"""
Middleware 装配层 —— Agent 的横切关注点。

四件事各自挂在 LangChain 1.x 的一个原生扩展点上，而不是塞进 ``chat_stream`` 手写：

    dynamic_prompt   每轮把长期画像 + Skill 索引 + 相关情节记忆注入 system prompt
    after_agent      一轮结束后把偏好与情节摘要写回长期记忆
    Summarization    历史超阈值时压缩（框架内置）
    ToolSelector     工具超过 20 个时先筛后调（框架内置）

前两个是本文件实现的；后两个在 ``build_middleware`` 里配置。

**为什么记忆写入必须挂在 ``after_agent``**：改造前它是
``asyncio.create_task(extract_and_save_preferences(...))``，而 Streamlit 每轮用
``asyncio.run()`` 跑流式消费。``asyncio.run`` 返回时会关闭事件循环并取消未完成
任务，抛出的 ``CancelledError`` 继承自 ``BaseException``，被
``except Exception`` 漏掉 —— 于是写入从不发生，也从不报错。实测证据：画像文件
的 mtime 停在某次显式 ``await`` 的脚本运行，之后所有浏览器对话都没写进去。
挂进 ``after_agent`` 后它是图里的一个节点，图没跑完流式迭代就不会结束。
"""

import json
import sys
from datetime import date
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    LLMToolSelectorMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    after_agent,
    dynamic_prompt,
)
from langchain_core.messages import AIMessage, HumanMessage

from movie_agent.llm import get_utility_llm
from movie_agent.skills import skill_index_prompt
from movie_agent.store import (
    PREFERENCE_KEYS,
    add_episode,
    aload_profile,
    asave_profile,
    has_preferences,
    recall_episodes,
)

BASE_SYSTEM_PROMPT = """\
You are a movie recommendation assistant.

Your tools come from three places:

1. **TMDB tools** (`search_movies`, `get_movie_details`, `get_recommendations`,
   `discover_movies`, `get_popular_movies`, `get_genres`) — real-time movie data.
2. **Local knowledge base** (`search_local_knowledge`) — a small curated corpus of
   film reviews and genre articles. Use it for critical opinion, genre theory, and
   aesthetic vocabulary. It is not a substitute for TMDB metadata.
3. **Watchlist files** (`list_directory`, `read_text_file`, `write_file`, `edit_file`,
   and siblings) — the user's saved lists, in a sandboxed directory.

Reason step by step. After a tool returns, keep going until you can answer properly.

Always include the TMDB URL for each movie: https://www.themoviedb.org/movie/{movie_id}
Never invent movie data. If a tool fails or returns nothing, say so.
"""

_PROFILE_LABELS = {
    "liked_genres": "Liked genres",
    "disliked_genres": "Disliked genres",
    "liked_tones": "Liked tones",
    "disliked_tones": "Disliked tones",
    "liked_movies": "Movies the user enjoyed",
    "disliked_movies": "Movies the user disliked",
}

_EXTRACTION_PROMPT = """\
Analyse this exchange from a movie conversation.

User: {user}
Assistant: {assistant}

The user's profile already contains:
{current}

Return ONLY a JSON object with these keys:
{{"liked_genres": [], "disliked_genres": [], "liked_tones": [], "disliked_tones": [],
 "liked_movies": [], "disliked_movies": [], "episode": ""}}

- The six list fields: only preference signals that are NEW (not already listed above)
  and stated or clearly implied by the user. Empty lists if nothing new.
- "episode": one factual sentence recording what the user asked for and what they
  reacted to, written so it is useful weeks later. Empty string if this exchange
  carries nothing worth remembering.
"""


# ---------------------------------------------------------------------------
# 1. 每轮注入长期记忆与 Skill 索引
# ---------------------------------------------------------------------------

def _render_profile(profile: dict[str, Any]) -> str:
    if not has_preferences(profile):
        return ""
    lines = ["--- User Profile (long-term memory) ---"]
    lines += [
        f"{label}: {', '.join(profile[key])}"
        for key, label in _PROFILE_LABELS.items()
        if profile.get(key)
    ]
    lines.append("Use this to personalise recommendations; treat dislikes as filters.")
    return "\n".join(lines)


def _last_human_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


@dynamic_prompt
async def memory_prompt(request: ModelRequest) -> str:
    """把长期记忆和 Skill 索引拼进 system prompt。

    每次模型调用都重跑，所以同一轮里 Skill 加载之后的后续调用也看得到最新画像。
    Store 不可用时静默退回基础 prompt —— 记忆是增强项，不该让对话失败。
    """
    parts = [BASE_SYSTEM_PROMPT]

    # 模型没有查日期的工具，不给就会在写观影清单时把 added 列填成 TBD
    # 并反过来向用户解释（已实测）。
    parts.append(f"Today's date is {date.today().isoformat()}.")

    index = skill_index_prompt()
    if index:
        parts.append(index)

    store = getattr(request.runtime, "store", None)
    if store is not None:
        try:
            profile = await aload_profile(store)
            rendered = _render_profile(profile)
            if rendered:
                parts.append(rendered)

            query = _last_human_text(request.state["messages"])
            if query:
                episodes = await recall_episodes(store, query, limit=2)
                if episodes:
                    parts.append(
                        "--- Relevant past exchanges ---\n"
                        + "\n".join(f"- {e}" for e in episodes)
                    )
        except Exception as exc:  # noqa: BLE001 - 读记忆失败不该中断对话
            print(f"[memory] prompt injection skipped: {type(exc).__name__}: {exc}", file=sys.stderr)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2. 一轮结束后写回长期记忆
# ---------------------------------------------------------------------------

def _parse_json_object(raw: str) -> dict[str, Any]:
    """解析 LLM 返回的 JSON，容忍 markdown 代码块包裹。"""
    text = raw.strip()
    if text.startswith("```"):
        segments = text.split("```")
        text = segments[1] if len(segments) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _last_exchange(messages: list[Any]) -> tuple[str, str]:
    """本轮的用户输入与最终回复。工具调用中间态不算回复。"""
    user_text = _last_human_text(messages)
    reply = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            reply = str(msg.content)
            break
    return user_text, reply


@after_agent
async def persist_memory(state: dict[str, Any], runtime: Any) -> None:
    """抽取偏好与情节摘要并写入 Store。

    异常只记日志不上抛：记忆写失败不该让用户看到的回复变成报错。
    但和改造前不同，它**确实会被执行完**，失败也**确实会打出来**。
    """
    store = getattr(runtime, "store", None)
    if store is None:
        return

    user_text, reply = _last_exchange(state.get("messages", []))
    if not user_text or not reply:
        return

    try:
        profile = await aload_profile(store)
        current = "\n".join(
            f"- {label}: {profile.get(key) or '[]'}"
            for key, label in _PROFILE_LABELS.items()
        )
        response = await get_utility_llm().ainvoke([
            HumanMessage(content=_EXTRACTION_PROMPT.format(
                user=user_text[:2000], assistant=reply[:2000], current=current,
            ))
        ])
        extracted = _parse_json_object(str(response.content))

        changed = False
        for key in PREFERENCE_KEYS:
            items = extracted.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                value = str(item).strip()
                if value and value not in profile[key]:
                    profile[key].append(value)
                    changed = True
        if changed:
            await asave_profile(store, profile)

        episode = str(extracted.get("episode", "")).strip()
        if episode:
            await add_episode(store, episode)

    except Exception as exc:  # noqa: BLE001 - 边界层
        print(f"[memory] extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 3 + 4. 内置中间件的装配
# ---------------------------------------------------------------------------

# 工具筛选的启用门槛。低于这个数就没有筛的必要，反而多花一次 LLM 调用。
TOOL_SELECTION_THRESHOLD = 12
MAX_TOOLS_PER_TURN = 6

# 工具筛选阶段必须始终可见的工具：Skill 加载是所有流程的入口，
# 被筛掉的话模型就永远不知道该按哪套流程走。
ALWAYS_AVAILABLE_TOOLS = ["load_skill"]


def build_middleware(tool_count: int) -> list[AgentMiddleware]:
    """按工具规模装配中间件链。顺序即执行顺序。"""
    middleware: list[AgentMiddleware] = [
        SummarizationMiddleware(
            model=get_utility_llm(),
            trigger={"messages": 24},
            keep=("messages", 12),
        ),
    ]

    if tool_count > TOOL_SELECTION_THRESHOLD:
        middleware.append(
            LLMToolSelectorMiddleware(
                model=get_utility_llm(),
                max_tools=MAX_TOOLS_PER_TURN,
                always_include=ALWAYS_AVAILABLE_TOOLS,
                # 解析失败时放行全部工具，而不是让这一轮直接失败
                on_parsing_failure="all",
            )
        )

    middleware += [memory_prompt, persist_memory]
    return middleware
