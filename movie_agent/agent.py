"""
基于 LangChain ReAct Agent + MiniMax LLM 的电影推荐 Agent。
对外暴露 chat()、chat_stream() 函数供 app.py 调用。
"""

import asyncio
import os
from typing import Any, AsyncGenerator

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as _create_langchain_agent

from movie_agent.callbacks import ToolDebugHandler, TokenTracker
from movie_agent.memory import (
    extract_and_save_preferences,
    load_profile,
    maybe_compress_history,
)
from movie_agent.tools import TOOLS

SYSTEM_PROMPT = """\
You are a movie recommendation assistant. You have access to two types of tools:

1. TMDB tools — for real-time movie data:
   search_movies, get_movie_details, get_recommendations, discover_movies,
   get_popular_movies, get_genres

2. Local knowledge base tool — for editorial content:
   search_local_knowledge — searches curated movie reviews, genre articles, and film art books.
   Use this when the user asks for critical opinions, review excerpts, background
   on film genres (cyberpunk, film noir), or film art theory and filmmaking concepts.
   Covered movies: Inception, Harry Potter and the Prisoner of Azkaban, Jane Eyre,
   Spirited Away, Wuthering Heights.
   Also contains film art book content for questions about cinematography,
   directing techniques, or film theory.

When you need to call a tool, reason step by step. After receiving tool results,
continue reasoning until you have enough information to give a final answer.

Always include the TMDB URL for each movie: https://www.themoviedb.org/movie/{movie_id}
"""


def _build_system_prompt(profile: dict[str, Any]) -> SystemMessage:
    """组合基础 prompt 与用户画像，生成每轮对话的 SystemMessage。"""
    parts = [SYSTEM_PROMPT]

    has_prefs = any(
        profile.get(k)
        for k in (
            "liked_genres", "disliked_genres",
            "liked_tones", "disliked_tones",
            "liked_movies", "disliked_movies",
        )
    )

    if has_prefs or profile.get("conversation_summary"):
        parts.append("\n--- User Profile (from long-term memory) ---")
        if profile.get("liked_genres"):
            parts.append(f"Liked genres: {', '.join(profile['liked_genres'])}")
        if profile.get("disliked_genres"):
            parts.append(f"Disliked genres: {', '.join(profile['disliked_genres'])}")
        if profile.get("liked_tones"):
            parts.append(f"Liked tones: {', '.join(profile['liked_tones'])}")
        if profile.get("disliked_tones"):
            parts.append(f"Disliked tones: {', '.join(profile['disliked_tones'])}")
        if profile.get("liked_movies"):
            parts.append(f"Movies the user has enjoyed: {', '.join(profile['liked_movies'])}")
        if profile.get("disliked_movies"):
            parts.append(f"Movies the user disliked: {', '.join(profile['disliked_movies'])}")
        if profile.get("conversation_summary"):
            parts.append(f"\nPrevious conversation summary:\n{profile['conversation_summary']}")
        parts.append("\nUse this profile to personalize your recommendations.")

    return SystemMessage(content="\n".join(parts))


def create_agent_minimax() -> dict[str, Any]:
    """初始化 LangChain ReAct Agent，使用 MiniMax LLM 和 TMDB 工具集。

    Returns:
        包含 'agent'、'history'、'llm'、'profile' 的字典，供 chat() 跨轮次使用。
    """
    api_key = os.getenv("MINIMAX_API_KEY", "")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")
    if not api_key or not group_id:
        raise ValueError("MINIMAX_API_KEY and MINIMAX_GROUP_ID must be set.")

    llm = ChatOpenAI(
        model="MiniMax-M2.7",
        base_url="https://api.minimaxi.com/v1",
        api_key=api_key,
        streaming=True,  # required for on_llm_new_token / on_chat_model_stream
        temperature=0.7,
        http_client=httpx.Client(headers={"GroupId": group_id}),
        http_async_client=httpx.AsyncClient(headers={"GroupId": group_id}),
    )

    agent = _create_langchain_agent(
        model=llm,
        tools=TOOLS,
    )

    return {"agent": agent, "history": [], "llm": llm, "profile": load_profile()}


# Backward-compatible alias (used by app.py)
create_agent = create_agent_minimax


def _format_tool_input(raw: Any) -> str:
    """Normalise tool input for display — dicts become compact JSON strings."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # If it's a dict with a single key whose value is a dict, unwrap it
        # (e.g. {"search_movies": {"query": "..."}} → {"query": "..."})
        if len(raw) == 1:
            inner = next(iter(raw.values()))
            if isinstance(inner, dict):
                raw = inner
        import json as _json
        return _json.dumps(raw, ensure_ascii=False)
    return str(raw)


# ---------------------------------------------------------------------------
# Streaming chat — yields events for progressive UI updates
# ---------------------------------------------------------------------------

async def chat_stream(
    agent_state: dict[str, Any],
    user_message: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming version of chat().

    Uses ``astream_events`` to yield tokens as they are generated, plus
    tool-start / tool-end events for the debug panel.
    Token usage is tracked via :class:`TokenTracker`.

    Yields:
        dict events:
        - ``{"type": "token", "content": "..."}``
        - ``{"type": "tool_start", "name": "...", "input": "..."}``
        - ``{"type": "tool_end", "name": "...", "output": "..."}``
        - ``{"type": "done", "response": "...", "token_usage": {...}, "tool_calls": [...]}``
    """
    profile = agent_state["profile"]
    llm = agent_state["llm"]

    agent_state["history"].append(HumanMessage(content=user_message))

    # 历史超过阈值时压缩旧消息
    agent_state["history"] = await maybe_compress_history(
        agent_state["history"], llm, profile
    )

    # 每轮注入最新用户画像
    messages_to_send = [_build_system_prompt(profile)] + agent_state["history"]

    tool_debug = ToolDebugHandler()
    token_tracker = TokenTracker()

    full_response = ""
    # Keep the last LLMResult seen so we can extract usage even when the
    # callback's on_llm_end doesn't fire (or llm_output is empty).
    last_llm_result = None

    async for event in agent_state["agent"].astream_events(
        {"messages": messages_to_send},
        config={"callbacks": [tool_debug, token_tracker]},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                full_response += chunk.content
                yield {"type": "token", "content": chunk.content}

        elif kind == "on_chat_model_end":
            output = event["data"].get("output")
            if output is not None:
                last_llm_result = output

            # Fallback: if streaming produced no tokens (e.g., provider doesn't
            # support streaming), extract the full response from LLMResult.
            if not full_response and hasattr(output, "generations"):
                last_gen = output.generations[0][0]
                msg = getattr(last_gen, "message", None)
                if msg and msg.content:
                    full_response = msg.content

        elif kind == "on_tool_start":
            # input may be a JSON string or a dict; normalise for display
            raw_input = event["data"].get("input")
            if isinstance(raw_input, dict):
                raw_input = _format_tool_input(raw_input)
            yield {
                "type": "tool_start",
                "name": event["name"],
                "input": raw_input,
            }

        elif kind == "on_tool_end":
            output = event["data"].get("output")
            yield {
                "type": "tool_end",
                "name": event["name"],
                "output": str(output)[:500] if output else "",
            }

    # --- If the callback never captured token usage, try last_llm_result ---
    if not token_tracker.last_call and last_llm_result is not None:
        usage = TokenTracker._extract_usage(last_llm_result)
        if usage:
            token_tracker.record_usage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            )

    # 记录最终回复到历史
    agent_state["history"].append(AIMessage(content=full_response))

    # 异步提取偏好，不阻塞回复返回
    asyncio.create_task(
        extract_and_save_preferences(user_message, full_response, llm, profile)
    )

    yield {
        "type": "done",
        "response": full_response,
        "token_usage": token_tracker.last_call,
        "tool_calls": tool_debug.calls,
    }


# ---------------------------------------------------------------------------
# Non-streaming wrapper — backward-compatible, returns whole response as str
# ---------------------------------------------------------------------------

async def chat(agent_state: dict[str, Any], user_message: str) -> str:
    """向 Agent 发送消息并返回文本回复。

    内部调用 :func:`chat_stream`，收集所有 token 后返回完整字符串。
    如需流式输出请直接使用 ``chat_stream()``。
    """
    response_text = ""
    async for event in chat_stream(agent_state, user_message):
        if event["type"] == "done":
            response_text = event["response"]
    return response_text
