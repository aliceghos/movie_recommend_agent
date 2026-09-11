"""
电影推荐 Agent 的装配入口。

编排全部交给 LangChain 1.x 的 ``create_agent``：

    工具   = 本地工具 + MCP 工具（tmdb Server + watchlist Server），Server 挂了则补本地降级工具
    记忆   = InMemorySaver（短期，按 thread_id）+ PersistentStore（长期，跨会话）
    横切   = middleware 链（画像注入 / 记忆写回 / 历史压缩 / 工具筛选）

对外只暴露 ``create_movie_agent()`` 与 ``chat_stream()``，供 app.py 调用。

**Checkpointer 为何是 InMemorySaver**：Streamlit 每次 rerun 都 ``asyncio.run()``
新建事件循环，连接型 checkpointer（AsyncSqliteSaver 之类）会绑定创建时的循环并
在下一轮失效。放在 ``st.session_state`` 里的纯内存 saver 恰好等于「本次会话的
短期记忆」，重启失忆是预期语义；跨会话该记住的东西全部走 Store。
"""

import json
import uuid
from typing import Any, AsyncGenerator

from langchain.agents import create_agent as _create_langchain_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from movie_agent.callbacks import TokenTracker, ToolDebugHandler
from movie_agent.llm import get_chat_llm
from movie_agent.mcp_client import failed_servers, load_mcp_tools_safe
from movie_agent.middleware import build_middleware
from movie_agent.skills import list_skills
from movie_agent.store import get_store, load_profile
from movie_agent.tools import LOCAL_TOOLS, TMDB_TOOLS


async def create_movie_agent() -> dict[str, Any]:
    """装配 Agent。

    Returns:
        ``agent`` / ``llm`` / ``checkpointer`` / ``store`` / ``thread_id`` /
        ``warnings``（MCP 降级说明）/ ``tool_names`` / ``skills``。
        注意这里**没有** ``history`` —— 消息历史由 checkpointer 按 thread_id 管理。
    """
    llm = get_chat_llm()

    mcp_tools, warnings = await load_mcp_tools_safe()
    tools = [*LOCAL_TOOLS, *mcp_tools]

    # tmdb Server 不可达时补上本地实现，其余 Server 缺失只是少一块能力
    if "tmdb" in failed_servers(warnings):
        tools += TMDB_TOOLS
        warnings.append("falling back to in-process TMDB tools")

    store = get_store()
    checkpointer = InMemorySaver()

    agent = _create_langchain_agent(
        model=llm,
        tools=tools,
        middleware=build_middleware(len(tools)),
        checkpointer=checkpointer,
        store=store,
    )

    return {
        "agent": agent,
        "llm": llm,
        "checkpointer": checkpointer,
        "store": store,
        "thread_id": uuid.uuid4().hex,
        "warnings": warnings,
        "tool_names": [t.name for t in tools],
        "skills": [s.name for s in list_skills()],
    }


def get_profile(agent_state: dict[str, Any]) -> dict[str, Any]:
    """读当前长期画像（供 UI 展示）。"""
    return load_profile(agent_state["store"])


def _format_tool_input(raw: Any) -> str:
    """把工具入参整理成便于展示的一行。"""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # {"search_movies": {"query": "..."}} → {"query": "..."}
        if len(raw) == 1:
            inner = next(iter(raw.values()))
            if isinstance(inner, dict):
                raw = inner
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


# ---------------------------------------------------------------------------
# Streaming chat — yields events for progressive UI updates
# ---------------------------------------------------------------------------

async def chat_stream(
    agent_state: dict[str, Any],
    user_message: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """流式对话。

    只投递本轮的新消息，历史与 system prompt 分别由 checkpointer 和
    middleware 负责，这里不再手工拼装。

    Yields:
        - ``{"type": "token", "content": "..."}``
        - ``{"type": "tool_start", "name": "...", "input": "..."}``
        - ``{"type": "tool_end", "name": "...", "output": "..."}``
        - ``{"type": "done", "response": "...", "token_usage": {...}, "tool_calls": [...]}``
    """
    tool_debug = ToolDebugHandler()
    token_tracker = TokenTracker()

    full_response = ""
    # 保留最后一个 LLMResult：回调的 on_llm_end 可能拿不到 usage，
    # 用它做兜底提取。
    last_llm_result = None

    async for event in agent_state["agent"].astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config={
            "configurable": {"thread_id": agent_state["thread_id"]},
            "callbacks": [tool_debug, token_tracker],
        },
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

            # 流式没吐出 token（网关不支持流式）时，从 LLMResult 里取全文
            if not full_response and hasattr(output, "generations"):
                last_gen = output.generations[0][0]
                msg = getattr(last_gen, "message", None)
                if msg and msg.content:
                    full_response = msg.content

        elif kind == "on_tool_start":
            raw_input = event["data"].get("input")
            if isinstance(raw_input, dict):
                raw_input = _format_tool_input(raw_input)
            yield {"type": "tool_start", "name": event["name"], "input": raw_input}

        elif kind == "on_tool_end":
            output = event["data"].get("output")
            yield {
                "type": "tool_end",
                "name": event["name"],
                "output": str(output)[:500] if output else "",
            }

    # 回调没抓到 usage 时用最后一个 LLMResult 补
    if not token_tracker.last_call and last_llm_result is not None:
        usage = TokenTracker._extract_usage(last_llm_result)
        if usage:
            token_tracker.record_usage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            )

    yield {
        "type": "done",
        "response": full_response,
        "token_usage": token_tracker.last_call,
        "tool_calls": tool_debug.calls,
    }


# ---------------------------------------------------------------------------
# Non-streaming wrapper — returns the whole response as str
# ---------------------------------------------------------------------------

async def chat(agent_state: dict[str, Any], user_message: str) -> str:
    """向 Agent 发送消息并返回完整文本回复。"""
    response_text = ""
    async for event in chat_stream(agent_state, user_message):
        if event["type"] == "done":
            response_text = event["response"]
    return response_text
