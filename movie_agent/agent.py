"""
电影推荐 Agent 的装配入口。

编排全部交给 LangChain 1.x 的 ``create_agent``：

    工具   = 本地工具 + MCP 工具（tmdb Server + watchlist Server），Server 挂了则补本地降级工具
    记忆   = checkpointer（短期，按 thread_id）+ store（长期，跳会话），两者由 server 注入
    横切   = middleware 链（授权 / 画像注入 / 记忆写回 / 历史压缩 / 工具筛选）
    身份   = ``context_schema=MovieAgentContext``，每请求现传

**从每会话一个改为进程级单例**。改造前每个 Streamlit session 都
``create_movie_agent()`` 一次，每次都重拉 MCP 工具表、重建 LLM 客户端。现在
装配只在 ``server/main.py`` 的 lifespan 里做一次，跟用户无关的东西（工具、
模型、中间件）全部共享；跟用户有关的只有两样，都每请求传：

    thread_id  -> config["configurable"]，定位哪个会话的 checkpoint
    context    -> MovieAgentContext，定位请求者是谁、权限如何

因此 ``build_agent_runtime()`` **不再生成 thread_id**。它返回的东西里一个字节都
不应该和具体用户绑定 —— 这是单例安全的前提。
"""

import json
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain.agents import create_agent as _create_langchain_agent
from langchain_core.messages import HumanMessage

from movie_agent.callbacks import TokenTracker, ToolDebugHandler
from movie_agent.context import MovieAgentContext
from movie_agent.llm import get_chat_llm
from movie_agent.mcp_client import failed_servers, load_mcp_tools_safe
from movie_agent.middleware import build_middleware
from movie_agent.skills import list_skills
from movie_agent.tools import LOCAL_TOOLS, TMDB_TOOLS


async def build_agent_runtime(
    checkpointer: Any,
    store: Any,
    watchlist_root: Path,
) -> dict[str, Any]:
    """装配进程级 Agent。在 lifespan 里调一次。

    Args:
        checkpointer: ``AsyncPostgresSaver``，已 ``setup()``。
        store: ``AsyncPostgresStore``，已 ``setup()``。
        watchlist_root: 沙箱根，每用户一个子目录。

    Returns:
        ``agent`` / ``llm`` / ``store`` / ``warnings``（MCP 降级说明）/
        ``tools`` / ``tool_names`` / ``skills``。没有 ``thread_id``，也没有任何
        用户相关的字段。
    """
    llm = get_chat_llm()

    mcp_tools, warnings = await load_mcp_tools_safe()
    tools = [*LOCAL_TOOLS, *mcp_tools]

    # tmdb Server 不可达时补上本地实现，其余 Server 缺失只是少一块能力
    if "tmdb" in failed_servers(warnings):
        tools += TMDB_TOOLS
        warnings.append("falling back to in-process TMDB tools")

    agent = _create_langchain_agent(
        model=llm,
        tools=tools,
        middleware=build_middleware(len(tools), watchlist_root),
        checkpointer=checkpointer,
        store=store,
        context_schema=MovieAgentContext,
    )

    return {
        "agent": agent,
        "llm": llm,
        "store": store,
        "warnings": warnings,
        "tools": tools,
        "tool_names": [t.name for t in tools],
        "skills": [s.name for s in list_skills()],
    }


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
    runtime: dict[str, Any],
    user_message: str,
    *,
    thread_id: str,
    context: MovieAgentContext,
) -> AsyncGenerator[dict[str, Any], None]:
    """流式对话。

    只投递本轮的新消息，历史与 system prompt 分别由 checkpointer 和
    middleware 负责，这里不再手工拼装。

    ``thread_id`` 与 ``context`` 必须由调用方显式传：``runtime`` 是全进程共享的
    单例，身份不能从里面读。两个参数都是 keyword-only，避免位置传错。

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

    async for event in runtime["agent"].astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [tool_debug, token_tracker],
        },
        context=context,
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

async def chat(
    runtime: dict[str, Any],
    user_message: str,
    *,
    thread_id: str,
    context: MovieAgentContext,
) -> str:
    """向 Agent 发送消息并返回完整文本回复。"""
    response_text = ""
    async for event in chat_stream(
        runtime, user_message, thread_id=thread_id, context=context
    ):
        if event["type"] == "done":
            response_text = event["response"]
    return response_text
