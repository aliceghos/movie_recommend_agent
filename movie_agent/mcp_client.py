"""
MCP 客户端装配层。

聚合两个 MCP Server：
  - ``tmdb``      —— 自建，streamable HTTP，见 ``mcp_servers/tmdb_server.py``
  - ``watchlist`` —— 第三方 ``@modelcontextprotocol/server-filesystem``（npx / stdio），
                     沙箱在 ``data/watchlist/``，承载观影清单的读写

设计要点：``MultiServerMCPClient.get_tools()`` 返回的工具「每次调用新建 session」，
所以客户端对象不持有长命的连接，可以安全地跨 Streamlit rerun（每次 rerun 都是
一个新的 ``asyncio.run`` 事件循环）复用。

任一 Server 不可用时只记 warning、不抛异常：调用方据此补上本地降级工具。
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TMDB_URL = "http://127.0.0.1:8931/mcp"
DEFAULT_WATCHLIST_DIR = _REPO_ROOT / "data" / "watchlist"

_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 1.0


def _watchlist_dir() -> Path:
    raw = os.getenv("MCP_WATCHLIST_DIR", "").strip()
    path = Path(raw).expanduser() if raw else DEFAULT_WATCHLIST_DIR
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_connections() -> dict[str, dict[str, Any]]:
    """MCP Server 连接配置。URL 与沙箱目录可通过环境变量覆盖。"""
    return {
        "tmdb": {
            "transport": "streamable_http",
            "url": os.getenv("MCP_TMDB_URL", "").strip() or DEFAULT_TMDB_URL,
            "timeout": 15,
        },
        "watchlist": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(_watchlist_dir()),
            ],
        },
    }


def _describe(exc: BaseException) -> str:
    """把异常压成一行可读文案。

    MCP 客户端跑在 anyio 的 task group 里，连接失败会被包成
    ``ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)`` ——
    这话对着侧边栏的用户等于什么都没说，所以剥到最内层的真实原因。
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


async def load_mcp_tools_safe() -> tuple[list[BaseTool], list[str]]:
    """逐个 Server 拉取工具，失败的只记 warning。

    每个 Server 试两次：``npx`` 首次启动要解析并解包 filesystem server，慢到
    会把会话初始化熬死（实测表现为 ``McpError: Connection closed``）；而降级是
    整个 session 级别的 —— 一次瞬时失败就让用户得重开页面才能拿回观影清单能力，
    成本太高。第二次尝试时包已进缓存，通常瞬通。

    Returns:
        ``(tools, warnings)``。``warnings`` 里每条形如
        ``"tmdb: <原因>"``，调用方用 server 名判断该补哪些降级工具。
    """
    connections = build_connections()
    client = MultiServerMCPClient(connections)

    tools: list[BaseTool] = []
    warnings: list[str] = []

    for server_name in connections:
        last_error = ""
        for attempt in range(_MAX_ATTEMPTS):
            try:
                server_tools = await client.get_tools(server_name=server_name)
            except Exception as exc:  # noqa: BLE001 - 单个 Server 挂掉不能拖垮整个 Agent
                last_error = _describe(exc)
            else:
                last_error = "" if server_tools else "connected but exposed no tools"
                if server_tools:
                    tools.extend(server_tools)
                    break
            if attempt + 1 < _MAX_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
        if last_error:
            warnings.append(f"{server_name}: {last_error}")

    return tools, warnings


def failed_servers(warnings: list[str]) -> set[str]:
    """从 warning 列表里解析出失败的 server 名。"""
    return {w.split(":", 1)[0] for w in warnings}
