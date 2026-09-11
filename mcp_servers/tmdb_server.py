"""
TMDB MCP Server —— 把 6 个 TMDB 工具通过 Model Context Protocol 暴露出去。

作为独立进程常驻，Agent 侧通过 streamable HTTP 连接：

    python mcp_servers/tmdb_server.py          # 监听 127.0.0.1:8931/mcp

这里只做协议层封装：工具实现与 docstring 全部来自
``movie_agent.tmdb_tools``，FastMCP 从函数签名推导 input schema，
因此 MCP 路径与 Agent 侧的本地降级路径对模型呈现完全一致。
"""

import os
import sys
from pathlib import Path

# 以脚本方式启动时 repo 根目录不在 sys.path 上，手动补一下
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 独立进程，需要自己读 TMDB_API_KEY / TMDB_BASE_URL
load_dotenv(_REPO_ROOT / ".env")

from movie_agent.tmdb_tools import TMDB_FUNCTIONS  # noqa: E402 - 必须在 sys.path 修补之后

HOST = os.getenv("MCP_TMDB_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_TMDB_PORT", "8931"))

mcp = FastMCP(
    name="tmdb",
    instructions="Real-time movie data from The Movie Database (TMDB).",
    host=HOST,
    port=PORT,
    # 每个请求独立 transport —— 客户端「每次工具调用新建 session」的用法下，
    # 无状态模式不会积累会话，也不依赖事件循环的连续性。
    stateless_http=True,
)

for _fn in TMDB_FUNCTIONS:
    mcp.tool()(_fn)


if __name__ == "__main__":
    if not os.getenv("TMDB_API_KEY", "").strip():
        print("[tmdb_server] warning: TMDB_API_KEY is empty, every tool call will fail.", file=sys.stderr)
    print(f"[tmdb_server] serving {len(TMDB_FUNCTIONS)} tools on http://{HOST}:{PORT}/mcp", file=sys.stderr)
    mcp.run(transport="streamable-http")
