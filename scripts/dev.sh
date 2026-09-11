#!/usr/bin/env bash
# 起全栈：TMDB MCP Server（后台）+ Streamlit 界面（前台）。
#
#   ./scripts/dev.sh
#
# MCP Server 是独立进程，所以必须先起：Agent 初始化时会去拉它的工具清单，
# 拉不到就降级用进程内的本地 TMDB 工具（功能一样，只是不经 MCP 协议）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-.}"

MCP_LOG=/tmp/tmdb_mcp.log

cleanup() {
  if [[ -n "${MCP_PID:-}" ]] && kill -0 "$MCP_PID" 2>/dev/null; then
    echo "stopping TMDB MCP server (pid $MCP_PID)"
    kill "$MCP_PID" 2>/dev/null || true
    wait "$MCP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "starting TMDB MCP server → $MCP_LOG"
python -u mcp_servers/tmdb_server.py >"$MCP_LOG" 2>&1 &
MCP_PID=$!

# 给 uvicorn 一点时间绑端口，否则 Agent 初始化会误判成 Server 不可用
for _ in $(seq 1 20); do
  if grep -q "Uvicorn running" "$MCP_LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$MCP_PID" 2>/dev/null; then
    echo "MCP server exited early:" >&2
    cat "$MCP_LOG" >&2
    exit 1
  fi
  sleep 0.5
done

# 预热第三方 filesystem MCP Server。npx 首次运行要解析并解包包体，慢到会把
# MCP 会话初始化熬死（实测表现为 McpError: Connection closed），于是首次打开页面
# 就看到“MCP degraded”、观影清单能力默默缺失。
# ``-p <pkg> true`` 只装包、拿 /usr/bin/true 当入口，因此不会真的启一个等 stdio
# 的 server 把脚本挂住。
if command -v npx >/dev/null 2>&1; then
  echo "warming up filesystem MCP server (npx cache)"
  npx -y -p @modelcontextprotocol/server-filesystem true >/dev/null 2>&1 || true
else
  echo "npx not found — watchlist tools will be unavailable" >&2
fi

echo "starting Streamlit"
streamlit run app.py
