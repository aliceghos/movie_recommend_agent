#!/usr/bin/env bash
# 起全栈：TMDB MCP Server → uvicorn（agent + API）→ Streamlit 界面（前台）。
#
#   ./scripts/dev.sh
#
# 顺序是有依赖的：
#   1. TMDB MCP Server 独立进程，必须先起 —— uvicorn 的 lifespan 会去拉它的
#      工具清单，拉不到就降级用进程内的本地 TMDB 工具。
#   2. uvicorn 装配 agent 单例并连 Postgres。Streamlit 现在只是 HTTP 客户端，
#      后端没起它连登录框都不显示（/readyz 探测失败）。
#
# 前置条件：postgresql@17 已启动，且 `alembic upgrade head` 跑过。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-.}"

MCP_LOG=/tmp/tmdb_mcp.log
API_LOG=/tmp/movie_agent_api.log
API_PORT="${API_PORT:-8000}"
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:$API_PORT}"

cleanup() {
  for pid in "${API_PID:-}" "${MCP_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# --- 1. TMDB MCP Server ----------------------------------------------------

echo "starting TMDB MCP server → $MCP_LOG"
python -u mcp_servers/tmdb_server.py >"$MCP_LOG" 2>&1 &
MCP_PID=$!

# 给它一点时间绑端口，否则 agent 装配时会误判成 Server 不可用
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
# MCP 会话初始化熬死（实测表现为 McpError: Connection closed），于是首次启动
# 就 “MCP degraded”、观影清单能力默默缺失。
# ``-p <pkg> true`` 只装包、拿 /usr/bin/true 当入口，因此不会真的启一个等 stdio
# 的 server 把脚本挂住。
if command -v npx >/dev/null 2>&1; then
  echo "warming up filesystem MCP server (npx cache)"
  npx -y -p @modelcontextprotocol/server-filesystem true >/dev/null 2>&1 || true
else
  echo "npx not found — watchlist tools will be unavailable" >&2
fi

# --- 2. API 服务 -----------------------------------------------------------

echo "starting API on :$API_PORT → $API_LOG"
# 刻意单 worker：advisory lock 跨 worker 也有效，但多 worker 下每个进程都要
# 自己拉一遍 MCP 工具、各驻留一份嵌入模型，本地开发不划算。
uvicorn server.main:app --host 127.0.0.1 --port "$API_PORT" >"$API_LOG" 2>&1 &
API_PID=$!

# lifespan 要连 DB、建框架表、拉 MCP 工具、载嵌入模型，冷启动可能几十秒。
# 这里等 /readyz 真的通，而不是等端口 —— 端口开着但 agent 没装好的话，
# Streamlit 一上来就会看到 “Backend not ready”。
echo -n "waiting for /readyz "
for _ in $(seq 1 120); do
  if curl -fsS "$API_BASE_URL/readyz" >/dev/null 2>&1; then
    echo "ok"
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo >&2
    echo "API exited early:" >&2
    tail -40 "$API_LOG" >&2
    exit 1
  fi
  echo -n "."
  sleep 1
done

# --- 3. Streamlit ----------------------------------------------------------

echo "starting Streamlit (API at $API_BASE_URL)"
streamlit run app.py
