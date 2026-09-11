"""健康检查与能力自述。

``/healthz`` 与 ``/readyz`` 的区别不是形式主义：

  - ``healthz`` —— 进程还活着，不查依赖。给重启策略用。
  - ``readyz``  —— **真探依赖**：DB 能查通、MCP 工具数 > 0。给流量准入用。

``readyz`` 必须真探，因为我们踩过 npx 冷启动导致 MCP 静默降级的情况：进程活得
好好的、``/healthz`` 一路 200，但观影清单功能已经没了。那种状态必须被健康检查
发现，而不是等用户发现。
"""

from fastapi import APIRouter, Request
from sqlalchemy import text

from server.deps import AgentRuntime, DbDep, SettingsDep
from server.errors import ServiceUnavailable
from server.schemas import CapabilitiesOut

router = APIRouter(tags=["meta"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """存活探针。刻意不查任何依赖 —— DB 挂了不该导致进程被反复重启。"""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request, db: DbDep) -> dict[str, object]:
    """就绪探针。DB 不通或 agent 没装好直接 503。"""
    checks: dict[str, object] = {}

    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - 探测失败本身就是结果
        raise ServiceUnavailable(f"database unreachable: {type(exc).__name__}") from exc

    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise ServiceUnavailable("agent runtime is not ready")

    tool_count = len(runtime.get("tool_names", []))
    if tool_count == 0:
        raise ServiceUnavailable("agent has no tools available")

    checks["tools"] = tool_count
    # warnings 非空说明有 Server 降级了。不算 not-ready（本地降级工具还在），
    # 但必须出现在响应里，否则这种"能力缺了一块"的状态没人看得见。
    checks["warnings"] = runtime.get("warnings", [])
    checks["status"] = "ok"
    return checks


@router.get("/v1/capabilities", response_model=CapabilitiesOut)
async def capabilities(runtime: AgentRuntime, settings: SettingsDep) -> CapabilitiesOut:
    """当前进程实际装到的能力。不需要认证 —— 这里没有任何用户数据。"""
    return CapabilitiesOut(
        tool_count=len(runtime["tool_names"]),
        tool_names=runtime["tool_names"],
        skills=runtime["skills"],
        warnings=runtime["warnings"],
        model=settings.openai_model,
    )
