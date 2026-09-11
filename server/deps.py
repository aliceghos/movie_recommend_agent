"""依赖注入。

这里是授权的收口处，三条硬规则：

1. **``user_id`` 只从 token 取**，永远不从请求体或 query 取。所以没有任何
   依赖接受 ``user_id`` 参数 —— 想传也传不进来。
2. **越权返回 404 而非 403**。403 等于告诉对方"这个资源存在，只是你不能看"，
   把资源 id 变成了可枚举的信息。404 让"不存在"和"不属于你"从外部无法区分。
3. **每个带 ``{conversation_id}`` 的路由都必须过 ``get_owned_conversation``**。
   不能拿路径参数直接去查 checkpointer：LangGraph 只认 ``thread_id`` 字符串，
   它不认人，谁传都给。所有权只能在业务表这一层查。
"""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, Path, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import Settings, get_settings
from server.db.models import Conversation, User
from server.db.session import session_scope
from server.errors import NotFound, ServiceUnavailable, Unauthorized
from server.security import decode_access_token

# auto_error=False：缺 header 时我们自己抛 Unauthorized，好让响应走统一的错误
# 信封，而不是 FastAPI 默认的 {"detail": ...}。
_bearer = HTTPBearer(auto_error=False)

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_db() -> AsyncIterator[AsyncSession]:
    async for session in session_scope():
        yield session


DbDep = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    settings: SettingsDep,
    db: DbDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    """解析 Bearer token 并取出用户。

    尽管 role 就在 token 里，这里仍然查一次库 —— 为了拿到 ``is_active``。
    停用一个账号必须立刻生效，不能等 15 分钟后 token 自然过期。
    """
    if credentials is None or not credentials.credentials:
        raise Unauthorized("missing bearer token")

    payload = decode_access_token(settings, credentials.credentials)
    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except (ValueError, KeyError):
        raise Unauthorized("invalid token subject") from None

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise Unauthorized("account is not available")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_owned_conversation(
    conversation_id: Annotated[uuid.UUID, Path()],
    user: CurrentUser,
    db: DbDep,
) -> Conversation:
    """取会话，并确认它属于当前用户。

    所有权条件写在 ``WHERE`` 里而不是取出来再 ``if``：少一次"忘了判断"的机会。
    """
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    )
    conversation = (await db.execute(stmt)).scalar_one_or_none()
    if conversation is None:
        # 别人的会话和不存在的会话，对外必须长得一模一样
        raise NotFound("conversation not found")
    return conversation


OwnedConversation = Annotated[Conversation, Depends(get_owned_conversation)]


def get_agent_runtime(request: Request) -> dict[str, Any]:
    """取 lifespan 里装好的进程级 agent 单例。"""
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise ServiceUnavailable("agent runtime is not ready")
    return runtime


AgentRuntime = Annotated[dict[str, Any], Depends(get_agent_runtime)]
