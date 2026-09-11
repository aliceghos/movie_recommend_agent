"""用户自助路由。都作用于 token 里的那个用户，没有任何路径参数。"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import func, select

from movie_agent.store import aclear_profile, aload_profile
from server.db.models import UsageRecord
from server.deps import AgentRuntime, CurrentUser, DbDep
from server.schemas import ProfileOut, UsageOut, UserOut

router = APIRouter(prefix="/v1/users", tags=["users"])


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser):
    return user


@router.get("/me/profile", response_model=ProfileOut)
async def my_profile(user: CurrentUser, runtime: AgentRuntime) -> ProfileOut:
    """读长期画像。uid 来自 token，取不到别人的。"""
    profile = await aload_profile(runtime["store"], str(user.id))
    return ProfileOut(**profile)


@router.delete("/me/profile", status_code=status.HTTP_204_NO_CONTENT)
async def clear_my_profile(user: CurrentUser, runtime: AgentRuntime) -> None:
    """清空长期画像。

    只删画像不删情节记忆：画像是模型推断出来的结论，用户说"这不是我"就该能抹掉；
    情节记忆是发生过的事实，属于会话历史，删会话时才跟着走。
    """
    await aclear_profile(runtime["store"], str(user.id))


@router.get("/me/usage", response_model=UsageOut)
async def my_usage(
    user: CurrentUser,
    db: DbDep,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> UsageOut:
    """token 用量汇总。走 ``ix_usage_user_created`` 复合索引。"""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(
        func.coalesce(func.sum(UsageRecord.prompt_tokens), 0),
        func.coalesce(func.sum(UsageRecord.completion_tokens), 0),
        func.coalesce(func.sum(UsageRecord.total_tokens), 0),
        func.count(UsageRecord.id),
    ).where(UsageRecord.user_id == user.id, UsageRecord.created_at >= since)

    prompt, completion, total, count = (await db.execute(stmt)).one()
    return UsageOut(
        prompt_tokens=int(prompt),
        completion_tokens=int(completion),
        total_tokens=int(total),
        request_count=int(count),
    )
