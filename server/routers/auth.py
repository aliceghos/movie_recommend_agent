"""认证路由。

``/register`` / ``/login`` / ``/refresh`` / ``/logout``。

**refresh 轮换 + 重放检测**是这里唯一不平凡的地方：每次 refresh 都作废旧
token、发一对新的。如果收到一个**已经被 revoke 过**的 token，说明它被用了两次
—— 要么是客户端重试，要么是 token 泄露后攻击者在用。无法区分这两种情况，所以
按最坏情况处理：撤销该用户全部 refresh token，强制重新登录。
"""

from datetime import datetime, timezone

from fastapi import APIRouter, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from server.db.models import RefreshToken, User
from server.deps import DbDep, SettingsDep
from server.errors import Conflict, Unauthorized
from server.schemas import LoginIn, RefreshIn, RegisterIn, TokenOut, UserOut
from server.security import (
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    refresh_expiry,
    verify_password,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


async def _issue_tokens(db: DbDep, settings: SettingsDep, user: User) -> TokenOut:
    access, expires_in = create_access_token(settings, user.id, user.role)
    raw_refresh, hashed = new_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hashed,
            expires_at=refresh_expiry(settings),
        )
    )
    await db.commit()
    return TokenOut(access_token=access, refresh_token=raw_refresh, expires_in=expires_in)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterIn, db: DbDep) -> User:
    """注册。

    唯一性交给数据库的 UNIQUE 约束兜底，而不是"先 SELECT 再 INSERT" —— 那之间
    有并发窗口，两个同时注册的相同邮箱都能通过检查。
    """
    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise Conflict("email is already registered") from None
    await db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn, db: DbDep, settings: SettingsDep) -> TokenOut:
    """登录。

    邮箱不存在与密码错误返回同一句话：区分开就等于送了一个账号枚举接口。
    """
    stmt = select(User).where(User.email == payload.email.lower())
    user = (await db.execute(stmt)).scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.password_hash):
        raise Unauthorized("invalid email or password")
    if not user.is_active:
        raise Unauthorized("account is not available")

    return await _issue_tokens(db, settings, user)


@router.post("/refresh", response_model=TokenOut)
async def refresh(payload: RefreshIn, db: DbDep, settings: SettingsDep) -> TokenOut:
    """用 refresh token 换一对新 token。旧的立即作废。"""
    hashed = hash_refresh_token(payload.refresh_token)
    stmt = select(RefreshToken).where(RefreshToken.token_hash == hashed)
    token = (await db.execute(stmt)).scalar_one_or_none()

    if token is None:
        raise Unauthorized("invalid refresh token")

    now = datetime.now(timezone.utc)

    if token.revoked_at is not None:
        # 已作废的 token 又被用了一次。可能是客户端重试，也可能是 token 泄露
        # 后攻击者在用 —— 从服务端看这两者完全一样，所以按最坏情况处理：
        # 把这个用户的全部 refresh 都撤掉，逼他重新登录。
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == token.user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        await db.commit()
        raise Unauthorized("refresh token was already used; please sign in again")

    if token.expires_at <= now:
        raise Unauthorized("refresh token has expired")

    user = await db.get(User, token.user_id)
    if user is None or not user.is_active:
        raise Unauthorized("account is not available")

    token.revoked_at = now
    return await _issue_tokens(db, settings, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshIn, db: DbDep) -> None:
    """作废这一个 refresh token。

    不校验 token 是否存在、也不要求带 access token：登出必须是幂等且总是"成功"
    的。报错只会让客户端卡在一个登不出去的状态，而且会顺带确认某个 token 的
    有效性。
    """
    hashed = hash_refresh_token(payload.refresh_token)
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.token_hash == hashed, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    )
    await db.commit()
