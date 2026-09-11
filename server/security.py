"""密码哈希与 JWT。

  - 密码：argon2id。bcrypt 有 72 字节静默截断的坑，argon2 是当前的默认推荐。
  - access token：短命（默认 15 分钟）、无状态、不查库。
  - refresh token：长命（默认 14 天）、**存库可撤销**、轮换使用。

**为什么 refresh 只存哈希**：库里存的是 sha256，拿到备份也换不出 access token。
这里用 sha256 而不是 argon2 —— refresh token 本身是 32 字节的高熵随机串，不存在
字典攻击的空间，慢哈希只会给每次刷新白加几十毫秒；密码是人选的、低熵，才需要慢
哈希。两处用不同算法是刻意的，不是疏漏。
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from server.config import Settings
from server.errors import Unauthorized

_hasher = PasswordHasher()

TOKEN_TYPE_ACCESS = "access"


def hash_password(raw: str) -> str:
    return _hasher.hash(raw)


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, raw)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def new_refresh_token() -> tuple[str, str]:
    """生成 refresh token，返回 ``(原文, 哈希)``。

    原文只在这一刻存在，返回给客户端之后服务端再也拿不回来。
    """
    raw = secrets.token_urlsafe(32)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_access_token(settings: Settings, user_id: uuid.UUID, role: str) -> tuple[str, int]:
    """签 access token，返回 ``(token, 有效秒数)``。

    ``role`` 进 payload 是为了省掉每个请求查一次库。代价是改角色后旧 token 在
    过期前仍带旧角色 —— 15 分钟的窗口，对这个项目可以接受；真要即时生效就得
    引入吊销名单，那是另一个量级的复杂度。
    """
    ttl = timedelta(minutes=settings.jwt_access_ttl_minutes)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": TOKEN_TYPE_ACCESS,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, int(ttl.total_seconds())


def decode_access_token(settings: Settings, token: str) -> dict:
    """校验并解出 access token 的 payload。

    ``algorithms`` 必须写死成单元素列表：允许多算法（尤其带上 ``none``）是 JWT
    最经典的绕过方式 —— 攻击者改 header 里的 alg 就能让服务端跳过签名校验。
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise Unauthorized("token has expired") from None
    except jwt.InvalidTokenError:
        raise Unauthorized("invalid token") from None

    if payload.get("type") != TOKEN_TYPE_ACCESS:
        # 不加这一条，refresh token 就能当 access token 用 —— 等于把 14 天的
        # 长命凭据直接当门票，短命 access token 的意义全没了。
        raise Unauthorized("invalid token")
    return payload


def refresh_expiry(settings: Settings) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_ttl_days)
