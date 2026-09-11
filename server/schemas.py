"""请求与响应模型。

这一层的价值不只是序列化，而是**把边界收窄**：``UserOut`` 里没有
``password_hash`` 字段，所以就算路由不小心把整个 ORM 对象丢出去，密码哈希也
出不了这道门。响应模型是最后一层泄露防护，不是样板代码。
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

class RegisterIn(BaseModel):
    email: EmailStr
    # 8 位下限是底线而非好建议。上限 128 是为了防 DoS：argon2 是刻意慢的，
    # 让人提交 10MB 的"密码"就能把 CPU 吃满。
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshIn(BaseModel):
    refresh_token: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------

class ConversationCreateIn(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationOut(BaseModel):
    """``updated_at`` 同时是列表接口的分页游标。

    刻意不用 offset：列表按 ``updated_at`` 倒序，翻页期间有新消息进来会让 offset
    漂移，翻出重复项或漏掉一整条。
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    thread_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime


# ---------------------------------------------------------------------------
# users / capabilities
# ---------------------------------------------------------------------------

class ProfileOut(BaseModel):
    liked_genres: list[str] = []
    disliked_genres: list[str] = []
    liked_tones: list[str] = []
    disliked_tones: list[str] = []
    liked_movies: list[str] = []
    disliked_movies: list[str] = []
    last_updated: str = ""


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int


class CapabilitiesOut(BaseModel):
    tool_count: int
    tool_names: list[str]
    skills: list[str]
    warnings: list[str]
    model: str
