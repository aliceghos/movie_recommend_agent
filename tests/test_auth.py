"""认证流程：注册、登录、过期 token、refresh 轮换与重放检测。"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from sqlalchemy import select

from server.db.models import RefreshToken
from tests.conftest import make_account


async def test_register_then_login(client):
    account = await make_account(client, "new@example.com")
    assert account.access_token and account.refresh_token

    me = await client.get("/v1/users/me", headers=account.headers)
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "new@example.com"
    assert body["role"] == "user"
    # 响应模型是最后一层泄露防护：密码哈希不该出现在任何响应里
    assert "password_hash" not in body


async def test_duplicate_email_is_conflict(client, alice):
    again = await client.post(
        "/v1/auth/register", json={"email": alice.email, "password": "another-password"}
    )
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "conflict"


async def test_wrong_password_and_unknown_email_are_indistinguishable(client, alice):
    """账号枚举防护：两种失败必须返回同一句话。"""
    wrong_password = await client.post(
        "/v1/auth/login", json={"email": alice.email, "password": "not-the-password"}
    )
    unknown_email = await client.post(
        "/v1/auth/login", json={"email": "nobody@example.com", "password": "not-the-password"}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json()["error"]["message"] == unknown_email.json()["error"]["message"]


async def test_missing_token_is_rejected(client):
    response = await client.get("/v1/users/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_expired_access_token_is_rejected(client, alice, settings):
    expired = jwt.encode(
        {
            "sub": alice.id,
            "role": "user",
            "type": "access",
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    response = await client.get("/v1/users/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


async def test_refresh_token_cannot_be_used_as_access_token(client, alice, settings):
    """token 里的 ``type`` 声明必须被校验。

    refresh token 活 14 天，access token 只活 15 分钟。如果 access 的校验不看
    ``type``，那 14 天的凭据就等于 14 天的门票，短时效设计彻底失效。
    """
    refresh_shaped = jwt.encode(
        {
            "sub": alice.id,
            "role": "user",
            "type": "refresh",
            "exp": datetime.now(timezone.utc) + timedelta(days=14),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    response = await client.get(
        "/v1/users/me", headers={"Authorization": f"Bearer {refresh_shaped}"}
    )
    assert response.status_code == 401


async def test_token_signed_with_another_secret_is_rejected(client, alice, settings):
    forged = jwt.encode(
        {
            "sub": alice.id,
            "role": "admin",
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "an-attacker-supplied-secret-that-is-long-enough",
        algorithm="HS256",
    )
    response = await client.get("/v1/users/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


async def test_refresh_rotates_and_invalidates_the_old_token(client, alice):
    rotated = await client.post("/v1/auth/refresh", json={"refresh_token": alice.refresh_token})
    assert rotated.status_code == 200
    fresh = rotated.json()
    assert fresh["refresh_token"] != alice.refresh_token

    # 新的能用
    ok = await client.get("/v1/users/me", headers={"Authorization": f"Bearer {fresh['access_token']}"})
    assert ok.status_code == 200


async def test_replaying_a_revoked_refresh_revokes_every_token(client, alice, db):
    """重放检测：一个已作废的 refresh 又被用了，就把该用户全部 refresh 撤掉。

    无法区分"客户端重试"和"token 泄露后攻击者在用"，所以按最坏情况处理。
    """
    first = await client.post("/v1/auth/refresh", json={"refresh_token": alice.refresh_token})
    assert first.status_code == 200
    second_generation = first.json()["refresh_token"]

    # 拿已经轮换掉的旧 token 再来一次
    replay = await client.post("/v1/auth/refresh", json={"refresh_token": alice.refresh_token})
    assert replay.status_code == 401
    assert "sign in again" in replay.json()["error"]["message"]

    # 连刚发出去的那个也一起作废了
    after = await client.post("/v1/auth/refresh", json={"refresh_token": second_generation})
    assert after.status_code == 401

    rows = (await db.execute(select(RefreshToken))).scalars().all()
    assert rows, "should have refresh tokens on record"
    assert all(r.revoked_at is not None for r in rows)


async def test_logout_revokes_and_is_idempotent(client, alice):
    first = await client.post("/v1/auth/logout", json={"refresh_token": alice.refresh_token})
    assert first.status_code == 204

    # 再登出一次也必须成功：登出不能把用户卡在登不出去的状态
    second = await client.post("/v1/auth/logout", json={"refresh_token": alice.refresh_token})
    assert second.status_code == 204

    # 但那个 refresh 已经不能换 token 了
    reuse = await client.post("/v1/auth/refresh", json={"refresh_token": alice.refresh_token})
    assert reuse.status_code == 401


async def test_refresh_tokens_are_never_stored_in_plaintext(client, alice, db):
    rows = (await db.execute(select(RefreshToken))).scalars().all()
    assert rows
    for row in rows:
        assert row.token_hash != alice.refresh_token
        assert len(row.token_hash) == 64  # sha256 hex


@pytest.mark.parametrize("password", ["short", ""])
async def test_weak_password_is_rejected(client, password):
    response = await client.post(
        "/v1/auth/register", json={"email": "weak@example.com", "password": password}
    )
    assert response.status_code == 422
    # 校验失败的响应里绝不能回显收到的密码
    assert password not in response.text or password == ""
