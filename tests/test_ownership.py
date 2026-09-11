"""资源所有权：A 绝不能碰 B 的会话。

核心断言是**404 而不是 403**。403 等于承认"这个资源存在，只是你不能看"，把
会话 id 变成可枚举的信息；404 让"不存在"和"不属于你"从外部完全无法区分。
"""

import uuid

from server.db.models import Conversation


async def test_conversation_list_is_scoped_to_the_caller(client, alice, bob):
    await client.post("/v1/conversations", json={"title": "alice's"}, headers=alice.headers)
    await client.post("/v1/conversations", json={"title": "bob's"}, headers=bob.headers)

    alice_list = (await client.get("/v1/conversations", headers=alice.headers)).json()
    bob_list = (await client.get("/v1/conversations", headers=bob.headers)).json()

    assert [c["title"] for c in alice_list] == ["alice's"]
    assert [c["title"] for c in bob_list] == ["bob's"]


async def test_reading_someone_elses_messages_is_404(client, alice, bob):
    created = await client.post("/v1/conversations", json={}, headers=alice.headers)
    conversation_id = created.json()["id"]

    # 主人读得到
    mine = await client.get(f"/v1/conversations/{conversation_id}/messages", headers=alice.headers)
    assert mine.status_code == 200

    # 别人读不到，而且看不出它存在
    theirs = await client.get(f"/v1/conversations/{conversation_id}/messages", headers=bob.headers)
    assert theirs.status_code == 404
    assert theirs.json()["error"]["code"] == "not_found"


async def test_deleting_someone_elses_conversation_is_404_and_changes_nothing(
    client, alice, bob, db
):
    created = await client.post("/v1/conversations", json={}, headers=alice.headers)
    conversation_id = created.json()["id"]

    attempt = await client.delete(f"/v1/conversations/{conversation_id}", headers=bob.headers)
    assert attempt.status_code == 404

    still_there = await db.get(Conversation, uuid.UUID(conversation_id))
    assert still_there is not None, "a failed authorization check must not delete anything"


async def test_posting_to_someone_elses_conversation_is_404(client, alice, bob):
    """越权必须在**流开始之前**就被拒。

    一旦 SSE 响应头发出去，状态码就定死是 200 了 —— 那时候再发现越权已经晚了，
    只能塞一个 error 事件，而客户端很可能已经开始渲染。
    """
    created = await client.post("/v1/conversations", json={}, headers=alice.headers)
    conversation_id = created.json()["id"]

    response = await client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "let me read that"},
        headers=bob.headers,
    )
    assert response.status_code == 404
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_unknown_and_foreign_conversations_look_identical(client, alice, bob):
    created = await client.post("/v1/conversations", json={}, headers=alice.headers)
    foreign_id = created.json()["id"]
    nonexistent_id = str(uuid.uuid4())

    foreign = await client.get(f"/v1/conversations/{foreign_id}/messages", headers=bob.headers)
    missing = await client.get(f"/v1/conversations/{nonexistent_id}/messages", headers=bob.headers)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json()["error"]["message"] == missing.json()["error"]["message"]


async def test_thread_id_is_server_generated(client, alice):
    """客户端不能指定 ``thread_id``。

    thread_id 是 checkpoint 的唯一 key，LangGraph 只认字符串不认人。让客户端
    自带就等于让它去猜别人的。
    """
    created = await client.post(
        "/v1/conversations",
        json={"title": "t", "thread_id": "i-picked-this-myself"},
        headers=alice.headers,
    )
    assert created.status_code == 201
    assert created.json()["thread_id"] != "i-picked-this-myself"
    assert len(created.json()["thread_id"]) == 32  # uuid4().hex


async def test_usage_is_scoped_to_the_caller(client, alice, bob):
    alice_usage = (await client.get("/v1/users/me/usage", headers=alice.headers)).json()
    bob_usage = (await client.get("/v1/users/me/usage", headers=bob.headers)).json()
    assert alice_usage["request_count"] == bob_usage["request_count"] == 0


async def test_deactivated_account_loses_access_immediately(client, alice, db):
    """停用必须立刻生效，不能等 15 分钟后 token 自然过期。

    这就是 ``get_current_user`` 明知 role 在 token 里、仍然要查一次库的原因。
    """
    ok = await client.get("/v1/users/me", headers=alice.headers)
    assert ok.status_code == 200

    from server.db.models import User

    user = await db.get(User, uuid.UUID(alice.id))
    user.is_active = False
    await db.commit()

    denied = await client.get("/v1/users/me", headers=alice.headers)
    assert denied.status_code == 401
