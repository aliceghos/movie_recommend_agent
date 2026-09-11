"""记忆隔离 —— A 的画像绝不能出现在 B 的 system prompt 里。

这是改造前最隐蔽的一个问题：``store.py`` 的 ``user_id()`` 读环境变量、默认回落
到字符串 ``"default"``，所有人共用一个命名空间。它不会报错，只会静默串号 ——
没有任何日志，没有任何异常，直到有人在自己的推荐里看到别人的口味。

所以这里的断言方式是**直接看 ``memory_prompt`` 吐出来的字符串**，而不是隔着
HTTP 看结果：串号的表现形式就是 prompt 里多了一段本不该有的文字。
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from movie_agent.context import MovieAgentContext
from movie_agent.middleware import memory_prompt
from movie_agent.store import (
    add_episode,
    aload_profile,
    asave_profile,
    aclear_profile,
    episodes_ns,
    profile_ns,
    recall_episodes,
)


@dataclass
class FakeRuntime:
    store: Any
    context: Any


@dataclass
class FakeModelRequest:
    runtime: FakeRuntime
    state: dict[str, Any]
    system_message: Any = None
    model: Any = None
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)

    def override(self, **changes: Any) -> "FakeModelRequest":
        merged = {
            "runtime": self.runtime,
            "state": self.state,
            "system_message": self.system_message,
            **changes,
        }
        return FakeModelRequest(**merged)


async def render_prompt(store, user_id: str, question: str) -> str:
    """跑一遍 ``memory_prompt``，返回它注入的 system prompt 全文。"""
    request = FakeModelRequest(
        runtime=FakeRuntime(store=store, context=MovieAgentContext(user_id=user_id)),
        state={"messages": [HumanMessage(content=question)]},
    )
    captured: dict[str, str] = {}

    async def handler(req):
        captured["prompt"] = str(req.system_message.content)
        return AIMessage(content="ok")

    await memory_prompt.awrap_model_call(request, handler)
    return captured["prompt"]


# ---------------------------------------------------------------------------
# 命名空间
# ---------------------------------------------------------------------------

def test_namespaces_are_per_user():
    assert profile_ns("alice") != profile_ns("bob")
    assert profile_ns("alice") == ("users", "alice", "profile")
    assert episodes_ns("alice") == ("users", "alice", "episodes")


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_missing_uid_raises_instead_of_defaulting(bad):
    """uid 必填，没有默认值。

    改造前这里会回落到 ``"default"``。那个后门在单用户 demo 里无害，多用户下是
    最危险的一类 bug：忘传 uid 的代码路径不报错，而是静默把 A 的画像写进公共
    命名空间，再当成 B 的记忆读出来 —— 越权发生了但没有任何痕迹。
    """
    with pytest.raises(ValueError, match="uid is required"):
        profile_ns(bad)
    with pytest.raises(ValueError, match="uid is required"):
        episodes_ns(bad)


def test_context_also_refuses_an_empty_user_id():
    with pytest.raises(ValueError, match="user_id is required"):
        MovieAgentContext(user_id="")


# ---------------------------------------------------------------------------
# 画像隔离
# ---------------------------------------------------------------------------

async def test_profiles_do_not_leak_between_users(store):
    await asave_profile(store, {**await aload_profile(store, "alice"), "liked_genres": ["film noir"]}, "alice")
    await asave_profile(store, {**await aload_profile(store, "bob"), "liked_genres": ["musical"]}, "bob")

    assert await aload_profile(store, "alice") != await aload_profile(store, "bob")
    assert (await aload_profile(store, "alice"))["liked_genres"] == ["film noir"]
    assert (await aload_profile(store, "bob"))["liked_genres"] == ["musical"]


async def test_alices_profile_is_absent_from_bobs_system_prompt(store):
    profile = await aload_profile(store, "alice")
    profile["liked_genres"] = ["film noir"]
    profile["disliked_movies"] = ["Cats"]
    await asave_profile(store, profile, "alice")

    alice_prompt = await render_prompt(store, "alice", "what should I watch?")
    bob_prompt = await render_prompt(store, "bob", "what should I watch?")

    assert "film noir" in alice_prompt
    assert "Cats" in alice_prompt

    assert "film noir" not in bob_prompt
    assert "Cats" not in bob_prompt
    assert "User Profile" not in bob_prompt, "bob has no profile, so no profile block at all"


async def test_a_fresh_user_gets_the_base_prompt_only(store):
    prompt = await render_prompt(store, "nobody", "hi")
    assert "User Profile" not in prompt
    assert "Relevant past exchanges" not in prompt
    # 但基础内容还在：日期是给模型的，跟用户无关
    assert "Today's date is" in prompt


# ---------------------------------------------------------------------------
# 情节记忆隔离
# ---------------------------------------------------------------------------

async def test_episodes_do_not_leak_between_users(store):
    await add_episode(store, "alice asked about Wong Kar-wai", "alice")
    await add_episode(store, "bob asked about Pixar", "bob")

    alice_recalled = await recall_episodes(store, "movies", "alice", limit=5)
    bob_recalled = await recall_episodes(store, "movies", "bob", limit=5)

    assert any("Wong Kar-wai" in e for e in alice_recalled)
    assert not any("Pixar" in e for e in alice_recalled)
    assert any("Pixar" in e for e in bob_recalled)
    assert not any("Wong Kar-wai" in e for e in bob_recalled)


async def test_alices_episodes_are_absent_from_bobs_system_prompt(store):
    await add_episode(store, "alice loved the ending of Chungking Express", "alice")

    alice_prompt = await render_prompt(store, "alice", "recommend something")
    bob_prompt = await render_prompt(store, "bob", "recommend something")

    assert "Chungking Express" in alice_prompt
    assert "Chungking Express" not in bob_prompt


# ---------------------------------------------------------------------------
# 清空画像
# ---------------------------------------------------------------------------

async def test_clearing_a_profile_leaves_episodes_and_other_users_alone(store):
    profile = await aload_profile(store, "alice")
    profile["liked_genres"] = ["film noir"]
    await asave_profile(store, profile, "alice")
    await add_episode(store, "alice watched Le Samourai", "alice")

    bob_profile = await aload_profile(store, "bob")
    bob_profile["liked_genres"] = ["musical"]
    await asave_profile(store, bob_profile, "bob")

    await aclear_profile(store, "alice")

    # 画像清了
    assert (await aload_profile(store, "alice"))["liked_genres"] == []
    # 情节记忆没动 —— 它是发生过的事实，属于会话历史
    assert await recall_episodes(store, "movies", "alice", limit=5)
    # 别人的画像当然也没动
    assert (await aload_profile(store, "bob"))["liked_genres"] == ["musical"]


# ---------------------------------------------------------------------------
# 缺 uid 时的行为
# ---------------------------------------------------------------------------

async def test_prompt_skips_memory_entirely_when_context_is_missing(store):
    """没有 uid 就整段跳过，而不是拿 ``"default"`` 去读。

    宁可丢个性化，也不能读错人的画像。
    """
    profile = await aload_profile(store, "alice")
    profile["liked_genres"] = ["film noir"]
    await asave_profile(store, profile, "alice")

    request = FakeModelRequest(
        runtime=FakeRuntime(store=store, context=None),
        state={"messages": [HumanMessage(content="what should I watch?")]},
    )
    captured: dict[str, str] = {}

    async def handler(req):
        captured["prompt"] = str(req.system_message.content)
        return AIMessage(content="ok")

    await memory_prompt.awrap_model_call(request, handler)

    assert "film noir" not in captured["prompt"]
    assert "Today's date is" in captured["prompt"]
