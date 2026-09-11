"""测试夹具。

三条设计决定：

1. **跑在 ``movie_agent_test`` 库上**，表用 ``Base.metadata.create_all`` 建而不是
   跑 Alembic —— 建表是被测对象的前提，不是被测对象本身；迁移文件的正确性由
   ``alembic upgrade head`` 在真库上验证过。
2. **不跑 lifespan**。``httpx.ASGITransport`` 默认不触发 lifespan，正好：真
   lifespan 会拉 MCP（npx 冷启动）、载 384 维嵌入模型，让每次跑测试变成分钟级。
   这里手动往 ``app.state`` 塞一个假 runtime。
3. **假 LLM / 假 MCP，真 Store / 真 Postgres**。安全边界（越权、幂等、会话锁、
   命名空间隔离）全都长在数据库语义上，把 DB 也换成假的等于把要测的东西测没了。
   ``InMemoryStore`` 是官方 ``BaseStore`` 实现，命名空间语义与 Postgres 版一致。
"""

import os
import uuid
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql://localhost:5432/movie_agent_test"
)
# 必须在 import server.config 之前就位：Settings 是 lru_cache 单例
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ.setdefault("JWT_SECRET", "t" * 16 + uuid.uuid4().hex)

from httpx import ASGITransport, AsyncClient  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from server.config import get_settings  # noqa: E402
from server.db.models import Base  # noqa: E402
from server.db.session import dispose_engine, get_sessionmaker, init_engine  # noqa: E402
from server.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest_asyncio.fixture
async def engine(settings) -> AsyncIterator[Any]:
    """每个用例一套干净的表。

    用 drop+create 而不是"每个用例一个回滚的事务"：被测代码自己会 commit
    （幂等键、advisory lock 都依赖真实事务边界），外层包一个大事务反而会改变
    被测行为。
    """
    eng = init_engine(settings)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await dispose_engine()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def fake_runtime(store) -> dict[str, Any]:
    """冒充 ``build_agent_runtime()`` 的返回值。

    形状必须和真的一致 —— 路由只认这几个 key，多一个少一个都会在真环境下才炸。
    """
    return {
        "agent": None,
        "llm": None,
        "store": store,
        "warnings": [],
        "tools": [],
        "tool_names": ["search_movies", "read_text_file", "write_file"],
        "skills": ["movie-critique"],
    }


@pytest_asyncio.fixture
async def app(engine, fake_runtime):
    application = create_app()
    application.state.agent_runtime = fake_runtime
    return application


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db() -> AsyncIterator[Any]:
    """给用例直接查库用的 session，与被测请求各自独立。"""
    async with get_sessionmaker()() as session:
        yield session


# ---------------------------------------------------------------------------
# 账号辅助
# ---------------------------------------------------------------------------

class Account:
    """一个已登录的账号 + 它的鉴权头。"""

    def __init__(self, email: str, password: str, tokens: dict, user: dict) -> None:
        self.email = email
        self.password = password
        self.access_token = tokens["access_token"]
        self.refresh_token = tokens["refresh_token"]
        self.id = user["id"]
        self.role = user["role"]

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}


async def make_account(client: AsyncClient, email: str, password: str = "correct-horse-battery") -> Account:
    created = await client.post("/v1/auth/register", json={"email": email, "password": password})
    assert created.status_code == 201, created.text
    logged_in = await client.post("/v1/auth/login", json={"email": email, "password": password})
    assert logged_in.status_code == 200, logged_in.text
    return Account(email, password, logged_in.json(), created.json())


@pytest_asyncio.fixture
async def alice(client) -> Account:
    return await make_account(client, "alice@example.com")


@pytest_asyncio.fixture
async def bob(client) -> Account:
    return await make_account(client, "bob@example.com")
