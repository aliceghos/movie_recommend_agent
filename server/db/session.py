"""数据库连接。

进程里有**两个**连接池，这是刻意的：

  - ``engine`` / ``AsyncSession`` —— SQLAlchemy，业务表的读写
  - ``AsyncConnectionPool`` —— 裸 psycopg3，交给 LangGraph 的
    ``AsyncPostgresSaver`` 与 ``AsyncPostgresStore``

框架要的是 psycopg 原生连接（而且 ``setup()`` 建表需要 autocommit），套一层
SQLAlchemy 反而要不断 unwrap。两个池都指向同一个库，驱动同为 psycopg3，所以
只是连接分账，没有第二套依赖。

配 max_size 时记住总数是 ``(db_pool_max_size × 2) × worker 数``，别超
Postgres 的 ``max_connections``（默认 100）—— 这是最常见的上线事故。
"""

from collections.abc import AsyncIterator

from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from server.config import Settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(
            settings.sqlalchemy_url,
            pool_size=settings.db_pool_max_size,
            max_overflow=0,
            pool_pre_ping=True,
            future=True,
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, autoflush=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("init_engine() must run before sessions are requested")
    return _sessionmaker


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def session_scope() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：一个请求一个 session，异常回滚。

    不在这里 commit —— 提交时机由路由决定，因为对话接口要把 assistant 消息、
    工具轨迹、用量记录放在同一个事务里一次落盘。
    """
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def build_psycopg_pool(settings: Settings) -> AsyncConnectionPool:
    """给 LangGraph 用的裸连接池。

    ``autocommit=True`` 是必须的：``AsyncPostgresSaver.setup()`` 里有建索引这类
    不能跑在事务块里的语句。
    """
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
