"""FastAPI 应用装配。

lifespan 里做四件事，顺序有讲究：

    1. 开 SQLAlchemy engine（业务表）
    2. 开裸 psycopg 池 + ``checkpointer.setup()`` / ``store.setup()``（框架表）
    3. ``build_agent_runtime()`` —— 拉 MCP 工具、装中间件，进程级一次
    4. 挂到 ``app.state``

第 3 步会拉 MCP 工具（npx 冷启动可能要几秒），所以放在最后：前面的都失败得快，
没必要为一个连不上的数据库等 npx。

**agent 是进程级单例**，跟用户无关的东西全部共享；身份每请求通过
``MovieAgentContext`` 传（见 ``movie_agent/context.py``）。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from movie_agent.agent import build_agent_runtime
from movie_agent.store import build_index_config
from server.config import get_settings
from server.db.session import build_psycopg_pool, dispose_engine, init_engine
from server.errors import install_error_handlers
from server.routers import auth, conversations, meta, users

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("movie_agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    init_engine(settings)

    pool = build_psycopg_pool(settings)
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    index = build_index_config()
    store = AsyncPostgresStore(pool, index=index) if index else AsyncPostgresStore(pool)
    await store.setup()
    if index is None:
        logger.warning("semantic memory disabled: no embedding model available")

    settings.watchlist_root.mkdir(parents=True, exist_ok=True)

    app.state.settings = settings
    app.state.pool = pool
    app.state.agent_runtime = await build_agent_runtime(
        checkpointer, store, settings.watchlist_root
    )
    logger.info(
        "agent ready: %d tools, %d skills, warnings=%s",
        len(app.state.agent_runtime["tool_names"]),
        len(app.state.agent_runtime["skills"]),
        app.state.agent_runtime["warnings"] or "none",
    )

    try:
        yield
    finally:
        await pool.close()
        await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Movie Recommendation Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    install_error_handlers(app)

    app.include_router(meta.router)
    app.include_router(auth.router)
    app.include_router(conversations.router)
    app.include_router(users.router)
    return app


app = create_app()
