"""服务配置 —— 全部配置项收拢在这里，替掉散落各处的 ``os.getenv``。

改造前 API key、MCP 地址、沙箱目录分别写在 ``llm.py`` / ``mcp_client.py`` /
``store.py`` 里各自 ``os.getenv``，缺一个要跑起来才知道。这里用
pydantic-settings 做一次集中校验：``JWT_SECRET`` 之类必填项缺失时进程直接
起不来，而不是等第一个请求打进来才 500。
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# pydantic-settings 自己会读 ``.env`` 填下面的 Settings，但 ``movie_agent/*`` 是
# 独立于 server 的领域层，仍然直接读 ``os.environ``（换任何 OpenAI 兼容网关只改
# 配置，不用回头动领域代码）。改造前是 Streamlit 入口替所有人 ``load_dotenv()``；
# agent 搬进 uvicorn 之后没人做这件事了，lifespan 会在 ``get_chat_llm()`` 上直接
# 炸 ``OPENAI_API_KEY must be set``。所以在这里把 ``.env`` 真正灌进环境变量。
#
# ``override=False``：进程里已有的真实环境变量优先。测试在 import 本模块之前就
# 把 ``DATABASE_URL`` 指向 ``movie_agent_test``，不能被 ``.env`` 盖回生产库。
load_dotenv(REPO_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 数据库 ---
    # 驱动写死 psycopg3：langgraph-checkpoint-postgres 只支持它，业务层跟着对齐，
    # 免得同一个进程里跑两个驱动两套连接池。
    database_url: str = Field(
        default="postgresql://localhost:5432/movie_agent",
        alias="DATABASE_URL",
    )
    db_pool_min_size: int = Field(default=2, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=10, alias="DB_POOL_MAX_SIZE")

    # --- 认证 ---
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_ttl_minutes: int = Field(default=15, alias="JWT_ACCESS_TTL_MINUTES")
    jwt_refresh_ttl_days: int = Field(default=14, alias="JWT_REFRESH_TTL_DAYS")

    # --- LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="", alias="OPENAI_MODEL")

    # --- 外部数据与 MCP ---
    tmdb_api_key: str = Field(default="", alias="TMDB_API_KEY")
    mcp_tmdb_url: str = Field(default="", alias="MCP_TMDB_URL")
    watchlist_root: Path = Field(
        default=REPO_ROOT / "data" / "watchlist",
        alias="MCP_WATCHLIST_DIR",
    )

    # --- 服务 ---
    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")

    @field_validator("jwt_secret")
    @classmethod
    def _reject_weak_secret(cls, value: str) -> str:
        """拒绝空或明显是占位符的密钥。

        签名密钥一旦是 ``changeme`` 这类默认值，任何人都能自己签一个 token
        冒充任意用户 —— 这比没有认证更危险，因为看起来是有的。
        """
        if len(value) < 32:
            raise ValueError("JWT_SECRET must be at least 32 chars; generate one with `openssl rand -hex 32`")
        if value.lower() in {"changeme", "secret", "please-change-me"} or set(value) == {"x"}:
            raise ValueError("JWT_SECRET is a placeholder; generate a real one")
        return value

    @field_validator("watchlist_root")
    @classmethod
    def _absolute_watchlist_root(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy 要显式的 ``+psycopg`` 才会走 psycopg3 而不是找 psycopg2。"""
        url = self.database_url
        if url.startswith("postgresql+"):
            return url
        return url.replace("postgresql://", "postgresql+psycopg://", 1)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """进程级单例。测试里可 ``get_settings.cache_clear()`` 后重建。"""
    return Settings()
