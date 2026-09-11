# Movie Recommendation Agent

A conversational movie assistant built on **LangChain 1.x / LangGraph**, wired up with the four
capabilities that make an agent more than a prompt: **MCP** tools, **hybrid RAG**, **Agent Skills**,
and **two-tier memory** — served behind a **FastAPI** HTTP API with Postgres persistence, JWT auth,
and per-user isolation.

Everything is assembled through framework-native extension points — `create_agent` plus a
middleware chain — rather than a hand-rolled orchestration loop.

## Capabilities

| Capability | How it is implemented | What it buys |
|---|---|---|
| **MCP** | Self-hosted TMDB server (FastMCP, streamable HTTP) + third-party `server-filesystem` (npx, stdio), aggregated by `MultiServerMCPClient` | Tools live behind a protocol boundary, not inside the app. Either server can fail without taking the agent down |
| **RAG** | BM25 + vector hybrid recall (20 + 20) → LLM rerank → cited passages, with an index manifest for staleness detection | Single-route retrieval demonstrably misses; see [Why hybrid recall](#why-hybrid-recall) |
| **Skills** | `skills/*/SKILL.md` with three-level progressive disclosure | Methodology stays out of the system prompt until needed — ~150 tokens resident instead of thousands |
| **Memory** | `AsyncPostgresSaver` (short-term, per thread) + `AsyncPostgresStore` (long-term, pgvector semantic search over episodes) | Preferences and history both survive restarts, namespaced per user |
| **API & auth** | FastAPI + SSE streaming, argon2 + JWT with refresh rotation, ownership checks, role-based tool authorization | Multiple users on one deployment without reading each other's data |

22 tools are live at runtime: 6 TMDB (via MCP) + 14 filesystem (via MCP) + 2 local
(`search_local_knowledge`, `load_skill`).

## Architecture

```
Streamlit UI (app.py)  ── HTTP + SSE ──▸  FastAPI (server/)
                                            ├─ /v1/auth       register / login / refresh / logout
                                            ├─ /v1/conversations   SSE streaming, advisory lock, idempotency
                                            ├─ /v1/users/me    profile, usage
                                            └─ /readyz         probes DB + MCP tool count
                                                 │
                                                 ▾  agent singleton, identity per request
create_agent  ──  middleware chain
    │              ├─ ToolAuthorizationMiddleware  role whitelist + per-user path confinement
    │              ├─ SummarizationMiddleware      compresses history past 24 messages
    │              ├─ LLMToolSelectorMiddleware    22 tools → 6 per turn
    │              ├─ @dynamic_prompt              injects profile + skill index + recalled episodes
    │              └─ @after_agent                 extracts preferences, writes to the Store
    ├─ MCP: tmdb server        → mcp_servers/tmdb_server.py → TMDB API
    ├─ MCP: filesystem server  → data/watchlist/{user_id}/  (npx, sandboxed per user)
    ├─ search_local_knowledge  → hybrid RAG over data/{reviews,knowledge,books}
    └─ load_skill              → skills/*/SKILL.md
    ↓
Postgres  ──  business tables (Alembic)  +  checkpoint/store tables (framework .setup())
```

The agent is a **process-wide singleton**; the caller's identity travels per request through
LangGraph's `context_schema`, which is what selects the memory namespace and the filesystem
sandbox. `user_id` is read from the access token only — never from a request body.

The agent talks to any OpenAI-compatible chat endpoint — the default is DeepSeek
(`deepseek-flash`). Swapping providers is a `.env` edit. Embeddings run locally via
HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim).

## Prerequisites

- Python 3.12 (managed by [uv](https://docs.astral.sh/uv/))
- PostgreSQL 17 with the `pgvector` extension
- Node.js + `npx` — for the third-party filesystem MCP server. Without it the watchlist
  tools are unavailable; everything else still works
- A [TMDB API key](https://www.themoviedb.org/settings/api)
- An API key for any OpenAI-compatible provider (default: [DeepSeek](https://platform.deepseek.com/))

## Setup

```bash
git clone <repo-url>
cd movie_recommend_agent

brew install postgresql@17 pgvector
brew services start postgresql@17
createdb movie_agent && createdb movie_agent_test
psql movie_agent -c 'CREATE EXTENSION IF NOT EXISTS vector;'
psql movie_agent_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'

uv venv --python 3.12
uv pip install -r requirements.txt

cp .env.example .env    # then fill in the API keys and generate a JWT secret
alembic upgrade head    # business tables
```

`.env` essentials:

```
DATABASE_URL=postgresql://localhost:5432/movie_agent
JWT_SECRET=<openssl rand -hex 32>
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-flash
TMDB_API_KEY=your_tmdb_api_key
PYTHONPATH=.
```

A `JWT_SECRET` shorter than 32 characters, or an obvious placeholder, makes the service refuse to
start. A weak signing key is worse than no authentication at all, because anyone can mint a token
for any user while the system still looks protected.

Everything else has a working default. Optional overrides:

| Variable | Use when |
|----------|----------|
| `API_BASE_URL=http://127.0.0.1:8000` | The Streamlit client should talk to an API elsewhere |
| `JWT_ACCESS_TTL_MINUTES` / `JWT_REFRESH_TTL_DAYS` | 15 minutes / 14 days don't suit you |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | Tuning the connection pool |
| `TMDB_BASE_URL=https://api.tmdb.org/3` | `api.themoviedb.org` is unreachable (equivalent official domain) |
| `HF_ENDPOINT=https://hf-mirror.com` | `huggingface.co` is unreachable — needed to download embedding weights on first run |
| `MCP_TMDB_URL` | The TMDB MCP server runs somewhere other than `http://127.0.0.1:8931/mcp` |
| `MCP_WATCHLIST_DIR` | The watchlist root should live outside `./data/watchlist` |
| `OPENAI_UTILITY_REASONING_EFFORT` | Your gateway rejects `reasoning_effort`; set it empty to stop sending it |

Multiple profiles no longer need a variable — register a second account.

## Running

Build the index first — first run downloads ~450 MB of embedding weights, and you would
rather not watch a blank page while that happens:

```bash
source .venv/bin/activate
python scripts/build_index.py
```

Then start the full stack (TMDB MCP server → uvicorn → Streamlit):

```bash
./scripts/dev.sh
```

Open `http://localhost:8501` and register an account. The script waits for `/readyz` to pass
before starting the UI, because a cold start has to connect to Postgres, create the framework
tables, discover MCP tools, and load the embedding model.

Running `streamlit run app.py` alone now shows "Backend not ready" — the UI holds no agent of its
own. The API itself still tolerates a missing TMDB MCP server: it falls back to the in-process
TMDB tools and reports the degradation through `/v1/capabilities`.

Coming from the single-user version? `scripts/migrate_store_to_pg.py` imports an existing
`data/memory/store.json` into a registered account:

```bash
python -m scripts.migrate_store_to_pg --email you@example.com --dry-run
python -m scripts.migrate_store_to_pg --email you@example.com
```

## Tests

```bash
pytest          # 89 cases against movie_agent_test
```

Fake LLM and fake MCP, but a real Postgres and a real store — the security boundaries are all
expressed in database semantics, so faking the database would fake away the thing under test.
The suite covers refresh-token rotation and replay detection, cross-user access (404, not 403),
memory isolation asserted directly on the rendered system prompt, tool authorization, path-escape
attempts including symlinks, idempotent replay, and same-conversation serialization.


## Usage Examples

| Intent | Example query | Path exercised |
|--------|--------------|----------------|
| Recommendation | `"I loved Inception, what should I watch next?"` | skill → MCP TMDB → profile write |
| Discovery | `"Show me top-rated sci-fi movies from the 90s"` | MCP TMDB `discover_movies` |
| Film analysis | `"聊聊黑色电影的视觉风格"` | skill → hybrid RAG with citations |
| Watchlist | `"Add Blade Runner to my watchlist"` | skill → filesystem MCP `write_file`, confined to your own directory |
| Memory | `"我刚才说过我喜欢什么风格？"` | checkpointer + long-term profile |

## API

```
POST   /v1/auth/register | login | refresh | logout
GET    /v1/conversations                  list (cursor paging)
POST   /v1/conversations                  create → {id, thread_id}
DELETE /v1/conversations/{id}
GET    /v1/conversations/{id}/messages    history
POST   /v1/conversations/{id}/messages    send, SSE: token / tool_start / tool_end / done / error
GET    /v1/users/me | /me/profile | /me/usage
DELETE /v1/users/me/profile               clear the long-term profile
GET    /v1/capabilities                   tool count / MCP status / skills
GET    /healthz | /readyz
```

Two concurrency protections are treated as correctness, not optimization: sending a message takes
a transaction-scoped Postgres advisory lock on the conversation, so two quick sends cannot
overwrite each other's checkpoint; and an `Idempotency-Key` header makes a client retry return the
first reply instead of paying for a second one.


## Why hybrid recall

Measured on this corpus, single-route retrieval fails in *opposite* directions depending on
query language:

| Query | Vector top-6 | BM25 top-6 |
|---|---|---|
| 「赛博朋克电影的视觉特征」 | 2 of top-3 wrong (matched an unrelated review) | all 6 correct |
| `"cyberpunk visual style"` | all 6 correct | all 6 wrong |

Neither route is reliable alone, which is why recall merges both and an LLM rerank picks the
final `k`. Rerank costs one extra LLM call (~2–6 s); pass `rerank=False` to skip it.

## Project Structure

```
movie_recommend_agent/
├── app.py                       # Streamlit UI — an HTTP client, imports no agent code
├── ui/api_client.py             # API client: error envelope + SSE parsing
├── server/
│   ├── main.py                  # FastAPI app + lifespan (agent assembled once)
│   ├── config.py                # pydantic-settings, single validation point
│   ├── deps.py                  # current user + resource ownership
│   ├── security.py              # argon2 hashing, JWT issue/verify, refresh rotation
│   ├── errors.py                # {"error": {code, message, request_id}}
│   ├── schemas.py               # request/response models
│   ├── db/                      # engine, ORM models, Alembic migrations
│   └── routers/                 # auth / conversations / users / meta
├── mcp_servers/
│   └── tmdb_server.py           # self-hosted MCP server (FastMCP, streamable HTTP)
├── movie_agent/
│   ├── agent.py                 # build_agent_runtime + chat_stream
│   ├── context.py               # MovieAgentContext — per-request identity
│   ├── authz.py                 # role→tool whitelist, path confinement
│   ├── middleware.py            # authorization / summarization / tool selection / prompt / memory
│   ├── mcp_client.py            # multi-server MCP aggregation with per-server fallback
│   ├── store.py                 # long-term memory domain functions over BaseStore
│   ├── skills.py                # SKILL.md discovery + progressive disclosure
│   ├── rag.py                   # hybrid recall + LLM rerank + index manifest
│   ├── tools.py                 # local tools; TMDB fallback tool set
│   ├── tmdb_tools.py            # TMDB tool bodies, shared by the MCP server and the fallback
│   ├── tmdb_client.py           # TMDB REST wrapper
│   ├── llm.py                   # chat model / utility model factory
│   └── callbacks.py             # tool-debug and token-accounting handlers
├── skills/
│   ├── movie-recommendation/SKILL.md
│   ├── film-analysis/SKILL.md   # + references/knowledge-base.md (level-3 disclosure)
│   └── watchlist-curation/SKILL.md
├── tests/                       # auth, ownership, memory isolation, tool authz, conversations
├── scripts/
│   ├── dev.sh                   # MCP server → uvicorn → Streamlit
│   ├── build_index.py           # offline index build (--force to rebuild)
│   └── migrate_store_to_pg.py   # one-shot legacy memory import
└── data/                        # not committed
    ├── reviews/                 # English review .txt
    ├── knowledge/               # Chinese genre articles .txt
    ├── books/                   # PDFs (see note below)
    ├── index/                   # FAISS index + manifest.json
    └── watchlist/{user_id}/     # written by the filesystem MCP server, one dir per user
```

> `data/` is excluded from git. Drop your own `.txt` files into `data/reviews/` or
> `data/knowledge/` and re-run `scripts/build_index.py`; the manifest detects changed
> sources and rebuilds automatically on the next query too.
>
> Long-term memory now lives in Postgres, not in `data/memory/store.json`. That file is only an
> import source for `scripts/migrate_store_to_pg.py`.
>
> **Scanned PDFs contribute nothing.** `pypdf` extracts a text layer, so an image-only
> scan yields zero chunks. The build prints a warning per file when this happens.

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` / `langchain-openai` / `langchain-community` | Agent framework, middleware, model adapters |
| `langgraph` | Graph runtime, checkpointer, store |
| `langgraph-checkpoint-postgres` | Postgres-backed checkpointer and store (pgvector) |
| `langchain-mcp-adapters` / `mcp` | MCP client and the FastMCP server |
| `fastapi` / `uvicorn` / `sse-starlette` | HTTP API and SSE streaming |
| `sqlalchemy` / `psycopg` / `alembic` | Business-table ORM, psycopg3 driver, migrations |
| `pydantic-settings` | Configuration with startup validation |
| `pyjwt` / `argon2-cffi` | Token signing, password hashing |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | Lexical recall for the hybrid retriever |
| `sentence-transformers` | Local embedding model |
| `langchain-text-splitters` | Chunking |
| `pypdf` | PDF text extraction |
| `streamlit` / `httpx` | Web UI and its API client |
| `requests` | TMDB HTTP calls |
| `python-dotenv` | Environment loading |
| `pytest` / `pytest-asyncio` | Test suite |


## License

MIT
