# CLAUDE.md

## Project Overview

A conversational movie recommendation agent built on LangChain 1.x / LangGraph, served behind a
FastAPI HTTP API with Postgres persistence and JWT auth. It demonstrates four capabilities wired
through framework-native extension points: MCP tools, hybrid RAG, Agent Skills, and two-tier
memory.

The orchestration is **not** hand-rolled: there is no manual history list, no manual system-prompt
assembly, no manual compression. All of that lives in a middleware chain passed to `create_agent`.

Streamlit is now only a client. It talks to the API over HTTP/SSE and imports nothing from
`movie_agent` — the agent runs in the uvicorn process, as a single process-wide instance shared by
all users, with identity passed per request.

## Architecture

```
Streamlit UI (app.py)  →  ui/api_client.py  ──HTTP/SSE──┐
                                                        │
FastAPI (server/main.py)  ←─────────────────────────────┘
    ├─ lifespan, once per process:
    │    SQLAlchemy engine (business tables)
    │    AsyncConnectionPool (autocommit) → AsyncPostgresSaver.setup() / AsyncPostgresStore.setup()
    │    build_agent_runtime()  →  app.state.agent_runtime
    ├─ routers: auth / conversations / users / meta
    └─ per request: MovieAgentContext(user_id, role) from the JWT — never from the body
         ↓
create_agent (movie_agent/agent.py)
    ├─ middleware (movie_agent/middleware.py), in order:
    │    1. ToolAuthorizationMiddleware    role filter + per-user path confinement
    │    2. SummarizationMiddleware        trigger={"messages": 24}, keep=("messages", 12)
    │    3. LLMToolSelectorMiddleware      only when tool_count > 12; max_tools=6
    │    4. @dynamic_prompt memory_prompt  base prompt + date + skill index + profile + episodes
    │    5. @after_agent  persist_memory   extracts preferences → Store
    ├─ tools
    │    ├─ MCP "tmdb"       → mcp_servers/tmdb_server.py (streamable HTTP, 127.0.0.1:8931)
    │    ├─ MCP "watchlist"  → npx @modelcontextprotocol/server-filesystem → data/watchlist/
    │    ├─ search_local_knowledge → movie_agent/rag.py
    │    └─ load_skill            → movie_agent/skills.py
    ├─ checkpointer: AsyncPostgresSaver   (short-term, keyed by thread_id)
    └─ store: AsyncPostgresStore          (long-term, pgvector-indexed)
```

Two connection pools on purpose: SQLAlchemy owns the business tables, a raw psycopg pool
(`autocommit=True`) belongs to LangGraph. Alembic manages only the business tables; the framework
tables are created by `.setup()` and their layout is the framework's business, not ours.

## Key Components

| File | Role |
|------|------|
| `app.py` | Streamlit UI — login, conversation list, SSE consumption, sidebar. Imports no `movie_agent` code and contains no `asyncio.run()` |
| `ui/api_client.py` | Thin HTTP client; unwraps the `{"error": {...}}` envelope, parses SSE lines |
| `server/main.py` | FastAPI app + lifespan (agent assembled once per process) |
| `server/config.py` | pydantic-settings; also `load_dotenv()` so the domain layer's `os.getenv` still resolves |
| `server/deps.py` | `CurrentUser` (token only) and `get_owned_conversation` (ownership → 404) |
| `server/security.py` | argon2 hashing, JWT issue/verify, refresh rotation with replay detection |
| `server/errors.py` | `{"error": {code, message, request_id}}`; no traceback ever reaches a client |
| `server/db/models.py` | 7 business tables, all FKs `ON DELETE CASCADE` |
| `server/routers/conversations.py` | SSE streaming, advisory-lock serialization, idempotency keys, persistence |
| `movie_agent/agent.py` | `build_agent_runtime()` (takes checkpointer/store, no thread_id) and `chat_stream()` |
| `movie_agent/context.py` | `MovieAgentContext` — the `context_schema`; rejects an empty `user_id` |
| `movie_agent/authz.py` | Role→tool whitelist and `confine_path()` |
| `movie_agent/middleware.py` | The five middleware; `BASE_SYSTEM_PROMPT` lives here |
| `movie_agent/mcp_client.py` | `load_mcp_tools_safe()` → `(tools, warnings)`; per-server failure isolation |
| `mcp_servers/tmdb_server.py` | FastMCP server, standalone process, reads its own `.env` |
| `movie_agent/tmdb_tools.py` | The 6 TMDB tool bodies. Imported by both the MCP server and `tools.py` so schema/docstring exist once |
| `movie_agent/store.py` | Domain functions over `BaseStore` (`aload_profile` / `add_episode` / …); uid is **required** |
| `movie_agent/skills.py` | `SKILL.md` discovery, frontmatter parsing, `load_skill` tool |
| `movie_agent/rag.py` | Hybrid recall, LLM rerank, index manifest |
| `movie_agent/llm.py` | `get_chat_llm()` (streaming, temp 0.7) / `get_utility_llm()` (deterministic singleton) |
| `scripts/migrate_store_to_pg.py` | One-shot import of the legacy `data/memory/store.json` into a registered user |
| `data/` | Corpus, index, watchlist — all excluded from git |


## Environment Variables

Required in `.env`:

```
DATABASE_URL=postgresql://localhost:5432/movie_agent
JWT_SECRET=<openssl rand -hex 32>            # < 32 chars and the process refuses to start
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-flash
TMDB_API_KEY=<key>
PYTHONPATH=.
```

Optional — all have defaults:

```
DB_POOL_MIN_SIZE=2 / DB_POOL_MAX_SIZE=10
JWT_ACCESS_TTL_MINUTES=15 / JWT_REFRESH_TTL_DAYS=14
API_BASE_URL=http://127.0.0.1:8000            # where the Streamlit client looks for the API
CORS_ORIGINS=                                 # comma-separated
TMDB_BASE_URL=https://api.tmdb.org/3          # when api.themoviedb.org is unreachable
HF_ENDPOINT=https://hf-mirror.com             # when huggingface.co is unreachable
MCP_TMDB_URL=http://127.0.0.1:8931/mcp
MCP_TMDB_HOST / MCP_TMDB_PORT                 # server-side bind address
MCP_WATCHLIST_DIR=./data/watchlist
OPENAI_UTILITY_REASONING_EFFORT=none          # empty string = don't send the field
```

There is no `MOVIE_AGENT_USER_ID` any more. Identity comes from the JWT and nowhere else; see
Hard Constraints.

## Running the Project

```bash
brew install postgresql@17 pgvector && brew services start postgresql@17
createdb movie_agent && createdb movie_agent_test
psql movie_agent -c 'CREATE EXTENSION IF NOT EXISTS vector;'
psql movie_agent_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'

uv venv --python 3.12
uv pip install -r requirements.txt
alembic upgrade head              # business tables; framework tables come from .setup()
python scripts/build_index.py     # do this first; downloads ~450 MB of weights
./scripts/dev.sh                  # MCP server → uvicorn → Streamlit
pytest                            # runs against movie_agent_test
```

`streamlit run app.py` alone now shows "Backend not ready" — the UI has no agent of its own. The
API, on the other hand, runs fine without the TMDB MCP server: it falls back to the in-process
TMDB tools and reports the degradation through `/v1/capabilities`.

Cold start takes tens of seconds (DB connect, framework table setup, MCP discovery, embedding
model load), so `dev.sh` waits for `/readyz` to actually pass rather than for the port to open.

## Hard Constraints

**`user_id` comes from the access token, never from a request body, query parameter, or path.**
Everything downstream depends on this: `MovieAgentContext.user_id` selects the memory namespace and
the filesystem sandbox, so a caller-supplied id would be a complete authorization bypass. Related
rules that must hold together:

- `profile_ns(uid)` / `episodes_ns(uid)` **raise** on an empty uid — there is no `"default"`
  fallback. A missing uid must crash loudly rather than silently write into a shared namespace.
- Accessing someone else's resource returns **404, not 403**, so the API does not confirm that an
  id exists.
- Every route carrying a `{conversation_id}` goes through `get_owned_conversation`. Never hand a
  path parameter to the checkpointer directly: LangGraph only knows `thread_id` strings, it has no
  concept of who owns one.
- Tool authorization is two-layered: `wrap_model_call` hides unauthorized tools from the model, and
  `awrap_tool_call` re-checks and returns a `ToolMessage` (not an exception) so the model can
  correct itself. The first layer alone is not a control — a model can invent a tool name.

**The Streamlit event-loop constraint is gone.** It used to be the central limitation here:
Streamlit builds a fresh loop per rerun via `asyncio.run()` and destroys it on exit, so nothing
loop-affine could survive in `st.session_state` — which is why the checkpointer was `InMemorySaver`
and the store was a hand-written JSON dumper. Now that the agent lives in uvicorn, the constraint
dissolved: `app.py` contains no `asyncio.run()` at all, and both connection pools live for the
lifetime of the process. Do not reintroduce agent construction into the UI process.

**Business tables and framework tables are separate, and Alembic only manages the former.**
checkpoint/store tables are the framework's internal serialization format and change across
versions; business queries must never join against them. `messages` holds the copy we display and
audit, the checkpoint holds what the agent needs to resume — different lifecycles, and the
checkpoint is safe to expire.

**MCP tools are safe to keep across requests** because `MultiServerMCPClient` opens a new session
per tool call. Verified: the filesystem server logs its startup banner once per call.


## Tools Available to the Agent (22)

Via MCP `tmdb` (6): `search_movies`, `get_movie_details`, `get_recommendations`,
`discover_movies`, `get_popular_movies`, `get_genres`.
The identical set exists in-process as `tools.TMDB_TOOLS`, used only when the MCP server is down.

Via MCP `watchlist` (14): the `@modelcontextprotocol/server-filesystem` tool set
(`read_text_file`, `write_file`, `edit_file`, `list_directory`, `directory_tree`,
`search_files`, …), sandboxed to `data/watchlist/`.

Local (2): `search_local_knowledge` (RAG), `load_skill` (skill disclosure).

## RAG Details

- Chunking: `chunk_size=800`, `chunk_overlap=100`. Corpus categories: `movie_review`
  (English .txt), `genre_knowledge` (Chinese .txt), `film_books` (PDF).
- Recall is hybrid: vector top-20 + BM25 top-20, deduplicated by `(source, first 200 chars)`
  with the vector route winning ties. Then an LLM rerank selects `k`; on any parse failure or
  timeout it falls back to vector order — retrieval never raises.
- BM25 needs a custom tokenizer: the default splits on whitespace, which does not segment
  Chinese at all. `_tokenize()` emits lowercase Latin words plus CJK unigrams and bigrams.
- `data/index/manifest.json` records per-source `sha256[:16]`/size plus the embedding model and
  chunk parameters. Any mismatch triggers a rebuild, so editing a source file is enough.
- Citations are rendered as `【类型知识 · 赛博朋克电影.txt】`, with a page number for PDFs.
- **Query in the corpus's own language.** Vector search and BM25 fail in opposite directions
  across languages (measured; see README).
- **Scanned PDFs yield zero chunks.** `data/books/电影艺术词典.pdf` is a 607-page image-only
  scan with no text layer — it has never contributed to the index. The build warns per file.
  Do not let prompts or tool docstrings promise film-dictionary content.

## Memory Details

- Namespaces: `("users", uid, "profile")` key `"main"` for the structured profile;
  `("users", uid, "episodes")` key uuid for episode summaries. `uid` is the user's UUID.
- Episodes are semantically searchable — `IndexConfig(dims=384, fields=["text"])` reuses the
  RAG embedding model, and `AsyncPostgresStore` persists the vectors in `store_vectors`. Recall
  works across languages (measured: an English query retrieves a Chinese episode at 0.585 vs
  0.348 for the English one).
- `scripts/migrate_store_to_pg.py` imports the legacy `data/memory/store.json` into a **registered**
  user. It deliberately does not create the account — that would be a second path around
  `/v1/auth/register` — and it keeps the original keys and timestamps, so re-running it is
  idempotent instead of duplicating every episode with today's date.
- Short-term memory now survives a restart (it is in Postgres), but it is still per-`thread_id`
  and still separate from the long-term profile. `DELETE /v1/users/me/profile` clears only the
  profile; episodes are records of things that happened and stay.


## Pitfalls Already Hit (do not regress these)

1. **Fire-and-forget memory writes silently vanish.** The original code ended a turn with
   `asyncio.create_task(extract_and_save_preferences(...))` under `asyncio.run()`. The loop
   closed before the task ran, raising `CancelledError` — which inherits from `BaseException`
   and so slipped straight through `except Exception: pass`. Nothing was ever written and
   nothing was ever logged. Preference extraction now lives in `@after_agent`, inside the
   graph, and its fallback handler writes to stderr instead of passing.

2. **`with_structured_output` defaults to `method="json_schema"`.** That emits
   `response_format={"type": "json_schema"}`, an OpenAI-only extension; DeepSeek returns
   400 `This response_format type is unavailable now`. `LLMToolSelectorMiddleware` uses this
   path, so *every* turn crashed on its first model call. `llm._CompatChatOpenAI` overrides the
   default to `function_calling`.

3. **Thinking mode rejects a forced `tool_choice`.** `function_calling` pins the tool, and
   DeepSeek answers 400 `Thinking mode does not support this tool_choice`. The utility model
   therefore sends `reasoning_effort="none"` — which is also just correct for summarization,
   reranking, and tool selection.

4. **`npx` cold start looks like a dead server.** The first spawn resolves and unpacks the
   filesystem package, slow enough that session init fails with `McpError: Connection closed`,
   leaving the whole Streamlit session degraded. Mitigated twice over: `scripts/dev.sh` warms
   the npm cache with `npx -y -p <pkg> true`, and `load_mcp_tools_safe()` retries once per
   server.

5. **`sentence-transformers` probes files that ignore `HF_ENDPOINT`.** Loading requests
   `adapter_config.json` directly from `huggingface.co` even when the mirror is configured,
   hanging through five retries. `rag._weights_are_cached()` checks the local cache and passes
   `local_files_only=True` when the weights are already there.

6. **`langchain-core` gates all tool callbacks behind `ignore_agent`.** There is no
   `ignore_tool` flag. A tool-observing handler must return `False` from `ignore_agent`.

7. **`Path.relative_to` is a string comparison, not a containment check.**
   `Path(f"{sandbox}/../../evil.md").is_relative_to(sandbox)` is `True`. `confine_path()` must
   `resolve()` first and only then compare, or `../` walks straight out of the sandbox. The
   second version of that function had the opposite bug: it silently rewrote `/etc/passwd` into
   `<sandbox>/etc/passwd`, which is safe but throws away the audit signal that someone tried.
   Both cases are pinned by tests.

8. **Nothing loads `.env` for the uvicorn process.** `movie_agent/*` is a domain layer that reads
   `os.getenv` directly, and it used to be the Streamlit entrypoint that called `load_dotenv()`
   for everyone. After the agent moved into uvicorn, startup died on
   `ValueError: OPENAI_API_KEY must be set`. `server/config.py` now calls `load_dotenv(...,
   override=False)` at import — `override=False` matters, because the test suite points
   `DATABASE_URL` at `movie_agent_test` before importing it and `create_all`/`drop_all` would
   otherwise run against the real database.

9. **Tool callbacks arrive interleaved, so pairing them by arrival order mis-attributes them.**
   The model fires tool calls in parallel batches; three `on_tool_start` before the first
   `on_tool_end` is normal. `ToolDebugHandler` originally wrote each end event into
   `self.calls[-1]` with a single scalar start time, which produced real damage visible in
   `tool_invocations`: of six rows in one turn, three had `NULL` duration and no output, while
   the others held another tool's output timed from an unrelated start. Records are keyed by
   `run_id` now.


## Conventions

- TMDB tool bodies live only in `tmdb_tools.py`; `tools.py` and the MCP server both derive
  their schemas from those signatures and docstrings. Never fork them.
- Tools return plain text, not JSON. TMDB results are capped at 5 items.
- Every tool wraps failures into a readable return string rather than raising — an exception at
  the tool boundary becomes a stack trace in the model's context.
- `load_skill` resolves paths and re-checks that the result is still under `skills/`;
  without that it is an arbitrary-file-read primitive.
- The `chat_stream` event contract is `token` / `tool_start` / `tool_end` / `done`, and the SSE
  endpoint forwards it verbatim plus an `error` event. `app.py` renders against it; keep it stable.
- **Stream failures cannot be reported by status code** — the response headers are already gone by
  the time the first token is sent. They become an `error` event, and the turn is rolled back so a
  failed exchange does not leave a user message with no answer behind it.
- Every error response is `{"error": {code, message, request_id}}`. Login and registration failures
  share one message on purpose, so the API cannot be used to enumerate accounts.
- Sending a message takes a transaction-scoped advisory lock on the `thread_id`. The lock is per
  conversation, not per user — locking per user would mean one person cannot have two tabs open.
- `Idempotency-Key` is unique on `(user_id, key)`. Scoping it globally would let anyone fetch
  someone else's reply by guessing a key.
- `/readyz` probes its dependencies for real (DB reachable, tool count > 0). We have shipped an
  npx cold start that silently degraded MCP, leaving a live process with missing capabilities;
  that state has to be visible to a health check rather than discovered by a user.
- Tests run against `movie_agent_test` with a fake LLM and fake MCP but a **real** Postgres and a
  real store: the security boundaries are all expressed in database semantics, so faking the
  database would fake away the thing under test.
- Streamlit asks for an email on first ever run, which blocks a non-interactive launch. Set
  `STREAMLIT_SERVER_HEADLESS=true` to skip it.
