# CLAUDE.md

## Project Overview

A conversational movie recommendation agent built on LangChain 1.x / LangGraph. It demonstrates
four capabilities wired through framework-native extension points: MCP tools, hybrid RAG,
Agent Skills, and two-tier memory. Users interact via a Streamlit chat UI.

The orchestration is **not** hand-rolled: there is no manual history list, no manual system-prompt
assembly, no manual compression. All of that lives in a middleware chain passed to `create_agent`.

## Architecture

```
Streamlit UI (app.py)
    ↓  asyncio.run(create_movie_agent()) once per session, then chat_stream() per turn
create_agent (movie_agent/agent.py)
    ├─ middleware (movie_agent/middleware.py), in order:
    │    1. SummarizationMiddleware        trigger={"messages": 24}, keep=("messages", 12)
    │    2. LLMToolSelectorMiddleware      only when tool_count > 12; max_tools=6
    │    3. @dynamic_prompt memory_prompt  base prompt + date + skill index + profile + episodes
    │    4. @after_agent  persist_memory   extracts preferences → Store
    ├─ tools
    │    ├─ MCP "tmdb"       → mcp_servers/tmdb_server.py (streamable HTTP, 127.0.0.1:8931)
    │    ├─ MCP "watchlist"  → npx @modelcontextprotocol/server-filesystem → data/watchlist/
    │    ├─ search_local_knowledge → movie_agent/rag.py
    │    └─ load_skill            → movie_agent/skills.py
    ├─ checkpointer: InMemorySaver    (short-term, keyed by thread_id)
    └─ store: PersistentStore        (long-term, data/memory/store.json)
```

## Key Components

| File | Role |
|------|------|
| `app.py` | Streamlit UI. Order is deliberate: keys → agent init → conversation → **sidebar last** (it shows this turn's token count and profile) |
| `movie_agent/agent.py` | `create_movie_agent()` (async, awaits MCP tool discovery) and `chat_stream()` |
| `movie_agent/middleware.py` | The four middleware; `BASE_SYSTEM_PROMPT` lives here |
| `movie_agent/mcp_client.py` | `load_mcp_tools_safe()` → `(tools, warnings)`; per-server failure isolation |
| `mcp_servers/tmdb_server.py` | FastMCP server, standalone process, reads its own `.env` |
| `movie_agent/tmdb_tools.py` | The 6 TMDB tool bodies. Imported by both the MCP server and `tools.py` so schema/docstring exist once |
| `movie_agent/store.py` | `PersistentStore(InMemoryStore)` — overrides `batch`/`abatch`, dumps JSON after writes |
| `movie_agent/skills.py` | `SKILL.md` discovery, frontmatter parsing, `load_skill` tool |
| `movie_agent/rag.py` | Hybrid recall, LLM rerank, index manifest |
| `movie_agent/llm.py` | `get_chat_llm()` (streaming, temp 0.7) / `get_utility_llm()` (deterministic singleton) |
| `data/` | Corpus, index, memory, watchlist — all excluded from git |

## Environment Variables

Required in `.env`:

```
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-flash
TMDB_API_KEY=<key>
PYTHONPATH=.
```

Optional — all have defaults:

```
TMDB_BASE_URL=https://api.tmdb.org/3          # when api.themoviedb.org is unreachable
HF_ENDPOINT=https://hf-mirror.com             # when huggingface.co is unreachable
MCP_TMDB_URL=http://127.0.0.1:8931/mcp
MCP_TMDB_HOST / MCP_TMDB_PORT                 # server-side bind address
MCP_WATCHLIST_DIR=./data/watchlist
MOVIE_AGENT_USER_ID=default
OPENAI_UTILITY_REASONING_EFFORT=none          # empty string = don't send the field
```

## Running the Project

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
python scripts/build_index.py     # do this first; downloads ~450 MB of weights
./scripts/dev.sh                  # MCP server (background) + Streamlit (foreground)
```

`streamlit run app.py` alone is valid: the TMDB MCP server will be unreachable, the agent
falls back to in-process TMDB tools and reports the degradation in the sidebar.

## Hard Constraints

**Streamlit creates a fresh event loop on every rerun** (`asyncio.run()`), and destroys it on
exit. Anything stored in `st.session_state` across turns must therefore hold no loop-affine
resources. Consequences:

- Checkpointer is `InMemorySaver` — pure in-memory, no loop affinity. `AsyncSqliteSaver` binds
  its connection to the creating loop and would break on the next rerun.
- Store is `PersistentStore` — in-memory state, dumped to JSON after each write. `langgraph`
  ships no persistent store backend (`langgraph.store.sqlite` does not exist).
- MCP tools are safe to keep across reruns because `MultiServerMCPClient` opens a **new session
  per tool call**. Verified: the filesystem server logs its startup banner once per call.

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
  `("users", uid, "episodes")` key uuid for episode summaries.
- Episodes are semantically searchable — `IndexConfig(dims=384, fields=["text"])` reuses the
  RAG embedding model. Embeddings are **not** persisted; they are recomputed on load, since
  they are derived data and would bloat the JSON.
- `migrate_legacy_profile()` imports a pre-existing `data/memory/user_profile.json` once.
- Short-term memory dies with the process; that is by design. Long-term survives. The
  "Clear conversation" button resets only the former.

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

## Conventions

- TMDB tool bodies live only in `tmdb_tools.py`; `tools.py` and the MCP server both derive
  their schemas from those signatures and docstrings. Never fork them.
- Tools return plain text, not JSON. TMDB results are capped at 5 items.
- Every tool wraps failures into a readable return string rather than raising — an exception at
  the tool boundary becomes a stack trace in the model's context.
- `load_skill` resolves paths and re-checks that the result is still under `skills/`;
  without that it is an arbitrary-file-read primitive.
- The `chat_stream` event contract is `token` / `tool_start` / `tool_end` / `done`. `app.py`
  renders against it; keep it stable.
- Streamlit asks for an email on first ever run, which blocks a non-interactive launch. Set
  `STREAMLIT_SERVER_HEADLESS=true` to skip it.
