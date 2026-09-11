# Movie Recommendation Agent

A conversational movie assistant built on **LangChain 1.x / LangGraph**, wired up with the four
capabilities that make an agent more than a prompt: **MCP** tools, **hybrid RAG**, **Agent Skills**,
and **two-tier memory**.

Everything is assembled through framework-native extension points — `create_agent` plus a
middleware chain — rather than a hand-rolled orchestration loop.

## Capabilities

| Capability | How it is implemented | What it buys |
|---|---|---|
| **MCP** | Self-hosted TMDB server (FastMCP, streamable HTTP) + third-party `server-filesystem` (npx, stdio), aggregated by `MultiServerMCPClient` | Tools live behind a protocol boundary, not inside the app. Either server can fail without taking the agent down |
| **RAG** | BM25 + vector hybrid recall (20 + 20) → LLM rerank → cited passages, with an index manifest for staleness detection | Single-route retrieval demonstrably misses; see [Why hybrid recall](#why-hybrid-recall) |
| **Skills** | `skills/*/SKILL.md` with three-level progressive disclosure | Methodology stays out of the system prompt until needed — ~150 tokens resident instead of thousands |
| **Memory** | `InMemorySaver` (short-term, per thread) + `PersistentStore` (long-term, semantic search over episodes) | Preferences survive restarts; conversation history intentionally does not |

22 tools are live at runtime: 6 TMDB (via MCP) + 14 filesystem (via MCP) + 2 local
(`search_local_knowledge`, `load_skill`).

## Architecture

```
Streamlit UI (app.py)
    ↓
create_agent  ──  middleware chain
    │              ├─ SummarizationMiddleware      compresses history past 24 messages
    │              ├─ LLMToolSelectorMiddleware    22 tools → 6 per turn
    │              ├─ @dynamic_prompt              injects profile + skill index + recalled episodes
    │              └─ @after_agent                 extracts preferences, writes to the Store
    ├─ MCP: tmdb server        → mcp_servers/tmdb_server.py → TMDB API
    ├─ MCP: filesystem server  → data/watchlist/  (npx, sandboxed to one directory)
    ├─ search_local_knowledge  → hybrid RAG over data/{reviews,knowledge,books}
    └─ load_skill              → skills/*/SKILL.md
    ↓
checkpointer: InMemorySaver   (short-term, keyed by thread_id)
store:        PersistentStore (long-term, data/memory/store.json)
```

The agent talks to any OpenAI-compatible chat endpoint — the default is DeepSeek
(`deepseek-flash`). Swapping providers is a `.env` edit. Embeddings run locally via
HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim).

## Prerequisites

- Python 3.12 (managed by [uv](https://docs.astral.sh/uv/))
- Node.js + `npx` — for the third-party filesystem MCP server. Without it the watchlist
  tools are unavailable; everything else still works
- A [TMDB API key](https://www.themoviedb.org/settings/api)
- An API key for any OpenAI-compatible provider (default: [DeepSeek](https://platform.deepseek.com/))

## Setup

```bash
git clone <repo-url>
cd movie_recommend_agent

uv venv --python 3.12
uv pip install -r requirements.txt

cp .env.example .env    # then fill in the two API keys
```

`.env` essentials:

```
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-flash
TMDB_API_KEY=your_tmdb_api_key
PYTHONPATH=.
```

Everything else has a working default. Optional overrides:

| Variable | Use when |
|----------|----------|
| `TMDB_BASE_URL=https://api.tmdb.org/3` | `api.themoviedb.org` is unreachable (equivalent official domain) |
| `HF_ENDPOINT=https://hf-mirror.com` | `huggingface.co` is unreachable — needed to download embedding weights on first run |
| `MCP_TMDB_URL` | The TMDB MCP server runs somewhere other than `http://127.0.0.1:8931/mcp` |
| `MCP_WATCHLIST_DIR` | The watchlist should live outside `./data/watchlist` |
| `MOVIE_AGENT_USER_ID` | You want more than one long-term profile on the same machine |
| `OPENAI_UTILITY_REASONING_EFFORT` | Your gateway rejects `reasoning_effort`; set it empty to stop sending it |

## Running

Build the index first — first run downloads ~450 MB of embedding weights, and you would
rather not watch a blank page while that happens:

```bash
source .venv/bin/activate
python scripts/build_index.py
```

Then start the full stack (TMDB MCP server in the background, Streamlit in front):

```bash
./scripts/dev.sh
```

Open `http://localhost:8501`.

Running `streamlit run app.py` on its own also works — the agent detects that the MCP
server is absent, falls back to in-process TMDB tools, and says so in the sidebar.

## Usage Examples

| Intent | Example query | Path exercised |
|--------|--------------|----------------|
| Recommendation | `"I loved Inception, what should I watch next?"` | skill → MCP TMDB → profile write |
| Discovery | `"Show me top-rated sci-fi movies from the 90s"` | MCP TMDB `discover_movies` |
| Film analysis | `"聊聊黑色电影的视觉风格"` | skill → hybrid RAG with citations |
| Watchlist | `"Add Blade Runner to my watchlist"` | skill → filesystem MCP `write_file` |
| Memory | `"我刚才说过我喜欢什么风格？"` | checkpointer + long-term profile |

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
├── app.py                       # Streamlit UI
├── mcp_servers/
│   └── tmdb_server.py           # self-hosted MCP server (FastMCP, streamable HTTP)
├── movie_agent/
│   ├── agent.py                 # create_agent assembly + chat_stream
│   ├── middleware.py            # dynamic_prompt / after_agent / summarization / tool selection
│   ├── mcp_client.py            # multi-server MCP aggregation with per-server fallback
│   ├── store.py                 # PersistentStore — long-term memory on disk
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
├── scripts/
│   ├── dev.sh                   # start MCP server + Streamlit
│   └── build_index.py           # offline index build (--force to rebuild)
└── data/                        # not committed
    ├── reviews/                 # English review .txt
    ├── knowledge/               # Chinese genre articles .txt
    ├── books/                   # PDFs (see note below)
    ├── index/                   # FAISS index + manifest.json
    ├── memory/store.json        # long-term profile and episodes
    └── watchlist/               # written by the filesystem MCP server
```

> `data/` is excluded from git. Drop your own `.txt` files into `data/reviews/` or
> `data/knowledge/` and re-run `scripts/build_index.py`; the manifest detects changed
> sources and rebuilds automatically on the next query too.
>
> **Scanned PDFs contribute nothing.** `pypdf` extracts a text layer, so an image-only
> scan yields zero chunks. The build prints a warning per file when this happens.

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` / `langchain-openai` / `langchain-community` | Agent framework, middleware, model adapters |
| `langgraph` | Graph runtime, checkpointer, store |
| `langchain-mcp-adapters` / `mcp` | MCP client and the FastMCP server |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | Lexical recall for the hybrid retriever |
| `sentence-transformers` | Local embedding model |
| `langchain-text-splitters` | Chunking |
| `pypdf` | PDF text extraction |
| `streamlit` | Web UI |
| `requests` | TMDB HTTP calls |
| `python-dotenv` | Environment loading |

## License

MIT
