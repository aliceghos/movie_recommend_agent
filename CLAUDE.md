# CLAUDE.md

## Project Overview

This is a conversational AI movie recommendation agent that combines real-time movie data from TMDB with a local curated knowledge base. Users interact via a Streamlit chat UI, and a LangChain ReAct agent intelligently routes requests across multiple tools.

## Architecture

```
Streamlit UI (app.py)
    ↓
async chat() — movie_agent/agent.py
    ↓
LangChain ReAct Agent (LangGraph)
    ├→ TMDB Tools (movie_agent/tools.py → movie_agent/tmdb_client.py → TMDB API)
    └→ RAG Tool (movie_agent/tools.py → movie_agent/rag.py → FAISS + local data/)
```

## Key Components

| File | Role |
|------|------|
| `app.py` | Streamlit web UI, session management |
| `movie_agent/agent.py` | ReAct agent init and async `chat()` loop |
| `movie_agent/tools.py` | 7 LangChain `@tool` definitions |
| `movie_agent/tmdb_client.py` | TMDB REST API wrapper |
| `movie_agent/rag.py` | FAISS-based RAG with local HuggingFace embeddings |
| `data/reviews/` | Movie review documents (excluded from git) |
| `data/knowledge/` | Genre knowledge articles in Chinese (excluded from git) |
| `data/books/` | Film art book PDFs (excluded from git) |
| `data/index/` | FAISS vector index (excluded from git, rebuilt at runtime) |

## Environment Variables

Required in `.env`:

```
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-flash
TMDB_API_KEY=<key>
PYTHONPATH=.
```

Optional, for restricted networks:

```
TMDB_BASE_URL=https://api.tmdb.org/3   # when api.themoviedb.org is unreachable
HF_ENDPOINT=https://hf-mirror.com      # when huggingface.co is unreachable
```

## Running the Project

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## External APIs

- **TMDB**: Movie search, details, recommendations, discovery, genres — base URL `https://api.themoviedb.org/3`, overridable via `TMDB_BASE_URL`
- **LLM**: any OpenAI-compatible endpoint, configured entirely through `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`. Default is DeepSeek (`https://api.deepseek.com/v1`, models `deepseek-flash` and `deepseek-v4-pro`). `stream_usage=True` is required for token accounting under streaming.
- **Embeddings**: run locally via `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), no API call

## RAG Details

- Documents split at chunk_size=800, overlap=100
- FAISS index is a lazy singleton (`rag.py:get_vectorstore()`), built on first query
- Three categories: `movie_review` (English .txt), `genre_knowledge` (Chinese .txt), and `film_books` (PDF art books)
- `data/` directory is excluded from git; the index is rebuilt from source files each session

## Tools Available to the Agent

1. `search_movies` — keyword/title search via TMDB
2. `get_movie_details` — full metadata for a movie ID
3. `get_recommendations` — TMDB-based similar movies
4. `discover_movies` — filter by genre, rating, year, sort
5. `get_popular_movies` — current trending movies
6. `get_genres` — full genre list with IDs (used with `discover_movies`)
7. `search_local_knowledge` — RAG search over reviews, genre articles, and film art books

## Conventions

- TMDB client functions return dicts; results are capped at 5 items to keep agent output manageable
- Tools format output as plain text strings (not JSON) for direct agent consumption
- The agent system prompt instructs including TMDB URLs in responses
- Async: agent uses `ainvoke()`; Streamlit bridges with `asyncio.run()`
- Callbacks: langchain-core gates **all** tool events (`on_tool_start/end/error`) behind
  `ignore_agent`; there is no `ignore_tool` flag. A tool-observing handler must return
  `False` from `ignore_agent`.
