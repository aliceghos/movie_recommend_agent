# Movie Recommendation Agent

A conversational AI assistant for movie recommendations, powered by a LangChain ReAct agent with real-time TMDB data and a local RAG knowledge base.

## Features

- **Natural language queries** — ask in plain English or Chinese
- **Real-time movie data** — search, details, recommendations, discovery via TMDB API
- **Local knowledge base** — curated movie reviews and genre articles retrieved via FAISS vector search
- **Multi-turn conversation** — maintains session history across the conversation
- **Streamlit chat UI** — clean web interface with example prompts

## Architecture

```
Streamlit UI (app.py)
    ↓
LangChain ReAct Agent  (LangGraph)
    ├─ TMDB Tools      → The Movie Database API
    └─ RAG Tool        → FAISS index over local reviews & genre articles
```

The agent talks to any OpenAI-compatible chat endpoint — the default is DeepSeek
(`deepseek-flash`). Swapping providers is a `.env` edit, no code change. Embeddings for RAG
run locally via HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`).

## Prerequisites

- Python 3.12 (managed by [uv](https://docs.astral.sh/uv/))
- A [TMDB API key](https://www.themoviedb.org/settings/api)
- An API key for any OpenAI-compatible provider (default: [DeepSeek](https://platform.deepseek.com/))

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd movie_recommend_agent

# 2. Create the virtual environment and install dependencies
uv venv --python 3.12
uv pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and fill in the values below
```

`.env` file:

```
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-flash
TMDB_API_KEY=your_tmdb_api_key
PYTHONPATH=.
```

Two optional escape hatches for restricted networks:

| Variable | Use when |
|----------|----------|
| `TMDB_BASE_URL=https://api.tmdb.org/3` | `api.themoviedb.org` is unreachable (equivalent official domain) |
| `HF_ENDPOINT=https://hf-mirror.com` | `huggingface.co` is unreachable — needed to download the embedding weights on first run |

## Running

```bash
source .venv/bin/activate
streamlit run app.py
```

Or without activating:

```bash
.venv/bin/streamlit run app.py
```

Open your browser to `http://localhost:8501`.

## Usage Examples

| Intent | Example query |
|--------|--------------|
| Search | `"Find movies about artificial intelligence"` |
| Recommendations | `"I loved Inception, what should I watch next?"` |
| Discovery | `"Show me top-rated sci-fi movies from the 90s"` |
| Trending | `"What are the most popular movies right now?"` |
| Reviews | `"What do critics say about Spirited Away?"` |
| Genre knowledge | `"Tell me about cyberpunk films"` |

## Project Structure

```
movie_recommend_agent/
├── app.py                    # Streamlit UI entry point
├── requirements.txt
├── .env                      # API keys (not committed)
├── movie_agent/
│   ├── agent.py              # ReAct agent and chat loop
│   ├── tools.py              # LangChain tool definitions (7 tools)
│   ├── tmdb_client.py        # TMDB API wrapper
│   └── rag.py                # FAISS-based RAG module
└── data/                     # Local knowledge base (not committed)
    ├── reviews/              # Movie review documents
    ├── knowledge/            # Genre knowledge articles (Chinese)
    └── index/                # FAISS vector index (auto-generated)
```

> `data/` is excluded from git. Add your own `.txt` files to `data/reviews/` or `data/knowledge/` before starting the app — the FAISS index is built automatically on first run.

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `langchain` / `langchain-core` / `langchain-openai` / `langchain-community` | LangChain framework |
| `langgraph` | ReAct agent orchestration |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local embedding model for RAG |
| `pypdf` | Film art book (PDF) parsing |
| `requests` | TMDB HTTP calls |
| `python-dotenv` | Environment variable loading |

## License

MIT
