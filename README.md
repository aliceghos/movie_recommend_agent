# Movie Recommendation Agent

A conversational movie recommendation assistant powered by **MiniMax M2.7**, **LlamaIndex**, and the **TMDB API**, with a **Streamlit** chat interface.

## Features

- Search movies by title or keyword
- Get similar-movie recommendations based on a reference film
- Discover movies by genre, rating, and release year
- Browse currently popular movies
- TMDB URLs for every recommended film

## Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | MiniMax `MiniMax-M2.7` |
| Agent framework | LlamaIndex `ReActAgent` |
| Movie data | TMDB API |
| UI | Streamlit |

## Quick Start

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys**

Copy `.env.example` to `.env` and fill in your keys:

```env
MINIMAX_API_KEY=your_minimax_key_here
MINIMAX_GROUP_ID=your_minimax_group_id_here
TMDB_API_KEY=your_tmdb_key_here
```

- **MINIMAX_API_KEY** / **MINIMAX_GROUP_ID** — from [minimaxi.com](https://www.minimaxi.com/)
- **TMDB_API_KEY** — free, from [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)

**3. Run the app**

```bash
streamlit run app.py
```

## Example Queries

- `向我推荐几部和《呼啸山庄》风格类似的电影`
- `我喜欢《盗梦空间》，有类似的电影吗？`
- `推荐几部90年代的科幻电影`
- `现在最流行的电影有哪些？`

## Project Structure

```
movie_recommend_agent/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
└── movie_agent/
    ├── agent.py            # ReActAgent + MiniMaxOutputParser
    ├── tools.py            # LlamaIndex FunctionTool definitions
    └── tmdb_client.py      # TMDB API wrapper
```

## Architecture

MiniMax returns tool calls in a non-standard XML format rather than the OpenAI JSON protocol. `MiniMaxOutputParser` (in `agent.py`) intercepts both known variants and reformats them as standard ReAct `Thought/Action/Action Input` text before the agent processes them.

```
User message
     │
     ▼
ReActAgent  ──── LLM call ────►  MiniMax M2.7
     │                               │
     │         XML tool call         │
     │◄──────────────────────────────┘
     │
     ▼
MiniMaxOutputParser
  (regex → Pydantic validation → reconstruct as ReAct text)
     │
     ▼
FunctionTool execution (TMDB API)
     │
     ▼
Final response to user
```
