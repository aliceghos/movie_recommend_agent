# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A conversational movie recommendation agent built with LlamaIndex ReActAgent + MiniMax LLM + TMDB API + Streamlit.

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Agent framework | LlamaIndex `ReActAgent` | Text-based ReAct loop; MiniMax does not use OpenAI function-calling protocol |
| LLM | MiniMax `MiniMax-M2.7` | Via `llama-index-llms-openai-like` (`OpenAILike`) with `is_function_calling_model=False` |
| Output parser | `MiniMaxOutputParser` | Custom subclass of `ReActOutputParser`; converts MiniMax's non-standard XML tool call format to standard ReAct text format |
| Movie data | TMDB API | `https://api.themoviedb.org/3/` |
| UI | Streamlit | Async chat interface |

## Project Structure

```
movie_recommend_agent/
├── CLAUDE.md
├── README.md
├── .env                       # API keys (never commit)
├── .env.example               # Template for .env
├── requirements.txt
├── app.py                     # Streamlit entry point — run this
└── movie_agent/
    ├── __init__.py
    ├── agent.py               # MiniMaxOutputParser, ReActAgent setup, chat()
    ├── tools.py               # LlamaIndex FunctionTool definitions
    └── tmdb_client.py         # Pure TMDB API wrapper (no LlamaIndex dependency)
```

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

Test TMDB connection independently:
```bash
python -c "from movie_agent.tmdb_client import get_popular_movies; print(get_popular_movies())"
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
MINIMAX_API_KEY=your_minimax_key_here
MINIMAX_GROUP_ID=your_minimax_group_id_here
TMDB_API_KEY=your_tmdb_key_here
```

- **MINIMAX_API_KEY** / **MINIMAX_GROUP_ID** — from [minimaxi.com](https://www.minimaxi.com/)
- **TMDB_API_KEY** — from [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) (free)

## Architecture Notes

### MiniMax Tool Call Format Fix (critical)

MiniMax does not return tool calls in OpenAI JSON format. It emits one of two XML variants:

**Variant 1:**

```
minimax:tool_call <invoke name="search_movies">
<tool_caller_parameters>{"query": "Wuthering Heights"}</tool_caller_parameters>
</invoke>
```

**Variant 2:**

```
minimax:tool_call <action>search_movies</action>
<action_input>{"query": "Wuthering Heights"}</action>
```

`MiniMaxOutputParser` in `agent.py` intercepts both variants, validates parameters with a Pydantic model (`MiniMaxToolCall`), and reconstructs them as standard ReAct `Thought/Action/Action Input` text before passing to the parent `ReActOutputParser`.

`is_function_calling_model=False` is set on `OpenAILike` to ensure the agent uses the text-based ReAct loop instead of OpenAI structured function-calling mode.

### Other Notes

- `tmdb_client.py` has no LlamaIndex dependency — it can be tested independently
- `tools.py` wraps tmdb_client functions as `FunctionTool` objects; docstrings are the tool descriptions the LLM reads to decide which tool to call
- The agent is initialized once per Streamlit session (`st.session_state.agent`) to avoid re-creation on every rerun
- `chat()` in `agent.py` is `async`; `app.py` calls it via `asyncio.run()`
