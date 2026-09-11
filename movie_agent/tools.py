"""
本地 LangChain Tool 定义。

改造后这里只剩两类：

- ``LOCAL_TOOLS``：始终加载。RAG 检索与 Skill 加载都在本进程内，没有外部依赖。
- ``TMDB_TOOLS``：**降级用**。正常路径下这 6 个工具由 MCP Server 提供
  （见 ``mcp_servers/tmdb_server.py``），只有 Server 连不上时才挂上本地版本。

两条路径共用 ``movie_agent.tmdb_tools`` 里的同一份实现与 docstring，
所以降级不会改变模型看到的工具描述。

工具 description 会被 LLM 读取以决定何时调用，因此保持英文。
"""

from langchain_core.tools import tool

from movie_agent import tmdb_tools
from movie_agent.rag import search_knowledge
from movie_agent.skills import load_skill


@tool
def search_local_knowledge(query: str) -> str:
    """Search the local knowledge base of film reviews and genre articles.

    Use this tool when the user asks about:
    - Critical opinion or analysis of a specific film ("what do critics say about Inception?")
    - Film genres and their conventions ("what is film noir?", "cyberpunk aesthetics")
    - Aesthetic or tonal vocabulary that TMDB metadata does not capture

    Retrieval combines keyword and semantic recall, then reranks. Query in the source's
    own language for best results: the genre articles are Chinese, the reviews English.

    The corpus covers:
    - Reviews: Inception, Harry Potter and the Prisoner of Azkaban, Jane Eyre,
      Spirited Away, Wuthering Heights
    - Genre articles: cyberpunk film, film noir

    It does NOT cover other films. If your subject is not listed above, say the
    critical layer is unavailable rather than reasoning from loosely related passages.

    Args:
        query: The question or topic to look up

    Returns:
        Relevant passages, each labelled with its source (type · filename · page).
    """
    return search_knowledge(query)


# 本进程内的工具，永远加载
LOCAL_TOOLS = [
    search_local_knowledge,
    load_skill,
]

# MCP Server 不可达时的兜底，与 Server 暴露的是同一份实现
TMDB_TOOLS = [tool(fn) for fn in tmdb_tools.TMDB_FUNCTIONS]
