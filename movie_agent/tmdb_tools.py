"""
TMDB 工具的纯函数实现 —— 唯一的一份。

这些函数不带任何框架装饰器，签名与 docstring 同时被两处消费：
  - ``movie_agent.tools``   用 ``langchain_core.tools.tool()`` 包成 LangChain Tool（MCP 降级时的兜底）
  - ``mcp_servers.tmdb_server`` 用 ``FastMCP.tool()`` 包成 MCP Tool（正常路径）

两边都从签名和 docstring 推导 schema，因此工具描述只需在此维护一份，
也保证了 MCP 路径与降级路径对模型呈现的行为完全一致。

本模块刻意不 import rag / faiss / sentence-transformers，
这样 MCP Server 子进程可以很轻量地只依赖 requests 启动。
"""

import functools
from typing import Callable, Optional, TypeVar

from movie_agent import tmdb_client

_F = TypeVar("_F", bound=Callable[..., str])


def _safe(fn: _F) -> _F:
    """把异常转成人类可读的返回值。

    工具抛异常时，MCP 协议层会把整个栈回传给模型，既浪费 token 又可能泄漏
    内部路径。统一降级成一句话，模型可以据此改换策略或告知用户。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 边界层，任何失败都要转成文本
            return f"TMDB request failed ({type(exc).__name__}): {exc}"

    return wrapper  # type: ignore[return-value]


def _year_of(movie: dict) -> str:
    release = movie.get("release_date") or ""
    return release[:4] if release else "N/A"


@_safe
def search_movies(query: str, year: Optional[int] = None) -> str:
    """Search for movies by title or keywords.

    Use this tool when the user mentions a specific movie title, asks about movies
    related to a topic or keyword, or wants to find a movie they partially remember.

    Args:
        query: Movie title or keywords to search for (e.g. "Inception", "space adventure")
        year: Optional release year to narrow down results (e.g. 2010)

    Returns:
        A formatted list of matching movies with title, release date, rating, and overview.
    """
    results = tmdb_client.search_movies(query, year)
    if not results:
        return f"No movies found for query: '{query}'."
    lines = [f"Search results for '{query}':"]
    for m in results:
        lines.append(
            f"- [{m['id']}] {m['title']} ({_year_of(m)}) "
            f"⭐ {m['vote_average']:.1f}\n  {m['overview'][:120]}..."
        )
    return "\n".join(lines)


@_safe
def get_movie_details(movie_id: int) -> str:
    """Get detailed information about a specific movie using its TMDB ID.

    Use this tool when you already have a movie's TMDB ID (from a previous search or
    recommendation) and need more information such as genres, runtime, or tagline.

    Args:
        movie_id: The TMDB numeric ID of the movie (e.g. 27205 for Inception)

    Returns:
        Detailed movie info including genres, runtime, rating, and overview.
    """
    m = tmdb_client.get_movie_details(movie_id)
    genres = ", ".join(m["genres"]) if m["genres"] else "N/A"
    runtime = f"{m['runtime']} min" if m["runtime"] else "N/A"
    return (
        f"{m['title']} ({_year_of(m)})\n"
        f"Genres: {genres}\n"
        f"Runtime: {runtime}\n"
        f"Rating: ⭐ {m['vote_average']:.1f}/10\n"
        f"Tagline: {m['tagline'] or 'N/A'}\n"
        f"Overview: {m['overview']}"
    )


@_safe
def get_recommendations(movie_id: int) -> str:
    """Get movie recommendations based on a specific movie.

    Use this tool when the user likes a particular movie and wants to find similar ones,
    or says things like "more like X" or "similar to X".

    Args:
        movie_id: The TMDB numeric ID of the movie to base recommendations on

    Returns:
        A list of recommended movies similar to the given movie.
    """
    results = tmdb_client.get_recommendations(movie_id)
    if not results:
        return "No recommendations found for this movie."
    lines = ["Movies you might also like:"]
    for m in results:
        lines.append(
            f"- [{m['id']}] {m['title']} ({_year_of(m)}) ⭐ {m['vote_average']:.1f}"
        )
    return "\n".join(lines)


@_safe
def discover_movies(
    genre_ids: Optional[str] = None,
    min_rating: Optional[float] = None,
    year: Optional[int] = None,
    sort_by: str = "popularity.desc",
) -> str:
    """Discover movies by filtering on genre, rating, year, and sort order.

    Use this tool when the user asks for movies by genre (e.g. "action movies"),
    wants highly rated films, movies from a specific era, or asks for recommendations
    without a specific reference movie.

    Common genre IDs: Action=28, Comedy=35, Drama=18, Horror=27, Romance=10749,
    Sci-Fi=878, Thriller=53, Animation=16, Documentary=99, Fantasy=14, Crime=80.

    Args:
        genre_ids: Comma-separated TMDB genre IDs as a string (e.g. "28,878" for action sci-fi)
        min_rating: Minimum vote average on a 0–10 scale (e.g. 7.5 for well-rated films)
        year: Filter by release year (e.g. 2023)
        sort_by: Sort order — "popularity.desc", "vote_average.desc", or "release_date.desc"

    Returns:
        A list of movies matching the filters.
    """
    parsed_genre_ids = [int(g.strip()) for g in genre_ids.split(",")] if genre_ids else None
    results = tmdb_client.discover_movies(parsed_genre_ids, min_rating, year, sort_by)
    if not results:
        return "No movies found with those filters."
    lines = ["Movies matching your criteria:"]
    for m in results:
        lines.append(
            f"- [{m['id']}] {m['title']} ({_year_of(m)}) "
            f"⭐ {m['vote_average']:.1f}\n  {m['overview'][:120]}..."
        )
    return "\n".join(lines)


@_safe
def get_popular_movies() -> str:
    """Get the most popular movies right now.

    Use this tool when the user asks what's popular, trending, or currently hot,
    or when they have no specific preferences and want general recommendations.

    Returns:
        A list of currently popular movies.
    """
    results = tmdb_client.get_popular_movies()
    if not results:
        return "Could not fetch popular movies."
    lines = ["Currently popular movies:"]
    for m in results:
        lines.append(
            f"- [{m['id']}] {m['title']} ({_year_of(m)}) ⭐ {m['vote_average']:.1f}"
        )
    return "\n".join(lines)


@_safe
def get_genres() -> str:
    """Get the full list of available movie genres and their TMDB IDs.

    Use this tool when you need to look up a genre ID before calling discover_movies,
    or when the user asks what genres are available.

    Returns:
        A list of all genre names and their corresponding TMDB IDs.
    """
    genres = tmdb_client.get_genres()
    lines = ["Available genres:"]
    for g in genres:
        lines.append(f"  {g['name']} (ID: {g['id']})")
    return "\n".join(lines)


# MCP Server 与本地降级路径共同的注册清单，顺序即工具列表顺序
TMDB_FUNCTIONS = [
    search_movies,
    get_movie_details,
    get_recommendations,
    discover_movies,
    get_popular_movies,
    get_genres,
]
