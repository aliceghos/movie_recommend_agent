"""
LangChain Tool 工具定义。
每个函数的 description 会被 LLM 读取以决定何时调用该工具，因此保持英文描述。
"""

from typing import Optional

from langchain_core.tools import tool

from movie_agent import tmdb_client
from movie_agent.rag import search_knowledge


@tool
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
            f"- [{m['id']}] {m['title']} ({m['release_date'][:4] if m['release_date'] else 'N/A'}) "
            f"⭐ {m['vote_average']:.1f}\n  {m['overview'][:120]}..."
        )
    return "\n".join(lines)


@tool
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
        f"{m['title']} ({m['release_date'][:4] if m['release_date'] else 'N/A'})\n"
        f"Genres: {genres}\n"
        f"Runtime: {runtime}\n"
        f"Rating: ⭐ {m['vote_average']:.1f}/10\n"
        f"Tagline: {m['tagline'] or 'N/A'}\n"
        f"Overview: {m['overview']}"
    )


@tool
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
            f"- [{m['id']}] {m['title']} ({m['release_date'][:4] if m['release_date'] else 'N/A'}) "
            f"⭐ {m['vote_average']:.1f}"
        )
    return "\n".join(lines)


@tool
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
            f"- [{m['id']}] {m['title']} ({m['release_date'][:4] if m['release_date'] else 'N/A'}) "
            f"⭐ {m['vote_average']:.1f}\n  {m['overview'][:120]}..."
        )
    return "\n".join(lines)


@tool
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
            f"- [{m['id']}] {m['title']} ({m['release_date'][:4] if m['release_date'] else 'N/A'}) "
            f"⭐ {m['vote_average']:.1f}"
        )
    return "\n".join(lines)


@tool
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


@tool
def search_local_knowledge(query: str) -> str:
    """Search the local knowledge base for movie reviews, genre knowledge, and film art books.

    Use this tool when the user asks about:
    - Specific movie reviews or critical analysis (e.g. "What do critics say about Inception?")
    - Film genres and their characteristics (e.g. "What is film noir?", "Tell me about cyberpunk films")
    - Film art theory, cinematography techniques, or filmmaking concepts (from curated film art books)
    - Background context about a movie or film style covered in the local library
    - Anything requiring editorial opinion rather than raw TMDB metadata

    The knowledge base currently contains:
    - Reviews: Inception, Harry Potter and the Prisoner of Azkaban, Jane Eyre,
      Spirited Away, Wuthering Heights
    - Genre articles: Cyberpunk films, Film Noir
    - Film books: curated film art and theory books (PDF)

    Args:
        query: The question or topic to look up in the local knowledge base

    Returns:
        Relevant excerpts from reviews, genre articles, or film books, with source labels.
    """
    return search_knowledge(query)


TOOLS = [
    search_movies,
    get_movie_details,
    get_recommendations,
    discover_movies,
    get_popular_movies,
    get_genres,
    search_local_knowledge,
]
