"""
TMDB API 封装层，不依赖 LlamaIndex，可独立测试：
  python -c "from movie_agent.tmdb_client import get_popular_movies; print(get_popular_movies())"
"""

import os
import requests
from typing import List, Optional, Dict

BASE_URL = "https://api.themoviedb.org/3"


def _get_api_key() -> str:
    key = os.getenv("TMDB_API_KEY", "")
    if not key:
        raise ValueError("TMDB_API_KEY is not set in environment variables.")
    return key


def _get(endpoint: str, params: Optional[Dict] = None) -> Dict:
    params = params or {}
    params["api_key"] = _get_api_key()
    params.setdefault("language", "en-US")
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def _format_movie(m: dict) -> dict:
    """从 TMDB 原始电影对象中提取所需字段。"""
    return {
        "id": m.get("id"),
        "title": m.get("title"),
        "overview": m.get("overview", ""),
        "release_date": m.get("release_date", ""),
        "vote_average": m.get("vote_average", 0),
        "genre_ids": m.get("genre_ids", []),
        "popularity": m.get("popularity", 0),
    }


def search_movies(query: str, year: Optional[int] = None) -> Optional[List[Dict]]:
    """按标题或关键词搜索电影，最多返回 5 条结果。"""
    params = {"query": query, "page": 1}
    if year:
        params["year"] = year
    data = _get("/search/movie", params)
    return [_format_movie(m) for m in data.get("results", [])[:5]]


def get_movie_details(movie_id: int) -> dict:
    """根据 TMDB ID 获取电影详情。"""
    data = _get(f"/movie/{movie_id}")
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "overview": data.get("overview", ""),
        "release_date": data.get("release_date", ""),
        "vote_average": data.get("vote_average", 0),
        "runtime": data.get("runtime"),
        "genres": [g["name"] for g in data.get("genres", [])],
        "tagline": data.get("tagline", ""),
        "status": data.get("status", ""),
    }


def get_recommendations(movie_id: int) -> list[dict]:
    """获取基于指定电影的 TMDB 推荐，最多返回 5 条结果。"""
    data = _get(f"/movie/{movie_id}/recommendations", {"page": 1})
    return [_format_movie(m) for m in data.get("results", [])[:5]]


def discover_movies(
    genre_ids: Optional[List[int]] = None,
    min_rating: Optional[float] = None,
    year: Optional[int] = None,
    sort_by: str = "popularity.desc",
) -> list[dict]:
    """按可选条件筛选发现电影，最多返回 5 条结果。"""
    params = {"page": 1, "sort_by": sort_by}
    if genre_ids:
        params["with_genres"] = ",".join(str(g) for g in genre_ids)
    if min_rating is not None:
        params["vote_average.gte"] = min_rating
        params["vote_count.gte"] = 100  # 过滤投票数过少的冷门电影
    if year:
        params["primary_release_year"] = year
    data = _get("/discover/movie", params)
    return [_format_movie(m) for m in data.get("results", [])[:5]]


def get_popular_movies() -> list[dict]:
    """获取当前最热门的电影，最多返回 5 条结果。"""
    data = _get("/movie/popular", {"page": 1})
    return [_format_movie(m) for m in data.get("results", [])[:5]]


def get_genres() -> list[dict]:
    """获取 TMDB 全部电影类型及其 ID。"""
    data = _get("/genre/movie/list")
    return data.get("genres", [])

if __name__ == '__main__':
    print(get_popular_movies())