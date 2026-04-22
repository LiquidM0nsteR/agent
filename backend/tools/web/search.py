from __future__ import annotations

import os
import re
import sys
from typing import Any
from urllib.parse import urlparse

import httpx


def _score_web_result(query: str, title: str, url: str, snippet: str) -> float:
    text = f"{title} {snippet}".lower()
    query_lower = query.lower().strip()
    host = urlparse(url).netloc.lower()
    score = 0.0

    if query_lower:
        query_terms = [term for term in re.split(r"\s+", query_lower) if len(term) >= 2]
        overlap_count = sum(1 for term in query_terms if term in text)
        if overlap_count > 0:
            score += min(6.0, float(overlap_count) * 1.5)

    if host.endswith(".gov") or host.endswith(".edu") or host.endswith(".org"):
        score += 2.0

    return score


def _classify_web_source(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.endswith(".gov") or host.endswith(".edu"):
        return "institution"
    return "web"


async def _fetch_serper_results(
    client: httpx.AsyncClient, api_key: str, query: str
) -> dict[str, Any]:
    response = await client.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "gl": "us", "hl": "zh-cn"},
    )
    response.raise_for_status()
    return response.json()


def _build_web_queries(query: str) -> list[str]:
    return [query]


async def run_web_search_query(query: str) -> dict[str, Any]:
    api_key = os.getenv("SERPER_API_KEY") or "d9753c011e9e1a8d9618a3b4038ff1eb08e66837"
    if not api_key:
        message = "Web search is not configured. Set SERPER_API_KEY to enable it."
        return {
            "status": "unavailable",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
        }

    queries = _build_web_queries(query)
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            organic_results: list[dict[str, Any]] = []
            seen_links: set[str] = set()
            for search_query in queries:
                data = await _fetch_serper_results(client, api_key, search_query)
                for item in data.get("organic") or []:
                    link = item.get("link") or ""
                    if link and link in seen_links:
                        continue
                    if link:
                        seen_links.add(link)
                    organic_results.append(item)
    except httpx.HTTPError as exc:
        print(
            f"[web_search] query={query!r} exc_type={type(exc).__name__} detail={exc}",
            file=sys.stderr,
            flush=True,
        )
        message = f"Web search failed: {exc}"
        return {
            "status": "error",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
        }
    except Exception as exc:
        print(
            f"[web_search] query={query!r} exc_type={type(exc).__name__} detail={exc}",
            file=sys.stderr,
            flush=True,
        )
        message = f"Web search failed: {exc}"
        return {
            "status": "error",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
        }

    ranked_results = sorted(
        organic_results,
        key=lambda item: _score_web_result(
            query,
            item.get("title") or "",
            item.get("link") or "",
            item.get("snippet") or "",
        ),
        reverse=True,
    )

    condensed_results = []
    for item in ranked_results[:5]:
        title = item.get("title") or "Untitled"
        link = item.get("link") or ""
        snippet = item.get("snippet") or ""
        score = _score_web_result(query, title, link, snippet)
        condensed_results.append(
            {
                "title": title,
                "url": link,
                "snippet": snippet,
                "source_tier": _classify_web_source(link),
                "score": score,
            }
        )

    if not condensed_results:
        message = "Web search returned no results."
        return {
            "status": "empty",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
            "provider": "serper",
            "queries": queries,
            "results": [],
            "references": [],
            "possible_answer": "",
        }

    possible_answer = "\n".join(
        f"{index}. [{item['source_tier']}] {item['title']}：{item['snippet']} ({item['url']})"
        for index, item in enumerate(condensed_results[:3], start=1)
        if item["snippet"] or item["url"]
    )

    references = [
        {
            "source_path": item["url"],
            "file_name": item["title"],
            "doc_type": "web",
            "page": None,
            "section": None,
            "chunk_id": None,
            "score": item.get("score", 0.0),
        }
        for item in condensed_results
        if item["url"]
    ]

    return {
        "status": "ok",
        "query": query,
        "provider": "serper",
        "queries": queries,
        "results": condensed_results,
        "possible_answer": possible_answer,
        "answer": possible_answer,
        "local_answer": possible_answer,
        "references": references,
    }
