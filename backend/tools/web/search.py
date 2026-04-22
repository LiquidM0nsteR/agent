from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_LATIN_TERM_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{1,}")
_CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
_CJK_STOP_CHARS = set(
    "的了是在和与及就都而又还把被于让向对给请问帮想要一下一个一些这那其该吗呢吧啊哦噢么什如何怎么为何为什么是否多少哪些有没有什么"
)
_LATIN_STOP_TERMS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "and",
    "or",
    "with",
    "about",
    "from",
    "into",
}
_TRUSTED_WEB_HOST_KEYWORDS = (
    "reuters.com",
    "bbc.",
    "apnews.com",
    "nature.com",
    "science.org",
    "cell.com",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
)


def _normalize_search_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _extract_latin_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in _LATIN_TERM_RE.findall(query.lower()):
        normalized = term.strip()
        if (
            len(normalized) < 2
            or normalized in _LATIN_STOP_TERMS
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _clean_cjk_span(value: str) -> str:
    return "".join(char for char in value if char not in _CJK_STOP_CHARS)


def _extract_cjk_spans(query: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for span in _CJK_SPAN_RE.findall(query):
        cleaned = _clean_cjk_span(span)
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        spans.append(cleaned)
    return spans


def _extract_cjk_ngrams(query: str) -> tuple[list[str], list[str]]:
    bigrams: list[str] = []
    trigrams: list[str] = []
    seen_bigrams: set[str] = set()
    seen_trigrams: set[str] = set()

    for span in _extract_cjk_spans(query):
        for size, bucket, seen in (
            (3, trigrams, seen_trigrams),
            (2, bigrams, seen_bigrams),
        ):
            if len(span) < size:
                continue
            for index in range(len(span) - size + 1):
                gram = span[index : index + size]
                if gram in seen:
                    continue
                seen.add(gram)
                bucket.append(gram)

    return bigrams, trigrams


def _score_web_result(query: str, title: str, url: str, snippet: str) -> float:
    title_text = _normalize_search_text(title)
    snippet_text = _normalize_search_text(snippet)
    combined_text = f"{title_text} {snippet_text}".strip()
    query_lower = _normalize_search_text(query)
    host = urlparse(url).netloc.lower()
    score = 0.0

    if query_lower and query_lower in combined_text:
        score += 3.0

    latin_terms = _extract_latin_terms(query_lower)
    for term in latin_terms:
        if term in title_text:
            score += 1.8
        elif term in snippet_text:
            score += 1.0

    cjk_spans = _extract_cjk_spans(query)
    for span in cjk_spans:
        if span in title_text:
            score += 2.2
        elif span in combined_text:
            score += 1.4

    bigrams, trigrams = _extract_cjk_ngrams(query)
    matched_trigrams = sum(1 for gram in trigrams if gram in combined_text)
    matched_bigrams = sum(1 for gram in bigrams if gram in combined_text)
    if matched_trigrams or matched_bigrams:
        score += min(3.2, matched_trigrams * 0.8 + matched_bigrams * 0.45)

    if host.endswith(".gov") or host.endswith(".edu") or host.endswith(".org"):
        score += 2.0
    elif any(keyword in host for keyword in _TRUSTED_WEB_HOST_KEYWORDS):
        score += 0.8

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
    normalized_query = " ".join((query or "").split())
    if not normalized_query:
        return []

    queries: list[str] = [normalized_query]
    keyword_parts: list[str] = []

    for span in _extract_cjk_spans(normalized_query):
        keyword_parts.append(span[:16])
    keyword_parts.extend(_extract_latin_terms(normalized_query)[:6])

    keyword_query = " ".join(
        part for part in dict.fromkeys(keyword_parts) if len(part) >= 2
    ).strip()
    if keyword_query and keyword_query != normalized_query:
        queries.append(keyword_query)

    return queries[:2]


async def run_web_search_query(query: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    api_key = str(os.getenv("SERPER_API_KEY", "") or "").strip()
    if not api_key:
        message = "Web search is not configured. Set SERPER_API_KEY to enable it."
        return {
            "status": "unavailable",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
            "metrics": {
                "search_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
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
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.warning(
            "[web_search] query=%r exc_type=%s detail=%s search_ms=%.2f",
            query,
            type(exc).__name__,
            exc,
            elapsed_ms,
        )
        message = f"Web search failed: {exc}"
        return {
            "status": "error",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
            "metrics": {"search_ms": elapsed_ms},
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "[web_search] query=%r exc_type=%s detail=%s search_ms=%.2f",
            query,
            type(exc).__name__,
            exc,
            elapsed_ms,
        )
        message = f"Web search failed: {exc}"
        return {
            "status": "error",
            "message": message,
            "answer": message,
            "local_answer": message,
            "query": query,
            "metrics": {"search_ms": elapsed_ms},
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
            "metrics": {
                "search_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
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

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "[web_search] query=%r results=%s search_ms=%.2f",
        query,
        len(condensed_results),
        elapsed_ms,
    )
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
        "metrics": {"search_ms": elapsed_ms},
    }
