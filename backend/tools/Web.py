from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

import requests
from rank_bm25 import BM25Okapi

from .RAG import (
    RAGChunk,
    RAGConfig,
    SearchHit,
    _encode,
    _load_embedder,
    _rank_dense_multi,
    _rank_sparse_multi,
    _rerank,
    _rrf,
    _tokenize,
    infer_project_root,
    normalize_queries,
    resolve_multi_query_count,
)

SearchType = Literal["search", "news", "images", "videos", "places", "maps", "shopping", "scholar", "patents", "autocomplete"]


class WebSearchError(RuntimeError):
    pass


@dataclass(slots=True)
class WebResult:
    title: str
    link: str = ""
    snippet: str = ""
    source: str = ""
    date: str = ""
    position: int = 0
    result_type: str = "organic"
    image_url: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WebSearchOutput:
    query: str
    search_type: str
    answer: str
    results: list[WebResult]
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "query": self.query,
            "search_type": self.search_type,
            "answer": self.answer,
            "results": [item.to_dict() for item in self.results],
        }
        if include_raw:
            data["raw"] = self.raw
        return data

    def format_for_llm(self, max_results: int = 8) -> str:
        lines = [f"Query: {self.query}", f"Search Type: {self.search_type}"]
        if self.answer:
            lines += ["", "Direct Answer / Summary:", self.answer]
        if self.results:
            lines += ["", "Search Results:"]
        for index, item in enumerate(self.results[:max_results], start=1):
            lines.append(f"[{index}] {item.title}")
            if item.link:
                lines.append(f"URL: {item.link}")
            meta = "; ".join(part for part in [f"source={item.source}" if item.source else "", f"date={item.date}" if item.date else "", f"type={item.result_type}"] if part)
            if meta:
                lines.append("Meta: " + meta)
            if item.snippet:
                lines.append(f"Snippet: {item.snippet}")
            if item.image_url:
                lines.append(f"Image: {item.image_url}")
            lines.append("")
        return "\n".join(lines).strip()


class SerperClient:
    BASE_URL = "https://google.serper.dev"
    RESULT_KEYS = {
        "search": ["organic", "peopleAlsoAsk", "relatedSearches"],
        "news": ["news"],
        "images": ["images"],
        "videos": ["videos"],
        "places": ["places"],
        "maps": ["places"],
        "shopping": ["shopping"],
        "scholar": ["organic", "scholar"],
        "patents": ["organic", "patents"],
        "autocomplete": ["suggestions"],
    }

    def __init__(self, api_key: str | None = None, timeout: int = 20, default_gl: str = "us", default_hl: str = "en") -> None:
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.timeout = timeout
        self.default_gl = default_gl
        self.default_hl = default_hl
        if not self.api_key:
            raise WebSearchError("SERPER_API_KEY is not set.")

    def search(
        self,
        query: str,
        search_type: SearchType = "search",
        num: int = 8,
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        tbs: str | None = None,
        page: int | None = None,
        include_raw: bool = True,
    ) -> WebSearchOutput:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")
        if search_type not in self.RESULT_KEYS:
            raise ValueError(f"Unsupported search_type: {search_type}")
        payload: Dict[str, Any] = {"q": query, "num": max(1, min(int(num), 20)), "gl": gl or self.default_gl, "hl": hl or self.default_hl}
        if location:
            payload["location"] = location
        if tbs:
            payload["tbs"] = tbs
        if page is not None:
            payload["page"] = page
        raw = self._post(search_type, payload)
        return WebSearchOutput(query=query, search_type=search_type, answer=self._extract_answer(raw), results=self._parse_results(raw, search_type, payload["num"]), raw=raw if include_raw else {})

    def _post(self, search_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.BASE_URL}/{search_type}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WebSearchError("Serper response must be a JSON object.")
        return data

    @staticmethod
    def _extract_answer(raw: Dict[str, Any]) -> str:
        parts: list[str] = []
        answer_box = raw.get("answerBox")
        if isinstance(answer_box, dict):
            parts.extend(str(answer_box.get(key) or "") for key in ("title", "answer", "snippet"))
        knowledge_graph = raw.get("knowledgeGraph")
        if isinstance(knowledge_graph, dict):
            title_type = " - ".join(str(knowledge_graph.get(key) or "") for key in ("title", "type") if knowledge_graph.get(key))
            parts.extend([title_type, str(knowledge_graph.get("description") or "")])
        return "\n".join(part.strip() for part in parts if part and part.strip())

    def _parse_results(self, raw: Dict[str, Any], search_type: str, limit: int) -> list[WebResult]:
        parsed: list[WebResult] = []
        for key in self.RESULT_KEYS[search_type]:
            for item in raw.get(key) or []:
                if isinstance(item, dict):
                    result = self._parse_item(item, key)
                    if result.title:
                        parsed.append(result)
                    if len(parsed) >= limit:
                        return parsed
        return parsed

    @staticmethod
    def _parse_item(item: Dict[str, Any], result_type: str) -> WebResult:
        title = str(item.get("title") or item.get("name") or item.get("value") or item.get("question") or "").strip()
        link = str(item.get("link") or item.get("url") or item.get("website") or item.get("sourceLink") or "").strip()
        snippet = str(item.get("snippet") or item.get("description") or item.get("answer") or "").strip()
        source = str(item.get("source") or item.get("domain") or item.get("publisher") or "").strip()
        date = str(item.get("date") or item.get("publishedDate") or item.get("year") or "").strip()
        image_url = str(item.get("imageUrl") or item.get("thumbnailUrl") or item.get("thumbnail") or "").strip()
        position = int(item.get("position") or 0)
        ignored = {"title", "name", "value", "question", "link", "url", "website", "sourceLink", "snippet", "description", "answer", "source", "domain", "publisher", "date", "publishedDate", "year", "imageUrl", "thumbnailUrl", "thumbnail", "position"}
        return WebResult(title=title, link=link, snippet=snippet, source=source, date=date, position=position, result_type=result_type, image_url=image_url, extra={k: v for k, v in item.items() if k not in ignored})


_CLIENT: SerperClient | None = None


def get_serper_client() -> SerperClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = SerperClient()
    return _CLIENT


def freshness_to_tbs(freshness: str | None) -> str | None:
    return {None: None, "h": "qdr:h", "hour": "qdr:h", "d": "qdr:d", "day": "qdr:d", "w": "qdr:w", "week": "qdr:w", "m": "qdr:m", "month": "qdr:m", "y": "qdr:y", "year": "qdr:y"}.get(freshness, freshness)


def _web_hybrid_config(k: int, multi_query_count: Any = None) -> RAGConfig:
    root = infer_project_root()
    model_dir = root / "models"
    query_count = resolve_multi_query_count(multi_query_count or os.getenv("AGENT_MULTI_QUERY_COUNT", os.getenv("RETRIEVAL_MULTI_QUERY_COUNT", "3")))
    return RAGConfig(
        project_root=root,
        knowledge_base_path=Path(),
        index_dir=Path(),
        chunks_path=Path(),
        bm25_path=Path(),
        embeddings_path=Path(),
        embedding_model_path=os.getenv("EMBEDDING_MODEL_PATH", str(model_dir / "bge-m3")),
        reranker_model_path=os.getenv("RERANK_MODEL_PATH", str(model_dir / "bge-reranker-v2-m3")),
        top_k_bm25=int(os.getenv("WEB_TOP_K_BM25", os.getenv("RAG_TOP_K_BM25", "20"))),
        top_k_vector=int(os.getenv("WEB_TOP_K_VECTOR", os.getenv("RAG_TOP_K_VECTOR", "20"))),
        top_k_fused=int(os.getenv("WEB_TOP_K_FUSED", os.getenv("RAG_TOP_K_FUSED", "30"))),
        top_k_rerank=max(1, int(k)),
        rrf_k=int(os.getenv("WEB_RRF_K", os.getenv("RAG_RRF_K", "60"))),
        rerank_batch_size=int(os.getenv("RAG_RERANK_BATCH_SIZE", "8")),
        rerank_max_length=int(os.getenv("RAG_RERANK_MAX_LENGTH", "512")),
        confidence_threshold=0.0,
        multi_query_count=query_count,
    )


def _web_text(result: WebResult) -> str:
    return "\n".join(
        part
        for part in (
            result.title,
            result.snippet,
            result.source,
            result.date,
            result.result_type,
        )
        if part
    )


def _dedupe_web_results(results: list[WebResult]) -> list[WebResult]:
    deduped: dict[str, WebResult] = {}
    for item in results:
        key = item.link or f"{item.title}\n{item.snippet}"
        deduped.setdefault(key, item)
    return list(deduped.values())


def _rank_web_results(query: str, query_list: list[str], results: list[WebResult], config: RAGConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = [_web_text(item) for item in results]
    chunks = [
        RAGChunk(
            chunk_id=f"web-{index}",
            doc_id=f"web-{index}",
            source_path=item.link,
            title=item.title,
            text=documents[index],
            chunk_index=index,
            metadata={"web_result": item.to_dict()},
        )
        for index, item in enumerate(results)
    ]
    bm25_bundle = {"bm25": BM25Okapi([_tokenize(text) for text in documents])}
    embeddings = _encode(_load_embedder(config.embedding_model_path), documents)
    dense = _rank_dense_multi(query_list, embeddings, config)
    sparse = _rank_sparse_multi(query_list, bm25_bundle, config)
    merged = _rrf(dense, sparse, config)
    fused_hits = [
        SearchHit(
            chunk=chunks[idx],
            rrf_score=float(values["rrf_score"]),
            vector_score=float(values.get("vector_score") or 0.0),
            bm25_score=float(values.get("bm25_score") or 0.0),
            vector_rank=values.get("vector_rank"),
            bm25_rank=values.get("bm25_rank"),
            hit_sources=tuple(sorted(values.get("hit_sources") or ())),
            matched_queries=tuple(values.get("matched_queries") or ()),
        )
        for idx, values in sorted(merged.items(), key=lambda item: item[1]["rrf_score"], reverse=True)[: config.top_k_fused]
    ]
    hits = _rerank(query, fused_hits, config)[: config.top_k_rerank]
    ranked_results: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        row = dict(hit.chunk.metadata.get("web_result") or {})
        row.update(
            {
                "url": row.get("link", ""),
                "score": round(float(hit.reranker_score), 6),
                "rrf_score": round(float(hit.rrf_score), 6),
                "reranker_score": round(float(hit.reranker_score), 6),
                "reranker_rank": rank,
                "vector_score": round(float(hit.vector_score), 6),
                "bm25_score": round(float(hit.bm25_score), 6),
                "vector_rank": hit.vector_rank,
                "bm25_rank": hit.bm25_rank,
                "retrieval_source": "+".join(hit.hit_sources),
                "matched_queries": list(hit.matched_queries),
            }
        )
        ranked_results.append(row)
    trace = {
        "web_search_called": True,
        "bm25_executed": True,
        "bm25_candidate_count": sum(len(run) for _, run in sparse),
        "vector_executed": True,
        "vector_candidate_count": sum(len(run) for _, run in dense),
        "rrf_executed": True,
        "rrf_fused_count": len(fused_hits),
        "reranker_executed": True,
        "reranker_model_path": config.reranker_model_path,
        "reranker_top_score": round(float(hits[0].reranker_score), 6) if hits else 0.0,
        "final_result_count": len(ranked_results),
        "multi_query_count_configured": config.multi_query_count,
        "multi_query_count_used": len(query_list),
        "queries_used": query_list,
        "final_information_source": "web search",
    }
    return ranked_results, trace


def _format_ranked_web_results(query: str, search_type: str, answer: str, results: list[dict[str, Any]], max_results: int) -> str:
    lines = [f"Query: {query}", f"Search Type: {search_type}"]
    if answer:
        lines += ["", "Direct Answer / Summary:", answer]
    if results:
        lines += ["", "Search Results:"]
    for index, item in enumerate(results[:max_results], start=1):
        lines.append(f"[{index}] {item.get('title') or 'Untitled'}")
        if item.get("link"):
            lines.append(f"URL: {item['link']}")
        meta = "; ".join(part for part in [f"source={item.get('source')}" if item.get("source") else "", f"date={item.get('date')}" if item.get("date") else "", f"type={item.get('result_type') or 'web'}", f"reranker_score={float(item.get('reranker_score') or 0):.4f}"] if part)
        lines.append("Meta: " + meta)
        if item.get("snippet"):
            lines.append(f"Snippet: {item['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()


def web_search(
    query: str,
    k: int = 6,
    search_type: SearchType = "search",
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    freshness: str | None = None,
    return_json: bool = False,
    include_raw: bool = False,
    queries: list[str] | None = None,
    multi_query_count: Any = None,
) -> str | Dict[str, Any]:
    started_at = time.perf_counter()
    config = _web_hybrid_config(k, multi_query_count)
    query_list = normalize_queries(query, queries, config.multi_query_count)
    outputs = [
        get_serper_client().search(query=item, search_type=search_type, num=k, gl=gl, hl=hl, location=location, tbs=freshness_to_tbs(freshness), include_raw=include_raw)
        for item in query_list
    ]
    direct_answer = "\n".join(item.answer.strip() for item in outputs if item.answer.strip())
    raw_results = _dedupe_web_results([result for output in outputs for result in output.results])
    ranked_results, retrieval_trace = _rank_web_results(query, query_list, raw_results, config) if raw_results else ([], {"web_search_called": True, "bm25_executed": False, "vector_executed": False, "rrf_executed": False, "reranker_executed": False, "final_result_count": 0, "multi_query_count_configured": config.multi_query_count, "multi_query_count_used": len(query_list), "queries_used": query_list, "final_information_source": "web search"})
    answer = _format_ranked_web_results(query, search_type, direct_answer, ranked_results, k)
    if not return_json:
        return _format_ranked_web_results(query, search_type, direct_answer, ranked_results, k)
    data = {
        "query": query,
        "queries": query_list,
        "search_type": search_type,
        "answer": answer,
        "results": ranked_results,
    }
    if include_raw:
        data["raw"] = [item.raw for item in outputs]
    references = [
        {
            "id": idx,
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "link": item.get("link", ""),
            "source": item.get("source", ""),
            "date": item.get("date", ""),
            "snippet": item.get("snippet", ""),
            "result_type": item.get("result_type", ""),
            "score": float(item.get("score") or 0.0),
            "rrf_score": float(item.get("rrf_score") or 0.0),
            "reranker_score": float(item.get("reranker_score") or 0.0),
        }
        for idx, item in enumerate(ranked_results, start=1)
    ]
    data.update({
        "status": "ok",
        "tool_name": "web_search",
        "answer": answer,
        "message": answer,
        "local_answer": answer,
        "artifacts": [],
        "references": references,
        "metrics": {
            "tool_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "result_count": len(ranked_results),
            "raw_result_count": len(raw_results),
            "bm25_candidate_count": retrieval_trace.get("bm25_candidate_count", 0),
            "vector_candidate_count": retrieval_trace.get("vector_candidate_count", 0),
            "rrf_fused_count": retrieval_trace.get("rrf_fused_count", 0),
            "reranker_top_score": retrieval_trace.get("reranker_top_score", 0.0),
        },
        "meta": {
            "search_type": search_type,
            "gl": gl,
            "hl": hl,
            "location": location or "",
            "freshness": freshness or "",
            "include_raw": include_raw,
            "retrieval_params": {
                "top_k_bm25": config.top_k_bm25,
                "top_k_vector": config.top_k_vector,
                "rrf_k": config.rrf_k,
                "top_k_fused": config.top_k_fused,
                "top_k_rerank": config.top_k_rerank,
                "multi_query_count": config.multi_query_count,
            },
            "retrieval_trace": retrieval_trace,
        },
        "retrieval_trace": retrieval_trace,
    })
    return data


def web_news_search(query: str, k: int = 6, freshness: str | None = "week", gl: str = "us", hl: str = "en") -> str:
    return str(web_search(query=query, k=k, search_type="news", gl=gl, hl=hl, freshness=freshness))


def web_image_search(query: str, k: int = 6, gl: str = "us", hl: str = "en") -> Dict[str, Any]:
    return dict(web_search(query=query, k=k, search_type="images", gl=gl, hl=hl, return_json=True))
