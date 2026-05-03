from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .RAG_build_index import build_rag_index as _build_rag_index_impl

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".doc"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = DOCUMENT_SUFFIXES | TEXT_SUFFIXES
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


@dataclass(slots=True)
class RAGConfig:
    project_root: Path
    knowledge_base_path: Path
    index_dir: Path
    chunks_path: Path
    bm25_path: Path
    embeddings_path: Path
    embedding_model_path: str
    reranker_model_path: str
    chunk_size: int = 900
    chunk_overlap: int = 180
    top_k_bm25: int = 20
    top_k_vector: int = 20
    top_k_fused: int = 30
    top_k_rerank: int = 6
    rrf_k: int = 60
    rerank_batch_size: int = 8
    rerank_max_length: int = 512
    confidence_threshold: float = 0.58
    multi_query_count: int = 3


@dataclass(slots=True)
class RAGChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    chunk_index: int
    page: int | None = None
    metadata: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RAGChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            source_path=str(data["source_path"]),
            title=str(data["title"]),
            text=str(data["text"]),
            chunk_index=int(data.get("chunk_index") or 0),
            page=data.get("page"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class SearchHit:
    chunk: RAGChunk
    rrf_score: float
    reranker_score: float = 0.0
    reranker_rank: int | None = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    hit_sources: tuple[str, ...] = ()
    matched_queries: tuple[str, ...] = ()

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source_path": self.chunk.source_path,
            "title": self.chunk.title,
            "page": self.chunk.page,
            "chunk_index": self.chunk.chunk_index,
            "score": round(float(self.reranker_score), 6),
            "rrf_score": round(float(self.rrf_score), 6),
            "reranker_score": round(float(self.reranker_score), 6),
            "reranker_rank": self.reranker_rank,
            "vector_score": round(float(self.vector_score), 6),
            "bm25_score": round(float(self.bm25_score), 6),
            "dense_score": round(float(self.vector_score), 6),
            "sparse_score": round(float(self.bm25_score), 6),
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.vector_rank,
            "sparse_rank": self.bm25_rank,
            "hit_sources": list(self.hit_sources),
            "retrieval_source": "+".join(self.hit_sources),
            "matched_queries": list(self.matched_queries),
            "metadata": dict(self.chunk.metadata or {}),
        }
        if include_text:
            payload["text"] = self.chunk.text
        return payload


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_rag_config(knowledge_base_path: str = "", index_dir: str = "", project_root: str = "", embedding_model_path: str = "") -> RAGConfig:
    root = Path(project_root).expanduser().resolve() if project_root else infer_project_root()
    data_dir = root / "data"
    kb_path = Path(knowledge_base_path).expanduser().resolve() if knowledge_base_path else data_dir / "local_knowledge"
    idx_dir = Path(index_dir).expanduser().resolve() if index_dir else data_dir / "local_knowledge_index"
    local_bge = root / "models" / "bge-m3"
    local_reranker = root / "models" / "bge-reranker-v2-m3"
    emb_path = embedding_model_path or os.getenv("EMBEDDING_MODEL_PATH") or (str(local_bge) if local_bge.exists() else DEFAULT_EMBEDDING_MODEL_NAME)
    reranker_path = os.getenv("RERANK_MODEL_PATH") or (str(local_reranker) if local_reranker.exists() else "BAAI/bge-reranker-v2-m3")
    return RAGConfig(
        project_root=root,
        knowledge_base_path=kb_path,
        index_dir=idx_dir,
        chunks_path=idx_dir / "chunks.jsonl",
        bm25_path=idx_dir / "bm25.pkl",
        embeddings_path=idx_dir / "embeddings.npy",
        embedding_model_path=emb_path,
        reranker_model_path=reranker_path,
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "180")),
        top_k_bm25=int(os.getenv("RAG_TOP_K_BM25", os.getenv("RAG_SPARSE_TOP_K", "20"))),
        top_k_vector=int(os.getenv("RAG_TOP_K_VECTOR", os.getenv("RAG_DENSE_TOP_K", "20"))),
        top_k_fused=int(os.getenv("RAG_TOP_K_FUSED", "30")),
        top_k_rerank=int(os.getenv("RAG_TOP_K_RERANK", os.getenv("RAG_FINAL_TOP_K", "6"))),
        rrf_k=int(os.getenv("RAG_RRF_K", "60")),
        rerank_batch_size=int(os.getenv("RAG_RERANK_BATCH_SIZE", "8")),
        rerank_max_length=int(os.getenv("RAG_RERANK_MAX_LENGTH", "512")),
        confidence_threshold=float(os.getenv("AGENT_RAG_CONFIDENCE_THRESHOLD", os.getenv("RAG_RERANKER_CONFIDENCE_THRESHOLD", "0.58"))),
        multi_query_count=resolve_multi_query_count(os.getenv("AGENT_MULTI_QUERY_COUNT", os.getenv("RETRIEVAL_MULTI_QUERY_COUNT", "3"))),
    )


def _stable_id(text: str, n: int = 20) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _normalize_text(text: Any) -> str:
    value = str(text or "").replace("\x00", " ")
    value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def resolve_multi_query_count(value: Any = None, default: int = 3) -> int:
    raw = default if value is None or isinstance(value, bool) else value
    count = raw if isinstance(raw, int) else int(str(raw).strip()) if str(raw).strip().isdecimal() else int(default)
    return max(1, min(count, 8))


def normalize_queries(query: str, queries: Sequence[Any] | None = None, count: Any = None) -> list[str]:
    limit = resolve_multi_query_count(count)
    values: list[str] = []
    for item in [query, *(list(queries or []))]:
        normalized = _normalize_text(item)
        if normalized and normalized not in values:
            values.append(normalized)
        if len(values) >= limit:
            break
    return values


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        return {_clean_json_value(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="ignore") as handle:
        for row in rows:
            handle.write(json.dumps(_clean_json_value(row), ensure_ascii=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _iter_files(base_path: Path, files: list[str] | None = None) -> list[Path]:
    base = base_path.resolve()
    if files:
        resolved = []
        for item in files:
            path = Path(item).expanduser()
            path = path.resolve() if path.is_absolute() else (base_path / path).resolve()
            path.relative_to(base)
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                resolved.append(path)
        return sorted(set(resolved), key=lambda p: p.as_posix())
    return sorted([p.resolve() for p in base_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES], key=lambda p: p.as_posix())


def _extract_text(path: Path) -> list[tuple[str, int | None, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return [(_normalize_text(path.read_text(encoding="utf-8", errors="ignore")), None, {"block_type": "text"})]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(_normalize_text(page.extract_text() or ""), index, {"block_type": "pdf_text"}) for index, page in enumerate(reader.pages, start=1)]
    if suffix == ".docx":
        from docx import Document

        document = Document(str(path))
        return [(_normalize_text("\n".join(p.text for p in document.paragraphs)), None, {"block_type": "docx_text"})]
    raise ValueError(f"不支持的文件类型：{suffix}")


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


@lru_cache(maxsize=2)
def _load_embedder(model_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_path)


def _encode(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def build_rag_index(knowledge_base_path: str = "", index_dir: str = "", files: list[str] | None = None, clean: bool = True, **_: Any) -> dict[str, Any]:
    return _build_rag_index_impl(knowledge_base_path=knowledge_base_path, index_dir=index_dir, files=files, clean=clean)


def _load_index(config: RAGConfig) -> tuple[list[RAGChunk], dict[str, Any], np.ndarray]:
    chunks = [RAGChunk.from_json(row) for row in _read_jsonl(config.chunks_path)]
    with config.bm25_path.open("rb") as handle:
        bm25_bundle = pickle.load(handle)
    embeddings = np.load(config.embeddings_path)
    return chunks, bm25_bundle, embeddings


def _rank_dense(query: str, chunks: list[RAGChunk], embeddings: np.ndarray, config: RAGConfig) -> list[tuple[int, float]]:
    query_vector = _encode(_load_embedder(config.embedding_model_path), [query])[0]
    scores = embeddings @ query_vector
    order = np.argsort(-scores)[: config.top_k_vector]
    return [(int(idx), float(scores[idx])) for idx in order]


def _rank_sparse(query: str, bm25_bundle: dict[str, Any], config: RAGConfig) -> list[tuple[int, float]]:
    scores = bm25_bundle["bm25"].get_scores(_tokenize(query))
    order = np.argsort(-scores)[: config.top_k_bm25]
    return [(int(idx), float(scores[idx])) for idx in order]


def _rank_dense_multi(queries: Sequence[str], embeddings: np.ndarray, config: RAGConfig) -> list[tuple[str, list[tuple[int, float]]]]:
    query_vectors = _encode(_load_embedder(config.embedding_model_path), list(queries))
    rankings: list[tuple[str, list[tuple[int, float]]]] = []
    for query, query_vector in zip(queries, query_vectors):
        scores = embeddings @ query_vector
        order = np.argsort(-scores)[: config.top_k_vector]
        rankings.append((query, [(int(idx), float(scores[idx])) for idx in order]))
    return rankings


def _rank_sparse_multi(queries: Sequence[str], bm25_bundle: dict[str, Any], config: RAGConfig) -> list[tuple[str, list[tuple[int, float]]]]:
    return [(query, _rank_sparse(query, bm25_bundle, config)) for query in queries]


def _rrf(
    dense: list[tuple[str, list[tuple[int, float]]]],
    sparse: list[tuple[str, list[tuple[int, float]]]],
    config: RAGConfig,
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for source, runs in (("vector", dense), ("bm25", sparse)):
        for run_query, ranked in runs:
            for rank, (idx, raw_score) in enumerate(ranked, start=1):
                item = merged.setdefault(
                    idx,
                    {
                        "rrf_score": 0.0,
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "vector_rank": None,
                        "bm25_rank": None,
                        "hit_sources": set(),
                        "matched_queries": set(),
                    },
                )
                item["rrf_score"] += 1.0 / (config.rrf_k + rank)
                item[f"{source}_score"] = max(float(item[f"{source}_score"]), float(raw_score))
                item[f"{source}_rank"] = rank if item[f"{source}_rank"] is None else min(int(item[f"{source}_rank"]), rank)
                item["hit_sources"].add(source)
                item["matched_queries"].add(run_query)
    return merged


def _rank_trace_results(runs: Sequence[tuple[str, list[tuple[int, float]]]], chunks: list[RAGChunk], score_name: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for matched_query, run in runs:
        for rank, (chunk_index, score) in enumerate(run, start=1):
            chunk = chunks[int(chunk_index)]
            results.append({
                "rank": rank,
                "query": matched_query,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "source_path": chunk.source_path,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                score_name: round(float(score), 6),
            })
    return results


@lru_cache(maxsize=1)
def _load_reranker(model_path: str) -> tuple[Any, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _rerank(query: str, hits: list[SearchHit], config: RAGConfig) -> list[SearchHit]:
    if not hits:
        return []
    tokenizer, model, device = _load_reranker(config.reranker_model_path)
    raw_scores: list[float] = []
    pairs = [(query, hit.chunk.text) for hit in hits]
    batch_size = max(1, int(config.rerank_batch_size))
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=config.rerank_max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits.detach().float().cpu().numpy().reshape(-1)
            raw_scores.extend(float(item) for item in logits)
    normalized = _sigmoid(np.asarray(raw_scores, dtype=np.float32))
    reranked = [
        SearchHit(
            chunk=hit.chunk,
            rrf_score=hit.rrf_score,
            reranker_score=float(score),
            vector_score=hit.vector_score,
            bm25_score=hit.bm25_score,
            vector_rank=hit.vector_rank,
            bm25_rank=hit.bm25_rank,
            hit_sources=hit.hit_sources,
            matched_queries=hit.matched_queries,
        )
        for hit, score in zip(hits, normalized)
    ]
    reranked.sort(key=lambda item: item.reranker_score, reverse=True)
    for rank, hit in enumerate(reranked, start=1):
        hit.reranker_rank = rank
    return reranked


_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "what", "how", "why", "when", "where", "which", "please",
    "请", "什么", "如何", "为什么", "以及", "一个", "这个", "那个", "进行", "分析", "说明",
}


def _content_token_coverage(query: str, hits: list[SearchHit]) -> float:
    query_tokens = [token for token in _tokenize(query) if token not in _QUERY_STOPWORDS and len(token) > 1]
    if not query_tokens:
        return 0.0
    joined = "\n".join(hit.chunk.text.lower() for hit in hits[:3])
    covered = sum(1 for token in query_tokens if token in joined)
    return covered / max(1, len(query_tokens))


def _rag_confidence(query: str, hits: list[SearchHit], config: RAGConfig) -> tuple[float, str]:
    if not hits:
        return 0.0, "没有可用的 rerank 后 chunk。"
    scores = [float(hit.reranker_score) for hit in hits[: max(1, min(3, len(hits)))]]
    top_score = scores[0]
    top_mean = float(np.mean(scores))
    coverage = _content_token_coverage(query, hits)
    support = min(1.0, len([hit for hit in hits[:5] if hit.reranker_score >= max(0.0, top_score - 0.15)]) / 3.0)
    confidence = 0.55 * top_score + 0.20 * top_mean + 0.20 * coverage + 0.05 * support
    confidence = max(0.0, min(1.0, float(confidence)))
    verdict = "sufficient" if confidence >= config.confidence_threshold else "insufficient"
    reason = (
        f"{verdict}: top reranker score={top_score:.3f}, top_mean={top_mean:.3f}, "
        f"query_token_coverage={coverage:.3f}, support={support:.3f}, "
        f"threshold={config.confidence_threshold:.3f}"
    )
    return confidence, reason


def run_rag(
    query: str,
    knowledge_base_path: str = "",
    index_dir: str = "",
    files: list[str] | None = None,
    history: Any = None,
    queries: Sequence[Any] | None = None,
    multi_query_count: Any = None,
    **_: Any,
) -> dict[str, Any]:
    del history
    started_at = time.perf_counter()
    config = get_rag_config(knowledge_base_path, index_dir)
    if files or not (config.chunks_path.exists() and config.bm25_path.exists() and config.embeddings_path.exists()):
        build_rag_index(knowledge_base_path=knowledge_base_path, index_dir=index_dir, files=files, clean=True)
    chunks, bm25_bundle, embeddings = _load_index(config)
    if len(chunks) != int(embeddings.shape[0]):
        build_rag_index(knowledge_base_path=knowledge_base_path, index_dir=index_dir, files=files, clean=True)
        chunks, bm25_bundle, embeddings = _load_index(config)
    query_count = resolve_multi_query_count(multi_query_count, config.multi_query_count)
    query_list = normalize_queries(query, queries, query_count)
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
    reranked_hits = _rerank(query, fused_hits, config)
    reranker_executed = True
    reranker_error = ""
    hits = reranked_hits[: config.top_k_rerank]
    confidence, confidence_reason = _rag_confidence(query, hits, config)
    references = [hit.to_dict(include_text=False) | {"id": index, "url": hit.chunk.source_path} for index, hit in enumerate(hits, start=1)]
    context = "\n\n".join(
        f"[{index}] {hit.chunk.title} p.{hit.chunk.page or '-'} "
        f"(sources={'+'.join(hit.hit_sources)}, rrf={hit.rrf_score:.4f}, rerank={hit.reranker_score:.4f})\n{hit.chunk.text}"
        for index, hit in enumerate(hits, start=1)
    )
    answer = (
        f"本地知识库混合检索完成，最终采用 BGE Reranker 重排后的 {len(hits)} 个片段。\n"
        f"RAG confidence={confidence:.3f} ({confidence_reason})\n\n{context}"
    )
    retrieval_trace = {
        "professional_qa_flow": True,
        "rag_called": True,
        "bm25_executed": True,
        "bm25_candidate_count": sum(len(run) for _, run in sparse),
        "vector_executed": True,
        "vector_candidate_count": sum(len(run) for _, run in dense),
        "rrf_executed": True,
        "rrf_fused_count": len(fused_hits),
        "multi_query_count_configured": query_count,
        "multi_query_count_used": len(query_list),
        "queries_used": query_list,
        "reranker_executed": reranker_executed,
        "reranker_model_path": config.reranker_model_path,
        "reranker_top_score": round(float(hits[0].reranker_score), 6) if hits else 0.0,
        "final_chunk_count": len(hits),
        "confidence": round(float(confidence), 6),
        "confidence_threshold": config.confidence_threshold,
        "confidence_result": "sufficient" if confidence >= config.confidence_threshold else "insufficient",
        "confidence_reason": confidence_reason,
        "web_search_triggered": False,
        "final_information_source": "local RAG",
        "degraded_without_reranker": not reranker_executed,
        "reranker_error": reranker_error,
    }
    retrieval_params = {
        "top_k_bm25": config.top_k_bm25,
        "top_k_vector": config.top_k_vector,
        "rrf_k": config.rrf_k,
        "top_k_fused": config.top_k_fused,
        "top_k_rerank": config.top_k_rerank,
        "multi_query_count": query_count,
        "reranker_confidence_threshold": config.confidence_threshold,
    }
    eval_trace = {
        "bm25": {"called": True, "top_k": config.top_k_bm25, "results": _rank_trace_results(sparse, chunks, "bm25_score")},
        "vector": {"called": True, "top_k": config.top_k_vector, "results": _rank_trace_results(dense, chunks, "vector_score")},
        "rrf": {"called": True, "rrf_k": config.rrf_k, "results": [hit.to_dict(include_text=False) for hit in fused_hits]},
        "reranker": {"called": reranker_executed, "model": config.reranker_model_path, "results": [hit.to_dict(include_text=False) for hit in hits]},
        "final_context_chunks": [hit.to_dict(include_text=True) for hit in hits],
    }
    return {
        "status": "ok",
        "tool_name": "local_knowledge_base",
        "query": query,
        "answer": answer,
        "local_answer": answer,
        "message": answer,
        "chunks": [hit.to_dict(include_text=True) for hit in hits],
        "references": references,
        "artifacts": [],
        "metrics": {
            "tool_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "hit_count": len(hits),
            "chunk_count": len(chunks),
            "bm25_candidate_count": retrieval_trace["bm25_candidate_count"],
            "vector_candidate_count": retrieval_trace["vector_candidate_count"],
            "rrf_fused_count": len(fused_hits),
            "reranker_top_score": retrieval_trace["reranker_top_score"],
            "confidence": round(float(confidence), 6),
        },
        "meta": {
            "knowledge_base_path": str(config.knowledge_base_path),
            "index_dir": str(config.index_dir),
            "files": files or [],
            "retrieval_params": retrieval_params,
            "retrieval_trace": retrieval_trace,
            "eval_trace": eval_trace,
            "confidence": round(float(confidence), 6),
            "confidence_reason": confidence_reason,
            "evidence_sufficient": confidence >= config.confidence_threshold and reranker_executed,
        },
        "retrieval_trace": retrieval_trace,
        "eval_trace": eval_trace,
    }
