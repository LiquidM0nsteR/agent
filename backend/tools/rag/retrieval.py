from __future__ import annotations

import math
import pickle
import re
import gc
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import portalocker
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from .config import RAGConfig, get_config
from .knowledge_base import ChunkRecord


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    retrieval_source: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.chunk.text,
            "score": self.score,
            "retrieval_source": self.retrieval_source,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
            "metadata": {
                "source_path": self.chunk.source_path,
                "file_name": self.chunk.file_name,
                "doc_type": self.chunk.doc_type,
                "chunk_id": self.chunk.chunk_id,
                "page": self.chunk.page,
                "section": self.chunk.section,
            },
        }


class BgeM3Embedder:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not config.embedding_model_path.exists():
            raise FileNotFoundError(
                "BGE-M3 embedding model was not found locally. "
                f"Expected path: {config.embedding_model_path}"
            )
        model_source = str(config.embedding_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_source, local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        """Release model resources to avoid long-lived GPU residency."""
        try:
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
        finally:
            # Drop references so CUDA memory can be reclaimed.
            if hasattr(self, "model"):
                self.model = None  # type: ignore[assignment]
            if hasattr(self, "tokenizer"):
                self.tokenizer = None  # type: ignore[assignment]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def encode(self, texts: Iterable[str], batch_size: int = 8) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        text_list = list(texts)
        if not text_list:
            return all_vectors

        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_vectors.extend(embeddings.cpu().tolist())
        return all_vectors

    def encode_query(self, query: str) -> list[float]:
        return self.encode([query], batch_size=1)[0]

    @staticmethod
    def _mean_pooling(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class QdrantVectorStore:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.config.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.qdrant_access_lock_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked_client(self) -> Iterable[QdrantClient]:
        with portalocker.Lock(
            self.config.qdrant_access_lock_path,
            mode="a+",
            timeout=self.config.qdrant_lock_timeout_seconds,
        ):
            deadline = time.monotonic() + self.config.qdrant_lock_timeout_seconds
            while True:
                try:
                    client = QdrantClient(path=str(self.config.qdrant_path))
                    break
                except RuntimeError as exc:
                    if (
                        "already accessed by another instance of Qdrant client"
                        not in str(exc)
                    ):
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for the local Qdrant storage lock to be released."
                        ) from exc
                    time.sleep(1.0)

            try:
                yield client
            finally:
                client.close()

    def replace_collection(
        self, chunks: list[ChunkRecord], embeddings: list[list[float]], batch_size: int = 64
    ) -> None:
        if not embeddings:
            return
        self.config.qdrant_path.mkdir(parents=True, exist_ok=True)
        with self._locked_client() as client:
            client.recreate_collection(
                collection_name=self.config.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=len(embeddings[0]), distance=Distance.COSINE
                ),
            )
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start : start + batch_size]
                batch_vectors = embeddings[start : start + batch_size]
                points = [
                    PointStruct(
                        id=chunk.chunk_uid, vector=vector, payload=chunk.to_payload()
                    )
                    for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
                ]
                client.upsert(
                    collection_name=self.config.qdrant_collection_name,
                    points=points,
                )

    def upsert_chunks(
        self, chunks: list[ChunkRecord], embeddings: list[list[float]], batch_size: int = 64
    ) -> None:
        with self._locked_client() as client:
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start : start + batch_size]
                batch_vectors = embeddings[start : start + batch_size]
                points = [
                    PointStruct(
                        id=chunk.chunk_uid, vector=vector, payload=chunk.to_payload()
                    )
                    for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
                ]
                client.upsert(
                    collection_name=self.config.qdrant_collection_name,
                    points=points,
                )

    def dense_retrieval(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        with self._locked_client() as client:
            response = client.query_points(
                collection_name=self.config.qdrant_collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        results = response.points
        chunks: list[RetrievedChunk] = []
        for result in results:
            chunk = ChunkRecord.from_payload(result.payload)
            chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(result.score),
                    retrieval_source="dense",
                    dense_score=float(result.score),
                )
            )
        return chunks


class BM25Index:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def load(cls, index_path: Path) -> "BM25Index":
        with index_path.open("rb") as handle:
            payload = pickle.load(handle)
        chunks = [ChunkRecord(**chunk) for chunk in payload["chunks"]]
        instance = cls(chunks)
        instance.tokenized_corpus = payload["tokenized_corpus"]
        instance.bm25 = BM25Okapi(instance.tokenized_corpus)
        return instance

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "tokenized_corpus": self.tokenized_corpus,
        }
        with index_path.open("wb") as handle:
            pickle.dump(payload, handle)

    def sparse_retrieval(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[index],
                    score=score,
                    retrieval_source="sparse",
                    sparse_score=score,
                )
            )
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


class Reranker:
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        query_terms = set(BM25Index._tokenize(query))
        reranked: list[RetrievedChunk] = []

        for candidate in candidates:
            text_terms = set(BM25Index._tokenize(candidate.chunk.text))
            overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
            base_score = self._sigmoid(candidate.score)
            rerank_score = 0.65 * base_score + 0.35 * overlap
            reranked.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    score=rerank_score,
                    retrieval_source=candidate.retrieval_source,
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    rerank_score=rerank_score,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))


class HybridRetriever:
    def __init__(
        self,
        config: RAGConfig,
        embedder: BgeM3Embedder,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        reranker: Reranker,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker

    @classmethod
    def from_index(cls, config: RAGConfig) -> "HybridRetriever":
        embedder = get_shared_bge_m3_embedder(config)
        vector_store = QdrantVectorStore(config)
        bm25_index = BM25Index.load(config.bm25_index_path)
        reranker = Reranker()
        return cls(config, embedder, vector_store, bm25_index, reranker)

    def dense_retrieval(self, query: str) -> list[RetrievedChunk]:
        query_vector = self.embedder.encode_query(query)
        return self.vector_store.dense_retrieval(query_vector, self.config.top_k_dense)

    def sparse_retrieval(self, query: str) -> list[RetrievedChunk]:
        return self.bm25_index.sparse_retrieval(query, self.config.top_k_sparse)

    def fuse_results(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        score_accumulator: defaultdict[str, float] = defaultdict(float)

        for rank, item in enumerate(dense_results, start=1):
            score_accumulator[item.chunk.chunk_uid] += 1.0 / (60 + rank)
            merged[item.chunk.chunk_uid] = item

        for rank, item in enumerate(sparse_results, start=1):
            score_accumulator[item.chunk.chunk_uid] += 1.0 / (60 + rank)
            if item.chunk.chunk_uid in merged:
                existing = merged[item.chunk.chunk_uid]
                existing.sparse_score = item.sparse_score
                existing.retrieval_source = "hybrid"
            else:
                merged[item.chunk.chunk_uid] = item

        fused = []
        for chunk_uid, item in merged.items():
            fused.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    score=score_accumulator[chunk_uid],
                    retrieval_source="hybrid",
                    dense_score=item.dense_score,
                    sparse_score=item.sparse_score,
                )
            )

        fused.sort(key=lambda result: result.score, reverse=True)
        return fused

    def hybrid_retrieve(self, query: str) -> tuple[list[RetrievedChunk], dict]:
        dense_results = self.dense_retrieval(query)
        sparse_results = self.sparse_retrieval(query)
        fused_results = self.fuse_results(dense_results, sparse_results)
        reranked_results = self.reranker.rerank(
            query, fused_results, self.config.top_k_final
        )
        trace = {
            "dense_hits": len(dense_results),
            "sparse_hits": len(sparse_results),
            "fused_hits": len(fused_results),
            "final_hits": len(reranked_results),
        }
        return reranked_results, trace

@lru_cache(maxsize=1)
def _get_cached_embedder(model_path: str) -> BgeM3Embedder:
    return BgeM3Embedder(replace(get_config(), embedding_model_path=Path(model_path)))


def get_shared_bge_m3_embedder(config: RAGConfig) -> BgeM3Embedder:
    return _get_cached_embedder(str(config.embedding_model_path))


def preload_bge_m3_embedder(config: RAGConfig) -> dict[str, str]:
    embedder = get_shared_bge_m3_embedder(config)
    return {
        "model_path": str(embedder.config.embedding_model_path),
        "device": str(embedder.device),
    }
