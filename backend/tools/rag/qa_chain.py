from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import logging
from pathlib import Path
import re
import time

from ...prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from ..llm.client import get_local_qwen_client
from .config import RAGConfig, get_config
from .retrieval import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextBuilder:
    config: RAGConfig

    def build(self, chunks: list[RetrievedChunk]) -> str:
        selected = chunks[: self.config.max_context_chunks]
        sections: list[str] = []
        for index, item in enumerate(selected, start=1):
            location = []
            if item.chunk.page is not None:
                location.append(f"page={item.chunk.page}")
            if item.chunk.section:
                location.append(f"section={item.chunk.section}")
            location_text = ", ".join(location) if location else "location=unknown"
            sections.append(
                f"[{index}] source={item.chunk.file_name} ({location_text})\n"
                f"{item.chunk.text}"
            )
        return "\n\n".join(sections)


class LocalKnowledgeQAChain:
    def __init__(
        self,
        config: RAGConfig,
        retriever: HybridRetriever,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.client = get_local_qwen_client()
        self.context_builder = ContextBuilder(config)

    @classmethod
    def from_config(cls, config: RAGConfig) -> "LocalKnowledgeQAChain":
        retriever = HybridRetriever.from_index(config)
        return cls(config, retriever)

    def ask(self, query: str, min_score: float | None = None) -> dict:
        # Query -> Hybrid Retrieval -> Reranker -> Context Builder -> LLM
        started_at = time.perf_counter()
        retrieval_started_at = time.perf_counter()
        retrieved_chunks, retrieval_trace = self.retriever.hybrid_retrieve(query)
        (
            retrieved_chunks,
            ambiguity_trace,
        ) = self._refine_ambiguous_entity_query(query, retrieved_chunks)
        retrieval_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 2)
        if ambiguity_trace:
            retrieval_trace["ambiguity_resolution"] = ambiguity_trace

        if min_score is not None:
            before_count = len(retrieved_chunks)
            retrieved_chunks = [
                item for item in retrieved_chunks if float(item.score) >= float(min_score)
            ]
            retrieval_trace["score_threshold"] = float(min_score)
            retrieval_trace["filtered_out_count"] = max(
                0,
                before_count - len(retrieved_chunks),
            )

        if not retrieved_chunks:
            metrics = {
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0.0,
                "total_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }
            logger.info(
                "[rag] query=%r evidence=insufficient chunks=%s retrieval_ms=%.2f total_ms=%.2f",
                query,
                len(retrieved_chunks),
                metrics["retrieval_ms"],
                metrics["total_ms"],
            )
            return {
                "answer": "根据当前知识库内容无法确定。",
                "message": "本地知识检索已执行，但未检索到达到当前置信度阈值的结果。",
                "references": [],
                "retrieved_chunks": [],
                "retrieval_trace": retrieval_trace,
                "evidence_status": "insufficient",
                "metrics": metrics,
            }

        context = self.context_builder.build(retrieved_chunks)
        generation_started_at = time.perf_counter()
        answer = self._generate_answer(query=query, context=context)
        generation_ms = round((time.perf_counter() - generation_started_at) * 1000, 2)
        references = self._build_references(retrieved_chunks)
        metrics = {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        logger.info(
            "[rag] query=%r evidence=sufficient chunks=%s references=%s retrieval_ms=%.2f generation_ms=%.2f total_ms=%.2f",
            query,
            len(retrieved_chunks),
            len(references),
            metrics["retrieval_ms"],
            metrics["generation_ms"],
            metrics["total_ms"],
        )
        return {
            "answer": answer,
            "message": answer,
            "references": references,
            "retrieved_chunks": [chunk.to_dict() for chunk in retrieved_chunks],
            "retrieval_trace": retrieval_trace,
            "evidence_status": "sufficient",
            "metrics": metrics,
        }

    def _generate_answer(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": RAG_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": build_rag_user_prompt(query, context)}],
            },
        ]
        return self.client.invoke(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            trace_label="rag_answer_generation",
        )

    @staticmethod
    def _build_references(chunks: list[RetrievedChunk]) -> list[dict]:
        seen: set[tuple[str, int | None, str | None]] = set()
        references: list[dict] = []
        for item in chunks:
            key = (item.chunk.source_path, item.chunk.page, item.chunk.section)
            if key in seen:
                continue
            seen.add(key)
            references.append(
                {
                    "source_path": item.chunk.source_path,
                    "file_name": item.chunk.file_name,
                    "doc_type": item.chunk.doc_type,
                    "page": item.chunk.page,
                    "section": item.chunk.section,
                    "chunk_id": item.chunk.chunk_id,
                    "score": item.score,
                }
            )
        return references

    def _refine_ambiguous_entity_query(
        self, query: str, retrieved_chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], dict | None]:
        alias = self._extract_alias(query)
        if not alias:
            return retrieved_chunks, None

        alias_lower = alias.lower()
        alias_matches = [
            chunk
            for chunk in self.retriever.bm25_index.chunks
            if alias_lower in chunk.text.lower() or alias_lower in chunk.file_name.lower()
        ]
        if not alias_matches:
            return retrieved_chunks, None

        grouped_matches: dict[str, list] = {
            "single_cell": [],
            "compression": [],
            "other": [],
        }
        for chunk in alias_matches:
            grouped_matches[self._classify_alias_domain(chunk.text)].append(chunk)

        if not grouped_matches["single_cell"] or not grouped_matches["compression"]:
            return retrieved_chunks, None

        prioritized = self._pick_alias_representatives(alias, grouped_matches)
        if not prioritized:
            return retrieved_chunks, None

        selected_chunk_uids = {item.chunk.chunk_uid for item in prioritized}
        for item in retrieved_chunks:
            if item.chunk.chunk_uid in selected_chunk_uids:
                continue
            prioritized.append(item)
            selected_chunk_uids.add(item.chunk.chunk_uid)
            if len(prioritized) >= self.config.top_k_final:
                break

        return prioritized[: self.config.top_k_final], {
            "alias": alias,
            "matched_chunks": len(alias_matches),
            "domains": {
                "single_cell": len(grouped_matches["single_cell"]),
                "compression": len(grouped_matches["compression"]),
            },
            "strategy": (
                "Prioritized single-cell references for an ambiguous entity name "
                "and retained one compression reference for disambiguation."
            ),
        }

    @staticmethod
    def _extract_alias(query: str) -> str | None:
        normalized = query.strip()
        if len(normalized) > 64:
            return None

        alias_candidates = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", normalized)
        deduplicated: list[str] = []
        seen: set[str] = set()
        for candidate in alias_candidates:
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduplicated.append(candidate)
        if len(deduplicated) != 1:
            return None
        return deduplicated[0]

    def _pick_alias_representatives(
        self, alias: str, grouped_matches: dict[str, list]
    ) -> list[RetrievedChunk]:
        prioritized_chunks: list[RetrievedChunk] = []
        ranked_single_cell = self._rank_alias_chunks(
            alias, grouped_matches["single_cell"], self._single_cell_keywords()
        )
        ranked_compression = self._rank_alias_chunks(
            alias, grouped_matches["compression"], self._compression_keywords()
        )

        for chunk in ranked_single_cell[:3]:
            prioritized_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0,
                    retrieval_source="alias_disambiguation",
                    rerank_score=1.0,
                )
            )
        for chunk in ranked_compression[:1]:
            prioritized_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=0.6,
                    retrieval_source="alias_disambiguation",
                    rerank_score=0.6,
                )
            )

        return prioritized_chunks

    def _rank_alias_chunks(
        self, alias: str, chunks: list, keywords: tuple[str, ...]
    ) -> list:
        alias_lower = alias.lower()
        ranked = sorted(
            chunks,
            key=lambda chunk: (
                self._keyword_hits(chunk.text, keywords),
                chunk.text.lower().count(alias_lower),
                1 if alias_lower in chunk.file_name.lower() else 0,
                1 if chunk.page == 1 else 0,
                -len(chunk.text),
            ),
            reverse=True,
        )

        selected: list = []
        seen_locations: set[tuple[str, int | None, str | None]] = set()
        for chunk in ranked:
            key = (chunk.source_path, chunk.page, chunk.section)
            if key in seen_locations:
                continue
            seen_locations.add(key)
            selected.append(chunk)
        return selected

    @staticmethod
    def _keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)

    @staticmethod
    def _classify_alias_domain(text: str) -> str:
        text_lower = text.lower()
        single_cell_hits = sum(
            1
            for keyword in LocalKnowledgeQAChain._single_cell_keywords()
            if keyword in text_lower
        )
        compression_hits = sum(
            1
            for keyword in LocalKnowledgeQAChain._compression_keywords()
            if keyword in text_lower
        )
        if single_cell_hits > compression_hits:
            return "single_cell"
        if compression_hits > single_cell_hits:
            return "compression"
        return "other"

    @staticmethod
    def _single_cell_keywords() -> tuple[str, ...]:
        return (
            "single-cell",
            "single cell",
            "scrna",
            "cell embeddings",
            "cell-type",
            "cell type",
            "gene expression",
            "batch integration",
            "cellular heterogeneity",
        )

    @staticmethod
    def _compression_keywords() -> tuple[str, ...]:
        return (
            "compression",
            "compress",
            "dna",
            "nucleotide",
            "sequence",
            "sequencing",
            "genome",
            "fastq",
            "genozip",
        )


@lru_cache(maxsize=1)
def _get_cached_qa_chain(
    *,
    chunk_manifest_path: str,
    bm25_index_path: str,
    qdrant_path: str,
    embedding_model_path: str,
    llm_model_path: str,
) -> LocalKnowledgeQAChain:
    base_config = get_config()
    config = replace(
        base_config,
        chunk_manifest_path=Path(chunk_manifest_path),
        bm25_index_path=Path(bm25_index_path),
        qdrant_path=Path(qdrant_path),
        embedding_model_path=Path(embedding_model_path),
        llm_model_path=Path(llm_model_path),
    )
    return LocalKnowledgeQAChain.from_config(config)


def get_shared_local_knowledge_qa_chain(config: RAGConfig) -> LocalKnowledgeQAChain:
    return _get_cached_qa_chain(
        chunk_manifest_path=str(config.chunk_manifest_path),
        bm25_index_path=str(config.bm25_index_path),
        qdrant_path=str(config.qdrant_path),
        embedding_model_path=str(config.embedding_model_path),
        llm_model_path=str(config.llm_model_path),
    )
