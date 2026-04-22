from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(slots=True)
class RAGConfig:
    project_root: Path
    data_dir: Path
    local_knowledge_dir: Path
    index_dir: Path
    qdrant_path: Path
    qdrant_access_lock_path: Path
    semantic_qdrant_path: Path
    chunk_manifest_path: Path
    bm25_index_path: Path
    embedding_model_name: str
    embedding_model_path: Path
    llm_model_path: Path
    qdrant_collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int
    max_context_chunks: int
    max_new_tokens: int
    temperature: float
    repetition_penalty: float
    short_term_max_messages: int
    short_term_max_approx_tokens: int
    short_term_summary_threshold: int
    long_term_top_k: int
    enable_profile_memory: bool
    enable_semantic_memory: bool
    semantic_memory_collection: str
    profile_storage_path: Path
    qdrant_lock_timeout_seconds: float


def get_config() -> RAGConfig:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    index_dir = data_dir / "local_knowledge_index"
    models_dir = project_root / "models"
    embedding_model_path = Path(
        os.getenv("EMBEDDING_MODEL_PATH", str(models_dir / "bge-m3"))
    )
    default_llm_model_path = Path(
        "./agent/models/Qwen2.5-VL-7B-Instruct"
    )
    llm_model_path = Path(
        os.getenv("LLM_MODEL_PATH", str(default_llm_model_path))
    ).expanduser()

    return RAGConfig(
        project_root=project_root,
        data_dir=data_dir,
        local_knowledge_dir=data_dir / "local_knowledge",
        index_dir=index_dir,
        qdrant_path=index_dir / "qdrant",
        qdrant_access_lock_path=index_dir / "qdrant.access.lock",
        semantic_qdrant_path=Path(
            os.getenv(
                "SEMANTIC_QDRANT_PATH",
                str(data_dir / "memory" / "qdrant"),
            )
        ),
        chunk_manifest_path=index_dir / "chunks.jsonl",
        bm25_index_path=index_dir / "bm25.pkl",
        embedding_model_name="BAAI/bge-m3",
        embedding_model_path=embedding_model_path,
        llm_model_path=llm_model_path,
        qdrant_collection_name="local_knowledge_chunks",
        chunk_size=900,
        chunk_overlap=180,
        top_k_dense=8,
        top_k_sparse=8,
        top_k_final=5,
        max_context_chunks=4,
        max_new_tokens=512,
        temperature=0.2,
        repetition_penalty=1.05,
        short_term_max_messages=int(os.getenv("SHORT_TERM_MAX_MESSAGES", "12")),
        short_term_max_approx_tokens=int(
            os.getenv("SHORT_TERM_MAX_APPROX_TOKENS", "2400")
        ),
        short_term_summary_threshold=int(
            os.getenv("SHORT_TERM_SUMMARY_THRESHOLD", "8")
        ),
        long_term_top_k=int(os.getenv("LONG_TERM_TOP_K", "3")),
        enable_profile_memory=os.getenv("ENABLE_PROFILE_MEMORY", "true").lower()
        in {"1", "true", "yes"},
        enable_semantic_memory=os.getenv("ENABLE_SEMANTIC_MEMORY", "true").lower()
        in {"1", "true", "yes"},
        semantic_memory_collection=os.getenv(
            "SEMANTIC_MEMORY_COLLECTION", "agent_semantic_memory"
        ),
        profile_storage_path=Path(
            os.getenv(
                "PROFILE_STORAGE_PATH",
                str(data_dir / "memory" / "profiles.sqlite3"),
            )
        ),
        qdrant_lock_timeout_seconds=float(
            os.getenv("QDRANT_LOCK_TIMEOUT_SECONDS", "180")
        ),
    )
