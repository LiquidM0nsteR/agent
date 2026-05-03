from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".doc"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = DOCUMENT_SUFFIXES | TEXT_SUFFIXES
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


@dataclass(slots=True)
class IndexBuildConfig:
    project_root: Path
    knowledge_base_path: Path
    index_dir: Path
    chunks_path: Path
    bm25_path: Path
    embeddings_path: Path
    embedding_model_path: str
    chunk_size: int = 900
    sentence_overlap: int = 1
    min_chunk_chars: int = 260


@dataclass(slots=True)
class IndexChunk:
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


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_index_build_config(knowledge_base_path: str = "", index_dir: str = "", project_root: str = "", embedding_model_path: str = "") -> IndexBuildConfig:
    root = Path(project_root).expanduser().resolve() if project_root else infer_project_root()
    data_dir = root / "data"
    kb_path = Path(knowledge_base_path).expanduser().resolve() if knowledge_base_path else data_dir / "local_knowledge"
    idx_dir = Path(index_dir).expanduser().resolve() if index_dir else data_dir / "local_knowledge_index"
    local_bge = root / "models" / "bge-m3"
    emb_path = embedding_model_path or os.getenv("EMBEDDING_MODEL_PATH") or (str(local_bge) if local_bge.exists() else DEFAULT_EMBEDDING_MODEL_NAME)
    return IndexBuildConfig(
        project_root=root,
        knowledge_base_path=kb_path,
        index_dir=idx_dir,
        chunks_path=idx_dir / "chunks.jsonl",
        bm25_path=idx_dir / "bm25.pkl",
        embeddings_path=idx_dir / "embeddings.npy",
        embedding_model_path=emb_path,
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
        sentence_overlap=max(0, int(os.getenv("RAG_SENTENCE_OVERLAP", "1"))),
        min_chunk_chars=max(80, int(os.getenv("RAG_MIN_CHUNK_CHARS", "260"))),
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


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        return {_clean_json_value(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]
    return value


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="ignore") as handle:
        for row in rows:
            handle.write(json.dumps(_clean_json_value(row), ensure_ascii=True, default=str) + "\n")


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


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _recursive_sentence_units(text: str, max_chars: int) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = _split_sentences(text)
    if len(sentences) > 1:
        units: list[str] = []
        for sentence in sentences:
            units.extend(_recursive_sentence_units(sentence, max_chars))
        return units
    for separator in ("\n\n", "\n", "。", "；", "，", ",", " "):
        if separator not in text:
            continue
        pieces = [piece.strip() for piece in text.split(separator) if piece.strip()]
        if len(pieces) <= 1:
            continue
        units = []
        for piece in pieces:
            suffix = separator if separator.strip() and separator not in {" ", "\n", "\n\n"} else ""
            units.extend(_recursive_sentence_units(piece + suffix, max_chars))
        return units
    return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]


def _chunk_text_by_recursive_sentences(text: str, max_chars: int, min_chars: int, sentence_overlap: int) -> list[str]:
    units = _recursive_sentence_units(text, max_chars)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_len = len(unit)
        if current and current_len + unit_len + 1 > max_chars and current_len >= min_chars:
            chunks.append(" ".join(current).strip())
            current = current[-sentence_overlap:] if sentence_overlap else []
            current_len = sum(len(item) + 1 for item in current)
        current.append(unit)
        current_len += unit_len + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _load_embedder(model_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_path)


def _encode(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def build_rag_index(knowledge_base_path: str = "", index_dir: str = "", files: list[str] | None = None, clean: bool = True, **_: Any) -> dict[str, Any]:
    del clean
    started_at = time.perf_counter()
    config = get_index_build_config(knowledge_base_path, index_dir)
    config.index_dir.mkdir(parents=True, exist_ok=True)
    paths = _iter_files(config.knowledge_base_path, files)
    if not paths:
        raise FileNotFoundError(f"没有找到可索引文件：{config.knowledge_base_path}")

    chunks: list[IndexChunk] = []
    skipped_sources: list[dict[str, str]] = []
    for path in paths:
        rel = path.resolve().relative_to(config.knowledge_base_path.resolve()).as_posix()
        doc_id = _stable_id(rel, 16)
        try:
            extracted_blocks = _extract_text(path)
        except Exception as exc:
            skipped_sources.append({"source_path": rel, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        for text, page, metadata in extracted_blocks:
            parts = _chunk_text_by_recursive_sentences(text, config.chunk_size, config.min_chunk_chars, config.sentence_overlap)
            for part in parts:
                chunks.append(IndexChunk(
                    chunk_id=_stable_id(f"{rel}:{page}:{len(chunks)}:{part[:120]}", 24),
                    doc_id=doc_id,
                    source_path=rel,
                    title=path.stem,
                    text=part,
                    chunk_index=len(chunks),
                    page=page,
                    metadata={**metadata, "chunking_strategy": "recursive_sentence", "sentence_overlap": config.sentence_overlap},
                ))

    if not chunks:
        raise ValueError(f"索引文件没有抽取到任何文本块。skipped_sources={skipped_sources}")

    _write_jsonl(config.chunks_path, [chunk.to_json() for chunk in chunks])
    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    with config.bm25_path.open("wb") as handle:
        pickle.dump({"chunk_ids": [chunk.chunk_id for chunk in chunks], "tokenized_corpus": tokenized, "bm25": BM25Okapi(tokenized)}, handle)
    embeddings = _encode(_load_embedder(config.embedding_model_path), [chunk.text for chunk in chunks])
    np.save(config.embeddings_path, embeddings)
    meta = {
        "embedding_model_path": config.embedding_model_path,
        "knowledge_base_path": str(config.knowledge_base_path),
        "index_dir": str(config.index_dir),
        "chunking": {
            "strategy": "recursive_sentence",
            "chunk_size": config.chunk_size,
            "min_chunk_chars": config.min_chunk_chars,
            "sentence_overlap": config.sentence_overlap,
        },
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "skipped_sources": skipped_sources,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (config.index_dir / "meta.json").write_text(json.dumps(_clean_json_value(meta), ensure_ascii=False, indent=2), encoding="utf-8", errors="ignore")
    return {
        "status": "ok",
        "chunk_count": len(chunks),
        "vector_count": int(embeddings.shape[0]),
        "index_dir": str(config.index_dir),
        "chunking_strategy": "recursive_sentence",
        "skipped_sources": skipped_sources,
        "metrics": {"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
    }
