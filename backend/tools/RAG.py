"""Retrieval-only local RAG module.

Responsibilities:
    1. Explicitly build local indexes by calling build_rag_index(...).
    2. Retrieve evidence by calling run_rag(...).
    3. Return formatted retrieval results only.

Supported local knowledge file types:
    PDF, DOCX/DOC, images, text, markdown.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from typing import Any, Callable, Iterable, Iterator, Sequence
from uuid import NAMESPACE_URL, uuid5

logger = logging.getLogger(__name__)
_OCR_UNAVAILABLE_LOGGED = False

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".doc"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = DOCUMENT_SUFFIXES | IMAGE_SUFFIXES | TEXT_SUFFIXES

DEFAULT_COLLECTION_NAME = "local_knowledge_chunks"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RAGConfig:
    project_root: Path
    knowledge_base_path: Path
    index_dir: Path
    chunks_path: Path
    bm25_path: Path
    qdrant_path: Path
    qdrant_lock_path: Path
    artifact_dir: Path
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_NAME
    rerank_model_path: str = DEFAULT_RERANK_MODEL_NAME
    chunk_size: int = 900
    chunk_overlap: int = 180
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_top_k: int = 30
    final_top_k: int = 6
    rrf_k: int = 60
    device: str | None = None
    enable_rerank: bool = True
    enable_pdf_figures: bool = True
    enable_pdf_tables: bool = True
    enable_ocr: bool = True
    enable_image_summary: bool = True
    semantic_chunking: bool = True
    semantic_chunk_min_chars: int = 260
    semantic_chunk_similarity_threshold: float = 0.58
    semantic_chunk_sentence_overlap: int = 1


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    source_path: str
    title: str
    suffix: str
    size_bytes: int
    mtime: float


@dataclass(slots=True)
class TextBlock:
    text: str
    page: int | None = None
    block_index: int = 0
    block_type: str = "text"
    bbox: tuple[float, float, float, float] | None = None
    artifact_path: str = ""
    structured_content: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    chunk_index: int
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RAGChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            source_path=str(data["source_path"]),
            title=str(data.get("title") or Path(str(data["source_path"])).stem),
            text=str(data["text"]),
            chunk_index=int(data.get("chunk_index") or 0),
            page=data.get("page"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class SearchHit:
    chunk: RAGChunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        metadata = dict(self.chunk.metadata or {})
        payload = {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source_path": self.chunk.source_path,
            "title": self.chunk.title,
            "page": self.chunk.page,
            "chunk_index": self.chunk.chunk_index,
            "block_type": metadata.get("block_type", "text"),
            "score": round(float(self.score), 6),
            "rrf_score": round(float(self.rrf_score), 6),
            "dense_score": round(float(self.dense_score), 6),
            "sparse_score": round(float(self.sparse_score), 6),
            "rerank_score": (
                round(float(self.rerank_score), 6)
                if self.rerank_score is not None
                else None
            ),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "bbox": metadata.get("bbox"),
            "artifact_path": metadata.get("artifact_path", ""),
            "metadata": metadata,
        }
        if include_text:
            payload["text"] = self.chunk.text
        return payload


@dataclass(slots=True)
class PDFTableObject:
    table_id: str
    source_path: str
    title: str
    pages: list[int]
    page_heights: list[float]
    bboxes: list[tuple[float, float, float, float]]
    rows: list[list[str]]
    caption: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    continued_markers: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(row) for row in self.rows), default=0)


@dataclass(slots=True)
class BM25Bundle:
    chunk_ids: list[str]
    tokenized_corpus: list[list[str]]
    bm25: Any

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Bundle":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            chunk_ids=list(payload["chunk_ids"]),
            tokenized_corpus=list(payload.get("tokenized_corpus") or []),
            bm25=payload["bm25"],
        )


# ---------------------------------------------------------------------------
# Configuration and generic helpers
# ---------------------------------------------------------------------------


def infer_project_root() -> Path:
    """Infer project root from agent/backend/tools/RAG.py -> agent."""
    return Path(__file__).resolve().parents[2]


def get_rag_config(
    knowledge_base_path: str = "",
    *,
    index_dir: str = "",
    project_root: str = "",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_path: str = "",
    rerank_model_path: str = "",
    enable_rerank: bool = True,
) -> RAGConfig:
    root = Path(project_root).expanduser().resolve() if project_root else infer_project_root()
    data_dir = root / "data"

    kb_path = (
        Path(knowledge_base_path).expanduser().resolve()
        if knowledge_base_path
        else data_dir / "local_knowledge"
    )
    idx_dir = (
        Path(index_dir).expanduser().resolve()
        if index_dir
        else data_dir / "local_knowledge_index"
    )

    local_bge = root / "models" / "bge-m3"
    local_reranker = root / "models" / "bge-reranker-v2-m3"

    emb_path = (
        embedding_model_path
        or os.getenv("EMBEDDING_MODEL_PATH")
        or (str(local_bge) if local_bge.exists() else DEFAULT_EMBEDDING_MODEL_NAME)
    )
    rr_path = (
        rerank_model_path
        or os.getenv("RERANK_MODEL_PATH")
        or (str(local_reranker) if local_reranker.exists() else DEFAULT_RERANK_MODEL_NAME)
    )

    return RAGConfig(
        project_root=root,
        knowledge_base_path=kb_path,
        index_dir=idx_dir,
        chunks_path=idx_dir / "chunks.jsonl",
        bm25_path=idx_dir / "bm25.pkl",
        qdrant_path=idx_dir / "qdrant",
        qdrant_lock_path=idx_dir / "qdrant.access.lock",
        artifact_dir=idx_dir / "artifacts",
        collection_name=collection_name,
        embedding_model_path=emb_path,
        rerank_model_path=rr_path,
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "180")),
        dense_top_k=int(os.getenv("RAG_DENSE_TOP_K", "20")),
        sparse_top_k=int(os.getenv("RAG_SPARSE_TOP_K", "20")),
        rrf_top_k=int(os.getenv("RAG_RRF_TOP_K", "30")),
        final_top_k=int(os.getenv("RAG_FINAL_TOP_K", "6")),
        rrf_k=int(os.getenv("RAG_RRF_K", "60")),
        device=os.getenv("RAG_DEVICE") or None,
        enable_rerank=enable_rerank,
        enable_pdf_figures=os.getenv("RAG_ENABLE_PDF_FIGURES", "true").lower()
        in {"1", "true", "yes"},
        enable_pdf_tables=os.getenv("RAG_ENABLE_PDF_TABLES", "true").lower()
        in {"1", "true", "yes"},
        enable_ocr=os.getenv("RAG_ENABLE_OCR", "true").lower() in {"1", "true", "yes"},
        enable_image_summary=os.getenv("RAG_ENABLE_IMAGE_SUMMARY", "true").lower()
        in {"1", "true", "yes"},
        semantic_chunking=os.getenv("RAG_SEMANTIC_CHUNKING", "true").lower()
        in {"1", "true", "yes"},
        semantic_chunk_min_chars=int(os.getenv("RAG_SEMANTIC_CHUNK_MIN_CHARS", "260")),
        semantic_chunk_similarity_threshold=float(
            os.getenv("RAG_SEMANTIC_CHUNK_SIMILARITY_THRESHOLD", "0.58")
        ),
        semantic_chunk_sentence_overlap=int(os.getenv("RAG_SEMANTIC_CHUNK_SENTENCE_OVERLAP", "1")),
    )


def _sha1(text: str, n: int = 20) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _strip_invalid_unicode(text: Any) -> str:
    return "".join(
        " "
        if unicodedata.category(ch).startswith("C") and ch not in {"\n", "\t"}
        else ch
        for ch in str(text or "")
    )


def _normalize_text(text: Any) -> str:
    value = _strip_invalid_unicode(text).replace("\x00", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _embedding_safe_text(text: Any) -> str:
    return _normalize_text(text) or " "


def _sanitize_jsonable(data: Any) -> Any:
    if isinstance(data, str):
        return _normalize_text(data)
    if isinstance(data, dict):
        return {
            _normalize_text(key): _sanitize_jsonable(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_sanitize_jsonable(item) for item in data]
    if isinstance(data, tuple):
        return [_sanitize_jsonable(item) for item in data]
    return data


def _json_dumps(data: Any) -> str:
    return json.dumps(_sanitize_jsonable(data), ensure_ascii=False, default=str)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _safe_relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()



def _make_doc_record(path: Path, kb_root: Path) -> DocumentRecord:
    stat = path.stat()
    rel = _safe_relative_path(path, kb_root)
    return DocumentRecord(
        doc_id=_sha1(rel, 16),
        source_path=rel,
        title=path.stem,
        suffix=path.suffix.lower(),
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
    )


def _iter_files(base_path: Path, files: list[str] | None = None) -> list[Path]:
    base = base_path.resolve()
    if files:
        resolved: list[Path] = []
        for item in files:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = base_path / path
            path = path.resolve()
            try:
                path.relative_to(base)
            except ValueError:
                logger.warning("Skip RAG file outside knowledge_base_path: file=%s root=%s", path, base)
                continue
            if path.exists() and path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                resolved.append(path)
        return sorted(set(resolved), key=lambda item: item.as_posix())

    if not base_path.exists():
        return []

    results: list[Path] = []
    for path in base_path.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            results.append(path.resolve())
    return sorted(results, key=lambda item: item.as_posix())


@contextmanager
def _qdrant_lock(config: RAGConfig) -> Iterator[None]:
    try:
        from filelock import FileLock
    except Exception:
        yield
        return

    config.index_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(config.qdrant_lock_path), timeout=180)
    with lock:
        yield


def _artifact_relpath(config: RAGConfig, path: Path) -> str:
    try:
        return path.resolve().relative_to(config.project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Document extraction: PDF / DOCX / DOC / images / text
# ---------------------------------------------------------------------------


def _extract_blocks(
    path: Path,
    doc: DocumentRecord,
    config: RAGConfig,
    *,
    image_summary_fn: Callable[[Path, dict[str, Any]], str] | None = None,
    table_summary_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[TextBlock]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_blocks(
            path=path,
            doc=doc,
            config=config,
            image_summary_fn=image_summary_fn,
            table_summary_fn=table_summary_fn,
        )
    if suffix == ".docx":
        return _extract_docx_blocks(path)
    if suffix == ".doc":
        return _extract_doc_blocks(path)
    if suffix in TEXT_SUFFIXES:
        return _extract_text_blocks(path)
    if suffix in IMAGE_SUFFIXES:
        return _extract_image_blocks(
            path=path,
            config=config,
            source_doc=doc,
            image_summary_fn=image_summary_fn,
        )

    raise ValueError(f"不支持的文件类型：{suffix}")


def _extract_pdf_blocks(
    path: Path,
    doc: DocumentRecord,
    config: RAGConfig,
    *,
    image_summary_fn: Callable[[Path, dict[str, Any]], str] | None = None,
    table_summary_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    errors: list[str] = []

    try:
        blocks.extend(_extract_pdf_text_blocks(path))
    except Exception as exc:
        errors.append(f"text: {exc}")

    if config.enable_pdf_figures:
        try:
            blocks.extend(
                _extract_pdf_figure_blocks(
                    path=path,
                    doc=doc,
                    config=config,
                    image_summary_fn=image_summary_fn,
                )
            )
        except Exception as exc:
            logger.warning("PDF figure extraction skipped: %s", exc)
            errors.append(f"figures: {exc}")

    if config.enable_pdf_tables:
        try:
            table_objects = _extract_pdf_table_objects(path=path, doc=doc, config=config)
            merged_tables = _merge_cross_page_tables(table_objects)
            blocks.extend(
                _table_objects_to_text_blocks(
                    tables=merged_tables,
                    table_summary_fn=table_summary_fn,
                )
            )
        except Exception as exc:
            logger.warning("PDF table extraction skipped: %s", exc)
            errors.append(f"tables: {exc}")

    if errors and not blocks:
        raise RuntimeError("PDF 解析失败：" + " | ".join(errors))

    blocks.sort(
        key=lambda block: (
            block.page if block.page is not None else 10**9,
            block.block_index,
            block.block_type,
        )
    )
    return blocks


def _extract_pdf_text_blocks(path: Path) -> list[TextBlock]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("缺少依赖 pypdf，请先安装：pip install pypdf") from exc

    reader = PdfReader(str(path))
    blocks: list[TextBlock] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if text:
            blocks.append(
                TextBlock(
                    text=text,
                    page=page_idx,
                    block_index=page_idx * 10000,
                    block_type="pdf_text",
                )
            )

    return blocks


def _open_pdf_with_fitz(path: Path) -> Any:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("缺少依赖 PyMuPDF，请先安装：pip install pymupdf") from exc
    return fitz.open(str(path))


def _crop_pdf_region(
    pdf_path: Path,
    page_number: int,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    *,
    zoom: float = 2.0,
) -> None:
    import fitz

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(pdf_path)) as pdf:
        page = pdf[page_number - 1]
        rect = fitz.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width <= 1 or rect.height <= 1:
            raise ValueError(f"Figure bbox is outside page bounds: bbox={bbox}, page_rect={tuple(page.rect)}")
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        pix.save(str(output_path))


def _extract_page_text_blocks_with_bbox(page: Any) -> list[dict[str, Any]]:
    raw_blocks = page.get_text("blocks")
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_blocks):
        if len(item) < 5:
            continue
        x0, y0, x1, y1, text = item[:5]
        text = _normalize_text(text)
        if text:
            results.append(
                {
                    "index": idx,
                    "bbox": (float(x0), float(y0), float(x1), float(y1)),
                    "text": text,
                }
            )
    return results


def _rect_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return (dx * dx + dy * dy) ** 0.5


def _extract_nearby_text_from_page(
    page: Any,
    bbox: tuple[float, float, float, float],
    *,
    max_chars: int = 1200,
    top_k: int = 6,
) -> str:
    blocks = _extract_page_text_blocks_with_bbox(page)
    ranked = sorted(blocks, key=lambda block: _rect_distance(bbox, block["bbox"]))

    texts: list[str] = []
    total = 0
    for block in ranked[:top_k]:
        text = block["text"]
        texts.append(text)
        total += len(text)
        if total >= max_chars:
            break

    return _normalize_text("\n".join(texts))[:max_chars]


def _find_caption_near_bbox(
    page: Any,
    bbox: tuple[float, float, float, float],
    *,
    kind: str,
    max_chars: int = 500,
) -> str:
    patterns = {
        "figure": r"^\s*((fig\.?|figure|图)\s*[\dIVXivx0-9\-\.]+[:：.\s].*)",
        "table": r"^\s*((table|表)\s*[\dIVXivx0-9\-\.]+[:：.\s].*)",
    }
    pattern = re.compile(patterns[kind], re.IGNORECASE)

    candidates: list[tuple[float, str]] = []
    for block in _extract_page_text_blocks_with_bbox(page):
        text = block["text"]
        if pattern.search(text):
            candidates.append((_rect_distance(bbox, block["bbox"]), text))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0])
    return _normalize_text(candidates[0][1])[:max_chars]


def _ocr_image_file(image_path: Path, *, enable_ocr: bool = True) -> str:
    global _OCR_UNAVAILABLE_LOGGED

    if not enable_ocr:
        return ""

    configured_lang = os.getenv("RAG_OCR_LANG", "eng+chi_sim")
    languages = [configured_lang]
    if configured_lang != "eng":
        languages.append("eng")

    try:
        from PIL import Image
        import pytesseract

        with Image.open(image_path) as image:
            last_error: Exception | None = None
            for lang in languages:
                try:
                    text = pytesseract.image_to_string(image, lang=lang)
                    return _normalize_text(text)
                except Exception as exc:
                    last_error = exc
                    if lang == configured_lang and "chi_sim" in configured_lang:
                        continue
                    break
    except Exception as exc:
        last_error = exc

    if not _OCR_UNAVAILABLE_LOGGED:
        logger.info("OCR unavailable or failed; continuing without OCR text. First image=%s (%s)", image_path, last_error)
        _OCR_UNAVAILABLE_LOGGED = True
    else:
        logger.debug("OCR unavailable for image: %s (%s)", image_path, last_error)
    return ""


def _build_figure_search_text(
    *,
    source_path: str,
    page: int | None,
    caption: str,
    nearby_text: str,
    ocr_text: str,
    semantic_summary: str,
) -> str:
    parts = [
        "[Figure]",
        f"Source: {source_path}",
    ]
    if page is not None:
        parts.append(f"Page: {page}")
    if caption:
        parts.append(f"Caption: {caption}")
    if semantic_summary:
        parts.append(f"Semantic summary: {semantic_summary}")
    if ocr_text:
        parts.append(f"OCR text: {ocr_text}")
    if nearby_text:
        parts.append(f"Nearby text: {nearby_text}")
    return _normalize_text("\n".join(parts))


def _default_image_summary(image_path: Path, metadata: dict[str, Any]) -> str:
    prompt = f"""
你正在为本地 RAG 索引描述一张从文档中裁剪出来的图片。

请基于图片内容生成可检索的语义摘要，要求：
1. 如果是图表、架构图、流程图或示意图，说明图中元素、关系、趋势和结论。
2. 如果是论文插图，结合页码、caption、邻近文本理解它在文档中的作用。
3. 不要编造图片中看不到的信息。
4. 输出 3-6 条简洁中文要点，保留关键英文术语、模型名、指标名。

文档路径：{metadata.get("source_path", "")}
页码：{metadata.get("page", "")}
Caption：{metadata.get("caption", "")}
邻近文本：{metadata.get("nearby_text", "")}
""".strip()

    try:
        from .LLM import chat

        return _normalize_text(
            chat(
                prompt=prompt,
                images=[image_path],
                max_new_tokens=int(os.getenv("RAG_IMAGE_SUMMARY_MAX_NEW_TOKENS", "220")),
                do_sample=False,
            )
        )
    except Exception as exc:
        logger.warning("Image semantic summary failed for %s: %s", image_path, exc)
        return ""


def _extract_pdf_figure_blocks(
    path: Path,
    doc: DocumentRecord,
    config: RAGConfig,
    *,
    image_summary_fn: Callable[[Path, dict[str, Any]], str] | None = None,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    artifact_dir = config.artifact_dir / "figures" / _sha1(doc.source_path, 12)
    pdf = _open_pdf_with_fitz(path)

    try:
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            page_number = page_idx + 1
            seen_rects: set[tuple[int, int, int, int]] = set()
            figure_count = 0

            for image_info in page.get_images(full=True):
                xref = image_info[0]
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []

                for rect in rects:
                    bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if width < 40 or height < 40:
                        continue

                    rounded = tuple(int(v) for v in bbox)
                    if rounded in seen_rects:
                        continue
                    seen_rects.add(rounded)

                    figure_count += 1
                    image_path = artifact_dir / f"page_{page_number:04d}_figure_{figure_count:03d}.png"
                    try:
                        _crop_pdf_region(path, page_number, bbox, image_path)
                    except Exception:
                        logger.warning("Failed to crop PDF figure: %s page=%s bbox=%s", path, page_number, bbox)
                        continue

                    caption = _find_caption_near_bbox(page, bbox, kind="figure")
                    nearby_text = _extract_nearby_text_from_page(page, bbox)
                    ocr_text = _ocr_image_file(image_path, enable_ocr=config.enable_ocr)
                    artifact_path = _artifact_relpath(config, image_path)

                    metadata = {
                        "source_path": doc.source_path,
                        "page": page_number,
                        "bbox": bbox,
                        "caption": caption,
                        "nearby_text": nearby_text,
                        "ocr_text": ocr_text,
                        "artifact_path": artifact_path,
                    }
                    semantic_summary = ""
                    if image_summary_fn is not None:
                        try:
                            semantic_summary = _normalize_text(image_summary_fn(image_path, metadata))
                        except Exception as exc:
                            logger.warning("image_summary_fn failed: %s", exc)

                    text = _build_figure_search_text(
                        source_path=doc.source_path,
                        page=page_number,
                        caption=caption,
                        nearby_text=nearby_text,
                        ocr_text=ocr_text,
                        semantic_summary=semantic_summary,
                    )
                    if not text:
                        continue

                    blocks.append(
                        TextBlock(
                            text=text,
                            page=page_number,
                            block_index=page_number * 10000 + 2000 + figure_count,
                            block_type="pdf_figure",
                            bbox=bbox,
                            artifact_path=artifact_path,
                            structured_content={
                                "object_type": "figure",
                                "caption": caption,
                                "nearby_text": nearby_text,
                                "ocr_text": ocr_text,
                                "semantic_summary": semantic_summary,
                                "image_path": artifact_path,
                            },
                        )
                    )
    finally:
        pdf.close()

    return blocks


def _extract_pdf_table_objects(path: Path, doc: DocumentRecord, config: RAGConfig) -> list[PDFTableObject]:
    import pdfplumber

    tables: list[PDFTableObject] = []
    artifact_dir = config.artifact_dir / "tables" / _sha1(doc.source_path, 12)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                detected_tables = page.find_tables()
            except Exception:
                detected_tables = []

            for table_idx, table in enumerate(detected_tables, start=1):
                try:
                    rows = table.extract() or []
                except Exception:
                    rows = []
                rows = _normalize_table_rows(rows)
                if not rows:
                    continue

                bbox = tuple(float(v) for v in table.bbox)
                caption = _find_table_caption_pdfplumber(page, bbox)

                artifact_path = ""
                image_path = artifact_dir / f"page_{page_idx:04d}_table_{table_idx:03d}.png"
                try:
                    _crop_pdf_region(path, page_idx, bbox, image_path)
                    artifact_path = _artifact_relpath(config, image_path)
                except Exception:
                    logger.warning("Failed to crop PDF table: %s page=%s bbox=%s", path, page_idx, bbox)

                table_id = _sha1(
                    f"{doc.source_path}::page={page_idx}::table={table_idx}::{caption}::{rows[:2]}",
                    24,
                )
                tables.append(
                    PDFTableObject(
                        table_id=table_id,
                        source_path=doc.source_path,
                        title=doc.title,
                        pages=[page_idx],
                        page_heights=[float(page.height)],
                        bboxes=[bbox],
                        rows=rows,
                        caption=caption,
                        artifact_paths=[artifact_path] if artifact_path else [],
                        continued_markers=_extract_continued_markers(caption),
                    )
                )

    return tables


def _normalize_table_rows(rows: list[list[Any]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows:
        cells = [_normalize_text(cell) for cell in row]
        if any(cells):
            normalized.append(cells)

    if not normalized:
        return []

    max_cols = max(len(row) for row in normalized)
    padded = [row + [""] * (max_cols - len(row)) for row in normalized]

    keep_indices: list[int] = []
    for col_idx in range(max_cols):
        values = [row[col_idx] for row in padded]
        if any(value.strip() for value in values):
            keep_indices.append(col_idx)

    return [[row[idx] for idx in keep_indices] for row in padded]


def _find_table_caption_pdfplumber(
    page: Any,
    bbox: tuple[float, float, float, float],
    *,
    max_chars: int = 500,
) -> str:
    pattern = re.compile(
        r"^\s*((table|表)\s*[\dIVXivx0-9\-\.]+[:：.\s].*)",
        re.IGNORECASE,
    )
    try:
        words = page.extract_words() or []
    except Exception:
        return ""

    line_groups: dict[int, list[dict[str, Any]]] = {}
    for word in words:
        y = int(float(word.get("top", 0)) // 5)
        line_groups.setdefault(y, []).append(word)

    candidates: list[tuple[float, str]] = []
    for _, line_words in line_groups.items():
        line_words = sorted(line_words, key=lambda item: float(item.get("x0", 0)))
        text = _normalize_text(" ".join(str(word.get("text", "")) for word in line_words))
        if not pattern.search(text):
            continue

        lx0 = min(float(word.get("x0", 0)) for word in line_words)
        ltop = min(float(word.get("top", 0)) for word in line_words)
        lx1 = max(float(word.get("x1", 0)) for word in line_words)
        lbottom = max(float(word.get("bottom", 0)) for word in line_words)
        line_bbox = (lx0, ltop, lx1, lbottom)

        distance = _rect_distance(bbox, line_bbox)
        if distance <= 120:
            candidates.append((distance, text))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1][:max_chars]


def _extract_continued_markers(text: str) -> list[str]:
    lowered = text.lower()
    markers: list[str] = []
    for marker in ("continued", "cont.", "续表", "接上表", "continued table"):
        if marker in lowered or marker in text:
            markers.append(marker)
    return markers


def _table_header(row: list[str]) -> list[str]:
    return [_normalize_text(cell).lower() for cell in row if _normalize_text(cell)]


def _first_nonempty_row(rows: list[list[str]]) -> list[str]:
    for row in rows:
        if any(cell.strip() for cell in row):
            return row
    return []


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {item for item in a if item}
    sb = {item for item in b if item}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _extract_table_number(caption: str) -> str:
    if not caption:
        return ""
    match = re.search(
        r"(?:table|表)\s*([0-9]+(?:\.[0-9]+)*|[IVXivx]+)",
        caption,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else ""


def _bbox_near_page_bottom(bbox: tuple[float, float, float, float], page_height: float) -> bool:
    return bbox[3] >= page_height * 0.72


def _bbox_near_page_top(bbox: tuple[float, float, float, float], page_height: float) -> bool:
    return bbox[1] <= page_height * 0.35


def _looks_like_continued_table(prev: PDFTableObject, cur: PDFTableObject) -> bool:
    if not prev.pages or not cur.pages:
        return False
    if cur.pages[0] != prev.pages[-1] + 1:
        return False
    if prev.n_cols == 0 or cur.n_cols == 0 or prev.n_cols != cur.n_cols:
        return False

    prev_number = _extract_table_number(prev.caption)
    cur_number = _extract_table_number(cur.caption)
    same_table_number = bool(prev_number and cur_number and prev_number == cur_number)
    continued_marker = bool(prev.continued_markers or cur.continued_markers)

    prev_header = _table_header(_first_nonempty_row(prev.rows))
    cur_header = _table_header(_first_nonempty_row(cur.rows))
    repeated_header = _jaccard(prev_header, cur_header) >= 0.6

    prev_bottom = prev.bboxes[-1]
    cur_top = cur.bboxes[0]
    prev_height = prev.page_heights[-1] if prev.page_heights else 792.0
    cur_height = cur.page_heights[0] if cur.page_heights else 792.0
    layout_continuity = _bbox_near_page_bottom(prev_bottom, prev_height) and _bbox_near_page_top(cur_top, cur_height)

    if same_table_number:
        return True
    if continued_marker:
        return True
    if repeated_header and layout_continuity:
        return True
    if not cur.caption and layout_continuity:
        return True
    return False


def _merge_table_rows(prev_rows: list[list[str]], cur_rows: list[list[str]]) -> list[list[str]]:
    if not prev_rows:
        return cur_rows
    if not cur_rows:
        return prev_rows

    prev_header = _table_header(_first_nonempty_row(prev_rows))
    cur_first = _table_header(_first_nonempty_row(cur_rows))
    if prev_header and cur_first and _jaccard(prev_header, cur_first) >= 0.8:
        return prev_rows + cur_rows[1:]
    return prev_rows + cur_rows


def _merge_two_tables(prev: PDFTableObject, cur: PDFTableObject) -> PDFTableObject:
    return PDFTableObject(
        table_id=prev.table_id,
        source_path=prev.source_path,
        title=prev.title,
        pages=prev.pages + cur.pages,
        page_heights=prev.page_heights + cur.page_heights,
        bboxes=prev.bboxes + cur.bboxes,
        rows=_merge_table_rows(prev.rows, cur.rows),
        caption=prev.caption or cur.caption,
        artifact_paths=prev.artifact_paths + cur.artifact_paths,
        continued_markers=list(dict.fromkeys(prev.continued_markers + cur.continued_markers)),
    )


def _merge_cross_page_tables(tables: list[PDFTableObject]) -> list[PDFTableObject]:
    if not tables:
        return []

    ordered = sorted(
        tables,
        key=lambda table: (
            table.pages[0],
            table.bboxes[0][1] if table.bboxes else 0,
            table.bboxes[0][0] if table.bboxes else 0,
        ),
    )

    merged: list[PDFTableObject] = []
    for table in ordered:
        if merged and _looks_like_continued_table(merged[-1], table):
            merged[-1] = _merge_two_tables(merged[-1], table)
        else:
            merged.append(table)

    return merged


def _table_to_markdown(rows: list[list[str]], *, max_rows: int = 80) -> str:
    if not rows:
        return ""

    limited = rows[:max_rows]
    max_cols = max(len(row) for row in limited)
    padded = [row + [""] * (max_cols - len(row)) for row in limited]

    def clean(cell: str) -> str:
        return _normalize_text(cell).replace("|", "\\|")

    header = padded[0]
    lines = [
        "| " + " | ".join(clean(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(clean(cell) for cell in row) + " |")
    return "\n".join(lines)


def _table_to_html(rows: list[list[str]], *, max_rows: int = 120) -> str:
    import html

    if not rows:
        return ""

    lines = ["<table>"]
    for row_idx, row in enumerate(rows[:max_rows]):
        tag = "th" if row_idx == 0 else "td"
        lines.append("  <tr>")
        for cell in row:
            lines.append(f"    <{tag}>{html.escape(cell)}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _summarize_table_heuristic(table: PDFTableObject) -> str:
    header = _first_nonempty_row(table.rows)
    header_text = "、".join(cell for cell in header if cell.strip())
    pages = str(table.pages[0]) if len(table.pages) == 1 else f"{table.pages[0]}-{table.pages[-1]}"

    parts = [
        "表格对象。",
        f"来源文件：{table.source_path}。",
        f"页码：{pages}。",
        f"规模：{table.n_rows} 行，{table.n_cols} 列。",
    ]
    if table.caption:
        parts.append(f"表题：{table.caption}。")
    if header_text:
        parts.append(f"主要字段：{header_text}。")

    flat_cells = [
        cell
        for row in table.rows[1:]
        for cell in row
        if cell and len(cell) <= 80
    ]
    common = [item for item, _ in Counter(flat_cells).most_common(12)]
    if common:
        parts.append("代表性条目：" + "；".join(common) + "。")

    return _normalize_text("".join(parts))


def _table_objects_to_text_blocks(
    tables: list[PDFTableObject],
    *,
    table_summary_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []

    for table_idx, table in enumerate(tables, start=1):
        markdown = _table_to_markdown(table.rows)
        html = _table_to_html(table.rows)
        payload = {
            "object_type": "table",
            "source_path": table.source_path,
            "pages": table.pages,
            "bboxes": table.bboxes,
            "caption": table.caption,
            "n_rows": table.n_rows,
            "n_cols": table.n_cols,
            "rows": table.rows,
            "markdown": markdown,
            "html": html,
            "artifact_paths": table.artifact_paths,
            "continued_markers": table.continued_markers,
        }

        if table_summary_fn is not None:
            try:
                summary = _normalize_text(table_summary_fn(payload))
            except Exception as exc:
                logger.warning("table_summary_fn failed: %s", exc)
                summary = _summarize_table_heuristic(table)
        else:
            summary = _summarize_table_heuristic(table)

        pages = str(table.pages[0]) if len(table.pages) == 1 else f"{table.pages[0]}-{table.pages[-1]}"
        text = _normalize_text(
            "\n".join(
                [
                    "[PDF Table]",
                    f"Source: {table.source_path}",
                    f"Pages: {pages}",
                    f"Caption: {table.caption}" if table.caption else "",
                    f"Summary: {summary}",
                    "Markdown:",
                    markdown,
                ]
            )
        )

        blocks.append(
            TextBlock(
                text=text,
                page=table.pages[0] if table.pages else None,
                block_index=(table.pages[0] if table.pages else 0) * 10000 + 4000 + table_idx,
                block_type="pdf_table",
                bbox=table.bboxes[0] if table.bboxes else None,
                artifact_path=table.artifact_paths[0] if table.artifact_paths else "",
                structured_content=payload,
            )
        )

    return blocks


def _extract_docx_blocks(path: Path) -> list[TextBlock]:
    from docx import Document

    doc = Document(str(path))
    blocks: list[TextBlock] = []

    parts: list[str] = []
    for para in doc.paragraphs:
        text = _normalize_text(para.text)
        if text:
            parts.append(text)

    if parts:
        blocks.append(
            TextBlock(
                text="\n\n".join(parts),
                page=None,
                block_index=0,
                block_type="docx_text",
            )
        )

    for table_idx, table in enumerate(doc.tables, start=1):
        rows: list[list[str]] = []
        for row in table.rows:
            cells = [_normalize_text(cell.text) for cell in row.cells]
            if any(cells):
                rows.append(cells)

        rows = _normalize_table_rows(rows)
        if not rows:
            continue

        markdown = _table_to_markdown(rows)
        payload = {
            "object_type": "table",
            "source_path": path.name,
            "pages": [],
            "bboxes": [],
            "caption": f"docx table {table_idx}",
            "n_rows": len(rows),
            "n_cols": max(len(row) for row in rows),
            "rows": rows,
            "markdown": markdown,
            "html": _table_to_html(rows),
            "artifact_paths": [],
            "continued_markers": [],
        }
        text = _normalize_text(
            "\n".join(
                [
                    "[DOCX Table]",
                    f"Source: {path.name}",
                    f"Caption: docx table {table_idx}",
                    "Markdown:",
                    markdown,
                ]
            )
        )
        blocks.append(
            TextBlock(
                text=text,
                page=None,
                block_index=10000 + table_idx,
                block_type="docx_table",
                structured_content=payload,
            )
        )

    return blocks


def _extract_doc_with_textract(path: Path) -> str:
    import textract
    raw = textract.process(str(path))
    return raw.decode("utf-8", errors="ignore")


def _extract_doc_with_antiword(path: Path) -> str:
    antiword = shutil.which("antiword")
    if not antiword:
        raise RuntimeError("antiword is not available")
    result = subprocess.run(
        [antiword, str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return result.stdout


def _extract_doc_with_libreoffice(path: Path) -> str:
    libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not libreoffice:
        raise RuntimeError("libreoffice/soffice is not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        subprocess.run(
            [
                libreoffice,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(tmp_path),
                str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        candidates = list(tmp_path.glob("*.docx"))
        if not candidates:
            raise RuntimeError("libreoffice conversion did not produce a docx file")
        return "\n\n".join(block.text for block in _extract_docx_blocks(candidates[0]))


def _extract_doc_blocks(path: Path) -> list[TextBlock]:
    errors: list[str] = []
    for extractor in (
        _extract_doc_with_textract,
        _extract_doc_with_antiword,
        _extract_doc_with_libreoffice,
    ):
        try:
            text = _normalize_text(extractor(path))
            if text:
                return [
                    TextBlock(
                        text=text,
                        page=None,
                        block_index=0,
                        block_type="doc_text",
                    )
                ]
        except Exception as exc:
            errors.append(f"{extractor.__name__}: {exc}")

    raise RuntimeError(
        "无法解析 .doc 文件。可安装 textract、antiword，或配置 libreoffice/soffice。"
        f" 详细错误：{' | '.join(errors)}"
    )


def _extract_text_blocks(path: Path) -> list[TextBlock]:
    text = _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    block_type = "markdown_text" if path.suffix.lower() in {".md", ".markdown"} else "text"
    return [
        TextBlock(
            text=text,
            page=None,
            block_index=0,
            block_type=block_type,
        )
    ]


def _extract_image_blocks(
    path: Path,
    config: RAGConfig,
    *,
    source_doc: DocumentRecord | None = None,
    image_summary_fn: Callable[[Path, dict[str, Any]], str] | None = None,
) -> list[TextBlock]:
    ocr_text = _ocr_image_file(path, enable_ocr=config.enable_ocr)
    metadata = {
        "source_path": source_doc.source_path if source_doc else path.name,
        "image_path": str(path),
        "ocr_text": ocr_text,
    }

    semantic_summary = ""
    if image_summary_fn is not None:
        try:
            semantic_summary = _normalize_text(image_summary_fn(path, metadata))
        except Exception as exc:
            logger.warning("image_summary_fn failed for standalone image: %s", exc)

    text = _build_figure_search_text(
        source_path=source_doc.source_path if source_doc else path.name,
        page=None,
        caption="",
        nearby_text="",
        ocr_text=ocr_text,
        semantic_summary=semantic_summary,
    )
    if not text:
        return []

    return [
        TextBlock(
            text=text,
            page=None,
            block_index=0,
            block_type="image",
            artifact_path=str(path),
            structured_content={
                "object_type": "image",
                "image_path": str(path),
                "ocr_text": ocr_text,
                "semantic_summary": semantic_summary,
            },
        )
    ]


# ---------------------------------------------------------------------------
# Sentence-level recursive splitting
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?；;])\s+|(?<=[。！？!?；;])|(?<=[.!?])\s+"
)


def _split_to_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []

    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
    units: list[str] = []
    for para in paragraphs:
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(para) if part.strip()]
        units.extend(parts)
        units.append("\n\n")

    if units and units[-1] == "\n\n":
        units.pop()
    return units


def _recursive_split_long_sentence(sentence: str, max_len: int) -> list[str]:
    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence] if sentence else []

    separators = ["\n", "，", ",", "、", "：", ":", " ", ""]

    def split_by_separator(text: str, sep_idx: int) -> list[str]:
        if len(text) <= max_len:
            return [text.strip()] if text.strip() else []

        sep = separators[sep_idx] if sep_idx < len(separators) else ""
        if sep == "":
            return [
                text[start : start + max_len].strip()
                for start in range(0, len(text), max_len)
                if text[start : start + max_len].strip()
            ]

        parts = [part.strip() for part in text.split(sep) if part.strip()]
        if len(parts) <= 1:
            return split_by_separator(text, sep_idx + 1)

        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    merged.extend(split_by_separator(current, sep_idx + 1))
                current = part
        if current:
            merged.extend(split_by_separator(current, sep_idx + 1))
        return merged

    return split_by_separator(sentence, 0)


def recursive_sentence_split(
    text: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 180,
) -> list[str]:
    sentence_units: list[str] = []
    for sentence in _split_to_sentences(text):
        if sentence == "\n\n":
            sentence_units.append(sentence)
        else:
            sentence_units.extend(_recursive_split_long_sentence(sentence, chunk_size))

    chunks: list[str] = []
    current = ""

    for unit in sentence_units:
        if unit == "\n\n":
            if current and not current.endswith("\n\n"):
                current += "\n\n"
            continue

        candidate = unit if not current else f"{current}{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            current = unit

    if current.strip():
        chunks.append(current.strip())

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            overlapped.append(chunk)
            continue
        prefix = chunks[idx - 1][-chunk_overlap:].strip()
        overlapped.append(f"{prefix}\n\n{chunk}" if prefix else chunk)

    return overlapped


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(vec_a, vec_b)))


def _join_sentence_units(units: Sequence[str]) -> str:
    text = ""
    for unit in units:
        if unit == "\n\n":
            if text and not text.endswith("\n\n"):
                text += "\n\n"
            continue
        if not text or text.endswith(("\n\n", " ", "\n")):
            text += unit
        else:
            text += f" {unit}"
    return _normalize_text(text)


def _split_units_for_semantic_chunking(text: str, max_len: int) -> list[str]:
    units: list[str] = []
    for sentence in _split_to_sentences(text):
        if sentence == "\n\n":
            if units and units[-1] != "\n\n":
                units.append(sentence)
            continue
        units.extend(_recursive_split_long_sentence(sentence, max_len))
    while units and units[-1] == "\n\n":
        units.pop()
    return units


def _segment_length(units: Sequence[str], start: int, end: int) -> int:
    return sum(len(unit) for unit in units[start:end] if unit != "\n\n")


def _segment_has_sentence_boundary(units: Sequence[str], start: int, end: int) -> bool:
    return sum(1 for unit in units[start:end] if unit != "\n\n") > 1


def _semantic_boundary_scores(
    units: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> dict[int, float]:
    sentence_indexes = [index for index, unit in enumerate(units) if unit != "\n\n"]
    sentence_position = {unit_index: pos for pos, unit_index in enumerate(sentence_indexes)}
    scores: dict[int, float] = {}

    for boundary in range(1, len(units)):
        left_index = boundary - 1
        while left_index >= 0 and units[left_index] == "\n\n":
            left_index -= 1
        right_index = boundary
        while right_index < len(units) and units[right_index] == "\n\n":
            right_index += 1

        left_pos = sentence_position.get(left_index)
        right_pos = sentence_position.get(right_index)
        if left_pos is None or right_pos is None:
            scores[boundary] = -1.0
            continue
        scores[boundary] = _cosine_similarity(vectors[left_pos], vectors[right_pos])

    return scores


def _choose_semantic_split(
    units: Sequence[str],
    boundary_scores: dict[int, float],
    start: int,
    end: int,
    *,
    min_chars: int,
    force_split: bool,
) -> int | None:
    candidates: list[tuple[float, int]] = []
    for boundary in range(start + 1, end):
        left_len = _segment_length(units, start, boundary)
        right_len = _segment_length(units, boundary, end)
        if left_len < min_chars or right_len < min_chars:
            continue
        score = boundary_scores.get(boundary)
        if score is None:
            continue
        midpoint_penalty = abs((boundary - start) / max(end - start, 1) - 0.5) * 0.08
        candidates.append((score + midpoint_penalty, boundary))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    if not force_split:
        return None

    midpoint = (start + end) // 2
    fallback_boundaries = sorted(
        range(start + 1, end),
        key=lambda boundary: abs(boundary - midpoint),
    )
    for boundary in fallback_boundaries:
        if _segment_has_sentence_boundary(units, start, boundary) and _segment_has_sentence_boundary(units, boundary, end):
            return boundary
    return None


def semantic_recursive_sentence_split(
    text: str,
    config: RAGConfig,
) -> list[str]:
    units = _split_units_for_semantic_chunking(text, config.chunk_size)
    sentence_units = [unit for unit in units if unit != "\n\n"]
    if not sentence_units:
        return []
    if len(sentence_units) == 1:
        return recursive_sentence_split(
            sentence_units[0],
            chunk_size=config.chunk_size,
            chunk_overlap=0,
        )

    try:
        vectors = _encode_texts(sentence_units, config)
    except Exception as exc:
        logger.warning("Semantic chunk embedding failed, fallback to recursive split: %s", exc)
        return recursive_sentence_split(
            text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    if len(vectors) != len(sentence_units):
        logger.warning(
            "Semantic chunk embedding count mismatch, fallback to recursive split: vectors=%s sentences=%s",
            len(vectors),
            len(sentence_units),
        )
        return recursive_sentence_split(
            text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    boundary_scores = _semantic_boundary_scores(units, vectors)
    min_chars = max(80, min(config.semantic_chunk_min_chars, config.chunk_size // 2))
    max_chars = max(config.chunk_size, min_chars * 2)
    threshold = float(config.semantic_chunk_similarity_threshold)
    ranges: list[tuple[int, int]] = []

    def split_range(start: int, end: int) -> None:
        segment_len = _segment_length(units, start, end)
        if segment_len <= max_chars and not _segment_has_sentence_boundary(units, start, end):
            ranges.append((start, end))
            return

        boundary_values = [
            boundary_scores[boundary]
            for boundary in range(start + 1, end)
            if boundary in boundary_scores
        ]
        weakest_similarity = min(boundary_values) if boundary_values else 1.0
        force_split = segment_len > max_chars
        should_split = force_split or (
            segment_len >= min_chars * 2
            and weakest_similarity < threshold
        )
        if not should_split:
            ranges.append((start, end))
            return

        boundary = _choose_semantic_split(
            units,
            boundary_scores,
            start,
            end,
            min_chars=min_chars,
            force_split=force_split,
        )
        if boundary is None:
            ranges.append((start, end))
            return

        split_range(start, boundary)
        split_range(boundary, end)

    split_range(0, len(units))

    overlap = max(0, int(config.semantic_chunk_sentence_overlap))
    chunks: list[str] = []
    for range_index, (start, end) in enumerate(ranges):
        expanded_start = start
        if overlap > 0 and range_index > 0:
            sentence_count = 0
            cursor = start - 1
            while cursor >= 0 and sentence_count < overlap:
                if units[cursor] != "\n\n":
                    sentence_count += 1
                expanded_start = cursor
                cursor -= 1
        chunk = _join_sentence_units(units[expanded_start:end])
        if chunk:
            chunks.append(chunk)
    return chunks


def _build_chunks_for_file(
    path: Path,
    kb_root: Path,
    config: RAGConfig,
    *,
    image_summary_fn: Callable[[Path, dict[str, Any]], str] | None = None,
    table_summary_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[RAGChunk]:
    doc = _make_doc_record(path, kb_root)
    blocks = _extract_blocks(
        path=path,
        doc=doc,
        config=config,
        image_summary_fn=image_summary_fn,
        table_summary_fn=table_summary_fn,
    )

    chunks: list[RAGChunk] = []
    chunk_index = 0

    for block in blocks:
        if not block.text.strip():
            continue

        if config.semantic_chunking:
            parts = semantic_recursive_sentence_split(block.text, config)
        else:
            parts = recursive_sentence_split(
                block.text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        for part in parts:
            raw_id = (
                f"{doc.source_path}::page={block.page}::block={block.block_index}::"
                f"type={block.block_type}::chunk={chunk_index}::{part[:120]}"
            )
            chunks.append(
                RAGChunk(
                    chunk_id=_sha1(raw_id, 24),
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    title=doc.title,
                    text=part,
                    chunk_index=chunk_index,
                    page=block.page,
                    metadata={
                        "suffix": doc.suffix,
                        "size_bytes": doc.size_bytes,
                        "mtime": doc.mtime,
                        "block_index": block.block_index,
                        "block_type": block.block_type,
                        "bbox": block.bbox,
                        "artifact_path": block.artifact_path,
                        "structured_content": block.structured_content,
                    },
                )
            )
            chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Index build: chunks.jsonl + BM25 + Qdrant dense vectors
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _build_bm25(chunks: Sequence[RAGChunk]) -> BM25Bundle:
    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(chunk.text) for chunk in chunks]
    return BM25Bundle(
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        tokenized_corpus=corpus,
        bm25=BM25Okapi(corpus),
    )


_EMBEDDER_CACHE: dict[str, Any] = {}
_RERANKER_CACHE: dict[str, Any] = {}


def _release_reranker_after_use_enabled() -> bool:
    return os.getenv("RAG_RELEASE_RERANKER_AFTER_USE", "true").lower() in {"1", "true", "yes"}


def _reranker_cache_key(config: RAGConfig) -> str:
    return f"{config.rerank_model_path}::{config.device or ''}"


def _get_embedder(config: RAGConfig) -> Any:
    cache_key = f"{config.embedding_model_path}::{config.device or ''}"
    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]

    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {}
    if config.device:
        kwargs["device"] = config.device
    model = SentenceTransformer(config.embedding_model_path, **kwargs)
    _EMBEDDER_CACHE[cache_key] = model
    return model


def _encode_texts(
    texts: Sequence[str],
    config: RAGConfig,
    *,
    batch_size: int = 16,
) -> list[list[float]]:
    if not texts:
        return []
    normalized_texts = [
        _embedding_safe_text(item)
        for item in texts
    ]
    model = _get_embedder(config)
    try:
        vectors = model.encode(
            normalized_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError as exc:
        logger.warning("Batch embedding failed, retrying item by item: %s", exc)
        recovered: list[list[float]] = []
        fallback_vector: list[float] | None = None
        for index, text in enumerate(normalized_texts):
            try:
                item_vectors = model.encode(
                    [text],
                    batch_size=1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vector = list(map(float, item_vectors[0]))
                recovered.append(vector)
                fallback_vector = vector
            except Exception as item_exc:
                logger.warning(
                    "Embedding failed for text segment index=%s, using fallback vector: %s",
                    index,
                    item_exc,
                )
                if fallback_vector is None:
                    fallback_vectors = model.encode(
                        ["unreadable text segment"],
                        batch_size=1,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    fallback_vector = list(map(float, fallback_vectors[0]))
                recovered.append(list(fallback_vector))
        return recovered
    return [list(map(float, vector)) for vector in vectors]


def _get_qdrant_client(config: RAGConfig) -> Any:
    from qdrant_client import QdrantClient
    config.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.qdrant_path))


def _recreate_collection(client: Any, collection_name: str, vector_size: int) -> None:
    from qdrant_client import models

    client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def _upsert_dense_vectors(
    chunks: Sequence[RAGChunk],
    vectors: Sequence[Sequence[float]],
    config: RAGConfig,
) -> None:
    if not chunks:
        return
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks/vectors 数量不一致：{len(chunks)} != {len(vectors)}")

    from qdrant_client import models

    with _qdrant_lock(config):
        client = _get_qdrant_client(config)
        _recreate_collection(client, config.collection_name, len(vectors[0]))

        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=list(vector),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "source_path": chunk.source_path,
                        "title": chunk.title,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "block_type": (chunk.metadata or {}).get("block_type", "text"),
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]
            client.upsert(collection_name=config.collection_name, points=points)


def build_rag_index(
    knowledge_base_path: str = "",
    files: list[str] | None = None,
    *,
    index_dir: str = "",
    clean: bool = True,
    embedding_model_path: str = "",
    rerank_model_path: str = "",
    image_summary_fn: Callable[[Path, dict[str, Any]], str] | None = None,
    table_summary_fn: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Build local RAG index explicitly.

    run_rag(...) never rebuilds the index. Call this function when:
        - local_knowledge files changed;
        - chunking strategy changed;
        - embedding model changed.
    """
    started = time.perf_counter()
    config = get_rag_config(
        knowledge_base_path,
        index_dir=index_dir,
        embedding_model_path=embedding_model_path,
        rerank_model_path=rerank_model_path,
    )
    if image_summary_fn is None and config.enable_image_summary:
        image_summary_fn = _default_image_summary

    config.knowledge_base_path.mkdir(parents=True, exist_ok=True)
    if clean and config.index_dir.exists():
        shutil.rmtree(config.index_dir)
    config.index_dir.mkdir(parents=True, exist_ok=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    source_files = _iter_files(config.knowledge_base_path, files)
    chunks: list[RAGChunk] = []
    errors: list[dict[str, str]] = []

    for path in source_files:
        try:
            chunks.extend(
                _build_chunks_for_file(
                    path=path,
                    kb_root=config.knowledge_base_path,
                    config=config,
                    image_summary_fn=image_summary_fn,
                    table_summary_fn=table_summary_fn,
                )
            )
        except Exception as exc:
            logger.exception("解析文件失败：%s", path)
            errors.append(
                {
                    "path": _safe_relative_path(path, config.knowledge_base_path),
                    "error": str(exc),
                }
            )

    _write_jsonl(config.chunks_path, (chunk.to_json() for chunk in chunks))

    vector_count = 0
    if chunks:
        bm25 = _build_bm25(chunks)
        bm25.save(config.bm25_path)

        vectors = _encode_texts([chunk.text for chunk in chunks], config)
        _upsert_dense_vectors(chunks, vectors, config)
        vector_count = len(vectors)
    else:
        if config.bm25_path.exists():
            config.bm25_path.unlink()
        if config.qdrant_path.exists():
            shutil.rmtree(config.qdrant_path)

    meta = {
        "collection_name": config.collection_name,
        "embedding_model_path": config.embedding_model_path,
        "rerank_model_path": config.rerank_model_path,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "chunking": {
            "strategy": "semantic_recursive_sentence_similarity"
            if config.semantic_chunking
            else "sentence_length_recursive",
            "semantic_chunk_min_chars": config.semantic_chunk_min_chars,
            "semantic_chunk_similarity_threshold": config.semantic_chunk_similarity_threshold,
            "semantic_chunk_sentence_overlap": config.semantic_chunk_sentence_overlap,
        },
        "knowledge_base_path": str(config.knowledge_base_path),
        "index_dir": str(config.index_dir),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "pdf_layout_objects": {
            "text": True,
            "figures": config.enable_pdf_figures,
            "tables": config.enable_pdf_tables,
            "ocr": config.enable_ocr,
            "image_semantic_summary": bool(image_summary_fn),
            "cross_page_table_merge": True,
        },
    }
    (config.index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "knowledge_base_path": str(config.knowledge_base_path),
        "index_dir": str(config.index_dir),
        "source_documents": len(source_files),
        "chunk_count": len(chunks),
        "vector_count": vector_count,
        "errors": errors,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Retrieval: BM25 + Dense + RRF + ReRank
# ---------------------------------------------------------------------------


def _load_chunks(config: RAGConfig) -> list[RAGChunk]:
    return [RAGChunk.from_json(row) for row in _read_jsonl(config.chunks_path)]


def _ensure_index_exists(config: RAGConfig) -> None:
    missing = [
        str(path)
        for path in (config.chunks_path, config.bm25_path, config.qdrant_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "本地 RAG 索引不存在或不完整，请先显式调用 build_rag_index(...)。"
            f" 缺失：{missing}"
        )


def _normalize_file_filter(files: list[str] | None) -> set[str]:
    if not files:
        return set()
    allowed: set[str] = set()
    for item in files:
        path = Path(item)
        allowed.add(str(item))
        allowed.add(path.name)
        allowed.add(path.as_posix())
    return allowed


def _chunk_allowed(chunk: RAGChunk, allowed_files: set[str]) -> bool:
    if not allowed_files:
        return True
    return chunk.source_path in allowed_files or Path(chunk.source_path).name in allowed_files


def _bm25_search(
    query: str,
    bm25: BM25Bundle,
    chunks_by_id: dict[str, RAGChunk],
    *,
    top_k: int,
    allowed_files: set[str] | None = None,
) -> list[tuple[str, float]]:
    tokens = _tokenize(query)
    if not tokens:
        return []

    allowed = allowed_files or set()
    scores = bm25.bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)

    results: list[tuple[str, float]] = []
    for idx, score in ranked:
        chunk_id = bm25.chunk_ids[idx]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None or not _chunk_allowed(chunk, allowed):
            continue
        results.append((chunk_id, float(score)))
        if len(results) >= top_k:
            break
    return results


def _dense_search(
    query: str,
    chunks_by_id: dict[str, RAGChunk],
    config: RAGConfig,
    *,
    top_k: int,
    allowed_files: set[str] | None = None,
) -> list[tuple[str, float]]:
    allowed = allowed_files or set()
    query_vector = _encode_texts([query], config, batch_size=32)[0]

    with _qdrant_lock(config):
        client = _get_qdrant_client(config)
        limit = top_k if not allowed else max(top_k * 5, 100)

        try:
            response = client.query_points(
                collection_name=config.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            points = list(getattr(response, "points", response))
        except Exception as exc:
            legacy_search = getattr(client, "search", None)
            if not callable(legacy_search):
                raise exc
            points = legacy_search(
                collection_name=config.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

    results: list[tuple[str, float]] = []
    for point in points:
        payload = getattr(point, "payload", None) or {}
        chunk_id = str(payload.get("chunk_id") or "")
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None or not _chunk_allowed(chunk, allowed):
            continue
        results.append((chunk_id, float(getattr(point, "score", 0.0) or 0.0)))
        if len(results) >= top_k:
            break

    return results


def _rrf_fusion(
    dense_results: Sequence[tuple[str, float]],
    sparse_results: Sequence[tuple[str, float]],
    chunks_by_id: dict[str, RAGChunk],
    config: RAGConfig,
) -> list[SearchHit]:
    dense_rank = {cid: idx for idx, (cid, _) in enumerate(dense_results, start=1)}
    sparse_rank = {cid: idx for idx, (cid, _) in enumerate(sparse_results, start=1)}
    dense_scores = {cid: score for cid, score in dense_results}
    sparse_scores = {cid: score for cid, score in sparse_results}

    hits: list[SearchHit] = []
    for chunk_id in set(dense_rank) | set(sparse_rank):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        rrf_score = 0.0
        if chunk_id in dense_rank:
            rrf_score += 1.0 / (config.rrf_k + dense_rank[chunk_id])
        if chunk_id in sparse_rank:
            rrf_score += 1.0 / (config.rrf_k + sparse_rank[chunk_id])

        hits.append(
            SearchHit(
                chunk=chunk,
                score=rrf_score,
                dense_score=dense_scores.get(chunk_id, 0.0),
                sparse_score=sparse_scores.get(chunk_id, 0.0),
                rrf_score=rrf_score,
                dense_rank=dense_rank.get(chunk_id),
                sparse_rank=sparse_rank.get(chunk_id),
            )
        )

    hits.sort(key=lambda hit: hit.rrf_score, reverse=True)
    return hits[: config.rrf_top_k]


def _get_reranker(config: RAGConfig) -> Any:
    cache_key = _reranker_cache_key(config)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    from FlagEmbedding import FlagReranker

    kwargs: dict[str, Any] = {"use_fp16": True}
    if config.device:
        kwargs["devices"] = [config.device]

    reranker = FlagReranker(config.rerank_model_path, batch_size=1, **kwargs)
    _RERANKER_CACHE[cache_key] = reranker
    return reranker


def _release_reranker(config: RAGConfig) -> None:
    cache_key = _reranker_cache_key(config)
    reranker = _RERANKER_CACHE.pop(cache_key, None)
    if reranker is None:
        return

    try:
        model = getattr(reranker, "model", None)
        if model is not None:
            model.to("cpu")
    except Exception as exc:
        logger.debug("Failed to move reranker to CPU before release: %s", exc)

    del reranker
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.debug("Failed to clear CUDA cache after reranker release: %s", exc)


def _rerank(query: str, hits: Sequence[SearchHit], config: RAGConfig) -> list[SearchHit]:
    if not hits or not config.enable_rerank:
        return list(hits)

    try:
        reranker = _get_reranker(config)
        pairs = [[query, hit.chunk.text] for hit in hits]
        raw_scores = reranker.compute_score(pairs)
    finally:
        if _release_reranker_after_use_enabled():
            _release_reranker(config)

    if isinstance(raw_scores, (int, float)):
        scores = [float(raw_scores)]
    else:
        scores = [float(score) for score in raw_scores]

    reranked: list[SearchHit] = []
    for hit, score in zip(hits, scores):
        reranked.append(
            SearchHit(
                chunk=hit.chunk,
                score=score,
                dense_score=hit.dense_score,
                sparse_score=hit.sparse_score,
                rrf_score=hit.rrf_score,
                rerank_score=score,
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
            )
        )

    reranked.sort(
        key=lambda hit: float(hit.rerank_score if hit.rerank_score is not None else hit.score),
        reverse=True,
    )
    return reranked


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _format_score(hit: SearchHit) -> float:
    return round(float(hit.rerank_score if hit.rerank_score is not None else hit.score), 6)


def _build_reference(hit: SearchHit, index: int) -> dict[str, Any]:
    metadata = dict(hit.chunk.metadata or {})
    return {
        "id": index,
        "source_path": hit.chunk.source_path,
        "title": hit.chunk.title,
        "page": hit.chunk.page,
        "chunk_index": hit.chunk.chunk_index,
        "block_type": metadata.get("block_type", "text"),
        "bbox": metadata.get("bbox"),
        "artifact_path": metadata.get("artifact_path", ""),
        "score": _format_score(hit),
        "rrf_score": round(float(hit.rrf_score), 6),
        "dense_score": round(float(hit.dense_score), 6),
        "sparse_score": round(float(hit.sparse_score), 6),
        "rerank_score": (
            round(float(hit.rerank_score), 6)
            if hit.rerank_score is not None
            else None
        ),
    }


def _format_evidence_block(hit: SearchHit, index: int) -> str:
    ref = _build_reference(hit, index)
    page_text = f", page={ref['page']}" if ref["page"] is not None else ""
    artifact_text = f", artifact={ref['artifact_path']}" if ref["artifact_path"] else ""
    return (
        f"[{index}] source={ref['source_path']}{page_text}, "
        f"type={ref['block_type']}, chunk={ref['chunk_index']}, "
        f"score={ref['score']}{artifact_text}\n"
        f"{hit.chunk.text}"
    )


def _format_result_content(hits: Sequence[SearchHit]) -> str:
    return "\n\n".join(
        _format_evidence_block(hit, idx)
        for idx, hit in enumerate(hits, start=1)
    )


def _format_grouped_results(hits: Sequence[SearchHit]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for idx, hit in enumerate(hits, start=1):
        key = hit.chunk.source_path
        if key not in groups:
            groups[key] = {
                "source_path": hit.chunk.source_path,
                "title": hit.chunk.title,
                "items": [],
            }
        groups[key]["items"].append(
            {
                "rank": idx,
                "page": hit.chunk.page,
                "chunk_index": hit.chunk.chunk_index,
                "block_type": (hit.chunk.metadata or {}).get("block_type", "text"),
                "score": _format_score(hit),
                "text": hit.chunk.text,
                "metadata": dict(hit.chunk.metadata or {}),
            }
        )
    return list(groups.values())


def run_rag(
    query: str,
    knowledge_base_path: str = "",
    index_dir: str = "",
    files: list[str] | None = None,
    history: list | None = None,
) -> str | dict:
    """Retrieve local evidence only.

    Args:
        query: user query.
        knowledge_base_path: default is agent/data/local_knowledge.
        index_dir: optional explicit index directory for the given knowledge base.
        files: optional file filter. It does not rebuild index.
        history: only counted in metadata; no query rewriting is done here.

    Returns:
        Formatted dict containing:
            - content: readable evidence string
            - chunks: machine-readable chunk list
            - references: compact evidence references
            - grouped_results: evidence grouped by source file
            - retrieval_trace and metrics
    """
    started = time.perf_counter()
    query = str(query or "").strip()
    if not query:
        return {
            "status": "error",
            "tool_name": "local_knowledge_base",
            "message": "query 不能为空",
            "answer": "",
            "query": query,
            "content": "",
            "artifacts": [],
            "chunks": [],
            "references": [],
            "grouped_results": [],
            "metrics": {"tool_ms": round((time.perf_counter() - started) * 1000, 2)},
            "meta": {},
        }

    config = get_rag_config(knowledge_base_path, index_dir=index_dir)
    _ensure_index_exists(config)

    chunks = _load_chunks(config)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    bm25 = BM25Bundle.load(config.bm25_path)
    allowed_files = _normalize_file_filter(files)

    dense_results = _dense_search(
        query,
        chunks_by_id,
        config,
        top_k=config.dense_top_k,
        allowed_files=allowed_files,
    )
    sparse_results = _bm25_search(
        query,
        bm25,
        chunks_by_id,
        top_k=config.sparse_top_k,
        allowed_files=allowed_files,
    )
    fused_hits = _rrf_fusion(dense_results, sparse_results, chunks_by_id, config)

    rerank_used = False
    rerank_error = ""
    try:
        final_hits = _rerank(query, fused_hits, config)[: config.final_top_k]
        rerank_used = config.enable_rerank and bool(fused_hits)
    except Exception as exc:
        logger.warning("ReRank failed, fallback to RRF results: %s", exc)
        final_hits = fused_hits[: config.final_top_k]
        rerank_error = str(exc)

    content = _format_result_content(final_hits)
    references = [
        _build_reference(hit, idx)
        for idx, hit in enumerate(final_hits, start=1)
    ]
    artifacts = [
        {
            "kind": "rag_artifact",
            "path": str(reference.get("artifact_path") or ""),
            "source_path": str(reference.get("source_path") or ""),
            "page": reference.get("page"),
            "block_type": reference.get("block_type"),
        }
        for reference in references
        if reference.get("artifact_path")
    ]
    retrieval_trace = {
        "pipeline": ["bm25", "dense_vector", "rrf", "rerank"],
        "dense_top_k": config.dense_top_k,
        "sparse_top_k": config.sparse_top_k,
        "rrf_top_k": config.rrf_top_k,
        "final_top_k": config.final_top_k,
        "rrf_k": config.rrf_k,
        "rerank_used": rerank_used,
        "rerank_error": rerank_error,
        "history_items": len(history or []),
        "file_filter": list(files or []),
        "index_dir": str(config.index_dir),
        "knowledge_base_path": str(config.knowledge_base_path),
        "collection_name": config.collection_name,
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
    }
    metrics = {
        "tool_ms": round((time.perf_counter() - started) * 1000, 2),
        "total_chunks": len(chunks),
        "dense_hits": len(dense_results),
        "sparse_hits": len(sparse_results),
        "rrf_hits": len(fused_hits),
        "final_hits": len(final_hits),
    }

    return {
        "status": "ok",
        "tool_name": "local_knowledge_base",
        "query": query,
        "answer": content,
        "message": content,
        "local_answer": content,
        "content": content,
        "artifacts": artifacts,
        "chunks": [hit.to_dict(include_text=True) for hit in final_hits],
        "references": references,
        "grouped_results": _format_grouped_results(final_hits),
        "retrieval_trace": retrieval_trace,
        "metrics": metrics,
        "meta": retrieval_trace,
    }


__all__ = [
    "RAGConfig",
    "RAGChunk",
    "SearchHit",
    "TextBlock",
    "PDFTableObject",
    "SUPPORTED_SUFFIXES",
    "DOCUMENT_SUFFIXES",
    "IMAGE_SUFFIXES",
    "build_rag_index",
    "recursive_sentence_split",
    "semantic_recursive_sentence_split",
    "run_rag",
]
