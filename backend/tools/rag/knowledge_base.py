from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from docx import Document
from pypdf import PdfReader

from .config import RAGConfig


@dataclass(slots=True)
class ChunkRecord:
    chunk_uid: str
    chunk_id: int
    text: str
    source_path: str
    file_name: str
    doc_type: str
    page: int | None = None
    section: str | None = None

    def to_payload(self) -> dict:
        payload = {
            "chunk_uid": self.chunk_uid,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
        }
        if self.page is not None:
            payload["page"] = self.page
        if self.section:
            payload["section"] = self.section
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> "ChunkRecord":
        return cls(
            chunk_uid=payload["chunk_uid"],
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            source_path=payload["source_path"],
            file_name=payload["file_name"],
            doc_type=payload["doc_type"],
            page=payload.get("page"),
            section=payload.get("section"),
        )


class KnowledgeBaseBuilder:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.supported_suffixes = {".pdf", ".docx"}

    def scan_documents(self) -> list[Path]:
        return sorted(
            path
            for path in self.config.local_knowledge_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.supported_suffixes
        )

    def build(self) -> tuple[list[ChunkRecord], list[str]]:
        chunks: list[ChunkRecord] = []
        errors: list[str] = []

        for path in self.scan_documents():
            try:
                chunks.extend(self._load_and_chunk_document(path))
            except Exception as exc:  # pragma: no cover - best effort ingestion
                errors.append(f"{path}: {exc}")

        return chunks, errors

    def _load_and_chunk_document(self, path: Path) -> list[ChunkRecord]:
        if path.suffix.lower() == ".pdf":
            segments = self._load_pdf(path)
            doc_type = "pdf"
        elif path.suffix.lower() == ".docx":
            segments = self._load_docx(path)
            doc_type = "docx"
        else:
            return []

        relative_source = path.relative_to(self.config.project_root).as_posix()
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for segment in segments:
            for piece in self._split_text(segment["text"]):
                chunk_index += 1
                stable_key = f"{relative_source}::{chunk_index}"
                chunks.append(
                    ChunkRecord(
                        chunk_uid=str(uuid5(NAMESPACE_URL, stable_key)),
                        chunk_id=chunk_index,
                        text=piece,
                        source_path=relative_source,
                        file_name=path.name,
                        doc_type=doc_type,
                        page=segment.get("page"),
                        section=segment.get("section"),
                    )
                )

        return chunks

    def _load_pdf(self, path: Path) -> list[dict]:
        reader = PdfReader(str(path))
        segments: list[dict] = []
        for page_index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = self._clean_text(raw_text)
            if cleaned:
                segments.append({"text": cleaned, "page": page_index, "section": None})
        return segments

    def _load_docx(self, path: Path) -> list[dict]:
        document = Document(str(path))
        current_section: str | None = None
        segments: list[dict] = []

        for paragraph in document.paragraphs:
            raw_text = paragraph.text or ""
            cleaned = self._clean_text(raw_text)
            if not cleaned:
                continue

            style_name = paragraph.style.name if paragraph.style else ""
            if style_name.lower().startswith("heading"):
                current_section = cleaned
                continue

            segments.append(
                {"text": cleaned, "page": None, "section": current_section}
            )

        return segments

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        while start < len(text):
            end = min(start + size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

        return chunks


def save_chunks(chunks: list[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(input_path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunks.append(ChunkRecord(**json.loads(line)))
    return chunks
