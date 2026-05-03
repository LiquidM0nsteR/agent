from __future__ import annotations

import json
import os
import re
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage

Intent = Literal["rag", "web_search", "sc_analysis", "chat", "unknown"]
InputKind = Literal["text", "pdf", "h5ad", "markdown", "mixed", "unknown"]
NextNode = Literal["RAG", "WebSearch", "scAnalysis", "FinalNode"]
StreamEventCallback = Callable[[dict[str, Any]], None]

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "")
SC_OUTPUT_DIR = os.getenv("SC_OUTPUT_DIR", "./outputs/sc_analysis")
UPLOAD_WORKDIR = os.getenv("AGENT_UPLOAD_WORKDIR", "./outputs/uploads")
MAX_GRAPH_STEPS = int(os.getenv("AGENT_MAX_GRAPH_STEPS", "8"))
MAX_LLM_GENERATION_CALLS = int(os.getenv("AGENT_MAX_LLM_GENERATION_CALLS", "8"))
DEFAULT_RAG_CONFIDENCE_THRESHOLD = float(os.getenv("AGENT_RAG_CONFIDENCE_THRESHOLD", "0.58"))
DEFAULT_MEMORY_SCORE_THRESHOLD = float(os.getenv("AGENT_MEMORY_VECTOR_SCORE_THRESHOLD", "0.58"))
SHORT_ENTITY_MEMORY_SCORE_THRESHOLD = float(os.getenv("AGENT_MEMORY_SHORT_QUERY_SCORE_THRESHOLD", "0.78"))

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
H5AD_SUFFIXES = {".h5ad"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
CODE_SUFFIXES = {
    ".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".java",
    ".js", ".ts", ".tsx", ".jsx", ".vue", ".go", ".rs", ".sh", ".bash",
    ".zsh", ".sql", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".r", ".R",
}
RAG_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES | CODE_SUFFIXES
MEMORY_BIOINFO_KEYWORDS = {
    "bioinfo", "生物", "医学", "单细胞", "多组学", "转录组", "基因", "细胞", "蛋白",
    "scgpt", "geneformer", "genemamba", "scrna", "rna", "h5ad", "scanpy",
}
MEMORY_CODING_KEYWORDS = {
    "python", "javascript", "typescript", "代码", "函数", "接口", "bug", "报错",
    "redis", "mysql", "qdrant", "langgraph", "api", "frontend", "backend",
}
MEMORY_PREFERENCE_KEYWORDS = {"记住", "偏好", "以后", "默认", "我喜欢", "我希望", "总是"}
MEMORY_ONE_SHOT_INSTRUCTION_WORDS = {"不要", "别", "无需", "不需要"}
MEMORY_TASK_STATE_KEYWORDS = {"待办", "进度", "任务", "已经完成", "下一步", "todo", "blocker"}
MEMORY_STOPWORDS = {"the", "and", "for", "with", "from", "this", "that", "what", "how", "why", "什么", "如何", "为什么", "请问", "请", "一下", "介绍", "解释", "具体", "表现"}

_STREAM_EVENT_CALLBACK: ContextVar[StreamEventCallback | None] = ContextVar(
    "agent_stream_event_callback",
    default=None,
)


class UploadedFileInfo(TypedDict, total=False):
    original_path: str
    normalized_path: str
    suffix: str
    kind: InputKind
    converted: bool


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"


def safe_id(value: str | None, default: str) -> str:
    raw = str(value or "").strip()
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_"})
    return safe or default


def safe_filename(filename: str | None) -> str:
    raw = str(filename or "upload.bin").strip()
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})
    return safe or "upload.bin"


def safe_relpath(value: str) -> Path:
    path = Path(str(value or "").replace("\\", "/").lstrip("/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid path")
    return path


def directory_size_bytes(root: str | Path) -> int:
    base = Path(root)
    if not base.exists():
        return 0
    return sum(path.stat().st_size for path in base.rglob("*") if path.is_file())


def inspect_upload_batch(
    uploads: list[Any],
    *,
    allowed_suffixes: set[str] | tuple[str, ...],
    max_files: int,
    max_file_size_bytes: int,
    quota_root: str | Path | None = None,
    quota_bytes: int = 0,
) -> list[dict[str, Any]]:
    upload_list = list(uploads or [])
    if not upload_list:
        raise ValueError("No files uploaded")
    if len(upload_list) > max_files:
        raise ValueError(f"Too many files: {len(upload_list)} > {max_files}")

    allowed = {item.lower() for item in allowed_suffixes}
    inspected: list[dict[str, Any]] = []
    incoming_bytes = 0
    for upload in upload_list:
        safe_name = safe_filename(getattr(upload, "filename", ""))
        suffix = Path(safe_name).suffix.lower()
        if suffix not in allowed:
            raise ValueError(f"Unsupported file suffix for {safe_name}: {suffix or '<none>'}. Allowed: {', '.join(sorted(allowed))}")
        file_obj = getattr(upload, "file", None)
        if file_obj is None:
            raise ValueError(f"Upload file handle is missing: {safe_name}")
        current_pos = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        size_bytes = int(file_obj.tell())
        file_obj.seek(current_pos)
        if size_bytes <= 0:
            raise ValueError(f"Uploaded file is empty: {safe_name}")
        if size_bytes > max_file_size_bytes:
            raise ValueError(f"Uploaded file is too large: {safe_name} ({size_bytes} > {max_file_size_bytes} bytes)")
        incoming_bytes += size_bytes
        inspected.append({"safe_name": safe_name, "suffix": suffix, "size_bytes": size_bytes})

    if quota_root is not None and quota_bytes > 0:
        used_bytes = directory_size_bytes(quota_root)
        if used_bytes + incoming_bytes > quota_bytes:
            raise ValueError(f"Knowledge upload quota exceeded: current={used_bytes}, incoming={incoming_bytes}, quota={quota_bytes} bytes")
    return inspected


def truncate_text(text: str, limit: int = 96) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def safe_text(obj: Any, max_len: int = 6000) -> str:
    text = str(obj)
    return text if len(text) <= max_len else text[:max_len] + "\n...[truncated]"


def compact_memory_text(value: Any, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def memory_query_features(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    word_tokens = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}|[0-9]+(?:\.[0-9]+)?", normalized))
    word_tokens.update(item for item in re.findall(r"[\u4e00-\u9fff]{2,}", normalized) if len(item) <= 12)
    word_tokens = {item for item in word_tokens if item not in MEMORY_STOPWORDS and len(item) > 1}
    entities = {item.lower() for item in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{1,}", str(text or ""))}
    compact = re.sub(r"\s+", "", str(text or ""))
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))
    tags: set[str] = set()
    if any(keyword.lower() in normalized for keyword in MEMORY_BIOINFO_KEYWORDS):
        tags.add("bioinfo")
    if any(keyword.lower() in normalized for keyword in MEMORY_CODING_KEYWORDS):
        tags.add("coding")
    has_stable_preference = any(keyword.lower() in normalized for keyword in MEMORY_PREFERENCE_KEYWORDS)
    has_one_shot_preference = any(keyword.lower() in normalized for keyword in MEMORY_ONE_SHOT_INSTRUCTION_WORDS) and any(keyword in normalized for keyword in {"以后", "默认", "总是", "记住"})
    if has_stable_preference or has_one_shot_preference:
        tags.add("user_preferences")
    if any(keyword.lower() in normalized for keyword in MEMORY_TASK_STATE_KEYWORDS):
        tags.add("task_state")
    for entity in sorted(entities)[:6]:
        tags.add(f"entity:{entity}")
    if not tags:
        tags.add("general")
    return {
        "normalized": normalized,
        "tokens": word_tokens,
        "entities": entities,
        "tags": tags,
        "is_short_entity": bool(compact)
        and (bool(entities) or chinese_char_count <= 4)
        and ((len(compact) <= 12 and len(entities) <= 2) or (len(word_tokens) <= 2 and len(compact) <= 16)),
    }


def summarize_memory_turn(user_input: str, final_answer: str, limit: int = 420) -> tuple[str, list[str], str]:
    cleaned = re.sub(r"```.*?```", " ", str(final_answer or ""), flags=re.S)
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"[-*]\s+", "", cleaned)
    parts = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s*", cleaned) if part.strip()]
    deduped_parts: list[str] = []
    seen_parts: set[str] = set()
    for part in parts:
        if part in seen_parts:
            continue
        seen_parts.add(part)
        deduped_parts.append(part)
        if len(deduped_parts) >= 3:
            break
    answer_summary = " ".join(deduped_parts)
    answer_summary = compact_memory_text(answer_summary or cleaned, limit)
    user_summary = compact_memory_text(user_input, 220)
    features = memory_query_features(f"{user_summary} {answer_summary}")
    tags = sorted(features["tags"])
    memory_type = "user_preference" if "user_preferences" in tags else "task_state" if "task_state" in tags else "conversation_summary"
    return f"User asked: {user_summary}. Answer summary: {answer_summary}".strip(), tags, memory_type


def sanitize_action_queries(user_input: str, queries: list[Any], max_count: int | None = None) -> list[str]:
    original = str(user_input or "").strip()
    if not original:
        return []
    features = memory_query_features(original)
    if features["is_short_entity"]:
        return [original]
    cleaned: list[str] = [original]
    for item in queries:
        query = compact_memory_text(item, 160)
        if not query or query == original or query in cleaned:
            continue
        cleaned.append(query)
        if max_count and len(cleaned) >= max_count:
            break
    return cleaned


class _StreamEventContext:
    def __init__(self, callback: StreamEventCallback | None) -> None:
        self.callback = callback
        self.token: Any = None

    def __enter__(self) -> None:
        self.token = _STREAM_EVENT_CALLBACK.set(self.callback)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _STREAM_EVENT_CALLBACK.reset(self.token)


def stream_event_callback(callback: StreamEventCallback | None) -> _StreamEventContext:
    return _StreamEventContext(callback)


def emit(event: dict[str, Any]) -> None:
    callback = _STREAM_EVENT_CALLBACK.get()
    if callback is not None:
        callback(event)


def last_user_text(state: dict[str, Any]) -> str:
    if state.get("user_input"):
        return str(state["user_input"])
    for msg in reversed(list(state.get("messages") or [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _guess_code_language(path: str | Path) -> str:
    suffix = Path(path).suffix
    return {
        ".py": "python", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c",
        ".h": "cpp", ".hpp": "cpp", ".java": "java", ".js": "javascript",
        ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".vue": "vue",
        ".go": "go", ".rs": "rust", ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
        ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".r": "r", ".R": "r",
    }.get(suffix, "")


def get_file_kind(path: str | Path) -> InputKind:
    suffix = Path(path).suffix.lower()
    if suffix in H5AD_SUFFIXES:
        return "h5ad"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".txt":
        return "text"
    if suffix in {item.lower() for item in CODE_SUFFIXES}:
        return "markdown"
    return "unknown"


def convert_code_file_to_markdown(file_path: str | Path, output_dir: str | Path = UPLOAD_WORKDIR) -> str:
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"上传文件不存在: {src}")
    dst_dir = Path(output_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{src.stem}{src.suffix.replace('.', '_')}.md"
    dst.write_text(
        f"# Source file: {src.name}\n\n"
        f"- Original path: `{src}`\n"
        f"- Converted from: `{src.suffix}`\n\n"
        f"```{_guess_code_language(src)}\n{src.read_text(encoding='utf-8', errors='ignore')}\n```\n",
        encoding="utf-8",
    )
    return str(dst)


def normalize_uploaded_files(file_paths: Optional[list[str]] = None, upload_workdir: str = UPLOAD_WORKDIR) -> list[UploadedFileInfo]:
    normalized: list[UploadedFileInfo] = []
    for raw_path in file_paths or []:
        path = Path(raw_path).expanduser()
        suffix = path.suffix
        suffix_lower = suffix.lower()
        if suffix_lower in {item.lower() for item in CODE_SUFFIXES}:
            normalized.append({
                "original_path": str(path),
                "normalized_path": convert_code_file_to_markdown(path, upload_workdir),
                "suffix": suffix,
                "kind": "markdown",
                "converted": True,
            })
        else:
            normalized.append({
                "original_path": str(path),
                "normalized_path": str(path),
                "suffix": suffix,
                "kind": get_file_kind(path),
                "converted": False,
            })
    return normalized


def extract_h5ad_paths_from_text(user_input: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"(?P<path>(?:~|/|\.{1,2}/|[A-Za-z0-9_.-]+/)?[^\s,，;；:：\"'<>]+\.h5ad)", str(user_input or ""), flags=re.IGNORECASE):
        raw_path = match.group("path").strip().strip("`'\"()[]{}<>，,。；;：:")
        if raw_path and raw_path not in paths:
            paths.append(raw_path)
    return paths


def infer_input_kind_by_files(normalized_files: list[UploadedFileInfo], user_input: str = "") -> InputKind:
    if not normalized_files:
        return "text" if user_input.strip() else "unknown"
    kinds = {item.get("kind", "unknown") for item in normalized_files}
    if "h5ad" in kinds:
        return "h5ad"
    known = kinds - {"unknown"}
    if len(known) > 1:
        return "mixed"
    return next(iter(known)) if known else "unknown"  # type: ignore[return-value]


def get_h5ad_files(normalized_files: list[UploadedFileInfo]) -> list[str]:
    return [item["normalized_path"] for item in normalized_files if item.get("kind") == "h5ad"]


def get_rag_files(normalized_files: list[UploadedFileInfo]) -> list[str]:
    return [item["normalized_path"] for item in normalized_files if item.get("kind") in {"pdf", "markdown", "text"}]


def is_rag_upload_candidate(saved_file: dict[str, Any]) -> bool:
    return str(saved_file.get("kind") or "") in {"pdf", "text", "markdown", "file"}


def has_session_rag_sources(uploads_root: Path) -> bool:
    return uploads_root.exists() and any(
        path.is_file() and path.suffix.lower() in {item.lower() for item in RAG_SUFFIXES}
        for path in uploads_root.rglob("*")
    )


def build_effective_user_text(raw_text: str, saved_files: list[dict[str, Any]]) -> str:
    text = raw_text.strip()
    if text:
        return text
    kinds = {str(item.get("kind") or "") for item in saved_files}
    if "h5ad" in kinds:
        return "请分析已上传的 h5ad 文件。"
    if kinds & {"pdf", "text", "markdown", "file"}:
        return "请处理已上传的文件。"
    return "请处理已上传的附件。"


def normalize_rag_files_for_base(files: list[str], knowledge_base_path: str) -> list[str]:
    base = Path(knowledge_base_path or KNOWLEDGE_BASE_PATH).expanduser().resolve()
    results: list[str] = []
    for item in files:
        path = Path(str(item)).expanduser()
        resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
        results.append(resolved.relative_to(base).as_posix() if base in resolved.parents or resolved == base else str(resolved))
    return results


def current_turn_id(state: dict[str, Any]) -> str:
    return str(state.get("current_turn_id") or "")


def is_current_turn_item(state: dict[str, Any], item: dict[str, Any]) -> bool:
    turn_id = current_turn_id(state)
    if not turn_id:
        return True
    metadata = item.get("metadata")
    return str((metadata if isinstance(metadata, dict) else item).get("turn_id") or "") == turn_id


def current_turn_observations(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in list(state.get("observations") or []) if isinstance(item, dict) and is_current_turn_item(state, item)]


def current_turn_step_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in list(state.get("step_records") or []) if isinstance(item, dict) and is_current_turn_item(state, item)]


def current_turn_llm_traces(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in list(state.get("llm_traces") or []) if isinstance(item, dict) and is_current_turn_item(state, item)]


def current_turn_steps(state: dict[str, Any]) -> list[str]:
    records = current_turn_step_records(state)
    if records:
        return [str(item.get("node") or "") for item in records if item.get("node")]
    return [] if current_turn_id(state) else list(state.get("steps") or [])


def current_turn_tool_results(state: dict[str, Any]) -> dict[str, Any]:
    current = state.get("current_tool_results")
    if isinstance(current, dict) and str(state.get("current_tool_results_turn_id") or "") == current_turn_id(state):
        return dict(current)
    return {} if current_turn_id(state) else dict(state.get("tool_results") or {})


def tool_result_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("answer", "local_answer", "message", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return safe_text(result)


def rag_confidence_from_result(result: Any) -> float:
    if not isinstance(result, dict):
        return 0.0
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and metrics.get("confidence") is not None:
        return max(0.0, min(1.0, float(metrics["confidence"])))
    meta = result.get("meta")
    if isinstance(meta, dict) and meta.get("confidence") is not None:
        return max(0.0, min(1.0, float(meta["confidence"])))
    rerank_scores: list[float] = []
    dense_scores: list[float] = []
    sparse_scores: list[float] = []
    fused_scores: list[float] = []
    for key in ("references", "chunks"):
        for item in result.get(key) or []:
            if isinstance(item, dict):
                if item.get("reranker_score") is not None:
                    rerank_scores.append(float(item["reranker_score"]))
                if item.get("dense_score") is not None:
                    dense_scores.append(float(item["dense_score"]))
                if item.get("vector_score") is not None:
                    dense_scores.append(float(item["vector_score"]))
                if item.get("sparse_score") is not None:
                    sparse_scores.append(float(item["sparse_score"]))
                if item.get("bm25_score") is not None:
                    sparse_scores.append(float(item["bm25_score"]))
                if item.get("rrf_score") is not None:
                    fused_scores.append(float(item["rrf_score"]))
                if item.get("score") is not None:
                    fused_scores.append(float(item["score"]))
    if rerank_scores:
        return max(0.0, min(1.0, max(rerank_scores)))
    if dense_scores:
        return max(0.0, min(1.0, max(dense_scores)))
    if sparse_scores:
        return max(0.0, min(1.0, max(sparse_scores) / 20.0))
    if fused_scores:
        return max(0.0, min(1.0, max(fused_scores) * 30.0))
    return 0.0


def standardize_tool_result(node: str, tool_name: str, result: Any, ok: bool, content: str, elapsed: float, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result) if isinstance(result, dict) else {}
    answer = str(payload.get("answer") or payload.get("local_answer") or payload.get("message") or content or "").strip()
    metrics = dict(payload.get("metrics") or {})
    metrics.setdefault("tool_ms", elapsed)
    meta = dict(payload.get("meta") or {})
    meta.update(metadata)
    payload.update({
        "status": str(payload.get("status") or ("ok" if ok else "error")),
        "tool_name": str(payload.get("tool_name") or tool_name),
        "answer": answer,
        "local_answer": str(payload.get("local_answer") or answer),
        "message": str(payload.get("message") or answer),
        "artifacts": list(payload.get("artifacts") or []),
        "references": list(payload.get("references") or []),
        "metrics": metrics,
        "meta": meta,
    })
    return payload


def build_step_record(node: str, elapsed: float, tool_name: str = "", detail: str = "", turn_id: str = "") -> dict[str, Any]:
    record: dict[str, Any] = {"node": node, "elapsed_ms": elapsed}
    if tool_name:
        record["tool_name"] = tool_name
    if detail:
        record["detail"] = detail
    if turn_id:
        record["turn_id"] = turn_id
    return record


def tool_node_result(
    state: dict[str, Any],
    *,
    node: str,
    tool_key: str,
    tool_name: str,
    ok: bool,
    result: Any,
    elapsed: float,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turn_id = current_turn_id(state)
    meta = dict(metadata or {})
    if turn_id:
        meta["turn_id"] = turn_id
    observation = {"node": node, "ok": ok, "content": content, "metadata": meta}
    emit({"node": node, "status": "end", "ok": ok})
    standardized = standardize_tool_result(node, tool_name, result, ok, content, elapsed, meta)
    current_results = dict(state.get("current_tool_results") or {})
    current_results[tool_key] = standardized
    return {
        "observations": [observation],
        "tool_results": {**dict(state.get("tool_results") or {}), tool_key: standardized},
        "current_tool_results": current_results,
        "current_tool_results_turn_id": turn_id,
        "steps": [node],
        "step_records": [build_step_record(node, elapsed, tool_name=tool_name, turn_id=turn_id)],
    }


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON output must be an object.")
    return data
