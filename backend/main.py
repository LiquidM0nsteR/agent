from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import AgentInput, UploadedAsset, stream_agent_response
from .memory import MemoryManager
from .tools.llm import preload_local_qwen_client
from .tools.rag.config import get_config
from .tools.rag.knowledge_base import KnowledgeBaseBuilder, save_chunks
from .tools.rag.retrieval import (
    BgeM3Embedder,
    BM25Index,
    QdrantVectorStore,
    preload_bge_m3_embedder,
)


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_ROOT = BASE_DIR / "data"
USER_SESSIONS_ROOT = DATA_ROOT / "users"
LOCAL_KNOWLEDGE_ROOT = DATA_ROOT / "local_knowledge"

USER_SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_KNOWLEDGE_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
H5AD_EXTENSIONS = {".h5ad"}
WORKSPACE_SETTINGS_DEFAULTS: dict[str, Any] = {
    "temperature": 0.2,
    "max_new_tokens": 512,
    "short_term_max_messages": 12,
    "short_term_summary_threshold": 8,
    "long_term_top_k": 3,
    "local_source_min_score": 0.35,
    "web_source_min_score": 1.5,
    "enable_profile_memory": True,
    "enable_semantic_memory": True,
    "search_provider": "serper",
    "search_prefers_official_sources": True,
}
TOOL_STATUS: dict[str, dict[str, Any]] = {
    "agent": {"state": "idle", "detail": "", "updated_at": ""},
    "local_llm": {"state": "idle", "detail": "", "updated_at": ""},
    "single_cell": {"state": "idle", "detail": "", "updated_at": ""},
    "retrieval_index": {"state": "idle", "detail": "", "updated_at": ""},
}
_MEMORY_MANAGER = MemoryManager(get_config())
logger = logging.getLogger(__name__)


def _safe_filename(filename: str | None) -> str:
    raw_name = filename or "upload.bin"
    sanitized = "".join(
        char for char in raw_name if char.isalnum() or char in {".", "-", "_"}
    )
    return sanitized or "upload.bin"


def _safe_id(value: str | None, default: str) -> str:
    raw = (value or "").strip()
    sanitized = "".join(char for char in raw if char.isalnum() or char in {"-", "_"})
    return sanitized or default


def _safe_relpath(value: str) -> Path:
    cleaned = value.replace("\\", "/").strip().lstrip("/")
    candidate = Path(cleaned)
    if not cleaned or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Invalid relative path")
    return candidate


def _get_session_root(user_id: str, session_id: str) -> Path:
    return USER_SESSIONS_ROOT / user_id / "sessions" / session_id


def _session_meta_path(user_id: str, session_id: str) -> Path:
    return _get_session_root(user_id, session_id) / "session.json"


def _memory_path(user_id: str, session_id: str) -> Path:
    return _get_session_root(user_id, session_id) / "memory" / "short_term.json"


def _chat_history_path(user_id: str, session_id: str) -> Path:
    return _get_session_root(user_id, session_id) / "history.jsonl"


def _user_settings_path(user_id: str) -> Path:
    return USER_SESSIONS_ROOT / user_id / "workspace_settings.json"


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _set_tool_status(tool: str, state: str, detail: str = "") -> None:
    TOOL_STATUS[tool] = {
        "state": state,
        "detail": detail,
        "updated_at": _now_iso(),
    }


def _truncate_text(value: str, limit: int = 90) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _load_memory_messages(user_id: str, session_id: str) -> list[dict[str, Any]]:
    try:
        short_term = _MEMORY_MANAGER.read_short_term(user_id, session_id)
    except RuntimeError:
        return []
    messages = short_term.to_dict().get("messages") or []
    return [item for item in messages if item.get("role") in {"user", "assistant"}]


def _append_chat_history_messages(
    user_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
) -> None:
    if not messages:
        return
    path = _chat_history_path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in messages:
            record = {
                "role": item.get("role", "assistant"),
                "content": str(item.get("content") or ""),
                "created_at": item.get("created_at") or _now_iso(),
                "metadata": item.get("metadata") or {},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_chat_history_messages(user_id: str, session_id: str) -> list[dict[str, Any]]:
    path = _chat_history_path(user_id, session_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("role") not in {"user", "assistant"}:
                continue
            records.append(
                {
                    "role": payload.get("role", "assistant"),
                    "content": str(payload.get("content") or ""),
                    "created_at": payload.get("created_at") or "",
                    "metadata": payload.get("metadata") or {},
                }
            )
    return records


def _build_history_payload(user_id: str, session_id: str) -> list[dict[str, Any]]:
    persisted = _load_chat_history_messages(user_id, session_id)
    if persisted:
        return persisted

    history: list[dict[str, Any]] = []
    for item in _load_memory_messages(user_id, session_id):
        history.append(
            {
                "role": item.get("role", "assistant"),
                "content": str(item.get("content") or ""),
                "created_at": item.get("created_at") or "",
                "metadata": item.get("metadata") or {},
            }
        )
    return history


def _load_session_meta(user_id: str, session_id: str) -> dict[str, Any] | None:
    path = _session_meta_path(user_id, session_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_session_meta(user_id: str, session_id: str, payload: dict[str, Any]) -> None:
    path = _session_meta_path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_session_title(user_text: str, saved_files: list[dict[str, Any]]) -> str:
    if user_text.strip():
        return _truncate_text(user_text.strip(), 42)
    if saved_files:
        return f"New chat with {saved_files[0]['original_name']}"
    return "New chat"


def _build_session_preview(
    user_text: str,
    agent_payload: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    tool_result = agent_payload.get("tool_result") or {}
    preview = (
        tool_result.get("answer")
        or tool_result.get("local_answer")
        or tool_result.get("message")
        or ""
    )
    if preview:
        return _truncate_text(str(preview), 90)
    if history:
        return _truncate_text(str(history[-1].get("content") or ""), 90)
    return _truncate_text(user_text, 90) or "No messages yet"


def _build_route_trace(agent_payload: dict[str, Any]) -> dict[str, Any]:
    decision = agent_payload.get("decision") or {}
    graph_execution = agent_payload.get("graph_execution") or {}
    selected_tools = decision.get("selected_tools") or []
    execution_steps = decision.get("execution_steps") or []
    llm_traces = decision.get("llm_traces") or []
    normalized_steps: list[dict[str, Any]] = []
    for item in execution_steps:
        if not isinstance(item, dict):
            continue
        normalized_steps.append(
            {
                "step_id": str(item.get("step_id") or ""),
                "description": str(item.get("description") or ""),
                "status": str(item.get("status") or ""),
                "tool_name": str(item.get("tool_name") or ""),
            }
        )
    normalized_llm_traces: list[dict[str, Any]] = []
    for item in llm_traces:
        if not isinstance(item, dict):
            continue
        normalized_llm_traces.append(
            {
                "label": str(item.get("label") or ""),
                "response": str(
                    item.get("response") or item.get("response_preview") or ""
                ),
                "prompt_preview": str(item.get("prompt_preview") or ""),
                "model_path": str(item.get("model_path") or ""),
                "elapsed_ms": item.get("elapsed_ms"),
            }
        )
    return {
        "intent": str(decision.get("intent") or ""),
        "reason": str(decision.get("reason") or ""),
        "dispatched_node": str(graph_execution.get("dispatched_node") or ""),
        "selected_tools": [str(item) for item in selected_tools],
        "execution_steps": normalized_steps,
        "llm_traces": normalized_llm_traces,
    }


def _upsert_session_meta(
    *,
    user_id: str,
    session_id: str,
    user_text: str,
    saved_files: list[dict[str, Any]],
    agent_payload: dict[str, Any],
) -> dict[str, Any]:
    existing = _load_session_meta(user_id, session_id) or {}
    history = _build_history_payload(user_id, session_id)
    created_at = existing.get("created_at") or _now_iso()
    title = existing.get("title") or _build_session_title(user_text, saved_files)
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "title": title,
        "preview": _build_session_preview(user_text, agent_payload, history),
        "created_at": created_at,
        "updated_at": _now_iso(),
        "message_count": len(history),
    }
    _write_session_meta(user_id, session_id, payload)
    return payload


def _list_user_sessions(user_id: str) -> list[dict[str, Any]]:
    user_root = USER_SESSIONS_ROOT / user_id / "sessions"
    if not user_root.exists():
        return []

    sessions: list[dict[str, Any]] = []
    for session_dir in user_root.iterdir():
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        meta = _load_session_meta(user_id, session_id) or {
            "session_id": session_id,
            "user_id": user_id,
            "title": session_id,
            "preview": "",
            "created_at": "",
            "updated_at": "",
            "message_count": len(_build_history_payload(user_id, session_id)),
        }
        sessions.append(meta)

    sessions.sort(
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        reverse=True,
    )
    return sessions


def _load_workspace_settings(user_id: str) -> dict[str, Any]:
    path = _user_settings_path(user_id)
    if not path.exists():
        return dict(WORKSPACE_SETTINGS_DEFAULTS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    merged = dict(WORKSPACE_SETTINGS_DEFAULTS)
    if isinstance(payload, dict):
        merged.update(payload)
    return merged


def _save_workspace_settings(user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    current = _load_workspace_settings(user_id)
    current.update(updates)
    path = _user_settings_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def _list_knowledge_files() -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in LOCAL_KNOWLEDGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(LOCAL_KNOWLEDGE_ROOT).as_posix()
        stat = path.stat()
        files.append(
            {
                "path": rel,
                "name": path.name,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    files.sort(key=lambda item: item["updated_at"], reverse=True)
    return files


def _rebuild_local_knowledge_index() -> dict[str, Any]:
    started_at = time.perf_counter()
    config = get_config()
    builder = KnowledgeBaseBuilder(config)
    chunks, errors = builder.build()
    save_chunks(chunks, config.chunk_manifest_path)

    bm25_index = BM25Index(chunks)
    bm25_index.save(config.bm25_index_path)

    vector_count = 0
    if chunks:
        embedder = BgeM3Embedder(config)
        try:
            vectors = embedder.encode([chunk.text for chunk in chunks], batch_size=8)
            vector_store = QdrantVectorStore(config)
            vector_store.replace_collection(chunks, vectors)
            vector_count = len(vectors)
        finally:
            embedder.close()
    else:
        # No chunks means knowledge base is empty; clear stale vector directory if present.
        if config.qdrant_path.exists():
            shutil.rmtree(config.qdrant_path)

    source_docs = len(builder.scan_documents())
    result = {
        "status": "ok",
        "source_documents": source_docs,
        "chunk_count": len(chunks),
        "vector_count": vector_count,
        "errors": errors,
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    logger.info(
        "[knowledge.rebuild] source_documents=%s chunks=%s vectors=%s elapsed_ms=%.2f",
        result["source_documents"],
        result["chunk_count"],
        result["vector_count"],
        result["elapsed_ms"],
    )
    return result


def _detect_file_kind(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in H5AD_EXTENSIONS:
        return "h5ad"
    if suffix in PDF_EXTENSIONS or (upload.content_type or "") == "application/pdf":
        return "pdf"
    if suffix in IMAGE_EXTENSIONS or (upload.content_type or "").startswith("image/"):
        return "image"
    return "other"


def _save_upload(upload: UploadFile, uploads_dir: Path) -> dict[str, Any]:
    file_kind = _detect_file_kind(upload)
    kind_dir = uploads_dir / file_kind
    kind_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(upload.filename)
    destination = kind_dir / f"{uuid4().hex}_{safe_name}"

    with destination.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)

    return {
        "original_name": upload.filename or safe_name,
        "saved_name": destination.name,
        "kind": file_kind,
        "content_type": upload.content_type or "application/octet-stream",
        "size_bytes": destination.stat().st_size,
        "path": str(destination.relative_to(BASE_DIR)),
    }


class LoginRequest(BaseModel):
    user_id: str


class WorkspaceSettingsUpdateRequest(BaseModel):
    settings: dict[str, Any]


class MemoryClearRequest(BaseModel):
    scope: str = "session"
    session_id: str = ""


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    _set_tool_status("local_llm", "loading", "Preloading local Qwen model into memory.")
    _set_tool_status("retrieval_index", "loading", "Preloading retrieval embedder into memory.")
    try:
        config = get_config()
        preload_meta = await asyncio.to_thread(preload_local_qwen_client)
        embedder_meta = await asyncio.to_thread(preload_bge_m3_embedder, config)
        detail = (
            f"Preloaded {preload_meta['model_path']} "
            f"on {preload_meta['device']}."
        )
        _set_tool_status("local_llm", "ready", detail)
        _set_tool_status(
            "retrieval_index",
            "ready",
            (
                f"Preloaded {embedder_meta['model_path']} "
                f"on {embedder_meta['device']}."
            ),
        )
    except Exception as exc:
        _set_tool_status("local_llm", "error", f"Preload failed: {exc}")
        _set_tool_status("retrieval_index", "error", f"Preload failed: {exc}")
    yield


app = FastAPI(title="Agent Entry API", version="0.1.0", lifespan=app_lifespan)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid4().hex[:12]
    started_at = time.perf_counter()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "[http] request_id=%s method=%s path=%s status=500 elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "[http] request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.middleware("http")
async def frontend_no_cache_middleware(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/sessions/{session_id}/artifacts/{artifact_path:path}")
async def get_session_artifact(session_id: str, artifact_path: str) -> FileResponse:
    raise HTTPException(
        status_code=410,
        detail="Use /api/users/{user_id}/sessions/{session_id}/artifacts/... instead.",
    )


@app.get("/api/users/{user_id}/sessions/{session_id}/artifacts/{artifact_path:path}")
async def get_user_session_artifact(
    user_id: str,
    session_id: str,
    artifact_path: str,
) -> FileResponse:
    session_root = _get_session_root(user_id, session_id).resolve()
    target = (session_root / artifact_path).resolve()
    if session_root not in target.parents and target != session_root:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.post("/api/auth/login")
async def login(payload: LoginRequest) -> dict[str, Any]:
    safe_user_id = _safe_id(payload.user_id, "anonymous")
    user_root = USER_SESSIONS_ROOT / safe_user_id / "sessions"
    user_root.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "user_id": safe_user_id,
        "sessions": _list_user_sessions(safe_user_id),
    }


@app.get("/api/users/{user_id}/sessions")
async def list_user_sessions(user_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    return {
        "user_id": safe_user_id,
        "sessions": _list_user_sessions(safe_user_id),
    }


@app.get("/api/users/{user_id}/workspace/settings")
async def get_workspace_settings(user_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    return {
        "user_id": safe_user_id,
        "settings": _load_workspace_settings(safe_user_id),
    }


@app.put("/api/users/{user_id}/workspace/settings")
async def update_workspace_settings(
    user_id: str,
    payload: WorkspaceSettingsUpdateRequest,
) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    settings = _save_workspace_settings(safe_user_id, payload.settings or {})
    return {
        "status": "ok",
        "user_id": safe_user_id,
        "settings": settings,
    }


@app.post("/api/users/{user_id}/workspace/memory/clear")
async def clear_workspace_memory(
    user_id: str,
    payload: MemoryClearRequest,
) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    scope = (payload.scope or "session").strip().lower()
    if scope not in {"session", "all"}:
        raise HTTPException(status_code=400, detail="scope must be 'session' or 'all'")

    cleared = 0
    if scope == "all":
        cleared = _MEMORY_MANAGER.clear_short_term(safe_user_id)
    else:
        safe_session_id = _safe_id(payload.session_id, "")
        if not safe_session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required when scope=session",
            )
        cleared = _MEMORY_MANAGER.clear_short_term(safe_user_id, safe_session_id)

    return {
        "status": "ok",
        "user_id": safe_user_id,
        "scope": scope,
        "cleared_files": cleared,
    }


@app.get("/api/workspace/knowledge/files")
async def list_knowledge_files() -> dict[str, Any]:
    return {
        "root": str(LOCAL_KNOWLEDGE_ROOT),
        "files": _list_knowledge_files(),
    }


@app.post("/api/workspace/knowledge/files")
async def upload_knowledge_files(
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    uploaded: list[dict[str, Any]] = []
    for upload in files:
        safe_name = _safe_filename(upload.filename)
        destination = LOCAL_KNOWLEDGE_ROOT / f"{uuid4().hex}_{safe_name}"
        with destination.open("wb") as output_file:
            shutil.copyfileobj(upload.file, output_file)
        uploaded.append(
            {
                "name": safe_name,
                "path": destination.relative_to(LOCAL_KNOWLEDGE_ROOT).as_posix(),
                "size_bytes": destination.stat().st_size,
            }
        )
    return {
        "status": "ok",
        "uploaded": uploaded,
        "files": _list_knowledge_files(),
    }


@app.delete("/api/workspace/knowledge/files")
async def delete_knowledge_file(path: str = Query(...)) -> dict[str, Any]:
    try:
        relpath = _safe_relpath(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = (LOCAL_KNOWLEDGE_ROOT / relpath).resolve()
    if LOCAL_KNOWLEDGE_ROOT.resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {
        "status": "ok",
        "deleted": relpath.as_posix(),
        "files": _list_knowledge_files(),
    }


@app.delete("/api/workspace/knowledge/files/{file_path:path}")
async def delete_knowledge_file_by_path(file_path: str) -> dict[str, Any]:
    try:
        relpath = _safe_relpath(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = (LOCAL_KNOWLEDGE_ROOT / relpath).resolve()
    if LOCAL_KNOWLEDGE_ROOT.resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {
        "status": "ok",
        "deleted": relpath.as_posix(),
        "files": _list_knowledge_files(),
    }


@app.post("/api/workspace/knowledge/rebuild-index")
async def rebuild_knowledge_index() -> dict[str, Any]:
    _set_tool_status("retrieval_index", "running", "Rebuilding local knowledge index")
    try:
        result = await asyncio.to_thread(_rebuild_local_knowledge_index)
        _set_tool_status(
            "retrieval_index",
            "completed",
            f"chunks={result.get('chunk_count', 0)}, vectors={result.get('vector_count', 0)}",
        )
        return result
    except Exception as exc:
        _set_tool_status("retrieval_index", "error", str(exc))
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {exc}") from exc


@app.get("/api/workspace/tool-status")
async def get_tool_status() -> dict[str, Any]:
    return {
        "status": "ok",
        "tools": TOOL_STATUS,
    }


@app.get("/api/users/{user_id}/sessions/{session_id}")
async def get_user_session(user_id: str, session_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "")
    if not safe_session_id:
        raise HTTPException(status_code=400, detail="Invalid session id")

    session_root = _get_session_root(safe_user_id, safe_session_id)
    if not session_root.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    meta = _load_session_meta(safe_user_id, safe_session_id) or {
        "session_id": safe_session_id,
        "user_id": safe_user_id,
        "title": safe_session_id,
        "preview": "",
        "created_at": "",
        "updated_at": "",
        "message_count": 0,
    }
    history = _build_history_payload(safe_user_id, safe_session_id)
    meta["message_count"] = len(history)
    return {
        "user_id": safe_user_id,
        "session": meta,
        "history": history,
    }


@app.delete("/api/users/{user_id}/sessions/{session_id}")
async def delete_user_session(user_id: str, session_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "")
    if not safe_session_id:
        raise HTTPException(status_code=400, detail="Invalid session id")

    session_root = _get_session_root(safe_user_id, safe_session_id).resolve()
    expected_root = (USER_SESSIONS_ROOT / safe_user_id / "sessions").resolve()
    if expected_root not in session_root.parents:
        raise HTTPException(status_code=400, detail="Invalid session path")
    if not session_root.exists() or not session_root.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")

    shutil.rmtree(session_root)
    return {
        "status": "ok",
        "user_id": safe_user_id,
        "session_id": safe_session_id,
    }


@app.post("/api/agent/submit")
async def submit_agent_input(
    request: Request,
    user_id: str = Form(default="anonymous"),
    session_id: str = Form(default=""),
    text: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
    images: list[UploadFile] | None = File(default=None),
    h5ad_files: list[UploadFile] | None = File(default=None),
) -> StreamingResponse:
    request_id = str(getattr(request.state, "request_id", "") or uuid4().hex[:12])
    request_started_at = time.perf_counter()
    attachments = [*(files or []), *(images or []), *(h5ad_files or [])]

    if not text.strip() and not attachments:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: text, image, PDF, h5ad, or file.",
        )

    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "") or _new_session_id()
    session_dir = _get_session_root(safe_user_id, safe_session_id)
    uploads_dir = session_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_files = [_save_upload(upload, uploads_dir) for upload in attachments]
    saved_images = [item for item in saved_files if item["kind"] == "image"]
    saved_pdfs = [item for item in saved_files if item["kind"] == "pdf"]
    saved_h5ad = [item for item in saved_files if item["kind"] == "h5ad"]
    saved_other = [item for item in saved_files if item["kind"] == "other"]
    workspace_settings = _load_workspace_settings(safe_user_id)

    agent_input = AgentInput(
        user_id=safe_user_id,
        session_id=safe_session_id,
        user_text=text,
        attachments=[
            UploadedAsset(
                name=item["original_name"],
                kind=item["kind"],
                content_type=item["content_type"],
                size_bytes=item["size_bytes"],
                path=item["path"],
            )
            for item in saved_files
        ],
        workspace_settings=workspace_settings,
    )
    logger.info(
        "[agent.request] request_id=%s user=%s session=%s text_len=%s files=%s",
        request_id,
        safe_user_id,
        safe_session_id,
        len(text or ""),
        len(saved_files),
    )

    _set_tool_status("agent", "running", "Agent routing and tool execution in progress")
    _set_tool_status("local_llm", "running", "Local model inference in progress")
    _set_tool_status(
        "single_cell",
        "running" if saved_h5ad else "idle",
        "Single-cell pipeline queued" if saved_h5ad else "",
    )
    async def _event_stream():
        try:
            yield _sse_event(
                "accepted",
                {
                    "request_id": request_id,
                    "user_id": safe_user_id,
                    "session_id": safe_session_id,
                    "text": text,
                },
            )
            async for item in stream_agent_response(agent_input):
                event_type = str(item.get("type") or "message")
                event_data = dict(item.get("data") or {})

                if event_type == "final":
                    agent_response = event_data
                    dispatched = (
                        agent_response.get("graph_execution", {}).get("dispatched_node")
                        or agent_response.get("decision", {}).get("intent")
                        or "unknown"
                    )
                    _set_tool_status("agent", "completed", f"Last route: {dispatched}")

                    agent_tool_result = agent_response.get("tool_result") or agent_response.get(
                        "decision", {}
                    ).get("tool_result", {})
                    request_elapsed_ms = round(
                        (time.perf_counter() - request_started_at) * 1000,
                        2,
                    )
                    agent_observability = dict(agent_response.get("observability") or {})
                    agent_observability.update(
                        {
                            "request_id": request_id,
                            "request_ms": request_elapsed_ms,
                        }
                    )
                    agent_response["observability"] = agent_observability
                    assistant_text = (
                        agent_tool_result.get("answer")
                        or agent_tool_result.get("local_answer")
                        or agent_tool_result.get("message")
                        or ""
                    )
                    _append_chat_history_messages(
                        safe_user_id,
                        safe_session_id,
                        [
                            {
                                "role": "user",
                                "content": text,
                                "metadata": {"file_count": len(saved_files)},
                            },
                            {
                                "role": "assistant",
                                "content": assistant_text,
                                "metadata": {
                                    "intent": agent_response.get("decision", {}).get(
                                        "intent", ""
                                    ),
                                    "dispatched_node": agent_response.get(
                                        "graph_execution", {}
                                    ).get("dispatched_node", ""),
                                    "route_trace": _build_route_trace(agent_response),
                                },
                            },
                        ],
                    )
                    session_payload = _upsert_session_meta(
                        user_id=safe_user_id,
                        session_id=safe_session_id,
                        user_text=text,
                        saved_files=saved_files,
                        agent_payload=agent_response,
                    )
                    final_payload = {
                        "user_id": safe_user_id,
                        "session_id": safe_session_id,
                        "session": session_payload,
                        "text": text,
                        "files": saved_files,
                        "images": saved_images,
                        "pdf_files": saved_pdfs,
                        "h5ad_files": saved_h5ad,
                        "other_files": saved_other,
                        "tool_result": agent_tool_result,
                        "agent": agent_response,
                        "observability": agent_observability,
                    }
                    logger.info(
                        "[agent.request] request_id=%s user=%s session=%s status=completed dispatched=%s request_ms=%.2f",
                        request_id,
                        safe_user_id,
                        safe_session_id,
                        dispatched,
                        request_elapsed_ms,
                    )
                    yield _sse_event("final", final_payload)
                    continue

                if event_type == "answer_start":
                    _set_tool_status("local_llm", "running", "Local model inference in progress")
                yield _sse_event(event_type, event_data)
        except Exception as exc:
            _set_tool_status("agent", "error", "Agent execution failed")
            logger.exception(
                "[agent.request] request_id=%s user=%s session=%s status=error detail=%s",
                request_id,
                safe_user_id,
                safe_session_id,
                exc,
            )
            yield _sse_event(
                "error",
                {
                    "request_id": request_id,
                    "message": str(exc) or "Agent execution failed.",
                },
            )
        finally:
            _set_tool_status("local_llm", "idle", "Ready")
            if saved_h5ad:
                _set_tool_status("single_cell", "idle", "Ready")

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
