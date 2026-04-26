from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import stream_agent
from .memory import get_memory_manager
from .tools.LLM import DEFAULT_LLM_INSTANCE_COUNT, initialize_llm_pool
from .tools.RAG import build_rag_index
from .util import (
    H5AD_SUFFIXES,
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    TEXT_SUFFIXES,
    build_effective_user_text as _build_effective_user_text,
    has_session_rag_sources as _has_session_rag_sources,
    is_local_knowledge_upload_request as _is_local_knowledge_upload_request,
    is_rag_upload_candidate as _is_rag_upload_candidate,
    new_session_id as _new_session_id,
    now_iso as _now_iso,
    safe_filename as _safe_filename,
    safe_id as _safe_id,
    safe_relpath as _safe_relpath,
    truncate_text as _truncate_text,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
LOCAL_KNOWLEDGE_DIR = DATA_DIR / "local_knowledge"
LOCAL_KNOWLEDGE_INDEX_DIR = DATA_DIR / "local_knowledge_index"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

USERS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KNOWLEDGE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

RAG_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES

WORKSPACE_SETTINGS_DEFAULTS: dict[str, Any] = {
    "temperature": 0.2,
    "max_new_tokens": 2048,
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

TOOL_STATUS: dict[str, dict[str, str]] = {
    "agent": {"state": "idle", "detail": "Ready", "updated_at": ""},
    "local_llm": {"state": "idle", "detail": "Waiting for startup preload", "updated_at": ""},
    "retrieval_index": {
        "state": "ready",
        "detail": str(LOCAL_KNOWLEDGE_INDEX_DIR),
        "updated_at": "",
    },
    "web_search": {
        "state": "ready" if os.getenv("SERPER_API_KEY") else "unavailable",
        "detail": "SERPER_API_KEY configured" if os.getenv("SERPER_API_KEY") else "SERPER_API_KEY missing",
        "updated_at": "",
    },
    "single_cell": {"state": "ready", "detail": "On-demand", "updated_at": ""},
}

TOOL_NAME_MAP = {
    "RAG": "local_knowledge_base",
    "WebSearch": "web_search",
    "scAnalysis": "single_cell_pipeline",
}
INTENT_MAP = {
    "rag": "local_knowledge_qa",
    "web_search": "web_search",
    "sc_analysis": "single_cell_analysis",
    "chat": "general_chat",
    "unknown": "general_chat",
}
STEP_DESCRIPTION_MAP = {
    "SupervisorNode": "分析输入并决定路由",
    "RAG": "执行本地知识检索",
    "WebSearch": "执行网页搜索",
    "scAnalysis": "执行单细胞分析",
    "FinalNode": "整理最终回答",
}


class LoginRequest(BaseModel):
    user_id: str


class WorkspaceSettingsRequest(BaseModel):
    settings: dict[str, Any]


class MemoryClearRequest(BaseModel):
    scope: str = "session"
    session_id: str = ""


def _set_tool_status(name: str, state: str, detail: str = "") -> None:
    TOOL_STATUS[name] = {
        "state": state,
        "detail": detail,
        "updated_at": _now_iso(),
    }


def _session_root(user_id: str, session_id: str) -> Path:
    return USERS_DIR / user_id / "sessions" / session_id


def _session_meta_path(user_id: str, session_id: str) -> Path:
    return _session_root(user_id, session_id) / "session.json"


def _session_history_path(user_id: str, session_id: str) -> Path:
    return _session_root(user_id, session_id) / "history.jsonl"


def _workspace_settings_path(user_id: str) -> Path:
    return USERS_DIR / user_id / "workspace_settings.json"


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


def _load_session_history(user_id: str, session_id: str) -> list[dict[str, Any]]:
    path = _session_history_path(user_id, session_id)
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
            records.append(
                {
                    "role": str(payload.get("role") or "assistant"),
                    "content": str(payload.get("content") or ""),
                    "created_at": str(payload.get("created_at") or ""),
                    "metadata": dict(payload.get("metadata") or {}),
                }
            )
    return records


def _append_session_history(
    user_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
) -> None:
    if not messages:
        return
    path = _session_history_path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in messages:
            record = {
                "role": str(item.get("role") or "assistant"),
                "content": str(item.get("content") or ""),
                "created_at": str(item.get("created_at") or _now_iso()),
                "metadata": dict(item.get("metadata") or {}),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_session_title(user_text: str, saved_files: list[dict[str, Any]]) -> str:
    if user_text.strip():
        return _truncate_text(user_text, 42)
    if saved_files:
        return saved_files[0]["original_name"]
    return "新对话"


def _upsert_session_meta(
    user_id: str,
    session_id: str,
    *,
    user_text: str,
    saved_files: list[dict[str, Any]],
    assistant_text: str,
) -> dict[str, Any]:
    existing = _load_session_meta(user_id, session_id) or {}
    history = _load_session_history(user_id, session_id)
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "title": existing.get("title") or _build_session_title(user_text, saved_files),
        "preview": _truncate_text(assistant_text or user_text or "No preview"),
        "created_at": existing.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
        "message_count": len(history),
    }
    _write_session_meta(user_id, session_id, payload)
    return payload


def _list_user_sessions(user_id: str) -> list[dict[str, Any]]:
    sessions_dir = USERS_DIR / user_id / "sessions"
    if not sessions_dir.exists():
        return []

    results: list[dict[str, Any]] = []
    for session_dir in sessions_dir.iterdir():
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
            "message_count": len(_load_session_history(user_id, session_id)),
        }
        results.append(meta)

    results.sort(
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        reverse=True,
    )
    return results


def _load_workspace_settings(user_id: str) -> dict[str, Any]:
    path = _workspace_settings_path(user_id)
    merged = dict(WORKSPACE_SETTINGS_DEFAULTS)
    if not path.exists():
        return merged
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        merged.update(payload)
    return merged


def _save_workspace_settings(user_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    merged = _load_workspace_settings(user_id)
    merged.update(settings or {})
    path = _workspace_settings_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def _clear_session_memory(user_id: str, session_id: str) -> int:
    cleared = 0
    history_path = _session_history_path(user_id, session_id)
    if history_path.exists():
        history_path.unlink()
        cleared += 1
    meta = _load_session_meta(user_id, session_id)
    if meta:
        meta["preview"] = ""
        meta["message_count"] = 0
        meta["updated_at"] = _now_iso()
        _write_session_meta(user_id, session_id, meta)
    return cleared


def _clear_user_memory(user_id: str) -> int:
    sessions_dir = USERS_DIR / user_id / "sessions"
    if not sessions_dir.exists():
        return 0
    cleared = 0
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            cleared += _clear_session_memory(user_id, session_dir.name)
    return cleared


def _list_knowledge_files() -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in LOCAL_KNOWLEDGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(LOCAL_KNOWLEDGE_DIR).as_posix(),
                "name": path.name,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    files.sort(key=lambda item: item["updated_at"], reverse=True)
    return files


def _detect_upload_kind(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    content_type = str(upload.content_type or "")
    if suffix in H5AD_SUFFIXES:
        return "h5ad"
    if suffix in IMAGE_SUFFIXES or content_type.startswith("image/"):
        return "image"
    if suffix == ".txt":
        return "text"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in RAG_SUFFIXES:
        return "pdf"
    return "file"


def _save_upload(upload: UploadFile, uploads_root: Path) -> dict[str, Any]:
    kind = _detect_upload_kind(upload)
    target_dir = uploads_root / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(upload.filename)
    destination = target_dir / f"{uuid4().hex}_{safe_name}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return {
        "original_name": upload.filename or safe_name,
        "kind": kind,
        "content_type": upload.content_type or "application/octet-stream",
        "size_bytes": destination.stat().st_size,
        "path": str(destination.resolve()),
    }


def _copy_uploaded_pdfs_to_local_knowledge(saved_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    LOCAL_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    for item in saved_files:
        if str(item.get("kind") or "") != "pdf":
            continue
        source = Path(str(item.get("path") or "")).expanduser()
        if not source.exists() or not source.is_file():
            continue
        safe_name = _safe_filename(str(item.get("original_name") or source.name))
        destination = LOCAL_KNOWLEDGE_DIR / f"{uuid4().hex}_{safe_name}"
        shutil.copy2(source, destination)
        copied.append(
            {
                "original_name": item.get("original_name") or source.name,
                "path": str(destination.resolve()),
                "knowledge_path": destination.relative_to(LOCAL_KNOWLEDGE_DIR).as_posix(),
                "size_bytes": destination.stat().st_size,
            }
        )
    return copied


def _stream_agent_in_thread(
    *,
    user_input: str,
    user_id: str,
    session_id: str,
    uploaded_files: list[str],
    knowledge_base_path: str,
    upload_workdir: str,
    rag_index_dir: str,
    workspace_settings: dict[str, Any],
) -> Queue:
    queue: Queue = Queue()

    def worker() -> None:
        try:
            def forward_event(payload: dict[str, Any]) -> None:
                queue.put(("event", ("custom", payload)))

            for event in stream_agent(
                user_input=user_input,
                user_id=user_id,
                session_id=session_id,
                uploaded_files=uploaded_files,
                knowledge_base_path=knowledge_base_path,
                upload_workdir=upload_workdir,
                rag_index_dir=rag_index_dir,
                workspace_settings=workspace_settings,
                event_callback=forward_event,
            ):
                queue.put(("event", event))
            queue.put(("done", None))
        except Exception as exc:
            queue.put(("error", exc))

    threading.Thread(target=worker, daemon=True).start()
    return queue


def _graph_event_to_sse(event: Any) -> tuple[str, dict[str, Any]] | None:
    mode = ""
    payload: Any = event
    if isinstance(event, tuple) and len(event) == 2:
        mode, payload = str(event[0]), event[1]

    if mode == "custom" and isinstance(payload, dict):
        status = str(payload.get("status") or "")
        if status == "thought":
            return "thought", payload
        if status == "answer_start":
            return "answer_start", payload
        if status == "answer_delta":
            return "answer_delta", payload
        node = str(payload.get("node") or "")
        if node:
            return "status", {"stage": node, "message": status or node, **payload}
    return None


def _normalize_intent(agent_result: dict[str, Any], primary_node: str) -> str:
    intent = str(agent_result.get("intent") or "").strip()
    if intent in INTENT_MAP:
        return INTENT_MAP[intent]
    if primary_node == "RAG":
        return "local_knowledge_qa"
    if primary_node == "WebSearch":
        return "web_search"
    if primary_node == "scAnalysis":
        return "single_cell_analysis"
    return "general_chat"


def _current_turn_id(agent_result: dict[str, Any]) -> str:
    return str(agent_result.get("current_turn_id") or "")


def _is_current_turn_item(agent_result: dict[str, Any], item: dict[str, Any]) -> bool:
    turn_id = _current_turn_id(agent_result)
    if not turn_id:
        return True
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("turn_id") or "") == turn_id
    return str(item.get("turn_id") or "") == turn_id


def _current_turn_observations(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in list(agent_result.get("observations") or [])
        if isinstance(item, dict) and _is_current_turn_item(agent_result, item)
    ]


def _current_turn_step_records(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in list(agent_result.get("step_records") or [])
        if isinstance(item, dict) and _is_current_turn_item(agent_result, item)
    ]


def _current_turn_llm_traces(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in list(agent_result.get("llm_traces") or [])
        if isinstance(item, dict) and _is_current_turn_item(agent_result, item)
    ]


def _current_turn_steps(agent_result: dict[str, Any]) -> list[str]:
    turn_id = _current_turn_id(agent_result)
    records = _current_turn_step_records(agent_result)
    if records:
        return [str(record.get("node") or "") for record in records if record.get("node")]
    if turn_id:
        return []
    return list(agent_result.get("steps") or [])


def _current_turn_tool_results(agent_result: dict[str, Any]) -> dict[str, Any]:
    current = agent_result.get("current_tool_results")
    if (
        isinstance(current, dict)
        and str(agent_result.get("current_tool_results_turn_id") or "") == _current_turn_id(agent_result)
    ):
        return dict(current)
    if _current_turn_id(agent_result):
        return {}
    return dict(agent_result.get("tool_results") or {})


def _build_execution_steps(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    step_records = _current_turn_step_records(agent_result)
    if step_records:
        execution_steps: list[dict[str, Any]] = []
        for index, record in enumerate(step_records, start=1):
            node = str(record.get("node") or "")
            execution_steps.append(
                {
                    "step_id": f"step-{index}",
                    "description": STEP_DESCRIPTION_MAP.get(node, node),
                    "detail": str(record.get("detail") or ""),
                    "status": "completed",
                    "tool_name": str(record.get("tool_name") or ""),
                    "elapsed_ms": float(record.get("elapsed_ms") or 0.0),
                }
            )
        return execution_steps

    steps = _current_turn_steps(agent_result)
    return [
        {
            "step_id": f"step-{index}",
            "description": STEP_DESCRIPTION_MAP.get(step, step),
            "detail": "",
            "status": "completed",
            "tool_name": TOOL_NAME_MAP.get(step, ""),
            "elapsed_ms": 0.0,
        }
        for index, step in enumerate(steps, start=1)
    ]


def _build_tool_result(agent_result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    observations = _current_turn_observations(agent_result)
    tool_results = _current_turn_tool_results(agent_result)
    successful = [item for item in observations if item.get("ok")]
    primary = successful[-1] if successful else (observations[-1] if observations else {})
    primary_node = str(primary.get("node") or "")
    final_answer = str(agent_result.get("final_answer") or "").strip()

    tool_result: dict[str, Any] = {
        "status": "ok" if final_answer else "error",
        "tool_name": TOOL_NAME_MAP.get(primary_node, "direct_llm"),
        "answer": final_answer,
        "local_answer": final_answer,
        "message": final_answer or str(primary.get("error") or "未返回有效结果。"),
        "references": [],
        "artifacts": [],
        "observation": primary,
        "metrics": {"step_count": len(_current_turn_steps(agent_result))},
        "meta": {},
    }

    if primary_node == "RAG" and isinstance(tool_results.get("rag"), dict):
        rag_result = tool_results["rag"]
        tool_result["tool_name"] = str(rag_result.get("tool_name") or tool_result["tool_name"])
        tool_result["references"] = list(rag_result.get("references") or [])
        tool_result["artifacts"] = list(rag_result.get("artifacts") or [])
        tool_result["metrics"].update(dict(rag_result.get("metrics") or {}))
        tool_result["meta"] = dict(rag_result.get("meta") or {})
        tool_result["retrieved_chunks"] = list(rag_result.get("chunks") or [])
        tool_result["grouped_results"] = list(rag_result.get("grouped_results") or [])
        tool_result["retrieval_trace"] = rag_result.get("retrieval_trace")

    elif primary_node == "WebSearch" and isinstance(tool_results.get("web_search"), dict):
        web_result = tool_results["web_search"]
        tool_result["tool_name"] = str(web_result.get("tool_name") or tool_result["tool_name"])
        tool_result["references"] = list(web_result.get("references") or [])
        tool_result["artifacts"] = list(web_result.get("artifacts") or [])
        tool_result["metrics"].update(dict(web_result.get("metrics") or {}))
        tool_result["meta"] = dict(web_result.get("meta") or {})
        raw_results = list(web_result.get("results") or [])
        web_results: list[dict[str, Any]] = []
        for index, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                continue
            web_results.append(
                {
                    **item,
                    "score": float(max(2, len(raw_results) - index + 1)),
                }
            )
        tool_result["results"] = web_results
        tool_result["web_search"] = {
            "results": web_results,
            "possible_answer": str(web_result.get("answer") or ""),
            "raw_results_count": len(raw_results),
            "retained_results_count": len(web_results),
        }

    elif primary_node == "scAnalysis" and isinstance(tool_results.get("sc_analysis"), dict):
        sc_result = tool_results["sc_analysis"]
        tool_result["tool_name"] = str(sc_result.get("tool_name") or tool_result["tool_name"])
        tool_result["references"] = list(sc_result.get("references") or [])
        tool_result["artifacts"] = list(sc_result.get("artifacts") or [])
        tool_result["metrics"].update(dict(sc_result.get("metrics") or {}))
        tool_result["meta"] = dict(sc_result.get("meta") or {})
        tool_result["pdf_report"] = sc_result.get("pdf_report")
        tool_result["analysis_result"] = sc_result.get("analysis_result")
        tool_result["report_context"] = sc_result.get("report_context")

    return tool_result, primary_node


def _build_agent_payload(agent_result: dict[str, Any], tool_result: dict[str, Any], primary_node: str) -> dict[str, Any]:
    steps = _current_turn_steps(agent_result)
    selected_tools = []
    for step in steps:
        tool_name = TOOL_NAME_MAP.get(step)
        if tool_name and tool_name not in selected_tools:
            selected_tools.append(tool_name)
    if not selected_tools:
        selected_tools = ["direct_llm"]

    intent = _normalize_intent(agent_result, primary_node)
    execution_steps = _build_execution_steps(agent_result)
    llm_traces = _current_turn_llm_traces(agent_result)

    decision = {
        "intent": intent,
        "reason": f"当前流程结束于 {primary_node or 'FinalNode'}。",
        "selected_tools": selected_tools,
        "execution_steps": execution_steps,
        "llm_traces": llm_traces,
        "tool_result": tool_result,
    }
    return {
        "decision": decision,
        "graph_execution": {
            "status": "completed",
            "used_create_react_agent": False,
            "dispatched_node": intent,
            "selected_tool": selected_tools,
        },
        "tool_result": tool_result,
        "state": {
            "intent": agent_result.get("intent"),
            "input_kind": agent_result.get("input_kind"),
            "steps": steps,
            "retry_count": agent_result.get("retry_count", 0),
        },
    }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    _set_tool_status(
        "local_llm",
        "running",
        f"Loading {DEFAULT_LLM_INSTANCE_COUNT} instance(s)",
    )
    count = initialize_llm_pool(instance_count=DEFAULT_LLM_INSTANCE_COUNT)
    _set_tool_status(
        "local_llm",
        "ready",
        f"Loaded {count} instance(s)",
    )
    yield


app = FastAPI(title="Agent Backend", version="0.1.0", lifespan=lifespan)

if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")


@app.get("/")
async def index() -> FileResponse:
    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend dist not found")
    return FileResponse(index_file)


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    icon_file = FRONTEND_DIST_DIR / "favicon.ico"
    if not icon_file.exists():
        raise HTTPException(status_code=404, detail="favicon not found")
    return FileResponse(icon_file)


@app.post("/api/auth/login")
async def login(payload: LoginRequest) -> dict[str, Any]:
    user_id = _safe_id(payload.user_id, "anonymous")
    (_session_root(user_id, "placeholder").parent).mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "user_id": user_id,
        "sessions": _list_user_sessions(user_id),
    }


@app.get("/api/users/{user_id}/sessions")
async def list_sessions(user_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    return {"user_id": safe_user_id, "sessions": _list_user_sessions(safe_user_id)}


@app.get("/api/users/{user_id}/sessions/{session_id}")
async def get_session(user_id: str, session_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "")
    if not safe_session_id:
        raise HTTPException(status_code=400, detail="Invalid session id")
    session_root = _session_root(safe_user_id, safe_session_id)
    if not session_root.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    history = _load_session_history(safe_user_id, safe_session_id)
    session = _load_session_meta(safe_user_id, safe_session_id) or {
        "session_id": safe_session_id,
        "user_id": safe_user_id,
        "title": safe_session_id,
        "preview": "",
        "created_at": "",
        "updated_at": "",
        "message_count": len(history),
    }
    session["message_count"] = len(history)
    return {"user_id": safe_user_id, "session": session, "history": history}


@app.delete("/api/users/{user_id}/sessions/{session_id}")
async def delete_session(user_id: str, session_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "")
    if not safe_session_id:
        raise HTTPException(status_code=400, detail="Invalid session id")
    session_root = _session_root(safe_user_id, safe_session_id)
    if not session_root.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    get_memory_manager().clear_session(safe_user_id, safe_session_id)
    shutil.rmtree(session_root)
    return {"status": "ok", "user_id": safe_user_id, "session_id": safe_session_id}


@app.get("/api/users/{user_id}/workspace/settings")
async def get_workspace_settings(user_id: str) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    return {"user_id": safe_user_id, "settings": _load_workspace_settings(safe_user_id)}


@app.put("/api/users/{user_id}/workspace/settings")
async def update_workspace_settings(user_id: str, payload: WorkspaceSettingsRequest) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    settings = _save_workspace_settings(safe_user_id, payload.settings or {})
    return {"status": "ok", "user_id": safe_user_id, "settings": settings}


@app.post("/api/users/{user_id}/workspace/memory/clear")
async def clear_workspace_memory(user_id: str, payload: MemoryClearRequest) -> dict[str, Any]:
    safe_user_id = _safe_id(user_id, "anonymous")
    scope = str(payload.scope or "session").strip().lower()
    memory_manager = get_memory_manager()
    if scope not in {"session", "all"}:
        raise HTTPException(status_code=400, detail="scope must be 'session' or 'all'")
    if scope == "all":
        cleared_memory = memory_manager.clear_user(safe_user_id)
        cleared_files = _clear_user_memory(safe_user_id)
    else:
        safe_session_id = _safe_id(payload.session_id, "")
        if not safe_session_id:
            raise HTTPException(status_code=400, detail="session_id is required when scope=session")
        cleared_memory = memory_manager.clear_session(safe_user_id, safe_session_id)
        cleared_files = _clear_session_memory(safe_user_id, safe_session_id)
    return {
        "status": "ok",
        "user_id": safe_user_id,
        "scope": scope,
        "cleared_items": cleared_memory + cleared_files,
        "cleared_memory": cleared_memory,
        "cleared_files": cleared_files,
    }


@app.get("/api/workspace/knowledge/files")
async def list_knowledge_files() -> dict[str, Any]:
    return {"root": str(LOCAL_KNOWLEDGE_DIR), "files": _list_knowledge_files()}


@app.post("/api/workspace/knowledge/files")
async def upload_knowledge_files(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    for upload in files:
        safe_name = _safe_filename(upload.filename)
        destination = LOCAL_KNOWLEDGE_DIR / f"{uuid4().hex}_{safe_name}"
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
    return {"status": "ok", "files": _list_knowledge_files()}


@app.delete("/api/workspace/knowledge/files/{file_path:path}")
async def delete_knowledge_file(file_path: str) -> dict[str, Any]:
    try:
        relpath = _safe_relpath(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = (LOCAL_KNOWLEDGE_DIR / relpath).resolve()
    if LOCAL_KNOWLEDGE_DIR.resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {"status": "ok", "deleted": relpath.as_posix(), "files": _list_knowledge_files()}


@app.post("/api/workspace/knowledge/rebuild-index")
async def rebuild_index() -> dict[str, Any]:
    _set_tool_status("retrieval_index", "running", "Rebuilding local knowledge index")
    try:
        result = await asyncio.to_thread(
            build_rag_index,
            knowledge_base_path=str(LOCAL_KNOWLEDGE_DIR),
            index_dir=str(LOCAL_KNOWLEDGE_INDEX_DIR),
            clean=True,
        )
    except Exception as exc:
        _set_tool_status("retrieval_index", "error", str(exc))
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {exc}") from exc
    _set_tool_status(
        "retrieval_index",
        "ready",
        f"chunks={result.get('chunk_count', 0)}, vectors={result.get('vector_count', 0)}",
    )
    return result


@app.get("/api/workspace/tool-status")
async def get_tool_status() -> dict[str, Any]:
    return {"status": "ok", "tools": TOOL_STATUS}


@app.get("/api/users/{user_id}/sessions/{session_id}/artifacts/{artifact_path:path}")
async def get_session_artifact(user_id: str, session_id: str, artifact_path: str) -> FileResponse:
    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "")
    session_root = _session_root(safe_user_id, safe_session_id).resolve()
    target = (session_root / artifact_path).resolve()
    if session_root not in target.parents and target != session_root:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.post("/api/agent/submit")
async def submit_agent(
    user_id: str = Form(default="anonymous"),
    session_id: str = Form(default=""),
    text: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
) -> StreamingResponse:
    uploads = list(files or [])
    if not text.strip() and not uploads:
        raise HTTPException(status_code=400, detail="请输入消息或上传附件。")

    safe_user_id = _safe_id(user_id, "anonymous")
    safe_session_id = _safe_id(session_id, "") or _new_session_id()
    session_root = _session_root(safe_user_id, safe_session_id)
    uploads_root = session_root / "uploads"
    converted_root = uploads_root / "converted"
    rag_index_dir = session_root / "tmp" / "rag_index"
    uploads_root.mkdir(parents=True, exist_ok=True)
    converted_root.mkdir(parents=True, exist_ok=True)
    rag_index_dir.mkdir(parents=True, exist_ok=True)

    saved_files = [_save_upload(upload, uploads_root) for upload in uploads]
    agent_uploaded_paths = [
        item["path"]
        for item in saved_files
        if str(item.get("kind") or "") != "pdf"
    ]
    effective_user_text = _build_effective_user_text(text, saved_files)
    workspace_settings = _load_workspace_settings(safe_user_id)

    has_uploaded_rag_files = any(_is_rag_upload_candidate(item) for item in saved_files)
    has_session_rag_files = has_uploaded_rag_files or _has_session_rag_sources(uploads_root)
    knowledge_base_path = (
        str(uploads_root.resolve())
        if has_session_rag_files
        else str(LOCAL_KNOWLEDGE_DIR.resolve())
    )
    effective_rag_index_dir = (
        str(rag_index_dir.resolve())
        if has_session_rag_files
        else str(LOCAL_KNOWLEDGE_INDEX_DIR.resolve())
    )

    async def event_stream():
        _set_tool_status("agent", "running", "Agent execution in progress")
        if any(item["kind"] == "h5ad" for item in saved_files):
            _set_tool_status("single_cell", "running", "Single-cell analysis in progress")
        yield _sse_event(
            "accepted",
            {
                "user_id": safe_user_id,
                "session_id": safe_session_id,
                "text": text,
            },
        )
        yield _sse_event(
            "status",
            {
                "stage": "agent",
                "message": "Agent 正在执行。",
            },
        )

        try:
            if _is_local_knowledge_upload_request(text) and any(item["kind"] == "pdf" for item in saved_files):
                copied_files = _copy_uploaded_pdfs_to_local_knowledge(saved_files)
                if copied_files:
                    assistant_text = (
                        f"已将 {len(copied_files)} 个 PDF 复制到本地知识库目录。"
                        "需要重建本地知识库索引后，RAG 才会检索到这些文件。"
                    )
                    tool_result = {
                        "status": "ok",
                        "tool_name": "local_knowledge_upload",
                        "answer": assistant_text,
                        "message": assistant_text,
                        "artifacts": [],
                        "references": [],
                        "metrics": {"copied_files": len(copied_files)},
                        "meta": {
                            "knowledge_base_path": str(LOCAL_KNOWLEDGE_DIR.resolve()),
                            "files": copied_files,
                            "index_rebuild_required": True,
                        },
                    }
                else:
                    assistant_text = "没有检测到可复制到本地知识库的 PDF 文件。"
                    tool_result = {
                        "status": "error",
                        "tool_name": "local_knowledge_upload",
                        "answer": assistant_text,
                        "message": assistant_text,
                        "artifacts": [],
                        "references": [],
                        "metrics": {"copied_files": 0},
                        "meta": {"knowledge_base_path": str(LOCAL_KNOWLEDGE_DIR.resolve())},
                    }
                agent_payload = {
                    "status": tool_result["status"],
                    "answer": assistant_text,
                    "tool_result": tool_result,
                    "decision": {
                        "intent": "local_knowledge_upload",
                        "reason": "用户明确要求将上传 PDF 加入本地知识库。",
                        "selected_tools": ["local_knowledge_upload"],
                        "execution_steps": [],
                    },
                    "graph_execution": {
                        "dispatched_node": "LocalKnowledgeUpload",
                        "steps": [],
                        "observations": [],
                    },
                }
                yield _sse_event("answer_start", {"label": "开始生成回答"})
                yield _sse_event("answer_delta", {"delta": assistant_text})
            else:
                event_queue = _stream_agent_in_thread(
                    user_input=effective_user_text,
                    user_id=safe_user_id,
                    session_id=safe_session_id,
                    uploaded_files=agent_uploaded_paths,
                    knowledge_base_path=knowledge_base_path,
                    upload_workdir=str(converted_root.resolve()),
                    rag_index_dir=effective_rag_index_dir,
                    workspace_settings=workspace_settings,
                )

                agent_result: dict[str, Any] | None = None
                while True:
                    kind, item = await asyncio.to_thread(event_queue.get)
                    if kind == "error":
                        raise item
                    if kind == "done":
                        break
                    if kind != "event":
                        continue
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "final_state":
                        agent_result = dict(item[1] or {})
                        continue
                    sse_event = _graph_event_to_sse(item)
                    if sse_event is not None:
                        yield _sse_event(sse_event[0], sse_event[1])

                if agent_result is None:
                    raise RuntimeError("Agent 未返回最终状态。")

                tool_result, primary_node = _build_tool_result(agent_result)
                agent_payload = _build_agent_payload(agent_result, tool_result, primary_node)
                assistant_text = str(tool_result.get("answer") or tool_result.get("message") or "").strip()

            _append_session_history(
                safe_user_id,
                safe_session_id,
                [
                    {
                        "role": "user",
                        "content": text.strip() or "已上传附件",
                        "metadata": {"file_count": len(saved_files)},
                    },
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "metadata": {
                            "intent": agent_payload["decision"]["intent"],
                            "dispatched_node": agent_payload["graph_execution"]["dispatched_node"],
                            "route_trace": {
                                "intent": agent_payload["decision"]["intent"],
                                "reason": agent_payload["decision"]["reason"],
                                "dispatched_node": agent_payload["graph_execution"]["dispatched_node"],
                                "selected_tools": agent_payload["decision"]["selected_tools"],
                                "execution_steps": agent_payload["decision"]["execution_steps"],
                                "llm_traces": [],
                            },
                        },
                    },
                ],
            )

            session = _upsert_session_meta(
                safe_user_id,
                safe_session_id,
                user_text=text.strip(),
                saved_files=saved_files,
                assistant_text=assistant_text,
            )
            _set_tool_status("agent", "idle", "Ready")
            _set_tool_status("single_cell", "ready", "On-demand")
            yield _sse_event(
                "final",
                {
                    "user_id": safe_user_id,
                    "session_id": safe_session_id,
                    "session": session,
                    "tool_result": tool_result,
                    "agent": agent_payload,
                },
            )
        except Exception as exc:
            _set_tool_status("agent", "error", str(exc))
            _set_tool_status("single_cell", "ready", "On-demand")
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
