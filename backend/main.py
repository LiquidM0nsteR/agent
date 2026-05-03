from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import portalocker
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C
from .agent import stream_agent
from .memory import get_memory_manager
from .tools.LLM import DEFAULT_LLM_INSTANCE_COUNT, initialize_llm_pool
from .tools.RAG import build_rag_index
from .util import (
    H5AD_SUFFIXES,
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    TEXT_SUFFIXES,
    build_effective_user_text,
    current_turn_llm_traces,
    current_turn_observations,
    current_turn_step_records,
    current_turn_steps,
    current_turn_tool_results,
    has_session_rag_sources,
    inspect_upload_batch,
    is_rag_upload_candidate,
    new_session_id,
    now_iso,
    safe_filename,
    safe_id,
    safe_relpath,
    truncate_text,
)

C.ensure_runtime_dirs()
C.export_runtime_env()

BASE_DIR = C.PROJECT_ROOT
DATA_DIR = C.DATA_DIR
USERS_DIR = C.USERS_DIR
LOCAL_KNOWLEDGE_DIR = C.LOCAL_KNOWLEDGE_DIR
LOCAL_KNOWLEDGE_INDEX_DIR = C.LOCAL_KNOWLEDGE_INDEX_DIR
FRONTEND_DIST_DIR = C.FRONTEND_DIST_DIR
FRONTEND_ASSETS_DIR = C.FRONTEND_ASSETS_DIR

RAG_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES
WORKSPACE_SETTINGS_DEFAULTS = C.WORKSPACE_SETTINGS_DEFAULTS
TOOL_NAME_MAP = C.TOOL_NAME_MAP
INTENT_MAP = C.INTENT_MAP
STEP_DESCRIPTION_MAP = C.STEP_DESCRIPTION_MAP
TOOL_STATUS: dict[str, dict[str, str]] = {
    "agent": {"state": "idle", "detail": "Ready", "updated_at": ""},
    "local_llm": {"state": "idle", "detail": "Waiting for startup preload", "updated_at": ""},
    "redis": {"state": "idle", "detail": "Waiting for startup", "updated_at": ""},
    "mysql": {"state": "idle", "detail": "Waiting for startup", "updated_at": ""},
    "vllm": {"state": "idle", "detail": "Waiting for startup", "updated_at": ""},
    "retrieval_index": {"state": "ready", "detail": str(LOCAL_KNOWLEDGE_INDEX_DIR), "updated_at": ""},
    "web_search": {
        "state": "ready" if C.SERPER_API_KEY else "unavailable",
        "detail": "SERPER_API_KEY configured" if C.SERPER_API_KEY else "SERPER_API_KEY missing",
        "updated_at": "",
    },
    "single_cell": {"state": "ready", "detail": "On-demand", "updated_at": ""},
}


class LoginRequest(BaseModel):
    user_id: str


class WorkspaceSettingsRequest(BaseModel):
    settings: dict[str, Any]


class MemoryClearRequest(BaseModel):
    scope: str = "session"
    session_id: str = ""


@dataclass(slots=True)
class SubmitContext:
    user_id: str
    session_id: str
    text: str
    effective_text: str
    saved_files: list[dict[str, Any]]
    uploaded_paths: list[str]
    session_root: Path
    uploads_root: Path
    converted_root: Path
    knowledge_base_path: str
    rag_index_dir: str
    workspace_settings: dict[str, Any]


def _set_tool_status(name: str, state: str, detail: str = "") -> None:
    TOOL_STATUS[name] = {"state": state, "detail": detail, "updated_at": now_iso()}


def _startup_log(message: str) -> None:
    print(f"[startup] {message}", flush=True)


def _session_root(user_id: str, session_id: str) -> Path:
    return USERS_DIR / user_id / "sessions" / session_id


def _session_meta_path(user_id: str, session_id: str) -> Path:
    return _session_root(user_id, session_id) / "session.json"


def _session_history_path(user_id: str, session_id: str) -> Path:
    return _session_root(user_id, session_id) / "history.jsonl"


def _workspace_settings_path(user_id: str) -> Path:
    return USERS_DIR / user_id / "workspace_settings.json"


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_session_meta(user_id: str, session_id: str) -> dict[str, Any] | None:
    payload = _load_json(_session_meta_path(user_id, session_id), None)
    return payload if isinstance(payload, dict) else None


def _write_session_meta(user_id: str, session_id: str, payload: dict[str, Any]) -> None:
    _write_json(_session_meta_path(user_id, session_id), payload)


def _load_session_history(user_id: str, session_id: str) -> list[dict[str, Any]]:
    path = _session_history_path(user_id, session_id)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_session_history(user_id: str, session_id: str, messages: list[dict[str, Any]]) -> None:
    path = _session_history_path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in messages:
            handle.write(json.dumps({"role": str(item.get("role") or "assistant"), "content": str(item.get("content") or ""), "created_at": str(item.get("created_at") or now_iso()), "metadata": dict(item.get("metadata") or {})}, ensure_ascii=False) + "\n")


def _build_session_title(user_text: str, saved_files: list[dict[str, Any]]) -> str:
    return truncate_text(user_text, 42) if user_text.strip() else (str(saved_files[0]["original_name"]) if saved_files else "新对话")


def _upsert_session_meta(user_id: str, session_id: str, user_text: str, saved_files: list[dict[str, Any]], assistant_text: str) -> dict[str, Any]:
    existing = _load_session_meta(user_id, session_id) or {}
    payload = {"session_id": session_id, "user_id": user_id, "title": existing.get("title") or _build_session_title(user_text, saved_files), "preview": truncate_text(assistant_text or user_text or "No preview"), "created_at": existing.get("created_at") or now_iso(), "updated_at": now_iso(), "message_count": len(_load_session_history(user_id, session_id))}
    _write_session_meta(user_id, session_id, payload)
    return payload


def _list_user_sessions(user_id: str) -> list[dict[str, Any]]:
    sessions_dir = USERS_DIR / user_id / "sessions"
    if not sessions_dir.exists():
        return []
    results = [_load_session_meta(user_id, item.name) or {"session_id": item.name, "user_id": user_id, "title": item.name, "preview": "", "created_at": "", "updated_at": "", "message_count": len(_load_session_history(user_id, item.name))} for item in sessions_dir.iterdir() if item.is_dir()]
    return sorted(results, key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)


def _load_workspace_settings(user_id: str) -> dict[str, Any]:
    payload = _load_json(_workspace_settings_path(user_id), {})
    settings = dict(WORKSPACE_SETTINGS_DEFAULTS)
    settings.update(payload if isinstance(payload, dict) else {})
    return settings


def _save_workspace_settings(user_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    merged = _load_workspace_settings(user_id)
    merged.update(settings or {})
    _write_json(_workspace_settings_path(user_id), merged)
    return merged


def _clear_session_history(user_id: str, session_id: str) -> int:
    path = _session_history_path(user_id, session_id)
    if not path.exists():
        return 0
    path.unlink()
    meta = _load_session_meta(user_id, session_id)
    if meta:
        meta.update({"preview": "", "message_count": 0, "updated_at": now_iso()})
        _write_session_meta(user_id, session_id, meta)
    return 1


def _clear_user_history(user_id: str) -> int:
    sessions_dir = USERS_DIR / user_id / "sessions"
    return sum(_clear_session_history(user_id, item.name) for item in sessions_dir.iterdir() if item.is_dir()) if sessions_dir.exists() else 0


def _list_knowledge_files() -> list[dict[str, Any]]:
    files = []
    for path in LOCAL_KNOWLEDGE_DIR.rglob("*"):
        if path.is_file():
            stat = path.stat()
            files.append({"path": path.relative_to(LOCAL_KNOWLEDGE_DIR).as_posix(), "name": path.name, "size_bytes": stat.st_size, "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")})
    return sorted(files, key=lambda item: item["updated_at"], reverse=True)


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
    if suffix in PDF_SUFFIXES:
        return "pdf"
    return "file"


def _save_upload(upload: UploadFile, uploads_root: Path) -> dict[str, Any]:
    kind = _detect_upload_kind(upload)
    target_dir = uploads_root / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(upload.filename)
    destination = target_dir / f"{uuid4().hex}_{safe_name}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return {"original_name": upload.filename or safe_name, "kind": kind, "content_type": upload.content_type or "application/octet-stream", "size_bytes": destination.stat().st_size, "path": str(destination.resolve())}


def _prepare_submit_context(user_id: str, session_id: str, text: str, files: list[UploadFile]) -> SubmitContext:
    safe_user_id = safe_id(user_id, "anonymous")
    safe_session_id = safe_id(session_id, "") or new_session_id()
    session_root = _session_root(safe_user_id, safe_session_id)
    uploads_root = session_root / "uploads"
    converted_root = uploads_root / "converted"
    session_rag_index = session_root / "tmp" / "rag_index"
    for directory in (uploads_root, converted_root, session_rag_index):
        directory.mkdir(parents=True, exist_ok=True)
    saved_files = [_save_upload(upload, uploads_root) for upload in files]
    has_session_rag = any(is_rag_upload_candidate(item) for item in saved_files) or has_session_rag_sources(uploads_root)
    return SubmitContext(
        user_id=safe_user_id,
        session_id=safe_session_id,
        text=text,
        effective_text=build_effective_user_text(text, saved_files),
        saved_files=saved_files,
        uploaded_paths=[item["path"] for item in saved_files],
        session_root=session_root,
        uploads_root=uploads_root,
        converted_root=converted_root,
        knowledge_base_path=str((uploads_root if has_session_rag else LOCAL_KNOWLEDGE_DIR).resolve()),
        rag_index_dir=str((session_rag_index if has_session_rag else LOCAL_KNOWLEDGE_INDEX_DIR).resolve()),
        workspace_settings=_load_workspace_settings(safe_user_id),
    )


def _graph_event_to_sse(event: Any) -> tuple[str, dict[str, Any]] | None:
    if isinstance(event, tuple) and len(event) == 2 and event[0] == "final_state":
        return None
    if isinstance(event, tuple) and len(event) == 2 and event[0] == "custom" and isinstance(event[1], dict):
        payload = event[1]
        status = str(payload.get("status") or "")
        return (status if status in {"thought", "answer_start", "answer_delta"} else "status", payload)
    return None


def _normalize_intent(agent_result: dict[str, Any], primary_node: str) -> str:
    intent_type = str(agent_result.get("intent_type") or "").strip()
    if intent_type in {"professional_qa", "non_professional_qa", "sc_analysis", "deep_sc_analysis", "unclear"}:
        return intent_type
    intent = str(agent_result.get("intent") or "").strip()
    return INTENT_MAP.get(intent) or {"Chat": "general_chat", "RAG": "local_knowledge_qa", "WebSearch": "web_search", "scAnalysis": "single_cell_analysis"}.get(primary_node, "general_chat")


def _llm_call_count(agent_result: dict[str, Any]) -> int:
    return len([item for item in current_turn_llm_traces(agent_result) if item.get("counts_as_llm_call", True)])


def _infer_final_information_source(tool_results: dict[str, Any]) -> str:
    sources: list[str] = []
    sc_result = tool_results.get("sc_analysis") if isinstance(tool_results.get("sc_analysis"), dict) else None
    if sc_result:
        sources.append("single-cell analysis")
        if sc_result.get("pdf_interpretation"):
            sources.append("PDF multimodal analysis")
    if isinstance(tool_results.get("rag"), dict):
        sources.append("local RAG")
    if isinstance(tool_results.get("web_search"), dict):
        sources.append("web search")
    return " + ".join(dict.fromkeys(sources))


def _normalize_source_statement(answer: str, source: str) -> str:
    if not answer.strip() or not source:
        return answer
    statement = f"主要信息来源：{source}。"
    if "主要信息来源：" in answer:
        return re.sub(r"主要信息来源：[^。\n]*(?:。)?", statement, answer).strip()
    return f"{answer.rstrip()}\n\n{statement}"


def _build_execution_steps(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    records = current_turn_step_records(agent_result)
    if records:
        return [{"step_id": f"step-{index}", "description": STEP_DESCRIPTION_MAP.get(str(record.get("node") or ""), str(record.get("node") or "")), "detail": str(record.get("detail") or ""), "status": "completed", "tool_name": str(record.get("tool_name") or ""), "elapsed_ms": float(record.get("elapsed_ms") or 0.0)} for index, record in enumerate(records, start=1)]
    return [{"step_id": f"step-{index}", "description": STEP_DESCRIPTION_MAP.get(step, step), "detail": "", "status": "completed", "tool_name": TOOL_NAME_MAP.get(step, ""), "elapsed_ms": 0.0} for index, step in enumerate(current_turn_steps(agent_result), start=1)]


def _build_tool_result(agent_result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    observations = current_turn_observations(agent_result)
    tool_results = current_turn_tool_results(agent_result)
    successful = [item for item in observations if item.get("ok")]
    primary = successful[-1] if successful else (observations[-1] if observations else {})
    primary_node = str(primary.get("node") or "")
    final_answer = str(agent_result.get("final_answer") or "").strip()
    tool_result = {"status": "ok" if final_answer else "error", "tool_name": TOOL_NAME_MAP.get(primary_node, "direct_llm"), "answer": final_answer, "local_answer": final_answer, "message": final_answer, "references": [], "artifacts": [], "observation": primary, "metrics": {"step_count": len(current_turn_steps(agent_result)), "llm_call_count": _llm_call_count(agent_result)}, "meta": {}}
    key = {"Chat": "chat", "RAG": "rag", "WebSearch": "web_search", "scAnalysis": "sc_analysis"}.get(primary_node)
    if key and isinstance(tool_results.get(key), dict):
        raw = tool_results[key]
        tool_result.update({"tool_name": str(raw.get("tool_name") or tool_result["tool_name"]), "references": list(raw.get("references") or []), "artifacts": list(raw.get("artifacts") or []), "meta": dict(raw.get("meta") or {})})
        tool_result["metrics"].update(dict(raw.get("metrics") or {}))
        for field in ("chunks", "grouped_results", "retrieval_trace", "results", "web_search", "pdf_report", "analysis_params", "analysis_result", "report_context", "pdf_interpretation"):
            if field in raw:
                tool_result[field] = raw[field]
    rag_raw = tool_results.get("rag") if isinstance(tool_results.get("rag"), dict) else {}
    rag_meta = rag_raw.get("meta") if isinstance(rag_raw, dict) else {}
    rag_meta = rag_meta if isinstance(rag_meta, dict) else {}
    rag_trace = dict(rag_meta.get("retrieval_trace") or {}) if isinstance(rag_raw, dict) else {}
    if rag_trace:
        tool_result["meta"]["rag_retrieval_trace"] = rag_trace
        tool_result.setdefault("retrieval_trace", rag_trace)
    web_raw = tool_results.get("web_search") if isinstance(tool_results.get("web_search"), dict) else {}
    raw_web_meta = web_raw.get("meta") if isinstance(web_raw, dict) else {}
    web_meta = dict(raw_web_meta or {}) if isinstance(raw_web_meta, dict) else {}
    if web_meta:
        tool_result["meta"]["web_search_meta"] = web_meta
    sc_raw = tool_results.get("sc_analysis") if isinstance(tool_results.get("sc_analysis"), dict) else {}
    if isinstance(sc_raw, dict) and sc_raw:
        sc_meta = sc_raw.get("meta") if isinstance(sc_raw.get("meta"), dict) else {}
        tool_result["meta"]["sc_analysis_meta"] = dict(sc_meta)
        for field in ("pdf_report", "analysis_params", "analysis_result", "report_context", "pdf_interpretation"):
            if field in sc_raw and field not in tool_result:
                tool_result[field] = sc_raw[field]
        if not tool_result.get("artifacts"):
            tool_result["artifacts"] = list(sc_raw.get("artifacts") or [])
    tool_result["meta"]["llm_call_count"] = _llm_call_count(agent_result)
    final_source = _infer_final_information_source(tool_results)
    if final_source:
        normalized_answer = _normalize_source_statement(str(tool_result.get("answer") or ""), final_source)
        tool_result["answer"] = normalized_answer
        tool_result["local_answer"] = normalized_answer
        tool_result["message"] = normalized_answer
        tool_result["meta"]["final_information_source"] = final_source
    return tool_result, primary_node


def _build_agent_payload(agent_result: dict[str, Any], tool_result: dict[str, Any], primary_node: str) -> dict[str, Any]:
    selected_tools = []
    for step in current_turn_steps(agent_result):
        tool_name = TOOL_NAME_MAP.get(step)
        if tool_name and tool_name not in selected_tools:
            selected_tools.append(tool_name)
    selected_tools = selected_tools or ["direct_llm"]
    intent = _normalize_intent(agent_result, primary_node)
    decision = {"intent": intent, "intent_type": agent_result.get("intent_type") or intent, "reason": agent_result.get("intent_reason") or f"当前流程结束于 {primary_node or 'FinalNode'}。", "selected_tools": selected_tools, "execution_steps": _build_execution_steps(agent_result), "llm_traces": current_turn_llm_traces(agent_result), "llm_call_count": _llm_call_count(agent_result), "tool_result": tool_result}
    return {"decision": decision, "graph_execution": {"status": "completed", "used_create_react_agent": False, "dispatched_node": intent, "selected_tool": selected_tools}, "tool_result": tool_result, "state": {"intent": agent_result.get("intent"), "intent_type": agent_result.get("intent_type"), "input_kind": agent_result.get("input_kind"), "steps": current_turn_steps(agent_result), "llm_call_count": _llm_call_count(agent_result), "retry_count": agent_result.get("retry_count", 0)}}


def _append_agent_history(ctx: SubmitContext, assistant_text: str, agent_payload: dict[str, Any]) -> None:
    _append_session_history(ctx.user_id, ctx.session_id, [{"role": "user", "content": ctx.text.strip() or "已上传附件", "metadata": {"file_count": len(ctx.saved_files)}}, {"role": "assistant", "content": assistant_text, "metadata": {"intent": agent_payload["decision"]["intent"], "dispatched_node": agent_payload["graph_execution"]["dispatched_node"], "route_trace": agent_payload["decision"]}}])


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"




# =========================
# Runtime service bootstrap
# =========================

REDIS_HOST = C.REDIS_HOST
REDIS_PORT = C.REDIS_PORT
REDIS_DATA_DIR = C.REDIS_STACK_DATA_DIR
REDIS_LOG_PATH = C.REDIS_STACK_LOG_PATH
REDIS_BIN = C.REDIS_BIN

MYSQL_HOST = C.MYSQL_HOST
MYSQL_PORT = C.MYSQL_PORT

VLLM_HOST = C.VLLM_HOST
VLLM_PORT = C.VLLM_PORT
VLLM_BIN = C.VLLM_BIN
VLLM_MODEL_PATH = C.VLLM_MODEL_PATH
VLLM_SERVED_MODEL_NAME = C.VLLM_SERVED_MODEL_NAME
VLLM_API_KEY = C.VLLM_API_KEY
VLLM_LOG_PATH = C.VLLM_LOG_PATH
VLLM_PID_PATH = C.VLLM_PID_PATH
SERVICE_STARTUP_TIMEOUT = C.SERVICE_STARTUP_TIMEOUT
_VLLM_PROCESS: subprocess.Popen[bytes] | None = None


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    code = sock.connect_ex((host, port))
    sock.close()
    return code == 0


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_open(host, port):
            return
        time.sleep(1)
    raise TimeoutError(f"Service port not ready: {host}:{port}")


def _mysql_ready_detail() -> str:
    if _port_open(MYSQL_HOST, MYSQL_PORT):
        return f"{MYSQL_HOST}:{MYSQL_PORT}"
    return ""


def _mysql_ready() -> bool:
    return bool(_mysql_ready_detail())


def _wait_for_mysql(timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        detail = _mysql_ready_detail()
        if detail:
            return detail
        time.sleep(1)
    raise TimeoutError(f"MySQL not ready: {MYSQL_HOST}:{MYSQL_PORT}")


def _wait_for_file(path: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(1)
    raise TimeoutError(f"Service file not ready: {path}")


def _start_redis_stack() -> None:
    if _port_open(REDIS_HOST, REDIS_PORT):
        _set_tool_status("redis", "ready", f"{REDIS_HOST}:{REDIS_PORT}")
        _startup_log(f"Redis ready at {REDIS_HOST}:{REDIS_PORT}")
        return

    _startup_log(f"Starting Redis Stack at {REDIS_HOST}:{REDIS_PORT}")
    REDIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REDIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            REDIS_BIN,
            "--dir", str(REDIS_DATA_DIR),
            "--bind", REDIS_HOST,
            "--port", str(REDIS_PORT),
            "--appendonly", "yes",
            "--daemonize", "yes",
            "--logfile", str(REDIS_LOG_PATH),
        ],
        check=True,
    )
    _wait_for_port(REDIS_HOST, REDIS_PORT, SERVICE_STARTUP_TIMEOUT)
    _set_tool_status("redis", "ready", f"{REDIS_HOST}:{REDIS_PORT}")
    _startup_log(f"Redis ready at {REDIS_HOST}:{REDIS_PORT}")


def _start_mysql() -> None:
    detail = _mysql_ready_detail()
    if detail:
        _set_tool_status("mysql", "ready", detail)
        _startup_log(f"MySQL ready at {detail}")
        return
    raise TimeoutError(f"MySQL not ready: {MYSQL_HOST}:{MYSQL_PORT}. Start it with ./start.sh.")


def _start_vllm() -> None:
    global _VLLM_PROCESS
    C.export_runtime_env()
    if _port_open(VLLM_HOST, VLLM_PORT):
        _set_tool_status("vllm", "ready", f"{VLLM_HOST}:{VLLM_PORT}")
        _startup_log(f"vLLM ready at {VLLM_HOST}:{VLLM_PORT}")
        return

    VLLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _startup_log(f"Starting vLLM at {VLLM_HOST}:{VLLM_PORT}; log={VLLM_LOG_PATH}")
    log_file = VLLM_LOG_PATH.open("ab")
    _VLLM_PROCESS = subprocess.Popen(C.vllm_command(), stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    VLLM_PID_PATH.write_text(str(_VLLM_PROCESS.pid), encoding="utf-8")
    _wait_for_port(VLLM_HOST, VLLM_PORT, SERVICE_STARTUP_TIMEOUT)
    _set_tool_status("vllm", "ready", f"{VLLM_HOST}:{VLLM_PORT}")
    _startup_log(f"vLLM ready at {VLLM_HOST}:{VLLM_PORT}")


def _stop_vllm() -> None:
    global _VLLM_PROCESS
    process = _VLLM_PROCESS
    if process is None:
        return
    if process.poll() is None:
        _startup_log(f"Stopping vLLM pid={process.pid}")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _startup_log(f"Force stopping vLLM pid={process.pid}")
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    _VLLM_PROCESS = None
    VLLM_PID_PATH.unlink(missing_ok=True)
    _startup_log("vLLM stopped; GPU memory released by process exit")


def _bootstrap_runtime_services() -> None:
    _startup_log("Bootstrapping runtime services")
    _set_tool_status("redis", "running", f"Starting {REDIS_BIN}")
    _start_redis_stack()

    _set_tool_status("mysql", "running", f"{MYSQL_HOST}:{MYSQL_PORT}")
    _start_mysql()

    _set_tool_status("local_llm", "running", f"Starting vLLM model={VLLM_SERVED_MODEL_NAME}")
    _set_tool_status("vllm", "running", f"{VLLM_HOST}:{VLLM_PORT}")
    _start_vllm()

    _startup_log("Initializing memory manager")
    get_memory_manager()
    _startup_log("Initializing LLM client pool and checking completion health")
    count = initialize_llm_pool(instance_count=DEFAULT_LLM_INSTANCE_COUNT)
    _set_tool_status("local_llm", "ready", f"vLLM ready, client_count={count}")
    _startup_log(f"Runtime services ready; llm_client_count={count}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    _bootstrap_runtime_services()
    try:
        yield
    finally:
        _stop_vllm()


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
    user_id = safe_id(payload.user_id, "anonymous")
    (_session_root(user_id, "placeholder").parent).mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "user_id": user_id, "sessions": _list_user_sessions(user_id)}


@app.get("/api/users/{user_id}/sessions")
async def list_sessions(user_id: str) -> dict[str, Any]:
    safe_user_id = safe_id(user_id, "anonymous")
    return {"user_id": safe_user_id, "sessions": _list_user_sessions(safe_user_id)}


@app.get("/api/users/{user_id}/sessions/{session_id}")
async def get_session(user_id: str, session_id: str) -> dict[str, Any]:
    safe_user_id, safe_session_id = safe_id(user_id, "anonymous"), safe_id(session_id, "")
    session_root = _session_root(safe_user_id, safe_session_id)
    if not safe_session_id or not session_root.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    history = _load_session_history(safe_user_id, safe_session_id)
    session = _load_session_meta(safe_user_id, safe_session_id) or {"session_id": safe_session_id, "user_id": safe_user_id, "title": safe_session_id, "preview": "", "created_at": "", "updated_at": "", "message_count": len(history)}
    session["message_count"] = len(history)
    return {"user_id": safe_user_id, "session": session, "history": history}


@app.delete("/api/users/{user_id}/sessions/{session_id}")
async def delete_session(user_id: str, session_id: str) -> dict[str, Any]:
    safe_user_id, safe_session_id = safe_id(user_id, "anonymous"), safe_id(session_id, "")
    root = _session_root(safe_user_id, safe_session_id)
    if not safe_session_id or not root.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    get_memory_manager().clear_session(safe_user_id, safe_session_id)
    shutil.rmtree(root)
    return {"status": "ok", "user_id": safe_user_id, "session_id": safe_session_id}


@app.get("/api/users/{user_id}/workspace/settings")
async def get_workspace_settings(user_id: str) -> dict[str, Any]:
    safe_user_id = safe_id(user_id, "anonymous")
    return {"user_id": safe_user_id, "settings": _load_workspace_settings(safe_user_id)}


@app.put("/api/users/{user_id}/workspace/settings")
async def update_workspace_settings(user_id: str, payload: WorkspaceSettingsRequest) -> dict[str, Any]:
    safe_user_id = safe_id(user_id, "anonymous")
    return {"status": "ok", "user_id": safe_user_id, "settings": _save_workspace_settings(safe_user_id, payload.settings or {})}


@app.post("/api/users/{user_id}/workspace/memory/clear")
async def clear_workspace_memory(user_id: str, payload: MemoryClearRequest) -> dict[str, Any]:
    safe_user_id, scope = safe_id(user_id, "anonymous"), str(payload.scope or "session").strip().lower()
    memory_manager = get_memory_manager()
    if scope == "all":
        cleared_memory, cleared_files = memory_manager.clear_user(safe_user_id), _clear_user_history(safe_user_id)
    elif scope == "session":
        safe_session_id = safe_id(payload.session_id, "")
        if not safe_session_id:
            raise HTTPException(status_code=400, detail="session_id is required when scope=session")
        cleared_memory, cleared_files = memory_manager.clear_session(safe_user_id, safe_session_id), _clear_session_history(safe_user_id, safe_session_id)
    else:
        raise HTTPException(status_code=400, detail="scope must be 'session' or 'all'")
    return {"status": "ok", "user_id": safe_user_id, "scope": scope, "cleared_items": cleared_memory + cleared_files, "cleared_memory": cleared_memory, "cleared_files": cleared_files}


@app.get("/api/workspace/knowledge/files")
async def list_knowledge_files() -> dict[str, Any]:
    return {"root": str(LOCAL_KNOWLEDGE_DIR), "files": _list_knowledge_files()}


@app.post("/api/workspace/knowledge/files")
async def upload_knowledge_files(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    try:
        with portalocker.Lock(str(C.KNOWLEDGE_UPLOAD_LOCK_PATH), timeout=C.KNOWLEDGE_UPLOAD_LOCK_TIMEOUT_SECONDS):
            inspected_files = inspect_upload_batch(
                files,
                allowed_suffixes=C.KNOWLEDGE_UPLOAD_ALLOWED_SUFFIXES,
                max_files=C.KNOWLEDGE_UPLOAD_MAX_FILES,
                max_file_size_bytes=C.KNOWLEDGE_UPLOAD_MAX_FILE_SIZE_BYTES,
                quota_root=LOCAL_KNOWLEDGE_DIR,
                quota_bytes=C.KNOWLEDGE_UPLOAD_TOTAL_QUOTA_BYTES,
            )
            saved_files = []
            for upload, inspected in zip(files, inspected_files):
                destination = LOCAL_KNOWLEDGE_DIR / f"{uuid4().hex}_{inspected['safe_name']}"
                upload.file.seek(0)
                with destination.open("wb") as handle:
                    shutil.copyfileobj(upload.file, handle)
                saved_files.append({"path": destination.relative_to(LOCAL_KNOWLEDGE_DIR).as_posix(), "name": destination.name, "size_bytes": destination.stat().st_size})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except portalocker.exceptions.LockException as exc:
        raise HTTPException(status_code=423, detail=f"Knowledge upload lock timeout: {exc}") from exc
    return {
        "status": "ok",
        "uploaded": saved_files,
        "limits": {
            "allowed_suffixes": sorted(C.KNOWLEDGE_UPLOAD_ALLOWED_SUFFIXES),
            "max_files": C.KNOWLEDGE_UPLOAD_MAX_FILES,
            "max_file_size_bytes": C.KNOWLEDGE_UPLOAD_MAX_FILE_SIZE_BYTES,
            "total_quota_bytes": C.KNOWLEDGE_UPLOAD_TOTAL_QUOTA_BYTES,
        },
        "files": _list_knowledge_files(),
    }


@app.delete("/api/workspace/knowledge/files/{file_path:path}")
async def delete_knowledge_file(file_path: str) -> dict[str, Any]:
    relpath = safe_relpath(file_path)
    target = (LOCAL_KNOWLEDGE_DIR / relpath).resolve()
    if LOCAL_KNOWLEDGE_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {"status": "ok", "deleted": relpath.as_posix(), "files": _list_knowledge_files()}


@app.post("/api/workspace/knowledge/rebuild-index")
async def rebuild_index() -> dict[str, Any]:
    _set_tool_status("retrieval_index", "running", "Rebuilding local knowledge index")
    result = await asyncio.to_thread(build_rag_index, knowledge_base_path=str(LOCAL_KNOWLEDGE_DIR), index_dir=str(LOCAL_KNOWLEDGE_INDEX_DIR), clean=True)
    _set_tool_status("retrieval_index", "ready", f"chunks={result.get('chunk_count', 0)}, vectors={result.get('vector_count', 0)}")
    return result


@app.get("/api/workspace/tool-status")
async def get_tool_status() -> dict[str, Any]:
    return {"status": "ok", "tools": TOOL_STATUS}


@app.get("/api/users/{user_id}/sessions/{session_id}/artifacts/{artifact_path:path}")
async def get_session_artifact(user_id: str, session_id: str, artifact_path: str) -> FileResponse:
    session_root = _session_root(safe_id(user_id, "anonymous"), safe_id(session_id, "")).resolve()
    target = (session_root / artifact_path).resolve()
    if session_root not in target.parents and target != session_root:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.post("/api/agent/submit")
async def submit_agent(user_id: str = Form(default="anonymous"), session_id: str = Form(default=""), text: str = Form(default=""), files: list[UploadFile] | None = File(default=None)) -> StreamingResponse:
    uploads = list(files or [])
    if not text.strip() and not uploads:
        raise HTTPException(status_code=400, detail="请输入消息或上传附件。")
    ctx = _prepare_submit_context(user_id, session_id, text, uploads)

    async def event_stream():
        _set_tool_status("agent", "running", "Agent execution in progress")
        if any(item["kind"] == "h5ad" for item in ctx.saved_files):
            _set_tool_status("single_cell", "running", "Single-cell analysis in progress")
        yield _sse_event("accepted", {"user_id": ctx.user_id, "session_id": ctx.session_id, "text": text})
        yield _sse_event("status", {"stage": "agent", "message": "Agent 正在执行。"})

        agent_result: dict[str, Any] | None = None
        for event in stream_agent(user_input=ctx.effective_text, user_id=ctx.user_id, session_id=ctx.session_id, uploaded_files=ctx.uploaded_paths, knowledge_base_path=ctx.knowledge_base_path, upload_workdir=str(ctx.converted_root.resolve()), rag_index_dir=ctx.rag_index_dir, workspace_settings=ctx.workspace_settings):
            if isinstance(event, tuple) and len(event) == 2 and event[0] == "final_state":
                agent_result = dict(event[1] or {})
            else:
                sse = _graph_event_to_sse(event)
                if sse is not None:
                    yield _sse_event(sse[0], sse[1])
        if agent_result is None:
            raise RuntimeError("Agent 未返回最终状态。")

        tool_result, primary_node = _build_tool_result(agent_result)
        agent_payload = _build_agent_payload(agent_result, tool_result, primary_node)
        assistant_text = str(tool_result.get("answer") or tool_result.get("message") or "").strip()
        _append_agent_history(ctx, assistant_text, agent_payload)
        session = _upsert_session_meta(ctx.user_id, ctx.session_id, user_text=text.strip(), saved_files=ctx.saved_files, assistant_text=assistant_text)
        _set_tool_status("agent", "idle", "Ready")
        _set_tool_status("single_cell", "ready", "On-demand")
        yield _sse_event("answer_start", {"label": "开始生成回答"})
        yield _sse_event("answer_delta", {"delta": assistant_text})
        yield _sse_event("final", {"user_id": ctx.user_id, "session_id": ctx.session_id, "session": session, "tool_result": tool_result, "agent": agent_payload})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
