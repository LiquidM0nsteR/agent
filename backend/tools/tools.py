from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import time
from typing import Any

from langchain_core.tools import BaseTool, tool
from pypdf import PdfReader

from ..prompts import build_general_chat_messages
from .llm import local_chat_completion
from .rag.config import get_config
from .rag.qa_chain import get_shared_local_knowledge_qa_chain
from .sc_analysis.skill import run_single_cell_skill
from .web import run_web_search_query

logger = logging.getLogger(__name__)

_COMMON_TOOL_RESULT_KEYS = {
    "tool_name",
    "status",
    "query",
    "answer",
    "message",
    "local_answer",
    "evidence_status",
    "references",
    "artifacts",
    "metrics",
    "meta",
    "observation",
}


# 这些 runner 同时给 LangGraph 工作流和兼容的 LangChain tool wrapper 复用。
async def run_general_chat(agent_input: Any) -> dict[str, Any]:
    return await run_general_chat_with_memory(agent_input)


def _resolve_attachment_path(asset: Any) -> Path:
    return get_config().project_root / str(getattr(asset, "path", "") or "")


def _load_general_chat_image_attachments(agent_input: Any) -> list[dict[str, Any]]:
    image_files: list[dict[str, Any]] = []
    for asset in getattr(agent_input, "attachments", []) or []:
        if str(getattr(asset, "kind", "") or "") != "image":
            continue
        image_path = _resolve_attachment_path(asset)
        if not image_path.exists():
            continue
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_files.append(
            {
                "name": str(getattr(asset, "name", "") or image_path.name),
                "data_url": f"data:{getattr(asset, 'content_type', 'image/png')};base64,{encoded}",
            }
        )
    return image_files


def _clean_pdf_excerpt(text: str) -> str:
    return " ".join(str(text or "").split())


def _load_general_chat_pdf_context(agent_input: Any) -> list[dict[str, Any]]:
    pdf_contexts: list[dict[str, Any]] = []
    for asset in getattr(agent_input, "attachments", []) or []:
        if str(getattr(asset, "kind", "") or "") != "pdf":
            continue
        pdf_path = _resolve_attachment_path(asset)
        if not pdf_path.exists():
            continue
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            continue

        excerpts: list[str] = []
        for page_index, page in enumerate(reader.pages[:4], start=1):
            page_text = _clean_pdf_excerpt(page.extract_text() or "")
            if not page_text:
                continue
            excerpts.append(f"[page {page_index}] {page_text[:900]}")
        if not excerpts:
            continue
        pdf_contexts.append(
            {
                "name": str(getattr(asset, "name", "") or pdf_path.name),
                "excerpt": "\n".join(excerpts)[:3200],
            }
        )
    return pdf_contexts


async def run_general_chat_with_memory(
    agent_input: Any,
    *,
    recent_messages: list[dict[str, Any]] | None = None,
    short_summary: str = "",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_files = _load_general_chat_image_attachments(agent_input)
    pdf_contexts = _load_general_chat_pdf_context(agent_input)
    messages = build_general_chat_messages(
        user_text=agent_input.user_text,
        recent_messages=recent_messages,
        short_summary=short_summary,
        profile=profile,
        pdf_contexts=pdf_contexts,
        image_files=image_files,
    )
    local_result = await local_chat_completion(
        messages,
        max_new_tokens=1024,
        temperature=0.2,
        trace_label="general_chat_answer",
    )
    final_answer = str(local_result.get("message") or "").strip() or "模型未返回内容。"
    result = _build_tool_result(
        "general_chat",
        status="ok",
        query=agent_input.user_text,
        answer=final_answer,
        evidence_status="not_applicable",
        metrics=_merge_metrics(
            local_result.get("metrics"),
            {"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
        ),
        meta={
            "model": str(local_result.get("model_path") or "local_qwen"),
            "device": str(local_result.get("device") or ""),
        },
        observation={
            "image_count": len(image_files),
            "pdf_count": len(pdf_contexts),
        },
        model=str(local_result.get("model_path") or "local_qwen"),
        image_count=len(image_files),
        pdf_count=len(pdf_contexts),
    )
    logger.info(
        "[tool.general_chat] session=%s images=%s pdfs=%s tool_ms=%.2f",
        getattr(agent_input, "session_id", ""),
        len(image_files),
        len(pdf_contexts),
        float(result.get("metrics", {}).get("tool_ms", 0.0)),
    )
    return result


async def run_local_knowledge_qa(agent_input: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    qa_chain = get_shared_local_knowledge_qa_chain(get_config())
    workspace_settings = dict(getattr(agent_input, "workspace_settings", {}) or {})
    try:
        min_score = max(
            0.0,
            float(workspace_settings.get("local_source_min_score", 0.35)),
        )
    except (TypeError, ValueError):
        min_score = 0.35
    result = qa_chain.ask(agent_input.user_text, min_score=min_score)
    canonical = _build_tool_result(
        "local_knowledge_qa",
        status="ok",
        query=agent_input.user_text,
        answer=str(result.get("answer") or ""),
        message=str(result.get("message") or result.get("answer") or ""),
        evidence_status=str(result.get("evidence_status") or "insufficient"),
        references=result.get("references"),
        metrics=_merge_metrics(
            result.get("metrics"),
            {"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
        ),
        observation={
            "retrieved_chunks": result.get("retrieved_chunks") or [],
            "retrieval_trace": result.get("retrieval_trace") or {},
            "local_source_min_score": min_score,
        },
        retrieved_chunks=result.get("retrieved_chunks", []),
        retrieval_trace=result.get("retrieval_trace"),
        local_source_min_score=min_score,
    )
    logger.info(
        "[tool.local_knowledge_qa] session=%s evidence=%s refs=%s tool_ms=%.2f",
        getattr(agent_input, "session_id", ""),
        canonical.get("evidence_status", ""),
        len(canonical.get("references") or []),
        float(canonical.get("metrics", {}).get("tool_ms", 0.0)),
    )
    return canonical


async def run_web_search(agent_input: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    raw_result = await run_web_search_query(agent_input.user_text)
    references = raw_result.get("references") or []
    evidence_status = "sufficient" if references else "insufficient"
    if str(raw_result.get("status") or "").lower() in {"error", "unavailable"}:
        evidence_status = "not_applicable"
    canonical = _build_tool_result(
        "web_search",
        status=str(raw_result.get("status") or "ok"),
        query=agent_input.user_text,
        answer=str(raw_result.get("answer") or ""),
        message=str(raw_result.get("message") or raw_result.get("answer") or ""),
        evidence_status=evidence_status,
        references=references,
        metrics=_merge_metrics(
            raw_result.get("metrics"),
            {"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
        ),
        meta={"provider": str(raw_result.get("provider") or "serper")},
        observation={
            "provider": raw_result.get("provider"),
            "queries": raw_result.get("queries") or [],
            "results": raw_result.get("results") or [],
            "possible_answer": str(raw_result.get("possible_answer") or ""),
        },
        provider=raw_result.get("provider"),
        queries=raw_result.get("queries"),
        results=raw_result.get("results"),
        possible_answer=raw_result.get("possible_answer"),
    )
    logger.info(
        "[tool.web_search] session=%s status=%s refs=%s tool_ms=%.2f",
        getattr(agent_input, "session_id", ""),
        canonical.get("status", ""),
        len(canonical.get("references") or []),
        float(canonical.get("metrics", {}).get("tool_ms", 0.0)),
    )
    return canonical


async def run_single_cell_analysis(agent_input: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    h5ad_asset = next(
        (asset for asset in agent_input.attachments if asset.kind == "h5ad"),
        None,
    )
    if h5ad_asset is None:
        return _build_tool_result(
            "single_cell_analysis",
            status="error",
            query=agent_input.user_text,
            message="Single-cell analysis requires an h5ad attachment.",
            evidence_status="not_applicable",
            metrics={"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
        )

    rag_config = get_config()
    h5ad_path = rag_config.project_root / h5ad_asset.path

    raw_result = await run_single_cell_skill(
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        user_text=agent_input.user_text,
        h5ad_path=str(h5ad_path),
    )
    canonical = _build_tool_result(
        "single_cell_analysis",
        status=str(raw_result.get("status") or "ok"),
        query=agent_input.user_text,
        answer=str(raw_result.get("answer") or ""),
        message=str(raw_result.get("message") or raw_result.get("answer") or ""),
        evidence_status=(
            "sufficient"
            if str(raw_result.get("status") or "").lower() in {"", "ok"}
            else "not_applicable"
        ),
        references=raw_result.get("references"),
        artifacts=raw_result.get("artifacts"),
        metrics=_merge_metrics(
            raw_result.get("metrics"),
            {"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
        ),
        observation=_extract_tool_observation(raw_result),
        analysis_params=raw_result.get("analysis_params"),
        analysis_result=raw_result.get("analysis_result"),
        report_context=raw_result.get("report_context"),
        pdf_report=raw_result.get("pdf_report"),
        subset_test_cells=raw_result.get("subset_test_cells"),
    )
    logger.info(
        "[tool.single_cell_analysis] session=%s status=%s artifacts=%s tool_ms=%.2f",
        getattr(agent_input, "session_id", ""),
        canonical.get("status", ""),
        len(canonical.get("artifacts") or []),
        float(canonical.get("metrics", {}).get("tool_ms", 0.0)),
    )
    return canonical


def _build_tool_result(
    tool_name: str,
    *,
    status: str,
    query: str,
    answer: str = "",
    message: str = "",
    evidence_status: str = "not_applicable",
    references: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    answer_text = str(answer or message or "").strip()
    message_text = str(message or answer_text or "").strip()
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "status": str(status or "ok"),
        "query": str(query or ""),
        "answer": answer_text,
        "message": message_text,
        "local_answer": answer_text or message_text,
        "evidence_status": str(evidence_status or "not_applicable"),
        "references": list(references or []),
        "artifacts": list(artifacts or []),
        "metrics": _merge_metrics(metrics),
        "meta": dict(meta or {}),
        "observation": dict(observation or {}),
    }
    for key, value in extras.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def _extract_tool_observation(raw_result: dict[str, Any]) -> dict[str, Any]:
    observation = raw_result.get("observation")
    merged = dict(observation) if isinstance(observation, dict) else {}
    for key, value in raw_result.items():
        if key in _COMMON_TOOL_RESULT_KEYS or value is None:
            continue
        merged[key] = value
    return merged


def _merge_metrics(*metric_maps: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metric_map in metric_maps:
        if not isinstance(metric_map, dict):
            continue
        for key, value in metric_map.items():
            if isinstance(value, (int, float)):
                merged[key] = round(float(value), 2)
            elif value is not None:
                merged[key] = value
    return merged


def _json_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


def build_agent_tools(*, agent_input: Any) -> list[BaseTool]:
    @tool("general_chat")
    async def general_chat_tool() -> str:
        result = await run_general_chat(agent_input)
        return _json_tool_result(result)

    @tool("local_knowledge_qa")
    async def local_knowledge_qa_tool() -> str:
        result = await run_local_knowledge_qa(agent_input)
        return _json_tool_result(result)

    @tool("web_search")
    async def web_search_tool() -> str:
        result = await run_web_search(agent_input)
        return _json_tool_result(result)

    @tool("single_cell_analysis")
    async def single_cell_analysis_tool() -> str:
        result = await run_single_cell_analysis(agent_input)
        return _json_tool_result(result)

    return [
        general_chat_tool,
        local_knowledge_qa_tool,
        web_search_tool,
        single_cell_analysis_tool,
    ]
