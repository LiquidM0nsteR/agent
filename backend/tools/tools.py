from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from pypdf import PdfReader

from ..prompts import build_general_chat_messages
from .llm import local_chat_completion
from .rag.config import get_config
from .rag.qa_chain import get_shared_local_knowledge_qa_chain
from .sc_analysis.skill import run_single_cell_skill
from .web import run_web_search_query


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
    return {
        "status": "ok",
        "query": agent_input.user_text,
        "model": str(local_result.get("model_path") or "local_qwen"),
        "answer": final_answer,
        "message": final_answer,
        "image_count": len(image_files),
        "pdf_count": len(pdf_contexts),
    }


async def run_local_knowledge_qa(agent_input: Any) -> dict[str, Any]:
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
    return {
        "status": "ok",
        "query": agent_input.user_text,
        "answer": result.get("answer", ""),
        "message": result.get("message", result.get("answer", "")),
        "references": result.get("references", []),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "retrieval_trace": result.get("retrieval_trace"),
        "evidence_status": result.get("evidence_status", ""),
        "local_source_min_score": min_score,
    }


async def run_web_search(agent_input: Any) -> dict[str, Any]:
    return await run_web_search_query(agent_input.user_text)


async def run_single_cell_analysis(agent_input: Any) -> dict[str, Any]:
    h5ad_asset = next(
        (asset for asset in agent_input.attachments if asset.kind == "h5ad"),
        None,
    )
    if h5ad_asset is None:
        return {
            "status": "error",
            "query": agent_input.user_text,
            "message": "Single-cell analysis requires an h5ad attachment.",
        }

    rag_config = get_config()
    h5ad_path = rag_config.project_root / h5ad_asset.path

    return await run_single_cell_skill(
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        user_text=agent_input.user_text,
        h5ad_path=str(h5ad_path),
    )


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
