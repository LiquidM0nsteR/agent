# agent/backend/agent.py

from __future__ import annotations

import asyncio
import operator
import os
import time
import uuid
from typing import Any, Optional, Literal, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .util import (
    KNOWLEDGE_BASE_PATH,
    MAX_GRAPH_STEPS,
    emit,
    last_user_text,
    safe_text,
    format_exception,
    normalize_uploaded_files,
    infer_input_kind_by_files,
    get_h5ad_files,
    get_rag_files,
    latest_observation,
    parse_router_output,
)
from .memory import get_memory_manager
from .prompts import build_supervisor_prompt, build_final_prompt


Intent = Literal["rag", "web_search", "sc_analysis", "chat", "unknown"]
InputKind = Literal["text", "pdf", "h5ad", "markdown", "mixed", "unknown"]
NextNode = Literal["RAG", "WebSearch", "scAnalysis", "FinalNode"]


class Observation(TypedDict, total=False):
    node: str
    ok: bool
    content: str
    error: str
    metadata: dict[str, Any]


class UploadedFileInfo(TypedDict, total=False):
    original_path: str
    normalized_path: str
    suffix: str
    kind: InputKind
    converted: bool


class StepRecord(TypedDict, total=False):
    node: str
    tool_name: str
    detail: str
    elapsed_ms: float


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    user_input: str
    user_id: str
    session_id: str
    current_turn_id: str
    workspace_settings: dict[str, Any]

    input_kind: InputKind
    intent: Intent
    next_node: NextNode

    uploaded_files: list[str]
    normalized_files: list[UploadedFileInfo]

    h5ad_files: list[str]
    rag_files: list[str]

    knowledge_base_path: str
    upload_workdir: str
    rag_index_dir: str

    observations: Annotated[list[Observation], operator.add]
    tool_results: dict[str, Any]
    current_tool_results: dict[str, Any]
    current_tool_results_turn_id: str
    memory_context: str
    long_term_memories: list[dict[str, Any]]

    steps: Annotated[list[str], operator.add]
    step_records: Annotated[list[StepRecord], operator.add]
    retry_count: int

    final_answer: str


STEP_TOOL_NAME_MAP = {
    "RAG": "local_knowledge_base",
    "WebSearch": "web_search",
    "scAnalysis": "single_cell_pipeline",
}
ROUTE_RESULT_LABEL_MAP = {
    "RAG": "本地知识库问答",
    "WebSearch": "网页搜索",
    "scAnalysis": "单细胞分析流程",
    "FinalNode": "整理回答",
}
DEFAULT_RAG_CONFIDENCE_THRESHOLD = float(os.getenv("AGENT_RAG_CONFIDENCE_THRESHOLD", "0.35"))
EXPLICIT_WEB_QUERY_MARKERS = (
    "最新",
    "最近",
    "新闻",
    "今天",
    "当前",
    "实时",
    "搜一下",
    "搜索",
    "网页",
    "网上",
    "互联网",
    "github",
    "latest",
    "recent",
    "news",
    "today",
    "current",
    "web",
)


# =========================
# 1. LLM 调用
# =========================

def _call_llm(
    prompt: str,
) -> str:
    from .tools.LLM import chat

    text = str(chat(prompt=prompt)).strip()

    if not text:
        raise RuntimeError("LLM.py 返回空结果。")

    return text


def _tool_result_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("answer", "local_answer", "message", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return safe_text(result)


def _is_explicit_web_request(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in EXPLICIT_WEB_QUERY_MARKERS)


def _rag_confidence_threshold(state: AgentState) -> float:
    settings = dict(state.get("workspace_settings") or {})
    raw_value = settings.get(
        "local_source_min_score",
        os.getenv("AGENT_RAG_CONFIDENCE_THRESHOLD", DEFAULT_RAG_CONFIDENCE_THRESHOLD),
    )
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_RAG_CONFIDENCE_THRESHOLD


def _rag_confidence_from_result(result: Any) -> float:
    if not isinstance(result, dict):
        return 0.0

    candidates: list[Any] = []
    for key in ("references", "chunks"):
        values = result.get(key)
        if isinstance(values, list):
            candidates.extend(values)

    scores: list[float] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("score", "rerank_score", "rrf_score", "dense_score", "sparse_score"):
            value = item.get(key)
            if value is None:
                continue
            try:
                scores.append(float(value))
                break
            except (TypeError, ValueError):
                continue

    return max(scores, default=0.0)


def _rag_evidence_is_sufficient(state: AgentState, observation: dict[str, Any]) -> bool:
    metadata = dict(observation.get("metadata") or {})
    confidence = float(metadata.get("confidence") or 0.0)
    threshold = float(metadata.get("confidence_threshold") or _rag_confidence_threshold(state))
    return confidence >= threshold


def _format_web_search_content(result: Any) -> str:
    if not isinstance(result, dict):
        return safe_text(result)

    parts: list[str] = []
    answer = str(result.get("answer") or "").strip()
    if answer:
        parts.append(answer)

    result_items = result.get("results") or []
    if isinstance(result_items, list):
        lines: list[str] = []
        for index, item in enumerate(result_items[:5], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("link") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if not any((title, link, snippet)):
                continue
            entry = [f"{index}. {title or link or '未命名结果'}"]
            if snippet:
                entry.append(snippet)
            if link:
                entry.append(link)
            lines.append("\n".join(entry))
        if lines:
            parts.append("\n\n".join(lines))

    return "\n\n".join(part for part in parts if part.strip()).strip() or safe_text(result)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _current_turn_id(state: AgentState) -> str:
    return str(state.get("current_turn_id") or "")


def _is_current_turn_item(state: AgentState, item: dict[str, Any]) -> bool:
    turn_id = _current_turn_id(state)
    if not turn_id:
        return True
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("turn_id") or "") == turn_id
    return str(item.get("turn_id") or "") == turn_id


def _current_turn_observations(state: AgentState) -> list[Observation]:
    return [
        obs
        for obs in list(state.get("observations") or [])
        if _is_current_turn_item(state, obs)
    ]


def _current_turn_step_records(state: AgentState) -> list[StepRecord]:
    return [
        record
        for record in list(state.get("step_records") or [])
        if _is_current_turn_item(state, record)
    ]


def _current_turn_steps(state: AgentState) -> list[str]:
    records = _current_turn_step_records(state)
    if records:
        return [str(record.get("node") or "") for record in records if record.get("node")]
    if _current_turn_id(state):
        return []
    return list(state.get("steps") or [])


def _current_turn_tool_results(state: AgentState) -> dict[str, Any]:
    current = state.get("current_tool_results")
    if (
        isinstance(current, dict)
        and str(state.get("current_tool_results_turn_id") or "") == _current_turn_id(state)
    ):
        return dict(current)
    if _current_turn_id(state):
        return {}
    return dict(state.get("tool_results") or {})


def _tool_name_for_step(node: str) -> str:
    return STEP_TOOL_NAME_MAP.get(node, "")


def _build_step_record(
    *,
    node: str,
    elapsed_ms: float,
    tool_name: str = "",
    detail: str = "",
    turn_id: str = "",
) -> StepRecord:
    record: StepRecord = {
        "node": node,
        "elapsed_ms": elapsed_ms,
    }
    if tool_name:
        record["tool_name"] = tool_name
    if detail:
        record["detail"] = detail
    if turn_id:
        record["turn_id"] = turn_id
    return record


def _build_route_result(next_node: str) -> tuple[str, str]:
    label = ROUTE_RESULT_LABEL_MAP.get(next_node, next_node or "未知节点")
    tool_name = _tool_name_for_step(next_node)
    return tool_name, f"路由结果：{label}"


def _tool_node_result(
    state: AgentState,
    *,
    node: str,
    tool_key: str,
    ok: bool,
    result: Any,
    elapsed_ms: float,
    content: str = "",
    error: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turn_id = _current_turn_id(state)
    metadata = dict(metadata or {})
    if turn_id:
        metadata["turn_id"] = turn_id
    obs: Observation = {
        "node": node,
        "ok": ok,
        "content": content,
        "metadata": metadata,
    }
    if error:
        obs["error"] = error

    event: dict[str, Any] = {
        "node": node,
        "status": "end" if ok else "error",
    }
    if ok:
        event["ok"] = True
    else:
        event["error"] = error
    emit(event)

    existing_current_tool_results = (
        dict(state.get("current_tool_results") or {})
        if str(state.get("current_tool_results_turn_id") or "") == turn_id
        else {}
    )
    current_tool_results = {
        **existing_current_tool_results,
        tool_key: result if ok else None,
    }

    return {
        "observations": [obs],
        "tool_results": {
            **state.get("tool_results", {}),
            tool_key: result if ok else None,
        },
        "current_tool_results": current_tool_results,
        "current_tool_results_turn_id": turn_id,
        "steps": [node],
        "step_records": [
            _build_step_record(
                node=node,
                elapsed_ms=elapsed_ms,
                tool_name=_tool_name_for_step(node),
                turn_id=turn_id,
            )
        ],
    }


def _build_initial_state(
    *,
    user_input: str,
    user_id: str,
    session_id: str,
    uploaded_files: list[str] | None,
    knowledge_base_path: str,
    upload_workdir: str,
    rag_index_dir: str,
    workspace_settings: dict[str, Any] | None,
) -> AgentState:
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "current_turn_id": uuid.uuid4().hex,
        "workspace_settings": dict(workspace_settings or {}),
        "uploaded_files": uploaded_files or [],
        "normalized_files": [],
        "h5ad_files": [],
        "rag_files": [],
        "knowledge_base_path": knowledge_base_path,
        "upload_workdir": upload_workdir,
        "rag_index_dir": rag_index_dir,
        "observations": [],
        "tool_results": {},
        "current_tool_results": {},
        "current_tool_results_turn_id": "",
        "memory_context": "",
        "long_term_memories": [],
        "steps": [],
        "step_records": [],
        "retry_count": 0,
    }


def _build_runtime_config(
    *,
    memory_manager: Any,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": memory_manager.thread_id(user_id, session_id),
            "session_id": session_id,
        }
    }


def _store_turn(
    *,
    memory_manager: Any,
    user_id: str,
    session_id: str,
    user_input: str,
    final_answer: str,
    state: dict[str, Any],
    workspace_settings: dict[str, Any] | None,
) -> None:
    memory_manager.store_turn(
        user_id=user_id,
        session_id=session_id,
        user_input=user_input,
        final_answer=final_answer,
        state=state,
        workspace_settings=workspace_settings,
    )


def _llm_route_for_text_only(state: AgentState) -> NextNode:
    """
    纯文本输入时，允许使用 LLM Router 判断下一步。
    不再使用关键词匹配。
    """
    user_input = last_user_text(state)

    prompt = build_supervisor_prompt(
        user_input=user_input,
        observations=_current_turn_observations(state),
        steps=_current_turn_steps(state),
        memory_context=str(state.get("memory_context") or ""),
    )

    router_output = _call_llm(prompt)

    next_node = parse_router_output(router_output)

    if next_node in {"RAG", "WebSearch", "FinalNode"}:
        return next_node  # type: ignore[return-value]

    return "FinalNode"


# =========================
# 2. SupervisorNode
# =========================

def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    SupervisorNode 新规则：

    1. 不使用字符串关键词匹配。
    2. 文件形态优先：
       - 有 h5ad -> scAnalysis
       - 有 pdf / md / txt / 代码转 md -> RAG
    3. 纯文本输入：
       - 使用 LLM Router 判断 RAG / WebSearch / FinalNode
       - LLM Router 不可用则进入 FinalNode
    """
    emit({"node": "SupervisorNode", "status": "start"})
    started_at = time.perf_counter()

    user_input = last_user_text(state)
    steps = _current_turn_steps(state)
    retry_count = state.get("retry_count", 0)

    if len(steps) >= MAX_GRAPH_STEPS:
        return {
            "next_node": "FinalNode",
            "steps": ["SupervisorNode"],
            "retry_count": retry_count,
        }

    latest_obs = latest_observation({"observations": _current_turn_observations(state)})

    # 第一次进入 Supervisor：根据输入形态路由
    if latest_obs is None:
        uploaded_files = state.get("uploaded_files", [])
        normalized_files = normalize_uploaded_files(
            uploaded_files,
            upload_workdir=state.get("upload_workdir", ""),
        )

        input_kind = infer_input_kind_by_files(
            normalized_files=normalized_files,
            user_input=user_input,
        )

        h5ad_files = get_h5ad_files(normalized_files)
        rag_files = get_rag_files(normalized_files)

        if h5ad_files:
            next_node: NextNode = "scAnalysis"
            intent: Intent = "sc_analysis"

        elif rag_files:
            next_node = "RAG"
            intent = "rag"

        elif input_kind == "text":
            next_node = _llm_route_for_text_only(state)
            if next_node == "WebSearch" and not _is_explicit_web_request(user_input):
                next_node = "RAG"

            if next_node == "RAG":
                intent = "rag"
            elif next_node == "WebSearch":
                intent = "web_search"
            else:
                intent = "chat"

        else:
            next_node = "FinalNode"
            intent = "unknown"

        emit(
            {
                "node": "SupervisorNode",
                "status": "route",
                "input_kind": input_kind,
                "intent": intent,
                "next_node": next_node,
                "uploaded_files": uploaded_files,
                "normalized_files": normalized_files,
            }
        )

        route_tool_name, route_detail = _build_route_result(next_node)

        return {
            "input_kind": input_kind,
            "intent": intent,
            "next_node": next_node,
            "normalized_files": normalized_files,
            "h5ad_files": h5ad_files,
            "rag_files": rag_files,
            "knowledge_base_path": state.get(
                "knowledge_base_path",
                KNOWLEDGE_BASE_PATH,
            ),
            "steps": ["SupervisorNode"],
            "step_records": [
                _build_step_record(
                    node="SupervisorNode",
                    elapsed_ms=_elapsed_ms(started_at),
                    tool_name=route_tool_name,
                    detail=route_detail,
                    turn_id=_current_turn_id(state),
                )
            ],
            "retry_count": retry_count,
        }

    # 工具执行后的观察判断
    obs_node = latest_obs.get("node", "")
    obs_ok = bool(latest_obs.get("ok", False))

    if obs_ok:
        if (
            obs_node == "RAG"
            and state.get("input_kind") == "text"
            and not _rag_evidence_is_sufficient(state, latest_obs)
        ):
            emit(
                {
                    "node": "SupervisorNode",
                    "status": "rag_low_confidence",
                    "from": obs_node,
                    "next_node": "WebSearch",
                    "confidence": latest_obs.get("metadata", {}).get("confidence"),
                    "confidence_threshold": latest_obs.get("metadata", {}).get("confidence_threshold"),
                }
            )

            route_tool_name, route_detail = _build_route_result("WebSearch")

            return {
                "intent": "web_search",
                "next_node": "WebSearch",
                "steps": ["SupervisorNode"],
                "step_records": [
                    _build_step_record(
                        node="SupervisorNode",
                        elapsed_ms=_elapsed_ms(started_at),
                        tool_name=route_tool_name,
                        detail="RAG 本地证据置信度低于阈值，转入网页搜索。",
                        turn_id=_current_turn_id(state),
                    )
                ],
                "retry_count": retry_count,
            }

        emit(
            {
                "node": "SupervisorNode",
                "status": "observation_ok",
                "from": obs_node,
                "next_node": "FinalNode",
            }
        )

        route_tool_name, route_detail = _build_route_result("FinalNode")

        return {
            "next_node": "FinalNode",
            "steps": ["SupervisorNode"],
            "step_records": [
                _build_step_record(
                    node="SupervisorNode",
                    elapsed_ms=_elapsed_ms(started_at),
                    tool_name=route_tool_name,
                    detail=route_detail,
                    turn_id=_current_turn_id(state),
                )
            ],
            "retry_count": retry_count,
        }

    # 工具失败后的回退策略
    # 注意：这里也不根据关键词判断。
    retry_count += 1

    if retry_count >= 2:
        next_node = "FinalNode"

    elif obs_node == "RAG" and state.get("input_kind") == "text":
        # 本地知识库没有给出有效证据时，才允许转入网页搜索。
        next_node = "WebSearch"

    elif obs_node == "WebSearch" and state.get("input_kind") == "text":
        # 网页搜索失败时，可以让纯文本问题回到 FinalNode。
        next_node = "FinalNode"

    else:
        next_node = "FinalNode"

    emit(
        {
            "node": "SupervisorNode",
            "status": "observation_failed",
            "from": obs_node,
            "next_node": next_node,
            "retry_count": retry_count,
        }
    )

    route_tool_name, route_detail = _build_route_result(next_node)

    return {
        "next_node": next_node,
        "steps": ["SupervisorNode"],
        "step_records": [
            _build_step_record(
                node="SupervisorNode",
                elapsed_ms=_elapsed_ms(started_at),
                tool_name=route_tool_name,
                detail=route_detail,
                turn_id=_current_turn_id(state),
            )
        ],
        "retry_count": retry_count,
    }

def memory_node(state: AgentState) -> dict[str, Any]:
    emit({"node": "MemoryNode", "status": "start"})
    patch = get_memory_manager().prepare_state(state)
    emit(
        {
            "node": "MemoryNode",
            "status": "end",
            "long_term_memory_count": len(patch.get("long_term_memories") or []),
            "has_memory_context": bool(str(patch.get("memory_context") or "").strip()),
        }
    )
    return patch


# =========================
# 3. RAG Node
# =========================

def rag_node(state: AgentState) -> dict[str, Any]:
    """
    RAG 节点。

    输入来源：
    1. PDF；
    2. Markdown；
    3. TXT；
    4. 由 .py / .cpp 等代码转换得到的 Markdown；
    5. LLM Router 判定需要本地知识库检索的纯文本问题。
    """
    emit({"node": "RAG", "status": "start"})
    started_at = time.perf_counter()

    query = last_user_text(state)
    knowledge_base_path = state.get("knowledge_base_path", KNOWLEDGE_BASE_PATH)
    rag_index_dir = state.get("rag_index_dir", "")
    rag_files = state.get("rag_files", [])

    try:
        from .tools.RAG import build_rag_index, run_rag

        if rag_files:
            build_rag_index(
                knowledge_base_path=knowledge_base_path,
                index_dir=rag_index_dir,
                files=rag_files,
                clean=True,
            )

        result = run_rag(
            query=query,
            knowledge_base_path=knowledge_base_path,
            index_dir=rag_index_dir,
            files=rag_files,
            history=state.get("messages", []),
        )

        content = _tool_result_text(result)
        confidence = _rag_confidence_from_result(result)
        confidence_threshold = _rag_confidence_threshold(state)
        return _tool_node_result(
            state,
            node="RAG",
            tool_key="rag",
            ok=bool(content.strip()),
            result=result,
            elapsed_ms=_elapsed_ms(started_at),
            content=content,
            metadata={
                "knowledge_base_path": knowledge_base_path,
                "index_dir": rag_index_dir,
                "files": rag_files,
                "confidence": confidence,
                "confidence_threshold": confidence_threshold,
                "evidence_sufficient": confidence >= confidence_threshold,
            },
        )

    except Exception as exc:
        error = format_exception(exc)
        return _tool_node_result(
            state,
            node="RAG",
            tool_key="rag",
            ok=False,
            result=None,
            elapsed_ms=_elapsed_ms(started_at),
            error=error,
            metadata={
                "knowledge_base_path": knowledge_base_path,
                "index_dir": rag_index_dir,
                "files": rag_files,
            },
        )


# =========================
# 4. WebSearch Node
# =========================

def web_search_node(state: AgentState) -> dict[str, Any]:
    """
    WebSearch 节点。

    注意：
    只有纯文本输入被 LLM Router 判定为需要网页搜索时，
    才会进入这个节点。
    """
    emit({"node": "WebSearch", "status": "start"})
    started_at = time.perf_counter()

    query = last_user_text(state)

    try:
        from .tools.Web import web_search

        results = web_search(
            query=query,
            k=6,
            return_json=True,
        )
        content = _format_web_search_content(results)
        return _tool_node_result(
            state,
            node="WebSearch",
            tool_key="web_search",
            ok=bool(content.strip()),
            result=results,
            elapsed_ms=_elapsed_ms(started_at),
            content=content,
            metadata={"query": query},
        )

    except Exception as exc:
        error = format_exception(exc)
        return _tool_node_result(
            state,
            node="WebSearch",
            tool_key="web_search",
            ok=False,
            result=None,
            elapsed_ms=_elapsed_ms(started_at),
            error=error,
            metadata={"query": query},
        )


# =========================
# 5. scAnalysis Node
# =========================

def sc_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    scAnalysis 节点。

    只由 h5ad 文件触发，不再通过关键词触发。

    建议 tools/SC.py 暴露：

        def run_sc_analysis(
            h5ad_path: str,
            output_dir: str = "./outputs/sc_analysis",
            **kwargs,
        ) -> str | dict:
            ...
    """
    emit({"node": "scAnalysis", "status": "start"})
    started_at = time.perf_counter()

    h5ad_files = state.get("h5ad_files", [])

    if not h5ad_files:
        error = "未检测到 h5ad 文件。scAnalysis 只接收 .h5ad 输入。"
        return _tool_node_result(
            state,
            node="scAnalysis",
            tool_key="sc_analysis",
            ok=False,
            result=None,
            elapsed_ms=_elapsed_ms(started_at),
            error=error,
        )

    h5ad_path = h5ad_files[0]

    try:
        from .tools.SC import run_sc_analysis

        result = run_sc_analysis(
            {
                "user_id": state.get("user_id", "anonymous"),
                "session_id": state.get("session_id", "default"),
                "user_text": last_user_text(state),
                "h5ad_path": h5ad_path,
            }
        )
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        content = _tool_result_text(result)
        return _tool_node_result(
            state,
            node="scAnalysis",
            tool_key="sc_analysis",
            ok=bool(content.strip()),
            result=result,
            elapsed_ms=_elapsed_ms(started_at),
            content=content,
            metadata={
                "h5ad_path": h5ad_path,
                "pdf_report": (
                    result.get("pdf_report", {})
                    if isinstance(result, dict)
                    else {}
                ),
            },
        )

    except Exception as exc:
        error = format_exception(exc)
        return _tool_node_result(
            state,
            node="scAnalysis",
            tool_key="sc_analysis",
            ok=False,
            result=None,
            elapsed_ms=_elapsed_ms(started_at),
            error=error,
            metadata={"h5ad_path": h5ad_path},
        )


# =========================
# 6. FinalNode
# =========================

def _fallback_final_answer(state: AgentState) -> str:
    observations = _current_turn_observations(state)
    tool_results = _current_turn_tool_results(state)

    if not observations:
        return (
            "当前没有调用工具。可能原因是：输入只有纯文本，但 LLM Router 不可用，"
            "因此系统没有进行 RAG 或 WebSearch 路由。"
        )

    successful = [obs for obs in observations if obs.get("ok")]
    failed = [obs for obs in observations if not obs.get("ok")]

    if successful:
        latest_success = successful[-1]
        node = latest_success.get("node", "")
        content = latest_success.get("content", "")

        if node == "scAnalysis":
            sc_result = tool_results.get("sc_analysis")
            report_path = None

            if isinstance(sc_result, dict):
                report_path = (
                    sc_result.get("report_path")
                    or sc_result.get("pdf_path")
                    or sc_result.get("report")
                )

            if report_path:
                return (
                    "单细胞分析任务已完成。\n\n"
                    f"输入文件：{latest_success.get('metadata', {}).get('h5ad_path')}\n"
                    f"分析报告：{report_path}\n\n"
                    "分析节点返回摘要：\n"
                    f"{content}"
                )

            return (
                "单细胞分析任务已执行完成，但当前结果中没有检测到 PDF 报告路径。\n\n"
                f"分析节点返回内容：\n{content}"
            )

        if node == "WebSearch":
            return f"网页搜索已完成，整理结果如下：\n\n{content}"

        if node == "RAG":
            return f"本地知识库检索已完成，结果如下：\n\n{content}"

        return content

    error_lines = []

    for obs in failed:
        error_lines.append(
            f"- {obs.get('node', 'UnknownNode')}: "
            f"{obs.get('error', '工具未返回有效结果')}"
        )

    return "本次请求没有得到有效工具结果。\n\n失败信息如下：\n" + "\n".join(error_lines)


def final_node(state: AgentState) -> dict[str, Any]:
    emit({"node": "FinalNode", "status": "start"})
    started_at = time.perf_counter()

    user_input = last_user_text(state)

    final_prompt = build_final_prompt(
        user_input=user_input,
        observations=_current_turn_observations(state),
        tool_results=_current_turn_tool_results(state),
        steps=_current_turn_steps(state),
        memory_context=str(state.get("memory_context") or ""),
    )

    llm_answer = _call_llm(final_prompt)
    final_answer = llm_answer or _fallback_final_answer(state)

    emit({"node": "FinalNode", "status": "end"})

    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "steps": ["FinalNode"],
        "step_records": [
            _build_step_record(
                node="FinalNode",
                elapsed_ms=_elapsed_ms(started_at),
                turn_id=_current_turn_id(state),
            )
        ],
    }


# =========================
# 7. 构建 LangGraph
# =========================

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("MemoryNode", memory_node)
    builder.add_node("SupervisorNode", supervisor_node)
    builder.add_node("RAG", rag_node)
    builder.add_node("WebSearch", web_search_node)
    builder.add_node("scAnalysis", sc_analysis_node)
    builder.add_node("FinalNode", final_node)

    builder.add_edge(START, "MemoryNode")
    builder.add_edge("MemoryNode", "SupervisorNode")

    builder.add_conditional_edges(
        "SupervisorNode",
        lambda state: state.get("next_node", "FinalNode"),
        {
            "RAG": "RAG",
            "WebSearch": "WebSearch",
            "scAnalysis": "scAnalysis",
            "FinalNode": "FinalNode",
        },
    )

    builder.add_edge("RAG", "SupervisorNode")
    builder.add_edge("WebSearch", "SupervisorNode")
    builder.add_edge("scAnalysis", "SupervisorNode")
    builder.add_edge("FinalNode", END)

    return builder.compile(checkpointer=get_memory_manager().checkpointer)


graph = None


def get_graph():
    global graph
    if graph is None:
        graph = build_graph()
    return graph


# =========================
# 8. 对外调用接口
# =========================

def run_agent(
    user_input: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    uploaded_files: Optional[list[str]] = None,
    knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
    upload_workdir: str = "",
    rag_index_dir: str = "",
    workspace_settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    同步调用接口。
    """
    memory_manager = get_memory_manager()
    initial_state = _build_initial_state(
        user_input=user_input,
        user_id=user_id,
        session_id=session_id,
        uploaded_files=uploaded_files,
        knowledge_base_path=knowledge_base_path,
        upload_workdir=upload_workdir,
        rag_index_dir=rag_index_dir,
        workspace_settings=workspace_settings,
    )
    runtime_config = _build_runtime_config(
        memory_manager=memory_manager,
        user_id=user_id,
        session_id=session_id,
    )

    result = get_graph().invoke(
        initial_state,
        config=runtime_config,
    )

    _store_turn(
        memory_manager=memory_manager,
        user_id=user_id,
        session_id=session_id,
        user_input=user_input,
        final_answer=str(result.get("final_answer") or ""),
        state=result,
        workspace_settings=workspace_settings,
    )

    return result


def stream_agent(
    user_input: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    uploaded_files: Optional[list[str]] = None,
    knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
    upload_workdir: str = "",
    rag_index_dir: str = "",
    workspace_settings: Optional[dict[str, Any]] = None,
):
    """
    流式调用接口。
    """
    memory_manager = get_memory_manager()
    initial_state = _build_initial_state(
        user_input=user_input,
        user_id=user_id,
        session_id=session_id,
        uploaded_files=uploaded_files,
        knowledge_base_path=knowledge_base_path,
        upload_workdir=upload_workdir,
        rag_index_dir=rag_index_dir,
        workspace_settings=workspace_settings,
    )
    runtime_config = _build_runtime_config(
        memory_manager=memory_manager,
        user_id=user_id,
        session_id=session_id,
    )
    compiled_graph = get_graph()

    for event in compiled_graph.stream(
        initial_state,
        config=runtime_config,
        stream_mode=["updates", "custom"],
    ):
        yield event

    state_snapshot = compiled_graph.get_state(runtime_config)
    values = dict(getattr(state_snapshot, "values", {}) or {})
    _store_turn(
        memory_manager=memory_manager,
        user_id=user_id,
        session_id=session_id,
        user_input=user_input,
        final_answer=str(values.get("final_answer") or ""),
        state=values,
        workspace_settings=workspace_settings,
    )
