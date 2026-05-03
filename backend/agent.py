from __future__ import annotations

import asyncio
import json
import operator
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .memory import get_memory_manager
from .prompts import (
    build_final_prompt,
    build_final_revision_prompt,
    build_intent_audit_prompt,
    build_pdf_report_interpretation_prompt,
    build_supervisor_prompt,
)
from .util import (
    DEFAULT_RAG_CONFIDENCE_THRESHOLD,
    KNOWLEDGE_BASE_PATH,
    MAX_GRAPH_STEPS,
    MAX_LLM_GENERATION_CALLS,
    SC_OUTPUT_DIR,
    build_step_record,
    current_turn_id,
    current_turn_llm_traces,
    current_turn_observations,
    current_turn_steps,
    current_turn_tool_results,
    elapsed_ms,
    emit,
    extract_h5ad_paths_from_text,
    get_h5ad_files,
    get_rag_files,
    infer_input_kind_by_files,
    last_user_text,
    normalize_rag_files_for_base,
    normalize_uploaded_files,
    parse_json_object,
    rag_confidence_from_result,
    sanitize_action_queries,
    stream_event_callback,
    tool_node_result,
    tool_result_text,
)

Intent = Literal["rag", "web_search", "sc_analysis", "chat", "unknown"]
IntentType = Literal["professional_qa", "non_professional_qa", "sc_analysis", "deep_sc_analysis", "unclear"]
InputKind = Literal["text", "pdf", "h5ad", "markdown", "mixed", "unknown"]
NextNode = Literal["RAG", "WebSearch", "scAnalysis", "FinalNode"]
SupervisorPhase = Literal["initial_route", "tool_result_review", "tool_error_recovery"]

TOOL_NAME = {"RAG": "local_knowledge_base", "WebSearch": "web_search", "scAnalysis": "single_cell_pipeline", "FinalNode": "direct_llm"}
NODE_INTENT: dict[str, Intent] = {"RAG": "rag", "WebSearch": "web_search", "scAnalysis": "sc_analysis", "FinalNode": "chat"}


class UploadedFileInfo(TypedDict, total=False):
    original_path: str
    normalized_path: str
    suffix: str
    kind: InputKind
    converted: bool


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    user_id: str
    session_id: str
    current_turn_id: str
    workspace_settings: dict[str, Any]
    input_kind: InputKind
    intent: Intent
    intent_type: IntentType
    intent_reason: str
    next_node: NextNode
    current_action: NextNode
    current_action_input: dict[str, Any]
    uploaded_files: list[str]
    normalized_files: list[UploadedFileInfo]
    h5ad_files: list[str]
    rag_files: list[str]
    knowledge_base_path: str
    upload_workdir: str
    rag_index_dir: str
    observations: Annotated[list[dict[str, Any]], operator.add]
    tool_results: dict[str, Any]
    current_tool_results: dict[str, Any]
    current_tool_results_turn_id: str
    llm_traces: Annotated[list[dict[str, Any]], operator.add]
    react_steps: Annotated[list[dict[str, Any]], operator.add]
    memory_context: str
    long_term_memories: list[dict[str, Any]]
    steps: Annotated[list[str], operator.add]
    step_records: Annotated[list[dict[str, Any]], operator.add]
    retry_count: int
    final_answer: str


def _call_llm_streaming(prompt: str, max_new_tokens: int = 2048) -> str:
    from .tools.LLM import GenerationConfig, chat_stream

    emit({"node": "FinalNode", "status": "answer_start", "label": "开始生成回答"})
    chunks: list[str] = []
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0, top_p=1.0, do_sample=False)
    for delta in chat_stream(prompt=prompt, gen_config=gen_config):
        text = str(delta or "")
        if text:
            chunks.append(text)
            emit({"node": "FinalNode", "status": "answer_delta", "delta": text})
    final_text = "".join(chunks).strip()
    if not final_text:
        raise RuntimeError("LLM.py 返回空结果。")
    return final_text


def _call_llm(prompt: str, max_new_tokens: int = 2048) -> str:
    from .tools.LLM import GenerationConfig, chat

    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0, top_p=1.0, do_sample=False)
    text = str(chat(prompt=prompt, gen_config=gen_config)).strip()
    if not text:
        raise RuntimeError("LLM.py 返回空结果。")
    return text


def _call_router(prompt: str) -> str:
    from .tools.LLM import GenerationConfig, chat

    emit({"node": "SupervisorNode", "status": "thought", "kind": "router_start", "action": "调用 ReAct Supervisor"})
    output = str(chat(prompt=prompt, gen_config=GenerationConfig(max_new_tokens=256, temperature=0, top_p=1.0, do_sample=False))).strip()
    if not output:
        raise RuntimeError("LLM Router 返回空结果。")
    emit({"node": "SupervisorNode", "status": "thought", "kind": "router_output", "content": output})
    return output


def _input_metadata(state: AgentState) -> dict[str, Any]:
    uploaded_files = list(state.get("uploaded_files") or [])
    normalized_files = list(state.get("normalized_files") or [])
    if uploaded_files and not normalized_files:
        normalized_files = normalize_uploaded_files(uploaded_files, upload_workdir=state.get("upload_workdir", ""))
    text_h5ad_files = extract_h5ad_paths_from_text(last_user_text(state))
    h5ad_files = list(state.get("h5ad_files") or []) or get_h5ad_files(normalized_files)
    for item in text_h5ad_files:
        if item not in h5ad_files:
            h5ad_files.append(item)
    input_kind = state.get("input_kind") or ("h5ad" if h5ad_files else infer_input_kind_by_files(normalized_files, last_user_text(state)))
    rag_files = list(state.get("rag_files") or []) or get_rag_files(normalized_files)
    return {"uploaded_files": uploaded_files, "normalized_files": normalized_files, "input_kind": input_kind or "unknown", "h5ad_files": h5ad_files, "rag_files": rag_files}


def _supervisor_phase(observations: list[dict[str, Any]]) -> SupervisorPhase:
    if not observations:
        return "initial_route"
    return "tool_result_review" if observations[-1].get("ok") else "tool_error_recovery"


def _tool_sets(observations: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    used: list[str] = []
    successful: list[str] = []
    for obs in observations:
        node = str(obs.get("node") or "")
        if node not in {"RAG", "WebSearch", "scAnalysis"}:
            continue
        if node not in used:
            used.append(node)
        if obs.get("ok") and node not in successful:
            successful.append(node)
    return used, successful


def _current_llm_call_count(state: AgentState) -> int:
    return len([item for item in current_turn_llm_traces(state) if item.get("counts_as_llm_call", True)])


def _llm_trace(label: str, *, elapsed: float = 0.0, turn_id: str = "", **extra: Any) -> dict[str, Any]:
    trace = {"label": label, "elapsed_ms": elapsed, "turn_id": turn_id, "counts_as_llm_call": True}
    trace.update({key: value for key, value in extra.items() if value is not None})
    return trace


def _available_actions(metadata: dict[str, Any]) -> list[str]:
    input_kind = str(metadata.get("input_kind") or "unknown")
    actions = ["FinalNode"]
    if metadata.get("h5ad_files") or input_kind in {"text", "mixed", "unknown"}:
        actions.append("scAnalysis")
    if metadata.get("rag_files") or input_kind in {"text", "pdf", "markdown", "mixed", "h5ad"}:
        actions.append("RAG")
    if input_kind in {"text", "mixed", "h5ad"}:
        actions.append("WebSearch")
    return actions


def _parse_supervisor_output(router_output: str, *, metadata: dict[str, Any], available_actions: list[str], successful_tools: list[str], latest_observation: dict[str, Any] | None) -> dict[str, Any]:
    try:
        payload = parse_json_object(router_output)
    except Exception:
        payload = {"thought": "Supervisor 输出不是合法 JSON，进入 FinalNode。", "action": "FinalNode", "action_input": {"query": ""}, "finish": True}
    action = str(payload.get("action") or payload.get("next_node") or "FinalNode")
    latest_node = str((latest_observation or {}).get("node") or "")
    if action not in available_actions:
        action = "FinalNode"
    action_input = payload.get("action_input") if isinstance(payload.get("action_input"), dict) else {}
    if action in successful_tools:
        action = "FinalNode"
    if latest_observation and not latest_observation.get("ok") and action == latest_node:
        action = "FinalNode"
    intent_type = str(payload.get("intent_type") or "").strip()
    if intent_type not in {"professional_qa", "non_professional_qa", "sc_analysis", "deep_sc_analysis", "unclear"}:
        intent_type = {"RAG": "professional_qa", "WebSearch": "non_professional_qa", "scAnalysis": "sc_analysis", "FinalNode": "unclear"}.get(action, "unclear")
    if action == "scAnalysis" and intent_type not in {"sc_analysis", "deep_sc_analysis"}:
        intent_type = "sc_analysis"
    return {
        "thought": str(payload.get("thought") or payload.get("reason") or ""),
        "intent_type": intent_type,
        "intent_reason": str(payload.get("reason") or payload.get("thought") or ""),
        "needs_rag": payload.get("needs_rag"),
        "needs_web_search": payload.get("needs_web_search"),
        "needs_sc_analysis": payload.get("needs_sc_analysis"),
        "needs_pdf_multimodal_analysis": payload.get("needs_pdf_multimodal_analysis"),
        "action": action,
        "action_input": dict(action_input),
        "finish": bool(payload.get("finish") or action == "FinalNode"),
    }


def _audit_initial_web_route(user_input: str) -> tuple[dict[str, Any], str]:
    output = _call_router(build_intent_audit_prompt(user_input))
    try:
        payload = parse_json_object(output)
    except Exception:
        payload = {"intent_type": "unclear", "prefer_rag_first": False, "reason": "审计输出不是合法 JSON。"}
    return payload, output


def _supervisor_decision(state: AgentState, metadata: dict[str, Any]) -> tuple[NextNode, list[dict[str, Any]], dict[str, Any]]:
    import time

    observations = current_turn_observations(state)
    latest_observation = observations[-1] if observations else None
    phase = _supervisor_phase(observations)
    used_tools, successful_tools = _tool_sets(observations)
    available_actions = _available_actions(metadata)
    user_input = last_user_text(state)
    prompt = build_supervisor_prompt(
        phase=phase,
        user_input=user_input,
        metadata=metadata,
        available_actions=available_actions,
        used_tools=used_tools,
        successful_tools=successful_tools,
        latest_observation=latest_observation,
        observations=observations,
        tool_results=current_turn_tool_results(state),
        workspace_settings=dict(state.get("workspace_settings") or {}),
        memory_context=str(state.get("memory_context") or ""),
    )
    started_at = time.perf_counter()
    router_output = _call_router(prompt)
    router_elapsed = elapsed_ms(started_at)
    decision = _parse_supervisor_output(
        router_output,
        metadata=metadata,
        available_actions=available_actions,
        successful_tools=successful_tools,
        latest_observation=latest_observation,
    )
    if phase == "initial_route" and metadata.get("h5ad_files") and "scAnalysis" in available_actions and decision.get("action") != "scAnalysis":
        decision.update({
            "action": "scAnalysis",
            "intent_type": "sc_analysis",
            "needs_sc_analysis": True,
            "needs_rag": False,
            "needs_web_search": False,
            "intent_reason": "检测到用户指定 h5ad 数据文件，按单细胞分析流程处理。",
            "thought": f"{decision.get('thought') or ''} 检测到 h5ad 数据文件，切换到 scAnalysis。".strip(),
        })
    audit_trace: dict[str, Any] | None = None
    if phase == "initial_route" and decision.get("action") == "WebSearch" and "RAG" in available_actions and not metadata.get("h5ad_files"):
        audit_started = time.perf_counter()
        audit_payload, audit_output = _audit_initial_web_route(user_input)
        audit_trace = {
            "label": "SupervisorNode.intent_audit",
            "counts_as_llm_call": True,
            "llm_call_type": "intent_routing_audit",
            "phase": phase,
            "response": audit_output,
            "parsed_intent_type": audit_payload.get("intent_type"),
            "prefer_rag_first": audit_payload.get("prefer_rag_first"),
            "reason": audit_payload.get("reason"),
            "elapsed_ms": elapsed_ms(audit_started),
            "turn_id": current_turn_id(state),
        }
        if str(audit_payload.get("intent_type") or "") == "professional_qa" and bool(audit_payload.get("prefer_rag_first", True)):
            decision.update({
                "action": "RAG",
                "intent_type": "professional_qa",
                "needs_rag": True,
                "needs_web_search": "decided_after_rag",
                "intent_reason": str(audit_payload.get("reason") or decision.get("intent_reason") or ""),
                "thought": f"{decision.get('thought') or ''} 语义审计判定为专业问题，先走本地 RAG。".strip(),
            })
    if phase == "initial_route" and decision.get("action") == "WebSearch" and not metadata.get("h5ad_files"):
        decision.update({"intent_type": "non_professional_qa", "needs_rag": False, "needs_web_search": True})
    action_input = dict(decision.get("action_input") or {})
    action_input["query"] = user_input
    if decision.get("action") in {"RAG", "WebSearch"}:
        max_queries = int(dict(state.get("workspace_settings") or {}).get("multi_query_count") or 1)
        action_input["queries"] = sanitize_action_queries(user_input, list(action_input.get("queries") or []), max_count=max_queries)
    decision["action_input"] = action_input
    next_node = decision["action"]
    trace = {
        "label": "SupervisorNode",
        "counts_as_llm_call": True,
        "llm_call_type": "intent_routing",
        "phase": phase,
        "response": router_output,
        "parsed_node": next_node,
        "intent_type": decision.get("intent_type"),
        "intent_reason": decision.get("intent_reason"),
        "thought": decision["thought"],
        "available_actions": available_actions,
        "used_tools": used_tools,
        "successful_tools": successful_tools,
        "action_input": decision["action_input"],
        "elapsed_ms": router_elapsed,
        "turn_id": current_turn_id(state),
    }
    traces = [trace]
    if audit_trace:
        traces.append(audit_trace)
    react_step = {
        "phase": phase,
        "thought": decision["thought"],
        "action": next_node,
        "action_input": decision["action_input"],
        "intent_type": decision.get("intent_type"),
        "intent_reason": decision.get("intent_reason"),
        "observation_count": len(observations),
        "turn_id": current_turn_id(state),
    }
    emit({
        "node": "SupervisorNode",
        "status": "thought",
        "kind": "supervisor_decision",
        "phase": phase,
        "thought": decision["thought"],
        "intent_type": decision.get("intent_type"),
        "intent_reason": decision.get("intent_reason"),
        "available_actions": available_actions,
        "used_tools": used_tools,
        "action": next_node,
        "next_node": next_node,
        "action_input": decision["action_input"],
    })
    return next_node, traces, react_step  # type: ignore[return-value]


def _initial_state(
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
        "current_action": "FinalNode",
        "current_action_input": {},
        "intent_type": "unclear",
        "intent_reason": "",
        "react_steps": [],
        "memory_context": "",
        "long_term_memories": [],
        "steps": [],
        "step_records": [],
        "retry_count": 0,
    }


def _runtime_config(memory_manager: Any, user_id: str, session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": memory_manager.thread_id(user_id, session_id), "session_id": session_id}}


def _store_turn(memory_manager: Any, user_id: str, session_id: str, user_input: str, final_answer: str, state: dict[str, Any], workspace_settings: dict[str, Any] | None) -> None:
    memory_manager.store_turn(user_id=user_id, session_id=session_id, user_input=user_input, final_answer=final_answer, state=state, workspace_settings=workspace_settings)


def memory_node(state: AgentState) -> dict[str, Any]:
    emit({"node": "MemoryNode", "status": "start"})
    patch = get_memory_manager().prepare_state(state)
    memory_debug = dict(patch.get("memory_debug") or {})
    emit({
        "node": "MemoryNode",
        "status": "end",
        "long_term_memory_count": len(patch.get("long_term_memories") or []),
        "has_memory_context": bool(str(patch.get("memory_context") or "").strip()),
        "current_query": memory_debug.get("current_query", ""),
        "selected_memories": memory_debug.get("selected_memories", []),
        "filtered_memories": memory_debug.get("filtered_memories", []),
        "session_summary_categories": memory_debug.get("session_summary_categories", []),
        "final_prompt_memory_block": memory_debug.get("final_prompt_memory_block", ""),
    })
    return patch


def supervisor_node(state: AgentState) -> dict[str, Any]:
    import time

    emit({"node": "SupervisorNode", "status": "start"})
    started_at = time.perf_counter()
    metadata = _input_metadata(state)
    if len(current_turn_steps(state)) >= MAX_GRAPH_STEPS:
        next_node: NextNode = "FinalNode"
        react_step = {"thought": "达到最大图执行步数，进入最终回答。", "action_input": {"query": ""}, "intent_type": state.get("intent_type", "unclear"), "intent_reason": state.get("intent_reason", "")}
        router_traces: list[dict[str, Any]] = [{"label": "Supervisor fallback", "parsed_node": next_node, "turn_id": current_turn_id(state), "counts_as_llm_call": False}]
    elif _current_llm_call_count(state) >= MAX_LLM_GENERATION_CALLS - 1:
        next_node = "FinalNode"
        react_step = {"thought": "接近单轮 LLM 调用上限，进入最终回答。", "action_input": {"query": ""}, "intent_type": state.get("intent_type", "unclear"), "intent_reason": state.get("intent_reason", "")}
        router_traces = [{"label": "Supervisor budget guard", "parsed_node": next_node, "turn_id": current_turn_id(state), "counts_as_llm_call": False}]
    else:
        next_node, router_traces, react_step = _supervisor_decision(state, metadata)
    tool_results_for_route = current_turn_tool_results(state)
    sc_result_for_route = tool_results_for_route.get("sc_analysis") if isinstance(tool_results_for_route, dict) else None
    if (
        next_node == "FinalNode"
        and isinstance(sc_result_for_route, dict)
        and dict(sc_result_for_route.get("meta") or {}).get("deep_analysis")
        and "rag" not in tool_results_for_route
        and "RAG" in _available_actions(metadata)
        and _current_llm_call_count(state) < MAX_LLM_GENERATION_CALLS - 1
    ):
        next_node = "RAG"
        react_step = {
            **react_step,
            "thought": "深入单细胞分析需要专业背景补充，先调用本地 RAG。",
            "action": "RAG",
            "action_input": {"query": last_user_text(state)},
            "intent_type": "deep_sc_analysis",
            "intent_reason": "scAnalysis 已生成深入分析报告，需要结合本地知识库补充专业背景。",
        }
    intent = NODE_INTENT.get(next_node, "unknown")
    intent_type = str(react_step.get("intent_type") or state.get("intent_type") or "unclear")
    intent_reason = str(react_step.get("intent_reason") or react_step.get("thought") or state.get("intent_reason") or "")
    sc_result = current_turn_tool_results(state).get("sc_analysis") if isinstance(current_turn_tool_results(state), dict) else None
    if next_node == "FinalNode" and isinstance(sc_result, dict):
        sc_meta = dict(sc_result.get("meta") or {})
        intent_type = "deep_sc_analysis" if sc_meta.get("deep_analysis") else "sc_analysis"
        intent_reason = intent_reason or "当前轮已完成单细胞分析，进入最终总结。"
    emit({"node": "SupervisorNode", "status": "route", "phase": react_step.get("phase", "forced_final"), "input_kind": metadata["input_kind"], "intent": intent, "intent_type": intent_type, "intent_reason": intent_reason, "next_node": next_node, "action_input": react_step.get("action_input") or {}, "uploaded_files": metadata["uploaded_files"], "normalized_files": metadata["normalized_files"]})
    return {
        "input_kind": metadata["input_kind"],
        "intent": intent,
        "intent_type": intent_type,
        "intent_reason": intent_reason,
        "next_node": next_node,
        "current_action": next_node,
        "current_action_input": dict(react_step.get("action_input") or {}),
        "normalized_files": metadata["normalized_files"],
        "h5ad_files": metadata["h5ad_files"],
        "rag_files": metadata["rag_files"],
        "knowledge_base_path": state.get("knowledge_base_path", KNOWLEDGE_BASE_PATH),
        "steps": ["SupervisorNode"],
        "step_records": [build_step_record("SupervisorNode", elapsed_ms(started_at), tool_name=TOOL_NAME.get(next_node, ""), detail=str(react_step.get("thought") or ""), turn_id=current_turn_id(state))],
        "llm_traces": router_traces,
        "react_steps": [react_step],
        "retry_count": state.get("retry_count", 0),
    }


def rag_node(state: AgentState) -> dict[str, Any]:
    import time
    from .tools.RAG import build_rag_index, run_rag

    emit({"node": "RAG", "status": "start"})
    started_at = time.perf_counter()
    action_input = dict(state.get("current_action_input") or {})
    query = str(action_input.get("query") or last_user_text(state))
    queries = action_input.get("queries") if isinstance(action_input.get("queries"), list) else []
    settings = dict(state.get("workspace_settings") or {})
    queries = sanitize_action_queries(query, queries, max_count=int(settings.get("multi_query_count") or 1))
    knowledge_base_path = state.get("knowledge_base_path", KNOWLEDGE_BASE_PATH)
    rag_index_dir = state.get("rag_index_dir", "")
    rag_files = normalize_rag_files_for_base(list(state.get("rag_files") or []), knowledge_base_path)
    index_root = Path(rag_index_dir).expanduser() if rag_index_dir else None
    index_missing = not index_root or not all((index_root / item).exists() for item in ("chunks.jsonl", "bm25.pkl", "embeddings.npy"))
    if rag_files or index_missing:
        build_rag_index(knowledge_base_path=knowledge_base_path, index_dir=rag_index_dir, files=rag_files or None, clean=True)
    result = run_rag(query=query, queries=queries, multi_query_count=settings.get("multi_query_count"), knowledge_base_path=knowledge_base_path, index_dir=rag_index_dir, files=rag_files)
    content = tool_result_text(result)
    confidence = rag_confidence_from_result(result)
    threshold = float(settings.get("local_source_min_score", DEFAULT_RAG_CONFIDENCE_THRESHOLD))
    result_meta = result.get("meta") if isinstance(result, dict) else {}
    evidence_sufficient = bool((result_meta if isinstance(result_meta, dict) else {}).get("evidence_sufficient", confidence >= threshold)) and confidence >= threshold
    retrieval_trace = dict((result_meta if isinstance(result_meta, dict) else {}).get("retrieval_trace") or {})
    retrieval_trace.update({
        "rag_confidence": round(float(confidence), 6),
        "rag_confidence_threshold": threshold,
        "rag_confidence_result": "sufficient" if evidence_sufficient else "insufficient",
        "llm_call_count": _current_llm_call_count(state),
    })
    if isinstance(result, dict):
        result["retrieval_trace"] = retrieval_trace
        result.setdefault("meta", {})["retrieval_trace"] = retrieval_trace
        result["meta"]["evidence_sufficient"] = evidence_sufficient
    return tool_node_result(state, node="RAG", tool_key="rag", tool_name=TOOL_NAME["RAG"], ok=bool(content.strip()), result=result, elapsed=elapsed_ms(started_at), content=content, metadata={"knowledge_base_path": knowledge_base_path, "index_dir": rag_index_dir, "files": rag_files, "confidence": confidence, "confidence_threshold": threshold, "evidence_sufficient": evidence_sufficient, "retrieval_trace": retrieval_trace})


def web_search_node(state: AgentState) -> dict[str, Any]:
    import time
    from .tools.Web import web_search

    emit({"node": "WebSearch", "status": "start"})
    started_at = time.perf_counter()
    action_input = dict(state.get("current_action_input") or {})
    query = str(action_input.get("query") or last_user_text(state))
    queries = action_input.get("queries") if isinstance(action_input.get("queries"), list) else []
    settings = dict(state.get("workspace_settings") or {})
    queries = sanitize_action_queries(query, queries, max_count=int(settings.get("multi_query_count") or 1))
    result = web_search(query=query, queries=queries, multi_query_count=settings.get("multi_query_count"), k=6, return_json=True)
    rag_result = current_turn_tool_results(state).get("rag") if isinstance(current_turn_tool_results(state), dict) else None
    triggered_after_low_confidence_rag = bool(isinstance(rag_result, dict) and not dict(rag_result.get("meta") or {}).get("evidence_sufficient", False))
    if isinstance(result, dict):
        result.setdefault("meta", {})["triggered_after_low_confidence_rag"] = triggered_after_low_confidence_rag
        result["meta"]["final_information_source"] = "local RAG + web search" if triggered_after_low_confidence_rag else "web search"
    content = tool_result_text(result)
    return tool_node_result(state, node="WebSearch", tool_key="web_search", tool_name=TOOL_NAME["WebSearch"], ok=bool(content.strip()), result=result, elapsed=elapsed_ms(started_at), content=content, metadata={"query": query, "triggered_after_low_confidence_rag": triggered_after_low_confidence_rag})


def _analyze_pdf_report_with_llm(pdf_path: str, user_text: str, report_context: str) -> str:
    import fitz

    from .tools.LLM import GenerationConfig, chat

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF report not found: {path}")
    image_dir = path.parent / "pdf_pages_for_llm"
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[str] = []
    document = fitz.open(path)
    try:
        for page_index in range(min(4, document.page_count)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_path = image_dir / f"page_{page_index + 1:02d}.png"
            pixmap.save(image_path)
            images.append(str(image_path))
    finally:
        document.close()
    if not images:
        raise RuntimeError(f"PDF report has no renderable pages: {path}")
    prompt = build_pdf_report_interpretation_prompt(user_text=user_text, report_context=report_context)
    return chat(prompt=prompt, images=images, gen_config=GenerationConfig(max_new_tokens=1024, temperature=0, top_p=1.0, do_sample=False))


def sc_analysis_node(state: AgentState) -> dict[str, Any]:
    import time
    from .tools.SC import run_sc_analysis

    emit({"node": "scAnalysis", "status": "start"})
    started_at = time.perf_counter()
    h5ad_files = list(state.get("h5ad_files") or [])
    action_input = dict(state.get("current_action_input") or {})
    h5ad_path = h5ad_files[0] if h5ad_files else str(action_input.get("h5ad_path") or "")
    result = run_sc_analysis({"user_id": state.get("user_id", "anonymous"), "session_id": state.get("session_id", "default"), "user_text": last_user_text(state), "h5ad_path": h5ad_path, "output_dir": SC_OUTPUT_DIR, "workspace_settings": dict(state.get("workspace_settings") or {})})
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    sc_llm_traces: list[dict[str, Any]] = []
    if isinstance(result, dict):
        meta = dict(result.get("meta") or {})
        function_call_count = int(meta.get("function_call_llm_count") or 0)
        for index in range(function_call_count):
            sc_llm_traces.append(_llm_trace("scAnalysis.configure_sc_analysis", turn_id=current_turn_id(state), llm_call_type="function_calling", function_call_name=meta.get("function_call_name") or "configure_sc_analysis", function_call_arguments=meta.get("function_call_arguments"), call_index=index + 1))
        analysis_params = dict(result.get("analysis_params") or {})
        pdf_report = result.get("pdf_report") if isinstance(result.get("pdf_report"), dict) else {}
        if analysis_params.get("deep_analysis") and isinstance(pdf_report, dict) and pdf_report.get("path"):
            sc_llm_traces.append(_llm_trace("scAnalysis.pdf_multimodal_interpretation", turn_id=current_turn_id(state), llm_call_type="pdf_multimodal_analysis", pdf_path=str(pdf_report["path"])))
            try:
                pdf_interpretation = _analyze_pdf_report_with_llm(str(pdf_report["path"]), last_user_text(state), str(result.get("report_context") or ""))
            except Exception as exc:
                pdf_interpretation = f"PDF 多模态解读失败：{exc}"
            result["pdf_interpretation"] = pdf_interpretation
            result["answer"] = str(result.get("answer") or result.get("message") or "").strip() + "\n\nPDF 图表解读：\n" + pdf_interpretation
            result["message"] = result["answer"]
            result["local_answer"] = result["answer"]
    content = tool_result_text(result)
    ok = bool(content.strip()) and (not isinstance(result, dict) or str(result.get("status") or "ok") == "ok")
    patch = tool_node_result(state, node="scAnalysis", tool_key="sc_analysis", tool_name=TOOL_NAME["scAnalysis"], ok=ok, result=result, elapsed=elapsed_ms(started_at), content=content, metadata={"h5ad_path": h5ad_path})
    if isinstance(result, dict):
        patch["intent_type"] = "deep_sc_analysis" if dict(result.get("meta") or {}).get("deep_analysis") else "sc_analysis"
    if sc_llm_traces:
        patch["llm_traces"] = sc_llm_traces
    return patch


def final_node(state: AgentState) -> dict[str, Any]:
    import time

    emit({"node": "FinalNode", "status": "start"})
    started_at = time.perf_counter()
    memory_context = str(state.get("memory_context") or "")
    tool_results = current_turn_tool_results(state)
    settings = dict(state.get("workspace_settings") or {})
    final_max_tokens = min(int(settings.get("max_new_tokens") or 2048), 1800)
    final_prompt = build_final_prompt(user_input=last_user_text(state), observations=current_turn_observations(state), tool_results=tool_results, steps=current_turn_steps(state), memory_context=memory_context)
    llm_traces: list[dict[str, Any]] = []
    if _current_llm_call_count(state) <= MAX_LLM_GENERATION_CALLS - 2:
        draft_started = time.perf_counter()
        draft_answer = _call_llm(final_prompt, max_new_tokens=min(final_max_tokens, 1200))
        llm_traces.append(_llm_trace("FinalNodeDraft", elapsed=elapsed_ms(draft_started), turn_id=current_turn_id(state), llm_call_type="final_answer_draft"))
        revision_prompt = build_final_revision_prompt(user_input=last_user_text(state), tool_results=tool_results, draft_answer=draft_answer, memory_context=memory_context)
        final_answer = _call_llm_streaming(revision_prompt, max_new_tokens=final_max_tokens)
    else:
        final_answer = _call_llm_streaming(final_prompt, max_new_tokens=final_max_tokens)
    emit({"node": "FinalNode", "status": "end"})
    elapsed = elapsed_ms(started_at)
    llm_traces.append(_llm_trace("FinalNode", elapsed=elapsed, turn_id=current_turn_id(state), llm_call_type="final_answer"))
    return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)], "steps": ["FinalNode"], "step_records": [build_step_record("FinalNode", elapsed, turn_id=current_turn_id(state))], "llm_traces": llm_traces}


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
    builder.add_conditional_edges("SupervisorNode", lambda state: state.get("next_node", "FinalNode"), {"RAG": "RAG", "WebSearch": "WebSearch", "scAnalysis": "scAnalysis", "FinalNode": "FinalNode"})
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
    memory_manager = get_memory_manager()
    initial = _initial_state(user_input=user_input, user_id=user_id, session_id=session_id, uploaded_files=uploaded_files, knowledge_base_path=knowledge_base_path, upload_workdir=upload_workdir, rag_index_dir=rag_index_dir, workspace_settings=workspace_settings)
    config = _runtime_config(memory_manager, user_id, session_id)
    result = get_graph().invoke(initial, config=config)
    _store_turn(memory_manager, user_id, session_id, user_input, str(result.get("final_answer") or ""), result, workspace_settings)
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
    event_callback: Optional[Any] = None,
):
    memory_manager = get_memory_manager()
    initial = _initial_state(user_input=user_input, user_id=user_id, session_id=session_id, uploaded_files=uploaded_files, knowledge_base_path=knowledge_base_path, upload_workdir=upload_workdir, rag_index_dir=rag_index_dir, workspace_settings=workspace_settings)
    config = _runtime_config(memory_manager, user_id, session_id)
    compiled_graph = get_graph()
    with stream_event_callback(event_callback):
        for event in compiled_graph.stream(initial, config=config, stream_mode=["updates"]):
            yield event
    snapshot = compiled_graph.get_state(config)
    values = dict(getattr(snapshot, "values", {}) or {})
    _store_turn(memory_manager, user_id, session_id, user_input, str(values.get("final_answer") or ""), values, workspace_settings)
    yield ("final_state", values)
