from __future__ import annotations

import asyncio
import base64
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Any, AsyncIterator, Awaitable, Callable, TypedDict

from .memory import BuiltMemoryContext, MemoryManager, SemanticMemoryHit
from .prompts import (
    build_augmented_analysis_messages,
    build_followup_contextualize_messages,
    build_react_decision_repair_messages,
    build_react_deliberation_messages,
    build_react_final_answer_messages,
)
from .tools.llm.client import (
    begin_llm_trace_session,
    end_llm_trace_session,
    local_chat_completion,
    local_chat_completion_stream,
)
from .tools.rag.config import get_config
from .tools.tools import (
    run_general_chat,
    run_general_chat_with_memory,
    run_local_knowledge_qa,
    run_single_cell_analysis,
    run_web_search,
)


class IntentType(StrEnum):
    LOCAL_KNOWLEDGE_QA = "local_knowledge_qa"
    WEB_SEARCH = "web_search"
    SINGLE_CELL_ANALYSIS = "single_cell_analysis"
    AUGMENTED_ANALYSIS = "augmented_analysis"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class ToolName(StrEnum):
    LOCAL_KNOWLEDGE_BASE = "local_knowledge_base"
    WEB_SEARCH = "web_search"
    SINGLE_CELL_PIPELINE = "single_cell_pipeline"
    DIRECT_LLM = "direct_llm"


@dataclass(slots=True)
class UploadedAsset:
    name: str
    kind: str
    content_type: str
    size_bytes: int
    path: str


@dataclass(slots=True)
class AgentInput:
    user_id: str
    session_id: str
    user_text: str
    attachments: list[UploadedAsset] = field(default_factory=list)
    workspace_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentStep:
    step_id: str
    description: str
    tool_name: ToolName | None = None
    status: str = "pending"


@dataclass(slots=True)
class AgentDecision:
    intent: IntentType
    reason: str
    selected_tools: list[ToolName] = field(default_factory=list)
    execution_steps: list[AgentStep] = field(default_factory=list)
    tool_result: dict[str, Any] = field(default_factory=dict)
    llm_traces: list[dict[str, Any]] = field(default_factory=list)


class AgentWorkflowState(TypedDict, total=False):
    agent_input: dict[str, Any]
    resolved_user_text: str
    attachment_lines: list[str]
    transcript: str
    memory_profile: dict[str, Any]
    memory_semantic_memories: list[dict[str, Any]]
    memory_recent_messages: list[dict[str, Any]]
    memory_short_summary: str
    memory_task_state: dict[str, Any]
    intent: str
    route_plan: str
    route_reason: str
    next_action: str
    next_tool_name: str
    selected_tool_names: list[str]
    raw_tool_results: dict[str, dict[str, Any]]
    tool_result: dict[str, Any]
    draft_final_answer: str
    final_answer: str
    deliberation_count: int


ToolRunner = Callable[[AgentInput], Awaitable[dict[str, Any]]]


TOOL_NAME_TO_INTENT: dict[str, IntentType] = {
    "general_chat": IntentType.GENERAL_CHAT,
    "local_knowledge_qa": IntentType.LOCAL_KNOWLEDGE_QA,
    "web_search": IntentType.WEB_SEARCH,
    "single_cell_analysis": IntentType.SINGLE_CELL_ANALYSIS,
    "augmented_analysis": IntentType.AUGMENTED_ANALYSIS,
}

INTENT_TO_TOOL_NAME: dict[IntentType, ToolName] = {
    IntentType.GENERAL_CHAT: ToolName.DIRECT_LLM,
    IntentType.LOCAL_KNOWLEDGE_QA: ToolName.LOCAL_KNOWLEDGE_BASE,
    IntentType.WEB_SEARCH: ToolName.WEB_SEARCH,
    IntentType.SINGLE_CELL_ANALYSIS: ToolName.SINGLE_CELL_PIPELINE,
    IntentType.AUGMENTED_ANALYSIS: ToolName.SINGLE_CELL_PIPELINE,
}

TOOL_DISPLAY_NAMES: dict[ToolName, str] = {
    ToolName.LOCAL_KNOWLEDGE_BASE: "本地知识库问答",
    ToolName.WEB_SEARCH: "网页搜索",
    ToolName.SINGLE_CELL_PIPELINE: "单细胞分析流程",
    ToolName.DIRECT_LLM: "基础对话模型",
}

EXECUTABLE_TOOL_NAMES = (
    "general_chat",
    "local_knowledge_qa",
    "web_search",
    "single_cell_analysis",
)
TOOL_RUNNERS: dict[str, ToolRunner] = {
    "general_chat": run_general_chat,
    "local_knowledge_qa": run_local_knowledge_qa,
    "web_search": run_web_search,
    "single_cell_analysis": run_single_cell_analysis,
}
ROUTE_DECISION_TEMPERATURE = 0.0
MAX_REACT_STEPS = 6
DEFAULT_ROUTE_REASON = "已根据问题内容自动完成处理。"
DEFAULT_EMPTY_ANSWER = "模型未返回内容。"

_MEMORY_MANAGER = MemoryManager(get_config())
AGENT_EVENT_EMITTER: ContextVar[
    Callable[[dict[str, Any]], Awaitable[None]] | None
] = ContextVar("agent_event_emitter", default=None)


class AgentRuntime:
    # 这个类只承接节点实现与运行时细节，图结构本身留在 agent.py。
    def build_decision(
        self,
        result: dict[str, Any],
        llm_traces: list[dict[str, Any]],
    ) -> AgentDecision:
        intent = _coerce_intent(result.get("intent"))
        selected_tools = _resolve_selected_tools(
            intent=intent,
            selected_tool_names=result.get("selected_tool_names") or [],
        )
        reason = str(result.get("route_reason") or "").strip() or DEFAULT_ROUTE_REASON
        tool_result = _normalize_tool_result(
            result.get("tool_result") or {},
            str(result.get("final_answer") or "").strip(),
        )
        return AgentDecision(
            intent=intent,
            reason=reason,
            selected_tools=selected_tools,
            execution_steps=_build_execution_steps(
                selected_tools=selected_tools,
                route_reason=reason,
                final_answer=tool_result.get("answer") or "",
            ),
            tool_result=tool_result,
            llm_traces=llm_traces,
        )

    def make_tool_node(
        self, tool_name: str
    ) -> Callable[[AgentWorkflowState], Awaitable[AgentWorkflowState]]:
        async def _tool_node(state: AgentWorkflowState) -> AgentWorkflowState:
            return await self.execute_tool_node(state, tool_name)

        return _tool_node

    async def prepare_context_node(
        self, state: AgentWorkflowState
    ) -> AgentWorkflowState:
        agent_input = _deserialize_agent_input(state["agent_input"])
        attachment_lines = _build_attachment_lines(agent_input.attachments)
        memory_context = _load_memory_context(agent_input)
        await _emit_agent_event(
            "status",
            {
                "stage": "prepare_context",
                "message": "已载入会话上下文与记忆。",
            },
        )
        transcript = _build_initial_transcript(
            user_text=agent_input.user_text,
            attachment_lines=attachment_lines,
            memory_context=memory_context,
            workspace_settings=agent_input.workspace_settings,
        )
        return {
            "resolved_user_text": agent_input.user_text,
            "attachment_lines": attachment_lines,
            "transcript": transcript,
            "memory_profile": memory_context.profile,
            "memory_semantic_memories": [
                item.to_dict() for item in memory_context.semantic_memories
            ],
            "memory_recent_messages": [
                item.to_dict() for item in memory_context.recent_messages
            ],
            "memory_short_summary": memory_context.short_summary,
            "memory_task_state": memory_context.task_state,
            "intent": "",
            "route_plan": "",
            "route_reason": "",
            "next_action": "",
            "next_tool_name": "",
            "selected_tool_names": [],
            "raw_tool_results": {},
            "tool_result": {},
            "draft_final_answer": "",
            "final_answer": "",
            "deliberation_count": 0,
        }

    async def contextualize_query_node(
        self, state: AgentWorkflowState
    ) -> AgentWorkflowState:
        agent_input = _deserialize_agent_input(state["agent_input"])
        recent_messages = state.get("memory_recent_messages") or []
        short_summary = str(state.get("memory_short_summary") or "")
        resolved_user_text = await _resolve_user_text_with_memory(
            user_text=agent_input.user_text,
            recent_messages=recent_messages,
            short_summary=short_summary,
        )
        if resolved_user_text != agent_input.user_text:
            await _emit_agent_event(
                "status",
                {
                    "stage": "contextualize_query",
                    "message": f"已将追问补全为：{resolved_user_text}",
                },
            )
        transcript = _build_initial_transcript(
            user_text=agent_input.user_text,
            resolved_user_text=resolved_user_text,
            attachment_lines=state.get("attachment_lines") or ["- 无附件"],
            memory_context=_memory_context_from_state(state, agent_input),
            workspace_settings=agent_input.workspace_settings,
        )
        return {
            "resolved_user_text": resolved_user_text,
            "transcript": transcript,
        }

    async def deliberate_node(
        self, state: AgentWorkflowState
    ) -> AgentWorkflowState:
        agent_input = _deserialize_agent_input(state["agent_input"])
        resolved_user_text = str(state.get("resolved_user_text") or agent_input.user_text)
        attachment_lines = state.get("attachment_lines") or ["- 无附件"]
        transcript = str(state.get("transcript") or "")
        deliberation_count = int(state.get("deliberation_count") or 0) + 1
        raw_tool_results = state.get("raw_tool_results") or {}
        selected_tool_names = state.get("selected_tool_names") or []

        if deliberation_count > MAX_REACT_STEPS:
            decision = {
                "intent": str(state.get("intent") or IntentType.GENERAL_CHAT.value),
                "plan": "已达到最大 ReAct 步数，基于现有 observation 收敛。",
                "action": "final",
                "tool_name": "",
                "reason": "已达到最大 ReAct 步数，停止继续调用工具。",
                "answer": "",
            }
        else:
            decision = await _run_route_decision(
                user_text=resolved_user_text,
                transcript=transcript,
                attachment_lines=attachment_lines,
            )

        decision = _normalize_react_decision_for_graph(
            decision=decision,
            selected_tool_names=selected_tool_names,
            raw_tool_results=raw_tool_results,
        )
        decision = _force_attachment_aware_general_chat_if_needed(
            decision=decision,
            agent_input=agent_input,
            raw_tool_results=raw_tool_results,
        )
        intent = _coerce_intent(decision.get("intent"))
        next_action = str(decision.get("action") or "").strip().lower()
        next_tool_name = str(decision.get("tool_name") or "").strip()
        updated_transcript = _append_agent_decision(
            transcript=transcript,
            decision=decision,
            deliberation_count=deliberation_count,
        )
        await _emit_agent_event(
            "thought",
            {
                "step": deliberation_count,
                "intent": intent.value,
                "plan": str(decision.get("plan") or "").strip(),
                "action": next_action,
                "tool_name": next_tool_name,
                "reason": str(decision.get("reason") or "").strip(),
            },
        )
        return {
            "intent": intent.value,
            "route_plan": str(decision.get("plan") or "").strip(),
            "route_reason": str(decision.get("reason") or "").strip(),
            "next_action": next_action,
            "next_tool_name": next_tool_name,
            "draft_final_answer": (
                str(decision.get("answer") or "").strip()
                if next_action == "final"
                else ""
            ),
            "transcript": updated_transcript,
            "deliberation_count": deliberation_count,
        }

    async def execute_tool_node(
        self,
        state: AgentWorkflowState,
        tool_name: str,
    ) -> AgentWorkflowState:
        agent_input = _build_effective_agent_input(
            _deserialize_agent_input(state["agent_input"]),
            str(state.get("resolved_user_text") or ""),
        )
        await _emit_agent_event(
            "tool_start",
            {
                "tool_name": tool_name,
                "label": _display_executable_tool_name(tool_name),
            },
        )
        if tool_name == "general_chat":
            result = await run_general_chat_with_memory(
                agent_input,
                recent_messages=state.get("memory_recent_messages") or [],
                short_summary=str(state.get("memory_short_summary") or ""),
                profile=state.get("memory_profile") or {},
            )
        else:
            result = await TOOL_RUNNERS[tool_name](agent_input)
        if tool_name == "web_search":
            result = _filter_web_result_by_score(
                result,
                min_score=_get_web_source_min_score(agent_input.workspace_settings),
            )
        transcript = _append_tool_observation(
            str(state.get("transcript") or ""),
            tool_name,
            result,
        )
        await _emit_agent_event(
            "tool_result",
            {
                "tool_name": tool_name,
                "label": _display_executable_tool_name(tool_name),
                "status": str(result.get("status") or ""),
                "summary": _summarize_single_tool_result(result),
            },
        )
        return {
            "selected_tool_names": _append_tool_name(
                state.get("selected_tool_names"), tool_name
            ),
            "raw_tool_results": _merge_tool_results(
                state.get("raw_tool_results"),
                tool_name,
                result,
            ),
            "tool_result": result,
            "transcript": transcript,
        }

    async def finalize_node(
        self, state: AgentWorkflowState
    ) -> AgentWorkflowState:
        agent_input = _deserialize_agent_input(state["agent_input"])
        raw_tool_results = state.get("raw_tool_results") or {}
        single_cell_result = dict(raw_tool_results.get("single_cell_analysis") or {})
        rag_result = dict(raw_tool_results.get("local_knowledge_qa") or {})
        web_result = dict(raw_tool_results.get("web_search") or {})
        web_result = _filter_web_result_by_score(
            web_result,
            min_score=_get_web_source_min_score(agent_input.workspace_settings),
        )
        intent = _coerce_intent(state.get("intent"))
        draft_final_answer = str(state.get("draft_final_answer") or "").strip()
        transcript = str(state.get("transcript") or "")
        tool_result = dict(state.get("tool_result") or {})
        final_answer = draft_final_answer
        normalized_result: dict[str, Any]
        answer_streamed = False

        if (
            intent is IntentType.AUGMENTED_ANALYSIS
            and single_cell_result
            and str(single_cell_result.get("status") or "").lower() in {"", "ok"}
        ):
            final_answer = await _run_augmented_final_answer(
                agent_input=agent_input,
                single_cell_result=single_cell_result,
                rag_result=rag_result,
                web_result=web_result,
            )
            answer_streamed = True
            if not final_answer:
                final_answer = draft_final_answer or await _run_react_final_answer(transcript)
                answer_streamed = True
            combined_result = {
                "status": "ok",
                "observation": {
                    "single_cell_analysis": single_cell_result,
                    "local_knowledge_qa": rag_result,
                    "web_search": web_result,
                },
                "artifacts": single_cell_result.get("artifacts") or [],
                "references": _merge_references(
                    rag_result.get("references"),
                    web_result.get("references"),
                ),
            }
            normalized_result = _normalize_tool_result(combined_result, final_answer)
        else:
            if not draft_final_answer:
                if str(tool_result.get("status") or "").lower() not in {"", "ok"}:
                    draft_final_answer = str(
                        tool_result.get("answer")
                        or tool_result.get("message")
                        or "工具执行失败。"
                    ).strip()
                elif _should_use_direct_tool_answer_in_finalize(
                    selected_tool_names=state.get("selected_tool_names") or [],
                    raw_tool_results=raw_tool_results,
                    tool_result=tool_result,
                ):
                    draft_final_answer = str(
                        tool_result.get("answer")
                        or tool_result.get("message")
                        or ""
                    ).strip()
                elif state.get("selected_tool_names"):
                    draft_final_answer = await _run_react_final_answer(transcript)
                    answer_streamed = True
            final_answer = draft_final_answer
            if rag_result and web_result:
                tool_result = _combine_external_knowledge_results(
                    rag_result=rag_result,
                    web_result=web_result,
                )
            normalized_result = _normalize_tool_result(tool_result, final_answer)

        if final_answer and not answer_streamed:
            await _emit_answer_text(final_answer)

        _persist_memory(
            agent_input=agent_input,
            state=state,
            tool_result=normalized_result,
            final_answer=final_answer,
        )
        return {
            "tool_result": normalized_result,
            "final_answer": final_answer,
        }

    @staticmethod
    def deliberation_edge(state: AgentWorkflowState) -> str:
        if str(state.get("next_action") or "").strip().lower() != "tool":
            return "finalize"
        tool_name = str(state.get("next_tool_name") or "").strip()
        return tool_name if tool_name in EXECUTABLE_TOOL_NAMES else "finalize"


async def emit_agent_event(event_type: str, data: dict[str, Any]) -> None:
    await _emit_agent_event(event_type, data)


def serialize_agent_input(agent_input: AgentInput) -> dict[str, Any]:
    return _serialize_agent_input(agent_input)


def deserialize_agent_input(payload: dict[str, Any]) -> AgentInput:
    return _deserialize_agent_input(payload)


async def invoke_graph_with_traces(graph: Any, agent_input: AgentInput) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_token = begin_llm_trace_session()
    try:
        result = await graph.ainvoke(
            {"agent_input": _serialize_agent_input(agent_input)},
            config={
                "configurable": {
                    "thread_id": agent_input.session_id,
                    "user_id": agent_input.user_id,
                }
            },
        )
    finally:
        llm_traces = end_llm_trace_session(trace_token)
    return result, llm_traces


async def _emit_agent_event(event_type: str, data: dict[str, Any]) -> None:
    emitter = AGENT_EVENT_EMITTER.get()
    if emitter is None:
        return
    await emitter({"type": event_type, "data": data})


async def _emit_answer_text(text: str) -> None:
    if not text.strip():
        return
    await _emit_agent_event("answer_start", {"label": "正在输出回答。"})
    await _emit_agent_event("answer_delta", {"delta": text})


async def _run_route_decision(
    *,
    user_text: str,
    transcript: str,
    attachment_lines: list[str],
) -> dict[str, Any]:
    tool_lines = _build_tool_lines()
    messages = build_react_deliberation_messages(
        user_text=user_text,
        transcript=transcript,
        tool_lines=tool_lines,
        attachment_lines=attachment_lines,
    )
    raw_output = await local_chat_completion(
        messages,
        max_new_tokens=512,
        temperature=ROUTE_DECISION_TEMPERATURE,
        trace_label="agent_route_decision",
    )
    decision = _parse_route_decision(raw_output.get("message") or "")
    if decision is not None:
        return decision

    repair_messages = build_react_decision_repair_messages(
        user_text=user_text,
        transcript=transcript,
        tool_lines=tool_lines,
        attachment_lines=attachment_lines,
        raw_output=str(raw_output.get("message") or ""),
    )
    repaired_output = await local_chat_completion(
        repair_messages,
        max_new_tokens=512,
        temperature=ROUTE_DECISION_TEMPERATURE,
        trace_label="agent_route_repair",
    )
    repaired = _parse_route_decision(repaired_output.get("message") or "")
    if repaired is not None:
        return repaired
    return {
        "intent": IntentType.GENERAL_CHAT.value,
        "plan": "路由结果解析失败，回退到直接对话。",
        "action": "final",
        "tool_name": "",
        "reason": "路由决策输出无法解析，已降级到基础对话流程。",
        "answer": "",
    }


async def _run_react_final_answer(transcript: str) -> str:
    messages = build_react_final_answer_messages(transcript)
    if AGENT_EVENT_EMITTER.get() is None:
        result = await local_chat_completion(
            messages,
            max_new_tokens=1024,
            temperature=0.2,
            trace_label="agent_final_answer",
        )
        return str(result.get("message") or "").strip() or "模型未返回内容。"

    collected: list[str] = []
    await _emit_agent_event(
        "answer_start",
        {"label": "正在生成最终回答。"},
    )
    async for chunk in local_chat_completion_stream(
        messages,
        max_new_tokens=1024,
        temperature=0.2,
        trace_label="agent_final_answer",
    ):
        collected.append(chunk)
        await _emit_agent_event("answer_delta", {"delta": chunk})
    return "".join(collected).strip() or "模型未返回内容。"


async def _run_augmented_final_answer(
    *,
    agent_input: AgentInput,
    single_cell_result: dict[str, Any],
    rag_result: dict[str, Any],
    web_result: dict[str, Any],
) -> str:
    image_files = _load_image_attachments(agent_input)
    messages = build_augmented_analysis_messages(
        user_text=agent_input.user_text,
        report_context=str(single_cell_result.get("report_context") or ""),
        analysis_json=json.dumps(
            single_cell_result.get("analysis_result") or {},
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
        rag_text=_stringify_external_result(rag_result),
        web_text=_stringify_external_result(web_result),
        image_files=image_files,
    )
    normalized_messages = _normalize_augmented_messages(messages)
    if AGENT_EVENT_EMITTER.get() is None:
        result = await local_chat_completion(
            normalized_messages,
            max_new_tokens=1536,
            temperature=0.2,
            trace_label="augmented_analysis_answer",
        )
        return str(result.get("message") or "").strip() or "模型未返回内容。"

    collected: list[str] = []
    await _emit_agent_event(
        "answer_start",
        {"label": "正在整理增强分析结论。"},
    )
    async for chunk in local_chat_completion_stream(
        normalized_messages,
        max_new_tokens=1536,
        temperature=0.2,
        trace_label="augmented_analysis_answer",
    ):
        collected.append(chunk)
        await _emit_agent_event("answer_delta", {"delta": chunk})
    return "".join(collected).strip() or "模型未返回内容。"


def _load_memory_context(agent_input: AgentInput) -> BuiltMemoryContext:
    try:
        return _MEMORY_MANAGER.build_context(
            user_id=agent_input.user_id,
            session_id=agent_input.session_id,
            query=agent_input.user_text,
        )
    except RuntimeError:
        return BuiltMemoryContext(
            user_id=agent_input.user_id,
            session_id=agent_input.session_id,
        )


async def _resolve_user_text_with_memory(
    *,
    user_text: str,
    recent_messages: list[dict[str, Any]],
    short_summary: str,
) -> str:
    normalized_user_text = str(user_text or "").strip()
    if not normalized_user_text:
        return ""
    if not recent_messages and not short_summary.strip():
        return normalized_user_text
    if not _looks_like_followup_question(normalized_user_text):
        return normalized_user_text

    messages = build_followup_contextualize_messages(
        user_text=normalized_user_text,
        recent_messages=recent_messages,
        short_summary=short_summary,
    )
    result = await local_chat_completion(
        messages,
        max_new_tokens=256,
        temperature=0.0,
        trace_label="followup_contextualize",
    )
    resolved_query = _parse_contextualized_query(result.get("message") or "")
    return resolved_query or normalized_user_text


def _parse_contextualized_query(raw_output: str) -> str:
    raw_text = str(raw_output or "").strip()
    if not raw_text:
        return ""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("resolved_query") or "").strip()


def _looks_like_followup_question(user_text: str) -> bool:
    normalized = str(user_text or "").strip().lower()
    if not normalized:
        return False
    followup_markers = (
        "它",
        "这个",
        "那个",
        "其",
        "该",
        "上述",
        "前者",
        "后者",
        "这些",
        "those",
        "that",
        "this",
        "it",
        "they",
    )
    if any(marker in normalized for marker in followup_markers):
        return True
    return len(normalized) <= 18


def _memory_context_from_state(
    state: AgentWorkflowState,
    agent_input: AgentInput,
) -> BuiltMemoryContext:
    semantic_memories = [
        SemanticMemoryHit(
            content=str(item.get("content") or ""),
            score=float(item.get("score") or 0.0),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in (state.get("memory_semantic_memories") or [])
        if isinstance(item, dict)
    ]
    return BuiltMemoryContext(
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        profile=state.get("memory_profile") or {},
        semantic_memories=semantic_memories,
        recent_messages=state.get("memory_recent_messages") or [],
        task_state=state.get("memory_task_state") or {},
        short_summary=str(state.get("memory_short_summary") or ""),
    )


def _build_effective_agent_input(
    agent_input: AgentInput,
    resolved_user_text: str,
) -> AgentInput:
    return AgentInput(
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        user_text=str(resolved_user_text or agent_input.user_text),
        attachments=list(agent_input.attachments),
        workspace_settings=dict(agent_input.workspace_settings),
    )


def _serialize_agent_input(agent_input: AgentInput) -> dict[str, Any]:
    return {
        "user_id": agent_input.user_id,
        "session_id": agent_input.session_id,
        "user_text": agent_input.user_text,
        "attachments": [asdict(item) for item in agent_input.attachments],
        "workspace_settings": dict(agent_input.workspace_settings),
    }


def _deserialize_agent_input(payload: dict[str, Any]) -> AgentInput:
    return AgentInput(
        user_id=str(payload.get("user_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        user_text=str(payload.get("user_text") or ""),
        attachments=[
            UploadedAsset(
                name=str(item.get("name") or ""),
                kind=str(item.get("kind") or ""),
                content_type=str(item.get("content_type") or ""),
                size_bytes=int(item.get("size_bytes") or 0),
                path=str(item.get("path") or ""),
            )
            for item in (payload.get("attachments") or [])
            if isinstance(item, dict)
        ],
        workspace_settings=dict(payload.get("workspace_settings") or {}),
    )


def _build_attachment_lines(attachments: list[UploadedAsset]) -> list[str]:
    if not attachments:
        return ["- 无附件"]
    return [
        (
            f"- {asset.name} "
            f"(kind={asset.kind}, content_type={asset.content_type}, size_bytes={asset.size_bytes})"
        )
        for asset in attachments
    ]


def _display_executable_tool_name(tool_name: str) -> str:
    intent = TOOL_NAME_TO_INTENT.get(tool_name)
    if intent is None:
        return tool_name
    mapped_tool = INTENT_TO_TOOL_NAME.get(intent)
    if mapped_tool is None:
        return tool_name
    return TOOL_DISPLAY_NAMES.get(mapped_tool, tool_name)


def _build_initial_transcript(
    *,
    user_text: str,
    resolved_user_text: str | None = None,
    attachment_lines: list[str],
    memory_context: BuiltMemoryContext | None = None,
    workspace_settings: dict[str, Any] | None = None,
) -> str:
    local_source_min_score = _get_local_source_min_score(workspace_settings)
    web_source_min_score = _get_web_source_min_score(workspace_settings)
    lines = [
        *_build_memory_context_lines(memory_context),
        f"本地知识最小置信度阈值：score >= {local_source_min_score:.2f}",
        "",
        f"网页来源最小置信度阈值：score >= {web_source_min_score:.2f}",
        "",
        "当前用户原始问题：",
        user_text.strip() or "(empty)",
    ]
    normalized_resolved = str(resolved_user_text or "").strip()
    if normalized_resolved and normalized_resolved != user_text.strip():
        lines.extend(
            [
                "",
                "追问补全后问题：",
                normalized_resolved,
            ]
        )
    lines.extend(
        [
            "",
            "附件列表：",
            *attachment_lines,
        ]
    )
    return "\n".join(lines).strip()


def _build_memory_context_lines(
    memory_context: BuiltMemoryContext | None,
) -> list[str]:
    if memory_context is None:
        return []

    lines: list[str] = []
    if memory_context.short_summary.strip():
        lines.extend(
            [
                "短期记忆摘要：",
                memory_context.short_summary.strip(),
                "",
            ]
        )

    if memory_context.profile:
        lines.extend(
            [
                "长期偏好：",
                json.dumps(
                    memory_context.profile,
                    ensure_ascii=False,
                    default=str,
                    indent=2,
                ),
                "",
            ]
        )

    if memory_context.semantic_memories:
        lines.append("相关长期记忆：")
        for item in memory_context.semantic_memories[:3]:
            content = (
                str(getattr(item, "content", "") or "")
                if not isinstance(item, dict)
                else str(item.get("content") or "")
            )
            score = (
                float(getattr(item, "score", 0.0) or 0.0)
                if not isinstance(item, dict)
                else float(item.get("score") or 0.0)
            )
            lines.append(f"- score={score:.3f} | {_trim_text(content, 220)}")
        lines.append("")

    recent_lines = _format_recent_memory_lines(memory_context.recent_messages)
    if recent_lines:
        lines.extend(["最近对话：", *recent_lines, ""])

    return lines


def _format_recent_memory_lines(messages: list[Any]) -> list[str]:
    normalized: list[str] = []
    for item in messages[-6:]:
        if hasattr(item, "role") and hasattr(item, "content"):
            role = str(getattr(item, "role") or "")
            content = str(getattr(item, "content") or "")
        elif isinstance(item, dict):
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
        else:
            continue
        if role not in {"user", "assistant"} or not content.strip():
            continue
        role_label = "用户" if role == "user" else "助手"
        normalized.append(f"- {role_label}: {_trim_text(content, 180)}")
    return normalized


def _append_tool_observation(
    transcript: str,
    tool_name: str,
    result: dict[str, Any],
) -> str:
    observation = json.dumps(result, ensure_ascii=False, default=str, indent=2)
    return (
        f"{transcript}\n\n"
        f"Tool Observation [{tool_name}]：\n{observation}"
    ).strip()


def _persist_memory(
    *,
    agent_input: AgentInput,
    state: AgentWorkflowState,
    tool_result: dict[str, Any],
    final_answer: str,
) -> None:
    answer_text = (
        final_answer.strip()
        or str(tool_result.get("answer") or tool_result.get("message") or "").strip()
        or DEFAULT_EMPTY_ANSWER
    )
    selected_tool_names = list(state.get("selected_tool_names") or [])
    selected_tool = selected_tool_names[-1] if selected_tool_names else ""
    state_update = dict(state.get("memory_task_state") or {})
    state_update.update(
        {
            "user_query": agent_input.user_text,
            "resolved_query": str(
                state.get("resolved_user_text") or agent_input.user_text
            ),
            "intent": str(state.get("intent") or ""),
            "selected_tool": selected_tool,
            "selected_tools": selected_tool_names,
            "tool_outputs": _summarize_tool_outputs(
                state.get("raw_tool_results") or {}
            ),
            "task_summary": answer_text,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    try:
        _MEMORY_MANAGER.write_short_term(
            agent_input.user_id,
            agent_input.session_id,
            role="user",
            content=agent_input.user_text,
            metadata={"attachments": _serialize_attachments(agent_input.attachments)},
        )
        short_term = _MEMORY_MANAGER.write_short_term(
            agent_input.user_id,
            agent_input.session_id,
            role="assistant",
            content=answer_text,
            metadata={
                "intent": str(state.get("intent") or ""),
                "selected_tools": selected_tool_names,
            },
            state_update=state_update,
        )
        _MEMORY_MANAGER.maybe_write_long_term(
            user_id=agent_input.user_id,
            user_text=agent_input.user_text,
            short_term=short_term,
            tool_result=tool_result,
        )
    except RuntimeError:
        return


def _get_web_source_min_score(workspace_settings: dict[str, Any] | None) -> float:
    raw_value = None if workspace_settings is None else workspace_settings.get(
        "web_source_min_score"
    )
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 1.5


def _get_local_source_min_score(workspace_settings: dict[str, Any] | None) -> float:
    raw_value = None if workspace_settings is None else workspace_settings.get(
        "local_source_min_score"
    )
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 0.35


def _build_web_possible_answer(results: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{index}. [{item.get('source_tier') or 'web'}] {item.get('title') or 'Untitled'}："
        f"{item.get('snippet') or ''} ({item.get('url') or ''})"
        for index, item in enumerate(results[:3], start=1)
        if item.get("snippet") or item.get("url")
    )


def _filter_web_result_by_score(
    result: dict[str, Any],
    *,
    min_score: float,
) -> dict[str, Any]:
    if not result:
        return {}

    filtered = dict(result)
    raw_results = [
        item for item in (result.get("results") or []) if isinstance(item, dict)
    ]
    raw_references = [
        item for item in (result.get("references") or []) if isinstance(item, dict)
    ]
    filtered_results = [
        item for item in raw_results if float(item.get("score") or 0.0) >= min_score
    ]
    filtered_references = [
        item for item in raw_references if float(item.get("score") or 0.0) >= min_score
    ]
    filtered["results"] = filtered_results
    filtered["references"] = filtered_references
    filtered["web_source_min_score"] = min_score
    filtered["filtered_out_count"] = max(0, len(raw_results) - len(filtered_results))

    if filtered_results:
        possible_answer = _build_web_possible_answer(filtered_results)
        filtered["possible_answer"] = possible_answer
        filtered["answer"] = possible_answer
        filtered["local_answer"] = possible_answer
        filtered["evidence_status"] = "sufficient"
    else:
        message = "网页搜索已执行，但未检索到达到当前置信度阈值的结果。"
        filtered["possible_answer"] = ""
        filtered["answer"] = message
        filtered["local_answer"] = message
        filtered["message"] = message
        filtered["evidence_status"] = "insufficient"

    return filtered


def _serialize_attachments(attachments: list[UploadedAsset]) -> list[dict[str, Any]]:
    return [
        {
            "name": asset.name,
            "kind": asset.kind,
            "content_type": asset.content_type,
            "size_bytes": asset.size_bytes,
            "path": asset.path,
        }
        for asset in attachments
    ]


def _summarize_tool_outputs(
    raw_tool_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        tool_name: _compact_tool_result(result)
        for tool_name, result in raw_tool_results.items()
    }


def _summarize_single_tool_result(result: dict[str, Any]) -> str:
    compact = _compact_tool_result(result)
    if compact.get("answer"):
        return str(compact["answer"])
    if compact.get("message"):
        return str(compact["message"])
    references_count = compact.get("references_count")
    artifacts_count = compact.get("artifacts_count")
    if references_count:
        return f"返回 {references_count} 条参考来源。"
    if artifacts_count:
        return f"生成 {artifacts_count} 个产物。"
    return f"状态：{compact.get('status') or 'ok'}"


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"status": str(result.get("status") or "")}
    for key in ("query", "answer", "message", "local_answer", "report_context"):
        value = str(result.get(key) or "").strip()
        if value:
            compact[key] = _trim_text(value, 240)

    references = result.get("references")
    if isinstance(references, list) and references:
        compact["references_count"] = len(references)

    artifacts = result.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        compact["artifacts_count"] = len(artifacts)

    analysis_result = result.get("analysis_result")
    if isinstance(analysis_result, dict) and analysis_result:
        compact["analysis_result_keys"] = list(analysis_result.keys())[:12]

    return compact


def _trim_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(limit - 3, 0)]}..."


def _append_agent_decision(
    *,
    transcript: str,
    decision: dict[str, Any],
    deliberation_count: int,
) -> str:
    lines = [
        transcript.strip(),
        "",
        f"Agent Decision [{deliberation_count}]：",
        f"intent={str(decision.get('intent') or '').strip()}",
        f"plan={str(decision.get('plan') or '').strip()}",
        f"action={str(decision.get('action') or '').strip()}",
        f"tool_name={str(decision.get('tool_name') or '').strip()}",
        f"reason={str(decision.get('reason') or '').strip()}",
    ]
    answer = str(decision.get("answer") or "").strip()
    if answer:
        lines.append(f"answer={answer}")
    return "\n".join(line for line in lines if line is not None).strip()


def _build_tool_lines() -> list[str]:
    return [f"- {item}" for item in EXECUTABLE_TOOL_NAMES]


def _parse_route_decision(raw_output: str) -> dict[str, Any] | None:
    raw_text = str(raw_output or "").strip()
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return None

    intent = _coerce_intent(parsed.get("intent"))
    action = str(parsed.get("action") or "").strip().lower()
    tool_name = str(parsed.get("tool_name") or "").strip()
    if action not in {"tool", "final"}:
        return None
    if action == "final":
        tool_name = ""
    elif not tool_name:
        tool_name = intent.value

    return {
        "intent": intent.value,
        "plan": str(parsed.get("plan") or "").strip(),
        "action": action,
        "tool_name": tool_name,
        "reason": str(parsed.get("reason") or "").strip(),
        "answer": str(parsed.get("answer") or "").strip(),
    }


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_intent(value: Any) -> IntentType:
    normalized = str(value or "").strip()
    return TOOL_NAME_TO_INTENT.get(normalized, IntentType.GENERAL_CHAT)


def _normalize_react_decision_for_graph(
    *,
    decision: dict[str, Any],
    selected_tool_names: list[str],
    raw_tool_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = dict(decision)
    intent = _coerce_intent(normalized.get("intent"))
    action = str(normalized.get("action") or "").strip().lower()
    tool_name = str(normalized.get("tool_name") or "").strip()
    selected_tool_name_set = set(selected_tool_names)
    rag_needs_web_fallback = _should_fallback_from_local_qa_to_web(
        intent=intent,
        raw_tool_results=raw_tool_results,
    )
    general_chat_needs_web_fallback = _should_fallback_from_general_chat_to_web(
        intent=intent,
        raw_tool_results=raw_tool_results,
    )

    if (
        rag_needs_web_fallback
        and "web_search" not in selected_tool_name_set
        and "web_search" not in raw_tool_results
    ):
        normalized["action"] = "tool"
        normalized["tool_name"] = "web_search"
        normalized["answer"] = ""
        reason = str(normalized.get("reason") or "").strip()
        normalized["reason"] = (
            f"{reason} 本地知识证据不足，继续使用网页搜索补充证据。".strip()
            if reason
            else "本地知识证据不足，继续使用网页搜索补充证据。"
        )
        plan = str(normalized.get("plan") or "").strip()
        normalized["plan"] = (
            f"{plan} 先基于网页搜索补足外部证据，再综合判断。".strip()
            if plan
            else "先基于网页搜索补足外部证据，再综合判断。"
        )
        action = "tool"
        tool_name = "web_search"

    if (
        general_chat_needs_web_fallback
        and "web_search" not in selected_tool_name_set
        and "web_search" not in raw_tool_results
    ):
        normalized["action"] = "tool"
        normalized["tool_name"] = "web_search"
        normalized["answer"] = ""
        reason = str(normalized.get("reason") or "").strip()
        normalized["reason"] = (
            f"{reason} 通用对话结果不足以回答用户问题，继续使用网页搜索补充信息。".strip()
            if reason
            else "通用对话结果不足以回答用户问题，继续使用网页搜索补充信息。"
        )
        plan = str(normalized.get("plan") or "").strip()
        normalized["plan"] = (
            f"{plan} 先使用网页搜索补充外部信息，再综合生成最终回答。".strip()
            if plan
            else "先使用网页搜索补充外部信息，再综合生成最终回答。"
        )
        action = "tool"
        tool_name = "web_search"

    if action == "tool":
        resolved_tool_name = _resolve_executable_tool_name(
            intent=intent,
            tool_name=tool_name,
            raw_tool_results=raw_tool_results,
        )
        if not resolved_tool_name:
            normalized["action"] = "final"
            normalized["tool_name"] = ""
            normalized["answer"] = str(normalized.get("answer") or "").strip()
            reason = str(normalized.get("reason") or "").strip()
            normalized["reason"] = reason or "当前 observation 已足够，或继续调用工具不再合适。"
            return normalized

        if resolved_tool_name in selected_tool_name_set:
            normalized["action"] = "final"
            normalized["tool_name"] = ""
            normalized["answer"] = str(normalized.get("answer") or "").strip()
            normalized["reason"] = (
                f"工具 {resolved_tool_name} 已执行过，避免重复调用；基于现有 observation 收敛。"
            )
            return normalized

        normalized["tool_name"] = resolved_tool_name
        normalized["answer"] = ""
        return normalized

    normalized["tool_name"] = ""
    return normalized


def _resolve_executable_tool_name(
    *,
    intent: IntentType,
    tool_name: str,
    raw_tool_results: dict[str, dict[str, Any]],
) -> str:
    normalized_tool_name = str(tool_name or "").strip()
    if normalized_tool_name in EXECUTABLE_TOOL_NAMES:
        return normalized_tool_name

    if _has_error_tool_result(raw_tool_results):
        return ""

    if intent is IntentType.AUGMENTED_ANALYSIS:
        if "single_cell_analysis" not in raw_tool_results:
            return "single_cell_analysis"
        if "local_knowledge_qa" not in raw_tool_results:
            return "local_knowledge_qa"
        if "web_search" not in raw_tool_results:
            return "web_search"
        return ""

    if intent is IntentType.LOCAL_KNOWLEDGE_QA:
        if "local_knowledge_qa" not in raw_tool_results:
            return "local_knowledge_qa"
        if _should_fallback_from_local_qa_to_web(
            intent=intent,
            raw_tool_results=raw_tool_results,
        ):
            return "web_search"
        return ""
    if intent is IntentType.WEB_SEARCH:
        return "web_search"
    if intent is IntentType.SINGLE_CELL_ANALYSIS:
        return "single_cell_analysis"
    if intent is IntentType.GENERAL_CHAT:
        if "general_chat" not in raw_tool_results:
            return "general_chat"
        if _should_fallback_from_general_chat_to_web(
            intent=intent,
            raw_tool_results=raw_tool_results,
        ):
            return "web_search"
        return ""
    return ""


def _force_attachment_aware_general_chat_if_needed(
    *,
    decision: dict[str, Any],
    agent_input: AgentInput,
    raw_tool_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = dict(decision)
    if _coerce_intent(normalized.get("intent")) is not IntentType.GENERAL_CHAT:
        return normalized
    if raw_tool_results:
        return normalized

    has_multimodal_attachment = any(
        asset.kind in {"image", "pdf"} for asset in agent_input.attachments
    )
    if not has_multimodal_attachment:
        return normalized

    normalized["action"] = "tool"
    normalized["tool_name"] = "general_chat"
    normalized["answer"] = ""
    reason = str(normalized.get("reason") or "").strip()
    normalized["reason"] = (
        f"{reason} 当前问题附带图片或 PDF，需要先执行支持附件理解的 general_chat 工具。".strip()
        if reason
        else "当前问题附带图片或 PDF，需要先执行支持附件理解的 general_chat 工具。"
    )
    plan = str(normalized.get("plan") or "").strip()
    normalized["plan"] = (
        f"{plan} 先读取附件内容并生成初步回答，再决定是否需要后续工具。".strip()
        if plan
        else "先读取附件内容并生成初步回答，再决定是否需要后续工具。"
    )
    return normalized


def _should_use_direct_tool_answer_in_finalize(
    *,
    selected_tool_names: list[str],
    raw_tool_results: dict[str, dict[str, Any]],
    tool_result: dict[str, Any],
) -> bool:
    if selected_tool_names != ["general_chat"]:
        return False
    if len(raw_tool_results) != 1 or "general_chat" not in raw_tool_results:
        return False
    answer_text = str(
        tool_result.get("answer") or tool_result.get("message") or ""
    ).strip()
    return bool(answer_text)


def _has_error_tool_result(raw_tool_results: dict[str, dict[str, Any]]) -> bool:
    for result in raw_tool_results.values():
        if str((result or {}).get("status") or "").lower() not in {"", "ok"}:
            return True
    return False


def _should_fallback_from_local_qa_to_web(
    *,
    intent: IntentType,
    raw_tool_results: dict[str, dict[str, Any]],
) -> bool:
    if intent is not IntentType.LOCAL_KNOWLEDGE_QA:
        return False
    if "web_search" in raw_tool_results:
        return False

    rag_result = dict(raw_tool_results.get("local_knowledge_qa") or {})
    if not rag_result:
        return False
    return _local_qa_result_is_insufficient(rag_result)


def _local_qa_result_is_insufficient(result: dict[str, Any]) -> bool:
    if str(result.get("status") or "").lower() not in {"", "ok"}:
        return False

    answer_text = " ".join(
        str(result.get(key) or "").strip()
        for key in ("answer", "message", "local_answer")
    ).strip()
    normalized_answer = answer_text.lower()

    if not answer_text:
        return True

    insufficiency_markers = (
        "根据当前知识库内容无法确定",
        "无法确定",
        "证据不足",
        "无法直接回答",
        "没有直接提供",
        "未直接提供",
        "知识库内容不足",
        "insufficient",
        "cannot determine",
        "not enough evidence",
    )
    if any(marker in answer_text for marker in insufficiency_markers):
        return True
    if any(marker in normalized_answer for marker in insufficiency_markers):
        return True

    references = result.get("references")
    if not isinstance(references, list) or not references:
        return True

    retrieved_chunks = result.get("retrieved_chunks")
    if isinstance(retrieved_chunks, list) and not retrieved_chunks:
        return True

    return False


def _should_fallback_from_general_chat_to_web(
    *,
    intent: IntentType,
    raw_tool_results: dict[str, dict[str, Any]],
) -> bool:
    if intent is not IntentType.GENERAL_CHAT:
        return False
    if "web_search" in raw_tool_results:
        return False

    general_chat_result = dict(raw_tool_results.get("general_chat") or {})
    if not general_chat_result:
        return False
    return _general_chat_result_is_insufficient(general_chat_result)


def _general_chat_result_is_insufficient(result: dict[str, Any]) -> bool:
    if str(result.get("status") or "").lower() not in {"", "ok"}:
        return False

    answer_text = " ".join(
        str(result.get(key) or "").strip()
        for key in ("answer", "message", "local_answer")
    ).strip()
    normalized_answer = answer_text.lower()

    if not answer_text:
        return True

    insufficiency_markers = (
        "抱歉，我无法",
        "抱歉，无法",
        "无法回答",
        "无法确定",
        "不确定",
        "无法确认",
        "我不知道",
        "我不清楚",
        "没有足够信息",
        "缺少足够信息",
        "需要更多信息",
        "无法回溯",
        "无法判断",
        "无法提供准确",
        "insufficient",
        "cannot answer",
        "cannot determine",
        "i don't know",
        "not enough information",
        "need more information",
    )
    if any(marker in answer_text for marker in insufficiency_markers):
        return True
    if any(marker in normalized_answer for marker in insufficiency_markers):
        return True

    return False


def _append_tool_name(
    existing: list[str] | None,
    tool_name: str,
) -> list[str]:
    merged = list(existing or [])
    if tool_name not in merged:
        merged.append(tool_name)
    return merged


def _merge_tool_results(
    existing: dict[str, dict[str, Any]] | None,
    tool_name: str,
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged = dict(existing or {})
    merged[tool_name] = dict(result)
    return merged


def _combine_external_knowledge_results(
    *,
    rag_result: dict[str, Any],
    web_result: dict[str, Any],
) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "status": "ok",
        "local_knowledge_qa": rag_result,
        "web_search": web_result,
        "observation": {
            "local_knowledge_qa": rag_result,
            "web_search": web_result,
        },
        "references": _merge_references(
            rag_result.get("references"),
            web_result.get("references"),
        ),
    }
    if rag_result.get("retrieved_chunks"):
        combined["retrieved_chunks"] = rag_result.get("retrieved_chunks")
    if rag_result.get("retrieval_trace"):
        combined["retrieval_trace"] = rag_result.get("retrieval_trace")
    return combined


def _normalize_tool_result(
    raw_tool_result: dict[str, Any],
    final_answer: str,
) -> dict[str, Any]:
    tool_result = dict(raw_tool_result)
    if final_answer:
        if raw_tool_result and "observation" not in tool_result:
            tool_result["observation"] = dict(raw_tool_result)
        tool_result["answer"] = final_answer
        tool_result["message"] = final_answer
        tool_result["local_answer"] = final_answer
    if tool_result and not tool_result.get("status"):
        tool_result["status"] = "ok"
    if not tool_result:
        tool_result = {
            "status": "ok",
            "answer": final_answer or DEFAULT_EMPTY_ANSWER,
            "message": final_answer or DEFAULT_EMPTY_ANSWER,
            "local_answer": final_answer or DEFAULT_EMPTY_ANSWER,
        }
    return tool_result


def _resolve_selected_tools(
    *,
    intent: IntentType,
    selected_tool_names: list[str],
) -> list[ToolName]:
    normalized: list[ToolName] = []
    for item in selected_tool_names:
        mapped_intent = TOOL_NAME_TO_INTENT.get(str(item).strip())
        if mapped_intent is None:
            continue
        tool_name = INTENT_TO_TOOL_NAME[mapped_intent]
        if tool_name not in normalized:
            normalized.append(tool_name)
    if normalized:
        return normalized
    return [INTENT_TO_TOOL_NAME.get(intent, ToolName.DIRECT_LLM)]


def _build_execution_steps(
    *,
    selected_tools: list[ToolName],
    route_reason: str,
    final_answer: str,
) -> list[AgentStep]:
    steps = [
        AgentStep(
            step_id="step-1",
            description=(
                f"分析问题并确定处理方式：{route_reason}"
                if route_reason
                else "分析问题并选择最合适的处理方式。"
            ),
            status="completed",
        )
    ]
    for index, tool_name in enumerate(selected_tools, start=2):
        steps.append(
            AgentStep(
                step_id=f"step-{index}",
                description=f"执行节点：{TOOL_DISPLAY_NAMES.get(tool_name, tool_name.value)}",
                tool_name=tool_name,
                status="completed",
            )
        )
    steps.append(
        AgentStep(
            step_id=f"step-{len(steps) + 1}",
            description="汇总节点输出并生成最终回答。" if final_answer else "完成当前处理流程。",
            status="completed",
        )
    )
    return steps


def _load_image_attachments(agent_input: AgentInput) -> list[dict[str, Any]]:
    config = get_config()
    image_files: list[dict[str, Any]] = []
    for asset in agent_input.attachments:
        if asset.kind != "image":
            continue
        image_path = config.project_root / asset.path
        if not image_path.exists():
            continue
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{asset.content_type};base64,{encoded}"
        image_files.append(
            {
                "name": asset.name,
                "data_url": data_url,
            }
        )
    return image_files


def _normalize_augmented_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        normalized_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                normalized_content.append({"type": "text", "text": str(item)})
                continue
            if str(item.get("type") or "").strip().lower() != "image_url":
                normalized_content.append(item)
                continue
            image_url = item.get("image_url") or {}
            normalized_content.append(
                {
                    "type": "image",
                    "image": str(image_url.get("url") or ""),
                }
            )
        normalized.append(
            {
                "role": str(message.get("role") or "user"),
                "content": normalized_content,
            }
        )
    return normalized


def _stringify_external_result(result: dict[str, Any]) -> str:
    if not result:
        return "无结果。"
    return json.dumps(result, ensure_ascii=False, default=str, indent=2)


def _merge_references(*groups: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, ensure_ascii=False, default=str, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged
