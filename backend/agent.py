from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict
from typing import Any, AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .agent_runtime import (
    AGENT_EVENT_EMITTER,
    EXECUTABLE_TOOL_NAMES,
    AgentDecision,
    AgentInput,
    AgentRuntime,
    AgentWorkflowState,
    IntentType,
    ToolName,
    UploadedAsset,
    invoke_graph_with_traces,
)


_CHECKPOINTER = InMemorySaver()


class AgentService:
    def __init__(self) -> None:
        self.runtime = AgentRuntime()
        self.checkpointer = _CHECKPOINTER
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        # agent.py 只保留 LangGraph 的流程图结构与对外入口。
        builder = StateGraph(AgentWorkflowState)
        builder.add_node("prepare_context", self.runtime.prepare_context_node)
        builder.add_node("contextualize_query", self.runtime.contextualize_query_node)
        builder.add_node("deliberate", self.runtime.deliberate_node)
        for tool_name in EXECUTABLE_TOOL_NAMES:
            builder.add_node(tool_name, self.runtime.make_tool_node(tool_name))
        builder.add_node("finalize", self.runtime.finalize_node)

        builder.add_edge(START, "prepare_context")
        builder.add_edge("prepare_context", "contextualize_query")
        builder.add_edge("contextualize_query", "deliberate")
        builder.add_conditional_edges(
            "deliberate",
            self.runtime.deliberation_edge,
            {
                "general_chat": "general_chat",
                "local_knowledge_qa": "local_knowledge_qa",
                "web_search": "web_search",
                "single_cell_analysis": "single_cell_analysis",
                "finalize": "finalize",
            },
        )
        for tool_name in EXECUTABLE_TOOL_NAMES:
            builder.add_edge(tool_name, "deliberate")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    async def run(self, agent_input: AgentInput) -> AgentDecision:
        result, llm_traces = await invoke_graph_with_traces(self.graph, agent_input)
        return self.runtime.build_decision(result, llm_traces)

    async def stream(self, agent_input: AgentInput) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        emitter_token = AGENT_EVENT_EMITTER.set(queue.put)

        async def _runner() -> None:
            try:
                result, llm_traces = await invoke_graph_with_traces(
                    self.graph,
                    agent_input,
                )
                decision = self.runtime.build_decision(result, llm_traces)
                await queue.put(
                    {
                        "type": "final",
                        "data": _build_agent_response_payload(agent_input, decision),
                    }
                )
            except Exception as exc:
                await queue.put(
                    {
                        "type": "error",
                        "data": {
                            "message": str(exc) or "Agent execution failed.",
                        },
                    }
                )
            finally:
                AGENT_EVENT_EMITTER.reset(emitter_token)
                await queue.put(None)

        runner = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not runner.done():
                runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner

    @staticmethod
    def describe_graph() -> dict[str, Any]:
        return {
            "enabled": True,
            "architecture": "StateGraph",
            "entrypoint": "prepare_context",
            "nodes": [
                "prepare_context",
                "contextualize_query",
                "deliberate",
                "general_chat",
                "local_knowledge_qa",
                "web_search",
                "single_cell_analysis",
                "finalize",
            ],
            "edges": [
                "prepare_context -> contextualize_query -> deliberate",
                "deliberate -> {general_chat|local_knowledge_qa|web_search|single_cell_analysis|finalize}",
                "{general_chat|local_knowledge_qa|web_search|single_cell_analysis} -> deliberate",
            ],
            "checkpointer": "InMemorySaver",
            "memory_layers": ["langgraph_checkpointer"],
        }


_AGENT_SERVICE = AgentService()


async def build_agent_response(agent_input: AgentInput) -> dict[str, Any]:
    decision = await _AGENT_SERVICE.run(agent_input)
    return _build_agent_response_payload(agent_input, decision)


async def stream_agent_response(agent_input: AgentInput) -> AsyncIterator[dict[str, Any]]:
    async for item in _AGENT_SERVICE.stream(agent_input):
        yield item


def _build_agent_response_payload(
    agent_input: AgentInput,
    decision: AgentDecision,
) -> dict[str, Any]:
    return {
        "architecture": "langgraph.StateGraph",
        "status": "active",
        "langgraph": _AGENT_SERVICE.describe_graph(),
        "graph_execution": {
            "status": "active",
            "used_create_react_agent": False,
            "dispatched_node": decision.intent.value,
            "selected_tool": [item.value for item in decision.selected_tools],
        },
        "agent_input": asdict(agent_input),
        "decision": asdict(decision),
        "tool_result": decision.tool_result,
    }


__all__ = [
    "AgentInput",
    "AgentDecision",
    "AgentService",
    "AgentWorkflowState",
    "IntentType",
    "ToolName",
    "UploadedAsset",
    "build_agent_response",
    "stream_agent_response",
]
