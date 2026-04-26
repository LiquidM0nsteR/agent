# agent/backend/prompts.py

from __future__ import annotations

from typing import Any

from .util import safe_text


FINAL_SYSTEM_PROMPT = """
你是 Agent 的最终回答节点，负责整合用户输入、工具调用结果和中间观察，生成逻辑清晰、准确、可执行的最终回答。

回答要求：
1. 优先基于工具结果回答，不要编造工具没有返回的信息。
2. 如果工具成功，说明完成了什么，并整理关键结果。
3. 如果工具失败，明确指出失败节点和失败原因。
4. 如果是单细胞分析任务，需要优先返回报告路径、输出目录、关键摘要。
5. 如果是 RAG 检索任务，需要整合本地知识库返回内容。
6. 如果是网页搜索任务，需要整合搜索摘要，并保留重要链接。
7. 回答应结构清晰，避免堆砌原始日志。
""".strip()


def format_observations(observations: list[dict[str, Any]]) -> str:
    """
    将 observations 转成 prompt 中可读的文本。
    """
    if not observations:
        return "暂无工具观察结果。"

    lines: list[str] = []

    for idx, obs in enumerate(observations, start=1):
        node = obs.get("node", "UnknownNode")
        ok = obs.get("ok", False)
        content = obs.get("content", "")
        error = obs.get("error", "")
        metadata = obs.get("metadata", {})

        lines.append(f"观察 {idx}:")
        lines.append(f"- 节点: {node}")
        lines.append(f"- 是否成功: {ok}")

        if content:
            lines.append(f"- 返回内容:\n{content}")

        if error:
            lines.append(f"- 错误信息:\n{error}")

        if metadata:
            lines.append(f"- 元信息:\n{metadata}")

        lines.append("")

    return "\n".join(lines).strip()


def build_final_prompt(
    user_input: str,
    observations: list[dict[str, Any]],
    tool_results: dict[str, Any],
    steps: list[str],
    memory_context: str = "",
) -> str:
    """
    FinalNode 使用的 prompt。
    后面接入本地 Qwen 或 API 模型时，可以把这个 prompt 直接交给 LLM。
    """
    obs_text = format_observations(observations)

    memory_section = ""
    if memory_context.strip():
        memory_section = f"\n长期记忆上下文：\n{memory_context}\n"

    return f"""
{FINAL_SYSTEM_PROMPT}

用户输入：
{user_input}

{memory_section}

执行轨迹：
{steps}

工具观察结果：
{obs_text}

工具结构化结果：
{tool_results}

请生成最终回答。
""".strip()


def _clip_text(value: Any, limit: int = 1800) -> str:
    text = safe_text(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_react_prompt(
    *,
    user_input: str,
    metadata: dict[str, Any],
    steps: list[str],
    react_steps: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    tool_results: dict[str, Any],
    memory_context: str = "",
) -> str:
    if observations:
        observation_lines: list[str] = []
        for index, obs in enumerate(observations[-4:], start=1):
            obs_metadata = dict(obs.get("metadata") or {})
            observation_lines.append(f"Observation {index}:")
            observation_lines.append(f"- tool: {obs.get('node', 'Unknown')}")
            observation_lines.append(f"- ok: {bool(obs.get('ok'))}")
            if obs.get("content"):
                observation_lines.append(f"- content: {_clip_text(obs.get('content'), 1200)}")
            if obs.get("error"):
                observation_lines.append(f"- error: {_clip_text(obs.get('error'), 700)}")
            if obs_metadata:
                observation_lines.append(f"- metadata: {_clip_text(obs_metadata, 700)}")
        observations_text = "\n".join(observation_lines).strip()
    else:
        observations_text = "暂无工具观察结果。"

    if react_steps:
        step_lines = []
        for index, step in enumerate(react_steps[-6:], start=1):
            step_lines.append(
                f"{index}. thought={_clip_text(step.get('thought'), 300)}; "
                f"action={step.get('action')}; action_input={_clip_text(step.get('action_input'), 300)}"
            )
        react_steps_text = "\n".join(step_lines)
    else:
        react_steps_text = "暂无 ReAct 步骤。"

    tool_summary = {
        key: {
            "status": value.get("status"),
            "answer": _clip_text(value.get("answer") or value.get("message"), 800),
            "metrics": value.get("metrics") or {},
            "meta": value.get("meta") or {},
        }
        for key, value in tool_results.items()
        if isinstance(value, dict)
    }

    return f"""
你是一个 ReAct Supervisor，必须在每一轮根据用户问题、历史 Thought/Action 和工具 Observation 决定下一步。

可用 Action 只能是：
- RAG: 检索本地知识库、项目文档、用户上传的非 PDF 文档。本地资料、已有文档问答、专业概念解释、非实时知识优先使用它。
- WebSearch: 联网搜索。最新/最近/当前进展、新闻、近期论文、GitHub、网页资料、外部资料核验必须使用它；专业主题也一样。
- scAnalysis: 只在存在 h5ad 文件且用户要求分析该数据时使用，生成单细胞分析结果和 PDF 报告。
- FinalNode: 已有结果足够回答，或无需工具、工具不可用、达到上限时结束。

硬性约束：
1. 每次只选择一个 Action。
2. 没有 h5ad 文件时不能选择 scAnalysis。
3. 如果用户问“最新/最近/当前/进展/新闻/近期论文/在线资料/GitHub”，选择 WebSearch。
4. 如果 RAG Observation 的 evidence_sufficient=false 或 confidence 低于 threshold，下一步通常选择 WebSearch。
5. 如果 WebSearch、RAG 或 scAnalysis 已经成功并足够回答，选择 FinalNode。
6. 不要重复调用同一个工具处理同一个 query；如果重复不会增加信息，选择 FinalNode。
7. 只输出 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。

JSON schema:
{{
  "thought": "一句简短、可展示的决策理由",
  "action": "RAG | WebSearch | scAnalysis | FinalNode",
  "action_input": {{
    "query": "给 RAG 或 WebSearch 使用的检索问题；没有则留空"
  }},
  "finish": false
}}

用户输入：
{user_input}

输入类型：
{metadata.get("input_kind")}

上传文件：
{_clip_text(metadata.get("normalized_files"), 1000)}

h5ad 文件：
{_clip_text(metadata.get("h5ad_files"), 600)}

RAG 文件：
{_clip_text(metadata.get("rag_files"), 600)}

长期记忆上下文：
{_clip_text(memory_context or "无", 1200)}

已执行图节点：
{steps}

ReAct 历史：
{react_steps_text}

工具 Observation：
{observations_text}

结构化工具结果摘要：
{_clip_text(tool_summary, 1400)}

请输出下一步 ReAct JSON。
""".strip()
