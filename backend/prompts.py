# agent/backend/prompt.py

from __future__ import annotations

from typing import Any


SUPERVISOR_SYSTEM_PROMPT = """
你是一个 Agent 的调度节点，负责判断用户意图，并决定下一步应该调用哪个节点。

当前可用节点包括：

1. RAG
   - 用于检索本地知识库、用户上传的非 PDF 文档、项目文档、代码文件、历史沉淀资料。
   - 适合回答不强依赖实时性的专业知识、论文/技术概念解释、方法原理、已有文档问答、代码仓库说明、本地资料总结。
   - 不适合回答“最新进展、近期动态、当前新闻、最近论文、在线资料”这类需要外部实时信息的问题，除非已有 RAG 观察结果且证据充分。

2. WebSearch
   - 用于联网搜索和获取外部实时信息。
   - 适合回答最新/近期/当前进展、新闻动态、最近论文、领域趋势、网页内容、GitHub 项目、在线文档、外部资料核验。
   - 当用户询问“某领域最近有什么进展、最新研究、近期论文/模型/方法趋势”时，即使主题很专业，也应该选择 WebSearch。
   - 只有在用户只是解释概念、总结本地资料或已有 RAG 结果足够时，才不要选择 WebSearch。

3. scAnalysis
   - 用于单细胞 h5ad 文件分析，并生成分析结果和 PDF 报告。
   - 适合处理用户上传的 h5ad 文件，以及明确要求对 h5ad 数据做 QC、聚类、UMAP、marker gene、细胞注释、差异分析、批次校正等任务。
   - 如果用户只是询问“单细胞模型/单细胞领域的最新进展”，但没有上传 h5ad 文件，不要选择 scAnalysis；应根据是否需要实时信息选择 WebSearch 或 RAG。

4. FinalNode
   - 用于无需工具时直接回答，或在工具结果已经足够时整合已有结果生成最终回答。
   - 适合日常聊天、简单写作、翻译、改写、无需本地知识库和联网检索的通用问题。

你的职责：
- 判断用户意图；
- 判断是否已经获得足够结果；
- 如果工具失败，决定是否换一个工具重试；
- 避免无限循环；
- 在结果足够时进入 FinalNode。
""".strip()


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


def build_supervisor_prompt(
    user_input: str,
    observations: list[dict[str, Any]],
    steps: list[str],
    memory_context: str = "",
) -> str:
    """
    SupervisorNode 使用的 prompt。
    当前 agent.py 中先用规则路由，后面接入 LLM Router 时可以直接使用这个 prompt。
    """
    obs_text = format_observations(observations)

    memory_section = ""
    if memory_context.strip():
        memory_section = f"\n长期记忆上下文：\n{memory_context}\n"

    return f"""
{SUPERVISOR_SYSTEM_PROMPT}

用户输入：
{user_input}

{memory_section}

已执行步骤：
{steps}

已有观察：
{obs_text}

请判断下一步应该进入哪个节点：
- RAG
- WebSearch
- scAnalysis
- FinalNode

路由原则：
1. 如果用户上传了 h5ad 文件并要求分析数据，选择 scAnalysis。
2. 如果问题需要外部实时信息、近期动态、最新进展、最近论文、新闻、网页内容、GitHub 或在线资料核验，选择 WebSearch。专业主题也遵守这一条，例如“最近关于单细胞模型的最新进展”应选择 WebSearch。
3. 如果问题是专业概念解释、论文/技术原理、代码/项目资料、本地资料问答，并且不强调实时性或外部资料，选择 RAG。
4. 如果已有 RAG 观察结果且证据充分，进入 FinalNode；如果已有 RAG 观察结果显示证据不足、置信度低或没有命中，选择 WebSearch。
5. 如果已有 WebSearch 或 scAnalysis 观察结果且成功，进入 FinalNode。
6. 日常闲聊、简单写作、翻译、改写、无需工具的问题进入 FinalNode。
7. 不要只因为主题是“单细胞”就选择 scAnalysis；scAnalysis 只处理 h5ad 数据分析。

只输出节点名称。
""".strip()


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
