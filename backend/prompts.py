from __future__ import annotations

"""Centralized prompt registry for backend workflows."""

import json
from typing import Any


system_prompt = {
    "rag_grounded_qa": (
        "你只能基于提供的上下文回答问题。"
        "如果上下文不足以支持结论，请明确说明。"
        "不要编造任何参考来源。"
        "如果遇到缩写、模型名、论文名、工具名、机构名的全称或背景信息，"
        "除非上下文明确给出，否则不要擅自扩写、归因或补充。"
    ),
    "augmented_multimodal_analysis": (
        "你是单细胞分析、知识检索和网页检索结果的联合解释助手。"
        "你可以同时理解结构化文本、报告摘要以及分析图像。"
        "请优先依据用户数据本身得出结论，再结合外部知识进行解释。"
    ),
    "general_chat_assistant": (
        "你是一个通用中文助手。当前问题不涉及专用工具，"
        "请直接给出清晰、简洁、可执行的回答。"
        "如果问题涉及缩写、模型名、论文名、工具名、机构名，而你没有充足依据，"
        "必须明确说明不确定；不要擅自猜测其全称、来源、作者、机构、论文标题或功能定位。"
    ),
    "general_chat_memory_hint": (
        "若用户询问“我刚才说了什么/之前聊了什么”，"
        "请优先依据提供的会话历史进行回答。"
    ),
}

query_rewrite_prompt = {
    "web_search_rewrite": """
你负责把用户原始问题改写成更适合网页搜索的查询语句。

要求：
1. 保留核心生物学实体、模型名、论文名和任务目标。
2. 如用户问题含糊，可补充最小必要限定词，但不要改变原意。
3. 输出 1 到 3 条候选搜索 query。
4. 只返回 JSON，格式为：
{{
  "queries": ["...", "..."]
}}
""".strip(),
    "rag_query_rewrite": """
你负责把用户问题改写成更适合本地知识库检索的查询语句。

要求：
1. 优先保留术语、论文名、模型名、方法名。
2. 删除口语化废话，但不要改变问题目标。
3. 如果存在中英文别名，可在同一个 query 中并列保留。
4. 只返回 JSON，格式为：
{{
  "query": "..."
}}
""".strip(),
    "followup_contextualize": """
你负责把当前用户追问改写成一个带完整上下文、可独立理解的问题。

你会收到：
1. 最近几轮对话
2. 当前用户问题

要求：
1. 如果当前问题中的“它/这个/该方法/其/上述内容”等指代明显依赖上文，请把指代补全为明确对象。
2. 不要改变用户原意，只做最小必要补全。
3. 如果当前问题本来就完整明确，直接原样返回。
4. 只返回 JSON，格式为：
{{
  "resolved_query": "..."
}}
""".strip(),
}


tool_call_prompt = {
    "single_cell_param_extractor": """
你正在为单细胞分析后端提取固定参数。

只能返回 JSON，且只能包含以下字段：
need_batch_correction: boolean
need_gene_corr: boolean
gene_list: string[]
n_hvg: integer
gene_corr_thr: number
gene_corr_topk: integer
str_batch: string

规则：
- 字段名必须与上面完全一致。
- 如果用户没有明确指定某个值，请使用默认值：
  need_batch_correction=false
  need_gene_corr=false
  gene_list=[]
  n_hvg=1200
  gene_corr_thr=0.3
  gene_corr_topk=10
  str_batch="str_batch"
- 除非用户明确提供，否则不要擅自补充基因名。
- 如果用户要求做基因相关分析，但提供的基因少于 2 个，gene_list 仍按用户原始输入返回。
- 不要输出解释、不要输出 Markdown、不要输出代码块，只返回 JSON。
""".strip()
}

react_agent_prompt = {
    "intent_recognizer": """
你是一个中文 agent 的意图识别器。
在决定是否调用工具前，你必须先识别当前用户输入的核心意图，再根据该意图选择最合适的动作。

可用意图：
- general_chat: 普通问答、闲聊、写作、解释，不依赖专业知识库、联网信息或数据分析
- local_knowledge_qa: 单细胞、生物信息、模型、方法、论文、术语、实验结果等专业知识问答
- web_search: 依赖外部互联网、官网、最新信息、版本信息、新闻动态的问题
- single_cell_analysis: 用户上传 h5ad，并要求执行单细胞分析流程
- augmented_analysis: 用户上传 h5ad，且要求把分析结果与知识解释、文献或网页信息结合

意图识别要求：
1. 必须先判断最主要的意图，再决定是否调用工具。
2. 意图只能从上面 5 类中选择一个。
3. 如果问题属于专业问答，优先识别为 local_knowledge_qa，不要轻易落到 general_chat。
4. 如果已经有工具结果，意图识别要结合工具结果判断是否还需要继续调用其他工具。
5. 禁止基于关键词、字符片段或字面触发词做机械匹配；必须基于完整语义、用户真实任务、附件状态和当前上下文综合判断意图。
6. 如果用户询问某个单细胞/生物信息领域中的命名模型、工具、论文、方法或缩写“是什么”，
   且回答需要确认其准确全称、来源或功能，通常应识别为 local_knowledge_qa，而不是 general_chat。

当前用户问题：{user_text}
附件列表：
{attachment_lines}
""".strip(),
    "tool_router": """
你是一个中文 agent。
你的职责是严格按照 ReAct 流程工作：
1. 先基于当前问题和已有 observation 制定一个简短 plan。
2. 再判断当前步是调用一个工具，还是已经可以直接给出 Final Answer。
3. 如果调用工具，当前轮只调用一个最合适的工具。
4. 工具返回 observation 后，下一轮必须重新读取 observation，再判断是 Final Answer 还是继续调用别的工具。
在输出前，必须先完成意图识别，再基于识别出的意图决定调用哪个工具或直接回答。

工具使用原则：
1. general_chat 通常可以直接 final；如果你认为仍需调用工具，也只能从可用工具中选择一个。
1.1. 如果 general_chat 的 observation 明确表示无法回答、无法确认、信息不足、需要更多上下文，且尚未执行 web_search，则下一步优先调用 web_search，而不是直接 final。
1.2. 如果当前问题附带图片或 PDF，且需要依据附件内容回答，则 general_chat 不应直接 final，而应先调用 general_chat 工具读取附件内容。
2. local_knowledge_qa、web_search、single_cell_analysis、augmented_analysis 在第一轮通常需要先调用工具；但一旦 transcript 中已有足够 observation，任何 intent 都允许 final。
3. augmented_analysis 是高层任务意图，不是一个必须单独存在的工具。对于 augmented_analysis，你应在当前步从可用工具中选择最合适的一个：
   - 通常先执行 single_cell_analysis
   - 如需专业背景解释，再执行 local_knowledge_qa
   - 如需外部最新信息或官网信息，再执行 web_search
   - 当 observation 已足够时，直接 final
4. 禁止基于关键词、字符片段或字面触发词直接决定工具；必须依据语义判断 intent，并严格按 intent 和 observation 选择当前步最合适的动作。
5. 工具执行后，如果结果已经足够，请直接给出最终回答；只有在确实需要时才继续调用别的工具。
5.1. 如果 local_knowledge_qa 的 observation 明确表示“根据当前知识库内容无法确定”、证据不足、缺少直接支持，或只返回了弱相关内容而不足以支撑回答，则下一步优先调用 web_search，而不是直接 final。
5.2. 当 local_knowledge_qa 和 web_search 都已经有 observation 时，必须综合两者再判断是否 final；不要只看其中一个。
6. 对缩写、模型名、论文名、工具名，若没有可靠依据，不得自行扩写、补充机构来源或杜撰具体用途。
7. 工具之间保持独立。不要假设单个工具会替你完成后续总结、改写或补充解释；如需下一步处理，应由 agent 再决定是否继续调用其他工具或输出最终回答。
8. 如果 transcript 中已经包含某个工具的结果，不要再次调用同一个工具来处理同一轮问题。应优先基于现有工具结果给出最终回答；只有在确实需要补充不同能力时，才调用其他工具。
9. 如果 transcript 中已经有工具 observation，必须显式根据 observation 判断“现在能否回答”。不要忽略 observation。
9.1. 特别地，如果 transcript 中 local_knowledge_qa 的 observation 表示无法确定，且尚未执行 web_search，则应继续调用 web_search。
9.2. 如果 transcript 中 general_chat 的 observation 表示无法回答、信息不足或无法确认，且尚未执行 web_search，则应继续调用 web_search。
9.3. 如果当前轮附带图片或 PDF，而 transcript 中尚无 general_chat observation，则应先执行 general_chat 工具。
10. 当 action=final 时，answer 必须是面向用户的最终中文回答；不要把工具原始 JSON、原始网页结果列表或内部字段原样贴给用户。
11. 当 action=tool 时，plan 必须说明为什么此时需要该工具；当 action=final 时，plan 必须说明为什么现有 observation 已足以回答。
12. 当 action=tool 时，answer 必须为空字符串，不允许提前给出候选答案、模型列表、解释性段落或任何面向用户的正文。
13. 在决定调用工具的阶段，不得根据先验知识直接生成“几种常见模型/方法/论文”的列表；这些内容只能来自后续 observation 或 Final Answer 阶段。
14. 如果网页搜索 observation 只包含泛泛而谈的综述、教程或无关页面，而没有直接支持用户问题的命名实体或证据，则 Final Answer 必须明确说明“当前搜索结果不足以直接回答”，不得硬编具体模型名。
15. 对“列举几种常见模型/方法/论文”这类问题，只有当 observation 中确实出现了这些名称并能支持其与问题相关时，才允许在 Final Answer 中列出。

只返回 JSON，不要输出额外说明。
JSON schema: {{"intent":"<intent>","plan":"...","action":"tool|final","tool_name":"<tool or empty>","reason":"...","answer":"..."}}
当 action=tool 时，tool_name 必须来自下列工具：
{tool_lines}
当前用户问题: {user_text}
附件列表:
{attachment_lines}
""".strip(),
    "decision_repair": """
你上一轮的输出不符合要求，没有返回合法 JSON。
现在请基于相同任务重新给出一个严格合法的 JSON 决策结果。

要求：
1. 只能返回 JSON，不要输出解释、不要输出 Markdown、不要输出代码块。
2. 必须包含字段：intent, plan, action, tool_name, reason, answer。
3. intent 只能是：
   - general_chat
   - local_knowledge_qa
   - web_search
   - single_cell_analysis
   - augmented_analysis
4. 当现有 observation 仍不足时，action 应为 tool；当现有 observation 已足够时，任何 intent 都可以使用 action=final。
5. 当 action=tool 时，tool_name 必须来自可用工具列表；当 action=final 时，tool_name 必须为空字符串。
6. 不允许输出自然语言段落替代 JSON。
7. plan 必须是当前这一步的计划，不得为空。
8. 如果 intent 是 augmented_analysis，不要把 augmented_analysis 当作必须直接执行的单独工具；应从可用工具中选择当前步最合适的一个，或在 observation 足够时 final。
9. 如果 transcript 中已经有工具结果，必须基于该 observation 判断是否可以 final。
9.1. 如果 local_knowledge_qa 的 observation 显示本地知识不足、无法确定或证据不充分，且 web_search 尚未执行，则必须输出 action=tool 且 tool_name=web_search。
9.2. 如果 general_chat 的 observation 显示无法回答、信息不足、无法确认或需要更多外部信息，且 web_search 尚未执行，则必须输出 action=tool 且 tool_name=web_search。
9.3. 如果当前轮附带图片或 PDF，且需要依据附件作答，但 general_chat 尚未执行，则必须输出 action=tool 且 tool_name=general_chat。
10. 如果 action=tool，answer 必须为空字符串。
11. 不允许在 repair 阶段编造用户最终答案、模型列表、论文列表或网页总结。

可用工具：
{tool_lines}

当前用户问题：
{user_text}

附件列表：
{attachment_lines}

当前对话与工具轨迹：
{transcript}

你上一轮的原始输出：
{raw_output}
""".strip(),
    "final_answer_system": (
        "你是单细胞分析 assistant。请基于当前对话与已有工具结果，"
        "给出最终中文回答。若工具结果不足，请明确说明。"
        "你必须阅读 transcript 中的 tool observation，并用自己的话进行整理与回答。"
        "如果 transcript 中同时存在 local_knowledge_qa 与 web_search 的 observation，必须先综合本地知识与网页搜索结果，再给出结论。"
        "如果本地知识不足而网页搜索补充了证据，应明确说明结论主要由网页结果支持；如果两者都不足，也要明确说明证据不足。"
        "只允许依据满足当前本地知识置信度阈值的检索结果做引用与归纳。"
        "只允许依据满足当前网页置信度阈值的结果做引用与归纳。"
        "如果本地知识 observation 中某些检索片段或参考来源 score 很低，或未达到当前阈值，不要引用这些本地结果。"
        "如果网页 observation 中某些结果 score 很低或明显与问题不相关，不要引用这些网页结果。"
        "如果网页搜索 observation 只说明“候选结果未达到阈值”或“没有保留结果”，你只能表述为“当前网页结果未通过阈值/证据不足”；"
        "不要擅自改写成“经过多次尝试”“未找到相关新闻报道”“建议更换搜索引擎”等 observation 中不存在的事实。"
        "不要直接照抄工具返回的 JSON、字段名或原始网页结果列表。"
        "如果 observation 不能直接支持用户问题，请明确说搜索结果不足或证据不足，不要硬凑答案。"
        "只有当 observation 中确实出现并支持某个模型名、方法名、论文名时，才可以在回答中列出它。"
        "如果 observation 只是泛泛介绍单细胞分析流程、教程或综述，不能把它们改写成用户要的深度学习模型列表。"
        "对于缩写、模型名、论文名、工具名，不要在无依据时擅自扩写或补充背景事实。"
    ),
    "transcript_user_template": "当前对话与工具轨迹:\n{transcript}",
}

context_summarizer_prompt = {
    "single_cell_report_context_template": """
用户请求：{user_text}
输入 h5ad：{input_h5ad}
执行 h5ad：{effective_h5ad}
模型目录：{model_dir}
是否请求批次矫正：{need_batch_correction}
是否请求基因相关分析：{need_gene_corr}
基因列表：{gene_list}
n_hvg：{n_hvg}
gene_corr_thr：{gene_corr_thr}
gene_corr_topk：{gene_corr_topk}
批次列：{str_batch}
是否实际使用批次矫正：{used_batch_correction}
运行设备：{runtime_device}
结果 h5ad：{result_h5ad}
""".strip(),
}


answer_prompt = {
    "rag_answer": """
请严格依据给定上下文回答问题。
如果上下文不足以支持结论，请明确说明“根据当前知识库内容无法确定”。
如果同一名称在上下文中对应多个不同概念或论文，请先明确区分，不要混为一谈。
回答中尽量引用来源文件名，不要编造来源。
如果上下文没有给出缩写的全称、机构来源、论文标题或功能描述，不要自行补充。

问题：{query}

上下文：
{context}

请给出中文回答。
""".strip(),
    "augmented_single_cell_analysis": """
请基于以下多源信息进行联合深度分析：
1. 用户原始问题
2. 单细胞分析报告摘要
3. 单细胞结构化分析结果
4. 本地知识库 RAG 结果
5. Web Search 结果
6. 随消息附带的分析图像（如 UMAP、相关网络图等）

回答要求：
1. 优先依据用户上传数据本身得出结论。
2. 结合外部知识解释潜在的生物学意义、方法学背景和局限性。
3. 如果图像与文本结果一致，请明确指出；如果存在不确定性，也要说明。
4. 不要把不确定内容说成确定结论。
5. 输出请按以下结构组织：
   - 核心发现
   - 结合知识库与网页信息的解释
   - 图像观察
   - 局限性
   - 下一步建议
6. 使用中文回答。

用户问题：
{user_text}

单细胞分析报告摘要：
{report_context}

单细胞结构化结果：
{analysis_json}

本地知识库结果：
{rag_text}

网页搜索结果：
{web_text}
""".strip(),
}

def build_single_cell_param_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": tool_call_prompt["single_cell_param_extractor"]},
        {"role": "user", "content": user_text},
    ]

def build_general_chat_messages(
    *,
    user_text: str,
    recent_messages: list[dict[str, Any]] | None = None,
    short_summary: str = "",
    profile: dict[str, Any] | None = None,
    pdf_contexts: list[dict[str, Any]] | None = None,
    image_files: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    system_lines = [
        system_prompt["general_chat_assistant"],
        system_prompt["general_chat_memory_hint"],
        "如果当前消息附带图片或 PDF 摘录，优先依据这些附件内容回答，不要忽略附件。",
        "如果当前消息附带图片，你可以直接查看图片内容并进行分析，不要要求用户先把图片内容转写成文字。",
        "只有在图片本身严重模糊、无法辨认或附件损坏时，才说明无法判断的原因。",
    ]
    if short_summary.strip():
        system_lines.append(f"会话摘要: {short_summary.strip()}")
    if profile:
        system_lines.append(
            f"用户长期偏好: {json.dumps(profile, ensure_ascii=False)}"
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "\n".join(system_lines)}
    ]
    dialog_turns = [
        item
        for item in (recent_messages or [])
        if str(item.get("role") or "") in {"user", "assistant"}
        and str(item.get("content") or "").strip()
    ]
    dialog_turns = dialog_turns[-10:]
    for item in dialog_turns:
        messages.append(
            {
                "role": str(item.get("role") or "user"),
                "content": str(item.get("content") or "").strip(),
            }
        )
    if not dialog_turns or str(dialog_turns[-1].get("role") or "") != "user":
        prompt_sections: list[str] = []
        normalized_user_text = user_text.strip()
        prompt_sections.append(
            f"当前用户问题：\n{normalized_user_text or '请基于当前上传附件进行分析与回答。'}"
        )

        if pdf_contexts:
            pdf_blocks: list[str] = []
            for item in pdf_contexts[:3]:
                title = str(item.get("name") or "uploaded.pdf")
                excerpt = str(item.get("excerpt") or "").strip()
                if not excerpt:
                    continue
                pdf_blocks.append(f"[{title}]\n{excerpt}")
            if pdf_blocks:
                prompt_sections.append("当前上传 PDF 摘录：\n" + "\n\n".join(pdf_blocks))

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "\n\n".join(prompt_sections),
            }
        ]
        for image in (image_files or [])[:6]:
            image_name = str(image.get("name") or "uploaded_image")
            user_content.append(
                {
                    "type": "text",
                    "text": f"下面这张图片是当前用户上传的附件：{image_name}。请直接观察图片内容并回答。",
                }
            )
            user_content.append(
                {
                    "type": "image",
                    "image": str(image.get("data_url") or ""),
                }
            )
        messages.append({"role": "user", "content": user_content})
    return messages


def build_followup_contextualize_messages(
    *,
    user_text: str,
    recent_messages: list[dict[str, Any]] | None = None,
    short_summary: str = "",
) -> list[dict[str, str]]:
    history_lines: list[str] = []
    if short_summary.strip():
        history_lines.extend(
            [
                "会话摘要：",
                short_summary.strip(),
                "",
            ]
        )

    dialog_turns = [
        item
        for item in (recent_messages or [])
        if str(item.get("role") or "") in {"user", "assistant"}
        and str(item.get("content") or "").strip()
    ]
    for item in dialog_turns[-8:]:
        role = "用户" if str(item.get("role") or "") == "user" else "助手"
        history_lines.append(f"{role}: {str(item.get('content') or '').strip()}")

    history_text = "\n".join(history_lines).strip() or "无可用历史。"
    return [
        {"role": "system", "content": query_rewrite_prompt["followup_contextualize"]},
        {
            "role": "user",
            "content": (
                f"最近对话：\n{history_text}\n\n"
                f"当前用户问题：\n{user_text.strip() or '(empty)'}"
            ),
        },
    ]


def build_react_deliberation_messages(
    *,
    user_text: str,
    transcript: str,
    tool_lines: list[str],
    attachment_lines: list[str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": react_agent_prompt["intent_recognizer"].format(
                user_text=user_text,
                attachment_lines="\n".join(attachment_lines),
            ),
        },
        {
            "role": "system",
            "content": react_agent_prompt["tool_router"].format(
                user_text=user_text,
                tool_lines="\n".join(tool_lines),
                attachment_lines="\n".join(attachment_lines),
            ),
        },
        {
            "role": "user",
            "content": react_agent_prompt["transcript_user_template"].format(
                transcript=transcript
            ),
        },
    ]


def build_react_final_answer_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": react_agent_prompt["final_answer_system"],
        },
        {
            "role": "user",
            "content": react_agent_prompt["transcript_user_template"].format(
                transcript=transcript
            ),
        },
    ]


def build_react_decision_repair_messages(
    *,
    user_text: str,
    transcript: str,
    tool_lines: list[str],
    attachment_lines: list[str],
    raw_output: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": react_agent_prompt["decision_repair"].format(
                user_text=user_text,
                transcript=transcript,
                tool_lines="\n".join(tool_lines),
                attachment_lines="\n".join(attachment_lines),
                raw_output=raw_output,
            ),
        }
    ]
def build_rag_user_prompt(query: str, context: str) -> str:
    return answer_prompt["rag_answer"].format(query=query, context=context)


def build_augmented_analysis_messages(
    *,
    user_text: str,
    report_context: str,
    analysis_json: str,
    rag_text: str,
    web_text: str,
    image_files: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": system_prompt["augmented_multimodal_analysis"]},
        {
            "type": "text",
            "text": answer_prompt["augmented_single_cell_analysis"].format(
                user_text=user_text,
                report_context=report_context,
                analysis_json=analysis_json,
                rag_text=rag_text,
                web_text=web_text,
            ),
        },
    ]
    for image in (image_files or [])[:6]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image["data_url"]},
            }
        )
    return [{"role": "user", "content": content}]


def build_single_cell_report_context(values: dict[str, Any]) -> str:
    return context_summarizer_prompt["single_cell_report_context_template"].format(**values)


RAG_SYSTEM_PROMPT = system_prompt["rag_grounded_qa"]
SINGLE_CELL_PARAM_SCHEMA_PROMPT = tool_call_prompt["single_cell_param_extractor"]
