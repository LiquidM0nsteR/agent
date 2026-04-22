export function formatRouteIntent(value) {
  const labels = {
    general_chat: "通用对话",
    local_knowledge_qa: "本地知识库问答",
    web_search: "网页搜索",
    single_cell_analysis: "单细胞分析",
    augmented_analysis: "增强分析",
  };
  return labels[value] || value || "";
}

function formatRouteTool(value) {
  const labels = {
    direct_llm: "基础对话模型",
    local_knowledge_base: "本地知识库问答",
    web_search: "网页搜索",
    single_cell_pipeline: "单细胞分析流程",
  };
  return labels[value] || value || "";
}

function buildRouteTraceModel(source) {
  if (!source || typeof source !== "object") {
    return null;
  }
  const reason = String(source.reason || "").trim();
  const selectedTools = Array.isArray(source.selected_tools)
    ? source.selected_tools.map((item) => formatRouteTool(String(item || ""))).filter(Boolean)
    : [];
  const executionSteps = Array.isArray(source.execution_steps)
    ? source.execution_steps
        .map((item) => ({
          description: String(item?.description || "").trim(),
          status: String(item?.status || "").trim(),
          toolName: formatRouteTool(String(item?.tool_name || "")),
        }))
        .filter((item) => item.description)
    : [];
  const intentLabel = formatRouteIntent(String(source.intent || ""));
  const dispatchedLabel = formatRouteIntent(String(source.dispatched_node || ""));
  const llmTraces = Array.isArray(source.llm_traces)
    ? source.llm_traces
        .map((item) => ({
          label: String(item?.label || ""),
          response: String(item?.response || item?.response_preview || ""),
          promptPreview: String(item?.prompt_preview || ""),
          modelPath: String(item?.model_path || ""),
          elapsedMs: item?.elapsed_ms,
        }))
        .filter((item) => item.response)
    : [];

  if (
    !reason &&
    !selectedTools.length &&
    !executionSteps.length &&
    !intentLabel &&
    !dispatchedLabel &&
    !llmTraces.length
  ) {
    return null;
  }

  if (llmTraces.length) {
    return {
      llmTraces,
    };
  }

  return {
    reason,
    selectedTools,
    executionSteps,
    intentLabel,
    dispatchedLabel,
    llmTraces,
  };
}

function appendRouteMetaItem(target, label, value) {
  if (!value) {
    return;
  }
  const item = document.createElement("div");
  item.className = "route-meta-item";

  const itemLabel = document.createElement("span");
  itemLabel.className = "route-meta-label";
  itemLabel.textContent = label;

  const itemValue = document.createElement("span");
  itemValue.className = "route-meta-value";
  itemValue.textContent = value;

  item.append(itemLabel, itemValue);
  target.append(item);
}

function renderStreamRouteEvents(slots) {
  const events = Array.isArray(slots.routeStreamEvents) ? slots.routeStreamEvents : [];
  if (!events.length) {
    return false;
  }

  const streamSection = document.createElement("div");
  streamSection.className = "route-section";

  const title = document.createElement("div");
  title.className = "route-section-title";
  title.textContent = "实时思考";

  const list = document.createElement("ol");
  list.className = "route-step-list";

  events.forEach((event) => {
    const item = document.createElement("li");
    item.className = "route-step-item";

    const desc = document.createElement("div");
    desc.className = "route-step-text";
    desc.textContent = event.title || event.message || "处理中";
    item.append(desc);

    if (event.detail) {
      const meta = document.createElement("div");
      meta.className = "route-step-meta";
      meta.textContent = event.detail;
      item.append(meta);
    }

    list.append(item);
  });

  streamSection.append(title, list);
  slots.routeBody.append(streamSection);
  return true;
}

export function renderRouteTrace(slots, source) {
  slots.routeTraceSource = source || slots.routeTraceSource || null;
  const model = buildRouteTraceModel(slots.routeTraceSource);
  if (!model && !slots.routeStreamEvents?.length) {
    slots.routeDetails.classList.add("is-hidden");
    slots.routeBody.innerHTML = "";
    return;
  }

  slots.routeBody.innerHTML = "";
  renderStreamRouteEvents(slots);

  if (model?.llmTraces?.length) {
    const tracesSection = document.createElement("div");
    tracesSection.className = "route-section";

    const title = document.createElement("div");
    title.className = "route-section-title";
    title.textContent = "LLM 原始输出";

    const list = document.createElement("div");
    list.className = "llm-trace-list";

    model.llmTraces.forEach((trace) => {
      const item = document.createElement("article");
      item.className = "llm-trace-item";

      const response = document.createElement("div");
      response.className = "llm-trace-response";
      response.textContent = trace.response;
      item.append(response);

      if (trace.elapsedMs) {
        const meta = document.createElement("div");
        meta.className = "route-step-meta";
        meta.textContent = `${trace.label || "llm"} · ${Number(trace.elapsedMs).toFixed(0)} ms`;
        item.append(meta);
      }

      list.append(item);
    });

    tracesSection.append(title, list);
    slots.routeBody.append(tracesSection);
  }

  if (model?.reason) {
    const reason = document.createElement("p");
    reason.className = "route-reason";
    reason.textContent = model.reason;
    slots.routeBody.append(reason);
  }

  if (model?.selectedTools?.length) {
    const toolsSection = document.createElement("div");
    toolsSection.className = "route-section";

    const title = document.createElement("div");
    title.className = "route-section-title";
    title.textContent = "调用工具";

    const chips = document.createElement("div");
    chips.className = "route-tool-chips";
    model.selectedTools.forEach((toolLabel) => {
      const chip = document.createElement("span");
      chip.className = "route-tool-chip";
      chip.textContent = toolLabel;
      chips.append(chip);
    });
    toolsSection.append(title, chips);
    slots.routeBody.append(toolsSection);
  }

  if (model?.executionSteps?.length) {
    const stepsSection = document.createElement("div");
    stepsSection.className = "route-section";

    const title = document.createElement("div");
    title.className = "route-section-title";
    title.textContent = "执行步骤";

    const list = document.createElement("ol");
    list.className = "route-step-list";

    model.executionSteps.forEach((step) => {
      const item = document.createElement("li");
      item.className = "route-step-item";

      const desc = document.createElement("div");
      desc.className = "route-step-text";
      desc.textContent = step.description;
      item.append(desc);

      const meta = document.createElement("div");
      meta.className = "route-step-meta";
      meta.textContent = [step.toolName, step.status].filter(Boolean).join(" · ");
      item.append(meta);

      list.append(item);
    });

    stepsSection.append(title, list);
    slots.routeBody.append(stepsSection);
  }

  if (model?.intentLabel || model?.dispatchedLabel) {
    const meta = document.createElement("div");
    meta.className = "route-meta-grid";
    appendRouteMetaItem(meta, "意图", model.intentLabel);
    appendRouteMetaItem(meta, "执行节点", model.dispatchedLabel);
    if (meta.childElementCount) {
      slots.routeBody.append(meta);
    }
  }

  slots.routeDetails.classList.remove("is-hidden");
}
