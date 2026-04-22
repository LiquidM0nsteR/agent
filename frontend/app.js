const STORAGE_KEYS = {
  userId: "agent_user_id",
  sessionId: "agent_active_session_id",
};
const DEFAULT_LOCAL_SOURCE_MIN_SCORE = 0.35;
const DEFAULT_WEB_SOURCE_MIN_SCORE = 1.5;

const form = document.getElementById("agent-form");
const filesInput = document.getElementById("files");
const fileList = document.getElementById("file-list");
const fileCount = document.getElementById("file-count");
const selectedFilesPanel = document.getElementById("selected-files");
const submitButton = document.getElementById("submit-button");
const chatPanel = document.getElementById("result-panel");
const chatThread = document.getElementById("chat-thread");
const chatEmpty = document.getElementById("chat-empty");
const textInput = document.getElementById("text");
const sourceDrawer = document.getElementById("source-drawer");
const sourceOverlay = document.getElementById("source-overlay");
const sourceClose = document.getElementById("source-close");
const sourceTabs = Array.from(document.querySelectorAll("[data-source-tab]"));
const sourceSummary = document.getElementById("source-summary");
const sourceLocalPanel = document.getElementById("source-local-panel");
const sourceWebPanel = document.getElementById("source-web-panel");
const authGate = document.getElementById("auth-gate");
const workspaceShell = document.getElementById("workspace-shell");
const loginForm = document.getElementById("login-form");
const loginInput = document.getElementById("login-user-id");
const loginButton = document.getElementById("login-button");
const logoutButton = document.getElementById("logout-button");
const newChatButton = document.getElementById("new-chat-button");
const sessionList = document.getElementById("session-list");
const sessionCount = document.getElementById("session-count");
const activeUserLabel = document.getElementById("active-user-label");
const activeSessionLabel = document.getElementById("active-session-label");
const activeSessionIdLabel = document.getElementById("active-session-id");
const quickCards = Array.from(document.querySelectorAll("[data-quick-prompt]"));
const workspaceSettingsButton = document.getElementById("workspace-settings-button");
const workspaceKnowledgeButton = document.getElementById("workspace-knowledge-button");
const workspaceToolsButton = document.getElementById("workspace-tools-button");
const workspaceOverlay = document.getElementById("workspace-overlay");
const workspaceDrawer = document.getElementById("workspace-drawer");
const workspaceClose = document.getElementById("workspace-close");
const workspaceTabs = Array.from(document.querySelectorAll("[data-workspace-tab]"));
const workspaceSummary = document.getElementById("workspace-summary");
const workspaceSettingsPanel = document.getElementById("workspace-settings-panel");
const workspaceKnowledgePanel = document.getElementById("workspace-knowledge-panel");
const workspaceToolsPanel = document.getElementById("workspace-tools-panel");

let activeController = null;
let toolStatusTimer = null;
let sourceState = {
  tab: "local",
  focusReferenceIndex: null,
  data: null,
};
let workspaceState = {
  tab: "settings",
  settings: null,
  knowledgeFiles: [],
  toolStatus: null,
};
let appState = {
  userId: "",
  activeSessionId: "",
  sessions: [],
};

async function parseJsonResponse(response) {
  const rawText = await response.text();
  try {
    return JSON.parse(rawText);
  } catch {
    return {
      detail: rawText || `HTTP ${response.status}`,
      _raw: rawText,
    };
  }
}

function parseSseEventBlock(block) {
  const lines = String(block || "").split(/\r?\n/);
  let eventType = "message";
  const dataLines = [];

  lines.forEach((line) => {
    if (!line || line.startsWith(":")) {
      return;
    }
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim() || "message";
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  });

  if (!dataLines.length) {
    return null;
  }

  const rawData = dataLines.join("\n");
  try {
    return {
      type: eventType,
      data: JSON.parse(rawData),
    };
  } catch {
    return {
      type: eventType,
      data: { raw: rawData },
    };
  }
}

async function consumeEventStream(response, onEvent) {
  if (!response.body) {
    throw new Error("浏览器不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary < 0) {
        break;
      }
      const rawBlock = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseEventBlock(rawBlock);
      if (parsed) {
        const shouldContinue = onEvent(parsed);
        if (shouldContinue === false) {
          await reader.cancel();
          return;
        }
      }
    }
  }

  buffer += decoder.decode();
  const tail = parseSseEventBlock(buffer.trim());
  if (tail) {
    onEvent(tail);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatFileSize(file) {
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB >= 1) {
    return `${sizeMB.toFixed(2)} MB`;
  }

  const sizeKB = file.size / 1024;
  return `${sizeKB.toFixed(1)} KB`;
}

function formatTimestamp(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatReferenceLabel(reference, index) {
  const parts = [`[${index + 1}]`, reference.file_name || "未知来源"];
  if (reference.page !== null && reference.page !== undefined) {
    parts.push(`p.${reference.page}`);
  }
  if (reference.section) {
    parts.push(reference.section);
  }
  return parts.join(" ");
}

function getLocalSourceMinScore() {
  const value = Number(workspaceState?.settings?.local_source_min_score);
  if (Number.isFinite(value)) {
    return value;
  }
  return DEFAULT_LOCAL_SOURCE_MIN_SCORE;
}

function getWebSourceMinScore() {
  const value = Number(workspaceState?.settings?.web_source_min_score);
  if (Number.isFinite(value)) {
    return value;
  }
  return DEFAULT_WEB_SOURCE_MIN_SCORE;
}

function filterWebResultsByScore(results) {
  const threshold = getWebSourceMinScore();
  return (results || []).filter((item) => Number(item?.score || 0) >= threshold);
}

function filterReferencesByScore(references) {
  const localThreshold = getLocalSourceMinScore();
  const webThreshold = getWebSourceMinScore();
  return (references || []).filter((item) => {
    const docType = String(item?.doc_type || "").toLowerCase();
    if (docType === "web") {
      return Number(item?.score || 0) >= webThreshold;
    }
    if (item?.score == null) {
      return true;
    }
    return Number(item?.score || 0) >= localThreshold;
  });
}

function filterLocalChunksByScore(chunks) {
  const threshold = getLocalSourceMinScore();
  return (chunks || []).filter((item) => {
    if (item?.score == null) {
      return true;
    }
    return Number(item?.score || 0) >= threshold;
  });
}

function autoResizeTextarea() {
  textInput.style.height = "0px";
  textInput.style.height = `${Math.min(textInput.scrollHeight, 240)}px`;
}

function detectFileKind(file) {
  const name = String(file?.name || "").toLowerCase();
  const contentType = String(file?.type || "").toLowerCase();
  if (name.endsWith(".h5ad")) {
    return "h5ad";
  }
  if (name.endsWith(".pdf") || contentType === "application/pdf") {
    return "pdf";
  }
  if (contentType.startsWith("image/")) {
    return "image";
  }
  return "file";
}

function formatAttachmentLabel(file) {
  const kind = detectFileKind(file);
  const kindLabelMap = {
    image: "图片",
    pdf: "PDF",
    h5ad: "h5ad",
    file: "文件",
  };
  return `${kindLabelMap[kind] || "文件"} | ${file.name} (${formatFileSize(file)})`;
}

function renderFiles() {
  const files = Array.from(filesInput.files || []);
  fileList.innerHTML = "";
  fileCount.textContent = `${files.length} files`;
  selectedFilesPanel.classList.toggle("is-collapsed", files.length === 0);

  if (!files.length) {
    const item = document.createElement("li");
    item.textContent = "未选择文件";
    fileList.appendChild(item);
    return;
  }

  files.forEach((file) => {
    const item = document.createElement("li");
    item.textContent = formatAttachmentLabel(file);
    fileList.appendChild(item);
  });
}

function ensureChatVisible() {
  chatEmpty.classList.add("is-hidden");
}

function scrollChatToBottom() {
  requestAnimationFrame(() => {
    chatPanel.scrollTop = chatPanel.scrollHeight;
  });
}

function createMessageShell(role) {
  const shell = document.createElement("article");
  shell.className = `chat-message chat-message-${role}`;

  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-bubble-${role}`;

  shell.appendChild(bubble);
  return { shell, bubble };
}

function appendUserMessage(text, files, createdAt = "") {
  const { shell, bubble } = createMessageShell("user");

  const body = document.createElement("div");
  body.className = "chat-text";
  body.textContent = text.trim() || "Files uploaded";
  bubble.append(body);

  if (files.length) {
    const attachmentList = document.createElement("ul");
    attachmentList.className = "chat-attachment-list";
    files.forEach((file) => {
      const item = document.createElement("li");
      item.textContent = formatAttachmentLabel(file);
      attachmentList.appendChild(item);
    });
    bubble.appendChild(attachmentList);
  }

  if (createdAt) {
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = formatTimestamp(createdAt);
    bubble.appendChild(meta);
  }

  chatThread.appendChild(shell);
  scrollChatToBottom();
}

function appendAssistantPlaceholder() {
  const { shell, bubble } = createMessageShell("assistant");

  const answerContent = document.createElement("div");
  answerContent.className = "chat-text";
  answerContent.textContent = "正在生成回复...";

  const sourceBar = document.createElement("div");
  sourceBar.className = "source-bar is-hidden";

  const sourceButton = document.createElement("button");
  sourceButton.type = "button";
  sourceButton.className = "source-button";
  sourceButton.textContent = "来源";

  const sourceHint = document.createElement("span");
  sourceHint.className = "source-hint";

  const artifactBar = document.createElement("div");
  artifactBar.className = "artifact-bar is-hidden";

  const routeDetails = document.createElement("details");
  routeDetails.className = "route-details is-hidden";
  const routeSummary = document.createElement("summary");
  routeSummary.className = "route-summary";
  routeSummary.textContent = "思考与路由过程";
  const routeBody = document.createElement("div");
  routeBody.className = "route-body";
  routeDetails.append(routeSummary, routeBody);

  sourceBar.append(sourceButton, sourceHint);
  bubble.append(answerContent, artifactBar, routeDetails, sourceBar);
  chatThread.appendChild(shell);
  scrollChatToBottom();

  return {
    shell,
    bubble,
    answerContent,
    sourceBar,
    sourceButton,
    sourceHint,
    artifactBar,
    routeDetails,
    routeSummary,
    routeBody,
    routeStreamEvents: [],
    routeTraceSource: null,
    streamedAnswer: "",
    sourceData: {
      localAnswer: "",
      webPossibleAnswer: "",
      webResults: [],
      references: [],
      chunks: [],
      trace: null,
      raw: "",
    },
  };
}

function appendAssistantHistoryMessage(text, createdAt = "", metadata = {}) {
  const slots = appendAssistantPlaceholder();
  renderAnswerContent(slots.answerContent, text, [], () => {});
  slots.sourceBar.classList.add("is-hidden");
  slots.artifactBar.classList.add("is-hidden");
  renderRouteTrace(slots, metadata?.route_trace || metadata || null);

  if (createdAt) {
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = formatTimestamp(createdAt);
    slots.bubble.appendChild(meta);
  }
}

function renderAnswerContent(target, answerText, references, onReferenceClick) {
  target.innerHTML = "";
  const paragraphs = String(answerText || "未返回结构化结果。")
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);

  const normalizedParagraphs = paragraphs.length
    ? paragraphs
    : ["未返回结构化结果。"];

  normalizedParagraphs.forEach((paragraph, index) => {
    const node = document.createElement("p");
    node.textContent = paragraph;

    if (index === normalizedParagraphs.length - 1 && references.length) {
      const inlineList = document.createElement("span");
      inlineList.className = "inline-citations";

      references.slice(0, 3).forEach((reference, refIndex) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "inline-citation";
        button.textContent = `[${refIndex + 1}]`;
        button.title = formatReferenceLabel(reference, refIndex);
        button.addEventListener("click", () => onReferenceClick(refIndex));
        inlineList.appendChild(button);
      });

      node.appendChild(document.createTextNode(" "));
      node.appendChild(inlineList);
    }

    target.appendChild(node);
  });
}

function buildReferenceItem(reference, index) {
  const item = document.createElement("article");
  item.className = "drawer-card";
  item.id = `source-reference-${index}`;

  const title = document.createElement("div");
  title.className = "drawer-card-title";
  title.textContent = formatReferenceLabel(reference, index);

  const meta = document.createElement("p");
  meta.className = "drawer-card-copy";
  const metaParts = [reference.source_path || "未知来源路径"];
  if (String(reference.doc_type || "").toLowerCase() === "web") {
    metaParts.push(`score=${Number(reference.score || 0).toFixed(2)}`);
  }
  meta.textContent = metaParts.join(" | ");

  item.append(title, meta);
  return item;
}

function buildChunkItem(chunk, index) {
  const item = document.createElement("article");
  item.className = "drawer-card";

  const title = document.createElement("div");
  title.className = "drawer-card-title";
  title.textContent = `Chunk ${index + 1} | ${chunk.metadata?.file_name || "unknown"}`;

  const body = document.createElement("p");
  body.className = "drawer-card-copy";
  body.textContent = chunk.text || "";

  const meta = document.createElement("div");
  meta.className = "drawer-card-meta";
  meta.textContent = `score=${Number(chunk.score || 0).toFixed(4)} | source=${chunk.retrieval_source || "unknown"}`;

  item.append(title, body, meta);
  return item;
}

function buildWebItem(result) {
  const item = document.createElement("article");
  item.className = "drawer-card";

  const titleRow = document.createElement("div");
  titleRow.className = "drawer-web-head";

  if (result.url) {
    const icon = document.createElement("img");
    icon.className = "drawer-favicon";
    icon.alt = "";
    icon.src = `https://www.google.com/s2/favicons?sz=64&domain_url=${encodeURIComponent(result.url)}`;
    titleRow.appendChild(icon);
  }

  const title = document.createElement("div");
  title.className = "drawer-card-title";
  title.textContent = `[${result.source_tier || "community"}] ${result.title || "Untitled"}`;
  titleRow.appendChild(title);

  const body = document.createElement("p");
  body.className = "drawer-card-copy";
  body.textContent = result.snippet || "";

  const meta = document.createElement("div");
  meta.className = "drawer-card-meta";
  meta.textContent = `score=${Number(result.score || 0).toFixed(2)} | tier=${result.source_tier || "web"}`;

  item.append(titleRow, body);
  item.appendChild(meta);

  if (result.url) {
    const link = document.createElement("a");
    link.className = "drawer-link";
    link.href = result.url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = result.url;
    item.appendChild(link);
  }

  return item;
}

function openSourceDrawer(data, tab, focusReferenceIndex = null) {
  sourceState = {
    tab,
    focusReferenceIndex,
    data,
  };

  sourceSummary.textContent = data.localAnswer || "Selected answer sources";
  sourceTabs.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.sourceTab === tab);
  });

  sourceLocalPanel.innerHTML = "";
  sourceWebPanel.innerHTML = "";

  if (data.references.length || data.chunks.length || data.trace || data.raw) {
    const localSections = [];

    if (data.references.length) {
      const referencesBlock = document.createElement("section");
      referencesBlock.className = "drawer-section";

      const title = document.createElement("h3");
      title.className = "drawer-section-title";
      title.textContent = "参考来源";
      referencesBlock.appendChild(title);

      data.references.forEach((reference, index) => {
        referencesBlock.appendChild(buildReferenceItem(reference, index));
      });

      localSections.push(referencesBlock);
    }

    if (data.chunks.length) {
      const chunksBlock = document.createElement("section");
      chunksBlock.className = "drawer-section";

      const title = document.createElement("h3");
      title.className = "drawer-section-title";
      title.textContent = "检索片段";
      chunksBlock.appendChild(title);

      data.chunks.forEach((chunk, index) => {
        chunksBlock.appendChild(buildChunkItem(chunk, index));
      });

      localSections.push(chunksBlock);
    }

    if (data.trace) {
      const traceBlock = document.createElement("section");
      traceBlock.className = "drawer-section";

      const title = document.createElement("h3");
      title.className = "drawer-section-title";
      title.textContent = "检索轨迹";

      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(data.trace, null, 2);

      traceBlock.append(title, pre);
      localSections.push(traceBlock);
    }

    if (data.raw) {
      const rawBlock = document.createElement("section");
      rawBlock.className = "drawer-section";

      const title = document.createElement("h3");
      title.className = "drawer-section-title";
      title.textContent = "原始返回";

      const pre = document.createElement("pre");
      pre.textContent = data.raw;

      rawBlock.append(title, pre);
      localSections.push(rawBlock);
    }

    localSections.forEach((section) => sourceLocalPanel.appendChild(section));
  } else {
    sourceLocalPanel.innerHTML = '<p class="drawer-empty">暂无本地来源详情。</p>';
  }

  if (data.webPossibleAnswer || data.webResults.length) {
    if (data.webPossibleAnswer) {
      const answerBlock = document.createElement("section");
      answerBlock.className = "drawer-section";

      const title = document.createElement("h3");
      title.className = "drawer-section-title";
      title.textContent = "网页候选答案";

      const body = document.createElement("p");
      body.className = "drawer-card-copy";
      body.textContent = data.webPossibleAnswer;

      answerBlock.append(title, body);
      sourceWebPanel.appendChild(answerBlock);
    }

    if (data.webResults.length) {
      const resultsBlock = document.createElement("section");
      resultsBlock.className = "drawer-section";

      const title = document.createElement("h3");
      title.className = "drawer-section-title";
      title.textContent = "网页检索结果";
      resultsBlock.appendChild(title);

      data.webResults.forEach((result) => {
        resultsBlock.appendChild(buildWebItem(result));
      });

      sourceWebPanel.appendChild(resultsBlock);
    }
  } else {
    sourceWebPanel.innerHTML = '<p class="drawer-empty">暂无网页来源详情。</p>';
  }

  sourceLocalPanel.classList.toggle("is-hidden", tab !== "local");
  sourceWebPanel.classList.toggle("is-hidden", tab !== "web");
  sourceOverlay.classList.remove("is-hidden");
  sourceDrawer.classList.add("is-open");

  if (tab === "local" && focusReferenceIndex !== null) {
    requestAnimationFrame(() => {
      const target = document.getElementById(`source-reference-${focusReferenceIndex}`);
      if (target) {
        target.scrollIntoView({ block: "center", behavior: "smooth" });
        target.classList.add("is-focused");
        window.setTimeout(() => target.classList.remove("is-focused"), 1200);
      }
    });
  }
}

function closeSourceDrawer() {
  sourceDrawer.classList.remove("is-open");
  sourceOverlay.classList.add("is-hidden");
}

function stopToolStatusPolling() {
  if (toolStatusTimer) {
    window.clearInterval(toolStatusTimer);
    toolStatusTimer = null;
  }
}

async function fetchWorkspaceSettings() {
  if (!appState.userId) return null;
  const response = await fetch(
    `/api/users/${encodeURIComponent(appState.userId)}/workspace/settings`,
  );
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "加载设置失败");
  return payload.settings || {};
}

async function saveWorkspaceSettings(settings) {
  if (!appState.userId) return null;
  const response = await fetch(
    `/api/users/${encodeURIComponent(appState.userId)}/workspace/settings`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings }),
    },
  );
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "保存设置失败");
  return payload.settings || {};
}

async function clearWorkspaceMemory(scope) {
  if (!appState.userId) return;
  const response = await fetch(
    `/api/users/${encodeURIComponent(appState.userId)}/workspace/memory/clear`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope,
        session_id: scope === "session" ? appState.activeSessionId : "",
      }),
    },
  );
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "清理记忆失败");
  return payload;
}

async function fetchKnowledgeFiles() {
  const response = await fetch("/api/workspace/knowledge/files");
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "加载文件列表失败");
  return payload.files || [];
}

async function uploadKnowledgeFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const response = await fetch("/api/workspace/knowledge/files", {
    method: "POST",
    body: formData,
  });
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "上传文件失败");
  return payload.files || [];
}

async function deleteKnowledgeFile(path) {
  const response = await fetch(
    `/api/workspace/knowledge/files/${encodeURIComponent(path)}`,
    { method: "DELETE" },
  );
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "删除文件失败");
  return payload.files || [];
}

async function rebuildKnowledgeIndex() {
  const response = await fetch("/api/workspace/knowledge/rebuild-index", {
    method: "POST",
  });
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "重建索引失败");
  return payload;
}

async function fetchToolStatus() {
  const response = await fetch("/api/workspace/tool-status");
  const payload = await parseJsonResponse(response);
  if (!response.ok) throw new Error(payload.detail || "加载工具状态失败");
  return payload.tools || {};
}

function openWorkspaceDrawer(tab = "settings") {
  if (!workspaceDrawer || !workspaceOverlay) {
    return;
  }
  workspaceState.tab = tab;
  workspaceSettingsButton?.classList.toggle("is-active", tab === "settings");
  workspaceKnowledgeButton?.classList.toggle("is-active", tab === "knowledge");
  workspaceToolsButton?.classList.toggle("is-active", tab === "tools");
  workspaceTabs.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.workspaceTab === tab);
  });
  workspaceSettingsPanel.classList.toggle("is-hidden", tab !== "settings");
  workspaceKnowledgePanel.classList.toggle("is-hidden", tab !== "knowledge");
  workspaceToolsPanel.classList.toggle("is-hidden", tab !== "tools");
  workspaceOverlay.classList.remove("is-hidden");
  workspaceDrawer.classList.add("is-open");
  void refreshWorkspaceTab(tab);
}

function closeWorkspaceDrawer() {
  if (!workspaceDrawer || !workspaceOverlay) {
    return;
  }
  workspaceDrawer.classList.remove("is-open");
  workspaceOverlay.classList.add("is-hidden");
  workspaceSettingsButton?.classList.remove("is-active");
  workspaceKnowledgeButton?.classList.remove("is-active");
  workspaceToolsButton?.classList.remove("is-active");
  stopToolStatusPolling();
}

function renderWorkspaceSettingsPanel() {
  const settings = workspaceState.settings || {};
  workspaceSummary.textContent = "常用参数、记忆管理与搜索偏好。";
  workspaceSettingsPanel.innerHTML = `
    <section class="drawer-section">
      <h3 class="drawer-section-title">常用参数</h3>
      <div class="workspace-form-grid">
        <label class="workspace-field">温度（Temperature）
          <input id="ws-temperature" type="number" step="0.1" min="0" max="2" value="${Number(settings.temperature ?? 0.2)}" />
        </label>
        <label class="workspace-field">最大生成长度（Max New Tokens）
          <input id="ws-max-new-tokens" type="number" min="64" max="4096" value="${Number(settings.max_new_tokens ?? 512)}" />
        </label>
        <label class="workspace-field">短期记忆最大消息数
          <input id="ws-short-msg" type="number" min="4" max="100" value="${Number(settings.short_term_max_messages ?? 12)}" />
        </label>
        <label class="workspace-field">摘要触发阈值
          <input id="ws-summary-threshold" type="number" min="2" max="100" value="${Number(settings.short_term_summary_threshold ?? 8)}" />
        </label>
        <label class="workspace-field">长期记忆检索 Top K
          <input id="ws-long-topk" type="number" min="1" max="20" value="${Number(settings.long_term_top_k ?? 3)}" />
        </label>
        <label class="workspace-field">本地知识最低置信度
          <input id="ws-local-min-score" type="number" step="0.05" min="0" max="10" value="${Number(settings.local_source_min_score ?? DEFAULT_LOCAL_SOURCE_MIN_SCORE)}" />
        </label>
        <label class="workspace-field">网页来源最低置信度
          <input id="ws-web-min-score" type="number" step="0.1" min="0" max="10" value="${Number(settings.web_source_min_score ?? DEFAULT_WEB_SOURCE_MIN_SCORE)}" />
        </label>
      </div>
      <div class="workspace-checkline">
        <label><input id="ws-profile-memory" type="checkbox" ${settings.enable_profile_memory ? "checked" : ""}/> 启用 Profile Memory</label>
        <label><input id="ws-semantic-memory" type="checkbox" ${settings.enable_semantic_memory ? "checked" : ""}/> 启用 Semantic Memory</label>
      </div>
      <h3 class="drawer-section-title">搜索偏好</h3>
      <div class="workspace-checkline">
        <label><input id="ws-prefers-official" type="checkbox" ${settings.search_prefers_official_sources ? "checked" : ""}/> 优先官方来源</label>
      </div>
      <div class="workspace-inline-actions">
        <button id="ws-save" type="button">保存设置</button>
        <button id="ws-clear-session-memory" type="button" class="ghost-button">清空当前会话记忆</button>
        <button id="ws-clear-all-memory" type="button" class="ghost-button">清空全部记忆</button>
      </div>
    </section>
  `;
  document.getElementById("ws-save")?.addEventListener("click", async () => {
    try {
      const updated = await saveWorkspaceSettings({
        temperature: Number(document.getElementById("ws-temperature")?.value || 0.2),
        max_new_tokens: Number(document.getElementById("ws-max-new-tokens")?.value || 512),
        short_term_max_messages: Number(document.getElementById("ws-short-msg")?.value || 12),
        short_term_summary_threshold: Number(document.getElementById("ws-summary-threshold")?.value || 8),
        long_term_top_k: Number(document.getElementById("ws-long-topk")?.value || 3),
        local_source_min_score: Number(document.getElementById("ws-local-min-score")?.value || DEFAULT_LOCAL_SOURCE_MIN_SCORE),
        web_source_min_score: Number(document.getElementById("ws-web-min-score")?.value || DEFAULT_WEB_SOURCE_MIN_SCORE),
        enable_profile_memory: Boolean(document.getElementById("ws-profile-memory")?.checked),
        enable_semantic_memory: Boolean(document.getElementById("ws-semantic-memory")?.checked),
        search_prefers_official_sources: Boolean(document.getElementById("ws-prefers-official")?.checked),
      });
      workspaceState.settings = updated;
      window.alert("设置已保存。");
    } catch (error) {
      window.alert(`Save failed: ${error.message}`);
    }
  });
  document.getElementById("ws-clear-session-memory")?.addEventListener("click", async () => {
    if (!appState.activeSessionId) {
      window.alert("No active session.");
      return;
    }
    if (!window.confirm("Clear memory for current session?")) return;
    try {
      await clearWorkspaceMemory("session");
      window.alert("Current session memory cleared.");
    } catch (error) {
      window.alert(`Clear failed: ${error.message}`);
    }
  });
  document.getElementById("ws-clear-all-memory")?.addEventListener("click", async () => {
    if (!window.confirm("Clear all memory for this user?")) return;
    try {
      await clearWorkspaceMemory("all");
      window.alert("All memory cleared.");
    } catch (error) {
      window.alert(`Clear failed: ${error.message}`);
    }
  });
}

function renderKnowledgePanel() {
  workspaceSummary.textContent = "管理本地知识库文件，新增或删除后建议重建索引。";
  const files = workspaceState.knowledgeFiles || [];
  const rows = files.length
    ? files
        .map(
          (file) => `
      <article class="drawer-card workspace-file-card">
        <div class="workspace-file-meta">
          <div class="drawer-card-title">${escapeHtml(file.name)}</div>
          <p class="drawer-card-copy">${escapeHtml(file.path)}</p>
          <div class="drawer-card-meta">${(file.size_bytes / 1024).toFixed(1)} KB | ${file.updated_at || ""}</div>
        </div>
        <button type="button" class="knowledge-delete-button" data-knowledge-delete="${encodeURIComponent(file.path)}">删除</button>
      </article>`,
        )
        .join("")
    : '<p class="drawer-empty">暂无知识库文件。</p>';
  workspaceKnowledgePanel.innerHTML = `
    <section class="drawer-section">
      <h3 class="drawer-section-title">知识库文件</h3>
      <div class="workspace-inline-actions">
        <button id="knowledge-upload-button" type="button">上传文件</button>
        <button id="knowledge-rebuild-button" type="button">重建向量数据库</button>
      </div>
      <div class="workspace-file-list">${rows}</div>
      <input id="knowledge-upload-input" type="file" multiple class="is-hidden" />
    </section>
  `;
  document.getElementById("knowledge-upload-button")?.addEventListener("click", () => {
    document.getElementById("knowledge-upload-input")?.click();
  });
  document.getElementById("knowledge-upload-input")?.addEventListener("change", async (event) => {
    const filesToUpload = Array.from(event.target.files || []);
    if (!filesToUpload.length) return;
    try {
      workspaceState.knowledgeFiles = await uploadKnowledgeFiles(filesToUpload);
      renderKnowledgePanel();
    } catch (error) {
      window.alert(`Upload failed: ${error.message}`);
    }
  });
  document.getElementById("knowledge-rebuild-button")?.addEventListener("click", async () => {
    if (!window.confirm("确认重建本地知识库向量索引吗？这可能需要一些时间。")) return;
    try {
      const result = await rebuildKnowledgeIndex();
      const message = `重建完成：文档 ${result.source_documents || 0}，分块 ${result.chunk_count || 0}，向量 ${result.vector_count || 0}`;
      window.alert(message);
    } catch (error) {
      window.alert(`重建失败: ${error.message}`);
    }
  });
  workspaceKnowledgePanel.querySelectorAll("[data-knowledge-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      const encodedPath = button.getAttribute("data-knowledge-delete") || "";
      const path = decodeURIComponent(encodedPath || "");
      if (!path) return;
      if (!window.confirm(`确认删除知识库文件：${path}？`)) return;
      try {
        workspaceState.knowledgeFiles = await deleteKnowledgeFile(path);
        renderKnowledgePanel();
      } catch (error) {
        window.alert(`删除失败: ${error.message}`);
      }
    });
  });
}

function renderToolStatusPanel() {
  workspaceSummary.textContent = "实时查看工具当前运行状态。";
  const tools = workspaceState.toolStatus || {};
  const rows = Object.entries(tools).map(
    ([name, item]) => `
      <article class="drawer-card">
        <div class="drawer-card-title">${name}</div>
        <p class="drawer-card-copy">状态：${item.state || "unknown"}</p>
        <div class="drawer-card-meta">${item.detail || ""}</div>
        <div class="drawer-card-meta">${item.updated_at || ""}</div>
      </article>`,
  );
  workspaceToolsPanel.innerHTML = `
    <section class="drawer-section">
      <h3 class="drawer-section-title">工具运行状态</h3>
      ${rows.length ? rows.join("") : '<p class="drawer-empty">No tool status yet.</p>'}
    </section>
  `;
}

async function refreshWorkspaceTab(tab) {
  stopToolStatusPolling();
  try {
    if (tab === "settings") {
      workspaceState.settings = await fetchWorkspaceSettings();
      renderWorkspaceSettingsPanel();
      return;
    }
    if (tab === "knowledge") {
      workspaceState.knowledgeFiles = await fetchKnowledgeFiles();
      renderKnowledgePanel();
      return;
    }
    if (tab === "tools") {
      workspaceState.toolStatus = await fetchToolStatus();
      renderToolStatusPanel();
      toolStatusTimer = window.setInterval(async () => {
        try {
          workspaceState.toolStatus = await fetchToolStatus();
          renderToolStatusPanel();
        } catch (_error) {
          stopToolStatusPolling();
        }
      }, 4000);
    }
  } catch (error) {
    const target = tab === "settings" ? workspaceSettingsPanel : tab === "knowledge" ? workspaceKnowledgePanel : workspaceToolsPanel;
    target.innerHTML = `<p class="drawer-empty">加载失败：${error.message}</p>`;
  }
}

function pickStructuredResult(payload) {
  const agentToolResult = payload?.tool_result || payload?.agent?.tool_result;
  if (agentToolResult?.answer) {
    return agentToolResult;
  }

  if (payload?.agent?.decision?.tool_result?.answer) {
    return payload.agent.decision.tool_result;
  }

  return null;
}

function formatRouteIntent(value) {
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
  target.appendChild(item);
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
    item.appendChild(desc);

    if (event.detail) {
      const meta = document.createElement("div");
      meta.className = "route-step-meta";
      meta.textContent = event.detail;
      item.appendChild(meta);
    }

    list.appendChild(item);
  });

  streamSection.append(title, list);
  slots.routeBody.appendChild(streamSection);
  return true;
}

function renderRouteTrace(slots, source) {
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
      item.appendChild(response);

      list.appendChild(item);
    });

    tracesSection.append(title, list);
    slots.routeBody.appendChild(tracesSection);
  }

  if (model?.reason) {
    const reason = document.createElement("p");
    reason.className = "route-reason";
    reason.textContent = model.reason;
    slots.routeBody.appendChild(reason);
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
      chips.appendChild(chip);
    });

    toolsSection.append(title, chips);
    slots.routeBody.appendChild(toolsSection);
  }

  if (model?.intentLabel || model?.dispatchedLabel) {
    const meta = document.createElement("div");
    meta.className = "route-meta-grid";
    appendRouteMetaItem(meta, "意图", model.intentLabel);
    appendRouteMetaItem(meta, "执行节点", model.dispatchedLabel);
    if (meta.childElementCount) {
      slots.routeBody.appendChild(meta);
    }
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

      item.appendChild(desc);
      if (step.toolName || step.status) {
        const meta = document.createElement("div");
        meta.className = "route-step-meta";
        meta.textContent = [step.toolName, step.status].filter(Boolean).join(" · ");
        item.appendChild(meta);
      }
      list.appendChild(item);
    });

    stepsSection.append(title, list);
    slots.routeBody.appendChild(stepsSection);
  }

  if (!slots.routeBody.childElementCount) {
    slots.routeDetails.classList.add("is-hidden");
    return;
  }
  slots.routeDetails.classList.remove("is-hidden");
}

function appendRouteStreamEvent(slots, title, detail = "") {
  slots.routeStreamEvents = Array.isArray(slots.routeStreamEvents)
    ? slots.routeStreamEvents
    : [];
  slots.routeStreamEvents.push({ title, detail });
  renderRouteTrace(slots, null);
  scrollChatToBottom();
}

function updateStreamingAnswer(slots, delta) {
  slots.streamedAnswer = `${slots.streamedAnswer || ""}${String(delta || "")}`;
  renderAnswerContent(
    slots.answerContent,
    slots.streamedAnswer || "正在生成回复...",
    [],
    () => {},
  );
  scrollChatToBottom();
}

function renderStructuredPayload(slots, payload) {
  const structured = pickStructuredResult(payload);
  const agentPayload = payload?.agent || null;
  renderRouteTrace(
    slots,
    agentPayload?.decision
        ? {
          intent: agentPayload.decision.intent || "",
          reason: agentPayload.decision.reason || "",
          dispatched_node: agentPayload?.graph_execution?.dispatched_node || "",
          selected_tools: agentPayload.decision.selected_tools || [],
          execution_steps: agentPayload.decision.execution_steps || [],
          llm_traces: agentPayload.decision.llm_traces || [],
        }
      : null,
  );

  if (structured?.answer) {
    const localAnswer = structured.local_answer || structured.answer;
    const references = filterReferencesByScore(structured.references || []);
    const localChunks = filterLocalChunksByScore(structured.retrieved_chunks || []);
    const webResults = filterWebResultsByScore(
      structured.web_search?.results || structured.results || [],
    );
    const hasLocalReferences = references.some(
      (item) => String(item?.doc_type || "").toLowerCase() !== "web",
    );
    const sourceData = {
      localAnswer,
      webPossibleAnswer:
        structured.web_possible_answer ||
        structured.web_search?.possible_answer ||
        structured.possible_answer ||
        "",
      webResults,
      references,
      chunks: localChunks,
      trace: structured.retrieval_trace || null,
      raw: "",
    };

    slots.sourceData = sourceData;
    renderAnswerContent(slots.answerContent, localAnswer, references, (refIndex) => {
      openSourceDrawer(sourceData, "local", refIndex);
    });

    if (
      references.length ||
      sourceData.chunks.length ||
      webResults.length ||
      sourceData.webPossibleAnswer ||
      sourceData.trace
    ) {
      slots.sourceHint.textContent = references.length
        ? references
            .slice(0, 3)
            .map((reference, index) => formatReferenceLabel(reference, index))
            .join(" · ")
        : webResults.length
          ? `${webResults.length} 条网页结果`
          : "View source details";
      slots.sourceBar.classList.remove("is-hidden");
      slots.sourceButton.onclick = () =>
        openSourceDrawer(sourceData, hasLocalReferences ? "local" : "web");
    }

    const pdfReport =
      structured.pdf_report ||
      (structured.artifacts || []).find((item) => item.kind === "pdf");
    slots.artifactBar.innerHTML = "";
    if (pdfReport?.url) {
      const link = document.createElement("a");
      link.className = "artifact-link";
      link.href = pdfReport.url;
      link.target = "_blank";
      link.rel = "noreferrer noopener";
      link.textContent = `Open PDF report: ${pdfReport.name || "report.pdf"}`;
      slots.artifactBar.appendChild(link);
      slots.artifactBar.classList.remove("is-hidden");
    } else {
      slots.artifactBar.classList.add("is-hidden");
    }

    return;
  }

  const fallbackMessage =
    payload?.agent?.tool_result?.message ||
    "未返回结构化结果。";

  slots.sourceData = {
    localAnswer: fallbackMessage,
    webPossibleAnswer: "",
    webResults: [],
    references: [],
    chunks: [],
    trace: null,
    raw: JSON.stringify(payload, null, 2),
  };

  renderAnswerContent(slots.answerContent, fallbackMessage, [], () => {});
  slots.sourceHint.textContent = "查看原始返回";
  slots.sourceBar.classList.remove("is-hidden");
  slots.sourceButton.onclick = () => openSourceDrawer(slots.sourceData, "local");
  slots.artifactBar.classList.add("is-hidden");
}

function getStoredUserId() {
  return window.localStorage.getItem(STORAGE_KEYS.userId) || "";
}

function getStoredSessionId() {
  return window.localStorage.getItem(STORAGE_KEYS.sessionId) || "";
}

function setActiveUser(userId) {
  appState.userId = userId;
  activeUserLabel.textContent = userId;
  window.localStorage.setItem(STORAGE_KEYS.userId, userId);
}

function setActiveSession(sessionId) {
  appState.activeSessionId = sessionId || "";
  if (appState.activeSessionId) {
    window.localStorage.setItem(STORAGE_KEYS.sessionId, appState.activeSessionId);
  } else {
    window.localStorage.removeItem(STORAGE_KEYS.sessionId);
  }
  syncActiveSessionHeader();
  renderSessionList();
}

function syncActiveSessionHeader() {
  const activeSession = appState.sessions.find(
    (item) => item.session_id === appState.activeSessionId,
  );
  activeSessionLabel.textContent = activeSession?.title || "新对话";
  activeSessionIdLabel.textContent = appState.activeSessionId || "未创建";
}

function renderSessionList() {
  sessionList.innerHTML = "";
  sessionCount.textContent = String(appState.sessions.length);

  if (!appState.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = "还没有历史对话，发送第一条消息后会出现在这里。";
    sessionList.appendChild(empty);
    return;
  }

  appState.sessions.forEach((session) => {
    const row = document.createElement("div");
    row.className = "session-row";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.classList.toggle("is-active", session.session_id === appState.activeSessionId);

    const main = document.createElement("div");
    main.className = "session-item-main";

    const title = document.createElement("span");
    title.className = "session-item-title";
    title.textContent = session.title || session.session_id;

    const preview = document.createElement("span");
    preview.className = "session-item-preview";
    preview.textContent = session.preview || "No preview";

    const meta = document.createElement("span");
    meta.className = "session-item-meta";
    meta.textContent = `${formatTimestamp(session.updated_at)} · ${session.message_count || 0}`;

    main.append(title, preview);
    button.append(main, meta);
    button.addEventListener("click", () => {
      if (session.session_id === appState.activeSessionId) {
        return;
      }
      loadSession(session.session_id);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete";
    deleteButton.textContent = "删除";
    deleteButton.title = "删除该会话";
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      const shouldDelete = window.confirm(`确认删除会话「${session.title || session.session_id}」吗？`);
      if (!shouldDelete) {
        return;
      }
      try {
        await deleteSession(session.session_id);
      } catch (error) {
        window.alert(`删除失败: ${error.message}`);
      }
    });

    row.append(button, deleteButton);
    sessionList.appendChild(row);
  });
}

async function deleteSession(sessionId) {
  if (!appState.userId || !sessionId) {
    return;
  }
  const response = await fetch(
    `/api/users/${encodeURIComponent(appState.userId)}/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
  );
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to delete session");
  }

  const deletedActive = appState.activeSessionId === sessionId;
  appState.sessions = appState.sessions.filter((item) => item.session_id !== sessionId);
  if (deletedActive) {
    startNewConversation();
  }
  await refreshSessions("");
}

function resetChatThread() {
  chatThread.innerHTML = "";
  chatEmpty.classList.remove("is-hidden");
}

function hydrateChatHistory(history) {
  resetChatThread();
  if (!history.length) {
    syncActiveSessionHeader();
    return;
  }

  ensureChatVisible();
  history.forEach((message) => {
    if (message.role === "user") {
      appendUserMessage(message.content || "", [], message.created_at || "");
      return;
    }
    appendAssistantHistoryMessage(
      message.content || "",
      message.created_at || "",
      message.metadata || {},
    );
  });
  syncActiveSessionHeader();
}

async function refreshSessions(preferredSessionId = "") {
  if (!appState.userId) {
    return;
  }
  const response = await fetch(`/api/users/${encodeURIComponent(appState.userId)}/sessions`);
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to load sessions");
  }
  appState.sessions = payload.sessions || [];

  if (preferredSessionId) {
    setActiveSession(preferredSessionId);
    return;
  }

  const storedSessionId = getStoredSessionId();
  const foundStored = appState.sessions.find((item) => item.session_id === storedSessionId);
  const nextSessionId = foundStored?.session_id || appState.activeSessionId;
  if (nextSessionId && appState.sessions.some((item) => item.session_id === nextSessionId)) {
    setActiveSession(nextSessionId);
  } else {
    setActiveSession("");
  }
}

async function loadSession(sessionId) {
  if (!appState.userId || !sessionId) {
    return;
  }
  const response = await fetch(
    `/api/users/${encodeURIComponent(appState.userId)}/sessions/${encodeURIComponent(sessionId)}`,
  );
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to load session");
  }

  const session = payload.session || {};
  const existingIndex = appState.sessions.findIndex((item) => item.session_id === session.session_id);
  if (existingIndex >= 0) {
    appState.sessions[existingIndex] = session;
  }
  setActiveSession(session.session_id || sessionId);
  hydrateChatHistory(payload.history || []);
}

async function loginUser(userId) {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_id: userId }),
  });
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload.detail || "Login failed");
  }

  setActiveUser(payload.user_id);
  appState.sessions = payload.sessions || [];
  try {
    workspaceState.settings = await fetchWorkspaceSettings();
  } catch (_error) {
    workspaceState.settings = workspaceState.settings || {};
  }
  authGate.classList.add("is-hidden");
  workspaceShell.classList.remove("is-hidden");

  const storedSessionId = getStoredSessionId();
  const matchingSession = appState.sessions.find((item) => item.session_id === storedSessionId);
  renderSessionList();
  syncActiveSessionHeader();

  if (matchingSession) {
    await loadSession(matchingSession.session_id);
  } else {
    startNewConversation();
  }
}

function startNewConversation() {
  setActiveSession("");
  resetChatThread();
  syncActiveSessionHeader();
}

function ensureSessionShell(sessionId, previewText = "") {
  if (!sessionId) {
    return;
  }
  const existingIndex = appState.sessions.findIndex(
    (item) => item.session_id === sessionId,
  );
  const now = new Date().toISOString();
  const shell = {
    session_id: sessionId,
    title: previewText.trim() || "新对话",
    preview: previewText.trim() || "处理中...",
    updated_at: now,
    message_count:
      existingIndex >= 0
        ? Number(appState.sessions[existingIndex]?.message_count || 0)
        : 0,
  };
  if (existingIndex >= 0) {
    appState.sessions[existingIndex] = {
      ...appState.sessions[existingIndex],
      ...shell,
    };
  } else {
    appState.sessions.unshift(shell);
  }
  renderSessionList();
  syncActiveSessionHeader();
}

function logoutUser() {
  window.localStorage.removeItem(STORAGE_KEYS.userId);
  window.localStorage.removeItem(STORAGE_KEYS.sessionId);
  appState = {
    userId: "",
    activeSessionId: "",
    sessions: [],
  };
  activeUserLabel.textContent = "anonymous";
  loginInput.value = "";
  sessionList.innerHTML = "";
  resetChatThread();
  closeWorkspaceDrawer();
  workspaceShell.classList.add("is-hidden");
  authGate.classList.remove("is-hidden");
}

async function boot() {
  renderFiles();
  autoResizeTextarea();
  const storedUserId = getStoredUserId();
  if (!storedUserId) {
    authGate.classList.remove("is-hidden");
    workspaceShell.classList.add("is-hidden");
    return;
  }

  loginInput.value = storedUserId;
  try {
    await loginUser(storedUserId);
  } catch (error) {
    console.error(error);
    logoutUser();
  }
}

filesInput.addEventListener("change", renderFiles);
textInput.addEventListener("input", autoResizeTextarea);
textInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }
  event.preventDefault();
  if (activeController) {
    activeController.abort();
    return;
  }
  form.requestSubmit();
});

sourceOverlay.addEventListener("click", closeSourceDrawer);
sourceClose.addEventListener("click", closeSourceDrawer);
sourceTabs.forEach((button) => {
  button.addEventListener("click", () => {
    if (!sourceState.data) {
      return;
    }
    openSourceDrawer(sourceState.data, button.dataset.sourceTab, null);
  });
});
workspaceOverlay?.addEventListener("click", closeWorkspaceDrawer);
workspaceClose?.addEventListener("click", closeWorkspaceDrawer);
workspaceTabs.forEach((button) => {
  button.addEventListener("click", () => {
    const tab = button.dataset.workspaceTab || "settings";
    openWorkspaceDrawer(tab);
  });
});
workspaceSettingsButton?.addEventListener("click", () => openWorkspaceDrawer("settings"));
workspaceKnowledgeButton?.addEventListener("click", () => openWorkspaceDrawer("knowledge"));
workspaceToolsButton?.addEventListener("click", () => openWorkspaceDrawer("tools"));

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sourceDrawer.classList.contains("is-open")) {
    closeSourceDrawer();
  }
  if (event.key === "Escape" && workspaceDrawer.classList.contains("is-open")) {
    closeWorkspaceDrawer();
  }
});

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const userId = loginInput.value.trim();
  if (!userId) {
    loginInput.focus();
    return;
  }

  loginButton.disabled = true;
  loginButton.textContent = "登录中...";
  try {
    await loginUser(userId);
  } catch (error) {
    window.alert(`登录失败: ${error.message}`);
  } finally {
    loginButton.disabled = false;
    loginButton.textContent = "进入工作区";
  }
});

logoutButton.addEventListener("click", logoutUser);
newChatButton.addEventListener("click", startNewConversation);
quickCards.forEach((card) => {
  card.addEventListener("click", () => {
    const quickPrompt = card.dataset.quickPrompt || "";
    textInput.value = quickPrompt;
    autoResizeTextarea();
    textInput.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!appState.userId) {
    window.alert("请先登录。");
    return;
  }

  if (activeController) {
    activeController.abort();
    return;
  }

  const inputText = textInput.value;
  const selectedFiles = Array.from(filesInput.files || []);
  const formData = new FormData();
  formData.append("user_id", appState.userId);
  formData.append("session_id", appState.activeSessionId || "");
  formData.append("text", inputText);
  selectedFiles.forEach((file) => {
    formData.append("files", file);
  });

  ensureChatVisible();
  appendUserMessage(inputText, selectedFiles);
  const slots = appendAssistantPlaceholder();

  textInput.value = "";
  form.reset();
  renderFiles();
  autoResizeTextarea();

  activeController = new AbortController();
  submitButton.textContent = "取消";

  try {
    const response = await fetch("/api/agent/submit", {
      method: "POST",
      body: formData,
      signal: activeController.signal,
    });
    if (!response.ok) {
      const payload = await parseJsonResponse(response);
      throw new Error(payload.detail || "请求失败");
    }

    let finalPayload = null;
    await consumeEventStream(response, ({ type, data }) => {
      if (type === "accepted") {
        if (data.session_id) {
          ensureSessionShell(data.session_id, inputText);
          setActiveSession(data.session_id);
        }
        appendRouteStreamEvent(
          slots,
          "请求已提交",
          `session=${data.session_id || appState.activeSessionId || "new"}`,
        );
        return;
      }
      if (type === "status") {
        appendRouteStreamEvent(slots, data.message || "状态更新", data.stage || "");
        return;
      }
      if (type === "thought") {
        const title = `第 ${Number(data.step || 0)} 轮思考`;
        const detail = [
          data.intent ? `意图=${formatRouteIntent(String(data.intent || ""))}` : "",
          data.plan ? `计划=${String(data.plan || "")}` : "",
          data.action ? `动作=${String(data.action || "")}` : "",
          data.tool_name ? `工具=${String(data.tool_name || "")}` : "",
          data.reason ? `原因=${String(data.reason || "")}` : "",
        ]
          .filter(Boolean)
          .join(" | ");
        appendRouteStreamEvent(slots, title, detail);
        return;
      }
      if (type === "tool_start") {
        appendRouteStreamEvent(
          slots,
          `开始执行工具：${data.label || data.tool_name || "unknown"}`,
        );
        return;
      }
      if (type === "tool_result") {
        appendRouteStreamEvent(
          slots,
          `工具完成：${data.label || data.tool_name || "unknown"}`,
          data.summary || data.status || "",
        );
        return;
      }
      if (type === "answer_start") {
        appendRouteStreamEvent(slots, data.label || "开始生成回答");
        return;
      }
      if (type === "answer_delta") {
        updateStreamingAnswer(slots, data.delta || "");
        return;
      }
      if (type === "error") {
        throw new Error(data.message || "请求失败");
      }
      if (type === "final") {
        finalPayload = data;
        renderStructuredPayload(slots, data);
        activeController = null;
        submitButton.textContent = "发送";
        return false;
      }
    });

    if (!finalPayload) {
      throw new Error("未收到最终结果。");
    }
    if (finalPayload.session_id) {
      await refreshSessions(finalPayload.session_id);
      setActiveSession(finalPayload.session_id);
    }
    if (workspaceDrawer.classList.contains("is-open") && workspaceState.tab === "tools") {
      await refreshWorkspaceTab("tools");
    }
  } catch (error) {
    const message =
      error?.name === "AbortError"
        ? "任务已取消。"
        : `请求失败: ${error.message}`;

    slots.sourceData = {
      localAnswer: message,
      webPossibleAnswer: "",
      webResults: [],
      references: [],
      chunks: [],
      trace: null,
      raw: "",
    };
    renderAnswerContent(slots.answerContent, message, [], () => {});
  } finally {
    activeController = null;
    submitButton.textContent = "发送";
    scrollChatToBottom();
  }
});

boot();
