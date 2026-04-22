import {
  consumeEventStream,
  formatFileSize,
  formatReferenceLabel,
  formatTimestamp,
  parseJsonResponse,
} from "./app-utils.js";
import { formatRouteIntent, renderRouteTrace } from "./route-trace.js";
import { createSourceDrawerController } from "./source-drawer.js";
import { createWorkspaceDrawerController } from "./workspace-drawer.js";

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
let appState = {
  userId: "",
  activeSessionId: "",
  sessions: [],
};

const sourceDrawerController = createSourceDrawerController({
  sourceDrawer,
  sourceOverlay,
  sourceClose,
  sourceTabs,
  sourceSummary,
  sourceLocalPanel,
  sourceWebPanel,
});

const workspaceDrawerController = createWorkspaceDrawerController({
  refs: {
    workspaceOverlay,
    workspaceDrawer,
    workspaceClose,
    workspaceTabs,
    workspaceSummary,
    workspaceSettingsPanel,
    workspaceKnowledgePanel,
    workspaceToolsPanel,
    workspaceSettingsButton,
    workspaceKnowledgeButton,
    workspaceToolsButton,
  },
  defaults: {
    localSourceMinScore: DEFAULT_LOCAL_SOURCE_MIN_SCORE,
    webSourceMinScore: DEFAULT_WEB_SOURCE_MIN_SCORE,
  },
  actions: {
    fetchWorkspaceSettings,
    saveWorkspaceSettings,
    clearWorkspaceMemory,
    fetchKnowledgeFiles,
    uploadKnowledgeFiles,
    deleteKnowledgeFile,
    rebuildKnowledgeIndex,
    fetchToolStatus,
  },
  getActiveSessionId: () => appState.activeSessionId,
});

function getLocalSourceMinScore() {
  const value = Number(workspaceDrawerController.getSettings()?.local_source_min_score);
  if (Number.isFinite(value)) {
    return value;
  }
  return DEFAULT_LOCAL_SOURCE_MIN_SCORE;
}

function getWebSourceMinScore() {
  const value = Number(workspaceDrawerController.getSettings()?.web_source_min_score);
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
      sourceDrawerController.open(sourceData, "local", refIndex);
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
        sourceDrawerController.open(
          sourceData,
          hasLocalReferences ? "local" : "web",
        );
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
  slots.sourceButton.onclick = () =>
    sourceDrawerController.open(slots.sourceData, "local");
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
    await workspaceDrawerController.loadSettings();
  } catch (_error) {
    workspaceDrawerController.setSettings({});
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
  workspaceDrawerController.setSettings({});
  activeUserLabel.textContent = "anonymous";
  loginInput.value = "";
  sessionList.innerHTML = "";
  resetChatThread();
  workspaceDrawerController.close();
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

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sourceDrawerController.isOpen()) {
    sourceDrawerController.close();
  }
  if (event.key === "Escape" && workspaceDrawerController.isOpen()) {
    workspaceDrawerController.close();
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
    if (
      workspaceDrawerController.isOpen()
      && workspaceDrawerController.getCurrentTab() === "tools"
    ) {
      await workspaceDrawerController.refreshCurrentTab();
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
