import { escapeHtml } from "./app-utils.js";

export function createWorkspaceDrawerController({
  refs,
  defaults,
  actions,
  getActiveSessionId,
}) {
  // 工作台抽屉独立维护设置、知识库和工具状态，减少主脚本的面向状态编程。
  const state = {
    tab: "settings",
    settings: null,
    knowledgeFiles: [],
    toolStatus: null,
  };
  let toolStatusTimer = null;

  function stopToolStatusPolling() {
    if (toolStatusTimer) {
      window.clearInterval(toolStatusTimer);
      toolStatusTimer = null;
    }
  }

  function renderWorkspaceSettingsPanel() {
    const settings = state.settings || {};
    refs.workspaceSummary.textContent = "常用参数、记忆管理与搜索偏好。";
    refs.workspaceSettingsPanel.innerHTML = `
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
            <input id="ws-local-min-score" type="number" step="0.05" min="0" max="10" value="${Number(settings.local_source_min_score ?? defaults.localSourceMinScore)}" />
          </label>
          <label class="workspace-field">网页来源最低置信度
            <input id="ws-web-min-score" type="number" step="0.1" min="0" max="10" value="${Number(settings.web_source_min_score ?? defaults.webSourceMinScore)}" />
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
        const updated = await actions.saveWorkspaceSettings({
          temperature: Number(document.getElementById("ws-temperature")?.value || 0.2),
          max_new_tokens: Number(document.getElementById("ws-max-new-tokens")?.value || 512),
          short_term_max_messages: Number(document.getElementById("ws-short-msg")?.value || 12),
          short_term_summary_threshold: Number(document.getElementById("ws-summary-threshold")?.value || 8),
          long_term_top_k: Number(document.getElementById("ws-long-topk")?.value || 3),
          local_source_min_score: Number(document.getElementById("ws-local-min-score")?.value || defaults.localSourceMinScore),
          web_source_min_score: Number(document.getElementById("ws-web-min-score")?.value || defaults.webSourceMinScore),
          enable_profile_memory: Boolean(document.getElementById("ws-profile-memory")?.checked),
          enable_semantic_memory: Boolean(document.getElementById("ws-semantic-memory")?.checked),
          search_prefers_official_sources: Boolean(document.getElementById("ws-prefers-official")?.checked),
        });
        state.settings = updated;
        window.alert("设置已保存。");
      } catch (error) {
        window.alert(`Save failed: ${error.message}`);
      }
    });

    document.getElementById("ws-clear-session-memory")?.addEventListener("click", async () => {
      if (!getActiveSessionId()) {
        window.alert("No active session.");
        return;
      }
      if (!window.confirm("Clear memory for current session?")) return;
      try {
        await actions.clearWorkspaceMemory("session");
        window.alert("Current session memory cleared.");
      } catch (error) {
        window.alert(`Clear failed: ${error.message}`);
      }
    });

    document.getElementById("ws-clear-all-memory")?.addEventListener("click", async () => {
      if (!window.confirm("Clear all memory for this user?")) return;
      try {
        await actions.clearWorkspaceMemory("all");
        window.alert("All memory cleared.");
      } catch (error) {
        window.alert(`Clear failed: ${error.message}`);
      }
    });
  }

  function renderKnowledgePanel() {
    refs.workspaceSummary.textContent = "管理本地知识库文件，新增或删除后建议重建索引。";
    const files = state.knowledgeFiles || [];
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

    refs.workspaceKnowledgePanel.innerHTML = `
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
        state.knowledgeFiles = await actions.uploadKnowledgeFiles(filesToUpload);
        renderKnowledgePanel();
      } catch (error) {
        window.alert(`Upload failed: ${error.message}`);
      }
    });

    document.getElementById("knowledge-rebuild-button")?.addEventListener("click", async () => {
      if (!window.confirm("确认重建本地知识库向量索引吗？这可能需要一些时间。")) return;
      try {
        const result = await actions.rebuildKnowledgeIndex();
        const message = `重建完成：文档 ${result.source_documents || 0}，分块 ${result.chunk_count || 0}，向量 ${result.vector_count || 0}`;
        window.alert(message);
      } catch (error) {
        window.alert(`重建失败: ${error.message}`);
      }
    });

    refs.workspaceKnowledgePanel.querySelectorAll("[data-knowledge-delete]").forEach((button) => {
      button.addEventListener("click", async () => {
        const encodedPath = button.getAttribute("data-knowledge-delete") || "";
        const path = decodeURIComponent(encodedPath || "");
        if (!path) return;
        if (!window.confirm(`确认删除知识库文件：${path}？`)) return;
        try {
          state.knowledgeFiles = await actions.deleteKnowledgeFile(path);
          renderKnowledgePanel();
        } catch (error) {
          window.alert(`删除失败: ${error.message}`);
        }
      });
    });
  }

  function renderToolStatusPanel() {
    refs.workspaceSummary.textContent = "实时查看工具当前运行状态。";
    const tools = state.toolStatus || {};
    const rows = Object.entries(tools).map(
      ([name, item]) => `
        <article class="drawer-card">
          <div class="drawer-card-title">${name}</div>
          <p class="drawer-card-copy">状态：${item.state || "unknown"}</p>
          <div class="drawer-card-meta">${item.detail || ""}</div>
          <div class="drawer-card-meta">${item.updated_at || ""}</div>
        </article>`,
    );

    refs.workspaceToolsPanel.innerHTML = `
      <section class="drawer-section">
        <h3 class="drawer-section-title">工具运行状态</h3>
        ${rows.length ? rows.join("") : '<p class="drawer-empty">No tool status yet.</p>'}
      </section>
    `;
  }

  async function refreshTab(tab = state.tab) {
    stopToolStatusPolling();
    state.tab = tab;
    try {
      if (tab === "settings") {
        state.settings = await actions.fetchWorkspaceSettings();
        renderWorkspaceSettingsPanel();
        return;
      }
      if (tab === "knowledge") {
        state.knowledgeFiles = await actions.fetchKnowledgeFiles();
        renderKnowledgePanel();
        return;
      }
      if (tab === "tools") {
        state.toolStatus = await actions.fetchToolStatus();
        renderToolStatusPanel();
        toolStatusTimer = window.setInterval(async () => {
          try {
            state.toolStatus = await actions.fetchToolStatus();
            renderToolStatusPanel();
          } catch (_error) {
            stopToolStatusPolling();
          }
        }, 4000);
      }
    } catch (error) {
      const target = tab === "settings"
        ? refs.workspaceSettingsPanel
        : tab === "knowledge"
          ? refs.workspaceKnowledgePanel
          : refs.workspaceToolsPanel;
      target.innerHTML = `<p class="drawer-empty">加载失败：${error.message}</p>`;
    }
  }

  function open(tab = "settings") {
    state.tab = tab;
    refs.workspaceSettingsButton?.classList.toggle("is-active", tab === "settings");
    refs.workspaceKnowledgeButton?.classList.toggle("is-active", tab === "knowledge");
    refs.workspaceToolsButton?.classList.toggle("is-active", tab === "tools");
    refs.workspaceTabs.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.workspaceTab === tab);
    });
    refs.workspaceSettingsPanel.classList.toggle("is-hidden", tab !== "settings");
    refs.workspaceKnowledgePanel.classList.toggle("is-hidden", tab !== "knowledge");
    refs.workspaceToolsPanel.classList.toggle("is-hidden", tab !== "tools");
    refs.workspaceOverlay.classList.remove("is-hidden");
    refs.workspaceDrawer.classList.add("is-open");
    void refreshTab(tab);
  }

  function close() {
    refs.workspaceDrawer.classList.remove("is-open");
    refs.workspaceOverlay.classList.add("is-hidden");
    refs.workspaceSettingsButton?.classList.remove("is-active");
    refs.workspaceKnowledgeButton?.classList.remove("is-active");
    refs.workspaceToolsButton?.classList.remove("is-active");
    stopToolStatusPolling();
  }

  refs.workspaceOverlay?.addEventListener("click", close);
  refs.workspaceClose?.addEventListener("click", close);
  refs.workspaceTabs.forEach((button) => {
    button.addEventListener("click", () => {
      open(button.dataset.workspaceTab || "settings");
    });
  });
  refs.workspaceSettingsButton?.addEventListener("click", () => open("settings"));
  refs.workspaceKnowledgeButton?.addEventListener("click", () => open("knowledge"));
  refs.workspaceToolsButton?.addEventListener("click", () => open("tools"));

  return {
    open,
    close,
    async loadSettings() {
      state.settings = await actions.fetchWorkspaceSettings();
      return state.settings;
    },
    setSettings(settings) {
      state.settings = settings || {};
    },
    getSettings() {
      return state.settings || {};
    },
    isOpen() {
      return refs.workspaceDrawer.classList.contains("is-open");
    },
    getCurrentTab() {
      return state.tab;
    },
    refreshCurrentTab() {
      return refreshTab(state.tab);
    },
  };
}
