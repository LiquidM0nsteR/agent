import { formatReferenceLabel } from "./app-utils.js";

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
    titleRow.append(icon);
  }

  const title = document.createElement("div");
  title.className = "drawer-card-title";
  title.textContent = `[${result.source_tier || "community"}] ${result.title || "Untitled"}`;
  titleRow.append(title);

  const body = document.createElement("p");
  body.className = "drawer-card-copy";
  body.textContent = result.snippet || "";

  const meta = document.createElement("div");
  meta.className = "drawer-card-meta";
  meta.textContent = `score=${Number(result.score || 0).toFixed(2)} | tier=${result.source_tier || "web"}`;

  item.append(titleRow, body, meta);

  if (result.url) {
    const link = document.createElement("a");
    link.className = "drawer-link";
    link.href = result.url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = result.url;
    item.append(link);
  }

  return item;
}

export function createSourceDrawerController(refs) {
  // 集中管理来源抽屉的展示与切换，避免 app.js 同时维护 DOM 和状态。
  const state = {
    tab: "local",
    focusReferenceIndex: null,
    data: null,
  };

  function open(data, tab, focusReferenceIndex = null) {
    state.tab = tab;
    state.focusReferenceIndex = focusReferenceIndex;
    state.data = data;

    refs.sourceSummary.textContent = data.localAnswer || "Selected answer sources";
    refs.sourceTabs.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.sourceTab === tab);
    });

    refs.sourceLocalPanel.innerHTML = "";
    refs.sourceWebPanel.innerHTML = "";

    if (data.references.length || data.chunks.length || data.trace || data.raw) {
      const localSections = [];

      if (data.references.length) {
        const referencesBlock = document.createElement("section");
        referencesBlock.className = "drawer-section";

        const title = document.createElement("h3");
        title.className = "drawer-section-title";
        title.textContent = "参考来源";
        referencesBlock.append(title);

        data.references.forEach((reference, index) => {
          referencesBlock.append(buildReferenceItem(reference, index));
        });

        localSections.push(referencesBlock);
      }

      if (data.chunks.length) {
        const chunksBlock = document.createElement("section");
        chunksBlock.className = "drawer-section";

        const title = document.createElement("h3");
        title.className = "drawer-section-title";
        title.textContent = "检索片段";
        chunksBlock.append(title);

        data.chunks.forEach((chunk, index) => {
          chunksBlock.append(buildChunkItem(chunk, index));
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

      localSections.forEach((section) => refs.sourceLocalPanel.append(section));
    } else {
      refs.sourceLocalPanel.innerHTML = '<p class="drawer-empty">暂无本地来源详情。</p>';
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
        refs.sourceWebPanel.append(answerBlock);
      }

      if (data.webResults.length) {
        const resultsBlock = document.createElement("section");
        resultsBlock.className = "drawer-section";

        const title = document.createElement("h3");
        title.className = "drawer-section-title";
        title.textContent = "网页检索结果";
        resultsBlock.append(title);

        data.webResults.forEach((result) => {
          resultsBlock.append(buildWebItem(result));
        });

        refs.sourceWebPanel.append(resultsBlock);
      }
    } else {
      refs.sourceWebPanel.innerHTML = '<p class="drawer-empty">暂无网页来源详情。</p>';
    }

    refs.sourceLocalPanel.classList.toggle("is-hidden", tab !== "local");
    refs.sourceWebPanel.classList.toggle("is-hidden", tab !== "web");
    refs.sourceOverlay.classList.remove("is-hidden");
    refs.sourceDrawer.classList.add("is-open");

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

  function close() {
    refs.sourceDrawer.classList.remove("is-open");
    refs.sourceOverlay.classList.add("is-hidden");
  }

  refs.sourceOverlay?.addEventListener("click", close);
  refs.sourceClose?.addEventListener("click", close);
  refs.sourceTabs.forEach((button) => {
    button.addEventListener("click", () => {
      if (!state.data) {
        return;
      }
      open(state.data, button.dataset.sourceTab || "local", null);
    });
  });

  return {
    open,
    close,
    isOpen() {
      return refs.sourceDrawer.classList.contains("is-open");
    },
  };
}
