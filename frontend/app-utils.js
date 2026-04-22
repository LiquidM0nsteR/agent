export async function parseJsonResponse(response) {
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

export async function consumeEventStream(response, onEvent) {
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

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function formatFileSize(file) {
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB >= 1) {
    return `${sizeMB.toFixed(2)} MB`;
  }

  const sizeKB = file.size / 1024;
  return `${sizeKB.toFixed(1)} KB`;
}

export function formatTimestamp(value) {
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

export function formatReferenceLabel(reference, index) {
  const parts = [`[${index + 1}]`, reference.file_name || "未知来源"];
  if (reference.page !== null && reference.page !== undefined) {
    parts.push(`p.${reference.page}`);
  }
  if (reference.section) {
    parts.push(reference.section);
  }
  return parts.join(" ");
}
