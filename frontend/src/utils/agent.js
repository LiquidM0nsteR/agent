export async function parseJsonResponse(response) {
  const rawText = await response.text()
  try {
    return JSON.parse(rawText)
  } catch {
    return {
      detail: rawText || `HTTP ${response.status}`,
      _raw: rawText,
    }
  }
}

function parseSseEventBlock(block) {
  const lines = String(block || '').split(/\r?\n/)
  let eventType = 'message'
  const dataLines = []

  lines.forEach((line) => {
    if (!line || line.startsWith(':')) {
      return
    }
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim() || 'message'
      return
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  })

  if (!dataLines.length) {
    return null
  }

  const rawData = dataLines.join('\n')
  try {
    return {
      type: eventType,
      data: JSON.parse(rawData),
    }
  } catch {
    return {
      type: eventType,
      data: { raw: rawData },
    }
  }
}

export async function consumeEventStream(response, onEvent) {
  if (!response.body) {
    throw new Error('浏览器不支持流式响应。')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })

    while (true) {
      const boundary = buffer.indexOf('\n\n')
      if (boundary < 0) {
        break
      }
      const rawBlock = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const parsed = parseSseEventBlock(rawBlock)
      if (!parsed) {
        continue
      }
      const shouldContinue = onEvent(parsed)
      if (shouldContinue === false) {
        await reader.cancel()
        return
      }
    }
  }

  buffer += decoder.decode()
  const tail = parseSseEventBlock(buffer.trim())
  if (tail) {
    onEvent(tail)
  }
}

export function formatFileSize(value) {
  const sizeBytes = typeof value === 'number' ? value : Number(value?.size || 0)
  const sizeMB = sizeBytes / (1024 * 1024)
  if (sizeMB >= 1) {
    return `${sizeMB.toFixed(2)} MB`
  }
  return `${(sizeBytes / 1024).toFixed(1)} KB`
}

export function formatTimestamp(value) {
  if (!value) {
    return ''
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export function formatReferenceLabel(reference, index) {
  const parts = [`[${index + 1}]`, reference.file_name || '未知来源']
  if (reference.page !== null && reference.page !== undefined) {
    parts.push(`p.${reference.page}`)
  }
  if (reference.section) {
    parts.push(reference.section)
  }
  return parts.join(' ')
}

export function formatRouteIntent(value) {
  const labels = {
    general_chat: '通用对话',
    local_knowledge_qa: '本地知识库问答',
    web_search: '网页搜索',
    single_cell_analysis: '单细胞分析',
    augmented_analysis: '增强分析',
  }
  return labels[value] || value || ''
}

export function formatToolLabel(value) {
  const labels = {
    direct_llm: '基础对话模型',
    local_knowledge_base: '本地知识库问答',
    web_search: '网页搜索',
    single_cell_pipeline: '单细胞分析流程',
  }
  return labels[value] || value || ''
}

export function buildRouteTraceModel(source) {
  if (!source || typeof source !== 'object') {
    return null
  }
  const reason = String(source.reason || '').trim()
  const selectedTools = Array.isArray(source.selected_tools)
    ? source.selected_tools.map((item) => formatToolLabel(String(item || ''))).filter(Boolean)
    : []
  const executionSteps = Array.isArray(source.execution_steps)
    ? source.execution_steps
        .map((item) => ({
          description: String(item?.description || '').trim(),
          detail: String(item?.detail || '').trim(),
          status: String(item?.status || '').trim(),
          toolName: formatToolLabel(String(item?.tool_name || '')),
          elapsedMs: Number(item?.elapsed_ms || 0),
        }))
        .filter((item) => item.description)
    : []
  const llmTraces = Array.isArray(source.llm_traces)
    ? source.llm_traces
        .map((item) => ({
          label: String(item?.label || ''),
          response: String(item?.response || item?.response_preview || ''),
          elapsedMs: item?.elapsed_ms,
        }))
        .filter((item) => item.response)
    : []

  if (
    !reason
    && !selectedTools.length
    && !executionSteps.length
    && !llmTraces.length
    && !source.intent
    && !source.dispatched_node
  ) {
    return null
  }

  return {
    reason,
    selectedTools,
    executionSteps,
    llmTraces,
    intentLabel: formatRouteIntent(String(source.intent || '')),
    dispatchedLabel: formatRouteIntent(String(source.dispatched_node || '')),
  }
}
