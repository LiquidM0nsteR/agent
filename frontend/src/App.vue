<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import {
  buildRouteTraceModel,
  consumeEventStream,
  formatFileSize,
  formatReferenceLabel,
  formatRouteIntent,
  formatTimestamp,
  parseJsonResponse,
} from './utils/agent'

const STORAGE_KEYS = {
  userId: 'agent_user_id',
  sessionId: 'agent_active_session_id',
}

const DEFAULT_LOCAL_SOURCE_MIN_SCORE = 0.35
const DEFAULT_WEB_SOURCE_MIN_SCORE = 1.5
const FILE_KIND_LABELS = {
  image: '图片',
  pdf: 'PDF',
  h5ad: 'h5ad',
  file: '文件',
}
const QUICK_PROMPTS = [
  '帮我解释一个生物信息学中的概念，并给出一个简单例子。',
  '帮我检索近期与单细胞分析相关的论文，并总结关键观点。',
  '我会上传一个 h5ad 文件，请先做单细胞分析并给出摘要。',
  '请给我一个后续实验设计建议，包含可执行步骤。',
]

const fileInputRef = ref(null)
const knowledgeInputRef = ref(null)
const chatBodyRef = ref(null)
const selectedFiles = ref([])
const activeController = ref(null)

const loginForm = reactive({
  userId: '',
  loading: false,
})

const state = reactive({
  userId: '',
  activeSessionId: '',
  sessions: [],
  messages: [],
  composerText: '',
  pending: false,
})

const workspace = reactive({
  visible: false,
  tab: 'settings',
  loading: false,
  settings: createDefaultWorkspaceSettings(),
  knowledgeFiles: [],
  toolStatus: {},
  toolPoller: null,
})

const sourceDrawer = reactive({
  visible: false,
  tab: 'local',
  data: createEmptySourceData(),
})

const activeSession = computed(() => {
  return state.sessions.find((item) => item.session_id === state.activeSessionId) || null
})

const sourceSummary = computed(() => {
  return sourceDrawer.data.localAnswer || '选择一条回答查看支撑材料。'
})

function createDefaultWorkspaceSettings() {
  return {
    temperature: 0.2,
    max_new_tokens: 512,
    short_term_max_messages: 12,
    short_term_summary_threshold: 8,
    long_term_top_k: 3,
    local_source_min_score: DEFAULT_LOCAL_SOURCE_MIN_SCORE,
    web_source_min_score: DEFAULT_WEB_SOURCE_MIN_SCORE,
    enable_profile_memory: true,
    enable_semantic_memory: true,
    search_provider: 'serper',
    search_prefers_official_sources: true,
  }
}

function createEmptySourceData() {
  return {
    localAnswer: '',
    webPossibleAnswer: '',
    webResults: [],
    webThreshold: null,
    webConfiguredThreshold: null,
    webThresholdRelaxed: false,
    webTopResultScore: null,
    webRawResultsCount: 0,
    webRetainedResultsCount: 0,
    references: [],
    chunks: [],
    trace: null,
    raw: '',
  }
}

function createMessageId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`
}

function detectFileKind(file) {
  const name = String(file?.name || '').toLowerCase()
  const contentType = String(file?.contentType || file?.type || '').toLowerCase()
  if (name.endsWith('.h5ad')) {
    return 'h5ad'
  }
  if (name.endsWith('.pdf') || contentType === 'application/pdf') {
    return 'pdf'
  }
  if (contentType.startsWith('image/')) {
    return 'image'
  }
  return 'file'
}

function buildDisplayAttachments(files) {
  return (files || []).map((file) => ({
    name: file.name || 'unknown',
    kind: detectFileKind(file),
    sizeBytes: Number(file.sizeBytes ?? file.size ?? 0),
    contentType: String(file.contentType || file.type || ''),
  }))
}

function formatAttachmentLabel(file) {
  const kind = file.kind || detectFileKind(file)
  return `${FILE_KIND_LABELS[kind] || '文件'} | ${file.name} (${formatFileSize(file.sizeBytes ?? file.size ?? 0)})`
}

function splitAnswerParagraphs(text) {
  const normalized = String(text || '').trim()
  if (!normalized) {
    return ['未返回结构化结果。']
  }
  const parts = normalized
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean)
  return parts.length ? parts : [normalized]
}

const URL_PATTERN_SOURCE = 'https?:\\/\\/[^\\s<>"\']+'
const TRAILING_URL_PUNCTUATION = /[),.，。；;:：!?！？\]}]+$/

function splitUrlToken(rawUrl) {
  const value = String(rawUrl || '')
  const trailing = value.match(TRAILING_URL_PUNCTUATION)?.[0] || ''
  return {
    url: trailing ? value.slice(0, -trailing.length) : value,
    trailing,
  }
}

function linkifyText(text) {
  const value = String(text || '')
  const segments = []
  let lastIndex = 0
  const urlPattern = new RegExp(URL_PATTERN_SOURCE, 'gi')

  value.replace(urlPattern, (match, offset) => {
    if (offset > lastIndex) {
      segments.push({ type: 'text', text: value.slice(lastIndex, offset) })
    }

    const { url, trailing } = splitUrlToken(match)
    if (url) {
      segments.push({ type: 'link', text: url, href: url })
    }
    if (trailing) {
      segments.push({ type: 'text', text: trailing })
    }

    lastIndex = offset + match.length
    return match
  })

  if (lastIndex < value.length) {
    segments.push({ type: 'text', text: value.slice(lastIndex) })
  }

  return segments.length ? segments : [{ type: 'text', text: value }]
}

function splitLinkedReferenceLines(text) {
  const normalized = String(text || '').trim()
  if (!normalized) {
    return [{ text: '未返回结构化结果。', segments: linkifyText('未返回结构化结果。') }]
  }

  return normalized
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => ({
      text: line,
      segments: linkifyText(line),
    }))
}

function formatAnswerBlocks(text) {
  return splitAnswerParagraphs(text).map((paragraph) => {
    const hasLink = new RegExp(URL_PATTERN_SOURCE, 'i').test(paragraph)
    if (!hasLink) {
      return {
        type: 'paragraph',
        text: paragraph,
        lines: [],
      }
    }
    return {
      type: 'references',
      text: paragraph,
      lines: splitLinkedReferenceLines(paragraph),
    }
  })
}

function prettyJson(value) {
  if (typeof value === 'string') {
    return value
  }
  return JSON.stringify(value, null, 2)
}

function getStoredUserId() {
  return window.localStorage.getItem(STORAGE_KEYS.userId) || ''
}

function getStoredSessionId() {
  return window.localStorage.getItem(STORAGE_KEYS.sessionId) || ''
}

function setActiveUser(userId) {
  state.userId = userId
  loginForm.userId = userId
  window.localStorage.setItem(STORAGE_KEYS.userId, userId)
}

function setActiveSession(sessionId) {
  state.activeSessionId = sessionId || ''
  if (state.activeSessionId) {
    window.localStorage.setItem(STORAGE_KEYS.sessionId, state.activeSessionId)
  } else {
    window.localStorage.removeItem(STORAGE_KEYS.sessionId)
  }
}

function createUserMessage(content, attachments = [], createdAt = '') {
  return {
    id: createMessageId('user'),
    role: 'user',
    content: content || '已上传附件',
    createdAt,
    attachments,
    routeTrace: null,
    routeStreamEvents: [],
    sourceData: createEmptySourceData(),
    artifact: null,
    loading: false,
    streamedAnswer: '',
    routeLlmOutput: '',
  }
}

function createAssistantMessage(content = '正在生成回复...') {
  return {
    id: createMessageId('assistant'),
    role: 'assistant',
    content,
    createdAt: '',
    attachments: [],
    routeTrace: null,
    routeStreamEvents: [],
    sourceData: createEmptySourceData(),
    artifact: null,
    loading: true,
    streamedAnswer: '',
    routeLlmOutput: '',
  }
}

function resetComposer() {
  state.composerText = ''
  selectedFiles.value = []
  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

function scrollChatToBottom() {
  nextTick(() => {
    if (chatBodyRef.value) {
      chatBodyRef.value.scrollTop = chatBodyRef.value.scrollHeight
    }
  })
}

function routeModel(message) {
  return buildRouteTraceModel(message?.routeTrace)
}

function hasRouteTrace(message) {
  return Boolean(message?.routeStreamEvents?.length || routeModel(message))
}

function messageHasLocalSources(message) {
  const data = message?.sourceData || {}
  return Boolean(data.references?.length || data.chunks?.length || data.trace || data.raw)
}

function messageHasWebSources(message) {
  const data = message?.sourceData || {}
  return Boolean(data.webResults?.length || data.webPossibleAnswer)
}

function localSourceButtonLabel(message) {
  const data = message?.sourceData || {}
  const count = Number(data.references?.length || 0) + Number(data.chunks?.length || 0)
  return count ? `本地来源 ${count}` : '本地来源'
}

function webSourceButtonLabel(message) {
  const count = Number(message?.sourceData?.webResults?.length || 0)
  return count ? `网页来源 ${count}` : '网页来源'
}

function currentLocalThreshold() {
  const value = Number(workspace.settings.local_source_min_score)
  return Number.isFinite(value) ? value : DEFAULT_LOCAL_SOURCE_MIN_SCORE
}

function currentWebThreshold() {
  const value = Number(workspace.settings.web_source_min_score)
  return Number.isFinite(value) ? value : DEFAULT_WEB_SOURCE_MIN_SCORE
}

function filterWebResultsByScore(results, thresholdOverride = null) {
  const threshold = Number.isFinite(Number(thresholdOverride))
    ? Number(thresholdOverride)
    : currentWebThreshold()
  return (results || []).filter((item) => Number(item?.score || 0) >= threshold)
}

function filterReferencesByScore(references) {
  const localThreshold = currentLocalThreshold()
  const webThreshold = currentWebThreshold()
  return (references || []).filter((item) => {
    const docType = String(item?.doc_type || '').toLowerCase()
    if (docType === 'web') {
      return Number(item?.score || 0) >= webThreshold
    }
    if (item?.score == null) {
      return true
    }
    return Number(item?.score || 0) >= localThreshold
  })
}

function filterLocalChunksByScore(chunks) {
  const threshold = currentLocalThreshold()
  return (chunks || []).filter((item) => {
    if (item?.score == null) {
      return true
    }
    return Number(item?.score || 0) >= threshold
  })
}

function pickStructuredResult(payload) {
  const agentToolResult = payload?.tool_result || payload?.agent?.tool_result
  if (agentToolResult?.answer) {
    return agentToolResult
  }
  if (payload?.agent?.decision?.tool_result?.answer) {
    return payload.agent.decision.tool_result
  }
  return null
}

function appendRouteStreamEvent(message, title, detail = '') {
  message.routeStreamEvents = Array.isArray(message.routeStreamEvents) ? message.routeStreamEvents : []
  message.routeStreamEvents.push({ title, detail })
  scrollChatToBottom()
}

function upsertRouteStreamEvent(message, key, title, detail = '') {
  message.routeStreamEvents = Array.isArray(message.routeStreamEvents) ? message.routeStreamEvents : []
  const existing = message.routeStreamEvents.find((item) => item.key === key)
  if (existing) {
    existing.title = title
    existing.detail = detail
  } else {
    message.routeStreamEvents.push({ key, title, detail })
  }
  scrollChatToBottom()
}

function updateRouteLlmOutput(message, data) {
  const delta = String(data.content_delta || '')
  const current = String(message.routeLlmOutput || '')
  message.routeLlmOutput = data.content != null ? String(data.content || '') : `${current}${delta}`
  upsertRouteStreamEvent(message, 'router-llm-output', '路由分析中', '正在生成路由决策...')
}

function appendRouteDecision(message, data) {
  const node = String(data.next_node || data.action || data.decision || '')
  const thought = String(data.thought || data.reason || '').trim()
  const query = String(data.action_input?.query || '').trim()
  const detail = [
    thought ? `决策=${thought}` : '',
    node ? `选择节点=${formatRouteIntent(node) || node}` : '',
    data.intent ? `意图=${formatRouteIntent(String(data.intent || ''))}` : '',
    query ? `检索问题=${query}` : '',
  ]
    .filter(Boolean)
    .join(' | ')
  upsertRouteStreamEvent(message, 'router-decision', '路由决策', detail)
}

function appendThoughtEvent(message, data) {
  const kind = String(data.kind || '')
  if (kind === 'router_start') {
    upsertRouteStreamEvent(
      message,
      'router-start',
      '开始路由',
      String(data.plan || data.action || '准备选择下一步节点。'),
    )
    return
  }

  const step = Number(data.step)
  const node = String(data.next_node || data.action || data.decision || '')
  const thought = String(data.thought || data.reason || '').trim()
  const detail = [
    thought ? `决策=${thought}` : '',
    node ? `选择节点=${formatRouteIntent(node) || node}` : '',
    data.intent ? `意图=${formatRouteIntent(String(data.intent || ''))}` : '',
    data.plan ? `计划=${String(data.plan || '')}` : '',
    data.tool_name ? `工具=${String(data.tool_name || '')}` : '',
  ]
    .filter(Boolean)
    .join(' | ')

  if (!detail) {
    return
  }

  const key = Number.isFinite(step) && step > 0 ? `thought-${step}` : `thought-${kind || data.node || 'latest'}`
  const title = Number.isFinite(step) && step > 0 ? `第 ${step} 轮思考` : '思考'
  upsertRouteStreamEvent(message, key, title, detail)
}

function updateStreamingAnswer(message, delta) {
  message.streamedAnswer = `${message.streamedAnswer || ''}${String(delta || '')}`
  message.content = message.streamedAnswer || '正在生成回复...'
  scrollChatToBottom()
}

function renderStructuredPayload(message, payload) {
  const structured = pickStructuredResult(payload)
  const agentPayload = payload?.agent || null

  if (agentPayload?.decision) {
    message.routeTrace = {
      intent: agentPayload.decision.intent || '',
      reason: agentPayload.decision.reason || '',
      dispatched_node: agentPayload?.graph_execution?.dispatched_node || '',
      selected_tools: agentPayload.decision.selected_tools || [],
      execution_steps: agentPayload.decision.execution_steps || [],
      llm_traces: agentPayload.decision.llm_traces || [],
    }
  }

  if (structured?.answer) {
    const localAnswer = structured.local_answer || structured.answer
    const references = filterReferencesByScore(structured.references || [])
    const localChunks = filterLocalChunksByScore(structured.retrieved_chunks || [])
    const effectiveWebThreshold = Number(
      structured.web_search?.web_source_effective_min_score
      ?? structured.web_source_effective_min_score
      ?? structured.web_search?.web_source_min_score
      ?? structured.web_source_min_score
    )
    const configuredWebThreshold = Number(
      structured.web_search?.web_source_configured_min_score
      ?? structured.web_source_configured_min_score
      ?? currentWebThreshold()
    )
    const webResults = filterWebResultsByScore(
      structured.web_search?.results || structured.results || [],
      effectiveWebThreshold,
    )
    const pdfReport = structured.pdf_report || (structured.artifacts || []).find((item) => item.kind === 'pdf')

    message.content = localAnswer
    message.sourceData = {
      localAnswer,
      webPossibleAnswer:
        structured.web_possible_answer
        || structured.web_search?.possible_answer
        || structured.possible_answer
        || '',
      webResults,
      webThreshold: Number.isFinite(effectiveWebThreshold) ? effectiveWebThreshold : null,
      webConfiguredThreshold: Number.isFinite(configuredWebThreshold) ? configuredWebThreshold : null,
      webThresholdRelaxed: Boolean(
        structured.web_search?.web_source_threshold_relaxed
        ?? structured.web_source_threshold_relaxed
      ),
      webTopResultScore: Number(
        structured.web_search?.top_result_score
        ?? structured.top_result_score
      ),
      webRawResultsCount: Number(
        structured.web_search?.raw_results_count
        ?? structured.raw_results_count
        ?? 0
      ),
      webRetainedResultsCount: Number(
        structured.web_search?.retained_results_count
        ?? structured.retained_results_count
        ?? webResults.length
      ),
      references,
      chunks: localChunks,
      trace: structured.retrieval_trace || null,
      raw: '',
    }
    message.artifact = pdfReport?.url ? pdfReport : null
    message.loading = false
    scrollChatToBottom()
    return
  }

  const fallbackMessage = payload?.agent?.tool_result?.message || payload?.tool_result?.message || '未返回结构化结果。'
  message.content = fallbackMessage
  message.sourceData = {
    localAnswer: fallbackMessage,
    webPossibleAnswer: '',
    webResults: [],
    references: [],
    chunks: [],
    trace: null,
    raw: JSON.stringify(payload, null, 2),
  }
  message.artifact = null
  message.loading = false
  scrollChatToBottom()
}

function openSourceDrawer(message, preferredTab = '') {
  const nextTab = preferredTab || (messageHasLocalSources(message) ? 'local' : 'web')
  sourceDrawer.visible = true
  sourceDrawer.tab = nextTab
  sourceDrawer.data = {
    ...createEmptySourceData(),
    ...(message?.sourceData || createEmptySourceData()),
  }
}

function ensureSessionShell(sessionId, previewText = '') {
  if (!sessionId) {
    return
  }
  const existingIndex = state.sessions.findIndex((item) => item.session_id === sessionId)
  const now = new Date().toISOString()
  const shell = {
    session_id: sessionId,
    title: previewText.trim() || '新对话',
    preview: previewText.trim() || '处理中...',
    updated_at: now,
    message_count: existingIndex >= 0 ? Number(state.sessions[existingIndex]?.message_count || 0) : 0,
  }

  if (existingIndex >= 0) {
    state.sessions[existingIndex] = {
      ...state.sessions[existingIndex],
      ...shell,
    }
  } else {
    state.sessions.unshift(shell)
  }
}

function hydrateChatHistory(history) {
  state.messages = (history || []).map((item) => {
    if (item.role === 'user') {
      return createUserMessage(item.content || '', [], item.created_at || '')
    }
    return {
      ...createAssistantMessage(item.content || ''),
      id: createMessageId('assistant-history'),
      content: item.content || '',
      createdAt: item.created_at || '',
      routeTrace: item.metadata?.route_trace || null,
      loading: false,
    }
  })
  scrollChatToBottom()
}

async function fetchWorkspaceSettings() {
  if (!state.userId) {
    return createDefaultWorkspaceSettings()
  }
  const response = await fetch(`/api/users/${encodeURIComponent(state.userId)}/workspace/settings`)
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '加载设置失败')
  }
  return payload.settings || {}
}

async function saveWorkspaceSettings() {
  const response = await fetch(`/api/users/${encodeURIComponent(state.userId)}/workspace/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings: workspace.settings }),
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '保存设置失败')
  }
  workspace.settings = {
    ...createDefaultWorkspaceSettings(),
    ...(payload.settings || {}),
  }
  ElMessage.success('设置已保存。')
}

async function clearWorkspaceMemory(scope) {
  const tip = scope === 'session' ? '确认清空当前会话记忆吗？' : '确认清空该用户全部记忆吗？'
  try {
    await ElMessageBox.confirm(tip, '清理记忆', {
      type: 'warning',
    })
  } catch {
    return
  }
  const response = await fetch(`/api/users/${encodeURIComponent(state.userId)}/workspace/memory/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scope,
      session_id: scope === 'session' ? state.activeSessionId : '',
    }),
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '清理记忆失败')
  }
  ElMessage.success(scope === 'session' ? '当前会话记忆已清空。' : '全部记忆已清空。')
}

async function fetchKnowledgeFiles() {
  const response = await fetch('/api/workspace/knowledge/files')
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '加载文件列表失败')
  }
  return payload.files || []
}

async function uploadKnowledgeFiles(files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  const response = await fetch('/api/workspace/knowledge/files', {
    method: 'POST',
    body: formData,
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '上传文件失败')
  }
  return payload.files || []
}

async function deleteKnowledgeFile(path) {
  const response = await fetch(`/api/workspace/knowledge/files/${encodeURIComponent(path)}`, {
    method: 'DELETE',
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '删除文件失败')
  }
  return payload.files || []
}

async function rebuildKnowledgeIndex() {
  const response = await fetch('/api/workspace/knowledge/rebuild-index', {
    method: 'POST',
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '重建索引失败')
  }
  return payload
}

async function fetchToolStatus() {
  const response = await fetch('/api/workspace/tool-status')
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '加载工具状态失败')
  }
  return payload.tools || {}
}

function stopToolStatusPolling() {
  if (workspace.toolPoller) {
    window.clearInterval(workspace.toolPoller)
    workspace.toolPoller = null
  }
}

function startToolStatusPolling() {
  stopToolStatusPolling()
  workspace.toolPoller = window.setInterval(async () => {
    try {
      workspace.toolStatus = await fetchToolStatus()
    } catch {
      stopToolStatusPolling()
    }
  }, 4000)
}

async function refreshWorkspaceTab(tab = workspace.tab) {
  workspace.loading = true
  stopToolStatusPolling()
  try {
    if (tab === 'settings') {
      workspace.settings = {
        ...createDefaultWorkspaceSettings(),
        ...(await fetchWorkspaceSettings()),
      }
      return
    }
    if (tab === 'knowledge') {
      workspace.knowledgeFiles = await fetchKnowledgeFiles()
      return
    }
    if (tab === 'tools') {
      workspace.toolStatus = await fetchToolStatus()
      startToolStatusPolling()
    }
  } catch (error) {
    ElMessage.error(error.message || '加载失败')
  } finally {
    workspace.loading = false
  }
}

function openWorkspace(tab = 'settings') {
  const shouldRefresh = !workspace.visible || workspace.tab === tab
  workspace.visible = true
  if (workspace.tab !== tab) {
    workspace.tab = tab
    return
  }
  if (shouldRefresh) {
    void refreshWorkspaceTab(tab)
  }
}

async function handleKnowledgeUpload(event) {
  const files = Array.from(event.target.files || [])
  if (!files.length) {
    return
  }
  try {
    workspace.knowledgeFiles = await uploadKnowledgeFiles(files)
    ElMessage.success(`已上传 ${files.length} 个文件。`)
  } catch (error) {
    ElMessage.error(error.message || '上传失败')
  } finally {
    if (knowledgeInputRef.value) {
      knowledgeInputRef.value.value = ''
    }
  }
}

async function handleDeleteKnowledgeFile(path) {
  try {
    await ElMessageBox.confirm(`确认删除知识库文件：${path}？`, '删除文件', {
      type: 'warning',
    })
  } catch {
    return
  }
  workspace.knowledgeFiles = await deleteKnowledgeFile(path)
  ElMessage.success('知识库文件已删除。')
}

async function handleRebuildKnowledgeIndex() {
  try {
    await ElMessageBox.confirm('确认重建本地知识库向量索引吗？这可能需要一些时间。', '重建索引', {
      type: 'warning',
    })
  } catch {
    return
  }
  const result = await rebuildKnowledgeIndex()
  ElMessage.success(`重建完成：文档 ${result.source_documents || 0}，分块 ${result.chunk_count || 0}，向量 ${result.vector_count || 0}`)
}

async function refreshSessions(preferredSessionId = '') {
  if (!state.userId) {
    return
  }
  const response = await fetch(`/api/users/${encodeURIComponent(state.userId)}/sessions`)
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '加载会话失败')
  }
  state.sessions = payload.sessions || []

  if (preferredSessionId) {
    setActiveSession(preferredSessionId)
    return
  }

  const storedSessionId = getStoredSessionId()
  const matchedSession = state.sessions.find((item) => item.session_id === storedSessionId)
  if (matchedSession) {
    setActiveSession(matchedSession.session_id)
    return
  }
  if (!state.sessions.some((item) => item.session_id === state.activeSessionId)) {
    setActiveSession('')
  }
}

async function loadSession(sessionId) {
  if (!state.userId || !sessionId) {
    return
  }
  const response = await fetch(`/api/users/${encodeURIComponent(state.userId)}/sessions/${encodeURIComponent(sessionId)}`)
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '加载会话失败')
  }

  const session = payload.session || {}
  const index = state.sessions.findIndex((item) => item.session_id === session.session_id)
  if (index >= 0) {
    state.sessions[index] = session
  }
  setActiveSession(session.session_id || sessionId)
  hydrateChatHistory(payload.history || [])
}

async function deleteSessionById(sessionId) {
  if (!state.userId || !sessionId) {
    return
  }
  try {
    await ElMessageBox.confirm('确认删除该会话吗？', '删除会话', {
      type: 'warning',
    })
  } catch {
    return
  }
  const response = await fetch(`/api/users/${encodeURIComponent(state.userId)}/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '删除会话失败')
  }
  state.sessions = state.sessions.filter((item) => item.session_id !== sessionId)
  if (state.activeSessionId === sessionId) {
    setActiveSession('')
    state.messages = []
  }
}

async function loginUser(userId) {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId }),
  })
  const payload = await parseJsonResponse(response)
  if (!response.ok) {
    throw new Error(payload.detail || '登录失败')
  }

  setActiveUser(payload.user_id)
  state.sessions = payload.sessions || []
  workspace.settings = {
    ...createDefaultWorkspaceSettings(),
    ...(await fetchWorkspaceSettings().catch(() => ({}))),
  }

  const storedSessionId = getStoredSessionId()
  const matchedSession = state.sessions.find((item) => item.session_id === storedSessionId)
  if (matchedSession) {
    await loadSession(matchedSession.session_id)
    return
  }
  setActiveSession('')
  state.messages = []
}

function logoutUser() {
  activeController.value?.abort()
  stopToolStatusPolling()
  window.localStorage.removeItem(STORAGE_KEYS.userId)
  window.localStorage.removeItem(STORAGE_KEYS.sessionId)
  loginForm.userId = ''
  state.userId = ''
  state.activeSessionId = ''
  state.sessions = []
  state.messages = []
  state.composerText = ''
  state.pending = false
  workspace.visible = false
  workspace.settings = createDefaultWorkspaceSettings()
  workspace.knowledgeFiles = []
  workspace.toolStatus = {}
  sourceDrawer.visible = false
  resetComposer()
}

async function handleLogin() {
  const userId = loginForm.userId.trim()
  if (!userId) {
    ElMessage.warning('请输入用户名。')
    return
  }
  loginForm.loading = true
  try {
    await loginUser(userId)
  } catch (error) {
    ElMessage.error(error.message || '登录失败')
  } finally {
    loginForm.loading = false
  }
}

function startNewConversation() {
  setActiveSession('')
  state.messages = []
}

function triggerFileInput() {
  fileInputRef.value?.click()
}

function triggerKnowledgeInput() {
  knowledgeInputRef.value?.click()
}

function handleFileSelection(event) {
  selectedFiles.value = Array.from(event.target.files || [])
}

function removeSelectedFile(index) {
  selectedFiles.value = selectedFiles.value.filter((_, itemIndex) => itemIndex !== index)
  if (fileInputRef.value) {
    const dataTransfer = new DataTransfer()
    selectedFiles.value.forEach((file) => dataTransfer.items.add(file))
    fileInputRef.value.files = dataTransfer.files
  }
}

function applyQuickPrompt(prompt) {
  state.composerText = prompt
}

function handleComposerKeydown(event) {
  if (event.key !== 'Enter' || event.shiftKey) {
    return
  }
  event.preventDefault()
  void submitOrAbort()
}

async function submitOrAbort() {
  if (state.pending) {
    activeController.value?.abort()
    return
  }
  await submitMessage()
}

async function submitMessage() {
  if (!state.userId) {
    ElMessage.warning('请先登录。')
    return
  }

  const rawText = state.composerText
  const files = [...selectedFiles.value]
  if (!rawText.trim() && !files.length) {
    ElMessage.warning('请输入消息或选择附件。')
    return
  }

  const userMessage = createUserMessage(rawText.trim() || '已上传附件', buildDisplayAttachments(files))
  const assistantMessage = createAssistantMessage()
  state.messages.push(userMessage, assistantMessage)
  scrollChatToBottom()

  resetComposer()
  state.pending = true
  const controller = new AbortController()
  activeController.value = controller

  const formData = new FormData()
  formData.append('user_id', state.userId)
  formData.append('session_id', state.activeSessionId || '')
  formData.append('text', rawText)
  files.forEach((file) => formData.append('files', file))

  try {
    const response = await fetch('/api/agent/submit', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
    if (!response.ok) {
      const payload = await parseJsonResponse(response)
      throw new Error(payload.detail || '请求失败')
    }

    let finalPayload = null
    await consumeEventStream(response, ({ type, data }) => {
      if (type === 'accepted') {
        if (data.session_id) {
          ensureSessionShell(data.session_id, rawText.trim() || '新对话')
          setActiveSession(data.session_id)
        }
        appendRouteStreamEvent(assistantMessage, '请求已提交', `session=${data.session_id || state.activeSessionId || 'new'}`)
        return
      }
      if (type === 'status') {
        appendRouteStreamEvent(assistantMessage, data.message || '状态更新', data.stage || '')
        return
      }
      if (type === 'thought') {
        if (data.kind === 'router_delta') {
          updateRouteLlmOutput(assistantMessage, data)
          return
        }
        if (data.kind === 'router_decision') {
          appendRouteDecision(assistantMessage, data)
          return
        }
        appendThoughtEvent(assistantMessage, data)
        return
      }
      if (type === 'tool_start') {
        appendRouteStreamEvent(assistantMessage, `开始执行工具：${data.label || data.tool_name || 'unknown'}`)
        return
      }
      if (type === 'tool_result') {
        appendRouteStreamEvent(
          assistantMessage,
          `工具完成：${data.label || data.tool_name || 'unknown'}`,
          data.summary || data.status || '',
        )
        return
      }
      if (type === 'answer_start') {
        appendRouteStreamEvent(assistantMessage, data.label || '开始生成回答')
        return
      }
      if (type === 'answer_delta') {
        updateStreamingAnswer(assistantMessage, data.delta || '')
        return
      }
      if (type === 'error') {
        throw new Error(data.message || '请求失败')
      }
      if (type === 'final') {
        finalPayload = data
        renderStructuredPayload(assistantMessage, data)
        return false
      }
    })

    if (!finalPayload) {
      throw new Error('未收到最终结果。')
    }

    if (finalPayload.session_id) {
      await refreshSessions(finalPayload.session_id)
      setActiveSession(finalPayload.session_id)
    }
    if (workspace.visible && workspace.tab === 'tools') {
      workspace.toolStatus = await fetchToolStatus()
    }
  } catch (error) {
    const message = error?.name === 'AbortError' ? '任务已取消。' : `请求失败: ${error.message}`
    assistantMessage.content = message
    assistantMessage.loading = false
    assistantMessage.sourceData = {
      ...createEmptySourceData(),
      localAnswer: message,
    }
    if (error?.name !== 'AbortError') {
      ElMessage.error(message)
    }
  } finally {
    if (activeController.value === controller) {
      activeController.value = null
    }
    state.pending = false
    scrollChatToBottom()
  }
}

watch(
  () => workspace.visible,
  (visible) => {
    if (!visible) {
      stopToolStatusPolling()
    }
  },
)

watch(
  () => workspace.tab,
  (tab) => {
    if (workspace.visible) {
      void refreshWorkspaceTab(tab)
    }
  },
)

onMounted(async () => {
  const storedUserId = getStoredUserId()
  if (!storedUserId) {
    return
  }
  loginForm.userId = storedUserId
  try {
    await loginUser(storedUserId)
  } catch (error) {
    ElMessage.error(error.message || '自动登录失败')
    logoutUser()
  }
})

onBeforeUnmount(() => {
  activeController.value?.abort()
  stopToolStatusPolling()
})
</script>

<template>
  <div class="app-shell">
    <section v-if="!state.userId" class="login-shell">
      <el-card class="login-card" shadow="never">
        <p class="login-eyebrow">智能体工作台</p>
        <h1 class="login-title">登录后开始对话</h1>
        <p class="login-copy">输入用户名即可进入工作区。当前使用轻量本地登录，用于区分用户和会话。</p>
        <el-form @submit.prevent="handleLogin">
          <el-form-item label="用户名">
            <el-input v-model="loginForm.userId" placeholder="例如 xzy 或 analyst_01" />
          </el-form-item>
          <el-button type="primary" :loading="loginForm.loading" @click="handleLogin">
            进入工作区
          </el-button>
        </el-form>
      </el-card>
    </section>

    <section v-else class="workspace-shell">
      <aside class="sidebar">
        <div class="sidebar-head">
          <div>
            <p class="sidebar-eyebrow">当前用户</p>
            <h2 class="sidebar-user">{{ state.userId }}</h2>
          </div>
          <el-button text @click="logoutUser">退出</el-button>
        </div>

        <el-button class="new-chat-button" type="primary" plain @click="startNewConversation">
          + 新建对话
        </el-button>

        <section class="sidebar-section sidebar-section-sessions">
          <div class="section-head">
            <span>会话列表</span>
            <el-tag size="small" round>{{ state.sessions.length }}</el-tag>
          </div>
          <div class="session-list">
            <div v-if="!state.sessions.length" class="session-empty">
              还没有历史对话，发送第一条消息后会出现在这里。
            </div>
            <div v-for="session in state.sessions" :key="session.session_id" class="session-row">
              <button
                class="session-item"
                :class="{ 'is-active': session.session_id === state.activeSessionId }"
                @click="loadSession(session.session_id)"
              >
                <span class="session-title">{{ session.title || session.session_id }}</span>
                <span class="session-preview">{{ session.preview || 'No preview' }}</span>
                <span class="session-meta">
                  {{ formatTimestamp(session.updated_at) }} · {{ session.message_count || 0 }}
                </span>
              </button>
              <div class="session-row-actions">
                <el-button
                  class="session-delete-button"
                  text
                  type="danger"
                  size="small"
                  @click.stop="deleteSessionById(session.session_id)"
                >
                  删除
                </el-button>
              </div>
            </div>
          </div>
        </section>

        <section class="sidebar-section sidebar-section-tools">
          <div class="section-head">
            <span>工作台</span>
          </div>
          <div class="sidebar-tools">
            <el-button plain @click="openWorkspace('settings')">设置</el-button>
            <el-button plain @click="openWorkspace('knowledge')">知识库</el-button>
            <el-button plain @click="openWorkspace('tools')">工具状态</el-button>
          </div>
        </section>
      </aside>

      <main class="chat-shell">
        <section ref="chatBodyRef" class="chat-body">
          <div class="chat-body-inner">
            <div v-if="!state.messages.length" class="empty-shell">
              <h3>今天想先做什么？</h3>
              <p>可以直接提问，或从下面的快捷入口开始。首条消息会自动创建新的 session。</p>
              <div class="quick-actions">
                <el-button
                  v-for="prompt in QUICK_PROMPTS"
                  :key="prompt"
                  class="quick-button"
                  plain
                  @click="applyQuickPrompt(prompt)"
                >
                  {{ prompt }}
                </el-button>
              </div>
            </div>

            <article
              v-for="message in state.messages"
              :key="message.id"
              class="message-row"
              :class="`message-row-${message.role}`"
            >
              <div class="message-meta">
                <span>{{ message.role === 'user' ? '用户' : '助手' }}</span>
                <span v-if="message.createdAt">{{ formatTimestamp(message.createdAt) }}</span>
              </div>

              <el-card shadow="never" class="message-card" :class="`message-card-${message.role}`">
                <div class="message-content">
                  <template
                    v-for="(block, blockIndex) in formatAnswerBlocks(message.content)"
                    :key="`${message.id}-content-${blockIndex}`"
                  >
                    <p v-if="block.type === 'paragraph'">
                      {{ block.text }}
                    </p>
                    <div v-else class="answer-reference-list">
                      <div
                        v-for="(line, lineIndex) in block.lines"
                        :key="`${message.id}-reference-${blockIndex}-${lineIndex}`"
                        class="answer-reference-line"
                      >
                        <template
                          v-for="(segment, segmentIndex) in line.segments"
                          :key="`${message.id}-reference-segment-${blockIndex}-${lineIndex}-${segmentIndex}`"
                        >
                          <a
                            v-if="segment.type === 'link'"
                            class="inline-reference-link"
                            :href="segment.href"
                            target="_blank"
                            rel="noreferrer noopener"
                          >
                            {{ segment.text }}
                          </a>
                          <span v-else>{{ segment.text }}</span>
                        </template>
                      </div>
                    </div>
                  </template>
                </div>

                <div v-if="message.attachments?.length" class="attachment-list">
                  <el-tag
                    v-for="attachment in message.attachments"
                    :key="`${attachment.name}-${attachment.sizeBytes}`"
                    round
                  >
                    {{ formatAttachmentLabel(attachment) }}
                  </el-tag>
                </div>

                <div v-if="message.role === 'assistant'" class="message-footer">
                  <div class="message-actions">
                    <el-button
                      v-if="messageHasLocalSources(message)"
                      text
                      size="small"
                      @click="openSourceDrawer(message, 'local')"
                    >
                      {{ localSourceButtonLabel(message) }}
                    </el-button>
                    <el-button
                      v-if="messageHasWebSources(message)"
                      text
                      size="small"
                      @click="openSourceDrawer(message, 'web')"
                    >
                      {{ webSourceButtonLabel(message) }}
                    </el-button>
                    <a
                      v-if="message.artifact?.url"
                      class="artifact-link"
                      :href="message.artifact.url"
                      target="_blank"
                      rel="noreferrer noopener"
                    >
                      打开 PDF 报告
                    </a>
                  </div>

                  <details v-if="hasRouteTrace(message)" class="route-details">
                    <summary>思考与路由过程</summary>
                    <div class="route-body">
                      <section v-if="message.routeStreamEvents?.length" class="route-section">
                        <h4>实时思考</h4>
                        <ol class="route-list">
                          <li v-for="(event, index) in message.routeStreamEvents" :key="`${message.id}-event-${index}`">
                            <div>{{ event.title }}</div>
                            <small v-if="event.detail">{{ event.detail }}</small>
                          </li>
                        </ol>
                      </section>

                      <section v-if="routeModel(message)?.llmTraces?.length" class="route-section">
                        <h4>路由决策</h4>
                        <article
                          v-for="(trace, index) in routeModel(message).llmTraces"
                          :key="`${message.id}-trace-${index}`"
                          class="trace-card"
                        >
                          <p>{{ trace.decision }}</p>
                          <small v-if="trace.selectedNode || trace.elapsedMs">
                            {{ trace.selectedNode || trace.label || '路由' }}
                            <span v-if="trace.elapsedMs"> · {{ Number(trace.elapsedMs).toFixed(0) }} ms</span>
                          </small>
                        </article>
                      </section>

                      <section v-if="routeModel(message)?.reason" class="route-section">
                        <h4>最终路由理由</h4>
                        <p>{{ routeModel(message).reason }}</p>
                      </section>

                      <section v-if="routeModel(message)?.selectedTools?.length" class="route-section">
                        <h4>调用工具</h4>
                        <div class="tool-chip-list">
                          <el-tag
                            v-for="tool in routeModel(message).selectedTools"
                            :key="`${message.id}-${tool}`"
                            round
                          >
                            {{ tool }}
                          </el-tag>
                        </div>
                      </section>

                      <section v-if="routeModel(message)?.executionSteps?.length" class="route-section">
                        <h4>执行步骤</h4>
                        <ol class="route-list">
                          <li
                            v-for="(step, index) in routeModel(message).executionSteps"
                            :key="`${message.id}-step-${index}`"
                          >
                            <div>{{ step.description }}</div>
                            <small v-if="step.detail">{{ step.detail }}</small>
                            <small>
                              {{ step.status || 'completed' }}
                              <span v-if="step.toolName">· {{ step.toolName }}</span>
                              <span v-if="step.elapsedMs">· {{ Number(step.elapsedMs).toFixed(0) }} ms</span>
                            </small>
                          </li>
                        </ol>
                      </section>
                    </div>
                  </details>
                </div>
              </el-card>
            </article>
          </div>
        </section>

        <footer class="composer-shell">
          <div class="composer-inner">
            <input
              ref="fileInputRef"
              hidden
              type="file"
              multiple
              accept=".png,.jpg,.jpeg,.webp,.gif,.bmp,.pdf,.h5ad,image/*,application/pdf"
              @change="handleFileSelection"
            >

            <div v-if="selectedFiles.length" class="selected-file-list">
              <el-tag
                v-for="(file, index) in selectedFiles"
                :key="`${file.name}-${file.size}-${index}`"
                closable
                round
                @close="removeSelectedFile(index)"
              >
                {{ formatAttachmentLabel(file) }}
              </el-tag>
            </div>

            <el-input
              v-model="state.composerText"
              type="textarea"
              :autosize="{ minRows: 2, maxRows: 8 }"
              resize="none"
              placeholder="输入消息，或描述你想执行的分析任务"
              @keydown="handleComposerKeydown"
            />

            <div class="composer-actions">
              <el-button plain @click="triggerFileInput">上传图片 / PDF / h5ad</el-button>
              <el-button type="primary" :loading="state.pending" @click="submitOrAbort">
                {{ state.pending ? '取消' : '发送' }}
              </el-button>
            </div>
          </div>
        </footer>
      </main>
    </section>

    <el-drawer
      v-model="sourceDrawer.visible"
      class="app-drawer"
      size="42%"
      title="来源详情"
      destroy-on-close
    >
      <p class="drawer-copy">{{ sourceSummary }}</p>
      <el-tabs v-model="sourceDrawer.tab">
        <el-tab-pane label="本地" name="local">
          <div v-if="sourceDrawer.data.references.length || sourceDrawer.data.chunks.length || sourceDrawer.data.trace || sourceDrawer.data.raw" class="drawer-stack">
            <section v-if="sourceDrawer.data.references.length" class="drawer-section">
              <h3>参考来源</h3>
              <el-card
                v-for="(reference, index) in sourceDrawer.data.references"
                :key="`${reference.source_path}-${index}`"
                shadow="never"
              >
                <template #header>
                  {{ formatReferenceLabel(reference, index) }}
                </template>
                <p>{{ reference.source_path || '未知来源路径' }}</p>
                <small v-if="reference.score != null">score={{ Number(reference.score).toFixed(4) }}</small>
              </el-card>
            </section>

            <section v-if="sourceDrawer.data.chunks.length" class="drawer-section">
              <h3>检索片段</h3>
              <el-card
                v-for="(chunk, index) in sourceDrawer.data.chunks"
                :key="`${chunk.metadata?.chunk_id || index}`"
                shadow="never"
              >
                <template #header>
                  Chunk {{ index + 1 }} | {{ chunk.metadata?.file_name || 'unknown' }}
                </template>
                <p>{{ chunk.text || '' }}</p>
                <small>score={{ Number(chunk.score || 0).toFixed(4) }} | source={{ chunk.retrieval_source || 'unknown' }}</small>
              </el-card>
            </section>

            <section v-if="sourceDrawer.data.trace" class="drawer-section">
              <h3>检索轨迹</h3>
              <pre>{{ prettyJson(sourceDrawer.data.trace) }}</pre>
            </section>

            <section v-if="sourceDrawer.data.raw" class="drawer-section">
              <h3>原始返回</h3>
              <pre>{{ sourceDrawer.data.raw }}</pre>
            </section>
          </div>
          <el-empty v-else description="暂无本地来源详情。" />
        </el-tab-pane>

        <el-tab-pane label="网页" name="web">
          <div v-if="sourceDrawer.data.webPossibleAnswer || sourceDrawer.data.webResults.length" class="drawer-stack">
            <section
              v-if="sourceDrawer.data.webThresholdRelaxed || sourceDrawer.data.webRawResultsCount > sourceDrawer.data.webRetainedResultsCount"
              class="drawer-section"
            >
              <el-alert
                :title="sourceDrawer.data.webThresholdRelaxed ? '已使用放宽阈值保留网页候选结果' : '部分网页候选结果因阈值被过滤'"
                type="info"
                :closable="false"
              >
                <template #default>
                  配置阈值={{ Number(sourceDrawer.data.webConfiguredThreshold || currentWebThreshold()).toFixed(2) }}
                  <span v-if="sourceDrawer.data.webThresholdRelaxed">
                    ，实际保留阈值={{ Number(sourceDrawer.data.webThreshold || currentWebThreshold()).toFixed(2) }}
                  </span>
                  <span v-if="Number.isFinite(Number(sourceDrawer.data.webTopResultScore))">
                    ，最高分={{ Number(sourceDrawer.data.webTopResultScore).toFixed(2) }}
                  </span>
                  ，返回 {{ Number(sourceDrawer.data.webRawResultsCount || 0) }} 条候选，保留 {{ Number(sourceDrawer.data.webRetainedResultsCount || 0) }} 条。
                </template>
              </el-alert>
            </section>

            <section v-if="sourceDrawer.data.webPossibleAnswer" class="drawer-section">
              <h3>网页候选答案</h3>
              <el-card shadow="never">
                <p
                  v-for="(line, index) in splitLinkedReferenceLines(sourceDrawer.data.webPossibleAnswer)"
                  :key="`web-possible-answer-${index}`"
                  class="drawer-reference-line"
                >
                  <template
                    v-for="(segment, segmentIndex) in line.segments"
                    :key="`web-possible-answer-segment-${index}-${segmentIndex}`"
                  >
                    <a
                      v-if="segment.type === 'link'"
                      class="inline-reference-link"
                      :href="segment.href"
                      target="_blank"
                      rel="noreferrer noopener"
                    >
                      {{ segment.text }}
                    </a>
                    <span v-else>{{ segment.text }}</span>
                  </template>
                </p>
              </el-card>
            </section>

            <section v-if="sourceDrawer.data.webResults.length" class="drawer-section">
              <h3>网页检索结果</h3>
              <el-card
                v-for="(result, index) in sourceDrawer.data.webResults"
                :key="`${result.url || result.title}-${index}`"
                shadow="never"
              >
                <template #header>
                  [{{ result.source_tier || 'web' }}] {{ result.title || 'Untitled' }}
                </template>
                <p>{{ result.snippet || '' }}</p>
                <small>score={{ Number(result.score || 0).toFixed(2) }} | tier={{ result.source_tier || 'web' }}</small>
                <a
                  v-if="result.url"
                  class="artifact-link"
                  :href="result.url"
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  {{ result.url }}
                </a>
              </el-card>
            </section>
          </div>
          <el-empty v-else description="暂无网页来源详情。" />
        </el-tab-pane>
      </el-tabs>
    </el-drawer>

    <el-drawer
      v-model="workspace.visible"
      class="app-drawer"
      size="42%"
      title="工作台"
      destroy-on-close
    >
      <p class="drawer-copy">管理常用设置、知识库文件和工具运行状态。</p>
      <el-tabs v-model="workspace.tab">
        <el-tab-pane label="设置" name="settings">
          <div v-if="workspace.loading" class="drawer-loading">加载中...</div>
          <div v-else class="drawer-stack">
            <el-form label-position="top" class="settings-form">
              <div class="settings-grid">
                <el-form-item label="温度（Temperature）">
                  <el-input-number v-model="workspace.settings.temperature" :min="0" :max="2" :step="0.1" />
                </el-form-item>
                <el-form-item label="最大生成长度">
                  <el-input-number v-model="workspace.settings.max_new_tokens" :min="64" :max="4096" />
                </el-form-item>
                <el-form-item label="短期记忆最大消息数">
                  <el-input-number v-model="workspace.settings.short_term_max_messages" :min="4" :max="100" />
                </el-form-item>
                <el-form-item label="摘要触发阈值">
                  <el-input-number v-model="workspace.settings.short_term_summary_threshold" :min="2" :max="100" />
                </el-form-item>
                <el-form-item label="长期记忆检索 Top K">
                  <el-input-number v-model="workspace.settings.long_term_top_k" :min="1" :max="20" />
                </el-form-item>
                <el-form-item label="本地知识最低置信度">
                  <el-input-number v-model="workspace.settings.local_source_min_score" :min="0" :max="10" :step="0.05" />
                </el-form-item>
                <el-form-item label="网页来源最低置信度">
                  <el-input-number v-model="workspace.settings.web_source_min_score" :min="0" :max="10" :step="0.1" />
                </el-form-item>
              </div>

              <div class="switch-grid">
                <div class="switch-item">
                  <span>启用 Profile Memory</span>
                  <el-switch v-model="workspace.settings.enable_profile_memory" />
                </div>
                <div class="switch-item">
                  <span>启用 Semantic Memory</span>
                  <el-switch v-model="workspace.settings.enable_semantic_memory" />
                </div>
                <div class="switch-item">
                  <span>优先官方来源</span>
                  <el-switch v-model="workspace.settings.search_prefers_official_sources" />
                </div>
              </div>

              <div class="drawer-actions">
                <el-button type="primary" @click="saveWorkspaceSettings">保存设置</el-button>
                <el-button plain :disabled="!state.activeSessionId" @click="clearWorkspaceMemory('session')">
                  清空当前会话记忆
                </el-button>
                <el-button plain @click="clearWorkspaceMemory('all')">清空全部记忆</el-button>
              </div>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="知识库" name="knowledge">
          <div v-if="workspace.loading" class="drawer-loading">加载中...</div>
          <div v-else class="drawer-stack">
            <div class="drawer-actions">
              <input ref="knowledgeInputRef" hidden type="file" multiple @change="handleKnowledgeUpload">
              <el-button type="primary" plain @click="triggerKnowledgeInput">上传文件</el-button>
              <el-button plain @click="handleRebuildKnowledgeIndex">重建向量数据库</el-button>
            </div>

            <el-empty v-if="!workspace.knowledgeFiles.length" description="暂无知识库文件。" />
            <el-card
              v-for="file in workspace.knowledgeFiles"
              :key="file.path"
              shadow="never"
            >
              <template #header>
                <div class="knowledge-card-head">
                  <span>{{ file.name }}</span>
                  <el-button text type="danger" @click="handleDeleteKnowledgeFile(file.path)">
                    删除
                  </el-button>
                </div>
              </template>
              <p>{{ file.path }}</p>
              <small>{{ (Number(file.size_bytes || 0) / 1024).toFixed(1) }} KB | {{ file.updated_at || '' }}</small>
            </el-card>
          </div>
        </el-tab-pane>

        <el-tab-pane label="工具状态" name="tools">
          <div v-if="workspace.loading" class="drawer-loading">加载中...</div>
          <div v-else class="drawer-stack">
            <el-empty v-if="!Object.keys(workspace.toolStatus).length" description="暂无工具状态。" />
            <el-card
              v-for="(item, name) in workspace.toolStatus"
              :key="name"
              shadow="never"
            >
              <template #header>{{ name }}</template>
              <p>状态：{{ item.state || 'unknown' }}</p>
              <small>{{ item.detail || '' }}</small>
              <small>{{ item.updated_at || '' }}</small>
            </el-card>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-drawer>
  </div>
</template>

<style>
:root {
  color-scheme: dark;
  font-family:
    "Noto Sans SC",
    "PingFang SC",
    "Microsoft YaHei",
    sans-serif;
  background: #212121;
  color: #ececec;
  --app-bg: #212121;
  --app-sidebar: #171717;
  --app-panel: #1e1e1e;
  --app-panel-2: #262626;
  --app-panel-3: #2f2f2f;
  --app-border: #3a3a3a;
  --app-border-strong: #4a4a4a;
  --app-text: #ececec;
  --app-muted: #a1a1aa;
  --app-subtle: #7b7b86;
  --app-accent: #10a37f;
  --app-accent-soft: rgba(16, 163, 127, 0.18);
  --app-shadow: 0 16px 48px rgba(0, 0, 0, 0.3);
  --el-color-primary: #10a37f;
  --el-color-primary-light-3: #1bbb93;
  --el-color-primary-light-5: #2fc9a2;
  --el-color-primary-light-7: #61ddbb;
  --el-color-primary-light-8: #84e6cb;
  --el-color-primary-light-9: #adf1e1;
  --el-color-primary-dark-2: #0b8467;
  --el-bg-color: #212121;
  --el-bg-color-page: #212121;
  --el-bg-color-overlay: #262626;
  --el-fill-color-blank: #262626;
  --el-fill-color: #262626;
  --el-fill-color-light: #2d2d2d;
  --el-fill-color-lighter: #303030;
  --el-fill-color-dark: #171717;
  --el-border-color: #3a3a3a;
  --el-border-color-light: #333333;
  --el-border-color-lighter: #2d2d2d;
  --el-border-color-dark: #4a4a4a;
  --el-text-color-primary: #ececec;
  --el-text-color-regular: #d4d4d8;
  --el-text-color-secondary: #a1a1aa;
  --el-text-color-placeholder: #6b7280;
  --el-mask-color: rgba(0, 0, 0, 0.72);
  --el-mask-color-extra-light: rgba(0, 0, 0, 0.4);
  --el-box-shadow-light: 0 12px 40px rgba(0, 0, 0, 0.28);
  --el-box-shadow: 0 18px 48px rgba(0, 0, 0, 0.38);
  --el-box-shadow-dark: 0 22px 60px rgba(0, 0, 0, 0.45);
  --el-card-bg-color: #262626;
  --el-overlay-color-lighter: rgba(0, 0, 0, 0.74);
  --el-disabled-bg-color: #27272a;
  --el-disabled-text-color: #71717a;
  --el-disabled-border-color: #323236;
}

* {
  box-sizing: border-box;
}

html {
  height: 100%;
  background: var(--app-bg);
}

body {
  margin: 0;
  height: 100%;
  overflow: hidden;
  background:
    radial-gradient(circle at top, rgba(16, 163, 127, 0.12) 0%, rgba(16, 163, 127, 0) 24%),
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0) 28%),
    var(--app-bg);
  color: var(--app-text);
}

#app {
  height: 100%;
  overflow: hidden;
}

.app-shell {
  height: 100dvh;
  overflow: hidden;
  background: transparent;
}

.login-shell {
  min-height: 100%;
  display: grid;
  place-items: center;
  padding: 32px;
  overflow: auto;
}

.login-card {
  width: min(520px, 100%);
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(46, 46, 46, 0.96) 0%, rgba(24, 24, 24, 0.96) 100%);
  box-shadow: 0 28px 80px rgba(0, 0, 0, 0.42);
}

.login-card .el-card__body {
  padding: 30px;
}

.login-eyebrow,
.sidebar-eyebrow,
.chat-eyebrow {
  margin: 0 0 10px;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #64d2b0;
}

.login-title,
.chat-title,
.sidebar-user {
  margin: 0;
  font-size: 28px;
  font-weight: 700;
  color: var(--app-text);
}

.login-copy {
  margin: 14px 0 24px;
  line-height: 1.7;
  color: var(--app-muted);
}

.workspace-shell {
  display: grid;
  grid-template-columns: 336px minmax(0, 1fr);
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 24px 20px;
  border-right: 1px solid rgba(255, 255, 255, 0.06);
  background:
    linear-gradient(180deg, rgba(18, 18, 18, 0.96) 0%, rgba(13, 13, 13, 0.98) 100%);
  backdrop-filter: blur(18px);
  min-height: 0;
  overflow: hidden;
}

.sidebar-head,
.section-head,
.knowledge-card-head,
.composer-actions,
.message-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.new-chat-button {
  width: 100%;
  min-height: 44px;
}

.sidebar-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  padding: 16px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.sidebar-section-sessions {
  flex: 1;
}

.sidebar-section-tools {
  flex: 0 0 auto;
  padding: 12px 14px;
  gap: 10px;
}

.session-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: 2px;
}

.session-empty,
.drawer-loading {
  padding: 18px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.05);
  color: var(--app-muted);
  text-align: center;
}

.session-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.session-item {
  width: 100%;
  border: 1px solid rgba(255, 255, 255, 0.04);
  background: rgba(255, 255, 255, 0.03);
  border-radius: 16px;
  padding: 16px 16px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-align: left;
  cursor: pointer;
  transition: 0.2s ease;
  color: inherit;
}

.session-item:hover,
.session-item.is-active {
  border-color: rgba(16, 163, 127, 0.28);
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
}

.session-title {
  font-weight: 600;
  color: var(--app-text);
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 1;
}

.session-preview,
.session-meta {
  color: var(--app-muted);
  font-size: 13px;
}

.session-preview {
  display: -webkit-box;
  line-height: 1.6;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.session-row-actions {
  display: flex;
  justify-content: flex-end;
  padding-right: 4px;
}

.session-delete-button {
  min-height: 28px;
  padding: 0 8px;
}

.chat-shell {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 20px 32px 18px;
  gap: 16px;
  background:
    radial-gradient(circle at top, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0) 18%),
    transparent;
}

.sidebar-tools {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.sidebar-tools .el-button {
  width: 100%;
  min-height: 36px;
  margin: 0;
  padding: 0 8px;
  font-size: 13px;
  white-space: nowrap;
}

.chat-body {
  min-height: 0;
  overflow-y: auto;
  padding-right: 6px;
  scrollbar-color: #4b5563 transparent;
}

.chat-body-inner {
  width: min(980px, 100%);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 6px 2px 28px;
}

.empty-shell {
  min-height: min(58vh, 520px);
  display: grid;
  place-content: center;
  gap: 16px;
  padding: 40px 32px;
  text-align: center;
  color: var(--app-muted);
  border-radius: 28px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  background: rgba(255, 255, 255, 0.02);
}

.quick-actions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px;
  margin-top: 12px;
}

.quick-button {
  white-space: normal;
  min-height: 60px;
  line-height: 1.55;
}

.message-row {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message-row-user {
  align-items: flex-end;
}

.message-row-assistant {
  align-items: flex-start;
}

.message-meta {
  display: flex;
  gap: 10px;
  font-size: 12px;
  color: var(--app-subtle);
  padding: 0 6px;
}

.message-card {
  width: min(860px, 100%);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  box-shadow: var(--app-shadow);
  overflow: hidden;
}

.message-card .el-card__body {
  padding: 22px 24px 20px;
}

.message-card-user {
  background: linear-gradient(180deg, rgba(53, 53, 53, 0.96) 0%, rgba(41, 41, 41, 0.96) 100%);
}

.message-card-assistant {
  background: linear-gradient(180deg, rgba(35, 35, 35, 0.96) 0%, rgba(29, 29, 29, 0.98) 100%);
}

.message-content {
  display: flex;
  flex-direction: column;
  gap: 14px;
  line-height: 1.82;
  color: var(--app-text);
  font-size: 15.5px;
}

.message-content p {
  margin: 0;
}

.answer-reference-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.answer-reference-line,
.drawer-reference-line {
  overflow-wrap: anywhere;
}

.inline-reference-link {
  color: var(--app-accent);
  text-decoration: none;
  border-bottom: 1px solid rgba(104, 168, 255, 0.45);
}

.inline-reference-link:hover {
  color: #9ec7ff;
  border-bottom-color: currentColor;
}

.attachment-list,
.tool-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}

.message-footer {
  margin-top: 18px;
  align-items: flex-start;
}

.message-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.artifact-link {
  color: #7fe8c9;
  text-decoration: none;
  font-size: 14px;
}

.artifact-link:hover {
  text-decoration: underline;
}

.route-details {
  width: 100%;
  margin-top: 18px;
  border-top: 1px dashed rgba(255, 255, 255, 0.08);
  padding-top: 16px;
}

.route-details summary {
  cursor: pointer;
  font-weight: 600;
  color: #d4d4d8;
}

.route-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 12px;
}

.route-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.route-section h4 {
  margin: 0;
  font-size: 14px;
  color: #f4f4f5;
}

.route-list {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.route-list li {
  color: #d4d4d8;
}

.route-list small,
.trace-card small {
  color: var(--app-subtle);
}

.trace-card {
  padding: 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.trace-card p {
  margin: 0 0 8px;
  white-space: pre-wrap;
  color: var(--app-text);
}

.composer-shell {
  padding: 18px 20px;
  border-radius: 24px;
  background:
    linear-gradient(180deg, rgba(40, 40, 40, 0.95) 0%, rgba(28, 28, 28, 0.98) 100%);
  border: 1px solid rgba(255, 255, 255, 0.06);
  box-shadow: var(--app-shadow);
}

.composer-inner {
  width: min(980px, 100%);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.selected-file-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.app-drawer .el-drawer__body {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding-top: 0;
}

.app-drawer .el-drawer {
  background: linear-gradient(180deg, rgba(27, 27, 27, 0.98) 0%, rgba(18, 18, 18, 0.98) 100%);
  color: var(--app-text);
}

.app-drawer .el-drawer__header {
  margin-bottom: 0;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.app-drawer .el-tabs__item {
  color: var(--app-muted);
}

.app-drawer .el-tabs__item.is-active,
.app-drawer .el-tabs__item:hover {
  color: #7fe8c9;
}

.app-drawer .el-tabs__nav-wrap::after {
  background-color: rgba(255, 255, 255, 0.06);
}

.drawer-copy {
  margin: 0;
  color: var(--app-muted);
  line-height: 1.7;
}

.drawer-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.drawer-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.drawer-section h3 {
  margin: 0;
  font-size: 15px;
  color: #f4f4f5;
}

.drawer-section pre {
  margin: 0;
  padding: 16px;
  border-radius: 16px;
  background: #121212;
  border: 1px solid rgba(255, 255, 255, 0.06);
  color: #e5eefb;
  overflow-x: auto;
  white-space: pre-wrap;
  line-height: 1.6;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.switch-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
}

.switch-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.drawer-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.app-shell .el-card {
  --el-card-bg-color: transparent;
  border-color: rgba(255, 255, 255, 0.06);
}

.app-shell .el-form-item__label,
.app-shell .el-empty__description p {
  color: var(--app-muted);
}

.app-shell .el-button.is-plain {
  background: rgba(255, 255, 255, 0.03);
  border-color: rgba(255, 255, 255, 0.08);
  color: var(--app-text);
  min-height: 40px;
}

.app-shell .el-button.is-plain:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(16, 163, 127, 0.3);
  color: #ecfdf5;
}

.app-shell .el-button--text {
  color: var(--app-muted);
}

.app-shell .el-button--text:hover {
  color: #f5f5f5;
}

.app-shell .el-button--primary {
  box-shadow: 0 12px 28px rgba(16, 163, 127, 0.22);
}

.app-shell .el-input__wrapper,
.app-shell .el-textarea__inner,
.app-shell .el-input-number .el-input__wrapper {
  background: rgba(255, 255, 255, 0.04);
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.07) inset !important;
}

.app-shell .el-input__wrapper.is-focus,
.app-shell .el-textarea__inner:focus,
.app-shell .el-input-number .el-input__wrapper.is-focus {
  box-shadow: 0 0 0 1px rgba(16, 163, 127, 0.45) inset !important;
}

.app-shell .el-textarea__inner {
  min-height: 96px !important;
  padding: 14px 16px;
  color: var(--app-text);
}

.app-shell .el-textarea__inner::placeholder,
.app-shell .el-input__inner::placeholder {
  color: var(--app-subtle);
}

.app-shell .el-tag {
  border-color: rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.05);
  color: var(--app-text);
}

.app-shell .el-input-number__increase,
.app-shell .el-input-number__decrease {
  background: rgba(255, 255, 255, 0.03);
  color: var(--app-muted);
}

.app-shell .el-switch__core {
  background: #3f3f46;
  border-color: #3f3f46;
}

.app-shell .el-overlay {
  backdrop-filter: blur(6px);
}

.chat-body::-webkit-scrollbar,
.session-list::-webkit-scrollbar {
  width: 10px;
}

.chat-body::-webkit-scrollbar-thumb,
.session-list::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  border: 2px solid transparent;
  background-clip: padding-box;
}

.chat-body::-webkit-scrollbar-track,
.session-list::-webkit-scrollbar-track {
  background: transparent;
}

@media (max-width: 1100px) {
  .workspace-shell {
    grid-template-columns: 1fr;
    overflow-y: auto;
  }

  .sidebar {
    border-right: 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    overflow: visible;
  }
}

@media (max-width: 780px) {
  .chat-shell,
  .sidebar {
    padding: 18px;
  }

  .chat-header-inner,
  .composer-shell,
  .sidebar-section {
    padding: 16px;
  }

  .settings-grid,
  .switch-grid {
    grid-template-columns: 1fr;
  }

  .sidebar-head {
    align-items: flex-start;
    flex-direction: column;
  }

  .composer-actions,
  .sidebar-tools {
    width: 100%;
  }

  .composer-actions {
    flex-direction: column;
    align-items: stretch;
  }

  .message-card {
    width: 100%;
  }
}
</style>
