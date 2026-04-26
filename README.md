# Agent Backend + Chat UI

本项目当前包含：
- FastAPI 后端入口：`backend/main.py`
- LangGraph Agent：`backend/agent.py`
- 本地 RAG、网页搜索、单细胞分析、多模态问答工具链
- 登录、会话管理、长短期记忆与前端工作台

## 1. 目录说明

- `backend/main.py`：API 入口、SSE 流式输出、会话与工作台接口。
- `backend/agent.py`：LangGraph 图结构、节点逻辑、ReAct 路由、工具执行、finalize 与记忆写回。
- `backend/tools/`：LLM、RAG、Web Search、单细胞分析工具。
- `backend/memory.py`：Redis 短期记忆 + MySQL 原始长期记忆 + Qdrant 语义向量检索。
- `build_index.py`：重建本地知识库 BM25 + Qdrant 索引。
- `frontend/`：聊天 UI。

## 2. 环境准备

```bash
conda activate agent
pip install -r requirements.txt
```

如果启用了 Redis 短期记忆，需要先确保本机 Redis 可用。最简单的方式是直接启动：

```bash
redis-server
```

## 3. 配置项

从 `.env.example` 复制为 `.env` 后按需填写。当前真实运行至少要关注这几组变量：

- 本地模型：
  - `LLM_MODEL_PATH`
  - `EMBEDDING_MODEL_PATH`
- Web Search：
  - `SERPER_API_KEY`
- 记忆系统：
  - `REDIS_URL`
  - `AGENT_MEMORY_TTL_MINUTES`
  - `AGENT_MEMORY_MYSQL_DB`
  - `AGENT_MEMORY_MYSQL_USER`
  - `AGENT_MEMORY_MYSQL_PASSWORD`
  - `AGENT_MEMORY_MYSQL_HOST`
  - `AGENT_MEMORY_MYSQL_PORT`
  - `AGENT_MEMORY_VECTOR_COLLECTION`
  - `AGENT_MEMORY_VECTOR_SCORE_THRESHOLD`
- 行为控制：
  - `SHORT_TERM_MAX_MESSAGES`
  - `SHORT_TERM_MAX_APPROX_TOKENS`
  - `SHORT_TERM_SUMMARY_THRESHOLD`
  - `LONG_TERM_TOP_K`
  - `ENABLE_PROFILE_MEMORY`
  - `ENABLE_SEMANTIC_MEMORY`
  - `SEMANTIC_MEMORY_COLLECTION`
  - `QDRANT_LOCK_TIMEOUT_SECONDS`

说明：
- Web Search 不再内置默认密钥；未配置 `SERPER_API_KEY` 时会直接报错并返回统一工具错误结果。
- 短期记忆使用 Redis；长期记忆原文落 MySQL，语义检索走本地 Qdrant 向量库。

## 4. 启动服务

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：`http://127.0.0.1:8000/`

## 5. 重建本地知识库索引

```bash
python build_index.py
```

该脚本会：
- 解析 `data/local_knowledge/`
- 生成 `data/local_knowledge_index/bm25.pkl`
- 重建 `data/local_knowledge_index/qdrant`
- 刷新 `data/local_knowledge_index/chunks.jsonl`

全局知识库只服务 `data/local_knowledge/`。用户在会话里普通上传的 PDF 只作为会话附件保存，不会自动构建 session RAG index，也不会混进全局知识库。只有当用户明确要求“上传/加入/保存到本地知识库”时，后端才会把该 PDF 复制到 `data/local_knowledge/`；复制后需要重建全局索引，RAG 才能检索到它。

Markdown、TXT、代码转 Markdown 等非 PDF RAG 文件仍使用会话临时目录：

```text
data/users/<user_id>/sessions/<session_id>/uploads/
data/users/<user_id>/sessions/<session_id>/tmp/rag_index/
```

也就是说，同一个 session 后续继续提问时，可以复用该 session 的非 PDF 临时 RAG index；没有 session 临时 RAG 文件时，才使用全局 `data/local_knowledge_index/`。

## 6. 工具返回协议

RAG、Web Search、单细胞分析在 Agent 内部统一保存为同一类 `tool_result` dict，至少包含：

```json
{
  "status": "ok",
  "tool_name": "local_knowledge_base",
  "answer": "...",
  "artifacts": [],
  "references": [],
  "metrics": {},
  "meta": {}
}
```

说明：
- `status`：`ok` 或 `error`。
- `tool_name`：例如 `local_knowledge_base`、`web_search`、`single_cell_analysis`。
- `answer`：工具可读摘要或证据文本。
- `artifacts`：PDF、图片、表格裁剪等可下载/可展示产物。
- `references`：RAG chunk、网页链接等引用来源。
- `metrics`：耗时、命中数量、chunk 数量等。
- `meta`：索引目录、检索参数、输入文件等调试信息。

## 7. 会话与记忆

- 登录接口：`POST /api/auth/login`
- 会话列表：`GET /api/users/{user_id}/sessions`
- 会话详情：`GET /api/users/{user_id}/sessions/{session_id}`
- 提交消息：`POST /api/agent/submit`
- 清理短期记忆：`POST /api/users/{user_id}/workspace/memory/clear`

默认会话目录：

```text
data/users/<user_id>/sessions/<session_id>/
```

## 8. 最小校验

```bash
python -m py_compile backend/main.py backend/agent.py
```

然后至少手工验证：

1. 登录后连续发送两条消息，确认仍在同一会话。
2. 上传图片、PDF、h5ad，确认各自路由正确。
3. 本地知识库无证据时，确认会自动回退到网页搜索。
4. Redis 关闭时，确认记忆层会报清晰错误而不是静默失效。
