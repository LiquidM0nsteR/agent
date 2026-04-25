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
- `backend/memory/`：Redis 短期记忆 + SQLite 长期记忆。
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
  - `MEMORY_REDIS_HOST`
  - `MEMORY_REDIS_PORT`
  - `MEMORY_REDIS_DB`
  - `MEMORY_REDIS_PASSWORD`
  - `PROFILE_STORAGE_PATH`
  - `SEMANTIC_QDRANT_PATH`
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
- Web Search 不再内置默认密钥；未配置 `SERPER_API_KEY` 时会返回 `unavailable`。
- 短期记忆使用 Redis，长期记忆使用 SQLite。

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

## 6. 会话与记忆

- 登录接口：`POST /api/auth/login`
- 会话列表：`GET /api/users/{user_id}/sessions`
- 会话详情：`GET /api/users/{user_id}/sessions/{session_id}`
- 提交消息：`POST /api/agent/submit`
- 清理短期记忆：`POST /api/users/{user_id}/workspace/memory/clear`

默认会话目录：

```text
data/users/<user_id>/sessions/<session_id>/
```

## 7. 最小校验

```bash
python -m py_compile backend/main.py backend/agent.py
```

然后至少手工验证：

1. 登录后连续发送两条消息，确认仍在同一会话。
2. 上传图片、PDF、h5ad，确认各自路由正确。
3. 本地知识库无证据时，确认会自动回退到网页搜索。
4. Redis 关闭时，确认记忆层会报清晰错误而不是静默失效。
