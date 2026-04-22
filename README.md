# Agent Backend + Chat UI

本项目当前包含：
- FastAPI 后端入口（`backend/main.py`）
- LangGraph 编排与工具路由（`backend/planning.py`）
- 本地 RAG + Web Search + 单细胞分析工具链
- 登录与会话管理（按 `user_id/session_id`）
- 长短期记忆（短期会话记忆 + 长期 profile/semantic memory）

## 1. 环境准备

```powershell
& E:\miniconda\shell\condabin\conda-hook.ps1
conda activate agent
pip install -r requirements.txt
```

## 2. 启动服务

```powershell
& E:\miniconda\shell\condabin\conda-hook.ps1; conda activate agent; uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：`http://127.0.0.1:8000/`

## 3. 配置项（.env）

可从 `.env.example` 复制后按需填写。核心分组如下：

- LLM 路由与生成：
  - `QWEN_API_KEY`
  - `QWEN_BASE_URL`
  - `QWEN_MODEL`
  - `QWEN_ROUTER_MODEL`
  - `QWEN_ANALYSIS_MODEL`
- Web 检索：
  - `SERPER_API_KEY`
- 记忆系统：
  - `SHORT_TERM_MAX_MESSAGES`
  - `SHORT_TERM_MAX_APPROX_TOKENS`
  - `SHORT_TERM_SUMMARY_THRESHOLD`
  - `LONG_TERM_TOP_K`
  - `ENABLE_PROFILE_MEMORY`
  - `ENABLE_SEMANTIC_MEMORY`
  - `SEMANTIC_MEMORY_COLLECTION`
  - `SEMANTIC_QDRANT_PATH`
  - `PROFILE_STORAGE_PATH`

## 4. 登录与会话

前端支持：
- 登录（本地轻量登录，使用 `user_id`）
- 新建对话（前端清空当前上下文，首条消息创建新 session）
- 切换历史会话（读取会话消息历史）

后端会话相关接口：
- `POST /api/auth/login`
- `GET /api/users/{user_id}/sessions`
- `GET /api/users/{user_id}/sessions/{session_id}`
- `POST /api/agent/submit`（支持传入 `user_id`、`session_id` 进行续聊）

会话数据默认落盘路径：
- `data/users/<user_id>/sessions/<session_id>/...`

## 5. 最小验证

```powershell
& E:\miniconda\shell\condabin\conda-hook.ps1; conda activate agent; python -m py_compile backend\main.py backend\planning.py backend\sc_analysis\skill.py
```

## 6. 发布前检查清单

1. `python -m py_compile backend\main.py backend\planning.py backend\sc_analysis\skill.py` 通过。  
2. 手工验证：登录 -> 新建对话 -> 连续发送两条消息 -> 切换会话 -> 返回原会话。  
3. 检查 `data/users/...` 下会话目录是否按用户隔离。  
4. 未配置 `QWEN_API_KEY` 时，接口应返回 mock/降级结果且不崩溃。  
5. 若启用语义记忆，确认 `data/local_knowledge_index/qdrant` 可写。  
6. 若启用 PDF 多模态分析，确认 `PyMuPDF` 可导入（`import fitz`）。  
