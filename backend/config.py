# backend/config.py
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
USERS_DIR = DATA_DIR / "users"
LOCAL_KNOWLEDGE_DIR = DATA_DIR / "local_knowledge"
LOCAL_KNOWLEDGE_INDEX_DIR = DATA_DIR / "local_knowledge_index"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
LOG_DIR = DATA_DIR / "logs"
BACKEND_LOG_PATH = DATA_DIR / "backend_log.txt"

MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_VLLM_MODEL_PATH = MODEL_DIR / "Qwen2.5-VL-7B-Instruct"
DEFAULT_EMBEDDING_MODEL_PATH = MODEL_DIR / "bge-m3"
DEFAULT_RERANK_MODEL_PATH = MODEL_DIR / "bge-reranker-v2-m3"


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


# =========================
# FastAPI / workspace
# =========================

APP_HOST = env_str("AGENT_HOST", "127.0.0.1")
APP_PORT = env_int("AGENT_PORT", 8000)
SERVICE_STARTUP_TIMEOUT = env_float("AGENT_SERVICE_STARTUP_TIMEOUT", 600.0)

WORKSPACE_SETTINGS_DEFAULTS: dict[str, Any] = {
    "temperature": env_float("AGENT_TEMPERATURE", 0.2),
    "max_new_tokens": env_int("AGENT_MAX_NEW_TOKENS", 2048),
    "short_term_max_messages": env_int("AGENT_SHORT_TERM_MAX_MESSAGES", 12),
    "short_term_summary_threshold": env_int("AGENT_SHORT_TERM_SUMMARY_THRESHOLD", 8),
    "long_term_top_k": env_int("AGENT_LONG_TERM_TOP_K", 3),
    "multi_query_count": env_int("AGENT_MULTI_QUERY_COUNT", env_int("RETRIEVAL_MULTI_QUERY_COUNT", 3)),
    "local_source_min_score": env_float("AGENT_LOCAL_SOURCE_MIN_SCORE", env_float("AGENT_RAG_CONFIDENCE_THRESHOLD", 0.58)),
    "web_source_min_score": env_float("AGENT_WEB_SOURCE_MIN_SCORE", 0.35),
    "enable_profile_memory": env_bool("AGENT_ENABLE_PROFILE_MEMORY", False),
    "enable_semantic_memory": env_bool("AGENT_ENABLE_SEMANTIC_MEMORY", True),
    "search_provider": env_str("AGENT_SEARCH_PROVIDER", "serper"),
    "search_prefers_official_sources": env_bool("AGENT_SEARCH_PREFERS_OFFICIAL_SOURCES", True),
}


# =========================
# Redis Stack: short-term memory / LangGraph checkpoint
# =========================

REDIS_BIN = env_str("REDIS_STACK_BIN", "redis-stack-server")
REDIS_HOST = env_str("REDIS_HOST", "127.0.0.1")
REDIS_PORT = env_int("REDIS_PORT", 6379)
REDIS_DB = env_int("REDIS_DB", 0)
REDIS_URL = env_str("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
REDIS_STACK_DATA_DIR = env_path("REDIS_STACK_DATA_DIR", Path.home() / "local" / "redis-stack-data")
REDIS_STACK_LOG_PATH = env_path("REDIS_STACK_LOG_PATH", Path.home() / "local" / "redis-stack-log" / "redis-stack.log")
REDIS_APPENDONLY = env_bool("REDIS_APPENDONLY", True)


# =========================
# MySQL: long-term memory
# =========================

MYSQL_BASE = Path.home() / "local" / "mysql"
MYSQL_CNF = MYSQL_BASE / "etc" / "my.cnf"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_DATABASE = "agent_memory"
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQLD_SAFE = MYSQL_BASE / "bin" / "mysqld_safe"
MYSQL_LOG_PATH = MYSQL_BASE / "logs" / "mysqld_safe.out"


# =========================
# vLLM: local model serving
# =========================

VLLM_BIN = env_str("VLLM_BIN", "vllm")
VLLM_HOST = env_str("VLLM_HOST", "127.0.0.1")
VLLM_PORT = env_int("VLLM_PORT", 8080)
VLLM_BASE_URL = env_str("VLLM_BASE_URL", f"http://{VLLM_HOST}:{VLLM_PORT}/v1")
VLLM_API_KEY = env_str("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_PATH = env_path("VLLM_MODEL_PATH", DEFAULT_VLLM_MODEL_PATH)
VLLM_SERVED_MODEL_NAME = env_str("VLLM_SERVED_MODEL_NAME", env_str("VLLM_MODEL", str(VLLM_MODEL_PATH)))
VLLM_MODEL = env_str("VLLM_MODEL", VLLM_SERVED_MODEL_NAME)
VLLM_LOG_PATH = DATA_DIR / "vllm_log.txt"
VLLM_PID_PATH = DATA_DIR / "vllm.pid"
VLLM_DTYPE = env_str("VLLM_DTYPE", "auto")
VLLM_GENERATION_CONFIG = env_str("VLLM_GENERATION_CONFIG", "vllm")
VLLM_TRUST_REMOTE_CODE = env_bool("VLLM_TRUST_REMOTE_CODE", True)
VLLM_ALLOWED_LOCAL_MEDIA_PATH = env_str("VLLM_ALLOWED_LOCAL_MEDIA_PATH", str(DATA_DIR))
VLLM_EXTRA_ARGS = shlex.split(env_str("VLLM_EXTRA_ARGS", ""))

VLLM_TIMEOUT = env_float("VLLM_TIMEOUT", 600.0)
VLLM_HEALTHCHECK_TIMEOUT = env_float("VLLM_HEALTHCHECK_TIMEOUT", 30.0)
VLLM_MAX_NEW_TOKENS = env_int("VLLM_MAX_NEW_TOKENS", 4096)
VLLM_TEMPERATURE = env_float("VLLM_TEMPERATURE", 0.2)
VLLM_TOP_P = env_float("VLLM_TOP_P", 0.9)
VLLM_TOP_K = env_int("VLLM_TOP_K", 50)
VLLM_REPETITION_PENALTY = env_float("VLLM_REPETITION_PENALTY", 1.05)
VLLM_DO_SAMPLE = env_bool("VLLM_DO_SAMPLE", False)
AGENT_LLM_INSTANCE_COUNT = env_int("AGENT_LLM_INSTANCE_COUNT", 1)
VLLM_GPU_MEMORY_UTILIZATION = env_str("VLLM_GPU_MEMORY_UTILIZATION", "0.70")
VLLM_MAX_MODEL_LEN = env_str("VLLM_MAX_MODEL_LEN", "8192")


# =========================
# RAG / Web / SC
# =========================

SERPER_API_KEY = env_str("SERPER_API_KEY", "")
KNOWLEDGE_BASE_PATH = env_str("KNOWLEDGE_BASE_PATH", str(LOCAL_KNOWLEDGE_DIR))
UPLOAD_WORKDIR = env_str("AGENT_UPLOAD_WORKDIR", str(DATA_DIR / "uploads"))
SC_OUTPUT_DIR = env_str("SC_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "sc_analysis"))
MAX_GRAPH_STEPS = env_int("AGENT_MAX_GRAPH_STEPS", 8)
MAX_LLM_GENERATION_CALLS = env_int("AGENT_MAX_LLM_GENERATION_CALLS", 8)

EMBEDDING_MODEL_PATH = env_str("EMBEDDING_MODEL_PATH", str(DEFAULT_EMBEDDING_MODEL_PATH))
RERANK_MODEL_PATH = env_str("RERANK_MODEL_PATH", str(DEFAULT_RERANK_MODEL_PATH))
RAG_CHUNK_SIZE = env_int("RAG_CHUNK_SIZE", 900)
RAG_CHUNK_OVERLAP = env_int("RAG_CHUNK_OVERLAP", 180)
RAG_TOP_K_BM25 = env_int("RAG_TOP_K_BM25", env_int("RAG_SPARSE_TOP_K", 20))
RAG_TOP_K_VECTOR = env_int("RAG_TOP_K_VECTOR", env_int("RAG_DENSE_TOP_K", 20))
RAG_RRF_K = env_int("RAG_RRF_K", 60)
RAG_TOP_K_FUSED = env_int("RAG_TOP_K_FUSED", 30)
RAG_TOP_K_RERANK = env_int("RAG_TOP_K_RERANK", env_int("RAG_FINAL_TOP_K", 6))
RAG_RERANK_BATCH_SIZE = env_int("RAG_RERANK_BATCH_SIZE", 8)
RAG_RERANK_MAX_LENGTH = env_int("RAG_RERANK_MAX_LENGTH", 512)
RAG_CONFIDENCE_THRESHOLD = env_float("AGENT_RAG_CONFIDENCE_THRESHOLD", env_float("RAG_RERANKER_CONFIDENCE_THRESHOLD", 0.58))
RETRIEVAL_MULTI_QUERY_COUNT = env_int("AGENT_MULTI_QUERY_COUNT", env_int("RETRIEVAL_MULTI_QUERY_COUNT", 3))


# =========================
# Tool display maps
# =========================

TOOL_NAME_MAP = {
    "Chat": "direct_llm",
    "RAG": "local_knowledge_base",
    "WebSearch": "web_search",
    "scAnalysis": "single_cell_pipeline",
}
INTENT_MAP = {
    "rag": "local_knowledge_qa",
    "web_search": "web_search",
    "sc_analysis": "single_cell_analysis",
    "chat": "general_chat",
    "unknown": "general_chat",
}
STEP_DESCRIPTION_MAP = {
    "SupervisorNode": "分析输入并决定路由",
    "Chat": "执行普通对话回答",
    "RAG": "执行本地知识检索",
    "WebSearch": "执行网页搜索",
    "scAnalysis": "执行单细胞分析",
    "FinalNode": "整理最终回答",
}


def ensure_runtime_dirs() -> None:
    for path in (
        DATA_DIR,
        USERS_DIR,
        LOCAL_KNOWLEDGE_DIR,
        LOCAL_KNOWLEDGE_INDEX_DIR,
        LOG_DIR,
        BACKEND_LOG_PATH.parent,
        REDIS_STACK_DATA_DIR,
        REDIS_STACK_LOG_PATH.parent,
        MYSQL_LOG_PATH.parent,
        VLLM_LOG_PATH.parent,
        VLLM_PID_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def export_runtime_env() -> None:
    no_proxy_hosts = ["127.0.0.1", "localhost", "::1"]
    existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    merged_no_proxy = [
        item.strip()
        for item in existing_no_proxy.split(",")
        if item.strip()
    ]
    for host in no_proxy_hosts:
        if host not in merged_no_proxy:
            merged_no_proxy.append(host)

    values = {
        "REDIS_URL": REDIS_URL,
        "VLLM_BASE_URL": VLLM_BASE_URL,
        "VLLM_API_KEY": VLLM_API_KEY,
        "VLLM_MODEL": VLLM_MODEL,
        "VLLM_ALLOWED_LOCAL_MEDIA_PATH": VLLM_ALLOWED_LOCAL_MEDIA_PATH,
        "NO_PROXY": ",".join(merged_no_proxy),
        "no_proxy": ",".join(merged_no_proxy),
        "EMBEDDING_MODEL_PATH": EMBEDDING_MODEL_PATH,
        "RERANK_MODEL_PATH": RERANK_MODEL_PATH,
        "KNOWLEDGE_BASE_PATH": KNOWLEDGE_BASE_PATH,
        "AGENT_UPLOAD_WORKDIR": UPLOAD_WORKDIR,
        "SC_OUTPUT_DIR": SC_OUTPUT_DIR,
        "AGENT_MAX_GRAPH_STEPS": str(MAX_GRAPH_STEPS),
        "AGENT_MAX_LLM_GENERATION_CALLS": str(MAX_LLM_GENERATION_CALLS),
        "RAG_TOP_K_BM25": str(RAG_TOP_K_BM25),
        "RAG_TOP_K_VECTOR": str(RAG_TOP_K_VECTOR),
        "RAG_RRF_K": str(RAG_RRF_K),
        "RAG_TOP_K_FUSED": str(RAG_TOP_K_FUSED),
        "RAG_TOP_K_RERANK": str(RAG_TOP_K_RERANK),
        "RAG_RERANK_BATCH_SIZE": str(RAG_RERANK_BATCH_SIZE),
        "RAG_RERANK_MAX_LENGTH": str(RAG_RERANK_MAX_LENGTH),
        "AGENT_RAG_CONFIDENCE_THRESHOLD": str(RAG_CONFIDENCE_THRESHOLD),
    }
    for key, value in values.items():
        if value != "":
            os.environ[key] = str(value)


def vllm_command() -> list[str]:
    cmd = [
        VLLM_BIN,
        "serve",
        str(VLLM_MODEL_PATH),
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--served-model-name", VLLM_SERVED_MODEL_NAME,
        "--api-key", VLLM_API_KEY,
        "--dtype", VLLM_DTYPE,
        "--generation-config", VLLM_GENERATION_CONFIG,
        "--gpu-memory-utilization", VLLM_GPU_MEMORY_UTILIZATION,
        "--max-model-len", VLLM_MAX_MODEL_LEN,
        "--allowed-local-media-path", VLLM_ALLOWED_LOCAL_MEDIA_PATH,
    ]
    if VLLM_TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    cmd.extend(VLLM_EXTRA_ARGS)
    return cmd
