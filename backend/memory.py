from __future__ import annotations

import atexit
import configparser
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pymysql
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, get_buffer_string
from langgraph.checkpoint.redis import RedisSaver
from langgraph.runtime import Runtime
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "memory"
SHORT_TERM_DIR = DATA_DIR / "short_term"
LONG_TERM_DIR = DATA_DIR / "long_term"
MYSQL_CNF_PATH = Path(
    os.getenv(
        "AGENT_MEMORY_MYSQL_CNF",
        str(Path.home() / "local" / "mysql" / "etc" / "my.cnf"),
    )
)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_TTL_MINUTES = int(os.getenv("AGENT_MEMORY_TTL_MINUTES", "1440"))
DEFAULT_SUMMARY_TRIGGER = int(os.getenv("AGENT_SHORT_TERM_SUMMARY_TRIGGER", "24"))
DEFAULT_KEEP_MESSAGES = int(os.getenv("AGENT_SHORT_TERM_KEEP_MESSAGES", "12"))
DEFAULT_LONG_TERM_TOP_K = int(os.getenv("AGENT_LONG_TERM_TOP_K", "3"))
MAX_LONG_TERM_CANDIDATES = int(os.getenv("AGENT_LONG_TERM_MAX_CANDIDATES", "200"))
MYSQL_DATABASE = os.getenv("AGENT_MEMORY_MYSQL_DB", "agent_memory")
MYSQL_USER = os.getenv("AGENT_MEMORY_MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("AGENT_MEMORY_MYSQL_PASSWORD", "123456")

SHORT_TERM_DIR.mkdir(parents=True, exist_ok=True)
LONG_TERM_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PREFIX = "Here is a summary of the conversation to date:\n\n"


@dataclass(slots=True)
class LongTermMemoryHit:
    content: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": round(float(self.score), 4),
            "metadata": dict(self.metadata),
        }


class LocalSummaryChatModel(SimpleChatModel):
    model_path: str = "backend.tools.LLM.chat"

    @property
    def _llm_type(self) -> str:
        return "local_summary_chat_model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_path": self.model_path}

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        del stop, run_manager, kwargs
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            prompt = str(messages[0].content)
        else:
            prompt = get_buffer_string(messages)

        try:
            from .tools import LLM as llm_module
        except ImportError:
            from tools import LLM as llm_module

        return str(llm_module.chat(prompt=prompt)).strip()


class MemoryManager:
    def __init__(self) -> None:
        self.summary_model = LocalSummaryChatModel()
        self.checkpointer = RedisSaver(
            redis_url=REDIS_URL,
            ttl={
                "default_ttl": REDIS_TTL_MINUTES,
                "refresh_on_read": True,
            },
        )
        self.checkpointer.setup()
        self._mysql_base_config = self._load_mysql_base_config()
        self._ensure_mysql_schema()

    def thread_id(self, user_id: str, session_id: str) -> str:
        return f"{user_id}::{session_id}"

    def prepare_state(self, state: dict[str, Any]) -> dict[str, Any]:
        user_id = str(state.get("user_id") or "anonymous")
        session_id = str(state.get("session_id") or "default")
        query = self._last_user_text(state)
        workspace_settings = dict(state.get("workspace_settings") or {})

        patch: dict[str, Any] = {
            "memory_context": "",
            "long_term_memories": [],
        }

        summary_patch = self._summarize_messages(
            user_id=user_id,
            session_id=session_id,
            messages=list(state.get("messages") or []),
            workspace_settings=workspace_settings,
        )
        if summary_patch:
            patch["messages"] = summary_patch["messages"]

        memory_context = self.load_long_term_context(
            user_id=user_id,
            session_id=session_id,
            query=query,
            top_k=int(workspace_settings.get("long_term_top_k", DEFAULT_LONG_TERM_TOP_K)),
            enable_profile=bool(workspace_settings.get("enable_profile_memory", True)),
            enable_semantic=bool(workspace_settings.get("enable_semantic_memory", True)),
        )
        patch["memory_context"] = memory_context["context"]
        patch["long_term_memories"] = memory_context["hits"]
        return patch

    def load_long_term_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int,
        enable_profile: bool,
        enable_semantic: bool,
    ) -> dict[str, Any]:
        profile = self._load_profile(user_id) if enable_profile else {}
        hits: list[LongTermMemoryHit] = []
        if enable_semantic and query.strip():
            hits = self._search_long_term_memories(
                user_id=user_id,
                session_id=session_id,
                query=query,
                top_k=max(top_k, 0),
            )

        context = self._build_memory_context(profile, hits)
        self._write_retrieval_snapshot(user_id, profile, hits, query)
        return {
            "context": context,
            "profile": profile,
            "hits": [hit.to_dict() for hit in hits],
        }

    def store_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_input: str,
        final_answer: str,
        state: dict[str, Any],
        workspace_settings: dict[str, Any] | None = None,
    ) -> None:
        settings = dict(workspace_settings or {})
        if not final_answer.strip():
            return

        if bool(settings.get("enable_profile_memory", True)):
            profile_updates = self._extract_profile_updates(user_input, state)
            if profile_updates:
                self._upsert_profile(user_id, profile_updates)
                self._write_profile_snapshot(user_id, self._load_profile(user_id))

        if bool(settings.get("enable_semantic_memory", True)):
            record = {
                "user_id": user_id,
                "session_id": session_id,
                "memory_type": "conversation_turn",
                "content": self._build_long_term_record(user_input, final_answer, state),
                "metadata": {
                    "intent": str(state.get("intent") or ""),
                    "steps": list(state.get("steps") or []),
                    "tool_results": sorted(dict(state.get("tool_results") or {}).keys()),
                    "input_kind": str(state.get("input_kind") or ""),
                },
            }
            self._insert_long_term_memory(record)
            self._append_long_term_export(user_id, record)

    def clear_session(self, user_id: str, session_id: str) -> int:
        cleared = 0
        self.checkpointer.delete_thread(self.thread_id(user_id, session_id))
        summary_dir = SHORT_TERM_DIR / user_id / session_id
        if summary_dir.exists():
            for path in summary_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(summary_dir.glob("**/*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            if summary_dir.exists():
                summary_dir.rmdir()
            cleared += 1

        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM long_term_memories WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                cleared += int(cursor.rowcount or 0)
        return cleared

    def clear_user(self, user_id: str) -> int:
        cleared = 0
        sessions_dir = PROJECT_ROOT / "data" / "users" / user_id / "sessions"
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    cleared += self.clear_session(user_id, session_dir.name)

        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM user_profiles WHERE user_id = %s",
                    (user_id,),
                )
                cleared += int(cursor.rowcount or 0)

        long_term_dir = LONG_TERM_DIR / user_id
        if long_term_dir.exists():
            for path in long_term_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(long_term_dir.glob("**/*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            if long_term_dir.exists():
                long_term_dir.rmdir()
            cleared += 1
        return cleared

    def close(self) -> None:
        redis_client = getattr(self.checkpointer, "_redis", None)
        if redis_client is None:
            return
        redis_client.close()
        if getattr(redis_client, "connection_pool", None):
            redis_client.connection_pool.disconnect()

    def _summarize_messages(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[BaseMessage],
        workspace_settings: dict[str, Any],
    ) -> dict[str, Any] | None:
        trigger_messages = max(
            2,
            int(
                workspace_settings.get(
                    "short_term_summary_threshold",
                    DEFAULT_SUMMARY_TRIGGER,
                )
            ),
        )
        keep_messages = max(
            1,
            int(
                workspace_settings.get(
                    "short_term_max_messages",
                    DEFAULT_KEEP_MESSAGES,
                )
            ),
        )
        middleware = SummarizationMiddleware(
            self.summary_model,
            trigger=("messages", trigger_messages),
            keep=("messages", keep_messages),
        )
        patch = middleware.before_model(
            {"messages": messages},
            Runtime(),
        )
        if not patch:
            return None

        summary_text = self._extract_summary_text(patch.get("messages", []))
        if summary_text:
            self._write_short_term_summary(user_id, session_id, summary_text)
        return patch

    def _extract_summary_text(self, messages: list[Any]) -> str:
        for message in messages:
            content = getattr(message, "content", "")
            additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
            if additional_kwargs.get("lc_source") != "summarization":
                continue
            text = str(content or "")
            if text.startswith(SUMMARY_PREFIX):
                return text[len(SUMMARY_PREFIX) :].strip()
            return text.strip()
        return ""

    def _write_short_term_summary(self, user_id: str, session_id: str, summary: str) -> None:
        summary_dir = SHORT_TERM_DIR / user_id / session_id
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "summary.md").write_text(summary, encoding="utf-8")

    def _load_mysql_base_config(self) -> dict[str, Any]:
        parser = configparser.ConfigParser()
        parser.read(MYSQL_CNF_PATH, encoding="utf-8")

        client = parser["client"] if parser.has_section("client") else {}
        mysql = parser["mysql"] if parser.has_section("mysql") else {}
        mysqld = parser["mysqld"] if parser.has_section("mysqld") else {}

        config: dict[str, Any] = {
            "user": MYSQL_USER,
            "password": MYSQL_PASSWORD,
            "charset": "utf8mb4",
            "autocommit": True,
            "cursorclass": DictCursor,
        }

        socket_path = os.getenv("AGENT_MEMORY_MYSQL_SOCKET", "").strip()
        host = os.getenv(
            "AGENT_MEMORY_MYSQL_HOST",
            client.get("host", mysql.get("host", mysqld.get("bind-address", "127.0.0.1"))),
        ).strip()
        if host == "localhost":
            host = "127.0.0.1"
        port = int(
            os.getenv(
                "AGENT_MEMORY_MYSQL_PORT",
                client.get("port", mysql.get("port", mysqld.get("port", "3306"))),
            )
        )

        if socket_path:
            if not Path(socket_path).exists():
                raise FileNotFoundError(f"MySQL socket 不存在：{socket_path}")
            config["unix_socket"] = socket_path
        else:
            config["host"] = host or "127.0.0.1"
            config["port"] = port
        return config

    def _connect_server(self):
        return pymysql.connect(**self._mysql_base_config)

    def _connect_db(self):
        return pymysql.connect(db=MYSQL_DATABASE, **self._mysql_base_config)

    def _ensure_mysql_schema(self) -> None:
        with self._connect_server() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4")

        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id VARCHAR(128) NOT NULL,
                        profile_key VARCHAR(128) NOT NULL,
                        profile_value_json JSON NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, profile_key)
                    ) CHARACTER SET utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id BIGINT NOT NULL AUTO_INCREMENT,
                        user_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        memory_type VARCHAR(64) NOT NULL,
                        content MEDIUMTEXT NOT NULL,
                        metadata_json JSON NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        INDEX idx_memories_user_created (user_id, created_at DESC),
                        INDEX idx_memories_user_session_created (
                            user_id,
                            session_id,
                            created_at DESC
                        )
                    ) CHARACTER SET utf8mb4
                    """
                )

    def _load_profile(self, user_id: str) -> dict[str, Any]:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT profile_key, profile_value_json FROM user_profiles WHERE user_id = %s",
                    (user_id,),
                )
                rows = cursor.fetchall()
        profile: dict[str, Any] = {}
        for row in rows:
            try:
                profile[str(row["profile_key"])] = json.loads(row["profile_value_json"])
            except Exception:
                profile[str(row["profile_key"])] = row["profile_value_json"]
        return profile

    def _upsert_profile(self, user_id: str, updates: dict[str, Any]) -> None:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                for key, value in updates.items():
                    cursor.execute(
                        """
                        INSERT INTO user_profiles (user_id, profile_key, profile_value_json)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            profile_value_json = VALUES(profile_value_json)
                        """,
                        (user_id, key, json.dumps(value, ensure_ascii=False)),
                    )

    def _insert_long_term_memory(self, record: dict[str, Any]) -> None:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO long_term_memories (
                        user_id,
                        session_id,
                        memory_type,
                        content,
                        metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        record["user_id"],
                        record["session_id"],
                        record["memory_type"],
                        record["content"],
                        json.dumps(record["metadata"], ensure_ascii=False),
                    ),
                )

    def _search_long_term_memories(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int,
    ) -> list[LongTermMemoryHit]:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT session_id, memory_type, content, metadata_json, created_at
                    FROM long_term_memories
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, MAX_LONG_TERM_CANDIDATES),
                )
                rows = cursor.fetchall()

        hits: list[LongTermMemoryHit] = []
        for row in rows:
            raw_metadata = row.get("metadata_json") or "{}"
            try:
                metadata = json.loads(raw_metadata)
            except Exception:
                metadata = {}
            metadata.setdefault("session_id", row.get("session_id", ""))
            metadata.setdefault("memory_type", row.get("memory_type", ""))
            metadata.setdefault("created_at", str(row.get("created_at") or ""))

            content = str(row.get("content") or "")
            score = self._score_memory_hit(query, content, metadata, session_id)
            if score <= 0:
                continue
            hits.append(LongTermMemoryHit(content=content, score=score, metadata=metadata))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(top_k, 0)]

    def _score_memory_hit(
        self,
        query: str,
        content: str,
        metadata: dict[str, Any],
        current_session_id: str,
    ) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        haystack = f"{content}\n{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}"
        haystack_tokens = set(self._tokenize(haystack))
        overlap = sum(1 for token in query_tokens if token in haystack_tokens)

        score = overlap / max(len(set(query_tokens)), 1)
        if query.strip() and query.strip().lower() in haystack.lower():
            score += 0.5
        if str(metadata.get("session_id") or "") == current_session_id:
            score += 0.15
        return round(score, 6)

    def _build_memory_context(
        self,
        profile: dict[str, Any],
        hits: list[LongTermMemoryHit],
    ) -> str:
        sections: list[str] = []
        if profile:
            lines = []
            for key, value in profile.items():
                lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
            sections.append("用户长期画像:\n" + "\n".join(lines))

        if hits:
            lines = []
            for index, hit in enumerate(hits, start=1):
                lines.append(
                    f"{index}. [score={round(hit.score, 3)}] {self._truncate_text(hit.content, 280)}"
                )
            sections.append("相关长期记忆:\n" + "\n".join(lines))

        return "\n\n".join(sections).strip()

    def _current_turn_id(self, state: dict[str, Any]) -> str:
        return str(state.get("current_turn_id") or "")

    def _is_current_turn_item(self, state: dict[str, Any], item: dict[str, Any]) -> bool:
        turn_id = self._current_turn_id(state)
        if not turn_id:
            return True
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            return str(metadata.get("turn_id") or "") == turn_id
        return str(item.get("turn_id") or "") == turn_id

    def _current_turn_observations(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in list(state.get("observations") or [])
            if isinstance(item, dict) and self._is_current_turn_item(state, item)
        ]

    def _current_turn_tool_results(self, state: dict[str, Any]) -> dict[str, Any]:
        current = state.get("current_tool_results")
        if (
            isinstance(current, dict)
            and str(state.get("current_tool_results_turn_id") or "") == self._current_turn_id(state)
        ):
            return dict(current)
        if self._current_turn_id(state):
            return {}
        return dict(state.get("tool_results") or {})

    def _build_long_term_record(
        self,
        user_input: str,
        final_answer: str,
        state: dict[str, Any],
    ) -> str:
        lines = [
            f"用户问题: {user_input.strip()}",
            f"最终回答: {final_answer.strip()}",
        ]
        intent = str(state.get("intent") or "").strip()
        if intent:
            lines.append(f"意图: {intent}")
        tool_results = sorted(self._current_turn_tool_results(state).keys())
        if tool_results:
            lines.append(f"涉及工具: {', '.join(tool_results)}")
        observations = self._current_turn_observations(state)
        if observations:
            latest = observations[-1]
            latest_content = str(latest.get("content") or latest.get("error") or "").strip()
            if latest_content:
                lines.append(f"最近观察: {self._truncate_text(latest_content, 320)}")
        return "\n".join(lines).strip()

    def _extract_profile_updates(
        self,
        user_input: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        text = user_input.strip()
        if not text:
            return {}
        updates: dict[str, Any] = {}
        lowered = text.lower()
        if any(token in text for token in ("中文", "汉语")):
            updates["preferred_language"] = "zh"
        if any(token in lowered for token in ("english", "英文")):
            updates["preferred_language"] = "en"
        if any(token in text for token in ("单细胞", "scgpt", "h5ad")):
            updates["domain_focus"] = "single_cell_analysis"
        if any(token in lowered for token in ("记住", "偏好", "默认", "prefer", "remember", "always")):
            updates["user_preference"] = text
        selected_tools = sorted(self._current_turn_tool_results(state).keys())
        if selected_tools:
            updates["last_selected_tools"] = selected_tools
        return updates

    def _append_long_term_export(self, user_id: str, record: dict[str, Any]) -> None:
        user_dir = LONG_TERM_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "turns.jsonl"
        export_record = {
            **record,
            "created_at": self._now_iso(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(export_record, ensure_ascii=False) + "\n")

    def _write_profile_snapshot(self, user_id: str, profile: dict[str, Any]) -> None:
        user_dir = LONG_TERM_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_retrieval_snapshot(
        self,
        user_id: str,
        profile: dict[str, Any],
        hits: list[LongTermMemoryHit],
        query: str,
    ) -> None:
        user_dir = LONG_TERM_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": query,
            "profile": profile,
            "hits": [hit.to_dict() for hit in hits],
            "updated_at": self._now_iso(),
        }
        (user_dir / "last_retrieved.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text.lower()) if token]

    def _truncate_text(self, text: str, limit: int) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _last_user_text(self, state: dict[str, Any]) -> str:
        if state.get("user_input"):
            return str(state["user_input"])
        for message in reversed(list(state.get("messages") or [])):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")


_MEMORY_MANAGER: MemoryManager | None = None
_MEMORY_MANAGER_LOCK = Lock()


def get_memory_manager() -> MemoryManager:
    global _MEMORY_MANAGER
    with _MEMORY_MANAGER_LOCK:
        if _MEMORY_MANAGER is None:
            _MEMORY_MANAGER = MemoryManager()
            atexit.register(_MEMORY_MANAGER.close)
        return _MEMORY_MANAGER
