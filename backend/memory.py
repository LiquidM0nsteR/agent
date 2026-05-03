from __future__ import annotations

import atexit
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

import pymysql
from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.checkpoint.redis import RedisSaver
from pymysql.cursors import DictCursor

from . import config as C
from .util import (
    DEFAULT_MEMORY_SCORE_THRESHOLD,
    SHORT_ENTITY_MEMORY_SCORE_THRESHOLD,
    compact_memory_text,
    memory_query_features,
    summarize_memory_turn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "memory"
LONG_TERM_DIR = DATA_DIR / "long_term"
LONG_TERM_VECTOR_DIR = DATA_DIR / "long_term_vectors"
LOCAL_BGE_MODEL = PROJECT_ROOT / "models" / "bge-m3"

MYSQL_DATABASE = C.MYSQL_DATABASE
MYSQL_USER = C.MYSQL_USER
MYSQL_PASSWORD = C.MYSQL_PASSWORD
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_TTL_MINUTES = int(os.getenv("AGENT_MEMORY_TTL_MINUTES", "1440"))
DEFAULT_LONG_TERM_TOP_K = int(os.getenv("AGENT_LONG_TERM_TOP_K", "3"))
LONG_TERM_VECTOR_COLLECTION = os.getenv("AGENT_MEMORY_VECTOR_COLLECTION", "long_term_memories")
DEFAULT_MEMORY_EMBEDDING_MODEL = os.getenv("AGENT_MEMORY_EMBEDDING_MODEL_PATH") or os.getenv("EMBEDDING_MODEL_PATH") or (str(LOCAL_BGE_MODEL) if LOCAL_BGE_MODEL.exists() else "BAAI/bge-m3")

LONG_TERM_DIR.mkdir(parents=True, exist_ok=True)
LONG_TERM_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
_EMBEDDER_CACHE: dict[str, Any] = {}
_MEMORY_MANAGER: "MemoryManager | None" = None


@dataclass(slots=True)
class LongTermMemoryHit:
    content: str
    score: float
    metadata: dict[str, Any]
    summary: str = ""
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "summary": self.summary,
            "score": round(float(self.score), 4),
            "metadata": dict(self.metadata),
            "tags": list(self.tags or []),
        }


def _get_embedder() -> Any:
    if DEFAULT_MEMORY_EMBEDDING_MODEL not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[DEFAULT_MEMORY_EMBEDDING_MODEL] = SentenceTransformer(DEFAULT_MEMORY_EMBEDDING_MODEL)
    return _EMBEDDER_CACHE[DEFAULT_MEMORY_EMBEDDING_MODEL]


def _encode_texts(texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
    vectors = _get_embedder().encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, vector)) for vector in vectors]


class MemoryManager:
    def __init__(self) -> None:
        self.checkpointer = RedisSaver(redis_url=REDIS_URL, ttl={"default_ttl": REDIS_TTL_MINUTES, "refresh_on_read": True})
        self.checkpointer.setup()
        self._vector_lock = Lock()
        self._mysql_base_config = self._load_mysql_base_config()
        self._ensure_mysql_schema()

    def thread_id(self, user_id: str, session_id: str) -> str:
        return f"{user_id}::{session_id}"

    def prepare_state(self, state: dict[str, Any]) -> dict[str, Any]:
        user_id = str(state.get("user_id") or "anonymous")
        session_id = str(state.get("session_id") or "default")
        query = self._last_user_text(state)
        settings = dict(state.get("workspace_settings") or {})
        top_k = int(settings.get("long_term_top_k", DEFAULT_LONG_TERM_TOP_K))
        hits, filtered = ([], [])
        if bool(settings.get("enable_semantic_memory", True)) and query.strip() and top_k > 0:
            hits, filtered = self.search_long_term_memories_with_trace(user_id=user_id, session_id=session_id, query=query, top_k=top_k)
        recent_rows = self._load_recent_session_rows(user_id=user_id, session_id=session_id, limit=max(8, top_k * 3))
        recent_turns = self._select_recent_session_turns(query, recent_rows, limit=2)
        session_summary = self._build_session_summary(query, recent_rows)
        memory_sections = {
            "current_user_query": query,
            "retrieved_long_term_memory": [self._hit_prompt_item(hit) for hit in hits],
            "session_summary": session_summary,
            "recent_session_turns": recent_turns,
            "tool_context": [],
            "system_instruction": "Memory is auxiliary context only; current user query and current tool results have higher priority.",
        }
        memory_context = self._build_context(memory_sections)
        memory_debug = {
            "current_query": query,
            "query_features": self._debug_query_features(query),
            "selected_memories": [self._debug_hit(hit) for hit in hits],
            "filtered_memories": filtered[:10],
            "session_summary_categories": sorted(session_summary.keys()),
            "recent_session_turn_count": len(recent_turns),
            "final_prompt_memory_block": compact_memory_text(memory_context, 1200),
        }
        return {
            "memory_context": memory_context,
            "long_term_memories": [hit.to_dict() for hit in hits],
            "memory_sections": memory_sections,
            "memory_debug": memory_debug,
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
        del workspace_settings
        if not final_answer.strip():
            return
        summary, tags, memory_type = summarize_memory_turn(user_input, final_answer)
        checksum = hashlib.sha256(json.dumps({"user_id": user_id, "session_id": session_id, "summary": summary, "tags": tags}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        record = {
            "user_id": user_id,
            "session_id": session_id,
            "memory_type": memory_type,
            "content": summary,
            "summary": summary,
            "tags": tags,
            "source_turn_id": str(state.get("current_turn_id") or ""),
            "relevance": {"inject_only_when_relevant": True, "score_threshold": DEFAULT_MEMORY_SCORE_THRESHOLD},
            "version": 1,
            "checksum": checksum,
            "metadata": {
                "intent": str(state.get("intent") or ""),
                "intent_type": str(state.get("intent_type") or ""),
                "steps": list(state.get("steps") or []),
                "tool_results": sorted(dict(state.get("tool_results") or {}).keys()),
                "input_kind": str(state.get("input_kind") or ""),
                "summary": summary,
                "tags": tags,
            },
        }
        record["id"] = self._insert_long_term_memory(record)
        self._upsert_vector(record)
        self._append_export(user_id, record)
        self._invalidate_memory_cache(user_id=user_id, session_id=session_id, memory_type=memory_type)

    def search_long_term_memories(self, *, user_id: str, session_id: str, query: str, top_k: int) -> list[LongTermMemoryHit]:
        hits, _ = self.search_long_term_memories_with_trace(user_id=user_id, session_id=session_id, query=query, top_k=top_k)
        return hits

    def search_long_term_memories_with_trace(self, *, user_id: str, session_id: str, query: str, top_k: int) -> tuple[list[LongTermMemoryHit], list[dict[str, Any]]]:
        del session_id
        query_vector = _encode_texts([query], batch_size=1)[0]
        points = self._query_points(user_id=user_id, vector=query_vector, limit=max(top_k * 4, top_k + 4, 8))
        candidate_ids: list[int] = []
        scored_payloads: list[tuple[int, float, dict[str, Any]]] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            memory_id = int(payload.get("memory_id") or 0)
            if memory_id:
                candidate_ids.append(memory_id)
                scored_payloads.append((memory_id, float(getattr(point, "score", 0.0) or 0.0), payload))
        rows = self._fetch_active_memory_rows(candidate_ids)
        candidates: list[LongTermMemoryHit] = []
        for memory_id, score, payload in scored_payloads:
            row = rows.get(memory_id)
            if not row:
                metadata = self._metadata_from_payload(payload)
                metadata.update({"memory_id": memory_id, "stale_vector_payload": True})
                candidates.append(LongTermMemoryHit(content="", summary="", score=score, metadata=metadata, tags=[]))
            else:
                candidates.append(self._hit_from_row(row, score=score))
        return self._filter_hits(query, candidates, top_k=top_k)

    def clear_session(self, user_id: str, session_id: str) -> int:
        self.checkpointer.delete_thread(self.thread_id(user_id, session_id))
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE long_term_memories SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s AND session_id = %s AND deleted_at IS NULL",
                    (user_id, session_id),
                )
                deleted = int(cursor.rowcount or 0)
        self._delete_vectors(user_id=user_id, session_id=session_id)
        self._invalidate_memory_cache(user_id=user_id, session_id=session_id)
        return deleted

    def clear_user(self, user_id: str) -> int:
        self._delete_vectors(user_id=user_id)
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE long_term_memories SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s AND deleted_at IS NULL",
                    (user_id,),
                )
                deleted = int(cursor.rowcount or 0)
        self._invalidate_memory_cache(user_id=user_id)
        return deleted

    def update_memory(self, *, memory_id: int, user_id: str, session_id: str | None = None, summary: str | None = None, tags: list[str] | None = None, memory_type: str | None = None) -> bool:
        row = self._fetch_memory_row(memory_id=memory_id, user_id=user_id)
        if not row:
            return False
        next_summary = summary if summary is not None else str(row.get("summary") or row.get("content") or "")
        next_tags = tags if tags is not None else list(self._json_field(row.get("tags_json"), []))
        next_type = memory_type or str(row.get("memory_type") or "conversation_summary")
        checksum = hashlib.sha256(json.dumps({"user_id": user_id, "session_id": row.get("session_id"), "summary": next_summary, "tags": next_tags}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        params: list[Any] = [next_type, next_summary, next_summary, json.dumps(next_tags, ensure_ascii=False), checksum, user_id, memory_id]
        session_clause = ""
        if session_id is not None:
            session_clause = " AND session_id = %s"
            params.append(session_id)
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE long_term_memories
                    SET memory_type = %s, content = %s, summary = %s, tags_json = %s,
                        checksum = %s, version = version + 1, updated_at = CURRENT_TIMESTAMP, deleted_at = NULL
                    WHERE user_id = %s AND id = %s{session_clause}
                    """,
                    tuple(params),
                )
                updated = bool(cursor.rowcount)
        if updated:
            refreshed = self._fetch_memory_row(memory_id=memory_id, user_id=user_id)
            if refreshed:
                self._upsert_vector(self._record_from_row(refreshed))
                self._invalidate_memory_cache(user_id=user_id, session_id=str(refreshed.get("session_id") or ""), memory_type=str(refreshed.get("memory_type") or ""))
        return updated

    def delete_memory(self, *, memory_id: int, user_id: str, session_id: str | None = None) -> bool:
        row = self._fetch_memory_row(memory_id=memory_id, user_id=user_id)
        params: list[Any] = [user_id, memory_id]
        session_clause = ""
        if session_id is not None:
            session_clause = " AND session_id = %s"
            params.append(session_id)
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE long_term_memories SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s AND id = %s AND deleted_at IS NULL{session_clause}",
                    tuple(params),
                )
                deleted = bool(cursor.rowcount)
        if deleted:
            actual_session = str((row or {}).get("session_id") or session_id or "")
            self._delete_vectors(user_id=user_id, session_id=actual_session or None, memory_ids=[memory_id])
            self._invalidate_memory_cache(user_id=user_id, session_id=actual_session, memory_type=str((row or {}).get("memory_type") or ""))
        return deleted

    def close(self) -> None:
        redis_client = getattr(self.checkpointer, "_redis", None)
        if redis_client is not None:
            redis_client.close()
            if getattr(redis_client, "connection_pool", None):
                redis_client.connection_pool.disconnect()

    def _load_mysql_base_config(self) -> dict[str, Any]:
        return {
            "host": C.MYSQL_HOST,
            "port": C.MYSQL_PORT,
            "user": MYSQL_USER,
            "password": MYSQL_PASSWORD,
            "charset": "utf8mb4",
            "autocommit": True,
            "cursorclass": DictCursor,
            "connect_timeout": 2,
        }

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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id BIGINT NOT NULL AUTO_INCREMENT,
                        user_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        memory_type VARCHAR(64) NOT NULL,
                        content MEDIUMTEXT NOT NULL,
                        summary TEXT NULL,
                        tags_json JSON NULL,
                        metadata_json JSON NULL,
                        source_turn_id VARCHAR(128) NULL,
                        relevance_json JSON NULL,
                        version INT NOT NULL DEFAULT 1,
                        checksum CHAR(64) NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        deleted_at TIMESTAMP NULL DEFAULT NULL,
                        PRIMARY KEY (id),
                        INDEX idx_user_created (user_id, created_at DESC),
                        INDEX idx_user_session_created (user_id, session_id, created_at DESC),
                        INDEX idx_user_deleted_updated (user_id, deleted_at, updated_at),
                        INDEX idx_user_session_deleted (user_id, session_id, deleted_at)
                    ) CHARACTER SET utf8mb4
                """)
                self._ensure_memory_columns(cursor)
                self._ensure_memory_indexes(cursor)

    def _insert_long_term_memory(self, record: dict[str, Any]) -> int:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO long_term_memories
                    (user_id, session_id, memory_type, content, summary, tags_json, metadata_json,
                     source_turn_id, relevance_json, version, checksum)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record["user_id"],
                        record["session_id"],
                        record["memory_type"],
                        record["content"],
                        record.get("summary", ""),
                        json.dumps(record.get("tags") or [], ensure_ascii=False),
                        json.dumps(record["metadata"], ensure_ascii=False),
                        record.get("source_turn_id", ""),
                        json.dumps(record.get("relevance") or {}, ensure_ascii=False),
                        int(record.get("version") or 1),
                        record.get("checksum", ""),
                    ),
                )
                return int(cursor.lastrowid or 0)

    def _ensure_memory_columns(self, cursor: Any) -> None:
        cursor.execute(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'long_term_memories'",
            (MYSQL_DATABASE,),
        )
        columns = {str(row.get("COLUMN_NAME")) for row in cursor.fetchall() or []}
        additions = {
            "summary": "ALTER TABLE long_term_memories ADD COLUMN summary TEXT NULL AFTER content",
            "tags_json": "ALTER TABLE long_term_memories ADD COLUMN tags_json JSON NULL AFTER summary",
            "source_turn_id": "ALTER TABLE long_term_memories ADD COLUMN source_turn_id VARCHAR(128) NULL AFTER metadata_json",
            "relevance_json": "ALTER TABLE long_term_memories ADD COLUMN relevance_json JSON NULL AFTER source_turn_id",
            "version": "ALTER TABLE long_term_memories ADD COLUMN version INT NOT NULL DEFAULT 1 AFTER relevance_json",
            "checksum": "ALTER TABLE long_term_memories ADD COLUMN checksum CHAR(64) NULL AFTER version",
            "updated_at": "ALTER TABLE long_term_memories ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at",
            "deleted_at": "ALTER TABLE long_term_memories ADD COLUMN deleted_at TIMESTAMP NULL DEFAULT NULL AFTER updated_at",
        }
        for column, ddl in additions.items():
            if column not in columns:
                cursor.execute(ddl)

    def _ensure_memory_indexes(self, cursor: Any) -> None:
        cursor.execute(
            "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'long_term_memories'",
            (MYSQL_DATABASE,),
        )
        indexes = {str(row.get("INDEX_NAME")) for row in cursor.fetchall() or []}
        if "idx_user_deleted_updated" not in indexes:
            cursor.execute("CREATE INDEX idx_user_deleted_updated ON long_term_memories (user_id, deleted_at, updated_at)")
        if "idx_user_session_deleted" not in indexes:
            cursor.execute("CREATE INDEX idx_user_session_deleted ON long_term_memories (user_id, session_id, deleted_at)")

    def _load_recent_session_rows(self, *, user_id: str, session_id: str, limit: int) -> list[dict[str, Any]]:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, session_id, memory_type, content, summary, tags_json,
                           metadata_json, source_turn_id, relevance_json, version, checksum,
                           created_at, updated_at, deleted_at
                    FROM long_term_memories
                    WHERE user_id = %s AND session_id = %s AND deleted_at IS NULL
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT %s
                    """,
                    (user_id, session_id, int(limit)),
                )
                return list(cursor.fetchall() or [])

    def _fetch_memory_row(self, *, memory_id: int, user_id: str) -> dict[str, Any] | None:
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, session_id, memory_type, content, summary, tags_json,
                           metadata_json, source_turn_id, relevance_json, version, checksum,
                           created_at, updated_at, deleted_at
                    FROM long_term_memories
                    WHERE id = %s AND user_id = %s
                    LIMIT 1
                    """,
                    (memory_id, user_id),
                )
                return cursor.fetchone()

    def _fetch_active_memory_rows(self, memory_ids: list[int]) -> dict[int, dict[str, Any]]:
        ids = [int(item) for item in memory_ids if int(item or 0) > 0]
        if not ids:
            return {}
        placeholders = ", ".join(["%s"] * len(ids))
        with self._connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, user_id, session_id, memory_type, content, summary, tags_json,
                           metadata_json, source_turn_id, relevance_json, version, checksum,
                           created_at, updated_at, deleted_at
                    FROM long_term_memories
                    WHERE id IN ({placeholders}) AND deleted_at IS NULL
                    """,
                    tuple(ids),
                )
                return {int(row["id"]): row for row in cursor.fetchall() or []}

    def _record_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = self._json_field(row.get("metadata_json"), {})
        tags = self._json_field(row.get("tags_json"), metadata.get("tags") or [])
        return {
            "id": int(row.get("id") or 0),
            "user_id": str(row.get("user_id") or ""),
            "session_id": str(row.get("session_id") or ""),
            "memory_type": str(row.get("memory_type") or ""),
            "content": str(row.get("content") or ""),
            "summary": str(row.get("summary") or metadata.get("summary") or ""),
            "tags": list(tags),
            "metadata": dict(metadata),
            "source_turn_id": str(row.get("source_turn_id") or ""),
            "relevance": self._json_field(row.get("relevance_json"), {}),
            "version": int(row.get("version") or 1),
            "checksum": str(row.get("checksum") or ""),
            "created_at": str(row.get("created_at") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "deleted_at": str(row.get("deleted_at") or ""),
        }

    def _hit_from_row(self, row: dict[str, Any], *, score: float) -> LongTermMemoryHit:
        record = self._record_from_row(row)
        metadata = dict(record["metadata"])
        metadata.update({
            "memory_id": record["id"],
            "session_id": record["session_id"],
            "memory_type": record["memory_type"],
            "source_turn_id": record["source_turn_id"],
            "checksum": record["checksum"],
            "tags": record["tags"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "retrieval_source": "qdrant_vector",
        })
        return LongTermMemoryHit(content=record["content"], summary=record["summary"], score=score, metadata=metadata, tags=record["tags"])

    def _metadata_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = self._json_field(payload.get("metadata_json"), {})
        metadata.update({
            "session_id": payload.get("session_id", ""),
            "memory_type": payload.get("memory_type", ""),
            "tags": self._json_field(payload.get("tags_json"), []),
            "checksum": payload.get("checksum", ""),
            "retrieval_source": "qdrant_vector",
        })
        return metadata

    def _filter_hits(self, query: str, hits: list[LongTermMemoryHit], *, top_k: int) -> tuple[list[LongTermMemoryHit], list[dict[str, Any]]]:
        query_features = memory_query_features(query)
        substantive_query_tags = query_features["tags"] - {"general", "user_preferences", "task_state"}
        selected: list[LongTermMemoryHit] = []
        filtered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            display_text = self._display_memory_text(hit)
            hit_features = memory_query_features(f"{display_text} {hit.content}")
            hit_tags = set(hit.tags or hit.metadata.get("tags") or [])
            hit_tags.update(hit_features["tags"])
            substantive_hit_tags = hit_tags - {"general", "user_preferences", "task_state"}
            exact_entity = bool(query_features["entities"] and any(entity in hit_features["normalized"] for entity in query_features["entities"]))
            token_overlap = bool(query_features["tokens"] & hit_features["tokens"])
            topic_overlap = bool(substantive_query_tags & substantive_hit_tags)
            entity_mismatch = bool(query_features["entities"] and hit_features["entities"] and not exact_entity)
            checksum = str(hit.metadata.get("checksum") or hashlib.sha256(display_text.lower().encode("utf-8")).hexdigest())
            reason = ""
            if checksum in seen:
                reason = "duplicate"
            elif not display_text:
                reason = "deleted_or_empty_memory"
            elif query_features["is_short_entity"] and not exact_entity:
                reason = "short_query_without_exact_entity_match"
            elif entity_mismatch and hit.score < 0.82:
                reason = "entity_mismatch_without_high_semantic_match"
            elif hit.score < DEFAULT_MEMORY_SCORE_THRESHOLD and not exact_entity:
                reason = f"below_threshold:{hit.score:.3f}<{DEFAULT_MEMORY_SCORE_THRESHOLD:.3f}"
            elif query_features["is_short_entity"] and hit.score < SHORT_ENTITY_MEMORY_SCORE_THRESHOLD and not exact_entity:
                reason = f"short_query_below_threshold:{hit.score:.3f}<{SHORT_ENTITY_MEMORY_SCORE_THRESHOLD:.3f}"
            elif not (exact_entity or token_overlap or topic_overlap or hit.score >= max(0.72, DEFAULT_MEMORY_SCORE_THRESHOLD + 0.12)):
                reason = "no_lexical_topic_or_high_semantic_match"
            if reason:
                filtered.append(self._debug_hit(hit, reason=reason))
                continue
            seen.add(checksum)
            selected.append(LongTermMemoryHit(content=display_text, summary=hit.summary or display_text, score=hit.score, metadata=hit.metadata, tags=sorted(hit_tags)))
            if len(selected) >= top_k:
                break
        return selected, filtered

    def _select_recent_session_turns(self, query: str, rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        hits = [self._hit_from_row(row, score=1.0) for row in rows]
        selected, _ = self._filter_hits(query, hits, top_k=limit)
        return [self._hit_prompt_item(hit) for hit in selected]

    def _build_session_summary(self, query: str, rows: list[dict[str, Any]]) -> dict[str, list[str]]:
        query_features = memory_query_features(query)
        substantive_query_tags = query_features["tags"] - {"general", "user_preferences", "task_state"}
        summary: dict[str, list[str]] = {}
        for row in rows:
            text = str(row.get("summary") or row.get("content") or "")
            if not text:
                continue
            text_features = memory_query_features(text)
            exact_entity = bool(query_features["entities"] and any(entity in text_features["normalized"] for entity in query_features["entities"]))
            substantive_text_tags = text_features["tags"] - {"general", "user_preferences", "task_state"}
            if query_features["entities"] and text_features["entities"] and not exact_entity:
                continue
            if query_features["is_short_entity"] and not exact_entity:
                continue
            if not query_features["is_short_entity"] and not (substantive_query_tags & substantive_text_tags or exact_entity):
                continue
            categories = []
            if "user_preferences" in text_features["tags"]:
                categories.append("user_preferences")
            if "task_state" in text_features["tags"]:
                categories.append("task_state")
            if "bioinfo" in text_features["tags"]:
                categories.append("bioinfo_context")
            if "coding" in text_features["tags"]:
                categories.append("coding_context")
            if exact_entity:
                categories.append("entities")
            for category in categories:
                summary.setdefault(category, [])
                item = compact_memory_text(text, 220)
                if item not in summary[category] and len(summary[category]) < 3:
                    summary[category].append(item)
        return summary

    def _get_vector_client(self) -> Any:
        from qdrant_client import QdrantClient

        LONG_TERM_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(LONG_TERM_VECTOR_DIR))

    def _ensure_vector_collection(self, vector_size: int) -> None:
        from qdrant_client import models

        with self._vector_lock:
            client = self._get_vector_client()
            collections = [item.name for item in client.get_collections().collections]
            if LONG_TERM_VECTOR_COLLECTION not in collections:
                client.create_collection(collection_name=LONG_TERM_VECTOR_COLLECTION, vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE))
            client.close()

    def _upsert_vector(self, record: dict[str, Any]) -> None:
        from qdrant_client import models

        vector_text = str(record.get("summary") or record.get("content") or "")
        if not vector_text.strip():
            return
        vector = _encode_texts([vector_text], batch_size=1)[0]
        self._ensure_vector_collection(len(vector))
        with self._vector_lock:
            client = self._get_vector_client()
            client.upsert(collection_name=LONG_TERM_VECTOR_COLLECTION, points=[models.PointStruct(id=self._point_id(record), vector=vector, payload=self._payload(record))])
            client.close()

    def _query_points(self, *, user_id: str, vector: Sequence[float], limit: int) -> list[Any]:
        from qdrant_client import models

        query_filter = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
        with self._vector_lock:
            client = self._get_vector_client()
            collections = [item.name for item in client.get_collections().collections]
            if LONG_TERM_VECTOR_COLLECTION not in collections:
                client.close()
                return []
            result = client.query_points(collection_name=LONG_TERM_VECTOR_COLLECTION, query=list(vector), query_filter=query_filter, limit=limit, with_payload=True)
            client.close()
        return list(getattr(result, "points", result))

    def _delete_vectors(self, *, user_id: str, session_id: str | None = None, memory_ids: list[int] | None = None) -> None:
        from qdrant_client import models

        must = [models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        if session_id is not None:
            must.append(models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)))
        if memory_ids:
            must.append(models.FieldCondition(key="memory_id", match=models.MatchAny(any=[int(item) for item in memory_ids])))
        with self._vector_lock:
            client = self._get_vector_client()
            collections = [item.name for item in client.get_collections().collections]
            if LONG_TERM_VECTOR_COLLECTION not in collections:
                client.close()
                return
            client.delete(collection_name=LONG_TERM_VECTOR_COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(must=must)))
            client.close()

    def _point_id(self, record: dict[str, Any]) -> str:
        key = f"{MYSQL_DATABASE}:{record.get('id') or record.get('memory_id') or record.get('content')}"
        return str(uuid5(NAMESPACE_URL, key))

    def _payload(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory_id": int(record.get("id") or 0),
            "user_id": str(record.get("user_id") or ""),
            "session_id": str(record.get("session_id") or ""),
            "memory_type": str(record.get("memory_type") or ""),
            "content": str(record.get("content") or ""),
            "summary": str(record.get("summary") or ""),
            "tags_json": json.dumps(record.get("tags") or [], ensure_ascii=False),
            "metadata_json": json.dumps(dict(record.get("metadata") or {}), ensure_ascii=False),
            "source_turn_id": str(record.get("source_turn_id") or ""),
            "relevance_json": json.dumps(record.get("relevance") or {}, ensure_ascii=False),
            "version": int(record.get("version") or 1),
            "checksum": str(record.get("checksum") or ""),
        }

    def _append_export(self, user_id: str, record: dict[str, Any]) -> None:
        path = LONG_TERM_DIR / user_id / "long_term_memories.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _invalidate_memory_cache(self, *, user_id: str, session_id: str | None = None, memory_type: str | None = None) -> None:
        redis_client = getattr(self.checkpointer, "_redis", None)
        if redis_client is None:
            return
        patterns = [
            f"agent_memory_cache:{user_id}:*",
            f"agent_memory_cache:{user_id}:{session_id or '*'}:*",
            f"agent_memory_cache:{user_id}:{session_id or '*'}:{memory_type or '*'}:*",
        ]
        for pattern in patterns:
            keys = list(redis_client.scan_iter(match=pattern, count=100))
            if keys:
                redis_client.delete(*keys)

    def _display_memory_text(self, hit: LongTermMemoryHit) -> str:
        if hit.summary:
            return compact_memory_text(hit.summary, 500)
        metadata_summary = str(hit.metadata.get("summary") or "").strip()
        if metadata_summary:
            return compact_memory_text(metadata_summary, 500)
        content = str(hit.content or "")
        user_marker = "User:"
        assistant_marker = "Assistant:"
        if user_marker in content and assistant_marker in content:
            user_part = content.split(assistant_marker, 1)[0].replace(user_marker, "").strip()
            assistant_part = content.split(assistant_marker, 1)[1].split("\nIntent:", 1)[0].strip()
            return compact_memory_text(f"Previous user asked: {user_part}. Prior answer summary: {assistant_part}", 420)
        return compact_memory_text(content, 420)

    def _hit_prompt_item(self, hit: LongTermMemoryHit) -> dict[str, Any]:
        return {
            "memory_id": hit.metadata.get("memory_id"),
            "memory_type": hit.metadata.get("memory_type", ""),
            "session_id": hit.metadata.get("session_id", ""),
            "score": round(float(hit.score), 4),
            "summary": compact_memory_text(hit.summary or hit.content, 360),
            "tags": list(hit.tags or []),
        }

    def _debug_hit(self, hit: LongTermMemoryHit, *, reason: str = "") -> dict[str, Any]:
        payload = {
            "memory_id": hit.metadata.get("memory_id"),
            "session_id": hit.metadata.get("session_id"),
            "memory_type": hit.metadata.get("memory_type"),
            "score": round(float(hit.score), 4),
            "summary": compact_memory_text(hit.summary or self._display_memory_text(hit), 180),
            "tags": list(hit.tags or hit.metadata.get("tags") or []),
        }
        if reason:
            payload["reason"] = reason
        return payload

    @staticmethod
    def _debug_query_features(query: str) -> dict[str, Any]:
        features = memory_query_features(query)
        return {
            "tokens": sorted(features["tokens"]),
            "entities": sorted(features["entities"]),
            "tags": sorted(features["tags"]),
            "is_short_entity": features["is_short_entity"],
        }

    @staticmethod
    def _json_field(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        if isinstance(value, (dict, list)):
            return value
        return json.loads(str(value))

    @staticmethod
    def _last_user_text(state: dict[str, Any]) -> str:
        if state.get("user_input"):
            return str(state["user_input"])
        return get_buffer_string([msg for msg in state.get("messages") or [] if isinstance(msg, HumanMessage)])

    @staticmethod
    def _build_record_text(user_input: str, final_answer: str, state: dict[str, Any]) -> str:
        return "\n".join([f"User: {user_input}", f"Assistant: {final_answer}", f"Intent: {state.get('intent', '')}", f"Steps: {state.get('steps', [])}"]).strip()

    @staticmethod
    def _build_context(sections: dict[str, Any]) -> str:
        retrieved = list(sections.get("retrieved_long_term_memory") or [])
        session_summary = dict(sections.get("session_summary") or {})
        recent_turns = list(sections.get("recent_session_turns") or [])
        if not retrieved and not session_summary and not recent_turns:
            return ""
        lines = [
            "[memory_policy]",
            "- Memory is auxiliary context only; it must not override, expand, or reinterpret current_user_query.",
            "- Ignore memory that is not directly relevant to current_user_query.",
            "[current_user_query]",
            str(sections.get("current_user_query") or ""),
        ]
        if retrieved:
            lines.append("[retrieved_long_term_memory]")
            for index, item in enumerate(retrieved, start=1):
                lines.append(f"{index}. score={item.get('score')} type={item.get('memory_type')} tags={item.get('tags')}: {item.get('summary')}")
        if session_summary:
            lines.append("[session_summary]")
            for category, values in session_summary.items():
                lines.append(f"{category}:")
                for value in values:
                    lines.append(f"- {value}")
        if recent_turns:
            lines.append("[recent_session_turns]")
            for index, item in enumerate(recent_turns, start=1):
                lines.append(f"{index}. type={item.get('memory_type')} tags={item.get('tags')}: {item.get('summary')}")
        return "\n".join(lines).strip()


def get_memory_manager() -> MemoryManager:
    global _MEMORY_MANAGER
    if _MEMORY_MANAGER is None:
        _MEMORY_MANAGER = MemoryManager()
    return _MEMORY_MANAGER


def _close_memory_manager() -> None:
    if _MEMORY_MANAGER is not None:
        _MEMORY_MANAGER.close()


atexit.register(_close_memory_manager)
