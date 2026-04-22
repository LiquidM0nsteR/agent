from __future__ import annotations

import json
import os
import socket
import sqlite3
from typing import Any

from .schemas import SemanticMemoryHit, utcnow_iso


class RedisShortTermStore:
    def __init__(self) -> None:
        self.host = os.getenv("MEMORY_REDIS_HOST", os.getenv("REDIS_HOST", "127.0.0.1"))
        self.port = int(os.getenv("MEMORY_REDIS_PORT", os.getenv("REDIS_PORT", "6379")))
        self.db = int(os.getenv("MEMORY_REDIS_DB", os.getenv("REDIS_DB", "0")))
        self.password = os.getenv("MEMORY_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", ""))
        self.timeout = float(os.getenv("MEMORY_REDIS_TIMEOUT", "2.0"))
        self.key_prefix = os.getenv("MEMORY_REDIS_PREFIX", "agent:short_term")

    def load(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        raw = self._execute("GET", self._key(user_id, session_id))
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            return None
        return json.loads(raw)

    def save(self, user_id: str, session_id: str, payload: dict[str, Any]) -> None:
        self._execute(
            "SET",
            self._key(user_id, session_id),
            json.dumps(payload, ensure_ascii=False),
        )

    def delete(self, user_id: str, session_id: str) -> int:
        deleted = self._execute("DEL", self._key(user_id, session_id))
        return int(deleted or 0)

    def delete_user(self, user_id: str) -> int:
        keys = self._execute("KEYS", self._user_pattern(user_id)) or []
        normalized_keys = [str(item) for item in keys if item]
        if not normalized_keys:
            return 0
        deleted = self._execute("DEL", *normalized_keys)
        return int(deleted or 0)

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{self.key_prefix}:{user_id}:{session_id}"

    def _user_pattern(self, user_id: str) -> str:
        return f"{self.key_prefix}:{user_id}:*"

    def _execute(self, command: str, *args: Any) -> Any:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                reader = sock.makefile("rb")
                if self.password:
                    self._send(sock, "AUTH", self.password)
                    self._read_response(reader)
                if self.db:
                    self._send(sock, "SELECT", self.db)
                    self._read_response(reader)
                self._send(sock, command, *args)
                return self._read_response(reader)
        except OSError as exc:
            raise RuntimeError(f"Redis 不可用: {exc}") from exc

    def _send(self, sock: socket.socket, command: str, *args: Any) -> None:
        parts = [command, *args]
        payload = [f"*{len(parts)}\r\n".encode("utf-8")]
        for item in parts:
            data = str(item).encode("utf-8")
            payload.append(f"${len(data)}\r\n".encode("utf-8"))
            payload.append(data + b"\r\n")
        sock.sendall(b"".join(payload))

    def _read_response(self, reader: Any) -> Any:
        prefix = reader.read(1)
        if not prefix:
            raise RuntimeError("Redis 响应为空。")
        line = reader.readline().rstrip(b"\r\n")
        if prefix == b"+":
            return line.decode("utf-8")
        if prefix == b"-":
            raise RuntimeError(line.decode("utf-8"))
        if prefix == b":":
            return int(line)
        if prefix == b"$":
            length = int(line)
            if length < 0:
                return None
            data = reader.read(length)
            reader.read(2)
            return data.decode("utf-8")
        if prefix == b"*":
            length = int(line)
            if length < 0:
                return []
            return [self._read_response(reader) for _ in range(length)]
        raise RuntimeError(f"未知 Redis 响应类型: {prefix!r}")


class SQLiteLongTermStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def load_profile(self, user_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        profile: dict[str, Any] = {}
        for key, value_json in rows:
            try:
                profile[str(key)] = json.loads(value_json)
            except json.JSONDecodeError:
                continue
        return profile

    def update_profile(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        if not values:
            return self.load_profile(user_id)
        with self._connect() as connection:
            for key, value in values.items():
                connection.execute(
                    """
                    INSERT INTO user_profiles (user_id, key, value_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (user_id, key, json.dumps(value, ensure_ascii=False), utcnow_iso()),
                )
            connection.commit()
        return self.load_profile(user_id)

    def add_memory(self, user_id: str, content: str, metadata: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO long_term_memories (
                    user_id,
                    content,
                    memory_type,
                    importance,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    content,
                    str(metadata.get("memory_type") or "semantic"),
                    float(metadata.get("importance") or 0.5),
                    json.dumps(metadata, ensure_ascii=False),
                    str(metadata.get("created_at") or utcnow_iso()),
                ),
            )
            connection.commit()

    def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int,
    ) -> list[SemanticMemoryHit]:
        if not query.strip():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT content, metadata_json, importance, created_at
                FROM long_term_memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 500
                """,
                (user_id,),
            ).fetchall()

        ranked_hits: list[SemanticMemoryHit] = []
        for content, metadata_json, importance, created_at in rows:
            try:
                metadata = json.loads(metadata_json or "{}")
            except json.JSONDecodeError:
                metadata = {}
            metadata.setdefault("created_at", created_at)
            metadata.setdefault("importance", importance)
            score = score_memory_hit(query, str(content or ""), metadata)
            if score <= 0:
                continue
            ranked_hits.append(
                SemanticMemoryHit(
                    content=str(content or ""),
                    score=score,
                    metadata=metadata,
                )
            )
        ranked_hits.sort(key=lambda item: item.score, reverse=True)
        return ranked_hits[: max(top_k, 0)]

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_long_term_memories_user_created
                ON long_term_memories (user_id, created_at DESC)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def score_memory_hit(query: str, content: str, metadata: dict[str, Any]) -> float:
    haystack = normalize_search_text(
        "\n".join(
            [
                content,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ]
        )
    )
    if not haystack:
        return 0.0

    query_text = normalize_search_text(query)
    if not query_text:
        return 0.0

    score = 0.0
    if query_text in haystack:
        score += 6.0

    for token in query_features(query_text):
        if token in haystack:
            score += 1.2

    score += float(metadata.get("importance") or 0.0)
    return score


def normalize_search_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def query_features(query: str) -> list[str]:
    whitespace_tokens = [token for token in query.split(" ") if len(token) >= 2]
    if whitespace_tokens:
        return whitespace_tokens
    condensed = query.replace(" ", "")
    if len(condensed) < 2:
        return [condensed] if condensed else []
    return [condensed[index : index + 2] for index in range(len(condensed) - 1)]
