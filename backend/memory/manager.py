from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock
from typing import Any

from ..tools.rag.config import RAGConfig
from .schemas import BuiltMemoryContext, SemanticMemoryHit, SessionMemory, utcnow_iso
from .storage import RedisShortTermStore, SQLiteLongTermStore


class MemoryManager:
    """统一管理 Redis 短期记忆与 SQLite 长期记忆。"""

    _session_lock_registry_guard = Lock()
    _session_locks: dict[tuple[str, str], RLock] = {}

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.long_term_store = SQLiteLongTermStore(str(config.profile_storage_path))
        self.short_term_store = RedisShortTermStore()
        self._short_term_cache: dict[tuple[str, str], SessionMemory] = {}

    def _get_session_lock(self, user_id: str, session_id: str) -> RLock:
        cache_key = (user_id, session_id)
        with self._session_lock_registry_guard:
            lock = self._session_locks.get(cache_key)
            if lock is None:
                lock = RLock()
                self._session_locks[cache_key] = lock
            return lock

    @contextmanager
    def _guard_session(self, user_id: str, session_id: str):
        # 同一会话内的短期记忆读写必须串行，避免 cache 与 Redis 状态错位。
        with self._get_session_lock(user_id, session_id):
            yield

    def load_short_term(self, user_id: str, session_id: str) -> SessionMemory:
        with self._guard_session(user_id, session_id):
            cache_key = (user_id, session_id)
            cached = self._short_term_cache.get(cache_key)
            if cached is not None:
                return cached

            payload = self.short_term_store.load(user_id, session_id)
            if payload is None:
                memory = self._new_session_memory(session_id)
            else:
                memory = SessionMemory.from_dict(payload)
                memory.max_messages = self.config.short_term_max_messages
                memory.max_approx_tokens = self.config.short_term_max_approx_tokens
                memory.summary_trigger_threshold = self.config.short_term_summary_threshold

            self._short_term_cache[cache_key] = memory
            return memory

    def save_short_term(
        self,
        user_id: str,
        session_id: str,
        memory: SessionMemory,
    ) -> None:
        with self._guard_session(user_id, session_id):
            memory.summarize_if_needed()
            self.short_term_store.save(user_id, session_id, memory.to_dict())
            self._short_term_cache[(user_id, session_id)] = memory

    def write_short_term(
        self,
        user_id: str,
        session_id: str,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        state_update: dict[str, Any] | None = None,
    ) -> SessionMemory:
        with self._guard_session(user_id, session_id):
            memory = self.load_short_term(user_id, session_id)
            memory.append_message(role=role, content=content, metadata=metadata)
            if state_update:
                memory.update_state(state_update)
            self.save_short_term(user_id, session_id, memory)
            return memory

    def read_short_term(self, user_id: str, session_id: str) -> SessionMemory:
        return self.load_short_term(user_id, session_id)

    def clear_short_term(self, user_id: str, session_id: str | None = None) -> int:
        if session_id:
            with self._guard_session(user_id, session_id):
                self._short_term_cache.pop((user_id, session_id), None)
                return self.short_term_store.delete(user_id, session_id)

        cleared = self.short_term_store.delete_user(user_id)
        for cache_key in list(self._short_term_cache):
            if cache_key[0] == user_id:
                self._short_term_cache.pop(cache_key, None)
        return cleared

    def retrieve_long_term(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[SemanticMemoryHit]:
        if not self.config.enable_semantic_memory:
            return []
        return self.long_term_store.search_memories(
            user_id,
            query,
            top_k or self.config.long_term_top_k,
        )

    def maybe_write_long_term(
        self,
        *,
        user_id: str,
        user_text: str,
        short_term: SessionMemory,
        tool_result: dict[str, Any],
    ) -> dict[str, Any]:
        writes = {"profile_updated": False, "semantic_written": False}
        state = short_term.get_state()
        task_summary = str(state.get("task_summary") or short_term.summary or "").strip()

        if self.config.enable_profile_memory and _should_store_profile(user_text):
            profile_updates = _extract_profile_updates(user_text, state)
            if profile_updates:
                self.long_term_store.update_profile(user_id, profile_updates)
                writes["profile_updated"] = True

        if self.config.enable_semantic_memory and _should_store_semantic(
            user_text, task_summary, tool_result
        ):
            content = _build_semantic_memory_text(state, task_summary)
            metadata = {
                "memory_type": "task_summary",
                "created_at": state.get("updated_at") or utcnow_iso(),
                "importance": 0.7 if tool_result.get("status") == "ok" else 0.4,
                "source": "agent_finalize",
                "intent": state.get("intent", ""),
                "selected_tool": state.get("selected_tool", ""),
            }
            self.long_term_store.add_memory(user_id, content, metadata)
            writes["semantic_written"] = True

        return writes

    def build_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        short_term: SessionMemory | None = None,
    ) -> BuiltMemoryContext:
        with self._guard_session(user_id, session_id):
            effective_short_term = short_term or self.load_short_term(user_id, session_id)
            recent_messages = effective_short_term.get_recent_messages(
                self.config.short_term_max_messages
            )
            task_state = effective_short_term.get_state()
            short_summary = effective_short_term.summary

        profile = (
            self.long_term_store.load_profile(user_id)
            if self.config.enable_profile_memory
            else {}
        )
        semantic_memories = self.retrieve_long_term(
            user_id=user_id,
            query=query,
            top_k=self.config.long_term_top_k,
        )
        return BuiltMemoryContext(
            user_id=user_id,
            session_id=session_id,
            profile=profile,
            semantic_memories=semantic_memories,
            recent_messages=recent_messages,
            task_state=task_state,
            short_summary=short_summary,
        )

    def _new_session_memory(self, session_id: str) -> SessionMemory:
        return SessionMemory(
            session_id=session_id,
            max_messages=self.config.short_term_max_messages,
            max_approx_tokens=self.config.short_term_max_approx_tokens,
            summary_trigger_threshold=self.config.short_term_summary_threshold,
        )


def _should_store_profile(user_text: str) -> bool:
    normalized = user_text.lower()
    return any(
        token in normalized
        for token in (
            "记住",
            "记一下",
            "以后",
            "偏好",
            "习惯",
            "默认",
            "always",
            "prefer",
            "remember",
        )
    )


def _should_store_semantic(
    user_text: str,
    task_summary: str,
    tool_result: dict[str, Any],
) -> bool:
    normalized = user_text.lower()
    if any(token in normalized for token in ("记住", "总结", "结论", "经验", "复用")):
        return True
    if not task_summary.strip() or tool_result.get("status") != "ok":
        return False
    if any(
        tool_result.get(key)
        for key in ("references", "retrieved_chunks", "artifacts", "analysis_result")
    ):
        return True
    return False


def _extract_profile_updates(user_text: str, state: dict[str, Any]) -> dict[str, Any]:
    normalized = user_text.strip()
    if not normalized:
        return {}
    updates: dict[str, Any] = {}
    if "中文" in normalized:
        updates["preferred_language"] = "zh"
    if any(token in normalized for token in ("单细胞", "scrna", "scrna-seq")):
        updates["domain_focus"] = "single_cell_analysis"
    selected_tool = state.get("selected_tool")
    if selected_tool:
        updates["preferred_tool_path"] = selected_tool
    return updates


def _build_semantic_memory_text(state: dict[str, Any], task_summary: str) -> str:
    user_query = state.get("user_query", "")
    intent = state.get("intent", "")
    selected_tool = state.get("selected_tool", "")
    lines = [
        f"用户问题: {user_query}",
        f"识别意图: {intent}",
        f"选择工具: {selected_tool}",
    ]
    if task_summary.strip():
        lines.append(f"任务摘要: {task_summary.strip()}")
    tool_outputs = state.get("tool_outputs") or {}
    if tool_outputs:
        lines.append(f"关键输出: {tool_outputs}")
    return "\n".join(lines)
