from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryMessage:
    role: MessageRole
    content: str
    created_at: str = field(default_factory=utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryMessage":
        return cls(
            role=payload["role"],
            content=payload["content"],
            created_at=payload.get("created_at", utcnow_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class SessionMemory:
    session_id: str
    max_messages: int = 12
    max_approx_tokens: int = 2400
    summary_trigger_threshold: int = 8
    messages: list[MemoryMessage] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def append_message(
        self,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.messages.append(
            MemoryMessage(
                role=role,
                content=content,
                metadata=dict(metadata or {}),
            )
        )

    def get_recent_messages(self, limit: int | None = None) -> list[MemoryMessage]:
        if limit is None or limit >= len(self.messages):
            return list(self.messages)
        return list(self.messages[-limit:])

    def update_state(
        self,
        patch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if patch:
            self.state.update(patch)
        if kwargs:
            self.state.update(kwargs)

    def get_state(self) -> dict[str, Any]:
        return dict(self.state)

    def summarize_if_needed(self) -> bool:
        if not self._needs_summary():
            return False

        keep_count = max(self.max_messages // 2, 1)
        to_summarize = self.messages[:-keep_count]
        self.summary = _summarize_messages(self.summary, to_summarize)
        self.messages = self.messages[-keep_count:]
        return True

    def approx_token_count(self) -> int:
        total_chars = sum(len(message.content) for message in self.messages)
        total_chars += len(self.summary)
        return max(total_chars // 4, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "max_messages": self.max_messages,
            "max_approx_tokens": self.max_approx_tokens,
            "summary_trigger_threshold": self.summary_trigger_threshold,
            "messages": [message.to_dict() for message in self.messages],
            "state": self.state,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionMemory":
        return cls(
            session_id=payload["session_id"],
            max_messages=int(payload.get("max_messages", 12)),
            max_approx_tokens=int(payload.get("max_approx_tokens", 2400)),
            summary_trigger_threshold=int(
                payload.get("summary_trigger_threshold", 8)
            ),
            messages=[
                MemoryMessage.from_dict(item)
                for item in payload.get("messages") or []
            ],
            state=dict(payload.get("state") or {}),
            summary=str(payload.get("summary") or ""),
        )

    def _needs_summary(self) -> bool:
        return (
            len(self.messages) > self.max_messages
            or len(self.messages) >= self.summary_trigger_threshold
            or self.approx_token_count() > self.max_approx_tokens
        )


@dataclass(slots=True)
class SemanticMemoryHit:
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BuiltMemoryContext:
    user_id: str
    session_id: str
    profile: dict[str, Any] = field(default_factory=dict)
    semantic_memories: list[SemanticMemoryHit] = field(default_factory=list)
    recent_messages: list[MemoryMessage] = field(default_factory=list)
    task_state: dict[str, Any] = field(default_factory=dict)
    short_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "profile": self.profile,
            "semantic_memories": [item.to_dict() for item in self.semantic_memories],
            "recent_messages": [item.to_dict() for item in self.recent_messages],
            "task_state": self.task_state,
            "short_summary": self.short_summary,
        }


def _summarize_messages(
    existing_summary: str,
    messages: list[MemoryMessage],
) -> str:
    compact_lines: list[str] = []
    if existing_summary.strip():
        compact_lines.append(existing_summary.strip())

    for message in messages:
        content = " ".join(message.content.split())
        if not content:
            continue
        trimmed = content[:160]
        prefix = {
            "user": "用户",
            "assistant": "助手",
            "tool": "工具",
            "system": "系统",
        }.get(message.role, message.role)
        compact_lines.append(f"{prefix}: {trimmed}")

    if not compact_lines:
        return ""

    merged = " | ".join(compact_lines)
    return merged[:1200]
