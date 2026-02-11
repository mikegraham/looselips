"""Parse LLM chat export JSON into Conversation objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

CHATGPT_URL_PREFIX = "https://chatgpt.com/c/"
CLAUDE_URL_PREFIX = "https://claude.ai/chat/"


@dataclass
class Message:
    role: str  # "user", "assistant"
    text: str


@dataclass
class Conversation:
    id: str
    title: str
    messages: list[Message]
    create_time: datetime | None = None
    update_time: datetime | None = None
    url_prefix: str = CHATGPT_URL_PREFIX

    @property
    def url(self) -> str:
        return self.url_prefix + self.id


def _ts(val: float | str | None) -> datetime | None:
    if val is None:
        return None
    try:
        ts = float(val)
    except ValueError:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=UTC)
    except ValueError:
        return None


def _iso(val: str | None) -> datetime | None:
    if val is None:
        return None
    try:
        dt = datetime.fromisoformat(val)
    except ValueError:
        return None
    # Normalize naive timestamps to UTC for consistency with _ts()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _load_json(path: str | Path | IO[bytes]) -> list[dict[str, object]]:
    """Load a JSON array from a file path or file-like object."""
    if hasattr(path, "read"):
        return json.load(path)  # type: ignore[no-any-return]
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def parse_chatgpt(path: str | Path | IO[bytes]) -> list[Conversation]:
    """Parse a ChatGPT conversations.json export file.

    *path* can be a filesystem path (str or Path) or a readable
    file-like object containing UTF-8 JSON bytes.
    """
    data = _load_json(path)

    conversations = []
    for conv in data:
        mapping = conv.get("mapping", {})

        # Build children lookup
        children: dict[str, list[str]] = {}
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent is not None:
                children.setdefault(parent, []).append(node_id)

        # Find roots: nodes with no parent
        roots = [nid for nid, n in mapping.items() if n.get("parent") is None]
        if not roots:
            continue

        # Walk tree depth-first to extract messages in order
        messages: list[Message] = []

        def walk(node_id: str) -> None:
            node = mapping.get(node_id)
            if node is None:
                return
            msg = node.get("message")
            if msg and msg.get("content"):
                content = msg["content"]
                parts = content.get("parts", [])
                text_parts = [p for p in parts if isinstance(p, str)]
                text = "\n".join(text_parts).strip()
                role = msg.get("author", {}).get("role", "unknown")
                if text and role in ("user", "assistant"):
                    messages.append(Message(role=role, text=text))
            for child_id in children.get(node_id, []):
                walk(child_id)

        for root in roots:
            walk(root)

        conversations.append(
            Conversation(
                id=conv.get("id", "unknown"),
                title=conv.get("title") or "Untitled",
                messages=messages,
                create_time=_ts(conv.get("create_time")),
                update_time=_ts(conv.get("update_time")),
            )
        )

    return conversations


_CLAUDE_ROLE_MAP = {"human": "user", "assistant": "assistant"}


def parse_claude(path: str | Path | IO[bytes]) -> list[Conversation]:
    """Parse a Claude conversations.json export file.

    *path* can be a filesystem path (str or Path) or a readable
    file-like object containing UTF-8 JSON bytes.
    """
    data = _load_json(path)

    conversations = []
    for conv in data:
        messages: list[Message] = []
        for msg in conv.get("chat_messages", []):
            role = _CLAUDE_ROLE_MAP.get(msg.get("sender", ""))
            if role is None:
                continue
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            messages.append(Message(role=role, text=text))

        conversations.append(
            Conversation(
                id=conv.get("uuid", "unknown"),
                title=conv.get("name") or "Untitled",
                messages=messages,
                create_time=_iso(conv.get("created_at")),
                update_time=_iso(conv.get("updated_at")),
                url_prefix=CLAUDE_URL_PREFIX,
            )
        )

    return conversations
