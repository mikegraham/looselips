"""Parse LLM chat export JSON into Conversation objects."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

CHATGPT_URL_PREFIX = "https://chatgpt.com/c/"
CLAUDE_URL_PREFIX = "https://claude.ai/chat/"
_UNTITLED = "Untitled"
_UNKNOWN_ID = "unknown"


@dataclass
class Message:
    role: str
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


# Accepts str because json.load returns Any -- callers pass raw dict values
# that are usually float but could be anything in malformed exports.
def _ts(val: float | str | None) -> datetime | None:
    if val is None:
        return None
    try:
        ts = float(val)
    except ValueError:
        logger.error("unparseable timestamp: %r", val)
        return None
    try:
        return datetime.fromtimestamp(ts, tz=UTC)
    except ValueError:
        logger.error("out-of-range timestamp: %r", val)
        return None


def _iso(val: str | None) -> datetime | None:
    if val is None:
        return None
    try:
        dt = datetime.fromisoformat(val)
    except ValueError:
        logger.error("unparseable ISO date: %r", val)
        return None
    # Claude exports sometimes have naive timestamps (no tz suffix).
    # Treat them as UTC to stay consistent with _ts().
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def parse_chatgpt(data: bytes) -> list[Conversation]:
    """Parse a ChatGPT conversations.json export."""
    raw_list: list[dict[str, Any]] = json.loads(data)

    conversations = []
    for raw in raw_list:
        mapping = raw.get("mapping", {})

        # Build children lookup
        children: dict[str, list[str]] = {}
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent is not None:
                children.setdefault(parent, []).append(node_id)

        # Find roots: nodes with no parent
        roots = [nid for nid, n in mapping.items() if n.get("parent") is None]
        if not roots:
            logger.error("conversation %r has no root nodes in mapping",
                         raw.get("id", "?"))
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
                if text:
                    messages.append(Message(role=role, text=text))
            for child_id in children.get(node_id, []):
                walk(child_id)

        for root in roots:
            walk(root)

        if not messages:
            logger.debug("conversation %r produced 0 messages",
                         raw.get("id", "?"))

        conversations.append(
            Conversation(
                id=raw.get("id", _UNKNOWN_ID),
                # `or` not `get(_, default)` -- catches both missing key and null
                title=raw.get("title") or _UNTITLED,
                messages=messages,
                create_time=_ts(raw.get("create_time")),
                update_time=_ts(raw.get("update_time")),
            )
        )

    return conversations


def parse_claude(data: bytes) -> list[Conversation]:
    """Parse a Claude conversations.json export."""
    raw_list: list[dict[str, Any]] = json.loads(data)

    conversations = []
    for raw in raw_list:
        messages: list[Message] = []
        for msg in raw.get("chat_messages", []):
            role = msg.get("sender", "unknown")
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            messages.append(Message(role=role, text=text))

        if not messages:
            logger.debug("conversation %r produced 0 messages",
                         raw.get("uuid", "?"))

        conversations.append(
            Conversation(
                id=raw.get("uuid", _UNKNOWN_ID),
                title=raw.get("name") or _UNTITLED,
                messages=messages,
                create_time=_iso(raw.get("created_at")),
                update_time=_iso(raw.get("updated_at")),
                url_prefix=CLAUDE_URL_PREFIX,
            )
        )

    return conversations
