"""Load LLM chat exports from JSON or zip files.

Supports ChatGPT and Claude export formats. Format is auto-detected
from file contents.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path

from .parsers import Conversation, parse_chatgpt, parse_claude

logger = logging.getLogger(__name__)


class InputError(Exception):
    """Raised for missing files or unreadable exports."""


def _detect_format(data: Sequence[object]) -> str:
    """Return 'claude' or 'chatgpt' based on the first conversation object."""
    if not data:
        return "chatgpt"
    first = data[0]
    if isinstance(first, dict) and "chat_messages" in first:
        return "claude"
    return "chatgpt"


def _read_from_zip(path: Path) -> tuple[io.BytesIO, str]:
    """Extract conversations.json from an export zip and detect format.

    Returns (BytesIO of conversations.json, format string).
    """
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        logger.debug("Zip contains %d entries", len(names))
        if "conversations.json" not in names:
            raise InputError(
                f"Zip file {path} does not contain conversations.json"
            )
        # Claude exports include users.json alongside conversations.json
        if "users.json" in names:
            fmt = "claude"
        else:
            fmt = "chatgpt"
        data = zf.read("conversations.json")
        logger.debug("Read %d bytes from conversations.json in zip", len(data))
        return io.BytesIO(data), fmt


def load_conversations(path: str | Path) -> list[Conversation]:
    """Load conversations from a ChatGPT or Claude export (.json or .zip)."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"File not found: {p}")

    logger.debug("Loading from %s (%s, %.1f KB)", p, p.suffix, p.stat().st_size / 1024)

    if p.suffix == ".zip":
        buf, fmt = _read_from_zip(p)
        if fmt == "claude":
            return parse_claude(buf)
        return parse_chatgpt(buf)

    # Bare JSON -- read once, detect format, pass as BytesIO
    raw = p.read_bytes()
    fmt = _detect_format(json.loads(raw))
    buf = io.BytesIO(raw)
    if fmt == "claude":
        return parse_claude(buf)
    return parse_chatgpt(buf)
