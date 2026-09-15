"""Load LLM chat exports from JSON or zip files.

Supports ChatGPT and Claude export formats. Format is auto-detected
from file contents.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from .parsers import Conversation, parse_chatgpt, parse_claude

logger = logging.getLogger(__name__)


class InputError(Exception):
    """Raised for missing files or unreadable exports."""


_Format = Literal["chatgpt", "claude"]


# Sequence[object] not list[dict] -- list is invariant, so list[dict[str, Any]]
# from json.loads wouldn't be assignable to list[object].
def _detect_format(data: Sequence[object]) -> _Format:
    """Return 'claude' or 'chatgpt' based on the first conversation object."""
    if not data:
        return "chatgpt"
    first = data[0]
    if isinstance(first, dict) and "chat_messages" in first:
        return "claude"
    return "chatgpt"


def _read_from_zip(path: Path) -> bytes:
    """Extract conversations.json from an export zip."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        logger.debug("Zip contains %d entries", len(names))
        if "conversations.json" not in names:
            raise InputError(
                f"Zip file {path} does not contain conversations.json"
            )
        data = zf.read("conversations.json")
        logger.debug("Read %d bytes from conversations.json in zip", len(data))
        return data


def load_conversations(path: str | Path) -> list[Conversation]:
    """Load conversations from a ChatGPT or Claude export (.json or .zip)."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"File not found: {p}")

    logger.debug("Loading from %s (%s, %.1f KB)", p, p.suffix, p.stat().st_size / 1024)

    if p.suffix == ".zip":
        raw = _read_from_zip(p)
    else:
        raw = p.read_bytes()

    # Detect the format from the content, never from sibling files.  Claude
    # exports used to ship users.json next to conversations.json, but the
    # newer per-category export (a manifest pointing at conversations-000.zip
    # and friends) does not, and keying on that file misread them as
    # ChatGPT.  The JSON is parsed a second time by the parser; that costs a
    # few seconds on a very large export and keeps the parsers self-contained.
    fmt = _detect_format(json.loads(raw))
    logger.info("Detected format: %s", fmt)
    parser = parse_claude if fmt == "claude" else parse_chatgpt
    convs = parser(raw)

    if not convs:
        logger.error("export contained 0 conversations")

    return convs
