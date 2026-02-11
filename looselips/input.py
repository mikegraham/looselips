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


def _read_from_zip(path: Path) -> tuple[bytes, _Format]:
    """Extract conversations.json from an export zip and detect format."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        logger.debug("Zip contains %d entries", len(names))
        if "conversations.json" not in names:
            raise InputError(
                f"Zip file {path} does not contain conversations.json"
            )
        # Claude exports include users.json alongside conversations.json.
        # Annotation needed: mypy infers the ternary as str, not Literal.
        fmt: _Format = "claude" if "users.json" in names else "chatgpt"
        data = zf.read("conversations.json")
        logger.debug("Read %d bytes from conversations.json in zip", len(data))
        return data, fmt


def load_conversations(path: str | Path) -> list[Conversation]:
    """Load conversations from a ChatGPT or Claude export (.json or .zip)."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"File not found: {p}")

    logger.debug("Loading from %s (%s, %.1f KB)", p, p.suffix, p.stat().st_size / 1024)

    if p.suffix == ".zip":
        data, fmt = _read_from_zip(p)
        logger.info("Detected format: %s", fmt)
        parser = parse_claude if fmt == "claude" else parse_chatgpt
        convs = parser(data)
    else:
        # Bare JSON -- read once, detect format
        raw = p.read_bytes()
        fmt = _detect_format(json.loads(raw))
        logger.info("Detected format: %s", fmt)
        parser = parse_claude if fmt == "claude" else parse_chatgpt
        convs = parser(raw)

    if not convs:
        logger.error("export contained 0 conversations")

    return convs
