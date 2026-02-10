"""Load ChatGPT conversation exports from JSON or zip files.

Accepts either a raw conversations.json or a ChatGPT export zip
containing one.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

from .parsers import Conversation, parse_chatgpt

logger = logging.getLogger(__name__)


class InputError(Exception):
    """Raised for missing files or unreadable exports."""


def _read_json_from_zip(path: Path) -> io.BytesIO:
    """Extract conversations.json from a ChatGPT export zip file."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        logger.debug("Zip contains %d entries", len(names))
        if "conversations.json" not in names:
            raise InputError(
                f"Zip file {path} does not contain conversations.json"
            )
        data = zf.read("conversations.json")
        logger.debug("Read %d bytes from conversations.json in zip", len(data))
        return io.BytesIO(data)


def load_conversations(path: str | Path) -> list[Conversation]:
    """Load conversations from a ChatGPT export (.json or .zip)."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"File not found: {p}")

    logger.debug("Loading from %s (%s, %.1f KB)", p, p.suffix, p.stat().st_size / 1024)

    if p.suffix == ".zip":
        return parse_chatgpt(_read_json_from_zip(p))
    return parse_chatgpt(p)
