"""Tests for looselips.input."""

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from looselips.input import InputError, _detect_format, load_conversations


def _write_chatgpt(tmp_path: Path, data: list[dict[str, Any]]) -> Path:
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_chatgpt_zip(tmp_path: Path, data: list[dict[str, Any]]) -> Path:
    p = tmp_path / "export.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("conversations.json", json.dumps(data))
    return p


_MINIMAL_EXPORT: list[dict[str, Any]] = [
    {
        "id": "conv-1",
        "title": "Test",
        "mapping": {
            "root": {"parent": None, "message": None},
            "msg": {
                "parent": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Hello"]},
                },
            },
        },
    }
]


def test_load_json(tmp_path: Path) -> None:
    p = _write_chatgpt(tmp_path, _MINIMAL_EXPORT)
    convs = load_conversations(p)
    assert len(convs) == 1
    assert convs[0].id == "conv-1"


def test_load_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="File not found"):
        load_conversations(tmp_path / "nope.json")


def test_load_zip(tmp_path: Path) -> None:
    p = _write_chatgpt_zip(tmp_path, _MINIMAL_EXPORT)
    convs = load_conversations(p)
    assert len(convs) == 1
    assert convs[0].id == "conv-1"
    assert "chatgpt.com" in convs[0].url


def test_load_zip_missing_conversations_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("other.txt", "hello")
    with pytest.raises(InputError, match="does not contain conversations.json"):
        load_conversations(p)


# -- Claude format fixtures --

_MINIMAL_CLAUDE_EXPORT: list[dict[str, Any]] = [
    {
        "uuid": "claude-1",
        "name": "Claude Chat",
        "created_at": "2025-01-15T12:00:00+00:00",
        "updated_at": "2025-01-15T12:05:00+00:00",
        "chat_messages": [
            {"sender": "human", "text": "Hello", "content": []},
            {"sender": "assistant", "text": "Hi!", "content": []},
        ],
    }
]


def _write_claude_json(tmp_path: Path, data: list[dict[str, Any]]) -> Path:
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_claude_zip(tmp_path: Path, data: list[dict[str, Any]]) -> Path:
    p = tmp_path / "claude_export.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("conversations.json", json.dumps(data))
        zf.writestr("users.json", "[]")
        zf.writestr("projects.json", "[]")
        zf.writestr("memories.json", "[]")
    return p


# -- Auto-detection tests --


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_MINIMAL_CLAUDE_EXPORT, "claude"),
        (_MINIMAL_EXPORT, "chatgpt"),
        ([], "chatgpt"),
    ],
    ids=["claude", "chatgpt", "empty-defaults-chatgpt"],
)
def test_detect_format(data: list[dict[str, Any]], expected: str) -> None:
    assert _detect_format(data) == expected


def test_load_claude_json(tmp_path: Path) -> None:
    p = _write_claude_json(tmp_path, _MINIMAL_CLAUDE_EXPORT)
    convs = load_conversations(p)
    assert len(convs) == 1
    assert convs[0].id == "claude-1"
    assert convs[0].url == "https://claude.ai/chat/claude-1"


def test_load_claude_zip(tmp_path: Path) -> None:
    p = _write_claude_zip(tmp_path, _MINIMAL_CLAUDE_EXPORT)
    convs = load_conversations(p)
    assert len(convs) == 1
    assert convs[0].id == "claude-1"


