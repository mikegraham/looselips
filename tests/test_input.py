"""Tests for looselips.input."""

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from looselips.input import InputError, load_conversations


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


def test_load_zip_missing_conversations_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("other.txt", "hello")
    with pytest.raises(InputError, match="does not contain conversations.json"):
        load_conversations(p)
