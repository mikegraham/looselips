"""Tests for looselips.parsers."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from looselips.parsers import Conversation, Message, _ts, parse_chatgpt


def _write_export(tmp_path: Path, data: list[dict[str, Any]]) -> str:
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)



def test_ts_none() -> None:
    assert _ts(None) is None


def test_ts_valid() -> None:
    result = _ts(0)
    assert result == datetime(1970, 1, 1, tzinfo=UTC)


def test_ts_float() -> None:
    result = _ts(1700000000.0)
    assert isinstance(result, datetime)
    assert result.tzinfo == UTC


def test_ts_invalid() -> None:
    assert _ts("not-a-number") is None


def test_ts_nan() -> None:
    # NaN is a valid float but datetime.fromtimestamp rejects it.
    assert _ts(float("nan")) is None



def test_conversation_url() -> None:
    c = Conversation(id="abc", title="t", messages=[])
    assert c.url == "https://chatgpt.com/c/abc"



def test_parse_chatgpt_basic(tmp_path: Path) -> None:
    data: list[dict[str, Any]] = [
        {
            "id": "conv-1",
            "title": "Test Chat",
            "create_time": 1700000000,
            "update_time": 1700001000,
            "mapping": {
                "root": {
                    "parent": None,
                    "message": None,
                },
                "msg1": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                    },
                },
                "msg2": {
                    "parent": "msg1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hi there!"]},
                    },
                },
            },
        }
    ]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)

    assert len(convs) == 1
    c = convs[0]
    assert c.id == "conv-1"
    assert c.title == "Test Chat"
    assert c.create_time is not None
    assert len(c.messages) == 2
    assert c.messages[0] == Message(role="user", text="Hello")
    assert c.messages[1] == Message(role="assistant", text="Hi there!")


def test_parse_chatgpt_skips_system_messages(tmp_path: Path) -> None:
    data: list[dict[str, Any]] = [
        {
            "id": "conv-2",
            "title": "With System",
            "mapping": {
                "root": {"parent": None, "message": None},
                "sys": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["You are helpful"]},
                    },
                },
                "usr": {
                    "parent": "sys",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hi"]},
                    },
                },
            },
        }
    ]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)

    assert len(convs) == 1
    assert len(convs[0].messages) == 1
    assert convs[0].messages[0].role == "user"


def test_parse_chatgpt_empty_mapping(tmp_path: Path) -> None:
    data: list[dict[str, Any]] = [{"id": "conv-3", "title": "Empty", "mapping": {}}]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)
    assert len(convs) == 0


def test_parse_chatgpt_multipart(tmp_path: Path) -> None:
    data: list[dict[str, Any]] = [
        {
            "id": "conv-4",
            "title": "Multi",
            "mapping": {
                "root": {"parent": None, "message": None},
                "msg": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Part one", "Part two"]},
                    },
                },
            },
        }
    ]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)

    assert len(convs) == 1
    assert convs[0].messages[0].text == "Part one\nPart two"


def test_parse_chatgpt_untitled(tmp_path: Path) -> None:
    data: list[dict[str, Any]] = [
        {
            "id": "conv-5",
            "title": None,
            "mapping": {
                "root": {"parent": None, "message": None},
                "msg": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hi"]},
                    },
                },
            },
        }
    ]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)
    assert convs[0].title == "Untitled"


def test_parse_chatgpt_skips_non_string_parts(tmp_path: Path) -> None:
    """Non-string parts (e.g. image metadata dicts) are filtered out."""
    data: list[dict[str, Any]] = [
        {
            "id": "conv-6",
            "title": "Images",
            "mapping": {
                "root": {"parent": None, "message": None},
                "msg": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["text part", {"image": "data"}]},
                    },
                },
            },
        }
    ]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)
    assert convs[0].messages[0].text == "text part"


def test_parse_chatgpt_dangling_child_ref(tmp_path: Path) -> None:
    """A child reference pointing to a missing node is silently skipped."""
    data: list[dict[str, Any]] = [
        {
            "id": "conv-dangle",
            "title": "Dangling",
            "mapping": {
                "root": {"parent": None, "message": None},
                "msg": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                    },
                },
                # Points to a node that doesn't exist
                "ghost": {"parent": "msg"},
            },
        }
    ]
    path = _write_export(tmp_path, data)
    convs = parse_chatgpt(path)
    assert len(convs) == 1
    assert convs[0].messages[0].text == "Hello"


def test_parse_chatgpt_from_fileobj() -> None:
    """parse_chatgpt accepts a file-like object (BytesIO)."""
    data: list[dict[str, Any]] = [
        {
            "id": "conv-fo",
            "title": "File Object",
            "mapping": {
                "root": {"parent": None, "message": None},
                "msg": {
                    "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hey"]},
                    },
                },
            },
        }
    ]
    buf = io.BytesIO(json.dumps(data).encode("utf-8"))
    convs = parse_chatgpt(buf)
    assert len(convs) == 1
    assert convs[0].id == "conv-fo"
    assert convs[0].messages[0].text == "Hey"
