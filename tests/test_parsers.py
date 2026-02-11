"""Tests for looselips.parsers."""

import json
from datetime import UTC, datetime
from typing import Any

from looselips.parsers import (
    Conversation,
    Message,
    _iso,
    _ts,
    parse_chatgpt,
    parse_claude,
)


def _to_bytes(data: list[dict[str, Any]]) -> bytes:
    return json.dumps(data).encode()


def test_ts_none() -> None:
    assert _ts(None) is None


def test_ts_valid() -> None:
    result = _ts(0)
    assert result == datetime(1970, 1, 1, tzinfo=UTC)


def test_ts_invalid() -> None:
    assert _ts("not-a-number") is None


def test_ts_nan() -> None:
    # NaN is a valid float but datetime.fromtimestamp rejects it.
    assert _ts(float("nan")) is None



def test_conversation_url() -> None:
    c = Conversation(id="abc", title="t", messages=[])
    assert c.url == "https://chatgpt.com/c/abc"



def test_parse_chatgpt_basic() -> None:
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
    convs = parse_chatgpt(_to_bytes(data))

    assert len(convs) == 1
    c = convs[0]
    assert c.id == "conv-1"
    assert c.title == "Test Chat"
    assert c.create_time is not None
    assert len(c.messages) == 2
    assert c.messages[0] == Message(role="user", text="Hello")
    assert c.messages[1] == Message(role="assistant", text="Hi there!")


def test_parse_chatgpt_includes_all_roles() -> None:
    """All roles are included to avoid false negatives."""
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
                "tool": {
                    "parent": "sys",
                    "message": {
                        "author": {"role": "tool"},
                        "content": {"parts": ["Tool output"]},
                    },
                },
                "usr": {
                    "parent": "tool",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hi"]},
                    },
                },
            },
        }
    ]
    convs = parse_chatgpt(_to_bytes(data))

    assert len(convs) == 1
    assert len(convs[0].messages) == 3
    assert convs[0].messages[0].role == "system"
    assert convs[0].messages[1].role == "tool"
    assert convs[0].messages[2].role == "user"


def test_parse_chatgpt_empty_mapping() -> None:
    data: list[dict[str, Any]] = [{"id": "conv-3", "title": "Empty", "mapping": {}}]
    convs = parse_chatgpt(_to_bytes(data))
    assert len(convs) == 0


def test_parse_chatgpt_multipart() -> None:
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
    convs = parse_chatgpt(_to_bytes(data))

    assert len(convs) == 1
    assert convs[0].messages[0].text == "Part one\nPart two"


def test_parse_chatgpt_untitled() -> None:
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
    convs = parse_chatgpt(_to_bytes(data))
    assert convs[0].title == "Untitled"


def test_parse_chatgpt_skips_non_string_parts() -> None:
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
    convs = parse_chatgpt(_to_bytes(data))
    assert convs[0].messages[0].text == "text part"


def test_parse_chatgpt_dangling_child_ref() -> None:
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
    convs = parse_chatgpt(_to_bytes(data))
    assert len(convs) == 1
    assert convs[0].messages[0].text == "Hello"


# -- _iso tests --


def test_iso_none() -> None:
    assert _iso(None) is None


def test_iso_valid() -> None:
    result = _iso("2025-01-15T12:30:00+00:00")
    assert result == datetime(2025, 1, 15, 12, 30, tzinfo=UTC)


def test_iso_naive_gets_utc() -> None:
    """A naive ISO string (no timezone) is normalized to UTC."""
    result = _iso("2025-01-15T12:30:00")
    assert result is not None
    assert result.tzinfo == UTC


def test_iso_invalid() -> None:
    assert _iso("not-a-date") is None


# -- parse_claude tests --


def _claude_export(
    messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if messages is None:
        messages = [
            {"sender": "human", "text": "Hello", "content": []},
            {"sender": "assistant", "text": "Hi there!", "content": []},
        ]
    return [
        {
            "uuid": "claude-conv-1",
            "name": "Test Claude Chat",
            "created_at": "2025-01-15T12:00:00+00:00",
            "updated_at": "2025-01-15T12:05:00+00:00",
            "chat_messages": messages,
        }
    ]


def test_parse_claude_basic() -> None:
    convs = parse_claude(_to_bytes(_claude_export()))

    assert len(convs) == 1
    c = convs[0]
    assert c.id == "claude-conv-1"
    assert c.title == "Test Claude Chat"
    assert c.url == "https://claude.ai/chat/claude-conv-1"
    assert c.create_time == datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    assert len(c.messages) == 2
    assert c.messages[0] == Message(role="human", text="Hello")
    assert c.messages[1] == Message(role="assistant", text="Hi there!")


def test_parse_claude_skips_empty_text() -> None:
    data = _claude_export(
        messages=[
            {"sender": "human", "text": "Hello", "content": []},
            {"sender": "assistant", "text": "", "content": []},
            {"sender": "assistant", "text": "Real reply", "content": []},
        ]
    )
    convs = parse_claude(_to_bytes(data))

    assert len(convs[0].messages) == 2
    assert convs[0].messages[1].text == "Real reply"


def test_parse_claude_includes_all_senders() -> None:
    """All senders are included to avoid false negatives."""
    data = _claude_export(
        messages=[
            {"sender": "human", "text": "Hello", "content": []},
            {"sender": "system", "text": "System msg", "content": []},
            {"sender": "tool_use", "text": "Tool output", "content": []},
        ]
    )
    convs = parse_claude(_to_bytes(data))

    assert len(convs[0].messages) == 3
    assert convs[0].messages[0].role == "human"
    assert convs[0].messages[1].role == "system"
    assert convs[0].messages[2].role == "tool_use"


def test_parse_claude_untitled() -> None:
    data = _claude_export()
    data[0]["name"] = None
    convs = parse_claude(_to_bytes(data))
    assert convs[0].title == "Untitled"


def test_parse_claude_null_text() -> None:
    """Messages with null text are skipped."""
    data = _claude_export(
        messages=[
            {"sender": "human", "text": "Hello", "content": []},
            {"sender": "assistant", "text": None, "content": []},
        ]
    )
    convs = parse_claude(_to_bytes(data))
    assert len(convs[0].messages) == 1


def test_parse_claude_no_chat_messages() -> None:
    """Conversation with no chat_messages key produces zero messages."""
    data = [{"uuid": "empty", "name": "Empty", "created_at": None, "updated_at": None}]
    convs = parse_claude(_to_bytes(data))
    assert len(convs) == 1
    assert convs[0].messages == []
