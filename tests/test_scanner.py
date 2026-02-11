"""Tests for looselips.scanner."""

import re
from unittest.mock import patch

import pytest

from looselips.matchers import LLMParseError, Match
from looselips.parsers import Conversation, Message
from looselips.scanner import _chunk_conversation, scan


def _conv(
    messages_text: list[tuple[str, str]], conv_id: str = "c1", title: str = "Test"
) -> Conversation:
    return Conversation(
        id=conv_id,
        title=title,
        messages=[Message(role=r, text=t) for r, t in messages_text],
    )


SIMPLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Email", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
]


def test_scan_flags_matches_and_skips_clean() -> None:
    convs = [
        _conv([("user", "clean message")], conv_id="c1"),
        _conv([("user", "has test@x.com")], conv_id="c2"),
    ]
    result = scan(convs, patterns=SIMPLE_PATTERNS)
    assert result.total == 2
    assert len(result.flagged) == 1
    assert result.flagged[0].matches[0].matched_text == "test@x.com"


def test_chunk_conversation_short() -> None:
    """A short conversation fits in one chunk with formatted messages."""
    conv = _conv([("user", "hello"), ("assistant", "hi"), ("user", "bye")])
    chunks = _chunk_conversation(conv, max_chars=1000)
    assert len(chunks) == 1
    assert "[USER]: hello" in chunks[0]
    assert "[ASSISTANT]: hi" in chunks[0]
    assert "[USER]: bye" in chunks[0]


def test_chunk_conversation_splits_at_message_boundary() -> None:
    """Long conversations split at message boundaries, not mid-message."""
    conv = _conv([("user", "a" * 100), ("user", "b" * 100), ("user", "c" * 100)])
    chunks = _chunk_conversation(conv, max_chars=150)
    assert len(chunks) == 3
    assert "a" * 100 in chunks[0]
    assert "b" * 100 in chunks[1]
    assert "c" * 100 in chunks[2]


def test_scan_with_llm_model_but_no_matchers_skips_llm() -> None:
    """llm_model alone does not trigger scanning -- explicit matchers required."""
    with patch("looselips.scanner.llm_scan") as mock:
        result = scan(
            [_conv([("user", "My name is Alice")])],
            patterns=[],
            llm_model="test-model",
        )
        mock.assert_not_called()
        assert len(result.flagged) == 0


def test_scan_with_explicit_llm_matchers() -> None:
    mock_matches = [
        Match(category="Custom", matched_text="found", context="found", source="llm")
    ]
    with patch("looselips.scanner.llm_scan", return_value=mock_matches) as mock:
        matchers = [("Custom", "Find custom stuff", "custom-model")]
        result = scan(
            [_conv([("user", "test")])],
            patterns=[],
            llm_model="fallback",
            llm_matchers=matchers,
        )
        assert mock.call_args[0][2] == "custom-model"
        assert len(result.flagged) == 1


def test_scan_llm_matcher_no_model_raises() -> None:
    with pytest.raises(ValueError, match="has no model"):
        scan(
            [_conv([("user", "test")])],
            patterns=[],
            llm_model=None,
            llm_matchers=[("test", "prompt", None)],
        )


def test_scan_llm_parse_error_recorded() -> None:
    """LLMParseError is recorded, not swallowed or raised."""
    with patch("looselips.scanner.llm_scan", side_effect=LLMParseError("bad")):
        result = scan(
            [_conv([("user", "test")])],
            patterns=[],
            llm_model="test-model",
            llm_matchers=[("pii", "find pii", None)],
        )
    assert len(result.errors) == 1
    assert result.errors[0].matcher == "pii"
    assert "bad" in result.errors[0].error
    assert len(result.flagged) == 0


