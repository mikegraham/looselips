"""Tests for looselips.scanner."""

import re
from unittest.mock import patch

import pytest

from looselips.matchers import LLMParseError, Match
from looselips.parsers import Conversation, Message
from looselips.scanner import ScanResult, _chunk_conversation, scan


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


def test_chunk_conversation_splits_oversized_message() -> None:
    """A message longer than max_chars is split, never left oversized.

    Regression test: an oversized chunk used to be sent as-is, and the
    backend silently truncated it (a false negative for whatever was in the
    tail of the message).
    """
    text = "".join(f"line {i} secret{i}\n" for i in range(400))
    conv = _conv([("user", text)])
    chunks = _chunk_conversation(conv, max_chars=500)

    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Every piece keeps a role tag and says which part it is.
    assert chunks[0].startswith(f"[USER] (part 1/{len(chunks)}): ")
    assert all(c.startswith("[USER] (part ") for c in chunks)
    # Nothing is dropped: the pieces reassemble into the original message.
    rebuilt = "".join(
        re.sub(r"^\[USER\] \(part \d+/\d+\): ", "", c) for c in chunks
    )
    assert rebuilt == text
    assert "secret399" in rebuilt


def test_chunk_conversation_splits_at_newline_when_possible() -> None:
    """Pieces prefer to end on a line boundary."""
    text = "".join(f"line {i}\n" for i in range(300))
    chunks = _chunk_conversation(_conv([("user", text)]), max_chars=400)
    assert len(chunks) > 1
    # All but the last piece end where a line ended.
    assert all(c.endswith("\n") for c in chunks[:-1])


def test_chunk_conversation_hard_splits_without_newlines() -> None:
    """A single long line has no newline to split on, so it is cut hard."""
    text = "x" * 5000
    chunks = _chunk_conversation(_conv([("user", text)]), max_chars=600)
    assert len(chunks) >= 9
    assert all(len(c) <= 600 for c in chunks)
    rebuilt = "".join(
        re.sub(r"^\[USER\] \(part \d+/\d+\): ", "", c) for c in chunks
    )
    assert rebuilt == text


def test_chunk_conversation_mixes_split_pieces_with_short_messages() -> None:
    """Split pieces flow through the normal packing logic."""
    conv = _conv([
        ("user", "short one"),
        ("assistant", "y" * 3000),
        ("user", "short two"),
    ])
    chunks = _chunk_conversation(conv, max_chars=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "[USER]: short one" in chunks[0]
    assert "[ASSISTANT] (part 1/" in "\n".join(chunks)
    assert "[USER]: short two" in chunks[-1]


def test_chunk_conversation_at_limit_not_split() -> None:
    """A message that exactly fits keeps its plain role tag."""
    text = "z" * (500 - len("[USER]: "))
    chunks = _chunk_conversation(_conv([("user", text)]), max_chars=500)
    assert len(chunks) == 1
    assert chunks[0] == f"[USER]: {text}"
    assert len(chunks[0]) == 500



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
    from looselips.matchers import LLMResult

    mock_result = LLMResult(
        found=True,
        reasoning="found",
        matches=[Match(
            category="Custom", matched_text="found",
            context="found", source="llm",
        )],
        verdict_json='{"reasoning":"found","found":true}',
    )
    with patch("looselips.scanner.llm_scan", return_value=mock_result) as mock:
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


def test_scan_on_progress_reports_after_each_conversation() -> None:
    convs = [
        _conv([("user", f"hello test{i}@x.com")], conv_id=f"c{i}", title=f"t{i}")
        for i in range(3)
    ]
    seen: list[tuple[int, int, int]] = []

    def on_progress(partial: ScanResult, scanned: int) -> None:
        seen.append((scanned, partial.total, len(partial.flagged)))

    scan(convs, patterns=SIMPLE_PATTERNS, on_progress=on_progress)
    assert seen == [(1, 3, 1), (2, 3, 2), (3, 3, 3)]
