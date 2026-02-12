"""Tests for looselips.matchers."""

import json
import re
from unittest.mock import MagicMock, patch

import litellm.exceptions
import pytest

from looselips.matchers import (
    LLMParseError,
    LLMVerdict,
    _snippet,
    llm_scan,
    regex_scan,
)

SIMPLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Email", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
]


def test_snippet_middle() -> None:
    text = "a" * 100 + "SECRET" + "b" * 100
    result = _snippet(text, 100, 106, margin=10)
    assert "SECRET" in result
    assert result.startswith("...")
    assert result.endswith("...")


def test_snippet_at_start() -> None:
    text = "SECRET" + "x" * 200
    result = _snippet(text, 0, 6, margin=10)
    assert result.startswith("SECRET")
    assert result.endswith("...")


def test_snippet_at_end() -> None:
    text = "x" * 200 + "SECRET"
    result = _snippet(text, 200, 206, margin=10)
    assert result.endswith("SECRET")
    assert result.startswith("...")


def test_snippet_short_text() -> None:
    text = "hello"
    result = _snippet(text, 0, 5, margin=80)
    assert result == "hello"


def test_regex_scan_finds_email() -> None:
    matches = regex_scan("contact me at foo@bar.com please", SIMPLE_PATTERNS)
    assert len(matches) == 1
    assert matches[0].category == "Email"
    assert matches[0].matched_text == "foo@bar.com"
    assert matches[0].source == "regex"


def test_regex_scan_no_matches() -> None:
    matches = regex_scan("nothing here", SIMPLE_PATTERNS)
    assert matches == []



@patch("looselips.matchers.instructor")
@patch("looselips.matchers.completion")
def test_llm_scan_flagged(
    mock_completion: MagicMock, mock_instructor: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_instructor.from_litellm.return_value = mock_client
    mock_client.chat.completions.create.return_value = LLMVerdict(
        found=True, reasoning="User is Bob, works at Acme"
    )

    result = llm_scan("Test Chat", "Hello I am Bob", "ollama/llama3", name="pii")
    assert result.found is True
    assert result.reasoning == "User is Bob, works at Acme"
    assert len(result.matches) == 1
    assert result.matches[0].category == "pii"
    assert result.matches[0].matched_text == "User is Bob, works at Acme"
    assert result.matches[0].source == "llm"
    assert json.loads(result.verdict_json)["found"] is True


@patch("looselips.matchers.instructor")
@patch("looselips.matchers.completion")
def test_llm_scan_not_flagged(
    mock_completion: MagicMock, mock_instructor: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_instructor.from_litellm.return_value = mock_client
    mock_client.chat.completions.create.return_value = LLMVerdict(
        found=False, reasoning=""
    )

    result = llm_scan("Chat", "nothing personal", "model")
    assert result.found is False
    assert result.matches == []
    assert json.loads(result.verdict_json)["found"] is False


@patch("looselips.matchers.instructor")
@patch("looselips.matchers.completion")
def test_llm_scan_wraps_errors(
    mock_completion: MagicMock, mock_instructor: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_instructor.from_litellm.return_value = mock_client
    mock_client.chat.completions.create.side_effect = (
        litellm.exceptions.APIConnectionError(
            message="connection failed", model="test", llm_provider="ollama"
        )
    )

    with pytest.raises(LLMParseError, match="connection failed"):
        llm_scan("Title", "text", "model")
