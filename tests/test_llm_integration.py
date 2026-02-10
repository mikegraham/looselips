"""Integration tests that call a real LLM.

Run with: pytest tests/test_llm_integration.py --model ollama/qwen3:0.6b
Skipped automatically when --model is not provided.
"""

from __future__ import annotations

import pytest

from looselips.matchers import Match, llm_scan

_SYSTEM_PROMPT = """You are a privacy auditor. Identify personal facts about the user.
Look for: names, contact info, employment, financial info, medical info, credentials."""

_OBVIOUS_PII = (
    "[USER]: My name is John Smith and I live at 123 Oak Street. "
    "My SSN is 123-45-6789 and I work at Sun Microsystems.\n\n"
    "[ASSISTANT]: I can help you with that."
)

_NO_PII = (
    "[USER]: What's the weather like today?\n\n[ASSISTANT]: It's sunny and 72 degrees."
)


@pytest.fixture(scope="module")
def pii_matches(model: str) -> list[Match]:
    """Single LLM call shared across PII tests to avoid redundant inference."""
    return llm_scan(
        title="Personal Info Chat",
        messages_text=_OBVIOUS_PII,
        model=model,
        name="pii",
        system_prompt=_SYSTEM_PROMPT,
    )


def test_llm_detects_obvious_pii(pii_matches: list[Match]) -> None:
    assert len(pii_matches) == 1
    remarks = pii_matches[0].matched_text.lower()
    # Should mention at least name or SSN in remarks
    assert "john" in remarks or "smith" in remarks or "123" in remarks


def test_llm_returns_empty_for_no_pii(model: str) -> None:
    matches = llm_scan(
        title="Weather Chat",
        messages_text=_NO_PII,
        model=model,
        name="pii",
        system_prompt=_SYSTEM_PROMPT,
    )
    assert len(matches) == 0


def test_llm_match_fields_are_populated(pii_matches: list[Match]) -> None:
    assert len(pii_matches) == 1
    m = pii_matches[0]
    assert m.source == "llm"
    assert m.category == "pii"
    assert m.matched_text
