"""Tests for looselips.report."""

from pathlib import Path

import pytest

from looselips.matchers import Match
from looselips.parsers import Conversation, Message
from looselips.report import generate_html, write_report
from looselips.scanner import ConversationResult, ScanResult


@pytest.fixture
def empty_result() -> ScanResult:
    return ScanResult(total=1, flagged=[])


@pytest.fixture
def regex_result() -> ScanResult:
    matches = [
        Match(
            category="Email",
            matched_text="a@b.com",
            context="email a@b.com here",
            source="regex",
        )
    ]
    conv = Conversation(id="c1", title="Test Chat", messages=[Message("user", "hi")])
    return ScanResult(
        total=2, flagged=[ConversationResult(conversation=conv, matches=matches)]
    )


@pytest.fixture
def llm_result() -> ScanResult:
    matches = [
        Match(
            category="employment",
            matched_text="User works at Acme Corp making $200k",
            context="User works at Acme Corp making $200k",
            source="llm",
        )
    ]
    conv = Conversation(id="c1", title="Chat", messages=[Message("user", "hi")])
    return ScanResult(
        total=2, flagged=[ConversationResult(conversation=conv, matches=matches)]
    )


def test_empty_report_shows_no_findings(empty_result: ScanResult) -> None:
    html = generate_html(empty_result)
    assert "No findings" in html


def test_report_contains_html_structure(regex_result: ScanResult) -> None:
    html = generate_html(regex_result, input_name="conversations")
    assert "<html" in html
    assert "</html>" in html
    assert "looselips: conversations" in html


def test_report_shows_conversation_title(regex_result: ScanResult) -> None:
    html = generate_html(regex_result)
    assert "Test Chat" in html


def test_report_shows_match_category_and_text(regex_result: ScanResult) -> None:
    html = generate_html(regex_result)
    assert "Email" in html
    assert "a@b.com" in html


def test_report_escapes_xss() -> None:
    matches = [
        Match(
            category="test",
            matched_text="<img src=x onerror=alert(1)>",
            context="<img src=x onerror=alert(1)>",
            source="regex",
        )
    ]
    conv = Conversation(id="c1", title="<b>evil</b>", messages=[Message("user", "hi")])
    result = ScanResult(
        total=2, flagged=[ConversationResult(conversation=conv, matches=matches)]
    )
    html = generate_html(result)

    assert "&lt;b&gt;evil&lt;/b&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<b>evil</b>" not in html
    assert "<img src=x" not in html


def test_report_llm_match_shows_remarks(llm_result: ScanResult) -> None:
    html = generate_html(llm_result)
    assert "Acme Corp" in html
    assert "[llm]" in html


def test_report_stats(regex_result: ScanResult) -> None:
    result = ScanResult(total=53, flagged=regex_result.flagged)
    html = generate_html(result)
    assert ">53<" in html  # total
    assert ">1<" in html  # flagged
    assert ">52<" in html  # clean


def test_write_report_creates_file(tmp_path: Path, empty_result: ScanResult) -> None:
    out = tmp_path / "report.html"
    write_report(empty_result, str(out))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
