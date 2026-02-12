"""Tests for the bench package (report, render, DB ops, testcase loading)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from looselips.bench import load_testcases
from looselips.bench.db import init_db, load_cached, save_result
from looselips.bench.render import render_report
from looselips.bench.report import (
    BenchReport,
    CellResult,
    ReportRow,
    _natural_sort_key,
    build_report,
)
from looselips.bench.report import (
    testcase_to_conversation as tc_to_conv,
)
from looselips.parsers import Message

# -- Fixtures ----------------------------------------------------------------


def _make_report(
    models: list[str] | None = None,
    matchers: list[str] | None = None,
    rows: list[ReportRow] | None = None,
) -> BenchReport:
    """Build a BenchReport with sensible defaults for testing."""
    if models is None:
        models = ["model-a", "model-b"]
    if matchers is None:
        matchers = ["pii", "secrets"]
    if rows is None:
        rows = []
    return BenchReport(models=models, matchers=matchers, rows=rows)


def _row(
    title: str,
    expectations: dict[str, bool],
    cells: dict[str, dict[str, CellResult]] | None = None,
) -> ReportRow:
    return ReportRow(
        title=title,
        messages=[Message(role="user", text="hi")],
        expectations=expectations,
        cells=cells or {},
    )


def _cell(found: bool, elapsed: float = 1.0) -> CellResult:
    return CellResult(found=found, reasoning="test", elapsed=elapsed)


# -- CellResult --------------------------------------------------------------


def test_cell_result_defaults() -> None:
    c = CellResult(found=True, reasoning="r", elapsed=0.5)
    assert c.response_json is None


# -- BenchReport.model_score -------------------------------------------------


def test_model_score_all_correct() -> None:
    r = _make_report(
        rows=[_row("t1", {"pii": True}, {"model-a": {"pii": _cell(True)}})]
    )
    assert r.model_score("model-a") == (1, 1)


def test_model_score_wrong() -> None:
    r = _make_report(
        rows=[_row("t1", {"pii": True}, {"model-a": {"pii": _cell(False)}})]
    )
    assert r.model_score("model-a") == (0, 1)


def test_model_score_missing_cell_not_counted() -> None:
    r = _make_report(rows=[_row("t1", {"pii": True}, {})])
    assert r.model_score("model-a") == (0, 0)


def test_model_score_multiple_rows_and_matchers() -> None:
    r = _make_report(
        rows=[
            _row("t1", {"pii": True, "secrets": False}, {
                "model-a": {"pii": _cell(True), "secrets": _cell(False)},
            }),
            _row("t2", {"pii": False}, {
                "model-a": {"pii": _cell(True)},  # wrong: expected False
            }),
        ],
    )
    assert r.model_score("model-a") == (2, 3)


# -- BenchReport.crosstab ---------------------------------------------------


def test_crosstab_tp() -> None:
    r = _make_report(
        rows=[_row("t1", {"pii": True}, {"m": {"pii": _cell(True)}})]
    )
    ct = r.crosstab("m")
    assert ct == {"TP": 1, "TN": 0, "FP": 0, "FN": 0}


def test_crosstab_tn() -> None:
    r = _make_report(
        rows=[_row("t1", {"pii": False}, {"m": {"pii": _cell(False)}})]
    )
    assert r.crosstab("m")["TN"] == 1


def test_crosstab_fp() -> None:
    r = _make_report(
        rows=[_row("t1", {"pii": False}, {"m": {"pii": _cell(True)}})]
    )
    assert r.crosstab("m")["FP"] == 1


def test_crosstab_fn() -> None:
    r = _make_report(
        rows=[_row("t1", {"pii": True}, {"m": {"pii": _cell(False)}})]
    )
    assert r.crosstab("m")["FN"] == 1


def test_crosstab_missing_model() -> None:
    r = _make_report(rows=[_row("t1", {"pii": True}, {})])
    assert r.crosstab("no-such-model") == {"TP": 0, "TN": 0, "FP": 0, "FN": 0}


# -- BenchReport.crosstab_by_matcher ----------------------------------------


def test_crosstab_by_matcher_filters() -> None:
    r = _make_report(
        rows=[
            _row("t1", {"pii": True, "secrets": False}, {
                "m": {"pii": _cell(True), "secrets": _cell(True)},
            }),
        ],
    )
    ct_pii = r.crosstab_by_matcher("m", "pii")
    assert ct_pii == {"TP": 1, "TN": 0, "FP": 0, "FN": 0}
    ct_sec = r.crosstab_by_matcher("m", "secrets")
    assert ct_sec == {"TP": 0, "TN": 0, "FP": 1, "FN": 0}


def test_crosstab_by_matcher_irrelevant_row_skipped() -> None:
    """Row without the matcher in expectations is skipped."""
    r = _make_report(
        rows=[_row("t1", {"pii": True}, {"m": {"pii": _cell(True)}})]
    )
    ct = r.crosstab_by_matcher("m", "secrets")
    assert ct == {"TP": 0, "TN": 0, "FP": 0, "FN": 0}


# -- BenchReport.model_mean_elapsed -----------------------------------------


def test_model_mean_elapsed() -> None:
    r = _make_report(
        rows=[
            _row("t1", {"pii": True}, {"m": {"pii": _cell(True, elapsed=2.0)}}),
            _row("t2", {"pii": True}, {"m": {"pii": _cell(True, elapsed=4.0)}}),
        ],
    )
    assert r.model_mean_elapsed("m") == 3.0


def test_model_mean_elapsed_no_results() -> None:
    r = _make_report(rows=[_row("t1", {"pii": True}, {})])
    assert r.model_mean_elapsed("m") == 0.0


# -- BenchReport.suspect_labels ---------------------------------------------


def test_suspect_labels_needs_three_models() -> None:
    r = _make_report(models=["m1", "m2"])
    assert r.suspect_labels() == []


def test_suspect_labels_unanimous_wrong() -> None:
    """All top-3 models disagree with the label -> suspect."""
    cells = {
        "m1": {"pii": _cell(False)},
        "m2": {"pii": _cell(False)},
        "m3": {"pii": _cell(False)},
    }
    r = _make_report(
        models=["m1", "m2", "m3"],
        rows=[_row("t1", {"pii": True}, cells)],
    )
    suspects = r.suspect_labels()
    assert len(suspects) == 1
    assert suspects[0]["title"] == "t1"
    assert suspects[0]["matcher"] == "pii"
    assert suspects[0]["expected"] is True


def test_suspect_labels_not_unanimous() -> None:
    """If one top model agrees with the label, not suspect."""
    cells = {
        "m1": {"pii": _cell(False)},
        "m2": {"pii": _cell(True)},  # agrees
        "m3": {"pii": _cell(False)},
    }
    r = _make_report(
        models=["m1", "m2", "m3"],
        rows=[_row("t1", {"pii": True}, cells)],
    )
    assert r.suspect_labels() == []


def test_suspect_labels_short_model_names() -> None:
    """Model names with '/' are shortened in dissenter list."""
    cells = {
        "ollama/m1": {"pii": _cell(False)},
        "ollama/m2": {"pii": _cell(False)},
        "plain": {"pii": _cell(False)},
    }
    r = _make_report(
        models=["ollama/m1", "ollama/m2", "plain"],
        rows=[_row("t1", {"pii": True}, cells)],
    )
    suspects = r.suspect_labels()
    assert "m1" in suspects[0]["dissenters"]
    assert "plain" in suspects[0]["dissenters"]


# -- testcase_to_conversation ------------------------------------------------


def test_tc_to_conv() -> None:
    tc: dict[str, Any] = {
        "title": "Test Case",
        "messages": [
            {"user": "  Hello  "},
            {"assistant": "Hi!\n"},
        ],
    }
    conv = tc_to_conv(tc)
    assert conv.id == "bench"
    assert conv.title == "Test Case"
    assert len(conv.messages) == 2
    assert conv.messages[0] == Message(role="user", text="Hello")
    assert conv.messages[1] == Message(role="assistant", text="Hi!")


def test_testcase_to_conversation_no_messages() -> None:
    conv = tc_to_conv({"title": "Empty"})
    assert conv.messages == []


# -- _natural_sort_key -------------------------------------------------------


def test_natural_sort_key_numeric() -> None:
    names = ["model10", "model2", "model1"]
    assert sorted(names, key=_natural_sort_key) == ["model1", "model2", "model10"]


def test_natural_sort_key_mixed() -> None:
    names = ["qwen2.5:32b", "qwen2.5:7b", "llama3:8b"]
    result = sorted(names, key=_natural_sort_key)
    assert result[0] == "llama3:8b"
    assert result[1] == "qwen2.5:7b"
    assert result[2] == "qwen2.5:32b"


# -- build_report -----------------------------------------------------------


def _populate_db(conn: sqlite3.Connection) -> None:
    """Insert test rows for build_report tests."""
    for tc, matcher, model, found in [
        ("tc1", "pii", "model-a", 1),
        ("tc1", "pii", "model-b", 0),
        ("tc1", "secrets", "model-a", 0),
    ]:
        conn.execute(
            "INSERT INTO results (testcase, matcher, model, found, reasoning,"
            " elapsed, run_at) VALUES (?, ?, ?, ?, '', 1.0, '2025-01-01')",
            (tc, matcher, model, found),
        )
    conn.commit()


def test_build_report_populates_rows(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    _populate_db(conn)
    testcases = [
        {
            "name": "tc1",
            "title": "Test Case 1",
            "messages": [{"user": "hi"}],
            "expect": {"pii": True, "secrets": False},
        },
    ]
    report = build_report(conn, testcases, ["pii", "secrets"])
    conn.close()

    assert len(report.rows) == 1
    assert report.rows[0].title == "Test Case 1"
    assert "model-a" in report.models
    assert "model-b" in report.models
    # model-a got both right (pii=True->found=True, secrets=False->found=False)
    assert report.model_score("model-a") == (2, 2)


def test_build_report_skips_irrelevant_testcases(tmp_path: Path) -> None:
    """Testcases with no matching expectations are excluded."""
    conn = init_db(tmp_path / "test.db")
    _populate_db(conn)
    testcases = [
        {
            "name": "tc1",
            "title": "Has Expectations",
            "messages": [{"user": "hi"}],
            "expect": {"pii": True},
        },
        {
            "name": "tc2",
            "title": "No Matching Expectations",
            "messages": [{"user": "hi"}],
            "expect": {"unrelated": True},
        },
    ]
    report = build_report(conn, testcases, ["pii"])
    conn.close()

    assert len(report.rows) == 1
    assert report.rows[0].title == "Has Expectations"


def test_build_report_sorts_models_by_score(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    _populate_db(conn)
    testcases = [
        {
            "name": "tc1",
            "title": "TC1",
            "messages": [{"user": "hi"}],
            "expect": {"pii": True},
        },
    ]
    report = build_report(conn, testcases, ["pii"])
    conn.close()

    # model-a got pii right (found=True, expected=True)
    # model-b got pii wrong (found=False, expected=True)
    # model-a should sort first (better score)
    assert report.models[0] == "model-a"


# -- DB operations -----------------------------------------------------------


def test_init_db_creates_tables(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "results" in tables
    assert "results_old" in tables


def test_init_db_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn1 = init_db(db_path)
    conn1.close()
    conn2 = init_db(db_path)
    conn2.close()


def test_save_and_load_cached(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    save_result(conn, "tc1", "pii", "model-a", "local", True, "found it", 1.5)
    cached = load_cached(conn, "model-a")
    conn.close()

    assert "tc1" in cached
    assert "pii" in cached["tc1"]
    assert cached["tc1"]["pii"]["found"] is True
    assert cached["tc1"]["pii"]["reasoning"] == "found it"
    assert cached["tc1"]["pii"]["elapsed"] == 1.5


def test_load_cached_different_model(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    save_result(conn, "tc1", "pii", "model-a", "local", True, "r", 1.0)
    cached = load_cached(conn, "model-b")
    conn.close()
    assert cached == {}


def test_save_result_upserts(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    save_result(conn, "tc1", "pii", "m", "local", True, "first", 1.0)
    save_result(conn, "tc1", "pii", "m", "local", False, "second", 2.0)
    cached = load_cached(conn, "m")
    conn.close()

    assert cached["tc1"]["pii"]["found"] is False
    assert cached["tc1"]["pii"]["reasoning"] == "second"


def test_save_result_with_response_json(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "test.db")
    save_result(conn, "tc1", "pii", "m", "local", True, "r", 1.0,
                response_json='{"found": true}')
    row = conn.execute(
        "SELECT response_json FROM results WHERE testcase='tc1'"
    ).fetchone()
    conn.close()
    assert row[0] == '{"found": true}'


# -- load_testcases ----------------------------------------------------------


def test_load_testcases(tmp_path: Path) -> None:
    (tmp_path / "case_a.yaml").write_text(
        "title: Case A\nmessages:\n  - user: Hello\n", encoding="utf-8"
    )
    (tmp_path / "case_b.yaml").write_text(
        "title: Case B\nmessages:\n  - user: Hi\n", encoding="utf-8"
    )
    (tmp_path / "not_yaml.txt").write_text("ignored", encoding="utf-8")

    tcs = load_testcases(tmp_path)
    assert len(tcs) == 2
    assert tcs[0]["name"] == "case_a"
    assert tcs[1]["name"] == "case_b"
    assert tcs[0]["title"] == "Case A"


def test_load_testcases_empty_dir(tmp_path: Path) -> None:
    assert load_testcases(tmp_path) == []


# -- render_report -----------------------------------------------------------


def test_render_report_produces_html() -> None:
    report = _make_report(
        models=["ollama/qwen2.5:7b"],
        matchers=["pii"],
        rows=[
            _row("Test Case", {"pii": True}, {
                "ollama/qwen2.5:7b": {"pii": _cell(True)},
            }),
        ],
    )
    html = render_report(report)
    assert "<html" in html
    assert "Test Case" in html
    assert "qwen2.5:7b" in html


def test_render_report_escapes_html() -> None:
    report = _make_report(
        models=["m"],
        matchers=["pii"],
        rows=[_row("<script>alert(1)</script>", {"pii": True}, {})],
    )
    html = render_report(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_report_empty() -> None:
    report = _make_report(models=[], matchers=[], rows=[])
    html = render_report(report)
    assert "<html" in html
