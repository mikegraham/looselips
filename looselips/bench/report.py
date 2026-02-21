from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, TypedDict

from looselips.parsers import Conversation, Message


class SuspectLabel(TypedDict):
    title: str
    matcher: str
    expected: bool
    dissenters: str

# -- Report dataclasses (built from DB, not from a single run) ----------------


@dataclass
class CellResult:
    """Single model+matcher result for one testcase."""

    found: bool
    reasoning: str
    elapsed: float
    response_json: str | None = None


@dataclass
class ReportRow:
    """One testcase in the report."""

    title: str
    messages: list[Message]
    expectations: dict[str, bool]  # matcher -> expected
    cells: dict[str, dict[str, CellResult]] = field(
        default_factory=dict,
    )  # model -> matcher -> CellResult


@dataclass
class BenchReport:
    """Multi-model benchmark report built from DB contents."""

    models: list[str]
    matchers: list[str]
    rows: list[ReportRow]

    def model_score(self, model: str) -> tuple[int, int]:
        """(correct, total) across all testcases/matchers for a model."""
        correct = 0
        total = 0
        for row in self.rows:
            model_cells = row.cells.get(model, {})
            for matcher, expected in row.expectations.items():
                cell = model_cells.get(matcher)
                if cell is None:
                    continue
                total += 1
                if cell.found == expected:
                    correct += 1
        return correct, total

    def crosstab(self, model: str) -> dict[str, int]:
        """{'TP': n, 'TN': n, 'FP': n, 'FN': n} aggregated across all matchers."""
        counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for row in self.rows:
            model_cells = row.cells.get(model, {})
            for matcher, expected in row.expectations.items():
                cell = model_cells.get(matcher)
                if cell is None:
                    continue
                if expected and cell.found:
                    counts["TP"] += 1
                elif not expected and not cell.found:
                    counts["TN"] += 1
                elif not expected and cell.found:
                    counts["FP"] += 1
                else:
                    counts["FN"] += 1
        return counts

    def crosstab_by_matcher(self, model: str, matcher: str) -> dict[str, int]:
        """{'TP': n, 'TN': n, 'FP': n, 'FN': n} for a single matcher."""
        counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for row in self.rows:
            expected = row.expectations.get(matcher)
            if expected is None:
                continue
            cell = row.cells.get(model, {}).get(matcher)
            if cell is None:
                continue
            if expected and cell.found:
                counts["TP"] += 1
            elif not expected and not cell.found:
                counts["TN"] += 1
            elif not expected and cell.found:
                counts["FP"] += 1
            else:
                counts["FN"] += 1
        return counts

    def model_mean_elapsed(self, model: str) -> float:
        """Mean elapsed time (seconds) across all results for a model."""
        times = []
        for row in self.rows:
            model_cells = row.cells.get(model, {})
            for cell in model_cells.values():
                if cell is not None:
                    times.append(cell.elapsed)
        if not times:
            return 0.0
        return sum(times) / len(times)

    def suspect_labels(self) -> list[SuspectLabel]:
        """Flag testcase+matcher combos where top models unanimously disagree.

        A combo is suspect only if ALL of the top 3 models (that have
        results) got it wrong.  Returns a list of dicts with keys:
        title, matcher, expected, dissenters.
        """
        top = self.models[:min(3, len(self.models))]
        if len(top) < 3:
            return []

        def _short(model: str) -> str:
            return model.split("/", 1)[-1] if "/" in model else model

        suspects: list[SuspectLabel] = []
        for row in self.rows:
            for matcher, expected in row.expectations.items():
                voted = [
                    m for m in top
                    if row.cells.get(m, {}).get(matcher) is not None
                ]
                wrong = [
                    m for m in voted
                    if row.cells[m][matcher].found != expected
                ]
                if voted and len(wrong) == len(voted):
                    suspects.append({
                        "title": row.title,
                        "matcher": matcher,
                        "expected": expected,
                        "dissenters": ", ".join(_short(m) for m in wrong),
                    })
        return suspects


# -- Report building from DB --------------------------------------------------


def testcase_to_conversation(testcase: dict[str, Any]) -> Conversation:
    """Convert a YAML testcase dict to a Conversation."""
    messages = []
    for entry in testcase.get("messages", []):
        for role, text in entry.items():
            messages.append(Message(role=role, text=text.strip()))
    return Conversation(
        id="bench",
        title=testcase["title"],
        messages=messages,
    )


def _natural_sort_key(model: str) -> list[float | str]:
    """Tiebreaker: natural sort on the full model name."""
    parts: list[float | str] = []
    for tok in re.split(r"(\d+(?:\.\d+)?)", model):
        if tok:
            try:
                parts.append(float(tok))
            except ValueError:
                parts.append(tok.lower())
    return parts


def build_report(
    conn: sqlite3.Connection,
    testcases: list[dict[str, Any]],
    matcher_names: list[str],
) -> BenchReport:
    """Build a multi-model BenchReport from all DB results."""
    # Discover all models
    model_rows = conn.execute(
        "SELECT DISTINCT model FROM results",
    ).fetchall()
    models = [r[0] for r in model_rows]

    # Load all results: {testcase: {model: {matcher: CellResult}}}
    all_rows = conn.execute(
        "SELECT testcase, matcher, model, found, reasoning, elapsed,"
        " response_json "
        "FROM results",
    ).fetchall()
    db_data: dict[str, dict[str, dict[str, CellResult]]] = {}
    for testcase, matcher, model, found, reasoning, elapsed, rj in all_rows:
        db_data.setdefault(testcase, {}).setdefault(model, {})[matcher] = (
            CellResult(found=bool(found), reasoning=reasoning,
                       elapsed=elapsed, response_json=rj)
        )

    # Build rows
    rows = []
    for tc in testcases:
        expectations = tc.get("expect", {})
        relevant = {m: expectations[m] for m in matcher_names if m in expectations}
        if not relevant:
            continue
        conv = testcase_to_conversation(tc)
        row = ReportRow(
            title=tc["title"],
            messages=conv.messages,
            expectations=relevant,
            cells=db_data.get(tc["name"], {}),
        )
        rows.append(row)

    # Sort models by F2 score (descending), natural name as tiebreaker
    report = BenchReport(models=models, matchers=matcher_names, rows=rows)

    def _score_key(model: str) -> tuple[float, list[float | str]]:
        ct = report.crosstab(model)
        denom = 5 * ct["TP"] + 4 * ct["FN"] + ct["FP"]
        f2 = 5 * ct["TP"] / denom if denom else 0.0
        return (-f2, _natural_sort_key(model))

    report.models = sorted(models, key=_score_key)
    return report
