#!/usr/bin/env python3
"""Benchmark harness for LLM matchers.

Runs LLM matchers from a TOML config against labeled testcases (YAML files
in scripts/bench/ that pair a conversation with expected matcher outcomes)
and produces an HTML report comparing expected vs actual results.

Results are saved incrementally to a SQLite database so progress survives
interruptions and new testcases/matchers only run what's missing.

Usage:
    python scripts/bench.py --model ollama/qwen2.5:7b -c examples/example_config.toml
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import escape

from looselips.cli.config import load_config
from looselips.parsers import Conversation, Message
from looselips.scanner import scan_conversation_llm

TESTCASE_DIR = Path(__file__).resolve().parent / "bench"
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "looselips" / "templates"
DEFAULT_OUTPUT = "bench_report.html"


# -- Report dataclasses (built from DB, not from a single run) ----------------


@dataclass
class CellResult:
    """Single model+matcher result for one testcase."""

    found: bool
    reasoning: str
    elapsed: float


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

    def crosstab(self, model: str, matcher: str) -> dict[str, int]:
        """{'TP': n, 'TN': n, 'FP': n, 'FN': n} for a model-matcher pair."""
        counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for row in self.rows:
            if matcher not in row.expectations:
                continue
            cell = row.cells.get(model, {}).get(matcher)
            if cell is None:
                continue
            expected = row.expectations[matcher]
            if expected and cell.found:
                counts["TP"] += 1
            elif not expected and not cell.found:
                counts["TN"] += 1
            elif not expected and cell.found:
                counts["FP"] += 1
            else:
                counts["FN"] += 1
        return counts


# -- Testcase loading ---------------------------------------------------------


def load_testcases(testcase_dir: Path) -> list[dict]:
    """Load all YAML testcases from a directory."""
    testcases = []
    for p in sorted(testcase_dir.glob("*.yaml")):
        with open(p) as f:
            testcases.append(yaml.safe_load(f))
    return testcases


def testcase_to_conversation(testcase: dict) -> Conversation:
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


# -- SQLite persistence -------------------------------------------------------


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the results database and ensure the schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            testcase  TEXT NOT NULL,
            matcher   TEXT NOT NULL,
            model     TEXT NOT NULL,
            backend   TEXT NOT NULL DEFAULT 'local',
            found     INTEGER NOT NULL,
            reasoning TEXT NOT NULL DEFAULT '',
            elapsed   REAL NOT NULL DEFAULT 0,
            run_at    TEXT NOT NULL,
            PRIMARY KEY (testcase, matcher, model)
        )
    """)
    conn.commit()
    return conn


def load_cached(
    conn: sqlite3.Connection, model: str,
) -> dict[str, dict[str, dict]]:
    """Load cached results for a model.

    Returns {testcase_title: {matcher_name: {found, reasoning, elapsed}}}.
    """
    rows = conn.execute(
        "SELECT testcase, matcher, found, reasoning, elapsed "
        "FROM results WHERE model = ?",
        (model,),
    ).fetchall()
    cached: dict[str, dict[str, dict]] = {}
    for testcase, matcher, found, reasoning, elapsed in rows:
        cached.setdefault(testcase, {})[matcher] = {
            "found": bool(found),
            "reasoning": reasoning,
            "elapsed": elapsed,
        }
    return cached


def save_result(
    conn: sqlite3.Connection,
    testcase: str,
    matcher: str,
    model: str,
    backend: str,
    found: bool,
    reasoning: str,
    elapsed: float,
) -> None:
    """Insert or replace a result row."""
    conn.execute(
        "INSERT OR REPLACE INTO results "
        "(testcase, matcher, model, backend, found, reasoning, elapsed, run_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (testcase, matcher, model, backend, int(found), reasoning, elapsed,
         datetime.now().isoformat()),
    )
    conn.commit()


# -- Report building from DB --------------------------------------------------


def build_report(
    conn: sqlite3.Connection,
    testcases: list[dict],
    matcher_names: list[str],
) -> BenchReport:
    """Build a multi-model BenchReport from all DB results."""
    # Discover all models
    model_rows = conn.execute(
        "SELECT DISTINCT model FROM results ORDER BY model",
    ).fetchall()
    models = [r[0] for r in model_rows]

    # Load all results: {testcase: {model: {matcher: CellResult}}}
    all_rows = conn.execute(
        "SELECT testcase, matcher, model, found, reasoning, elapsed "
        "FROM results",
    ).fetchall()
    db_data: dict[str, dict[str, dict[str, CellResult]]] = {}
    for testcase, matcher, model, found, reasoning, elapsed in all_rows:
        db_data.setdefault(testcase, {}).setdefault(model, {})[matcher] = (
            CellResult(found=bool(found), reasoning=reasoning, elapsed=elapsed)
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
            cells=db_data.get(tc["title"], {}),
        )
        rows.append(row)

    return BenchReport(models=models, matchers=matcher_names, rows=rows)


# -- Bench runner --------------------------------------------------------------


def run_bench(
    testcases: list[dict],
    matchers: list[tuple[str, str, str]],
    conn: sqlite3.Connection,
    model: str,
    backend: str = "local",
    model_override: str | None = None,
    on_result: callable = None,
) -> None:
    """Run matchers against testcases, skipping already-completed work.

    Checks the database for cached results per (testcase, matcher, model,
    backend).  Only missing combos are run.  *on_result* is called after
    each testcase finishes so the report can be rebuilt.
    """
    cached = load_cached(conn, model, backend)
    total = len(testcases)

    for i, testcase in enumerate(testcases, 1):
        conv = testcase_to_conversation(testcase)
        expectations = testcase.get("expect", {})

        matcher_names = [name for name, _, _ in matchers]
        relevant = [m for m in matcher_names if m in expectations]
        if not relevant:
            if on_result:
                on_result()
            continue

        # Figure out which matchers still need running
        tc_cache = cached.get(testcase["title"], {})
        needed = [m for m in relevant if m not in tc_cache]

        if not needed:
            print(f"({i}/{total}) {conv.title!r} -- cached")
            if on_result:
                on_result()
            continue

        # Build effective matchers (apply model override, filter to needed)
        effective = [
            (name, prompt, model_override or orig_model)
            for name, prompt, orig_model in matchers
            if name in needed
        ]

        print(f"({i}/{total}) {conv.title!r} "
              f"(matchers: {', '.join(needed)})")

        for mr in scan_conversation_llm(conv, effective):
            if mr.error:
                print(f"  {mr.name}: ERROR -- {mr.error} ({mr.elapsed:.1f}s)")
                continue

            save_result(conn, testcase["title"], mr.name, model,
                        backend, mr.found, mr.reasoning, mr.elapsed)

            expected = expectations[mr.name]
            ok = mr.found == expected
            marker = "OK" if ok else "WRONG"
            print(f"  {mr.name}: {marker} (expected={expected}, "
                  f"got={mr.found}, {mr.elapsed:.1f}s)")

        if on_result:
            on_result()


# -- Rendering -----------------------------------------------------------------


def render_report(report: BenchReport) -> str:
    """Render the benchmark report to HTML."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(default=True),
    )
    env.filters["escape"] = escape
    template = env.get_template("bench_report.html.j2")
    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    return template.render(report=report, now=now_str)


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM matcher benchmarks against labeled testcases.",
    )
    parser.add_argument(
        "-c", "--config", required=True, type=Path,
        help="TOML config file with matcher definitions",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model for all matchers (any LiteLLM model string)",
    )
    parser.add_argument(
        "--testcases", type=Path, default=TESTCASE_DIR,
        help=f"Directory of YAML testcases (default: {TESTCASE_DIR})",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--backend", default="local",
        help="Compute backend label stored with results (default: local)",
    )
    args = parser.parse_args()

    # Load config and extract LLM matchers
    config = load_config(args.config)
    llm_matchers: list[tuple[str, str, str]] = []
    for m in config.matchers:
        if m.type != "llm":
            continue
        model = args.model or m.model or config.default_model
        if not model:
            parser.error(f"LLM matcher {m.name!r} has no model and no --model given")
        llm_matchers.append((m.name, m.prompt, model))

    if not llm_matchers:
        parser.error("no LLM matchers found in config")

    matcher_names = [name for name, _, _ in llm_matchers]
    effective_model = args.model or config.default_model or llm_matchers[0][2]
    print(f"Model: {effective_model}")
    print(f"Matchers: {', '.join(matcher_names)}")
    print()

    # Load testcases
    testcases = load_testcases(args.testcases)
    print(f"Loaded {len(testcases)} testcases from {args.testcases}")

    # Open SQLite database
    db_path = args.output.with_suffix(".db")
    conn = init_db(db_path)
    cached = load_cached(conn, effective_model)
    if cached:
        print(f"Found {len(cached)} cached results in {db_path}")
    print()

    # Callback: rebuild and re-render from DB after each testcase
    def on_result() -> None:
        report = build_report(conn, testcases, matcher_names)
        html = render_report(report)
        args.output.write_text(html, encoding="utf-8")

    # Run benchmarks
    run_bench(
        testcases, llm_matchers,
        conn=conn,
        model=effective_model,
        backend=args.backend,
        model_override=args.model,
        on_result=on_result,
    )

    # Final report
    report = build_report(conn, testcases, matcher_names)

    # Print summary
    print()
    print("=" * 60)
    print(f"SUMMARY -- model: {effective_model}")
    print("-" * 60)
    correct, total = report.model_score(effective_model)
    pct = (correct / total * 100) if total else 0
    print(f"  Score: {correct}/{total} correct ({pct:.0f}%)")
    for name in matcher_names:
        ct = report.crosstab(effective_model, name)
        tested = ct["TP"] + ct["TN"] + ct["FP"] + ct["FN"]
        ok = ct["TP"] + ct["TN"]
        mpct = (ok / tested * 100) if tested else 0
        print(f"  {name}: {ok}/{tested} ({mpct:.0f}%) "
              f"[TP={ct['TP']} TN={ct['TN']} FP={ct['FP']} FN={ct['FN']}]")
    print("=" * 60)

    # Final render
    html = render_report(report)
    args.output.write_text(html, encoding="utf-8")
    conn.close()
    print(f"\nReport written to {args.output}")
    print(f"Results saved to {db_path}")


if __name__ == "__main__":
    main()
