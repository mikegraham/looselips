#!/usr/bin/env python3
"""Benchmark harness for LLM matchers.

Runs LLM matchers from a TOML config against labeled testcases (YAML files
in scripts/bench/ that pair a conversation with expected matcher outcomes)
and produces an HTML report comparing expected vs actual results.

Results are saved incrementally to a SQLite database so progress survives
interruptions and new testcases/matchers only run what's missing.

Model resolution: --model overrides all matchers. Without it, each matcher
uses its own model from config, falling back to the config-level default_model.

Examples:

    # Run with config's default model against built-in testcases
    python scripts/bench.py --backend local -c config.toml

    # Override model for all matchers
    python scripts/bench.py --backend local --model ollama/qwen2.5:7b -c config.toml

    # Separate DB and report paths, custom testcases
    python scripts/bench.py --backend local -c config.toml \\
        --db results.db -o report.html --testcases /path/to/cases

    # Run only the 'shoe_size' matcher
    python scripts/bench.py --backend local -c config.toml -m shoe_size

    # Re-run a matcher from scratch (delete cached results first)
    python scripts/bench.py --backend local -c config.toml -m shoe_size --force

    # Re-render report from existing DB without running anything
    python scripts/bench.py --report-only --db results.db -o report.html
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections.abc import Callable
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

    def suspect_labels(self) -> list[dict]:
        """Flag testcase+matcher combos where top models disagree with the label.

        A combo is suspect if at least 3 of the top 4 models got it wrong.
        Returns a list of dicts with keys: title, matcher, expected, dissenters.
        """
        top = self.models[:min(4, len(self.models))]
        if len(top) < 3:
            return []

        def _short(model: str) -> str:
            return model.split("/", 1)[-1] if "/" in model else model

        suspects: list[dict] = []
        for row in self.rows:
            for matcher, expected in row.expectations.items():
                wrong = [
                    m for m in top
                    if (cell := row.cells.get(m, {}).get(matcher)) is not None
                    and cell.found != expected
                ]
                if len(wrong) >= 3:
                    suspects.append({
                        "title": row.title,
                        "matcher": matcher,
                        "expected": expected,
                        "dissenters": ", ".join(_short(m) for m in wrong),
                    })
        return suspects


# -- Testcase loading ---------------------------------------------------------


def load_testcases(testcase_dir: Path) -> list[dict]:
    """Load all YAML testcases from a directory.

    Each testcase dict gets a 'name' key set to the filename stem.
    """
    testcases = []
    for p in sorted(testcase_dir.glob("*.yaml")):
        with open(p) as f:
            tc = yaml.safe_load(f)
        tc["name"] = p.stem
        testcases.append(tc)
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
            testcase      TEXT NOT NULL,
            matcher       TEXT NOT NULL,
            model         TEXT NOT NULL,
            backend       TEXT,
            found         INTEGER NOT NULL,
            reasoning     TEXT NOT NULL DEFAULT '',
            elapsed       REAL NOT NULL DEFAULT 0,
            response_json TEXT,
            run_at        TEXT NOT NULL,
            PRIMARY KEY (testcase, matcher, model)
        )
    """)
    # Migrate existing databases that lack the response_json column.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(results)").fetchall()}
    if "response_json" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN response_json TEXT")
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
    response_json: str | None = None,
) -> None:
    """Insert or replace a result row."""
    conn.execute(
        "INSERT OR REPLACE INTO results "
        "(testcase, matcher, model, backend, found, reasoning, elapsed,"
        " response_json, run_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (testcase, matcher, model, backend, int(found), reasoning, elapsed,
         response_json, datetime.now().isoformat()),
    )
    conn.commit()


# -- Report building from DB --------------------------------------------------


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
    testcases: list[dict],
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

    # Sort models by accuracy * recall (descending), natural name as tiebreaker
    report = BenchReport(models=models, matchers=matcher_names, rows=rows)

    def _score_key(model: str) -> tuple[float, list[float | str]]:
        ct = report.crosstab(model)
        total = ct["TP"] + ct["TN"] + ct["FP"] + ct["FN"]
        pos = ct["TP"] + ct["FN"]
        acc = (ct["TP"] + ct["TN"]) / total if total else 0.0
        recall = ct["TP"] / pos if pos else 0.0
        return (-acc * recall, _natural_sort_key(model))

    report.models = sorted(models, key=_score_key)
    return report


# -- Bench runner --------------------------------------------------------------


def run_bench(
    testcases: list[dict],
    matchers: list[tuple[str, str, str]],
    conn: sqlite3.Connection,
    model: str,
    backend: str = "local",
    model_override: str | None = None,
    on_result: Callable[[], None] | None = None,
) -> None:
    """Run matchers against testcases, skipping already-completed work.

    Checks the database for cached results per (testcase, matcher, model).
    Only missing combos are run.  *on_result* is called after each testcase
    finishes so the report can be rebuilt.
    """
    cached = load_cached(conn, model)
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
        tc_cache = cached.get(testcase["name"], {})
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

            save_result(conn, testcase["name"], mr.name, model,
                        backend, mr.found, mr.reasoning, mr.elapsed,
                        mr.response_json)

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
    env.filters["short_model"] = lambda s: s.split("/", 1)[-1] if "/" in s else s
    template = env.get_template("bench_report.html.j2")
    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    return template.render(report=report, now=now_str)


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM matcher benchmarks against labeled testcases.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=None,
        help="TOML config file with matcher definitions "
             "(required unless --report-only)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model for all matchers, ignoring per-matcher and "
             "default_model from config (any LiteLLM model string)",
    )
    parser.add_argument(
        "--testcases", type=Path, default=TESTCASE_DIR,
        help=f"Directory of YAML testcases (default: {TESTCASE_DIR})",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output HTML report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="SQLite database path (default: <output>.db)",
    )
    parser.add_argument(
        "--backend", default=None,
        help="Compute backend tag stored with results (e.g. local, modal/L4). "
             "Required unless --report-only.",
    )
    parser.add_argument(
        "-m", "--matcher", action="append", dest="matchers", metavar="NAME",
        help="Only run these matchers (repeatable; default: all from config)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete cached results for the selected matchers before running",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Re-render report from existing DB without running any benchmarks",
    )
    args = parser.parse_args()

    # Load testcases
    testcases = load_testcases(args.testcases)

    # Open SQLite database
    db_path = args.db or args.output.with_suffix(".db")
    conn = init_db(db_path)

    if args.report_only:
        # Pull matcher names from DB -- no config needed
        matcher_rows = conn.execute(
            "SELECT DISTINCT matcher FROM results ORDER BY matcher",
        ).fetchall()
        matcher_names = [r[0] for r in matcher_rows]
        report = build_report(conn, testcases, matcher_names)
        html = render_report(report)
        args.output.write_text(html, encoding="utf-8")
        conn.close()
        print(f"Report written to {args.output}")
        return

    if not args.config:
        parser.error("--config is required (unless --report-only)")
    if not args.backend:
        parser.error("--backend is required (unless --report-only)")

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

    # Filter to requested matchers if -m/--matcher given
    if args.matchers:
        available = {name for name, _, _ in llm_matchers}
        for m in args.matchers:
            if m not in available:
                parser.error(
                    f"matcher {m!r} not in config "
                    f"(available: {', '.join(sorted(available))})"
                )
        llm_matchers = [
            (n, p, mod) for n, p, mod in llm_matchers if n in args.matchers
        ]

    matcher_names = [name for name, _, _ in llm_matchers]

    effective_model = args.model or config.default_model or llm_matchers[0][2]
    print(f"Model: {effective_model}")
    print(f"Matchers: {', '.join(matcher_names)}")
    print()

    # --force: delete cached results for selected matchers before running
    if args.force:
        placeholders = ", ".join("?" for _ in matcher_names)
        deleted = conn.execute(
            f"DELETE FROM results WHERE model = ? AND matcher IN ({placeholders})",
            [effective_model, *matcher_names],
        ).rowcount
        conn.commit()
        if deleted:
            print(f"Deleted {deleted} cached results (--force)")

    print(f"Loaded {len(testcases)} testcases from {args.testcases}")
    cached = load_cached(conn, effective_model)
    if cached:
        print(f"Found {len(cached)} cached results in {db_path}")
    print()

    # For the report, use all matchers from config (not just the filtered ones)
    all_matcher_names = [m.name for m in config.matchers if m.type == "llm"]

    # Callback: rebuild and re-render from DB after each testcase
    def on_result() -> None:
        report = build_report(conn, testcases, all_matcher_names)
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

    # Final report (all matchers, not just the filtered ones)
    report = build_report(conn, testcases, all_matcher_names)

    # Print summary
    print()
    print("=" * 60)
    print(f"SUMMARY -- model: {effective_model}")
    print("-" * 60)
    ct = report.crosstab(effective_model)
    tested = ct["TP"] + ct["TN"] + ct["FP"] + ct["FN"]
    acc = ((ct["TP"] + ct["TN"]) / tested * 100) if tested else 0
    recall = (ct["TP"] / (ct["TP"] + ct["FN"]) * 100) if (ct["TP"] + ct["FN"]) else 0
    print(f"  TP={ct['TP']} TN={ct['TN']} FP={ct['FP']} FN={ct['FN']}")
    print(f"  Accuracy: {acc:.0f}%  Recall: {recall:.0f}%")
    print("=" * 60)

    # Final render
    html = render_report(report)
    args.output.write_text(html, encoding="utf-8")
    conn.close()
    print(f"\nReport written to {args.output}")
    print(f"Results saved to {db_path}")


if __name__ == "__main__":
    main()
