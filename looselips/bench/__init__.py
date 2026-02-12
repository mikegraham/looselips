#!/usr/bin/env python3
"""Benchmark harness for LLM matchers.

Runs LLM matchers from a TOML config against labeled testcases (YAML files
in looselips/bench/testcases/ that pair a conversation with expected matcher
outcomes) and produces an HTML report comparing expected vs actual results.

Results are saved incrementally to a SQLite database so progress survives
interruptions and new testcases/matchers only run what's missing.

Model resolution: --model overrides all matchers. Without it, each matcher
uses its own model from config, falling back to the config-level default_model.

Examples:

    # Run with config's default model against built-in testcases
    python -m looselips.bench --backend local -c config.toml

    # Override model for all matchers
    python -m looselips.bench --backend local --model ollama/qwen2.5:7b -c config.toml

    # Separate DB and report paths, custom testcases
    python -m looselips.bench --backend local -c config.toml \\
        --db results.db -o report.html --testcases /path/to/cases

    # Run only the 'shoe_size' matcher
    python -m looselips.bench --backend local -c config.toml -m shoe_size

    # Re-run a matcher from scratch (delete cached results first)
    python -m looselips.bench --backend local -c config.toml -m shoe_size --force

    # Re-render report from existing DB without running anything
    python -m looselips.bench --report-only --db results.db -o report.html
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from looselips.bench.db import init_db, load_cached, save_result
from looselips.bench.render import render_report
from looselips.bench.report import build_report, testcase_to_conversation
from looselips.cli.config import load_config
from looselips.scanner import scan_conversation_llm

TESTCASE_DIR = Path(__file__).resolve().parent / "testcases"
DEFAULT_OUTPUT = "bench_report.html"


# -- Testcase loading ---------------------------------------------------------


def load_testcases(testcase_dir: Path) -> list[dict[str, Any]]:
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


# -- Bench runner --------------------------------------------------------------


def run_bench(
    testcases: list[dict[str, Any]],
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
    try:
        _main_with_conn(parser, args, conn, testcases, db_path)
    finally:
        conn.close()


def _main_with_conn(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    testcases: list[dict[str, Any]],
    db_path: Path,
) -> None:
    if args.report_only:
        # Pull matcher names from DB -- no config needed
        matcher_rows = conn.execute(
            "SELECT DISTINCT matcher FROM results ORDER BY matcher",
        ).fetchall()
        matcher_names = [r[0] for r in matcher_rows]
        report = build_report(conn, testcases, matcher_names)
        html = render_report(report)
        args.output.write_text(html, encoding="utf-8")
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

    # --force: archive cached results, then remove from active table
    if args.force:
        placeholders = ", ".join("?" for _ in matcher_names)
        params = [effective_model, *matcher_names]
        where = f"model = ? AND matcher IN ({placeholders})"
        conn.execute(
            f"INSERT INTO results_old "
            f"(testcase, matcher, model, backend, found, reasoning,"
            f" elapsed, response_json, run_at, archived_at) "
            f"SELECT testcase, matcher, model, backend, found, reasoning,"
            f" elapsed, response_json, run_at, ? "
            f"FROM results WHERE {where}",
            [datetime.now().isoformat(), *params],
        )
        archived = conn.execute(
            f"DELETE FROM results WHERE {where}", params,
        ).rowcount
        conn.commit()
        if archived:
            print(f"Archived {archived} cached results (--force)")

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
    print(f"\nReport written to {args.output}")
    print(f"Results saved to {db_path}")


if __name__ == "__main__":
    main()
