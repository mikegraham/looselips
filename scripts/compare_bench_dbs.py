#!/usr/bin/env python3
"""Compare two bench DBs verdict-by-verdict.

Usage:
    python scripts/compare_bench_dbs.py OLD.db NEW.db [--model MODEL]

Prints per-model counts of both-right, both-wrong, old-only-right
(regressions), and new-only-right (improvements), plus the specific
testcases that flipped.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import yaml

TESTCASE_DIR = Path(__file__).resolve().parent.parent / "looselips/bench/testcases"


def load_expects(tc_dir: Path) -> dict[tuple[str, str], bool]:
    expects: dict[tuple[str, str], bool] = {}
    for p in tc_dir.glob("*.yaml"):
        with open(p) as f:
            tc = yaml.safe_load(f)
        for m, v in tc.get("expect", {}).items():
            expects[(p.stem, m)] = v
    return expects


def get_verdicts(conn: sqlite3.Connection, model: str) -> dict[tuple[str, str], bool]:
    rows = conn.execute(
        "SELECT testcase, matcher, found FROM results WHERE model = ?",
        (model,),
    ).fetchall()
    return {(tc, m): bool(f) for tc, m, f in rows}


def compare(
    old_conn: sqlite3.Connection,
    new_conn: sqlite3.Connection,
    model: str,
    expects: dict[tuple[str, str], bool],
) -> None:
    old_map = get_verdicts(old_conn, model)
    new_map = get_verdicts(new_conn, model)
    common = set(old_map) & set(new_map)

    if not common:
        print(f"  (no common results)")
        return

    both_right = 0
    both_wrong = 0
    regressions: list[tuple[str, str]] = []
    improvements: list[tuple[str, str]] = []

    for key in sorted(common):
        exp = expects.get(key)
        if exp is None:
            continue
        old_ok = old_map[key] == exp
        new_ok = new_map[key] == exp
        if old_ok and new_ok:
            both_right += 1
        elif not old_ok and not new_ok:
            both_wrong += 1
        elif old_ok:
            regressions.append(key)
        else:
            improvements.append(key)

    total = both_right + both_wrong + len(regressions) + len(improvements)
    net = len(improvements) - len(regressions)
    print(f"  {total} pairs: {both_right} both-right, {both_wrong} both-wrong, "
          f"{len(improvements)} improved, {len(regressions)} regressed (net {net:+d})")

    if regressions:
        print(f"  Regressions:")
        for tc, m in regressions:
            print(f"    {tc} / {m}")
    if improvements:
        print(f"  Improvements:")
        for tc, m in improvements:
            print(f"    {tc} / {m}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_db", type=Path, help="Baseline DB")
    ap.add_argument("new_db", type=Path, help="New DB to compare")
    ap.add_argument("--model", action="append", dest="models",
                    help="Only compare these models (repeatable)")
    ap.add_argument("--testcases", type=Path, default=TESTCASE_DIR,
                    help="Testcase directory")
    args = ap.parse_args()

    expects = load_expects(args.testcases)
    old_conn = sqlite3.connect(args.old_db)
    new_conn = sqlite3.connect(args.new_db)

    if args.models:
        models = args.models
    else:
        old_models = {r[0] for r in old_conn.execute("SELECT DISTINCT model FROM results")}
        new_models = {r[0] for r in new_conn.execute("SELECT DISTINCT model FROM results")}
        models = sorted(old_models & new_models)

    for model in models:
        short = model.split("/", 1)[-1] if "/" in model else model
        print(f"=== {short} ===")
        compare(old_conn, new_conn, model, expects)
        print()

    old_conn.close()
    new_conn.close()


if __name__ == "__main__":
    main()
