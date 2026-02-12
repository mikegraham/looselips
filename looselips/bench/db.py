from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class CachedResult(TypedDict):
    found: bool
    reasoning: str
    elapsed: float


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results_old (
            testcase      TEXT NOT NULL,
            matcher       TEXT NOT NULL,
            model         TEXT NOT NULL,
            backend       TEXT,
            found         INTEGER NOT NULL,
            reasoning     TEXT NOT NULL DEFAULT '',
            elapsed       REAL NOT NULL DEFAULT 0,
            response_json TEXT,
            run_at        TEXT NOT NULL,
            archived_at   TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def load_cached(
    conn: sqlite3.Connection, model: str,
) -> dict[str, dict[str, CachedResult]]:
    """Load cached results for a model.

    Returns {testcase_name: {matcher_name: {found, reasoning, elapsed}}}.
    """
    rows = conn.execute(
        "SELECT testcase, matcher, found, reasoning, elapsed "
        "FROM results WHERE model = ?",
        (model,),
    ).fetchall()
    cached: dict[str, dict[str, CachedResult]] = {}
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
