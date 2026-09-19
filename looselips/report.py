"""Render scan results to a self-contained HTML report."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from .scanner import ScanResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _highlight(context: str, matched_text: str) -> Markup:
    """Wrap matched_text in a highlight span (inputs are escaped first)."""
    # Must escape BEFORE wrapping in Markup -- Markup() trusts its contents.
    # Bandit/semgrep flag Markup() as potential XSS; safe here because
    # escape() runs first on both inputs.
    safe_ctx = escape(context)
    safe_match = escape(matched_text)
    return Markup(  # noqa: S704  # nosec B704
        safe_ctx.replace(
            safe_match,
            Markup('<span class="hl">') + safe_match + Markup("</span>"),
            1,
        )
    )


def generate_html(
    result: ScanResult, input_name: str = "scan", scanned: int | None = None
) -> str:
    """Generate a self-contained HTML report.

    *scanned* marks a partial report written mid-scan: how many of
    result.total conversations have been processed so far.
    """
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default=True),
    )
    env.filters["highlight"] = _highlight
    template = env.get_template("report.html.j2")

    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total_matches = sum(len(r.matches) for r in result.flagged)
    # A conversation whose LLM call failed was not fully scanned: list it
    # and never count it as clean.  Deduplicated, since each failed matcher
    # records its own error.
    errored = list({id(e.conversation): e.conversation for e in result.errors}.values())
    not_clean = {id(c) for c in errored} | {id(r.conversation) for r in result.flagged}

    return template.render(
        input_name=input_name,
        now=now_str,
        total=result.total,
        flagged_count=len(result.flagged),
        # Mid-scan, conversations not scanned yet are not clean either.
        clean=(result.total if scanned is None else scanned) - len(not_clean),
        total_matches=total_matches,
        conversations=result.flagged,
        errored=errored,
        scanned=scanned,
    )


def write_report(
    result: ScanResult,
    path: str | Path,
    input_name: str = "scan",
    scanned: int | None = None,
) -> None:
    """Write the HTML report atomically (temp file + rename).

    The CLI rewrites the report periodically during a long scan, so a
    crash or interrupt must never leave a half-written file behind.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(generate_html(result, input_name=input_name, scanned=scanned))
    os.replace(tmp, path)
