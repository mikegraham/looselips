"""Render scan results to a self-contained HTML report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from .scanner import ScanResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _highlight(context: str, matched_text: str) -> Markup:
    """Wrap matched_text in a highlight span (inputs are escaped first)."""
    safe_ctx = escape(context)
    safe_match = escape(matched_text)
    return Markup(  # noqa: S704  # nosec B704 -- inputs are escaped above
        safe_ctx.replace(
            safe_match,
            Markup('<span class="hl">') + safe_match + Markup("</span>"),
            1,
        )
    )


def generate_html(result: ScanResult, input_name: str = "scan") -> str:
    """Generate a self-contained HTML report."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default=True),
    )
    env.filters["highlight"] = _highlight
    template = env.get_template("report.html.j2")

    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total_matches = sum(len(r.matches) for r in result.flagged)

    return template.render(
        input_name=input_name,
        now=now_str,
        total=result.total,
        flagged_count=len(result.flagged),
        clean=result.total - len(result.flagged),
        total_matches=total_matches,
        conversations=result.flagged,
    )


def write_report(
    result: ScanResult, path: str | Path, input_name: str = "scan"
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(generate_html(result, input_name=input_name))
