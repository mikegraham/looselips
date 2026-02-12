from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import escape

from looselips.bench.report import BenchReport

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


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
