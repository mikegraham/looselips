"""Tests for looselips.cli.app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from looselips.cli.app import main
from looselips.report import write_report
from looselips.scanner import scan


def _write_export(tmp_path: Path, data: list[dict[str, Any]] | None = None) -> str:
    p = tmp_path / "conversations.json"
    if data is None:
        data = [
            {
                "id": "conv-1",
                "title": "Test Chat",
                "mapping": {
                    "root": {"parent": None, "message": None},
                    "msg": {
                        "parent": "root",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["My email is test@example.com"]},
                        },
                    },
                },
            }
        ]
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def _write_config(tmp_path: Path, content: str) -> str:
    p = tmp_path / "looselips.toml"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_basic_run(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    output = str(tmp_path / "report.html")
    main([export, "-o", output])
    assert Path(output).exists()


def test_run_with_config(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    config = _write_config(
        tmp_path,
        """
[[matcher]]
type = "regex"
category = "Custom"
pattern = 'test@example'
""",
    )
    output = str(tmp_path / "report.html")
    main([export, "-c", config, "-o", output])
    html = Path(output).read_text()
    assert "Custom" in html



def test_no_input_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_config_error_exits(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    config = _write_config(tmp_path, '[[matcher]]\ntype = "bad"\n')
    with pytest.raises(SystemExit):
        main([export, "-c", config])


def test_file_not_found_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope.json")])


def test_verbose_flag(tmp_path: Path) -> None:
    """Single -v sets DEBUG on root but WARNING on litellm/httpx."""
    import logging

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    export = _write_export(tmp_path)
    output = str(tmp_path / "report.html")
    main([export, "-o", output, "-v"])
    assert root.level == logging.DEBUG
    assert logging.getLogger("LiteLLM").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_very_verbose_flag(tmp_path: Path) -> None:
    """-vv sets DEBUG globally without quieting litellm."""
    import logging

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.NOTSET)

    export = _write_export(tmp_path)
    output = str(tmp_path / "report.html")
    main([export, "-o", output, "-vv"])
    assert root.level == logging.DEBUG
    # Unlike -v, -vv should NOT quiet LiteLLM
    assert logging.getLogger("LiteLLM").level != logging.WARNING


def test_report_is_written_during_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the interval at 0, every conversation checkpoints a partial report.

    Regression test: results used to exist only in memory until the very
    end, so a crash at the final write lost the whole scan.
    """
    import looselips.cli.app as app

    monkeypatch.setattr(app, "REPORT_INTERVAL", 0)
    export = _write_export(tmp_path)
    output = tmp_path / "report.html"
    calls: list[int | None] = []
    real = write_report

    def spy(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs.get("scanned"))
        real(*args, **kwargs)

    monkeypatch.setattr(app, "write_report", spy)
    main([export, "-o", str(output)])
    assert calls[-1] is None  # final, complete report
    assert len(calls) >= 2
    assert all(c is not None for c in calls[:-1])
    assert output.exists()


def test_jobs_flag(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    output = str(tmp_path / "report.html")
    with patch("looselips.cli.app.scan", wraps=scan) as spy:
        main([export, "-o", output])
        assert spy.call_args.kwargs["jobs"] == 1
        main([export, "-o", output, "--jobs", "5"])
        assert spy.call_args.kwargs["jobs"] == 5


def test_jobs_flag_rejects_zero(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    with pytest.raises(SystemExit):
        main([export, "-o", str(tmp_path / "r.html"), "--jobs", "0"])
