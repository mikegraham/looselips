"""Tests for looselips.cli.app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from looselips.cli.app import main


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
