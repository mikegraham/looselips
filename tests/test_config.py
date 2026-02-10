"""Tests for looselips.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from looselips.cli.config import (
    ConfigError,
    build_llm_matchers,
    build_regex_patterns,
    load_config,
)


def _write(tmp_path: Path, content: str) -> str:
    p = tmp_path / "looselips.toml"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_load_empty_config(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    config = load_config(path)
    assert config.default_model is None
    assert config.matchers == []


def test_load_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path, 'model = "ollama/llama3"\n'
    )
    config = load_config(path)
    assert config.default_model == "ollama/llama3"



def test_load_regex_matcher(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[[matcher]]
type = "regex"
category = "Phone"
pattern = '212.?867.?5309'
ignore_case = true
""",
    )
    config = load_config(path)
    assert len(config.matchers) == 1
    m = config.matchers[0]
    assert m.type == "regex"
    assert m.category == "Phone"
    assert m.pattern == "212.?867.?5309"
    assert m.ignore_case is True


def test_load_llm_matcher(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[[matcher]]
type = "llm"
name = "illegal activity"
prompt = "Find illegal activity"
model = "gpt-4o-mini"
""",
    )
    config = load_config(path)
    assert len(config.matchers) == 1
    m = config.matchers[0]
    assert m.type == "llm"
    assert m.name == "illegal activity"
    assert m.prompt == "Find illegal activity"
    assert m.model == "gpt-4o-mini"


def test_load_multiple_matchers(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[[matcher]]
type = "regex"
category = "SSN"
pattern = '\\d{3}-\\d{2}-\\d{4}'

[[matcher]]
type = "llm"
name = "names"
prompt = "Find names"

[[matcher]]
type = "regex"
category = "Custom"
pattern = 'secret_word'
""",
    )
    config = load_config(path)
    assert len(config.matchers) == 3
    assert config.matchers[0].type == "regex"
    assert config.matchers[1].type == "llm"
    assert config.matchers[2].type == "regex"


def test_invalid_type(tmp_path: Path) -> None:
    path = _write(tmp_path, '[[matcher]]\ntype = "bad"\n')
    with pytest.raises(ConfigError, match="type must be"):
        load_config(path)


def test_regex_missing_pattern(tmp_path: Path) -> None:
    path = _write(tmp_path, '[[matcher]]\ntype = "regex"\ncategory = "X"\n')
    with pytest.raises(ConfigError, match="requires 'pattern'"):
        load_config(path)


def test_regex_missing_category(tmp_path: Path) -> None:
    path = _write(tmp_path, '[[matcher]]\ntype = "regex"\npattern = "x"\n')
    with pytest.raises(ConfigError, match="requires 'category'"):
        load_config(path)


def test_regex_invalid_pattern(tmp_path: Path) -> None:
    path = _write(
        tmp_path, '[[matcher]]\ntype = "regex"\ncategory = "X"\npattern = "[invalid"\n'
    )
    with pytest.raises(ConfigError, match="invalid regex"):
        load_config(path)


def test_llm_missing_name(tmp_path: Path) -> None:
    path = _write(tmp_path, '[[matcher]]\ntype = "llm"\nprompt = "Find stuff"\n')
    with pytest.raises(ConfigError, match="requires 'name'"):
        load_config(path)


def test_llm_missing_prompt(tmp_path: Path) -> None:
    path = _write(tmp_path, '[[matcher]]\ntype = "llm"\nname = "test"\n')
    with pytest.raises(ConfigError, match="requires 'prompt'"):
        load_config(path)


def test_unknown_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '[[matcher]]\ntype = "regex"\ncategory = "X"\npattern = "x"\ntypo_key = true\n',
    )
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)


def test_build_regex_patterns(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[[matcher]]
type = "regex"
category = "Phone"
pattern = '\\d{3}-\\d{4}'

[[matcher]]
type = "llm"
name = "ignored"
prompt = "ignored"

[[matcher]]
type = "regex"
category = "Custom"
pattern = 'foo'
ignore_case = true
""",
    )
    config = load_config(path)
    patterns = build_regex_patterns(config)
    assert len(patterns) == 2
    assert patterns[0][0] == "Phone"
    assert patterns[0][1].search("555-1234")
    assert patterns[1][0] == "Custom"
    assert patterns[1][1].search("FOO")  # ignore_case


def test_build_llm_matchers(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[[matcher]]
type = "regex"
category = "X"
pattern = "x"

[[matcher]]
type = "llm"
name = "names"
prompt = "Find names"

[[matcher]]
type = "llm"
name = "addresses"
prompt = "Find addresses"
model = "special-model"
""",
    )
    config = load_config(path)
    llm = build_llm_matchers(config)
    assert len(llm) == 2
    assert llm[0] == ("names", "Find names", None)
    assert llm[1] == ("addresses", "Find addresses", "special-model")
