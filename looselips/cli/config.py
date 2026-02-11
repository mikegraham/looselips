"""Parse looselips.toml config into matcher definitions."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MatcherDef:
    """A single matcher definition from config."""

    type: str  # "regex" or "llm"
    name: str = ""  # required for llm
    category: str = ""  # required for regex
    pattern: str = ""  # regex only
    prompt: str = ""  # llm only
    model: str | None = None  # llm only, overrides default


@dataclass
class Config:
    """Parsed config file."""

    default_model: str | None = None
    matchers: list[MatcherDef] = field(default_factory=list)


class ConfigError(Exception):
    """Raised for invalid config files."""


_VALID_MATCHER_KEYS = {"type", "category", "name", "pattern", "prompt", "model"}


def load_config(path: str | Path) -> Config:
    """Load and validate a TOML config file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = Config(
        default_model=raw.get("model"),
    )

    for i, m in enumerate(raw.get("matcher", [])):
        unknown = set(m.keys()) - _VALID_MATCHER_KEYS
        if unknown:
            raise ConfigError(f"matcher #{i + 1}: unknown keys: {unknown}")

        mtype = m.get("type", "")
        if mtype not in ("regex", "llm"):
            raise ConfigError(
                f"matcher #{i + 1}: type must be 'regex' or 'llm', got {mtype!r}"
            )

        if mtype == "regex":
            pattern = m.get("pattern", "")
            category = m.get("category", "")
            if not pattern:
                raise ConfigError(f"matcher #{i + 1}: regex matcher requires 'pattern'")
            if not category:
                raise ConfigError(
                    f"matcher #{i + 1}: regex matcher requires 'category'"
                )
            # Validate the regex compiles
            try:
                re.compile(pattern)
            except re.error as e:
                raise ConfigError(f"matcher #{i + 1}: invalid regex: {e}") from e

            config.matchers.append(
                MatcherDef(
                    type="regex",
                    category=category,
                    pattern=pattern,
                )
            )

        elif mtype == "llm":
            name = m.get("name", "")
            prompt = m.get("prompt", "")
            if not name:
                raise ConfigError(f"matcher #{i + 1}: llm matcher requires 'name'")
            if not prompt:
                raise ConfigError(f"matcher #{i + 1}: llm matcher requires 'prompt'")
            config.matchers.append(
                MatcherDef(
                    type="llm",
                    name=name,
                    prompt=prompt,
                    model=m.get("model"),
                )
            )

    return config


def build_regex_patterns(
    config: Config,
) -> list[tuple[str, re.Pattern[str]]]:
    """Compile regex matchers from config into (category, pattern) tuples."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for m in config.matchers:
        if m.type != "regex":
            continue
        patterns.append((m.category, re.compile(m.pattern)))
    return patterns


def build_llm_matchers(
    config: Config,
) -> list[tuple[str, str, str | None]]:
    """Extract LLM matchers as (name, prompt, model_override) tuples."""
    return [(m.name, m.prompt, m.model) for m in config.matchers if m.type == "llm"]
