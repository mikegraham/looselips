"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        default=None,
        help="LiteLLM model string for integration tests (e.g. ollama/qwen2.5:0.5b)",
    )


@pytest.fixture(scope="module")
def model(request: pytest.FixtureRequest) -> str:
    """LLM model for integration tests. Skips if --model not provided."""
    m: str | None = request.config.getoption("--model")
    if m is None:
        pytest.skip("no --model provided")
    return m
