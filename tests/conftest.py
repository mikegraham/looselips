"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

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


class FakeOllama:
    """Minimal Ollama HTTP server that records request bodies.

    Answers /api/generate and /api/chat with a fixed LLMVerdict-shaped JSON
    so the real instructor + litellm stack can be exercised end to end.
    """

    RESPONSE_JSON = '{"reasoning": "User is Bob", "found": true}'

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- http.server API
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/generate":
                    reply: dict[str, Any] = {"response": fake.RESPONSE_JSON}
                elif self.path == "/api/chat":
                    reply = {
                        "message": {"role": "assistant", "content": fake.RESPONSE_JSON},
                    }
                else:
                    # litellm also probes /api/show for model info; those
                    # are not completions, so leave them out of .requests.
                    self.send_response(404)
                    self.end_headers()
                    return
                fake.requests.append(body)
                reply.update({
                    "model": body.get("model", ""),
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                })
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOllama]:
    """Run a FakeOllama and point litellm's Ollama provider at it."""
    fake = FakeOllama()
    monkeypatch.setenv("OLLAMA_API_BASE", fake.url)
    try:
        yield fake
    finally:
        fake.close()
