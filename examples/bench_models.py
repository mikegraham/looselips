"""Quick benchmark: which small models can produce valid structured output?

Usage: .venv/bin/python examples/bench_models.py
Requires ollama running with models already pulled.
"""

from __future__ import annotations

import time

import instructor
import litellm
from litellm import completion

from looselips.matchers import LLM_BASE_PROMPT, LLMParseError, LLMVerdict, llm_scan

litellm.telemetry = False

MATCHER_PROMPT = """Identify personal facts about the user.
Look for: names, addresses, employers, financial info, medical info."""

TEXT = (
    "[USER]: My name is John Smith and I live at 123 Oak Street. "
    "My SSN is 123-45-6789 and I work at Sun Microsystems making $185k.\n\n"
    "[ASSISTANT]: I can help you with that."
)

MODELS = [
    "ollama/qwen2.5:0.5b",
    "ollama/qwen3:0.6b",
    "ollama/tinyllama:1.1b",
    "ollama/qwen2.5:1.5b",
    "ollama/llama3.2:1b",
]


FULL_PROMPT = LLM_BASE_PROMPT.format(instructions=MATCHER_PROMPT)


def bench(model: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print("=" * 60)

    # Raw verdict (bypass llm_scan to see exactly what the model returns)
    client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)
    t0 = time.time()
    try:
        verdict = client.chat.completions.create(
            model=model,
            response_model=LLMVerdict,
            messages=[
                {"role": "system", "content": FULL_PROMPT},
                {"role": "user", "content": f"Conversation title: Test\n\n{TEXT}"},
            ],
            max_tokens=2000,
            temperature=0.1,
            timeout=300,
            max_retries=1,
        )
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        print(f"Verdict: {verdict}")
        print(f"Flagged: {verdict.found}")
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        print(f"FAILED: {e}")


def main() -> None:
    print("=" * 60)
    print("FULL SYSTEM PROMPT")
    print("=" * 60)
    print(FULL_PROMPT)
    print()
    print("=" * 60)
    print("SCHEMA (LLMVerdict)")
    print("=" * 60)
    print(LLMVerdict.model_json_schema())
    print()

    for model in MODELS:
        bench(model)
    print()


if __name__ == "__main__":
    main()
