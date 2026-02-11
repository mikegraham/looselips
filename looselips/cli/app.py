"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path

import argcomplete

from looselips.input import InputError, load_conversations
from looselips.report import write_report
from looselips.scanner import scan

from .config import ConfigError, build_llm_matchers, build_regex_patterns, load_config

logger = logging.getLogger(__name__)


def _complete_model(**kwargs: object) -> list[str]:
    """List locally available ollama models for tab completion."""
    try:
        import ollama
    except ImportError:
        return []

    try:
        return [f"ollama/{m.model}" for m in ollama.list().models]
    except Exception:  # noqa: BLE001
        logger.debug("Ollama not reachable for tab completion", exc_info=True)
        return []


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scan LLM chat exports for personal information.",
    )
    parser.add_argument(
        "input",
        help="Path to ChatGPT or Claude export (.json or .zip)",
    )
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "-c", "--config", default=None, help="Path to looselips.toml config file"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for DEBUG, -vv for litellm debug too)",
    )
    model_arg = parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="LiteLLM model for LLM-based scanning (e.g. ollama/llama3.2)",
    )
    model_arg.completer = _complete_model  # type: ignore[attr-defined]
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose == 1:
        logging.basicConfig(level=logging.DEBUG)
        # Quiet down litellm/httpx unless -vv
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    patterns: list[tuple[str, re.Pattern[str]]] = []
    llm_matchers: list[tuple[str, str, str | None]] = []
    llm_model: str | None = args.model

    if args.config:
        logger.debug("Loading config from %s", args.config)
        try:
            config = load_config(args.config)
        except ConfigError as e:
            parser.error(str(e))

        patterns.extend(build_regex_patterns(config))
        llm_matchers = build_llm_matchers(config)
        llm_model = args.model or config.default_model
        logger.debug("Config: %d regex, %d llm matchers, default_model=%s",
                     len(patterns), len(llm_matchers), llm_model)

    input_path: str = args.input

    # Derive input name (for report title) and default output path
    input_stem = Path(input_path).stem
    output_path = args.output or f"{input_stem}_report.html"

    logger.info("Loading %s...", input_path)
    try:
        conversations = load_conversations(input_path)
    except InputError as e:
        parser.error(str(e))

    logger.info("  %d conversations", len(conversations))

    logger.info("Scanning...")
    t0 = time.time()
    result = scan(
        conversations,
        patterns=patterns,
        llm_model=llm_model,
        llm_matchers=llm_matchers or None,
    )
    elapsed = time.time() - t0

    logger.info("  %d/%d flagged (%.1fs)", len(result.flagged), result.total, elapsed)

    write_report(result, output_path, input_name=input_stem)
    logger.info("Report: %s", output_path)
    # TODO: optionally open the report in the user's browser (webbrowser.open)?


if __name__ == "__main__":
    main()
