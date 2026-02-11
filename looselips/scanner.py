"""Run regex and LLM matchers across a list of conversations."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .matchers import LLMParseError, Match, llm_scan, regex_scan
from .parsers import Conversation

logger = logging.getLogger(__name__)

LLM_CHUNK_CHARS = 6000


@dataclass
class ConversationResult:
    conversation: Conversation
    matches: list[Match] = field(default_factory=list)

    @property
    def has_matches(self) -> bool:
        return len(self.matches) > 0


@dataclass
class ScanResult:
    total: int
    flagged: list[ConversationResult]


def _format_messages(conv: Conversation) -> list[str]:
    """Format each message as ``[ROLE]: text``."""
    return [f"[{m.role.upper()}]: {m.text}" for m in conv.messages]


def _chunk_conversation(
    conv: Conversation, max_chars: int = LLM_CHUNK_CHARS
) -> list[str]:
    """Split a conversation into chunks that fit within *max_chars*.

    Splits at message boundaries so no single message is cut in half.
    A message longer than *max_chars* gets its own chunk.

    .. todo:: Smarter chunking with overlap for context continuity.
    """
    messages = _format_messages(conv)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for msg in messages:
        msg_len = len(msg)
        # Would adding this message exceed the limit?
        if current and current_len + msg_len + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(msg)
        current_len += msg_len + 2  # +2 for "\n\n" separator

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def scan(
    conversations: Sequence[Conversation],
    patterns: Sequence[tuple[str, re.Pattern[str]]],
    llm_model: str | None = None,
    llm_matchers: Sequence[tuple[str, str, str | None]] | None = None,
) -> ScanResult:
    """Scan conversations and return results.

    Parameters
    ----------
    conversations : sequence of Conversation
        Parsed conversations to scan.
    patterns : sequence of (category, compiled_regex)
        Regex matchers to run against each message.
    llm_model : str or None
        Default LiteLLM model string for LLM matchers.
    llm_matchers : sequence of (name, system_prompt, model_override) or None
        Explicit LLM matchers from config.  Falls back to the built-in
        system prompt when *llm_model* is set but no matchers are given.
    """
    logger.debug("scan: %d conversations, %d regex patterns, llm_model=%s",
                 len(conversations), len(patterns), llm_model)

    effective_llm: list[tuple[str, str, str]] = []
    if llm_matchers:
        for name, prompt, model_override in llm_matchers:
            model = model_override or llm_model
            if model:
                effective_llm.append((name, prompt, model))
            else:
                logger.warning("LLM matcher has no model, skipping: %s", name)

    if effective_llm:
        logger.debug("scan: %d LLM matchers active", len(effective_llm))
        for name, _, model in effective_llm:
            logger.debug("  matcher %r -> %s", name, model)

    flagged = []

    for conv in conversations:
        matches: list[Match] = []

        if patterns and conv.messages:
            full_text = "\n\n".join(m.text for m in conv.messages)
            matches.extend(regex_scan(full_text, patterns))

        if effective_llm and conv.messages:
            chunks = _chunk_conversation(conv)
            logger.debug("conv %r: %d messages, %d chunks",
                         conv.title, len(conv.messages), len(chunks))
            for name, system_prompt, model in effective_llm:
                for chunk in chunks:
                    try:
                        matches.extend(
                            llm_scan(
                                conv.title,
                                chunk,
                                model,
                                name=name,
                                system_prompt=system_prompt,
                            )
                        )
                    except LLMParseError as e:
                        logger.warning("%s", e)

        if matches:
            flagged.append(ConversationResult(conversation=conv, matches=matches))

    return ScanResult(total=len(conversations), flagged=flagged)
