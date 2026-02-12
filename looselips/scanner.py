"""Run regex and LLM matchers across a list of conversations."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from .matchers import LLMParseError, Match, llm_scan, regex_scan
from .parsers import Conversation

logger = logging.getLogger(__name__)

# TODO: infer from model context window (ollama /api/show, litellm.get_model_info)
LLM_CHUNK_CHARS = 24000


@dataclass
class MatcherResult:
    """Per-matcher result for a single conversation."""

    name: str
    found: bool
    matches: list[Match]
    reasoning: str
    elapsed: float
    error: str | None = None


@dataclass
class ConversationResult:
    conversation: Conversation
    matches: list[Match] = field(default_factory=list)

    @property
    def has_matches(self) -> bool:
        return len(self.matches) > 0


@dataclass
class ConversationError:
    """A conversation where one or more LLM matchers failed."""

    conversation: Conversation
    matcher: str
    error: str


@dataclass
class ScanResult:
    total: int
    flagged: list[ConversationResult]
    errors: list[ConversationError] = field(default_factory=list)


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


def scan_conversation_llm(
    conv: Conversation,
    matchers: Sequence[tuple[str, str, str]],
) -> list[MatcherResult]:
    """Run LLM matchers against a single conversation.

    Each matcher is a (name, system_prompt, model) tuple.  Returns one
    MatcherResult per matcher with found/reasoning/elapsed/error fields.
    """
    chunks = _chunk_conversation(conv)
    results: list[MatcherResult] = []

    for name, system_prompt, model in matchers:
        t0 = time.monotonic()
        found = False
        all_matches: list[Match] = []
        reasoning_parts: list[str] = []
        error: str | None = None

        try:
            for chunk in chunks:
                hits = llm_scan(
                    conv.title, chunk, model,
                    name=name, system_prompt=system_prompt,
                )
                if hits:
                    found = True
                    all_matches.extend(hits)
                    reasoning_parts.extend(m.matched_text for m in hits)
        except LLMParseError as e:
            error = str(e)

        elapsed = time.monotonic() - t0
        results.append(MatcherResult(
            name=name,
            found=found,
            matches=all_matches,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else "(no match)",
            elapsed=elapsed,
            error=error,
        ))

    return results


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

    if not patterns and not llm_matchers:
        logger.warning("no matchers configured -- nothing to scan")

    effective_llm: list[tuple[str, str, str]] = []
    if llm_matchers:
        for name, prompt, model_override in llm_matchers:
            model = model_override or llm_model
            if not model:
                raise ValueError(f"LLM matcher {name!r} has no model")
            effective_llm.append((name, prompt, model))

    if effective_llm:
        logger.debug("scan: %d LLM matchers active", len(effective_llm))
        for name, _, model in effective_llm:
            logger.debug("  matcher %r -> %s", name, model)

    flagged = []
    errors: list[ConversationError] = []

    total = len(conversations)
    for i, conv in enumerate(conversations, 1):
        if not conv.messages:
            logger.debug("(%d/%d) conversation %r has 0 messages, skipping",
                         i, total, conv.title)

        matches: list[Match] = []

        if patterns and conv.messages:
            full_text = "\n\n".join(m.text for m in conv.messages)
            matches.extend(regex_scan(full_text, patterns))

        if effective_llm and conv.messages:
            logger.info("(%d/%d) %r (%d messages)",
                         i, total, conv.title, len(conv.messages))
            for mr in scan_conversation_llm(conv, effective_llm):
                matches.extend(mr.matches)
                if mr.error:
                    logger.error("[%s] FAILED on %r: %s",
                                 mr.name, conv.title, mr.error)
                    errors.append(ConversationError(
                        conversation=conv, matcher=mr.name, error=mr.error,
                    ))

        if matches:
            flagged.append(ConversationResult(conversation=conv, matches=matches))

    return ScanResult(total=len(conversations), flagged=flagged, errors=errors)
