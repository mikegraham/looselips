"""Regex and LLM matching against conversation text.

regex_scan() runs compiled patterns against a string and returns Match objects.
llm_scan() sends conversation text to a model via instructor/litellm and parses
a flagged/evidence verdict back into Match objects.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

# Ugh -- litellm phones home on import to fetch a model cost map. Looks
# sketchy for a tool meant to be local-only.
# LITELLM_LOCAL_MODEL_COST_MAP=True uses the bundled fallback instead.
import instructor
import litellm
import litellm.exceptions
from instructor.core import InstructorRetryException
from litellm import completion
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

litellm.telemetry = False

SNIPPET_MARGIN = 80
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.1
LLM_DEFAULT_TIMEOUT = 300
LLM_DEFAULT_RETRIES = 3

LLM_BASE_PROMPT = """\
You are a SCANNER. You have one job: decide whether a conversation \
matches a specific search task. You are NOT a chatbot -- do not engage \
with the conversation content, answer questions in it, or comment on it.

SEARCH TASK:
{instructions}

RULES:
- Scan both the user's messages AND the assistant's replies.
- Check implicit info too: things in code, configs, URLs, file paths, \
or logs count.
- When in doubt, report it. Offhand remarks and passing references count.
- Stay on task. Your reasoning must be about whether the conversation \
matches the search task -- nothing else. Do not analyze, summarize, \
or comment on unrelated aspects of the conversation.

EXAMPLES (these use a DIFFERENT task to show the format -- \
do NOT look for animals, apply YOUR search task above):

  Task: "Find mentions of animals"
  Conversation: "I took my dog Max to the vet yesterday."
  Output: {{"reasoning": "Mentions a dog named Max", "found": true}}

  Task: "Find mentions of animals"
  Conversation: "I'm learning Python and my Jaguar needs new brakes"
  Output: {{"reasoning": "Python and Jaguar refer to a language and a car, not animals", "found": false}}

  Task: "Find mentions of animals"
  Conversation: "Can you help me practice Spanish? 'El gato esta en la mesa.'"
  Output: {{"reasoning": "'El gato' means 'the cat'", "found": true}}

Now scan the conversation below.

SEARCH TASK:
{instructions}"""


@dataclass
class Match:
    """A single finding from a regex or LLM matcher."""

    category: str
    matched_text: str
    context: str
    source: str  # "regex" or "llm"


def _snippet(text: str, start: int, end: int, margin: int = SNIPPET_MARGIN) -> str:
    """Extract a window of *margin* chars around a match region."""
    s = max(0, start - margin)
    e = min(len(text), end + margin)
    ctx = text[s:e]
    if s > 0:
        ctx = "..." + ctx
    if e < len(text):
        ctx = ctx + "..."
    return ctx


def regex_scan(
    text: str, patterns: Sequence[tuple[str, re.Pattern[str]]]
) -> list[Match]:
    """Run every (category, compiled_regex) pair against *text*."""
    matches: list[Match] = []
    for category, pattern in patterns:
        for m in pattern.finditer(text):
            logger.debug("regex hit: %s %r", category, m.group()[:80])
            matches.append(
                Match(
                    category=category,
                    matched_text=m.group(),
                    context=_snippet(text, m.start(), m.end()),
                    source="regex",
                )
            )
    return matches


class LLMVerdict(BaseModel):
    """Scan result."""

    reasoning: str = Field(
        description="What matched the search task, or why nothing matched. Stay on task -- only discuss relevance to the search.",
    )
    found: bool = Field(
        description="true if you found anything matching the task, false if not",
    )


class LLMParseError(Exception):
    """Raised when the LLM fails to return valid structured output after retries."""


def llm_scan(
    title: str,
    messages_text: str,
    model: str,
    name: str = "llm",
    system_prompt: str = "",
    timeout: int = LLM_DEFAULT_TIMEOUT,
    retries: int = LLM_DEFAULT_RETRIES,
) -> list[Match]:
    """Send conversation text to an LLM and return matches.

    The LLM returns a simple flagged/evidence verdict.  If flagged,
    a single Match is returned with *name* as the category and
    the LLM's evidence as matched_text.
    """
    logger.debug("[%s] querying %s (%d chars)", name, model, len(messages_text))
    client = instructor.from_litellm(completion, mode=instructor.Mode.JSON_SCHEMA)

    instructions = system_prompt or "Report all findings."
    full_prompt = LLM_BASE_PROMPT.format(instructions=instructions)

    user_msg = f"Conversation title: {title}\n\n{messages_text}"

    try:
        verdict = client.chat.completions.create(
            model=model,
            response_model=LLMVerdict,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            timeout=timeout,
            max_retries=retries,
        )
    except (
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.APIError,
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.BadRequestError,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.RateLimitError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.Timeout,
        InstructorRetryException,
    ) as e:
        raise LLMParseError(f"LLM failed for conversation '{title}': {e}") from e

    logger.debug("[%s] raw: %s", name, verdict.model_dump_json())

    if not verdict.found:
        logger.debug("[%s] NO MATCH -- %s", name, verdict.reasoning[:200])
        return []

    logger.info("[%s] MATCH -- %s", name, verdict.reasoning[:200])
    return [
        Match(
            category=name,
            matched_text=verdict.reasoning,
            context=verdict.reasoning,
            source="llm",
        )
    ]
