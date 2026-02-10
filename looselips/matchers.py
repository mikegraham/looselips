"""Regex and LLM matching against conversation text.

regex_scan() runs compiled patterns against a string and returns Match objects.
llm_scan() sends conversation text to a model via instructor/litellm and parses
a flagged/remarks verdict back into Match objects.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import instructor
import litellm
import litellm.exceptions
from instructor.core import InstructorRetryException
from litellm import completion
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

litellm.telemetry = False
# TODO: expose litellm settings (callbacks, logging, etc.) to users

SNIPPET_MARGIN = 80
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.0
LLM_DEFAULT_TIMEOUT = 300
LLM_DEFAULT_RETRIES = 2

LLM_BASE_PROMPT = """\
You are scanning a conversation for specific information. \
Read every message carefully and report what you find.

YOUR TASK:
{instructions}

GUIDELINES:
- Check both the user's messages AND the assistant's replies.
- Look for implicit info too: things embedded in code, configs, URLs, \
file paths, or logs count.
- Report specific details, not vague references. When in doubt, report it.
- Findings can appear anywhere -- offhand remarks count.

OUTPUT FORMAT:
Return a JSON object with "found" (true/false) and "remarks" (string).
If you found something: {{"found": true, "remarks": "what you found"}}
If you found nothing: {{"found": false, "remarks": "why nothing matched"}}

EXAMPLES (these use a DIFFERENT task to show the format -- \
do NOT look for animals, apply YOUR TASK above):

  Task: "Find mentions of animals"
  Conversation: "I took my dog Max to the vet yesterday."
  Correct output: {{"found": true, "remarks": "Dog named Max"}}

  Task: "Find mentions of animals"
  Conversation: "The server logs show systemd errors"
  Correct output: {{"found": false, "remarks": "No mentions of animals"}}

  Task: "Find mentions of animals"
  Conversation: "Can you write me a Python loop?"
  Correct output: {{"found": false, "remarks": "\"Python\" mentioned, but it refers to the programming language, not the animal"}}

Now scan the conversation below. Apply YOUR TASK:
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

    found: bool = Field(
        description="true if you found anything matching the task, false if not",
    )
    remarks: str = Field(
        default="",
        description="What you found, or why you found nothing",
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

    The LLM returns a simple flagged/remarks verdict.  If flagged,
    a single Match is returned with *name* as the category and
    the LLM's remarks as matched_text.
    """
    logger.debug("llm_scan: model=%s name=%r title=%r len=%d",
                 model, name, title, len(messages_text))
    client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)

    full_prompt = LLM_BASE_PROMPT.format(instructions=system_prompt or "Report all findings.")

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

    logger.debug("llm_scan: title=%r remarks=%r",
                 title, verdict.remarks[:120] if verdict.remarks else "")

    if not verdict.found:
        return []

    return [
        Match(
            category=name,
            matched_text=verdict.remarks,
            context=verdict.remarks,
            source="llm",
        )
    ]
