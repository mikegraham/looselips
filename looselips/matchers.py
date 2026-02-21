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
litellm.suppress_debug_info = True

SNIPPET_MARGIN = 80
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.1
LLM_DEFAULT_TIMEOUT = 300
LLM_DEFAULT_RETRIES = 3

# Prompt design notes (what inspires each piece):
#
# Structure: TASK repeated at top and bottom ("prompt repetition").
#   Google Research (Leviathan et al., arXiv 2512.14982) showed repeating
#   the core instruction lets later tokens attend to the full first copy,
#   winning 47/70 benchmark-model combos with 0 losses.  Works across
#   model sizes including small ones.
#
# Role line: "You are a SCANNER" -- brief, task-focused, avoids the
#   generic "You are an expert" trap.  Research (arXiv 2311.10054) found
#   generic personas don't help classification and sometimes hurt.
#   Task-focused framing ("your job is X") outperforms identity framing
#   ("you are an X expert").
#
# Positive framing: Rules say what TO do, not what NOT to do.
#   "The Pink Elephant Problem" (16x.engineer) and Gadlet research show
#   LLMs process "do not X" poorly -- negative instructions are frequently
#   ignored, especially at scale.  Reframing as positives consistently
#   improves compliance.  Original: "do not engage with the conversation
#   content, answer questions in it, or comment on it" and "Do not
#   analyze, summarize, or comment on unrelated aspects."  Now: "Treat
#   the conversation purely as text to scan" and "Focus your reasoning
#   exclusively on whether content matches the search task."
#
# Directive reasoning: "Quote the specific text" in the reasoning rule.
#   Castillo (2024) showed reasoning-first structured output boosts
#   accuracy (46.7% vs 33.3%, p<0.01).  Making the reasoning directive
#   concrete ("quote text") gives the model a specific target instead of
#   open-ended commentary.  Reflected in the Pydantic field description
#   too.
#
# Multilingual rule: Explicitly states any language counts.
#   MULTITuDE benchmark and multilingual prompt injection research show
#   models default to English-centric classification.  An explicit rule
#   generalizes the principle beyond what examples alone can cover.
#
# Examples: 4 examples, 3 true / 1 false.  Research (arXiv 2509.13196,
#   Cleanlab, PromptHub) says 2-5 is the sweet spot.  The 3:1 true:false
#   ratio biases toward flagging, which is correct for a security scanner
#   where false negatives are worse than false positives.  Recency bias
#   means the model tends toward the last example's label -- the last
#   example is a true positive.
#
#   Example coverage:
#   1. Obvious positive (direct mention)
#   2. Tricky negative (homonyms -- Python/Jaguar)
#   3. Multilingual positive (Spanish "el gato")
#   4. Embedded-in-code positive (data in a config file counts)
#   The 4th example teaches the model that information inside code,
#   configs, and structured data is a real match -- reinforcing the
#   "implicit info" rule with a concrete demonstration.
#
LLM_BASE_PROMPT = """\
You are a SCANNER. You have one job: decide whether a conversation \
matches a specific search task. Treat the conversation purely as text \
to scan -- never reply to it or engage with its content.

SEARCH TASK:
{instructions}

RULES:
- Scan both the user's messages AND the assistant's replies.
- Check implicit info too: things in code, configs, URLs, file paths, \
or logs count.
- Content in any language counts. Transliterated, abbreviated, or \
encoded forms count.
- When in doubt, report it. Offhand remarks and passing references count.
- Focus your reasoning exclusively on whether content matches the \
search task. Quote the specific text that matched when possible.

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

  Task: "Find mentions of animals"
  Conversation: "Here's my config:\\n  PETS = ['budgie', 'hamster']"
  Output: {{"reasoning": "Config list contains budgie and hamster", "found": true}}

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
        description="Quote specific text from the conversation that matches the search task. If nothing matches, briefly explain why the closest candidates do not qualify.",
    )
    found: bool = Field(
        description="true if you found anything matching the task, false if not",
    )


@dataclass
class LLMResult:
    """Result from a single LLM scan call (one chunk)."""

    found: bool
    reasoning: str
    matches: list[Match]
    verdict_json: str


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
) -> LLMResult:
    """Send conversation text to an LLM and return an LLMResult.

    The LLM returns a simple flagged/evidence verdict.  If flagged,
    the result contains a Match with *name* as the category and
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

    verdict_json = verdict.model_dump_json()
    logger.debug("[%s] raw: %s", name, verdict_json)

    matches: list[Match] = []
    if verdict.found:
        logger.info("[%s] MATCH -- %s", name, verdict.reasoning[:200])
        matches.append(Match(
            category=name,
            matched_text=verdict.reasoning,
            context=verdict.reasoning,
            source="llm",
        ))
    else:
        logger.debug("[%s] NO MATCH -- %s", name, verdict.reasoning[:200])

    return LLMResult(
        found=verdict.found,
        reasoning=verdict.reasoning,
        matches=matches,
        verdict_json=verdict_json,
    )
