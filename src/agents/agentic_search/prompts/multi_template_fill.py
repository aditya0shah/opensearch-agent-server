"""Prompt for multi-template fill: choose one of several templates and fill it.

Used only when a request carries **more than one** candidate template. A single
candidate goes through the ordinary single-template fill prompt instead, which
measurably fills more precisely: asking one call to both choose and fill costs a few
points of fill accuracy (wrong field scoping, over-filled text slots, sloppier date
math), so the combined prompt is reserved for requests that actually need a choice.

The expressibility check comes first on purpose. When the choice is asked for first the
model commits to a template and then fills it no matter what, so a question needing a
capability no template has (an unsupported projection, aggregation, or ranking signal)
gets a confidently wrong query instead of a fallback. Deciding expressibility first
restores the abstention.

Wording here is load-bearing and was tuned by measurement: a *required* abstain decision
and a "when unsure, abstain" framing both over-abstained and lost accuracy. Re-measure
before editing.
"""

from __future__ import annotations

from strands.types.content import SystemContentBlock

MULTI_FILL_MODEL_NAME = "SelectAndFillTemplate"

# Sentinel choice meaning "none of the candidates fits this question".
NONE_CHOICE = "none"

MULTI_FILL_SYSTEM_PROMPT = (
    "You answer a user's search question using one of several predefined OpenSearch "
    "search templates.\n"
    "Call the tool exactly once. Work in this order:\n"
    "1. First decide whether any candidate template can express the question. Set "
    "cannot_express=true when none can, and stop — do not fill parameters.\n"
    "2. Only if one can, choose that template and fill its parameters.\n"
    "Abstaining is the correct answer whenever the question needs a field, filter, "
    "projection, ranking signal, aggregation, count-only answer, exact phrase, "
    "prefix/wildcard/fuzzy match, or similarity that no candidate's parameters cover. "
    "Abstaining routes the question to a more capable path, so a near-miss fill is worse "
    "than declining.\n"
    "When you do fill: fill only the parameters the question clearly implies and leave "
    "the rest unset — do not guess. Put ONLY content/topic words in a free-text query "
    "parameter — never counts, filters, sort terms, or field names. For enum parameters, "
    "choose only from the options that parameter allows. Each parameter name is prefixed "
    "with the id of the template it belongs to; fill only parameters carrying the prefix "
    "of the template you chose."
)


def build_system_blocks(system_prompt: str) -> list[SystemContentBlock]:
    """Wrap a system prompt in content blocks with a trailing cache point.

    The candidate set is usually stable per caller, so the tool schema and system
    prefix are served from the prompt cache on warm calls.
    """
    return [{"text": system_prompt}, {"cachePoint": {"type": "default"}}]


MULTI_FILL_SYSTEM_BLOCKS: list[SystemContentBlock] = build_system_blocks(
    MULTI_FILL_SYSTEM_PROMPT
)

MULTI_FILL_USER_PROMPT = """\
Question: {question}

Candidate templates:
{candidates}

Choose the best template and fill its parameters for this question.
"""


def format_candidates(candidates: list[tuple[str, str]]) -> str:
    """Render ``(template_id, description)`` pairs as a list for the prompt."""
    return "\n".join(
        f"- {tid}: {desc or '(no description)'}" for tid, desc in candidates
    )
