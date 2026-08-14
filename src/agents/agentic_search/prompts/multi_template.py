"""Prompts for the multi-template (candidate-set) generation prototypes.

Three designs share these prompts so a benchmark can attribute differences to the
*mechanism* (two calls vs one combined call) rather than to prompt wording:

- ``SELECT_*``  the routing-only call used by the two-call design.
- ``COMBINED_*`` the single pick-and-fill call used by both single-call designs.
"""

from __future__ import annotations

from strands.types.content import SystemContentBlock

SELECT_MODEL_NAME = "SelectTemplate"
COMBINED_MODEL_NAME = "SelectAndFillTemplate"

# Kept in sync with template_schema.CANNOT_EXPRESS_FIELD; named here so the prompt text
# below can reference the field without importing the schema module.
CANNOT_EXPRESS_FIELD_NAME = "cannot_express"

# Sentinel choice meaning "no candidate can express this question".
NONE_CHOICE = "none"

_ABSTAIN_RULE = (
    "Choose {none} when no candidate can express the question: the question needs a "
    "field, filter, ranking signal, aggregation, count-only answer, phrase/prefix match, "
    "or similarity capability that no candidate covers. Do not force a near-miss."
)

SELECT_SYSTEM_PROMPT = (
    "You route a user's search question to the best-fitting OpenSearch search template.\n"
    "Call the SelectTemplate tool exactly once with the id of the single best candidate.\n"
    "Match on the question's domain and intent against each candidate's description.\n"
    + _ABSTAIN_RULE.format(none=NONE_CHOICE)
)

COMBINED_SYSTEM_PROMPT = (
    "You answer a user's search question by choosing one OpenSearch search template and "
    "filling its parameters.\n"
    "Call the tool exactly once. First choose the template whose description matches the "
    "question's domain and intent, then fill only that template's parameters.\n"
    "Fill only the parameters the question clearly implies; leave everything else unset — "
    "do not guess. Put ONLY content/topic words in any free-text query parameter — never "
    "counts, filters, sort terms, or field names. For enum parameters, choose only from the "
    "options that parameter allows.\n"
    "Do not fill parameters belonging to a template you did not choose.\n"
    + _ABSTAIN_RULE.format(none=NONE_CHOICE)
)


def build_system_blocks(system_prompt: str) -> list[SystemContentBlock]:
    """Wrap a system prompt in content blocks with a trailing cache point."""
    return [{"text": system_prompt}, {"cachePoint": {"type": "default"}}]


# Variant that leads with the expressibility check instead of the choice. The default
# combined prompt tells the model to choose a template first, which appears to commit it:
# a combined pick-and-fill call almost never abstains, while the two-call design's
# dedicated routing call abstains readily. This wording makes declining the first
# decision rather than an afterthought.
COMBINED_ABSTAIN_FIRST_PROMPT = (
    "You answer a user's search question using one of several predefined OpenSearch "
    "search templates.\n"
    "Call the tool exactly once. Work in this order:\n"
    "1. FIRST decide whether any candidate template can express the question. Set "
    f"{CANNOT_EXPRESS_FIELD_NAME}=true when it cannot, and stop — do not fill parameters.\n"
    "2. Only if it can, choose the matching template and fill its parameters.\n"
    "Step 1 is not optional. Abstaining is the correct answer whenever the question needs "
    "a field, filter, projection, ranking signal, aggregation, count-only answer, exact "
    "phrase, prefix/wildcard/fuzzy match, or similarity that no candidate's parameters "
    "cover. Abstaining routes the question to a more capable path, so a near-miss fill is "
    "strictly worse than declining.\n"
    "When you do fill: fill only the parameters the question clearly implies and leave the "
    "rest unset. Put ONLY content/topic words in a free-text query parameter — never "
    "counts, filters, sort terms, or field names. For enum parameters, choose only from "
    "the options that parameter allows. Do not fill parameters belonging to a template you "
    "did not choose."
)

SELECT_SYSTEM_BLOCKS: list[SystemContentBlock] = build_system_blocks(SELECT_SYSTEM_PROMPT)
COMBINED_SYSTEM_BLOCKS: list[SystemContentBlock] = build_system_blocks(COMBINED_SYSTEM_PROMPT)
COMBINED_ABSTAIN_FIRST_BLOCKS: list[SystemContentBlock] = build_system_blocks(
    COMBINED_ABSTAIN_FIRST_PROMPT
)

SELECT_USER_PROMPT = """\
Question: {question}

Candidate templates:
{candidates}

Choose the single best template id for this question.
"""

COMBINED_USER_PROMPT = """\
Question: {question}

Candidate templates:
{candidates}

Choose the best template and fill its parameters for this question.
"""


def format_candidates(candidates: list[tuple[str, str]]) -> str:
    """Render ``(template_id, description)`` pairs as a numbered list."""
    lines = []
    for tid, desc in candidates:
        lines.append(f"- {tid}: {desc or '(no description)'}")
    return "\n".join(lines)
