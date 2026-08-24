"""Prompt for multi-template fill: choose one of several templates and fill it.

Used only when a request carries more than one candidate template; a single candidate
uses the ordinary single-template fill prompt instead.

Each candidate template is offered to the model as its own tool: the tool's name is the
template, its description is what the template searches, and its parameters are the
template's fields. Selection is therefore ordinary tool choice — the model calls the one
tool whose template best fits the question and fills only that template's parameters.

A dedicated free-DSL tool is offered alongside the templates so the model has an explicit
"none of these fit" option: a question needing a capability no template exposes (an
unsupported field, filter, projection, aggregation, count-only answer, exact phrase,
ranking signal, or similarity) is answered by calling that tool, which routes to the
free-DSL fallback rather than forcing a wrong template fill.
"""

from __future__ import annotations

from strands.types.content import SystemContentBlock

# Name of the fallback tool the model calls when no candidate template fits. Kept in the
# ``[a-zA-Z0-9_-]`` set Bedrock allows for tool names, and distinct from any template id.
FREE_DSL_TOOL_NAME = "use_free_dsl"

MULTI_FILL_SYSTEM_PROMPT = (
    "You answer a user's search question by calling exactly one tool.\n"
    "Each tool is a predefined OpenSearch search template: the tool's parameters are that "
    "template's fields. Call the one tool whose template best expresses the question and "
    "fill only the parameters the question clearly implies — leave the rest unset, do not "
    "guess. Put ONLY content/topic words in a free-text query parameter — never counts, "
    "filters, sort terms, or field names. For enum parameters, choose only from the "
    "options that parameter allows.\n"
    f"If NO template can express the question, call the {FREE_DSL_TOOL_NAME} tool "
    "instead. Do that whenever the question needs a field, filter, projection, ranking "
    "signal, aggregation, count-only answer, exact phrase, prefix/wildcard/fuzzy match, "
    "or similarity that no template's parameters cover. Calling it routes the question to "
    "a more capable path, so it is the correct answer for a near-miss — a wrong-template "
    "fill is worse than declining."
)


def build_system_blocks(system_prompt: str) -> list[SystemContentBlock]:
    """Wrap a system prompt in content blocks with a trailing cache point.

    The candidate set is usually stable per caller, so the tool schemas and system
    prefix are served from the prompt cache on warm calls.
    """
    return [{"text": system_prompt}, {"cachePoint": {"type": "default"}}]


MULTI_FILL_SYSTEM_BLOCKS: list[SystemContentBlock] = build_system_blocks(
    MULTI_FILL_SYSTEM_PROMPT
)

MULTI_FILL_USER_PROMPT = """\
Question: {question}

Call the tool for the best-fitting template and fill its parameters for this question. \
If no template fits, call {free_dsl_tool}.
"""
