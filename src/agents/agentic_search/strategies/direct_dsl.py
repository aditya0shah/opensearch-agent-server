"""Direct-DSL generation: the model authors the whole ``_search`` body.

The default (and today's only) generation strategy. It hands the LLM the index
mapping and prompts it, via a single forced ``EmitSearch`` tool call, to write a
complete OpenSearch query body. Future strategies (e.g. search-template fill)
are sibling modules registered alongside this one.
"""

from __future__ import annotations

import logging
from typing import Any

from strands import Agent

from agents.agentic_search.prompts import (
    SYSTEM_BLOCKS,
    USER_PROMPT,
    EmitSearch,
)
from agents.agentic_search.strategies.base import GenerationRequest

logger = logging.getLogger(__name__)


class DirectDslStrategy:
    """Generate a ``_search`` body directly from the NLQ and index mapping."""

    name = "direct_dsl"

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        """Return the OpenSearch ``_search`` body as a dict for the NLQ.

        A fresh ``Agent`` per call keeps each request stateless; the shared
        ``request.model`` carries the cost-bearing connection. The system prompt
        is passed as content blocks so its cache point reaches the request, and
        the non-deprecated ``structured_output_model`` invocation is required —
        the older ``structured_output()`` flattens the prompt to a string and
        drops the cache point.
        """
        agent = Agent(
            model=request.model,
            system_prompt=SYSTEM_BLOCKS,
            tools=[],
            callback_handler=None,
        )
        user_msg = USER_PROMPT.format(
            question=request.question,
            index_name=request.index_name,
            mapping=request.mapping,
        )
        result = agent(user_msg, structured_output_model=EmitSearch).structured_output
        logger.info(
            "Generated DSL for index=%s (reason=%s)", request.index_name, result.reason
        )
        return result.dsl
