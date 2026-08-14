# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Generation strategies, selected per request by ``context["strategy"]``.

Each strategy turns a natural-language question into an OpenSearch ``_search``
body; the agent (``agent.py``) owns the shared plumbing (cluster client, model,
credentials, fallback) and dispatches here. To add one — e.g. search-template
fill — write its module and register it in ``STRATEGIES``; nothing else changes.
"""

from __future__ import annotations

from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.strategies.multi_template import (
    MultiTemplateShapeA2Strategy,
    MultiTemplateShapeA3Strategy,
    MultiTemplateShapeA4Strategy,
    MultiTemplateShapeA5Strategy,
    MultiTemplateShapeAStrategy,
    MultiTemplateShapeBStrategy,
    MultiTemplateTwoCallStrategy,
)
from agents.agentic_search.strategies.multi_template_fill import (
    MultiTemplateFillStrategy,
)
from agents.agentic_search.strategies.template_fill import TemplateFillStrategy

# Registry keyed by strategy name. `direct_dsl` is the default when a request omits
# `context.strategy`. The template strategies fill a search template's params instead of
# authoring DSL; the agent picks between them from how many distinct template ids a
# request carries (see AgenticSearchAgent._select_strategy), and both degrade to
# `direct_dsl`.
STRATEGIES = {
    DirectDslStrategy.name: DirectDslStrategy(),
    TemplateFillStrategy.name: TemplateFillStrategy(),
    MultiTemplateFillStrategy.name: MultiTemplateFillStrategy(),
    # Benchmark-only prototypes comparing multi-template call structures. Never selected
    # automatically; reachable only when a request names one via `context.strategy`.
    # This registration exists on the benchmark branch only.
    MultiTemplateTwoCallStrategy.name: MultiTemplateTwoCallStrategy(),
    MultiTemplateShapeAStrategy.name: MultiTemplateShapeAStrategy(),
    MultiTemplateShapeA2Strategy.name: MultiTemplateShapeA2Strategy(),
    MultiTemplateShapeA3Strategy.name: MultiTemplateShapeA3Strategy(),
    MultiTemplateShapeA4Strategy.name: MultiTemplateShapeA4Strategy(),
    MultiTemplateShapeA5Strategy.name: MultiTemplateShapeA5Strategy(),
    MultiTemplateShapeBStrategy.name: MultiTemplateShapeBStrategy(),
}
DEFAULT_STRATEGY = DirectDslStrategy.name
# One template to fill: the focused fill prompt, which fills more precisely than the
# combined pick-and-fill call.
SINGLE_TEMPLATE_STRATEGY = TemplateFillStrategy.name
# Several candidates: one call chooses among them and fills the winner.
MULTI_TEMPLATE_STRATEGY = MultiTemplateFillStrategy.name
