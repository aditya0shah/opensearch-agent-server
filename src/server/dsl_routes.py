"""Route handler for ``POST /generate_dsl``.

Unlike ``POST /runs`` (which streams AG-UI events), this endpoint returns a
single JSON response so an in-cluster ml-commons connector can consume it. The
response is an ``inference_results`` envelope with the DSL as a string in
``output[0].result``, which ml-commons' passthrough post-process function maps
to ``ModelTensor.result``. The generator is injected via
:func:`set_dsl_generator`.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from fastapi import Request
from pydantic import BaseModel, Field

from server.types import GenerateDslResponse

logger = logging.getLogger(__name__)


class GenerateDslInput(BaseModel):
    """Request body for ``POST /generate_dsl``."""

    question: str = Field(min_length=1, description="The natural-language query.")
    index_name: str = Field(min_length=1, description="Target index for DSL generation.")


# Returned when generation fails so the search degrades to matching everything
# rather than erroring.
FALLBACK_DSL = '{"size":10,"query":{"match_all":{}}}'


class DslGenerator(Protocol):
    """Pluggable engine that returns an OpenSearch ``_search`` body as a JSON string."""

    def generate(
        self, question: str, index_name: str, auth_token: str | None = None
    ) -> str: ...


_generator: DslGenerator | None = None


def set_dsl_generator(generator: DslGenerator) -> None:
    """Register the engine that backs ``POST /generate_dsl`` (call at startup)."""
    global _generator
    _generator = generator


def _wrap_inference_results(dsl_query: str) -> GenerateDslResponse:
    """Wrap a DSL string in the ml-commons inference_results passthrough envelope."""
    return {
        "inference_results": [
            {
                "output": [{"name": "response", "result": dsl_query}],
                "status_code": 200,
            }
        ]
    }


async def generate_dsl_route(
    *, request: Request, input_data: GenerateDslInput
) -> GenerateDslResponse:
    """Generate DSL for an NLQ and return it in the passthrough envelope.

    On any generation failure the fallback ``match_all`` DSL is returned with a
    200 so the upstream search degrades instead of erroring.
    """
    if _generator is None:
        logger.error("generate_dsl called but no DslGenerator is registered")
        return _wrap_inference_results(FALLBACK_DSL)

    # Forward the caller's bearer token so the generator can reach the cluster.
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    token = auth[len("bearer ") :] if auth and auth.lower().startswith("bearer ") else None

    try:
        dsl_query = _generator.generate(
            input_data.question, input_data.index_name, auth_token=token
        )
        # Fall back unless the generated DSL is a JSON object (a _search body).
        # A bare primitive/array (e.g. "ok", []) parses but is not valid DSL.
        if not isinstance(json.loads(dsl_query), dict):
            raise ValueError("generated DSL is not a JSON object")
    except Exception:  # noqa: BLE001 - any generation failure degrades to fallback
        logger.exception(
            "DSL generation failed for index=%s; returning fallback", input_data.index_name
        )
        return _wrap_inference_results(FALLBACK_DSL)

    return _wrap_inference_results(dsl_query)
