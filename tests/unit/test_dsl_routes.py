"""Unit tests for the non-streaming NLQ->DSL endpoint (POST /generate_dsl).

Verifies, with the generator mocked (no cluster, no LLM):
1. A successful generation is wrapped in the ml-commons inference_results
   envelope with the DSL string at output[0].result (so the connector
   passthrough lands it in ModelTensor.result).
2. A generator exception degrades to the fallback match_all DSL (still 200).
3. Non-JSON generator output is treated as failure -> fallback.
4. When no generator is registered, the fallback is returned.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from server import dsl_routes
from server.dsl_routes import (
    FALLBACK_DSL,
    GenerateDslInput,
    generate_dsl_route,
    set_dsl_generator,
)

pytestmark = pytest.mark.unit


def _request_with_auth(token: str | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {token}"} if token else {}
    return request


def _result_string(response: dict) -> str:
    return response["inference_results"][0]["output"][0]["result"]


@pytest.fixture(autouse=True)
def _reset_generator():
    """Ensure each test starts with no registered generator."""
    dsl_routes._generator = None
    yield
    dsl_routes._generator = None


@pytest.mark.asyncio
async def test_success_wraps_dsl_in_passthrough_envelope():
    dsl = '{"query":{"term":{"status":"active"}}}'
    gen = MagicMock()
    gen.generate.return_value = dsl
    set_dsl_generator(gen)

    resp = await generate_dsl_route(
        request=_request_with_auth("svc-token"),
        input_data=GenerateDslInput(question="active items", index_name="idx"),
    )

    # DSL must be a STRING at output[0].result, not a nested object.
    assert _result_string(resp) == dsl
    assert isinstance(_result_string(resp), str)
    assert resp["inference_results"][0]["status_code"] == 200
    # Bearer token forwarded to the generator for cluster access.
    gen.generate.assert_called_once_with("active items", "idx", auth_token="svc-token")


@pytest.mark.asyncio
async def test_generator_exception_falls_back():
    gen = MagicMock()
    gen.generate.side_effect = RuntimeError("bedrock unavailable")
    set_dsl_generator(gen)

    resp = await generate_dsl_route(
        request=_request_with_auth(),
        input_data=GenerateDslInput(question="x", index_name="idx"),
    )

    assert _result_string(resp) == FALLBACK_DSL
    assert json.loads(_result_string(resp)) == {"size": 10, "query": {"match_all": {}}}


@pytest.mark.asyncio
async def test_non_json_output_falls_back():
    gen = MagicMock()
    gen.generate.return_value = "not valid json"
    set_dsl_generator(gen)

    resp = await generate_dsl_route(
        request=_request_with_auth(),
        input_data=GenerateDslInput(question="x", index_name="idx"),
    )

    assert _result_string(resp) == FALLBACK_DSL


@pytest.mark.asyncio
@pytest.mark.parametrize("non_object", ['"ok"', "[]", "42", "true", "null"])
async def test_non_object_json_falls_back(non_object):
    # Valid JSON that isn't an object is not a usable _search body -> fallback.
    gen = MagicMock()
    gen.generate.return_value = non_object
    set_dsl_generator(gen)

    resp = await generate_dsl_route(
        request=_request_with_auth(),
        input_data=GenerateDslInput(question="x", index_name="idx"),
    )

    assert _result_string(resp) == FALLBACK_DSL


@pytest.mark.asyncio
async def test_no_generator_registered_falls_back():
    resp = await generate_dsl_route(
        request=_request_with_auth(),
        input_data=GenerateDslInput(question="x", index_name="idx"),
    )

    assert _result_string(resp) == FALLBACK_DSL


def test_input_requires_nonempty_fields():
    with pytest.raises(ValueError):
        GenerateDslInput(question="", index_name="idx")
    with pytest.raises(ValueError):
        GenerateDslInput(question="x", index_name="")
