# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for the tool-per-template multi-template fill strategy.

The Bedrock tool-choice call is monkeypatched, so these tests exercise routing,
candidate resolution, tool construction, selection handling, and the abstain/fallback
contract without a live model or cluster.
"""

from __future__ import annotations

import pytest

from agents.agentic_search.strategies import multi_template_fill as M
from agents.agentic_search.strategies.base import GenerationRequest
from agents.agentic_search.strategies.multi_template_fill import (
    FREE_DSL_TOOL_NAME,
    MAX_CANDIDATES,
    MultiTemplateFillStrategy,
    _sanitize_tool_name,
    _tool_names_for,
)
from agents.agentic_search.template_schema import TemplateSchema, build_fill_model

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_forced_choice():
    """Restore the module-level tool-choice function that tests monkeypatch in place."""
    original = M.forced_tool_choice_any
    yield
    M.forced_tool_choice_any = original


# --------------------------------------------------------------------------- fakes


def make_schema(tid, param_schema, *, index_binding=None, description="a template"):
    return TemplateSchema(
        template_id=tid,
        index_binding=index_binding,
        param_schema=param_schema,
        fill_model=build_fill_model(param_schema),
        description=description,
    )


class FakeCache:
    """Stands in for TemplateSchemaCache; raises for unknown ids like the real one."""

    def __init__(self, schemas):
        self._by_id = {s.template_id: s for s in schemas}

    def get(self, template_id, client):
        if template_id not in self._by_id:
            raise ValueError(f"no schema for {template_id}")
        return self._by_id[template_id]


class _ClientProxy:
    def converse_stream(self, **kwargs):  # pragma: no cover - never called in tests
        raise AssertionError("forced_tool_choice_any should be monkeypatched")


class FakeModel:
    """A Bedrock-like model: has a client exposing converse_stream + a config."""

    def __init__(self):
        self.client = _ClientProxy()
        self.config = {"model_id": "test-model"}


class NonBedrockModel:
    """No converse_stream client -> supports_forced_tool is False."""

    def __init__(self):
        self.client = object()
        self.config = {}


class FakeIndices:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_mapping(self, index):
        return self._mapping


class FakeClient:
    def __init__(self, template_output=None, mapping=None):
        self._output = template_output or {"query": {"match_all": {}}}
        self.rendered = []
        self.indices = FakeIndices(mapping or {"m": {}})

    def render_search_template(self, id, body):
        self.rendered.append((id, body.get("params")))
        return {"template_output": self._output}


class RecordingStrategy:
    """Records that it was invoked; returns a sentinel body."""

    def __init__(self, tag):
        self._tag = tag
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return {"_via": self._tag}


def make_request(context, *, model=None, client=None, mapping="", index="idx-1"):
    return GenerationRequest(
        question="find something",
        index_name=index,
        mapping=mapping,
        context=context,
        model=model or FakeModel(),
        client=client or FakeClient(),
    )


def make_strategy(
    schemas, *, single=None, fallback=None, max_candidates=MAX_CANDIDATES
):
    return MultiTemplateFillStrategy(
        single=single or RecordingStrategy("single"),
        fallback=fallback or RecordingStrategy("fallback"),
        schema_cache=FakeCache(schemas),
        max_candidates=max_candidates,
    )


# ------------------------------------------------------------------ tool naming


def test_sanitize_replaces_invalid_chars():
    assert _sanitize_tool_name("cat.a log v5") == "cat_a_log_v5"


def test_sanitize_preserves_dash_and_underscore():
    assert _sanitize_tool_name("catalog_v5-search") == "catalog_v5-search"


def test_sanitize_caps_length():
    assert len(_sanitize_tool_name("x" * 200)) == 64


def test_sanitize_empty_falls_back():
    assert _sanitize_tool_name("!!!") == "___"  # not empty; each char maps to _


def test_tool_names_dash_vs_underscore_stay_distinct():
    names = _tool_names_for(["a-b", "a_b"])
    assert names["a-b"] != names["a_b"]
    assert len(set(names.values())) == 2


def test_tool_names_disambiguate_true_collision():
    # Both sanitize to "a_b"; the second gets a positional suffix.
    names = _tool_names_for(["a.b", "a b"])
    assert names["a.b"] != names["a b"]


def test_tool_names_reserve_free_dsl_sentinel():
    # A template literally named the sentinel must not shadow the fallback tool.
    names = _tool_names_for([FREE_DSL_TOOL_NAME])
    assert names[FREE_DSL_TOOL_NAME] != FREE_DSL_TOOL_NAME


# ------------------------------------------------------------ candidate handling


def test_candidate_ids_dedup_and_scalar():
    ctx = {"template_ids": ["a", "a", "b"], "template_id": "b"}
    assert MultiTemplateFillStrategy._candidate_ids(ctx) == ["a", "b"]


def test_candidate_ids_scalar_only():
    assert MultiTemplateFillStrategy._candidate_ids({"template_id": "x"}) == ["x"]


def test_no_candidates_falls_back():
    fb = RecordingStrategy("fallback")
    strat = make_strategy([], fallback=fb)
    out = strat.generate(make_request({"template_ids": []}))
    assert out == {"_via": "fallback"}
    assert len(fb.calls) == 1


def test_single_candidate_delegates_to_single():
    single = RecordingStrategy("single")
    strat = make_strategy(
        [make_schema("only", {"q": {"type": "string"}}, index_binding="idx-1")],
        single=single,
    )
    out = strat.generate(make_request({"template_ids": ["only"]}))
    assert out == {"_via": "single"}
    # Delegation passes the scalar template_id and drops the list.
    ctx = single.calls[0].context
    assert ctx["template_id"] == "only"
    assert "template_ids" not in ctx


def test_index_binding_filters_cross_index():
    # Two candidates bound to different indices; only the matching one survives, so it
    # delegates to the single path rather than doing a multi pick.
    single = RecordingStrategy("single")
    strat = make_strategy(
        [
            make_schema("here", {"q": {"type": "string"}}, index_binding="idx-1"),
            make_schema("there", {"q": {"type": "string"}}, index_binding="idx-2"),
        ],
        single=single,
    )
    strat.generate(make_request({"template_ids": ["here", "there"]}, index="idx-1"))
    assert single.calls[0].context["template_id"] == "here"


def test_cap_stops_resolving():
    schemas = [
        make_schema(f"t{i}", {"q": {"type": "string"}}, index_binding="idx-1")
        for i in range(5)
    ]
    strat = make_strategy(schemas, max_candidates=2)
    seen = []

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        seen.append(tool_specs)
        return tool_specs[0]["name"], {"q": "x"}

    M.forced_tool_choice_any = fake_choice
    strat.generate(make_request({"template_ids": [s.template_id for s in schemas]}))
    # 2 template tools + 1 fallback tool.
    assert len(seen[0]) == 3


def test_unresolvable_candidate_skipped():
    # "ghost" is not in the cache; the two real ones remain -> multi pick.
    schemas = [
        make_schema("a", {"q": {"type": "string"}}, index_binding="idx-1"),
        make_schema("b", {"q": {"type": "string"}}, index_binding="idx-1"),
    ]
    strat = make_strategy(schemas)
    captured = {}

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        captured["names"] = [t["name"] for t in tool_specs]
        return tool_specs[0]["name"], {"q": "x"}

    M.forced_tool_choice_any = fake_choice
    strat.generate(make_request({"template_ids": ["a", "ghost", "b"]}))
    # a, b, and the fallback tool -> ghost dropped.
    assert set(captured["names"]) == {"a", "b", FREE_DSL_TOOL_NAME}


# --------------------------------------------------------------- tool building


def test_build_tools_has_fallback_and_one_per_candidate():
    strat = make_strategy([])
    cands = [
        make_schema("a", {"q": {"type": "string"}}),
        make_schema("b", {"n": {"type": "integer"}}),
    ]
    specs, choice_map = strat._build_tools(cands)
    names = [s["name"] for s in specs]
    assert FREE_DSL_TOOL_NAME in names
    assert "a" in names and "b" in names
    assert len(specs) == 3
    assert set(choice_map) == {"a", "b"}


def test_build_tools_schema_carries_required_field():
    strat = make_strategy([])
    cands = [make_schema("a", {"q": {"type": "string", "required": True}})]
    specs, _ = strat._build_tools(cands)
    tool = next(s for s in specs if s["name"] == "a")
    schema = tool["inputSchema"]["json"]
    assert "q" in schema.get("required", [])


def test_build_tools_fallback_tool_takes_no_args():
    strat = make_strategy([])
    specs, _ = strat._build_tools([make_schema("a", {"q": {"type": "string"}})])
    fb = next(s for s in specs if s["name"] == FREE_DSL_TOOL_NAME)
    assert fb["inputSchema"]["json"].get("properties") == {}


# ------------------------------------------------------------ select-and-fill


def _two_same_index():
    return [
        make_schema(
            "full",
            {"q": {"type": "string"}, "price": {"type": "integer"}},
            index_binding="idx-1",
        ),
        make_schema("facet", {"price": {"type": "integer"}}, index_binding="idx-1"),
    ]


def test_select_renders_chosen_template():
    client = FakeClient(template_output={"query": {"term": {"a": 1}}})
    strat = make_strategy(_two_same_index())

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return "full", {"q": "boots", "price": 100}

    M.forced_tool_choice_any = fake_choice
    out = strat.generate(
        make_request({"template_ids": ["full", "facet"]}, client=client)
    )
    assert out == {"query": {"term": {"a": 1}}}
    rendered_id, params = client.rendered[0]
    assert rendered_id == "full"
    assert params == {"q": "boots", "price": 100}


def test_select_drops_unset_optional_params():
    client = FakeClient()
    strat = make_strategy(_two_same_index())

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return "full", {"q": "boots"}  # price left unset

    M.forced_tool_choice_any = fake_choice
    strat.generate(make_request({"template_ids": ["full", "facet"]}, client=client))
    _, params = client.rendered[0]
    assert params == {"q": "boots"}


def test_select_ignores_hallucinated_key():
    client = FakeClient()
    strat = make_strategy(_two_same_index())

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return "facet", {"price": 50, "not_a_param": "x"}

    M.forced_tool_choice_any = fake_choice
    strat.generate(make_request({"template_ids": ["full", "facet"]}, client=client))
    rendered_id, params = client.rendered[0]
    assert rendered_id == "facet"
    assert params == {"price": 50}


def test_free_dsl_tool_abstains_to_fallback():
    fb = RecordingStrategy("fallback")
    strat = make_strategy(_two_same_index(), fallback=fb)

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return FREE_DSL_TOOL_NAME, {}

    M.forced_tool_choice_any = fake_choice
    out = strat.generate(make_request({"template_ids": ["full", "facet"]}))
    assert out == {"_via": "fallback"}
    assert len(fb.calls) == 1


def test_unknown_tool_name_falls_back():
    fb = RecordingStrategy("fallback")
    strat = make_strategy(_two_same_index(), fallback=fb)

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return "nonexistent_tool", {}

    M.forced_tool_choice_any = fake_choice
    out = strat.generate(make_request({"template_ids": ["full", "facet"]}))
    assert out == {"_via": "fallback"}


def test_non_bedrock_provider_falls_back():
    fb = RecordingStrategy("fallback")
    strat = make_strategy(_two_same_index(), fallback=fb)

    def fake_choice(
        *, model, tool_specs, system_blocks, user_message
    ):  # pragma: no cover
        raise AssertionError("should not reach the model on a non-Bedrock provider")

    M.forced_tool_choice_any = fake_choice
    out = strat.generate(
        make_request({"template_ids": ["full", "facet"]}, model=NonBedrockModel())
    )
    assert out == {"_via": "fallback"}


def test_fallback_fetches_mapping_when_absent():
    fb = RecordingStrategy("fallback")
    client = FakeClient(mapping={"idx-1": {"mappings": {}}})
    strat = make_strategy([], fallback=fb)  # no candidates -> straight to fallback
    strat.generate(make_request({"template_ids": []}, client=client, mapping=""))
    # The fallback request carries a fetched mapping.
    assert fb.calls[0].mapping != ""


# ------------------------------------------------------------------ rendering


def test_render_unwraps_string_output():
    client = FakeClient(template_output='{"query": {"match_all": {}}}')
    strat = make_strategy(_two_same_index())

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return "full", {"q": "x"}

    M.forced_tool_choice_any = fake_choice
    out = strat.generate(
        make_request({"template_ids": ["full", "facet"]}, client=client)
    )
    assert out == {"query": {"match_all": {}}}


def test_render_bad_output_falls_back():
    fb = RecordingStrategy("fallback")

    class BadClient(FakeClient):
        def render_search_template(self, id, body):
            return {"template_output": None}

    strat = make_strategy(_two_same_index(), fallback=fb)

    def fake_choice(*, model, tool_specs, system_blocks, user_message):
        return "full", {"q": "x"}

    M.forced_tool_choice_any = fake_choice
    out = strat.generate(
        make_request({"template_ids": ["full", "facet"]}, client=BadClient())
    )
    assert out == {"_via": "fallback"}
