"""Multi-template prototypes: pick one of N candidate templates, then fill it.

Three designs for the same job, so a benchmark can attribute differences to the
*mechanism* rather than to prompt or schema wording:

``multi_two_call``
    Two LLM calls. The first routes on ``{template_id, description}`` only; the
    second is the ordinary single-template fill on the winner.

``multi_shape_a``
    One LLM call. Every candidate's parameters are merged into one flat object,
    namespaced by template id and all optional, alongside a required
    ``template_id`` choice. Required-ness is recovered by validating the chosen
    template's parameters after the call.

``multi_shape_b``
    One LLM call whose schema is a discriminated union: one object variant per
    candidate, each carrying only that template's parameters.

All three share :func:`_params_schema` for the per-template parameter block and
validate the chosen fill through the production ``build_fill_model``, so the only
difference between arms is call structure. Any failure degrades to free-DSL, as in
the single-template strategy.

Prototype code for measurement, not a shipping path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from agents.agentic_search.prompts.multi_template import (
    COMBINED_MODEL_NAME,
    COMBINED_ABSTAIN_FIRST_BLOCKS,
    COMBINED_SYSTEM_BLOCKS,
    COMBINED_USER_PROMPT,
    NONE_CHOICE,
    SELECT_MODEL_NAME,
    SELECT_SYSTEM_BLOCKS,
    SELECT_USER_PROMPT,
    format_candidates,
)
from agents.agentic_search.prompts.template_fill import (
    FILL_MODEL_NAME,
    FILL_SYSTEM_BLOCKS,
    FILL_USER_PROMPT,
)
from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.template_schema import (
    _CANNOT_EXPRESS_DESCRIPTION,
    CANNOT_EXPRESS_FIELD,
    TemplateSchema,
    TemplateSchemaCache,
    build_fill_model,
)

logger = logging.getLogger(__name__)

TEMPLATE_IDS_KEY = "template_ids"
TEMPLATE_ID_KEY = "template_id"
CHOICE_FIELD = "template_id"
SELECTION_FIELD = "selection"

# param-schema "type" -> JSON Schema type for the tool's input schema.
#
# ``array`` maps to string on purpose: the production ``build_fill_model`` has no
# array entry in its scalar map, so it surfaces a multi-value slot as a string and
# the model emits a JSON-array literal that the body renders raw through a triple
# brace. Matching that here keeps the per-param typing identical across every arm,
# so the comparison isolates call structure rather than introducing a second
# difference. (The missing array mapping is a production gap worth fixing
# separately.)
_JSON_TYPES = {
    "string": "string",
    "text": "string",
    "keyword": "string",
    "integer": "integer",
    "int": "integer",
    "long": "integer",
    "number": "number",
    "float": "number",
    "double": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "string",
}


class _NoTemplateSelected(Exception):
    """The model chose no candidate; route to free-DSL."""


def _trace(record: dict[str, Any]) -> None:
    """Append one JSON line of per-request telemetry when tracing is enabled.

    The benchmark reads this to recover which template was selected and the
    per-phase cost, neither of which the ``/invoke`` response exposes today.
    """
    path = os.environ.get("MT_TRACE")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - telemetry must never break a request
        logger.warning("multi-template trace write failed")


def _param_json_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON Schema for one param entry from the stored param-schema."""
    out: dict[str, Any] = {}
    desc = spec.get("description")
    if desc:
        out["description"] = desc
    if "enum" in spec and isinstance(spec["enum"], list) and spec["enum"]:
        out["enum"] = list(spec["enum"])
        out["type"] = "string"
        return out
    raw_type = str(spec.get("type", "string")).lower()
    out["type"] = _JSON_TYPES.get(raw_type, "string")
    if raw_type == "array":
        hint = "JSON array literal, e.g. [\"a\",\"b\"]."
        out["description"] = f"{out['description']} {hint}".strip() if out.get("description") else hint
    return out


def _params_schema(
    param_schema: dict[str, Any], *, prefix: str = "", force_optional: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(properties, required)`` for a template's params.

    Shared by every arm so the parameter half of each tool schema is identical and
    only the surrounding structure differs. ``prefix`` namespaces names for the flat
    merge; ``force_optional`` drops required-ness (recovered by post-validation).
    """
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in param_schema.items():
        if not isinstance(spec, dict):
            continue
        key = f"{prefix}{name}"
        props[key] = _param_json_schema(spec)
        if spec.get("required", False) and not force_optional:
            required.append(key)
    return props, required


def _safe_prefix(template_id: str) -> str:
    """Namespace prefix for a template's params in the flat merge."""
    return "".join(c if c.isalnum() else "_" for c in template_id) + "__"


def forced_call(
    *,
    model: Any,
    tool_spec: dict[str, Any],
    system_blocks: list[dict],
    user_message: str,
) -> tuple[dict[str, Any], dict[str, int], int]:
    """Force one tool call and return ``(tool_input, usage, elapsed_ms)``.

    Mirrors the production forced-tool path but returns the raw tool input (each
    arm validates differently) plus Bedrock's token usage, which the benchmark
    needs to price the larger single-call schemas.
    """
    tool_name = tool_spec["name"]
    kwargs: dict[str, Any] = {
        "modelId": model.config.get("model_id"),
        "system": [dict(b) for b in system_blocks],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": tool_spec.get("description", "Emit the result."),
                        "inputSchema": tool_spec["inputSchema"],
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        },
    }
    temperature = model.config.get("temperature")
    if temperature is not None:
        kwargs["inferenceConfig"] = {"temperature": temperature}

    t0 = time.time()
    resp = model.client.converse_stream(**kwargs)
    tool_input = ""
    usage: dict[str, int] = {}
    for event in resp["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {}).get("toolUse")
        if delta and "input" in delta:
            tool_input += delta["input"]
        meta = event.get("metadata")
        if meta and isinstance(meta.get("usage"), dict):
            # Keep every field Bedrock reports (including the cache counters), since
            # the point of the comparison is what the larger schemas actually cost.
            usage = dict(meta["usage"])
    elapsed_ms = int((time.time() - t0) * 1000)

    if not tool_input.strip():
        raise ValueError("forced tool call produced no input")
    parsed = json.loads(tool_input)
    if not isinstance(parsed, dict):
        raise ValueError("forced tool input is not an object")
    return parsed, usage, elapsed_ms


class _MultiTemplateBase:
    """Shared candidate resolution, validation, render, and fallback."""

    needs_mapping = False
    name = "multi_base"

    def __init__(
        self,
        *,
        fallback: GenerationStrategy | None = None,
        schema_cache: TemplateSchemaCache | None = None,
    ) -> None:
        self._fallback = fallback if fallback is not None else DirectDslStrategy()
        self._schema_cache = schema_cache if schema_cache is not None else TemplateSchemaCache()

    # ---- request plumbing -------------------------------------------------

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        ids = self._candidate_ids(request)
        rec: dict[str, Any] = {
            "arm": self.name,
            # Benchmark-supplied correlation id, so a run can join telemetry to its
            # question without depending on the NLQ text being unique.
            "qid": request.context.get("qid"),
            "nlq": request.question,
            "index": request.index_name,
            "n_candidates": len(ids),
        }
        t0 = time.time()
        try:
            if not ids:
                raise ValueError("no template_ids in context")
            candidates = self._resolve(ids, request.client)
            rec["n_resolved"] = len(candidates)
            if not candidates:
                raise ValueError("no candidate resolved")
            dsl = self._run(request, candidates, rec)
            rec["outcome"] = "template"
            rec["total_ms"] = int((time.time() - t0) * 1000)
            _trace(rec)
            return dsl
        except _NoTemplateSelected as e:
            rec["outcome"] = "abstain"
            rec["abstain_reason"] = str(e)
        except Exception as e:  # noqa: BLE001 - any failure degrades to free-DSL
            rec["outcome"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            logger.warning("multi-template %s failed (%s); falling back", self.name, e)
        dsl = self._fallback_generate(request)
        rec["total_ms"] = int((time.time() - t0) * 1000)
        _trace(rec)
        return dsl

    @staticmethod
    def _candidate_ids(request: GenerationRequest) -> list[str]:
        raw = request.context.get(TEMPLATE_IDS_KEY)
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            one = request.context.get(TEMPLATE_ID_KEY)
            raw = [one] if one else []
        return [str(x) for x in raw if x]

    def _resolve(self, ids: list[str], client: Any) -> list[TemplateSchema]:
        """Resolve candidates, dropping any that cannot be read.

        Unlike the single-template path, one bad id must not fail the request.
        """
        out: list[TemplateSchema] = []
        for tid in ids:
            try:
                out.append(self._schema_cache.get(tid, client))
            except Exception as e:  # noqa: BLE001 - skip unreadable candidate
                logger.warning("candidate %s unresolved (%s); skipping", tid, e)
        return out

    def _run(
        self,
        request: GenerationRequest,
        candidates: list[TemplateSchema],
        rec: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    # ---- shared post-processing ------------------------------------------

    @staticmethod
    def _pick(candidates: list[TemplateSchema], choice: str | None) -> TemplateSchema:
        if not choice or choice == NONE_CHOICE:
            raise _NoTemplateSelected("model chose none")
        for c in candidates:
            if c.template_id == choice:
                return c
        raise ValueError(f"model chose unknown template '{choice}'")

    @staticmethod
    def _validate(chosen: TemplateSchema, params: dict[str, Any]) -> dict[str, Any]:
        """Validate raw params against the template's real model; return clean params.

        Recovers required-ness and enum constraints that a flattened/all-optional
        schema cannot express, and drops unset optionals so the body's inverted
        sections still work.
        """
        model_cls = build_fill_model(chosen.param_schema, add_abstain=False)
        inst = model_cls.model_validate(params)
        return inst.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def _render(client: Any, template_id: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = client.render_search_template(id=template_id, body={"params": params})
        output = resp.get("template_output") if isinstance(resp, dict) else None
        if output is None:
            raise ValueError("_render/template returned no template_output")
        if isinstance(output, str):
            output = json.loads(output)
        if not isinstance(output, dict):
            raise ValueError("rendered template_output is not a _search body object")
        return output

    def _fill_and_render(
        self, request: GenerationRequest, chosen: TemplateSchema, params: dict[str, Any]
    ) -> dict[str, Any]:
        clean = self._validate(chosen, params)
        return self._render(request.client, chosen.template_id, clean)

    def _fallback_generate(self, request: GenerationRequest) -> dict[str, Any]:
        req = request
        if not request.mapping:
            try:
                mapping = json.dumps(request.client.indices.get_mapping(index=request.index_name))
                req = replace(request, mapping=mapping)
            except Exception as e:  # noqa: BLE001
                logger.warning("fallback mapping fetch failed (%s)", e)
        return self._fallback.generate(req)


class MultiTemplateTwoCallStrategy(_MultiTemplateBase):
    """Route on descriptions, then fill the winner. Two LLM calls."""

    name = "multi_two_call"

    def _run(self, request, candidates, rec):
        ids = [c.template_id for c in candidates]
        select_spec = {
            "name": SELECT_MODEL_NAME,
            "description": "Choose the best-fitting search template for the question.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        CHOICE_FIELD: {
                            "type": "string",
                            "enum": [*ids, NONE_CHOICE],
                            "description": "Id of the best-fitting template, or 'none'.",
                        }
                    },
                    "required": [CHOICE_FIELD],
                }
            },
        }
        picked, usage, ms = forced_call(
            model=request.model,
            tool_spec=select_spec,
            system_blocks=SELECT_SYSTEM_BLOCKS,
            user_message=SELECT_USER_PROMPT.format(
                question=request.question,
                candidates=format_candidates([(c.template_id, c.description or "") for c in candidates]),
            ),
        )
        rec["select_ms"] = ms
        rec["select_usage"] = usage
        choice = picked.get(CHOICE_FIELD)
        rec["selected"] = choice
        chosen = self._pick(candidates, choice)

        # Second call: the ordinary single-template fill on the winner.
        props, required = _params_schema(chosen.param_schema)
        props[CANNOT_EXPRESS_FIELD] = {
            "type": "boolean",
            "description": "Set true when this template cannot express the question.",
        }
        fill_spec = {
            "name": FILL_MODEL_NAME,
            "description": "Fill the chosen template's parameters.",
            "inputSchema": {
                "json": {"type": "object", "properties": props, "required": required}
            },
        }
        filled, fusage, fms = forced_call(
            model=request.model,
            tool_spec=fill_spec,
            system_blocks=FILL_SYSTEM_BLOCKS,
            user_message=FILL_USER_PROMPT.format(question=request.question),
        )
        rec["fill_ms"] = fms
        rec["fill_usage"] = fusage
        if bool(filled.pop(CANNOT_EXPRESS_FIELD, False)):
            raise _NoTemplateSelected("fill cannot_express")
        return self._fill_and_render(request, chosen, filled)


class MultiTemplateShapeAStrategy(_MultiTemplateBase):
    """Flat merged all-optional params + a required choice. One LLM call."""

    name = "multi_shape_a"
    # Whether to offer the per-template abstain hatch alongside the choice. Choosing
    # `none` says "no candidate fits"; `cannot_express` says "this template is the
    # right one but it cannot express the question", which is the signal the
    # single-template path uses to route to free-DSL instead of forcing a wrong fill.
    add_abstain_field = False
    # Whether to make the abstain decision an explicit required field (the model must
    # emit true/false) and to lead the prompt with the expressibility check. Both push
    # the combined call to consider declining, which it otherwise almost never does.
    abstain_first = False
    require_abstain_decision = False

    def _run(self, request, candidates, rec):
        ids = [c.template_id for c in candidates]
        props: dict[str, Any] = {
            CHOICE_FIELD: {
                "type": "string",
                "enum": [*ids, NONE_CHOICE],
                "description": (
                    "Id of the template you are filling, or 'none'. Fill only the "
                    "parameters prefixed with the id you choose."
                ),
            }
        }
        required = [CHOICE_FIELD]
        if self.add_abstain_field:
            props[CANNOT_EXPRESS_FIELD] = {
                "type": "boolean",
                "description": _CANNOT_EXPRESS_DESCRIPTION,
            }
            if self.require_abstain_decision:
                required.append(CANNOT_EXPRESS_FIELD)
        for c in candidates:
            sub, _ = _params_schema(
                c.param_schema, prefix=_safe_prefix(c.template_id), force_optional=True
            )
            props.update(sub)
        spec = {
            "name": COMBINED_MODEL_NAME,
            "description": "Choose a template and fill its parameters.",
            "inputSchema": {
                "json": {"type": "object", "properties": props, "required": required}
            },
        }
        out, usage, ms = forced_call(
            model=request.model,
            tool_spec=spec,
            system_blocks=(
                COMBINED_ABSTAIN_FIRST_BLOCKS if self.abstain_first else COMBINED_SYSTEM_BLOCKS
            ),
            user_message=COMBINED_USER_PROMPT.format(
                question=request.question,
                candidates=format_candidates([(c.template_id, c.description or "") for c in candidates]),
            ),
        )
        rec["fill_ms"] = ms
        rec["fill_usage"] = usage
        rec["n_props"] = len(props)
        if bool(out.pop(CANNOT_EXPRESS_FIELD, False)):
            rec["selected"] = out.get(CHOICE_FIELD)
            raise _NoTemplateSelected("combined cannot_express")
        choice = out.get(CHOICE_FIELD)
        rec["selected"] = choice
        chosen = self._pick(candidates, choice)

        # Keep only the chosen template's namespace and strip the prefix.
        prefix = _safe_prefix(chosen.template_id)
        params = {k[len(prefix):]: v for k, v in out.items() if k.startswith(prefix)}
        rec["n_filled"] = len(params)
        # Count values the model put under a template it did not choose.
        rec["cross_fill"] = sum(
            1
            for k in out
            if k != CHOICE_FIELD and not k.startswith(prefix) and out[k] is not None
        )
        try:
            return self._fill_and_render(request, chosen, params)
        except ValidationError as e:
            rec["post_validate_failed"] = str(e)[:300]
            raise


class MultiTemplateShapeA2Strategy(_MultiTemplateBase):
    """Shape A with collision-aware naming: keep bare param names where possible.

    Shape A namespaces every parameter by template id, which renames a slot the
    model was trained-of-prompt to recognise (``lex_query`` becomes
    ``catalog_v5_search__lex_query``). This variant namespaces a name only when two
    candidates disagree about it, so the common case keeps the original slot name and
    only genuinely ambiguous names are qualified.
    """

    name = "multi_shape_a2"

    def _run(self, request, candidates, rec):
        ids = [c.template_id for c in candidates]

        # A bare name is safe when every candidate that declares it declares the same
        # JSON Schema for it; otherwise that name must be qualified per template.
        seen: dict[str, Any] = {}
        clashing: set[str] = set()
        for c in candidates:
            for name, spec in c.param_schema.items():
                if not isinstance(spec, dict):
                    continue
                js = json.dumps(_param_json_schema(spec), sort_keys=True)
                if name in seen and seen[name] != js:
                    clashing.add(name)
                else:
                    seen[name] = js

        props: dict[str, Any] = {
            CHOICE_FIELD: {
                "type": "string",
                "enum": [*ids, NONE_CHOICE],
                "description": (
                    "Id of the template you are filling, or 'none'. Fill only parameters "
                    "belonging to the template you choose."
                ),
            }
        }
        # name_map: template_id -> {emitted_key: real_param_name}
        name_map: dict[str, dict[str, str]] = {}
        for c in candidates:
            prefix = _safe_prefix(c.template_id)
            mapping: dict[str, str] = {}
            for name, spec in c.param_schema.items():
                if not isinstance(spec, dict):
                    continue
                key = f"{prefix}{name}" if name in clashing else name
                mapping[key] = name
                props[key] = _param_json_schema(spec)
            name_map[c.template_id] = mapping
        rec["n_props"] = len(props)
        rec["n_clashing_names"] = len(clashing)

        spec_tool = {
            "name": COMBINED_MODEL_NAME,
            "description": "Choose a template and fill its parameters.",
            "inputSchema": {
                "json": {"type": "object", "properties": props, "required": [CHOICE_FIELD]}
            },
        }
        out, usage, ms = forced_call(
            model=request.model,
            tool_spec=spec_tool,
            system_blocks=COMBINED_SYSTEM_BLOCKS,
            user_message=COMBINED_USER_PROMPT.format(
                question=request.question,
                candidates=format_candidates(
                    [(c.template_id, c.description or "") for c in candidates]
                ),
            ),
        )
        rec["fill_ms"] = ms
        rec["fill_usage"] = usage
        choice = out.get(CHOICE_FIELD)
        rec["selected"] = choice
        chosen = self._pick(candidates, choice)

        mapping = name_map[chosen.template_id]
        params = {mapping[k]: v for k, v in out.items() if k in mapping}
        rec["n_filled"] = len(params)
        try:
            return self._fill_and_render(request, chosen, params)
        except ValidationError as e:
            rec["post_validate_failed"] = str(e)[:300]
            raise


class MultiTemplateShapeBStrategy(_MultiTemplateBase):
    """Discriminated union: one object variant per candidate. One LLM call.

    Built as a hand-written JSON Schema because strands'
    ``convert_pydantic_to_tool_spec`` only understands ``anyOf`` for
    ``Optional[T]`` and silently drops a multi-variant union.
    """

    name = "multi_shape_b"

    def _run(self, request, candidates, rec):
        variants: list[dict[str, Any]] = []
        for c in candidates:
            props, required = _params_schema(c.param_schema)
            props[CHOICE_FIELD] = {"type": "string", "enum": [c.template_id]}
            variants.append(
                {
                    "type": "object",
                    "title": c.template_id,
                    "description": c.description or "",
                    "properties": props,
                    "required": [CHOICE_FIELD, *required],
                }
            )
        variants.append(
            {
                "type": "object",
                "title": NONE_CHOICE,
                "description": "No candidate template can express the question.",
                "properties": {CHOICE_FIELD: {"type": "string", "enum": [NONE_CHOICE]}},
                "required": [CHOICE_FIELD],
            }
        )
        # The union must sit under a property: Bedrock rejects oneOf/anyOf/allOf at the
        # top level of a tool input schema ("input_schema does not support oneOf, allOf,
        # or anyOf at the top level"). Nesting is accepted, but the model often emits the
        # variant as a JSON string rather than an object, so the reader below is lenient.
        spec = {
            "name": COMBINED_MODEL_NAME,
            "description": "Choose one template variant and fill its parameters.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        SELECTION_FIELD: {
                            "description": (
                                "Exactly one variant: the template you chose, with its "
                                "parameters filled."
                            ),
                            "oneOf": variants,
                        }
                    },
                    "required": [SELECTION_FIELD],
                }
            },
        }
        out, usage, ms = forced_call(
            model=request.model,
            tool_spec=spec,
            system_blocks=COMBINED_SYSTEM_BLOCKS,
            user_message=COMBINED_USER_PROMPT.format(
                question=request.question,
                candidates=format_candidates([(c.template_id, c.description or "") for c in candidates]),
            ),
        )
        rec["fill_ms"] = ms
        rec["fill_usage"] = usage
        sel = out
        if CHOICE_FIELD not in sel:
            # Tolerate a nested variant, including one emitted as a JSON string.
            inner = out.get(SELECTION_FIELD)
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except ValueError:
                    inner = None
                rec["union_stringified"] = True
            if isinstance(inner, dict):
                sel = inner
                rec["union_nested"] = True
            else:
                rec["raw_keys"] = sorted(out)[:20]
                rec["raw_sample"] = json.dumps(out)[:400]
        choice = sel.get(CHOICE_FIELD)
        rec["selected"] = choice
        chosen = self._pick(candidates, choice)
        params = {k: v for k, v in sel.items() if k != CHOICE_FIELD}
        rec["n_filled"] = len(params)
        try:
            return self._fill_and_render(request, chosen, params)
        except ValidationError as e:
            rec["post_validate_failed"] = str(e)[:300]
            raise


class MultiTemplateShapeA3Strategy(MultiTemplateShapeAStrategy):
    """Shape A plus the per-template ``cannot_express`` hatch.

    Isolates how much of the single-call designs' accuracy gap is simply the missing
    abstain signal: `none` covers "no candidate fits", but not "this is the right
    template and it still cannot express the question".
    """

    name = "multi_shape_a3"
    add_abstain_field = True


class MultiTemplateShapeA4Strategy(MultiTemplateShapeAStrategy):
    """Shape A + abstain hatch + abstain-first prompt.

    Tests whether wording that makes the model decide expressibility *before* it
    commits to a template restores the abstention the combined call otherwise skips.
    """

    name = "multi_shape_a4"
    add_abstain_field = True
    abstain_first = True


class MultiTemplateShapeA5Strategy(MultiTemplateShapeAStrategy):
    """Shape A + abstain-first prompt + a REQUIRED abstain decision.

    Forces the model to emit ``cannot_express`` true/false explicitly, so declining is
    a mandatory decision rather than an optional field it can silently omit.
    """

    name = "multi_shape_a5"
    add_abstain_field = True
    abstain_first = True
    require_abstain_decision = True
