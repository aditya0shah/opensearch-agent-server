"""Multi-template fill: offer each candidate template as its own tool and pick one.

Selected when a request carries more than one candidate in ``context.template_ids``.
Each candidate becomes a tool whose parameters are that template's real fields (with
their required-ness and enum constraints intact), and a forced ``toolChoice: any`` call
lets the model call exactly one — either a template tool (which it then fills) or a
dedicated free-DSL tool meaning "none of these fit". The called template's input is
validated against its own model and rendered by OpenSearch, exactly as in the
single-template path.

Offering one tool per template (rather than merging every candidate's parameters into a
single flat schema) keeps each tool's schema clean: the model sees only the chosen
template's parameters when it fills, cannot mix parameters across templates, and the
per-tool schema carries the constraints a flat all-optional merge cannot.

The combined call is used only when there is a choice to make:

- one candidate (after filtering) -> the single-template strategy, whose prompt is
  dedicated to filling one template;
- several candidates -> the tool-choice call.

Candidates are pre-filtered to the request's index and capped, because each candidate
adds a tool schema and an unbounded set would dominate the prompt.

Any failure, or an abstention (the free-DSL tool), degrades to the free-DSL fallback
rather than returning a wrong query, the same contract as :mod:`template_fill`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from strands.tools.structured_output import convert_pydantic_to_tool_spec

from agents.agentic_search.prompts.multi_template_fill import (
    FREE_DSL_TOOL_NAME,
    MULTI_FILL_SYSTEM_BLOCKS,
    MULTI_FILL_USER_PROMPT,
)
from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.strategies.forced_tool import (
    forced_tool_choice_any,
    supports_forced_tool,
)
from agents.agentic_search.strategies.template_fill import (
    TEMPLATE_ID_KEY,
    TemplateFillStrategy,
)
from agents.agentic_search.template_schema import (
    TemplateSchema,
    TemplateSchemaCache,
    build_fill_model,
)

logger = logging.getLogger(__name__)


class _NoCandidateExpresses(Exception):
    """The model declined: no candidate template can express the question.

    Raised on the happy path so :meth:`MultiTemplateFillStrategy.generate` routes to
    the free-DSL fallback -- the same handling as a structural failure, but triggered by
    the model's own judgment (it called the free-DSL tool) rather than a broken render.
    """


TEMPLATE_IDS_KEY = "template_ids"

# Upper bound on candidates fed to one call. Each candidate adds a tool schema, so an
# unbounded set would crowd out the question and blow past the point where prompt caching
# keeps it affordable. Extra candidates are dropped with a warning rather than silently,
# so a caller can see the set was truncated.
MAX_CANDIDATES = 8

# Bedrock tool names must match ``[a-zA-Z0-9_-]{1,64}``. Template ids usually already do,
# but sanitize and cap defensively.
_MAX_TOOL_NAME = 64


def _sanitize_tool_name(template_id: str) -> str:
    """Return a Bedrock-legal tool name for a template id (before disambiguation)."""
    name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in template_id)
    name = name[:_MAX_TOOL_NAME]
    return name or "template"


def _tool_names_for(template_ids: list[str]) -> dict[str, str]:
    """Map each template id to a tool name unique within this candidate set.

    Sanitizing is lossy, so distinct ids can collapse onto the same tool name; a
    collision gets a positional suffix so every candidate keeps its own tool. The
    free-DSL tool name is reserved so no template can shadow the abstain option.
    """
    out: dict[str, str] = {}
    used: set[str] = {FREE_DSL_TOOL_NAME}
    for position, template_id in enumerate(template_ids):
        name = _sanitize_tool_name(template_id)
        if name in used:
            name = f"{name[: _MAX_TOOL_NAME - 4]}_{position}"
        while name in used:
            name = f"{name[: _MAX_TOOL_NAME - 5]}_{position}x"
        used.add(name)
        out[template_id] = name
    return out


class MultiTemplateFillStrategy:
    """Offer each candidate template as a tool, pick one, and fill it."""

    name = "multi_template_fill"
    # Neither the choice nor the fill uses the index mapping, so skip the per-query
    # mapping fetch; the free-DSL fallback re-adds it only when it is actually needed.
    needs_mapping = False

    def __init__(
        self,
        *,
        single: GenerationStrategy | None = None,
        fallback: GenerationStrategy | None = None,
        schema_cache: TemplateSchemaCache | None = None,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        self._fallback = fallback if fallback is not None else DirectDslStrategy()
        self._schema_cache = (
            schema_cache if schema_cache is not None else TemplateSchemaCache()
        )
        # Single-template path, reused whenever only one candidate survives filtering.
        self._single = (
            single
            if single is not None
            else TemplateFillStrategy(
                fallback=self._fallback, schema_cache=self._schema_cache
            )
        )
        self._max_candidates = max_candidates

    # ---- entry point ------------------------------------------------------

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        ids = self._candidate_ids(request.context)
        if not ids:
            logger.warning(
                "multi_template_fill selected without candidates; falling back to free-DSL"
            )
            return self._fallback_generate(request)

        try:
            candidates = self._resolve(ids, request)
        except Exception as e:  # noqa: BLE001 - resolution failure degrades
            logger.warning("candidate resolution failed (%s); falling back", e)
            return self._fallback_generate(request)

        if not candidates:
            logger.warning(
                "no candidate template resolved for index=%s; falling back",
                request.index_name,
            )
            return self._fallback_generate(request)

        # One candidate needs no choice; use the dedicated single-template path.
        if len(candidates) == 1:
            return self._delegate_single(request, candidates[0].template_id)

        try:
            return self._select_and_fill(request, candidates)
        except _NoCandidateExpresses:
            logger.info(
                "no candidate can express the question (index=%s); routing to free-DSL",
                request.index_name,
            )
            return self._fallback_generate(request)
        except Exception as e:  # noqa: BLE001 - any failure degrades to free-DSL
            logger.warning(
                "multi-template fill failed for index=%s (%s); falling back to free-DSL",
                request.index_name,
                e,
            )
            return self._fallback_generate(request)

    # ---- candidate handling ----------------------------------------------

    @staticmethod
    def _candidate_ids(context: dict[str, Any]) -> list[str]:
        """Return the requested candidate ids, de-duplicated, order preserved.

        Accepts a single id under either key so a caller can send a one-element list
        without special-casing.
        """
        raw = context.get(TEMPLATE_IDS_KEY)
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raw = []
        ids = [str(x) for x in raw if x]
        one = context.get(TEMPLATE_ID_KEY)
        if one and str(one) not in ids:
            ids.append(str(one))
        seen: set[str] = set()
        return [i for i in ids if not (i in seen or seen.add(i))]

    def _resolve(
        self, ids: list[str], request: GenerationRequest
    ) -> list[TemplateSchema]:
        """Resolve candidates, dropping unusable ones and capping the set.

        A candidate that cannot be read (unregistered, or not readable by this caller)
        is skipped rather than failing the request: with several candidates, one bad id
        must not deny the others. Candidates bound to a different index are dropped
        because the rendered query runs against this request's index.
        """
        resolved: list[TemplateSchema] = []
        # Stop as soon as the cap is met. Each unresolved id costs a system-index read
        # and a model build, so capping only after the loop would let a caller's long
        # list drive that work regardless of how few candidates are actually used.
        for position, tid in enumerate(ids):
            if len(resolved) >= self._max_candidates:
                logger.warning(
                    "candidate set capped at %d; ignoring %s",
                    self._max_candidates,
                    ", ".join(ids[position:]),
                )
                break
            try:
                schema = self._schema_cache.get(tid, request.client)
            except Exception as e:  # noqa: BLE001 - skip this candidate only
                logger.warning("candidate %s unresolved (%s); skipping", tid, e)
                continue
            if (
                schema.index_binding
                and request.index_name
                and schema.index_binding != request.index_name
            ):
                logger.debug(
                    "candidate %s is bound to index %s, not %s; skipping",
                    tid,
                    schema.index_binding,
                    request.index_name,
                )
                continue
            resolved.append(schema)
        return resolved

    def _delegate_single(
        self, request: GenerationRequest, template_id: str
    ) -> dict[str, Any]:
        """Run the single-template strategy for ``template_id``."""
        context = {**request.context, TEMPLATE_ID_KEY: template_id}
        context.pop(TEMPLATE_IDS_KEY, None)
        return self._single.generate(replace(request, context=context))

    # ---- the combined call ------------------------------------------------

    def _select_and_fill(
        self, request: GenerationRequest, candidates: list[TemplateSchema]
    ) -> dict[str, Any]:
        """Pick a template tool and fill it in one forced tool-choice call, then render."""
        if not supports_forced_tool(request.model):
            # Non-Bedrock providers don't expose the converse_stream tool-choice path;
            # degrade to free-DSL rather than fail.
            raise RuntimeError("forced tool choice is unsupported for this provider")

        tool_specs, choice_map = self._build_tools(candidates)
        tool_name, tool_input = forced_tool_choice_any(
            model=request.model,
            tool_specs=tool_specs,
            system_blocks=MULTI_FILL_SYSTEM_BLOCKS,
            user_message=MULTI_FILL_USER_PROMPT.format(
                question=request.question, free_dsl_tool=FREE_DSL_TOOL_NAME
            ),
        )

        if tool_name == FREE_DSL_TOOL_NAME:
            raise _NoCandidateExpresses("model called the free-DSL tool")
        picked = choice_map.get(tool_name)
        if picked is None:
            raise ValueError(f"model called unknown tool '{tool_name}'")
        chosen, model_cls = picked

        # Validate the tool input against the template's own model. The tool's schema was
        # built from that same model, so a conforming call passes; validation also
        # recovers the real Mustache parameter names from the sanitized field aliases.
        # Dropping unset params lets the body's inverted sections and optional clauses
        # behave as authored.
        clean = model_cls.model_validate(tool_input).model_dump(
            by_alias=True, exclude_none=True
        )

        rendered = self._render(request.client, chosen.template_id, clean)
        logger.info(
            "Multi-template fill chose %s of %d candidates and rendered %d params",
            chosen.template_id,
            len(candidates),
            len(clean),
        )
        return rendered

    def _build_tools(
        self, candidates: list[TemplateSchema]
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[TemplateSchema, type]]]:
        """Build one tool per candidate plus the free-DSL fallback tool.

        Returns the Bedrock tool specs and a map from each tool name back to its
        ``(candidate, fill-model)`` pair, so the called tool is resolved by exact name
        rather than by inferring ownership from string prefixes.
        """
        names = _tool_names_for([c.template_id for c in candidates])
        tool_specs: list[dict[str, Any]] = []
        choice_map: dict[str, tuple[TemplateSchema, type]] = {}
        for candidate in candidates:
            # add_abstain=False: abstention is a separate tool, not a per-template field.
            model_cls = build_fill_model(candidate.param_schema, add_abstain=False)
            spec = convert_pydantic_to_tool_spec(model_cls)
            tool_name = names[candidate.template_id]
            tool_specs.append(
                {
                    "name": tool_name,
                    "description": candidate.description
                    or f"Search template '{candidate.template_id}'.",
                    "inputSchema": spec["inputSchema"],
                }
            )
            choice_map[tool_name] = (candidate, model_cls)

        tool_specs.append(
            {
                "name": FREE_DSL_TOOL_NAME,
                "description": (
                    "Call this when NO listed template can express the question — it "
                    "needs a field, filter, projection, aggregation, count-only answer, "
                    "exact phrase, prefix/wildcard/fuzzy match, ranking signal, or "
                    "similarity that no template's parameters cover."
                ),
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        )
        return tool_specs, choice_map

    # ---- shared tail ------------------------------------------------------

    @staticmethod
    def _render(
        client: Any, template_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Render the stored body via ``POST _render/template`` and unwrap the DSL."""
        resp = client.render_search_template(id=template_id, body={"params": params})
        output = resp.get("template_output") if isinstance(resp, dict) else None
        if output is None:
            raise ValueError("_render/template returned no template_output")
        if isinstance(output, str):
            output = json.loads(output)
        if not isinstance(output, dict):
            raise ValueError("rendered template_output is not a _search body object")
        return output

    def _fallback_generate(self, request: GenerationRequest) -> dict[str, Any]:
        """Run the free-DSL fallback, fetching the mapping it needs."""
        req = request
        if not request.mapping:
            try:
                mapping = json.dumps(
                    request.client.indices.get_mapping(index=request.index_name)
                )
                req = replace(request, mapping=mapping)
            except Exception as e:  # noqa: BLE001 - let the fallback try regardless
                logger.warning(
                    "fallback mapping fetch failed for index=%s (%s)",
                    request.index_name,
                    e,
                )
        return self._fallback.generate(req)


__all__ = ["MAX_CANDIDATES", "TEMPLATE_IDS_KEY", "MultiTemplateFillStrategy"]
