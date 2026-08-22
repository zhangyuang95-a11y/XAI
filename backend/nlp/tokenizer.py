"""Pretrained multilingual Unigram tokenizer with literal-span preservation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_TOKENIZER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_JSON_SCHEMA_HINTS: Mapping[str, str] = {
    "QuestionIRV2": (
        'Return exactly seven keys: {"i":intent,"t":canonical_target_id,'
        '"q":canonical_query_variable,"w":desired_value_or_null,'
        '"x":[],"r":[],"a":[]}. '
        'Allowed intents: factual, explanatory, why_not, predictive, '
        'counterfactual, comparative, diagnostic, mixed. Use only IDs from '
        'the supplied environment schema. Priority rules: (1) explicit '
        'if/suppose/change requests are counterfactual and put edits in x or r; '
        '(2) "why not X" / “为什么不X” is why_not and MUST preserve X in d; '
        '(3) an ordinary current-behavior why is explanatory. When X is an '
        'objective alias, use q="objective" and w=objective_id; when X is an '
        'action alias, use q="observed_action" and w=action_id. For questions '
        'without an explicitly desired alternative, w is null. “当前” does '
        'not cancel rule (2). '
        'A desired destination or objective is not a scene edit, so x and r '
        'stay empty. Each explicit primitive edit is '
        '{"op":operation,"e":entity_id,"p":property,"val":value,'
        '"span":exact_source_span}; each explicit relation edit is '
        '{"rel":relation,"s":[entity_ids],"val":value,'
        '"span":exact_source_span}. Never invent an edit, entity, value, or '
        'relation. d is always an object; a is always an array. Output JSON '
        'only and stop immediately after the closing brace.'
    ),
    "QueryPlan": (
        'Return a JSON INSTANCE, not JSON Schema. Never output keys named '
        '"type", "properties", "required", "items", "enum", or "value". '
        'Use this compact instance template: '
        '{"intent":"explanatory","frame_reference":0,'
        '"subjects":["robot_2"],"referenced_entities":[],'
        '"requires_scene_edit":false,"requires_policy_query":true,'
        '"requires_program_trace":true,"requires_simulation":false,'
        '"requires_baseline_comparison":false,"scene_edit_plan":null,'
        '"target_variables":["robot_2.observed_action"],"horizon":1,'
        '"rollout_count":2,"evidence_requirements":["state",'
        '"actor_observation","neural_policy","program_trace"],'
        '"desired_outcomes":{},'
        '"assumed_outcomes":{"robot_2.observed_action":"RIGHT"},'
        '"response_language":"zh-CN",'
        '"confidence":0.95,"clarification_required":false,'
        '"clarification_reason":null,"unsupported_components":[]}. '
        'When requires_scene_edit is true, scene_edit_plan must not be null. '
        'For example, moving an entity uses '
        '{"source_frame":0,"entity_references":["<entity_id>"],'
        '"primitive_interventions":[{"operation":"move_entity",'
        '"entity_id":"<entity_id>","property_name":null,'
        '"value":[5,5],"source_span":"<exact user phrase>",'
        '"metadata":{}}],"relational_constraints":[],'
        '"preserved_variables":[],"simulation_horizon":1}. '
        'Allowed intent values: factual, explanatory, why_not, predictive, '
        'counterfactual, comparative, diagnostic, mixed.'
    ),
    "SemanticQueryIR": (
        'Return a JSON INSTANCE, not JSON Schema. This is a semantic request, '
        'not an execution plan: never output requires_scene_edit, '
        'requires_policy_query, requires_program_trace, requires_simulation, '
        'requires_baseline_comparison, or scene_edit_plan. Use exactly one '
        'source of truth for scene changes: the two top-level arrays '
        'primitive_interventions and relational_constraints. Use this compact '
        'instance shape: {"intent":"counterfactual","frame_reference":5,'
        '"subjects":["robot_2"],"referenced_entities":["robot_1"],'
        '"primitive_interventions":[{"operation":"move_entity",'
        '"entity_id":"robot_1","property_name":null,"value":[5,5],'
        '"source_span":"<exact user phrase>","metadata":{}}],'
        '"relational_constraints":[],"preserved_variables":[],'
        '"target_variables":["robot_2.next_action"],"causal_variables":[],'
        '"desired_outcomes":{},'
        '"assumed_outcomes":{},"horizon":1,"rollout_count":2,'
        '"response_language":"zh-CN","ambiguities":[],'
        '"unsupported_components":[]}. subjects '
        'contains exactly the prediction_target entities whose behavior the '
        'user asks about; referenced_entities are '
        'other explicitly mentioned or edited entities. A hypothetical edit '
        'must appear in primitive_interventions or relational_constraints. '
        'For a question without an edit, both arrays must be empty. Use '
        'move_entity for position, set_battery for battery, set_direction for '
        'direction, and relational_constraints for relations such as surrounds '
        'or blocks. Copy all entities and values from the user question and do '
        'not invent facts.'
    ),
    "EntityRoleResolution": (
        'Return a JSON INSTANCE, not JSON Schema, with exactly five top-level '
        'keys: bindings, explicit_scene_edit, primitive_interventions, '
        'relational_constraints, and ambiguities. bindings is an array; each '
        'item has exactly entity_id, roles, and source_span. ambiguities is an '
        'array. explicit_scene_edit is true exactly when the user explicitly '
        'asks to change the scene. Every explicit hypothetical assignment must '
        'appear exactly once in primitive_interventions or '
        'relational_constraints. A battery assignment uses set_battery; a '
        'position assignment uses move_entity; a direction assignment uses '
        'set_direction. If there is no hypothetical change, return false and '
        'two empty edit arrays. '
        'Allowed roles are prediction_target, intervention_subject, '
        'comparison_target, and context_entity. The entity whose result is '
        'asked for is prediction_target; an entity changed by an if-clause is '
        'intervention_subject. Mention order has no semantic meaning. Every '
        'source_span must be copied literally from the current user question. '
        'Do not copy entity IDs, spans, or values from this contract.'
    ),
    "AtomicClaimList": (
        'Return a JSON INSTANCE, not JSON Schema. Never output keys named '
        '"type", "properties", "required", "items", "enum", or "value". '
        'Use this compact instance template: '
        '{"claims":[{"claim_id":"claim_1",'
        '"text":"<one exact atomic statement from the final answer>",'
        '"claim_type":"state",'
        '"entities":["<canonical entity actually present in the answer>"],'
        '"frame_scope":[],"time_scope":"current frame",'
        '"predicate":"<predicate representing only this statement>",'
        '"expected_outcome":{"<structured field>":"<asserted value>"},'
        '"modality":"observed","confidence":0.95},'
        '{"claim_id":"claim_2",'
        '"text":"<a different exact atomic statement later in the answer>",'
        '"claim_type":"state",'
        '"entities":["<canonical entity actually present in the answer>"],'
        '"frame_scope":[],"time_scope":"current frame",'
        '"predicate":"<predicate for that different statement>",'
        '"expected_outcome":{"<structured field>":"<asserted value>"},'
        '"modality":"observed","confidence":0.95}]}. '
        'The two items illustrate that the array must continue for every '
        'atomic statement; do not copy their placeholder text.'
    ),
    "ClaimEvidenceAlignment": (
        'Return a JSON INSTANCE, not JSON Schema. Never output keys named '
        '"type", "properties", "required", "items", "enum", or "value". '
        'Use this compact instance template: '
        '{"alignments":[{"claim_id":"<copy an exact supplied claim_id>",'
        '"evidence_assertions":[{"evidence_id":'
        '"<copy an exact supplied evidence_id>",'
        '"claim_value":"<value asserted by that claim>"}]}]}. '
        'claim_value is the value asserted by the claim after translating '
        'the natural language into the canonical type used by that evidence '
        'record. It is not the observed value copied from Evidence.'
    ),
    "GroundedUserExplanation": (
        'Return a JSON INSTANCE, not JSON Schema. Never output keys named '
        '"type", "properties", "required", "items", "enum", or "value". '
        'Use this compact instance template: '
        '{"answer":"<one or two natural-language sentences generated from '
        'the supplied evidence>",'
        '"objective_reason":"<sentence explaining why the selected objective '
        'exists, or an empty string when not required>",'
        '"action_reason":"<sentence explaining why this action helps or was '
        'required, or an empty string when not required>",'
        '"premise_correction":null,'
        '"used_evidence_ids":["<exact ID copied from the task input>"],'
        '"covered_requirement_keys":["<exact required key copied from the '
        'task input>"]}. '
        'The answer must be two or three short conversational sentences. '
        'For a why-question, the answer must state both why the current task '
        'exists and why the action helps or was required. Copy all IDs from '
        'required_reason_evidence_ids and every key from '
        'required_reason_requirements. Express every required value in the '
        'corresponding objective_reason or action_reason sentence. '
        'When task input premise_check.status is "contradicted", '
        'premise_correction must be a non-empty conversational correction '
        'that states the recorded action and says the assumed action did not '
        'happen. The answer, objective_reason, and action_reason must then '
        'explain the actual recorded action from the required evidence, never '
        'the action assumed by the question. Otherwise premise_correction must '
        'be null. '
        'Every used_evidence_ids item must be copied exactly from the '
        'available evidence IDs in the task input.'
    ),
    "NaturalLanguageExplanation": (
        'Return a JSON INSTANCE, not JSON Schema. Never output keys named '
        '"type", "properties", "required", "items", "enum", or "value". '
        'Use exactly this compact instance shape: '
        '{"answer":"<one to three short conversational sentences that '
        'answer the user from the supplied evidence>"}. '
        'The answer value must contain only natural language: do not copy '
        'JSON, field names, object representations, probability tables, or '
        'internal implementation terms into it.'
    ),
    "EvidenceBoundNaturalLanguageExplanation": (
        'Return a JSON INSTANCE, not JSON Schema. Never output keys named '
        '"type", "properties", "required", "items", "enum", or "value". '
        'Use exactly this compact instance shape: '
        '{"answer":"<one to three short conversational sentences based '
        'only on the supplied semantic facts>",'
        '"used_evidence_ids":["<exact supplied evidence ID>"],'
        '"covered_requirement_keys":["<exact supplied requirement key>"]}. '
        'Copy every mandatory requirement key after expressing that fact in '
        'the answer. Copy only evidence IDs that occur in the task input. '
        'The answer must contain natural language only; never place JSON, '
        'field names, probability tables, or internal implementation terms '
        'inside the answer string.'
    ),
    "ExplanationDocumentV2": (
        'Return the compact wire form {"s":[[role,text,[unit_ids]],...]}. '
        'role must be summary, task, rationale, process, or counterfactual. '
        'text is concise natural language and unit_ids contains only exact IDs '
        'supplied in the task. Example shape: {"s":[["summary","<direct '
        'answer>",["final"]],["process","<decision process>",'
        '["proposal","coordination"]]]}.'
    ),
    "LanguageIdentification": (
        'Return a JSON INSTANCE, not JSON Schema. This application supports '
        'exactly two response languages. input_language and response_language '
        'must each be either "zh-CN" or "en"; no other language tag is '
        'allowed. Classify the quoted user text, never the surrounding '
        'instructions. Unsupported languages use "en". Examples: Simplified '
        'Chinese text "为什么等待？" -> '
        '{"input_language":"zh-CN","response_language":"zh-CN",'
        '"confidence":0.99}; English text "Why is it waiting?" -> '
        '{"input_language":"en","response_language":"en",'
        '"confidence":0.99}. Return only the object for the actual task input.'
    ),
}

_JSON_REQUIRED_KEYS: Mapping[str, tuple[str, ...]] = {
    "QuestionIRV2": (
        "intent",
        "target_entity",
        "frame_reference",
        "entity_roles",
        "primitive_interventions",
        "relational_constraints",
        "target_variables",
        "desired_outcomes",
        "ambiguities",
    ),
    "QueryPlan": ("intent",),
    "SemanticQueryIR": (
        "intent",
        "subjects",
        "primitive_interventions",
        "relational_constraints",
        "target_variables",
        "causal_variables",
    ),
    "EntityRoleResolution": (
        "bindings",
        "explicit_scene_edit",
        "primitive_interventions",
        "relational_constraints",
        "ambiguities",
    ),
    "AtomicClaimList": ("claims",),
    "ClaimEvidenceAlignment": ("alignments",),
    "GroundedUserExplanation": (
        "answer",
        "used_evidence_ids",
    ),
    "NaturalLanguageExplanation": ("answer",),
    "EvidenceBoundNaturalLanguageExplanation": (
        "answer",
        "used_evidence_ids",
        "covered_requirement_keys",
    ),
    "ExplanationDocumentV2": ("sections",),
    "LanguageIdentification": (
        "input_language",
        "response_language",
    ),
}


@dataclass(frozen=True)
class ProtectedLiteral:
    text: str
    start: int
    end: int
    kind: str


class MultilingualQueryTokenizer:
    """SentencePiece-style tokenizer; never trains a Warehouse-specific vocabulary."""

    def __init__(self, model_name: str = DEFAULT_TOKENIZER, *, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.backend = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
        model = getattr(getattr(self.backend, "backend_tokenizer", None), "model", None)
        self.backend_model_type = type(model).__name__
        if self.backend_model_type != "Unigram":
            raise ValueError(
                f"{model_name} does not expose the required pretrained Unigram/SentencePiece tokenizer."
            )

    def encode(self, text: str, *, max_length: int = 192) -> dict[str, Any]:
        return self.backend(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

    def protected_literals(self, text: str) -> tuple[ProtectedLiteral, ...]:
        """Locate structural literals without assigning any semantic intent."""

        literals: list[ProtectedLiteral] = []
        index = 0
        while index < len(text):
            character = text[index]
            if character.isdigit():
                end = index + 1
                while end < len(text):
                    if text[end].isdigit():
                        end += 1
                        continue
                    if text[end] == "." and end + 1 < len(text) and text[end + 1].isdigit():
                        end += 1
                        continue
                    break
                if end < len(text) and text[end] in "%％":
                    end += 1
                literals.append(ProtectedLiteral(text[index:end], index, end, "number"))
                index = end
                continue
            index += 1
        return tuple(literals)

    def span_text(self, text: str, offsets: Sequence[Sequence[int]], token_indices: Sequence[int]) -> str:
        usable = [offsets[index] for index in token_indices if offsets[index][1] > offsets[index][0]]
        if not usable:
            return ""
        return text[min(item[0] for item in usable) : max(item[1] for item in usable)]


class StructuredTransformerBackend(Protocol):
    """Production NLP contract used by the open-ended planning pipeline."""

    def generate_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        max_new_tokens: int | None = None,
    ) -> Mapping[str, Any]: ...

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str: ...


class HuggingFaceStructuredTransformer:
    """JSON/text generation through a configured Transformer checkpoint.

    There is deliberately no keyword or regex fallback. A missing or
    incompatible model fails explicitly so a rule system can never silently
    impersonate natural-language understanding.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
        max_new_tokens: int = 768,
        max_input_tokens: int = 6144,
        json_repair_attempts: int = 0,
        dtype: str = "auto",
        attention_implementation: str = "sdpa",
    ) -> None:
        if not model_name_or_path:
            raise ValueError("A Transformer model path is required.")
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.max_input_tokens = max(512, int(max_input_tokens))
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.requested_dtype = str(dtype)
        self.dtype: str | torch.dtype = str(dtype)
        self.attention_implementation = str(attention_implementation)
        self.total_generation_calls = 0
        self.last_generation_metrics: dict[str, Any] = {}
        self.last_raw_text = ""
        self._generation_lock = threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            local_files_only=local_files_only,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if self.device.type == "cuda":
            if self.requested_dtype.casefold() == "auto":
                self.dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            model_kwargs["dtype"] = self.dtype
            if self.attention_implementation:
                model_kwargs["attn_implementation"] = self.attention_implementation
        self.is_encoder_decoder = True
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        except (OSError, TypeError, ValueError):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
            self.is_encoder_decoder = False
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        model_prompt = self._model_prompt(prompt)
        generation_limit = (
            self.max_new_tokens
            if max_new_tokens is None
            else max(32, int(max_new_tokens))
        )
        model_context = int(
            getattr(
                self.model.config,
                "max_position_embeddings",
                self.max_input_tokens + generation_limit,
            )
        )
        input_limit = max(
            512,
            min(
                self.max_input_tokens,
                model_context - generation_limit,
            ),
        )
        encoded = self.tokenizer(
            model_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_limit,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        started = time.perf_counter()
        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=generation_limit,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        tokens = generated[0]
        if not self.is_encoder_decoder:
            tokens = tokens[encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Transformer returned an empty response.")
        self.total_generation_calls += 1
        self.last_raw_text = text
        self.last_generation_metrics = {
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "output_tokens": int(tokens.shape[0]),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
        }
        return text

    def warmup(self) -> None:
        """Materialize CUDA kernels without entering the request critical path."""

        if self.device.type != "cuda":
            return
        previous = self.total_generation_calls
        try:
            self.generate_text("Return OK.", max_new_tokens=32)
        finally:
            # Warm-up is operational telemetry, not a user-request model call.
            self.total_generation_calls = previous

    def _model_prompt(self, prompt: str) -> str:
        """Format Instruct checkpoints exactly as they were trained."""

        chat_template = getattr(self.tokenizer, "chat_template", None)
        apply_chat_template = getattr(
            self.tokenizer,
            "apply_chat_template",
            None,
        )
        if (
            self.is_encoder_decoder
            or not chat_template
            or not callable(apply_chat_template)
        ):
            return prompt
        return str(
            apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the structured natural-language component "
                            "of an auditable multi-agent XAI system. Follow the "
                            "requested output format exactly. Never invent "
                            "simulator evidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        max_new_tokens: int | None = None,
    ) -> Mapping[str, Any]:
        shape = _JSON_SCHEMA_HINTS.get(
            schema_name,
            "one JSON object matching the requested schema",
        )
        request = (
            "OUTPUT CONTRACT:\n"
            f"Return exactly one valid JSON object for schema {schema_name}.\n"
            "Do not use Markdown fences. Do not add analysis, comments, or a "
            "second JSON object.\n"
            f"Required shape and allowed values:\n{shape}\n\n"
            "TASK INPUT:\n"
            f"{prompt}"
        )
        last_text = ""
        last_error: ValueError | None = None
        for attempt in range(self.json_repair_attempts + 1):
            generation_prompt = request
            if attempt:
                generation_prompt = (
                    "Correct the previous response into exactly one valid JSON "
                    f"object for schema {schema_name}. Preserve the intended "
                    "answer, remove all prose and duplicate objects, and follow "
                    f"this required shape:\n{shape}\n"
                    f"Previous response:\n{last_text}"
                )
            if max_new_tokens is None:
                last_text = self.generate_text(generation_prompt)
            else:
                last_text = self.generate_text(
                    generation_prompt,
                    max_new_tokens=max_new_tokens,
                )
            try:
                return _extract_schema_object(
                    last_text,
                    schema_name=schema_name,
                )
            except ValueError as exc:
                last_error = exc
        excerpt = " ".join(last_text.split())[:500]
        raise ValueError(
            f"Transformer returned invalid JSON for {schema_name} after "
            f"{self.json_repair_attempts + 1} attempt(s): {last_error}. "
            f"Output excerpt: {excerpt!r}"
        ) from last_error


class CallableTransformerBackend:
    """Dependency-injected backend for deterministic tests.

    The callable stands in for a Transformer at the interface boundary; it is
    never selected implicitly by production code.
    """

    def __init__(
        self,
        json_generator: Callable[[str, str], Mapping[str, Any]],
        text_generator: Callable[[str], str] | None = None,
    ) -> None:
        self._json_generator = json_generator
        self._text_generator = text_generator

    def generate_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        max_new_tokens: int | None = None,
    ) -> Mapping[str, Any]:
        del max_new_tokens
        return self._json_generator(prompt, schema_name)

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        del max_new_tokens
        if self._text_generator is None:
            raise RuntimeError("The injected Transformer backend has no text generator.")
        return self._text_generator(prompt)


def _extract_schema_object(
    text: str,
    *,
    schema_name: str,
) -> Mapping[str, Any]:
    """Return the first complete object matching the requested schema.

    Decoder.raw_decode stops at the end of one object, so harmless Markdown,
    prose, or a duplicated second object cannot turn a valid first object into
    the ``Extra data`` failure produced by ``json.loads(first_brace:last_brace)``.
    """

    decoder = json.JSONDecoder()
    candidates: list[Mapping[str, Any]] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            candidates.append(value)
            unwrapped = _unwrap_schema_object(
                value,
                schema_name=schema_name,
            )
            if _matches_schema(unwrapped, schema_name=schema_name):
                return unwrapped
    recovered_items = _recover_schema_item_objects(
        candidates,
        schema_name=schema_name,
    )
    if recovered_items is not None:
        return recovered_items
    if candidates:
        keys = sorted(
            {
                str(key)
                for candidate in candidates
                for key in candidate
            }
        )
        raise ValueError(
            f"found JSON object(s), but none match {schema_name}; "
            f"available keys are {keys}"
        )
    raise ValueError("no complete JSON object was found")


def _recover_schema_item_objects(
    candidates: Sequence[Mapping[str, Any]],
    *,
    schema_name: str,
) -> Mapping[str, Any] | None:
    """Recover arrays when a small LM omits only their outer wrapper."""

    if schema_name == "AtomicClaimList":
        items = [
            dict(item)
            for item in candidates
            if {
                "claim_id",
                "text",
                "claim_type",
            }.issubset(item)
        ]
        return {"claims": items} if items else None
    if schema_name == "ClaimEvidenceAlignment":
        items = [
            dict(item)
            for item in candidates
            if {
                "claim_id",
            }.issubset(item)
            and (
                (
                    isinstance(item.get("evidence_assertions"), Sequence)
                    and not isinstance(
                        item.get("evidence_assertions"),
                        (str, bytes),
                    )
                )
                or (
                    isinstance(item.get("evidence_ids"), Sequence)
                    and not isinstance(
                        item.get("evidence_ids"),
                        (str, bytes),
                    )
                )
            )
        ]
        return {"alignments": items} if items else None
    if schema_name == "ExplanationDocumentV2":
        items = [
            dict(item)
            for item in candidates
            if {"role", "text", "unit_ids"}.issubset(item)
        ]
        return {"sections": items} if items else None
    return None


def _unwrap_schema_object(
    payload: Mapping[str, Any],
    *,
    schema_name: str,
) -> Mapping[str, Any]:
    candidates: list[Mapping[str, Any]] = [payload]
    for key in (
        schema_name,
        schema_name.lower(),
        "result",
        "output",
    ):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for candidate in candidates:
        if _matches_schema(candidate, schema_name=schema_name):
            return candidate
        schema_instance = _schema_description_to_instance(candidate)
        if (
            schema_instance is not None
            and _matches_schema(
                schema_instance,
                schema_name=schema_name,
            )
        ):
            return schema_instance
    return payload


_MISSING_SCHEMA_VALUE = object()


def _schema_description_to_instance(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Recover an instance when a model fills values into JSON Schema.

    Some instruction-tuned models copy the output contract and attach concrete
    ``value`` fields instead of returning the requested object. This decoder
    accepts only explicit values, constants, array item values, object
    properties, and the first model-supplied enum candidate. The recovered
    object still has to pass the normal named-schema validation.
    """

    properties = payload.get("properties")
    if (
        payload.get("type") != "object"
        or not isinstance(properties, Mapping)
    ):
        return None
    result: dict[str, Any] = {}
    for key, description in properties.items():
        value = _schema_description_value(description)
        if value is not _MISSING_SCHEMA_VALUE:
            result[str(key)] = value
    return result


def _schema_description_value(description: Any) -> Any:
    if not isinstance(description, Mapping):
        return description
    if "value" in description:
        value = description["value"]
        if isinstance(value, Mapping):
            nested = _schema_description_to_instance(value)
            return nested if nested is not None else dict(value)
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            return [
                (
                    _schema_description_value(item)
                    if isinstance(item, Mapping)
                    else item
                )
                for item in value
            ]
        return value
    if "const" in description:
        return description["const"]

    node_type = description.get("type")
    if node_type == "array":
        items = description.get("items")
        if isinstance(items, Sequence) and not isinstance(
            items,
            (str, bytes),
        ):
            values = [
                _schema_description_value(item)
                for item in items
            ]
            return [
                value
                for value in values
                if value is not _MISSING_SCHEMA_VALUE
            ]
        return _MISSING_SCHEMA_VALUE
    if node_type == "object" or isinstance(
        description.get("properties"),
        Mapping,
    ):
        nested = _schema_description_to_instance(description)
        return (
            nested
            if nested is not None
            else _MISSING_SCHEMA_VALUE
        )

    enum_values = description.get("enum")
    if (
        isinstance(enum_values, Sequence)
        and not isinstance(enum_values, (str, bytes))
        and enum_values
    ):
        return enum_values[0]
    return _MISSING_SCHEMA_VALUE


def _matches_schema(
    payload: Mapping[str, Any],
    *,
    schema_name: str,
) -> bool:
    if schema_name == "QuestionIRV2":
        compact_v2 = {"i", "t", "x", "r", "v", "d", "a"}
        # Entity, query-variable, desired-value and empty-array omissions can
        # all be completed from literal ontology anchors and the UI focus.
        # Requiring those presentation fields here used to discard a valid
        # semantic intent before the environment binder could inspect it.
        compact_v3 = {"i"}
        expanded = set(_JSON_REQUIRED_KEYS["QuestionIRV2"])
        if (
            not compact_v2.issubset(payload)
            and not compact_v3.issubset(payload)
            and not expanded.issubset(payload)
        ):
            return False
        intent = str(payload.get("i", payload.get("intent", ""))).lower()
        if intent not in {
            "factual",
            "explanatory",
            "why_not",
            "predictive",
            "counterfactual",
            "comparative",
            "diagnostic",
            "mixed",
        }:
            return False
    elif schema_name == "ExplanationDocumentV2":
        if "sections" not in payload and "s" not in payload:
            return False
    else:
        required = _JSON_REQUIRED_KEYS.get(schema_name, ())
        if any(key not in payload for key in required):
            return False
    if schema_name == "QueryPlan":
        return str(payload.get("intent", "")).lower() in {
            "factual",
            "explanatory",
            "why_not",
            "predictive",
            "counterfactual",
            "comparative",
            "diagnostic",
            "mixed",
        }
    if schema_name == "AtomicClaimList":
        value = payload.get("claims")
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        )
    if schema_name == "LanguageIdentification":
        return (
            payload.get("input_language") in {"zh-CN", "en"}
            and payload.get("response_language") in {"zh-CN", "en"}
        )
    if schema_name == "ClaimEvidenceAlignment":
        value = payload.get("alignments")
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        )
    if schema_name == "GroundedUserExplanation":
        answer = payload.get("answer")
        evidence_ids = payload.get("used_evidence_ids")
        return bool(
            isinstance(answer, str)
            and answer.strip()
            and isinstance(evidence_ids, Sequence)
            and not isinstance(evidence_ids, (str, bytes))
        )
    return True
