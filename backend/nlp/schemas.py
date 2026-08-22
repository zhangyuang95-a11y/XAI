"""Environment-neutral, executable query intermediate representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class QueryIntent(str, Enum):
    """Runtime intent produced by a Transformer, never by the environment."""

    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    WHY_NOT = "why_not"
    PREDICTIVE = "predictive"
    COUNTERFACTUAL = "counterfactual"
    COMPARATIVE = "comparative"
    DIAGNOSTIC = "diagnostic"
    MIXED = "mixed"


class EntityQueryRole(str, Enum):
    """Semantic role played by an entity in one user request.

    Entity order is not a semantic signal.  In particular, the entity being
    edited in a counterfactual is often mentioned before the entity whose
    behaviour the user wants predicted.  Keeping those roles explicit avoids
    routing a correct scene edit to the wrong policy target.
    """

    PREDICTION_TARGET = "prediction_target"
    INTERVENTION_SUBJECT = "intervention_subject"
    COMPARISON_TARGET = "comparison_target"
    CONTEXT_ENTITY = "context_entity"


@dataclass(frozen=True)
class EntityRoleBinding:
    """Auditable binding between one canonical entity and its query roles."""

    entity_id: str
    roles: tuple[EntityQueryRole, ...]
    source_span: str = ""

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.entity_id.strip():
            errors.append("entity_role.entity_id is empty")
        if not self.roles:
            errors.append("entity_role.roles is empty")
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "roles": [role.value for role in self.roles],
            "source_span": self.source_span,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EntityRoleBinding":
        raw_roles = payload.get("roles", ())
        if isinstance(raw_roles, str):
            raw_roles = (raw_roles,)
        value = cls(
            entity_id=str(payload.get("entity_id", "")),
            roles=tuple(
                dict.fromkeys(
                    EntityQueryRole(str(role).lower()) for role in raw_roles
                )
            ),
            source_span=str(payload.get("source_span", "")),
        )
        errors = value.validate()
        if errors:
            raise ValueError("Invalid EntityRoleBinding: " + "; ".join(errors))
        return value


@dataclass(frozen=True)
class SemanticQueryGrounding:
    """Independent grounding contract for entities and requested scene edits.

    This structure is decoded before :class:`SemanticQueryIR`.  It is the
    authoritative record of *what the user explicitly changed* and *whose
    outcome the user asked for*.  Keeping these commitments outside the main
    planning pass prevents an if-clause from being silently dropped while the
    rest of the question is still accepted as an ordinary policy query.

    ``edit_extraction_complete`` is false only for partial injected backends
    that still return the old role-only payload.  Production Transformer
    outputs are schema-checked and must always provide the complete edit
    arrays and the explicit-scene-edit flag.
    """

    bindings: tuple[EntityRoleBinding, ...]
    explicit_scene_edit: bool
    primitive_interventions: tuple["PrimitiveIntervention", ...] = ()
    relational_constraints: tuple["RelationalConstraint", ...] = ()
    ambiguities: tuple[str, ...] = ()
    edit_extraction_complete: bool = True

    @property
    def has_scene_edit(self) -> bool:
        return bool(
            self.primitive_interventions or self.relational_constraints
        )

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        for binding in self.bindings:
            errors.extend(binding.validate())
        for intervention in self.primitive_interventions:
            errors.extend(intervention.validate())
        for constraint in self.relational_constraints:
            errors.extend(constraint.validate())
        if (
            self.edit_extraction_complete
            and self.explicit_scene_edit != self.has_scene_edit
        ):
            errors.append(
                "explicit_scene_edit must equal the presence of a primitive "
                "intervention or relational constraint"
            )
        edited_entities = {
            entity_id
            for intervention in self.primitive_interventions
            for entity_id in _primitive_intervention_entity_ids(intervention)
        }
        role_entities = {
            binding.entity_id
            for binding in self.bindings
            if EntityQueryRole.INTERVENTION_SUBJECT in binding.roles
        }
        if edited_entities - role_entities:
            errors.append(
                "grounded edit entities lack intervention_subject roles: "
                + ", ".join(sorted(edited_entities - role_entities))
            )
        if (
            self.edit_extraction_complete
            and role_entities
            and not self.has_scene_edit
        ):
            errors.append(
                "intervention_subject roles require an explicit scene edit"
            )
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bindings": [binding.to_dict() for binding in self.bindings],
            "explicit_scene_edit": self.explicit_scene_edit,
            "primitive_interventions": [
                item.to_dict() for item in self.primitive_interventions
            ],
            "relational_constraints": [
                item.to_dict() for item in self.relational_constraints
            ],
            "ambiguities": list(self.ambiguities),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "SemanticQueryGrounding":
        complete = all(
            key in payload
            for key in (
                "explicit_scene_edit",
                "primitive_interventions",
                "relational_constraints",
            )
        )
        primitives = tuple(
            item
            if isinstance(item, PrimitiveIntervention)
            else PrimitiveIntervention.from_dict(item)
            for item in payload.get("primitive_interventions", ())
        )
        constraints = tuple(
            item
            if isinstance(item, RelationalConstraint)
            else RelationalConstraint.from_dict(item)
            for item in payload.get("relational_constraints", ())
        )
        bindings = tuple(
            item
            if isinstance(item, EntityRoleBinding)
            else EntityRoleBinding.from_dict(item)
            for item in payload.get("bindings", ())
        )
        value = cls(
            bindings=bindings,
            explicit_scene_edit=(
                bool(payload.get("explicit_scene_edit"))
                if complete
                else bool(primitives or constraints)
            ),
            primitive_interventions=primitives,
            relational_constraints=constraints,
            ambiguities=tuple(
                str(item) for item in payload.get("ambiguities", ())
            ),
            edit_extraction_complete=complete,
        )
        errors = value.validate()
        if errors:
            raise ValueError(
                "Invalid SemanticQueryGrounding: " + "; ".join(errors)
            )
        return value


class PrimitiveOperation(str, Enum):
    SET_ATTRIBUTE = "set_attribute"
    MOVE_ENTITY = "move_entity"
    SET_DIRECTION = "set_direction"
    SET_BATTERY = "set_battery"
    ASSIGN_TASK = "assign_task"
    CHANGE_RESOURCE_STATE = "change_resource_state"
    ENABLE_OR_DISABLE_ENTITY = "enable_or_disable_entity"
    BATCH_INTERVENTION = "batch_intervention"


class RelationOperator(str, Enum):
    ADJACENT = "adjacent"
    DISTANCE = "distance"
    BLOCKS = "blocks"
    SURROUNDS = "surrounds"
    OCCUPIES = "occupies"
    NEAREST = "nearest"
    LINE_OF_SIGHT = "line_of_sight"
    CHARGER_AVAILABLE = "charger_available"
    ALL_AGENTS = "all_agents"
    EXISTS_AGENT = "exists_agent"


@dataclass(frozen=True)
class PrimitiveIntervention:
    """One domain-neutral scene-edit primitive emitted by a Transformer."""

    operation: PrimitiveOperation
    entity_id: str
    property_name: str | None = None
    value: Any = None
    source_span: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.entity_id.strip():
            errors.append("primitive_intervention.entity_id is empty")
        if self.operation == PrimitiveOperation.SET_ATTRIBUTE and not (self.property_name or "").strip():
            errors.append("set_attribute requires property_name")
        if self.operation == PrimitiveOperation.MOVE_ENTITY and self.value is None:
            errors.append("move_entity requires a destination")
        if self.operation == PrimitiveOperation.SET_DIRECTION and self.value is None:
            errors.append("set_direction requires a direction")
        if self.operation == PrimitiveOperation.SET_BATTERY and self.value is None:
            errors.append("set_battery requires a level")
        if self.operation == PrimitiveOperation.ASSIGN_TASK and self.value is None:
            errors.append("assign_task requires a task")
        if self.operation == PrimitiveOperation.CHANGE_RESOURCE_STATE and not (self.property_name or "").strip():
            errors.append("change_resource_state requires property_name")
        if self.operation == PrimitiveOperation.BATCH_INTERVENTION:
            if not isinstance(self.value, Sequence) or isinstance(
                self.value, (str, bytes)
            ):
                errors.append("batch_intervention requires a sequence of interventions")
            else:
                for nested in self.value:
                    if isinstance(nested, PrimitiveIntervention):
                        errors.extend(nested.validate())
                    elif isinstance(nested, Mapping):
                        try:
                            PrimitiveIntervention.from_dict(nested)
                        except (KeyError, TypeError, ValueError) as exc:
                            errors.append(str(exc))
                    else:
                        errors.append(
                            "batch_intervention items must be PrimitiveIntervention objects or mappings"
                        )
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if self.operation == PrimitiveOperation.BATCH_INTERVENTION and isinstance(
            value, Sequence
        ) and not isinstance(value, (str, bytes)):
            value = [
                item.to_dict()
                if isinstance(item, PrimitiveIntervention)
                else dict(item)
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        return {
            "operation": self.operation.value,
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "value": value,
            "source_span": self.source_span,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveIntervention":
        operation = PrimitiveOperation(str(payload["operation"]).lower())
        raw_value = payload.get("value")
        if operation == PrimitiveOperation.BATCH_INTERVENTION:
            if not isinstance(raw_value, Sequence) or isinstance(
                raw_value, (str, bytes)
            ):
                raise ValueError(
                    "batch_intervention requires a sequence of interventions"
                )
            raw_value = tuple(
                item
                if isinstance(item, PrimitiveIntervention)
                else PrimitiveIntervention.from_dict(item)
                for item in raw_value
            )
        item = cls(
            operation=operation,
            entity_id=str(payload.get("entity_id", "")),
            property_name=(
                str(payload["property_name"])
                if payload.get("property_name") is not None
                else None
            ),
            value=raw_value,
            source_span=str(payload.get("source_span", "")),
            metadata=dict(payload.get("metadata", {})),
        )
        errors = item.validate()
        if errors:
            raise ValueError("Invalid PrimitiveIntervention: " + "; ".join(errors))
        return item


@dataclass(frozen=True)
class RelationalConstraint:
    """A relation that the intervention solver must make true."""

    relation: RelationOperator
    subjects: tuple[str, ...]
    value: Any = None
    comparator: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source_span: str = ""

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.subjects and self.relation not in {
            RelationOperator.ALL_AGENTS,
            RelationOperator.EXISTS_AGENT,
        }:
            errors.append(f"{self.relation.value} requires at least one subject")
        if self.relation in {
            RelationOperator.ADJACENT,
            RelationOperator.DISTANCE,
            RelationOperator.BLOCKS,
            RelationOperator.LINE_OF_SIGHT,
        } and len(self.subjects) < 2:
            errors.append(f"{self.relation.value} requires two subjects")
        if self.relation == RelationOperator.SURROUNDS and len(self.subjects) < 2:
            errors.append("surrounds requires a target and at least one surrounding agent")
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation.value,
            "subjects": list(self.subjects),
            "value": self.value,
            "comparator": self.comparator,
            "parameters": dict(self.parameters),
            "source_span": self.source_span,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationalConstraint":
        item = cls(
            relation=RelationOperator(str(payload["relation"]).lower()),
            subjects=tuple(str(value) for value in payload.get("subjects", ())),
            value=payload.get("value"),
            comparator=(
                str(payload["comparator"]) if payload.get("comparator") is not None else None
            ),
            parameters=dict(payload.get("parameters", {})),
            source_span=str(payload.get("source_span", "")),
        )
        errors = item.validate()
        if errors:
            raise ValueError("Invalid RelationalConstraint: " + "; ".join(errors))
        return item


@dataclass(frozen=True)
class SceneEditPlan:
    """Serializable, auditable and reversible description of a scene edit."""

    source_frame: int | None
    entity_references: tuple[str, ...] = ()
    primitive_interventions: tuple[PrimitiveIntervention, ...] = ()
    relational_constraints: tuple[RelationalConstraint, ...] = ()
    preserved_variables: tuple[str, ...] = ()
    simulation_horizon: int = 1
    validation_result: Mapping[str, Any] = field(default_factory=dict)
    ambiguity: tuple[str, ...] = ()
    confidence: float = 0.0

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.source_frame is not None and self.source_frame < 0:
            errors.append("source_frame must be non-negative")
        if self.simulation_horizon < 1:
            errors.append("simulation_horizon must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be in [0, 1]")
        for item in self.primitive_interventions:
            errors.extend(item.validate())
        for item in self.relational_constraints:
            errors.extend(item.validate())
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "entity_references": list(self.entity_references),
            "primitive_interventions": [
                item.to_dict() for item in self.primitive_interventions
            ],
            "relational_constraints": [
                item.to_dict() for item in self.relational_constraints
            ],
            "preserved_variables": list(self.preserved_variables),
            "simulation_horizon": self.simulation_horizon,
            "validation_result": dict(self.validation_result),
            "ambiguity": list(self.ambiguity),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SceneEditPlan":
        value = cls(
            source_frame=(
                int(payload["source_frame"])
                if payload.get("source_frame") is not None
                else None
            ),
            entity_references=tuple(
                str(item) for item in payload.get("entity_references", ())
            ),
            primitive_interventions=tuple(
                PrimitiveIntervention.from_dict(item)
                for item in payload.get("primitive_interventions", ())
            ),
            relational_constraints=tuple(
                RelationalConstraint.from_dict(item)
                for item in payload.get("relational_constraints", ())
            ),
            preserved_variables=tuple(
                str(item) for item in payload.get("preserved_variables", ())
            ),
            simulation_horizon=int(payload.get("simulation_horizon", 1)),
            validation_result=dict(payload.get("validation_result", {})),
            ambiguity=tuple(str(item) for item in payload.get("ambiguity", ())),
            confidence=float(payload.get("confidence", 0.0)),
        )
        errors = value.validate()
        if errors:
            raise ValueError("Invalid SceneEditPlan: " + "; ".join(errors))
        return value


def _primitive_intervention_entities(
    interventions: Sequence[PrimitiveIntervention],
) -> tuple[tuple[str, str], ...]:
    """Flatten edited entities while preserving their literal source spans."""

    result: list[tuple[str, str]] = []
    for item in interventions:
        if item.operation == PrimitiveOperation.BATCH_INTERVENTION:
            nested = tuple(
                value
                if isinstance(value, PrimitiveIntervention)
                else PrimitiveIntervention.from_dict(value)
                for value in item.value
            )
            result.extend(_primitive_intervention_entities(nested))
        elif item.entity_id and item.entity_id != "batch":
            result.append((item.entity_id, item.source_span))
    return tuple(result)


def _primitive_intervention_entity_ids(
    intervention: PrimitiveIntervention,
) -> tuple[str, ...]:
    """Return every concrete entity edited by one possibly batched primitive."""

    if intervention.operation != PrimitiveOperation.BATCH_INTERVENTION:
        return (
            (intervention.entity_id,)
            if intervention.entity_id and intervention.entity_id != "batch"
            else ()
        )
    nested = tuple(
        value
        if isinstance(value, PrimitiveIntervention)
        else PrimitiveIntervention.from_dict(value)
        for value in intervention.value
    )
    return tuple(
        entity_id
        for item in nested
        for entity_id in _primitive_intervention_entity_ids(item)
    )


def _parse_or_derive_entity_roles(
    payload: Any,
    *,
    prediction_targets: Sequence[str],
    primitive_interventions: Sequence[PrimitiveIntervention],
    referenced_entities: Sequence[str],
) -> tuple[EntityRoleBinding, ...]:
    """Normalize v2 role bindings and complete compact subject-based records.

    New runtime plans receive explicit Transformer-grounded roles.  The
    derivation path exists only so stored v1 plans and named test fixtures keep
    loading; runtime execution no longer selects a target by mention order.
    """

    raw_bindings = (
        payload
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes))
        else ()
    )
    ordered_entities: list[str] = []
    roles_by_entity: dict[str, list[EntityQueryRole]] = {}
    spans_by_entity: dict[str, str] = {}

    def merge(
        entity_id: str,
        roles: Sequence[EntityQueryRole],
        source_span: str = "",
    ) -> None:
        entity_id = str(entity_id).strip()
        if not entity_id:
            return
        if entity_id not in roles_by_entity:
            ordered_entities.append(entity_id)
            roles_by_entity[entity_id] = []
        for role in roles:
            if role not in roles_by_entity[entity_id]:
                roles_by_entity[entity_id].append(role)
        if source_span and not spans_by_entity.get(entity_id):
            spans_by_entity[entity_id] = str(source_span)

    for raw in raw_bindings:
        binding = (
            raw
            if isinstance(raw, EntityRoleBinding)
            else EntityRoleBinding.from_dict(raw)
        )
        merge(binding.entity_id, binding.roles, binding.source_span)

    # These structural facts are deterministic and therefore override no
    # semantic role supplied by the model; they only complete compact records.
    for entity_id in prediction_targets:
        merge(entity_id, (EntityQueryRole.PREDICTION_TARGET,))
    for entity_id, source_span in _primitive_intervention_entities(
        primitive_interventions
    ):
        merge(
            entity_id,
            (EntityQueryRole.INTERVENTION_SUBJECT,),
            source_span,
        )
    for entity_id in referenced_entities:
        merge(entity_id, (EntityQueryRole.CONTEXT_ENTITY,))

    return tuple(
        EntityRoleBinding(
            entity_id=entity_id,
            roles=tuple(roles_by_entity[entity_id]),
            source_span=spans_by_entity.get(entity_id, ""),
        )
        for entity_id in ordered_entities
    )


def _expand_compact_question_ir(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand the latency-optimized model wire format into named fields."""

    if "i" not in payload:
        return dict(payload)
    target_entity = str(payload.get("t") or "").strip()
    query_variable = str(payload.get("q") or "").strip()
    target_variables = payload.get("v") or ()
    desired_outcomes = payload.get("d") or {}
    if query_variable:
        # The UI focus entity is authoritative for an implicit subject.  Keep
        # an unqualified variable until the deterministic binder has supplied
        # that entity instead of manufacturing a leading-dot path.
        target_path = (
            f"{target_entity}.{query_variable}"
            if target_entity
            else query_variable
        )
        target_variables = (target_path,)
        desired_outcomes = (
            {target_path: payload.get("w")}
            if "w" in payload and payload.get("w") is not None
            else {}
        )
    edited_entities = {
        str(item.get("e", item.get("entity_id", "")))
        for item in payload.get("x", ()) or ()
        if isinstance(item, Mapping)
    }
    bindings: list[dict[str, Any]] = []
    for item in payload.get("b", ()) or ():
        if isinstance(item, Mapping):
            entity_id = str(item.get("e", item.get("entity_id", "")))
            roles = item.get("roles") or tuple(
                role
                for role, present in (
                    ("prediction_target", entity_id == target_entity),
                    ("intervention_subject", entity_id in edited_entities),
                    ("context_entity", entity_id != target_entity),
                )
                if present
            )
            bindings.append(
                {
                    "entity_id": entity_id,
                    "roles": roles,
                    "source_span": item.get("span", item.get("source_span", "")),
                }
            )
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            entity_id = str(item[0])
            explicit_roles = item[1] if len(item) >= 3 else ()
            roles = explicit_roles or tuple(
                role
                for role, present in (
                    ("prediction_target", entity_id == target_entity),
                    ("intervention_subject", entity_id in edited_entities),
                    ("context_entity", entity_id != target_entity),
                )
                if present
            )
            bindings.append(
                {
                    "entity_id": entity_id,
                    "roles": (
                        roles
                        if isinstance(roles, Sequence)
                        and not isinstance(roles, (str, bytes))
                        else (roles,)
                    ),
                    "source_span": item[2] if len(item) >= 3 else item[1],
                }
            )
    edits = tuple(
        {
            "operation": item.get("op", item.get("operation", "")),
            "entity_id": item.get("e", item.get("entity_id", "")),
            "property_name": item.get("p", item.get("property_name")),
            "value": item.get("val", item.get("value")),
            "source_span": item.get("span", item.get("source_span", "")),
            "metadata": item.get("metadata", {}),
        }
        if isinstance(item, Mapping)
        else item
        for item in payload.get("x", ()) or ()
    )
    relations = tuple(
        {
            "relation": item.get(
                "rel",
                item.get("relation", item.get("p", item.get("predicate", ""))),
            ),
            "subjects": item.get("s", item.get("subjects", ())),
            "value": item.get("val", item.get("value")),
            "comparator": item.get("cmp", item.get("comparator")),
            "parameters": item.get("parameters", {}),
            "source_span": item.get("span", item.get("source_span", "")),
        }
        if isinstance(item, Mapping)
        else item
        for item in payload.get("r", ()) or ()
    )
    bound_ids = {
        str(item.get("entity_id", ""))
        for item in bindings
    }
    relation_entities = tuple(
        str(entity_id)
        for relation in relations
        if isinstance(relation, Mapping)
        for entity_id in relation.get("subjects", ())
    )
    for entity_id in dict.fromkeys(
        (target_entity, *edited_entities, *relation_entities)
    ):
        if not entity_id or entity_id in bound_ids:
            continue
        bindings.append(
            {
                "entity_id": entity_id,
                "roles": tuple(
                    role
                    for role, present in (
                        ("prediction_target", entity_id == target_entity),
                        ("intervention_subject", entity_id in edited_entities),
                        ("context_entity", entity_id != target_entity),
                    )
                    if present
                ),
                "source_span": "",
            }
        )
    return {
        "intent": payload.get("i", "mixed"),
        "target_entity": target_entity,
        "frame_reference": payload.get("f"),
        "entity_roles": bindings,
        "primitive_interventions": edits,
        "relational_constraints": relations,
        "preserved_variables": payload.get("k") or (),
        "target_variables": target_variables,
        "causal_variables": payload.get("c") or (),
        "desired_outcomes": desired_outcomes,
        "assumed_outcomes": payload.get("u") or {},
        "horizon": payload.get("h", 1),
        "rollout_count": payload.get("n", 1),
        "ambiguities": payload.get("a") or (),
        "unsupported_components": payload.get("z") or (),
    }


@dataclass(frozen=True)
class QuestionIRV2:
    """Single-pass model output for free-form question understanding."""

    intent: QueryIntent
    target_entity: str
    frame_reference: int | None = None
    referenced_entities: tuple[str, ...] = ()
    entity_roles: tuple[EntityRoleBinding, ...] = ()
    primitive_interventions: tuple[PrimitiveIntervention, ...] = ()
    relational_constraints: tuple[RelationalConstraint, ...] = ()
    preserved_variables: tuple[str, ...] = ()
    target_variables: tuple[str, ...] = ()
    causal_variables: tuple[str, ...] = ()
    desired_outcomes: Mapping[str, Any] = field(default_factory=dict)
    assumed_outcomes: Mapping[str, Any] = field(default_factory=dict)
    horizon: int = 1
    rollout_count: int = 1
    ambiguities: tuple[str, ...] = ()
    unsupported_components: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        selected_frame: int | None = None,
    ) -> "QuestionIRV2":
        payload = _expand_compact_question_ir(payload)
        role_payload = payload.get(
            "entity_roles",
            payload.get("bindings", ()),
        )
        explicit_roles = tuple(
            item
            if isinstance(item, EntityRoleBinding)
            else EntityRoleBinding.from_dict(item)
            for item in role_payload
        )
        role_target = next(
            (
                item.entity_id
                for item in explicit_roles
                if EntityQueryRole.PREDICTION_TARGET in item.roles
            ),
            "",
        )
        target = str(
            payload.get("target_entity")
            or role_target
            or next(iter(payload.get("subjects", ())), "")
        ).strip()
        normalized = dict(payload)
        normalized["subjects"] = [target] if target else []
        if explicit_roles:
            normalized["entity_roles"] = explicit_roles
        raw_intent = QueryIntent(
            str(normalized.get("intent", "mixed")).lower()
        )
        semantic_payload = dict(normalized)
        # Permit the environment binder to recover a literally named
        # relation that a small model misplaced in q/w.  The canonical
        # SemanticQueryIR quite correctly rejects an edit-free
        # counterfactual, but enforcing that invariant before source
        # grounding made recovery impossible.  The original intent is
        # restored below and must still validate after binding.
        if (
            raw_intent == QueryIntent.COUNTERFACTUAL
            and not semantic_payload.get("primitive_interventions")
            and not semantic_payload.get("relational_constraints")
        ):
            semantic_payload["intent"] = QueryIntent.MIXED.value
        semantic = SemanticQueryIR.from_dict(
            semantic_payload,
            selected_frame=selected_frame,
        )
        # An empty model target is valid at this boundary: the deterministic
        # environment binder may still resolve it from one literal entity
        # mention or from the UI's focused entity.  Validation happens only
        # after that binding step.
        return cls(
            intent=raw_intent,
            target_entity=target,
            frame_reference=semantic.frame_reference,
            referenced_entities=semantic.referenced_entities,
            entity_roles=semantic.entity_roles,
            primitive_interventions=semantic.primitive_interventions,
            relational_constraints=semantic.relational_constraints,
            preserved_variables=semantic.preserved_variables,
            target_variables=semantic.target_variables,
            causal_variables=semantic.causal_variables,
            desired_outcomes=semantic.desired_outcomes,
            assumed_outcomes=semantic.assumed_outcomes,
            horizon=semantic.horizon,
            rollout_count=semantic.rollout_count,
            ambiguities=semantic.ambiguities,
            unsupported_components=semantic.unsupported_components,
        )

    def to_semantic_ir(self, *, response_language: str) -> "SemanticQueryIR":
        return SemanticQueryIR(
            intent=self.intent,
            frame_reference=self.frame_reference,
            subjects=(self.target_entity,) if self.target_entity else (),
            referenced_entities=self.referenced_entities,
            entity_roles=self.entity_roles,
            primitive_interventions=self.primitive_interventions,
            relational_constraints=self.relational_constraints,
            preserved_variables=self.preserved_variables,
            target_variables=self.target_variables,
            causal_variables=self.causal_variables,
            desired_outcomes=self.desired_outcomes,
            assumed_outcomes=self.assumed_outcomes,
            horizon=self.horizon,
            rollout_count=self.rollout_count,
            response_language=response_language,
            ambiguities=self.ambiguities,
            unsupported_components=self.unsupported_components,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "question-ir.v2",
            "intent": self.intent.value,
            "target_entity": self.target_entity,
            "frame_reference": self.frame_reference,
            "referenced_entities": list(self.referenced_entities),
            "entity_roles": [item.to_dict() for item in self.entity_roles],
            "primitive_interventions": [
                item.to_dict() for item in self.primitive_interventions
            ],
            "relational_constraints": [
                item.to_dict() for item in self.relational_constraints
            ],
            "preserved_variables": list(self.preserved_variables),
            "target_variables": list(self.target_variables),
            "causal_variables": list(self.causal_variables),
            "desired_outcomes": dict(self.desired_outcomes),
            "assumed_outcomes": dict(self.assumed_outcomes),
            "horizon": self.horizon,
            "rollout_count": self.rollout_count,
            "ambiguities": list(self.ambiguities),
            "unsupported_components": list(self.unsupported_components),
        }


@dataclass(frozen=True)
class SemanticQueryIR:
    """Single, non-redundant semantic representation of a user request.

    A Transformer extracts facts into this structure.  It deliberately has no
    ``requires_*`` execution flags: those are derived later by the deterministic
    query compiler, so an edit request cannot claim that a scene edit exists
    while omitting the edit itself.
    """

    intent: QueryIntent
    frame_reference: int | None = None
    subjects: tuple[str, ...] = ()
    referenced_entities: tuple[str, ...] = ()
    entity_roles: tuple[EntityRoleBinding, ...] = ()
    primitive_interventions: tuple[PrimitiveIntervention, ...] = ()
    relational_constraints: tuple[RelationalConstraint, ...] = ()
    preserved_variables: tuple[str, ...] = ()
    target_variables: tuple[str, ...] = ()
    causal_variables: tuple[str, ...] = ()
    desired_outcomes: Mapping[str, Any] = field(default_factory=dict)
    assumed_outcomes: Mapping[str, Any] = field(default_factory=dict)
    horizon: int = 1
    rollout_count: int = 1
    response_language: str = "auto"
    ambiguities: tuple[str, ...] = ()
    unsupported_components: tuple[str, ...] = ()

    @property
    def has_scene_edit(self) -> bool:
        return bool(
            self.primitive_interventions
            or self.relational_constraints
        )

    def entities_with_role(self, role: EntityQueryRole) -> tuple[str, ...]:
        return tuple(
            binding.entity_id
            for binding in self.entity_roles
            if role in binding.roles
        )

    @property
    def prediction_targets(self) -> tuple[str, ...]:
        explicit = self.entities_with_role(EntityQueryRole.PREDICTION_TARGET)
        # Compatibility for serialized v1 IRs.  New production plans always
        # contain explicit bindings and never depend on subjects ordering.
        return explicit or self.subjects

    @property
    def intervention_targets(self) -> tuple[str, ...]:
        return self.entities_with_role(EntityQueryRole.INTERVENTION_SUBJECT)

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.frame_reference is not None and self.frame_reference < 0:
            errors.append("frame_reference must be non-negative")
        if self.horizon < 1:
            errors.append("horizon must be positive")
        if self.rollout_count < 1:
            errors.append("rollout_count must be positive")
        if not self.response_language.strip():
            errors.append("response_language is empty")
        for item in self.primitive_interventions:
            errors.extend(item.validate())
        for item in self.relational_constraints:
            errors.extend(item.validate())
        for binding in self.entity_roles:
            errors.extend(binding.validate())
        if self.entity_roles and not self.prediction_targets:
            errors.append("entity_roles requires a prediction_target")
        if self.intent == QueryIntent.COUNTERFACTUAL and not self.has_scene_edit:
            errors.append(
                "counterfactual intent requires at least one explicit "
                "intervention or relational constraint"
            )
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "frame_reference": self.frame_reference,
            "subjects": list(self.subjects),
            "referenced_entities": list(self.referenced_entities),
            "entity_roles": [item.to_dict() for item in self.entity_roles],
            "primitive_interventions": [
                item.to_dict() for item in self.primitive_interventions
            ],
            "relational_constraints": [
                item.to_dict() for item in self.relational_constraints
            ],
            "preserved_variables": list(self.preserved_variables),
            "target_variables": list(self.target_variables),
            "causal_variables": list(self.causal_variables),
            "desired_outcomes": dict(self.desired_outcomes),
            "assumed_outcomes": dict(self.assumed_outcomes),
            "horizon": self.horizon,
            "rollout_count": self.rollout_count,
            "response_language": self.response_language,
            "ambiguities": list(self.ambiguities),
            "unsupported_components": list(self.unsupported_components),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        selected_frame: int | None = None,
    ) -> "SemanticQueryIR":
        """Normalize both the canonical IR and compact QueryPlan-shaped output."""

        scene_payload = payload.get("scene_edit_plan")
        scene = scene_payload if isinstance(scene_payload, Mapping) else {}
        primitives = (
            payload.get("primitive_interventions")
            or payload.get("interventions")
            or scene.get("primitive_interventions")
            or ()
        )
        relations = (
            payload.get("relational_constraints")
            or payload.get("scene_constraints")
            or scene.get("relational_constraints")
            or ()
        )
        frame_value = payload.get("frame_reference")
        if frame_value is None:
            frame_value = scene.get("source_frame", selected_frame)
        ambiguity_values = list(payload.get("ambiguities", ()))
        if bool(payload.get("clarification_required", False)):
            reason = payload.get("clarification_reason")
            ambiguity_values.append(
                str(reason or "The Transformer marked the request as ambiguous.")
            )
        subjects = tuple(str(item) for item in payload.get("subjects", ()))
        parsed_primitives = tuple(
            item
            if isinstance(item, PrimitiveIntervention)
            else PrimitiveIntervention.from_dict(item)
            for item in primitives
        )
        entity_roles = _parse_or_derive_entity_roles(
            payload.get("entity_roles", ()),
            prediction_targets=subjects,
            primitive_interventions=parsed_primitives,
            referenced_entities=tuple(
                str(item)
                for item in payload.get(
                    "referenced_entities",
                    scene.get("entity_references", ()),
                )
            ),
        )
        value = cls(
            intent=QueryIntent(str(payload.get("intent", "mixed")).lower()),
            frame_reference=(
                int(frame_value) if frame_value is not None else selected_frame
            ),
            subjects=subjects,
            referenced_entities=tuple(
                str(item)
                for item in payload.get(
                    "referenced_entities",
                    scene.get("entity_references", ()),
                )
            ),
            entity_roles=entity_roles,
            primitive_interventions=parsed_primitives,
            relational_constraints=tuple(
                item
                if isinstance(item, RelationalConstraint)
                else RelationalConstraint.from_dict(item)
                for item in relations
            ),
            preserved_variables=tuple(
                str(item)
                for item in payload.get(
                    "preserved_variables",
                    scene.get("preserved_variables", ()),
                )
            ),
            target_variables=tuple(
                str(item) for item in payload.get("target_variables", ())
            ),
            causal_variables=tuple(
                str(item) for item in payload.get("causal_variables", ())
            ),
            desired_outcomes=dict(payload.get("desired_outcomes", {})),
            assumed_outcomes=dict(payload.get("assumed_outcomes", {})),
            horizon=int(
                payload.get(
                    "horizon",
                    scene.get("simulation_horizon", 1),
                )
            ),
            rollout_count=int(payload.get("rollout_count", 1)),
            response_language=str(payload.get("response_language", "auto")),
            ambiguities=tuple(dict.fromkeys(ambiguity_values)),
            unsupported_components=tuple(
                str(item)
                for item in payload.get("unsupported_components", ())
            ),
        )
        errors = value.validate()
        if errors:
            raise ValueError("Invalid SemanticQueryIR: " + "; ".join(errors))
        return value


@dataclass(frozen=True)
class QueryPlan:
    """General execution plan for one open-ended Warehouse question."""

    raw_text: str
    intent: QueryIntent
    frame_reference: int | None = None
    subjects: tuple[str, ...] = ()
    referenced_entities: tuple[str, ...] = ()
    entity_roles: tuple[EntityRoleBinding, ...] = ()
    requires_scene_edit: bool = False
    requires_policy_query: bool = True
    requires_program_trace: bool = False
    requires_simulation: bool = False
    requires_baseline_comparison: bool = False
    scene_edit_plan: SceneEditPlan | None = None
    target_variables: tuple[str, ...] = ()
    horizon: int = 1
    rollout_count: int = 1
    evidence_requirements: tuple[str, ...] = ()
    desired_outcomes: Mapping[str, Any] = field(default_factory=dict)
    assumed_outcomes: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    clarification_required: bool = False
    clarification_reason: str | None = None
    unsupported_components: tuple[str, ...] = ()
    response_language: str = "auto"

    @property
    def interventions(self) -> tuple[PrimitiveIntervention, ...]:
        """Top-level view required by the public QueryPlan contract."""

        return (
            self.scene_edit_plan.primitive_interventions
            if self.scene_edit_plan is not None
            else ()
        )

    @property
    def scene_constraints(self) -> tuple[RelationalConstraint, ...]:
        """Top-level view of relational scene constraints."""

        return (
            self.scene_edit_plan.relational_constraints
            if self.scene_edit_plan is not None
            else ()
        )

    def entities_with_role(self, role: EntityQueryRole) -> tuple[str, ...]:
        return tuple(
            binding.entity_id
            for binding in self.entity_roles
            if role in binding.roles
        )

    @property
    def prediction_targets(self) -> tuple[str, ...]:
        explicit = self.entities_with_role(EntityQueryRole.PREDICTION_TARGET)
        return explicit or self.subjects

    @property
    def intervention_targets(self) -> tuple[str, ...]:
        return self.entities_with_role(EntityQueryRole.INTERVENTION_SUBJECT)

    @property
    def comparison_targets(self) -> tuple[str, ...]:
        return self.entities_with_role(EntityQueryRole.COMPARISON_TARGET)

    @property
    def primary_prediction_target(self) -> str | None:
        targets = self.prediction_targets
        return targets[0] if len(targets) == 1 else None

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.raw_text.strip():
            errors.append("raw_text is empty")
        if self.frame_reference is not None and self.frame_reference < 0:
            errors.append("frame_reference must be non-negative")
        if self.horizon < 1:
            errors.append("horizon must be positive")
        if self.rollout_count < 1:
            errors.append("rollout_count must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be in [0, 1]")
        if not self.response_language.strip():
            errors.append("response_language is empty")
        if self.requires_scene_edit and self.scene_edit_plan is None:
            errors.append("requires_scene_edit needs a SceneEditPlan")
        if self.scene_edit_plan is not None:
            errors.extend(self.scene_edit_plan.validate())
        has_concrete_scene_edit = bool(
            self.scene_edit_plan is not None
            and (
                self.scene_edit_plan.primitive_interventions
                or self.scene_edit_plan.relational_constraints
            )
        )
        if self.requires_scene_edit and not has_concrete_scene_edit:
            errors.append(
                "requires_scene_edit needs at least one concrete intervention "
                "or relational constraint"
            )
        if has_concrete_scene_edit and not self.requires_scene_edit:
            errors.append(
                "a concrete SceneEditPlan requires requires_scene_edit=true"
            )
        if self.intent == QueryIntent.COUNTERFACTUAL:
            if not self.requires_scene_edit or not has_concrete_scene_edit:
                errors.append(
                    "counterfactual QueryPlan requires a concrete scene edit"
                )
            if not self.requires_simulation:
                errors.append(
                    "counterfactual QueryPlan requires simulation"
                )
            if not self.requires_baseline_comparison:
                errors.append(
                    "counterfactual QueryPlan requires a baseline comparison"
                )
        if self.requires_scene_edit and not self.requires_simulation:
            errors.append("scene-edit QueryPlan requires simulation")
        if self.requires_scene_edit and not self.requires_baseline_comparison:
            errors.append("scene-edit QueryPlan requires baseline comparison")
        for binding in self.entity_roles:
            errors.extend(binding.validate())
        if self.entity_roles and self.subjects != self.prediction_targets:
            errors.append(
                "subjects must equal the explicitly bound prediction_targets"
            )
        if self.requires_policy_query and self.entity_roles:
            if len(self.prediction_targets) != 1:
                errors.append(
                    "the current policy executor requires exactly one "
                    "prediction_target"
                )
        if self.entity_roles and self.scene_edit_plan is not None:
            edited_entities = {
                entity_id
                for entity_id, _source_span in _primitive_intervention_entities(
                    self.scene_edit_plan.primitive_interventions
                )
            }
            missing_roles = edited_entities - set(self.intervention_targets)
            if missing_roles:
                errors.append(
                    "edited entities lack intervention_subject roles: "
                    + ", ".join(sorted(missing_roles))
                )
        if self.entity_roles and len(self.prediction_targets) == 1:
            target = self.prediction_targets[0]
            role_entities = {
                binding.entity_id for binding in self.entity_roles
            }
            for variable in self.target_variables:
                prefix, separator, _suffix = variable.partition(".")
                if separator and prefix in role_entities and prefix != target:
                    errors.append(
                        f"target variable {variable!r} is bound to {prefix}, "
                        f"not prediction_target {target}"
                    )
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "intent": self.intent.value,
            "frame_reference": self.frame_reference,
            "subjects": list(self.subjects),
            "referenced_entities": list(self.referenced_entities),
            "entity_roles": [item.to_dict() for item in self.entity_roles],
            "requires_scene_edit": self.requires_scene_edit,
            "requires_policy_query": self.requires_policy_query,
            "requires_program_trace": self.requires_program_trace,
            "requires_simulation": self.requires_simulation,
            "requires_baseline_comparison": self.requires_baseline_comparison,
            "scene_edit_plan": (
                self.scene_edit_plan.to_dict() if self.scene_edit_plan else None
            ),
            "interventions": [item.to_dict() for item in self.interventions],
            "scene_constraints": [
                item.to_dict() for item in self.scene_constraints
            ],
            "target_variables": list(self.target_variables),
            "horizon": self.horizon,
            "rollout_count": self.rollout_count,
            "evidence_requirements": list(self.evidence_requirements),
            "desired_outcomes": dict(self.desired_outcomes),
            "assumed_outcomes": dict(self.assumed_outcomes),
            "confidence": self.confidence,
            "clarification_required": self.clarification_required,
            "clarification_reason": self.clarification_reason,
            "unsupported_components": list(self.unsupported_components),
            "response_language": self.response_language,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        raw_text: str | None = None,
    ) -> "QueryPlan":
        scene_payload = payload.get("scene_edit_plan")
        if not isinstance(scene_payload, Mapping) and (
            payload.get("interventions")
            or payload.get("primitive_interventions")
            or payload.get("scene_constraints")
            or payload.get("relational_constraints")
        ):
            scene_payload = {
                "source_frame": payload.get("frame_reference"),
                "entity_references": payload.get("referenced_entities", ()),
                "primitive_interventions": payload.get(
                    "interventions",
                    payload.get("primitive_interventions", ()),
                ),
                "relational_constraints": payload.get(
                    "scene_constraints",
                    payload.get("relational_constraints", ()),
                ),
                "preserved_variables": payload.get("preserved_variables", ()),
                "simulation_horizon": payload.get("horizon", 1),
                "confidence": payload.get("confidence", 0.0),
            }
        subjects = tuple(str(item) for item in payload.get("subjects", ()))
        parsed_scene_edit = (
            SceneEditPlan.from_dict(scene_payload)
            if isinstance(scene_payload, Mapping)
            else None
        )
        entity_roles = _parse_or_derive_entity_roles(
            payload.get("entity_roles", ()),
            prediction_targets=subjects,
            primitive_interventions=(
                parsed_scene_edit.primitive_interventions
                if parsed_scene_edit is not None
                else ()
            ),
            referenced_entities=tuple(
                str(item) for item in payload.get("referenced_entities", ())
            ),
        )
        value = cls(
            raw_text=str(raw_text if raw_text is not None else payload.get("raw_text", "")),
            intent=QueryIntent(str(payload.get("intent", "mixed")).lower()),
            frame_reference=(
                int(payload["frame_reference"])
                if payload.get("frame_reference") is not None
                else None
            ),
            subjects=subjects,
            referenced_entities=tuple(
                str(item) for item in payload.get("referenced_entities", ())
            ),
            entity_roles=entity_roles,
            requires_scene_edit=bool(payload.get("requires_scene_edit", False)),
            requires_policy_query=bool(payload.get("requires_policy_query", True)),
            requires_program_trace=bool(payload.get("requires_program_trace", False)),
            requires_simulation=bool(payload.get("requires_simulation", False)),
            requires_baseline_comparison=bool(
                payload.get("requires_baseline_comparison", False)
            ),
            scene_edit_plan=parsed_scene_edit,
            target_variables=tuple(
                str(item) for item in payload.get("target_variables", ())
            ),
            horizon=int(payload.get("horizon", 1)),
            rollout_count=int(payload.get("rollout_count", 1)),
            evidence_requirements=tuple(
                str(item) for item in payload.get("evidence_requirements", ())
            ),
            desired_outcomes=dict(payload.get("desired_outcomes", {})),
            assumed_outcomes=dict(payload.get("assumed_outcomes", {})),
            confidence=float(payload.get("confidence", 0.0)),
            clarification_required=bool(
                payload.get("clarification_required", False)
            ),
            clarification_reason=(
                str(payload["clarification_reason"])
                if payload.get("clarification_reason") is not None
                else None
            ),
            unsupported_components=tuple(
                str(item) for item in payload.get("unsupported_components", ())
            ),
            response_language=str(
                payload.get("response_language", "auto")
            ).strip() or "auto",
        )
        errors = value.validate()
        if errors:
            raise ValueError("Invalid QueryPlan: " + "; ".join(errors))
        return value


class ClaimVerdictStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    claim_type: str
    entities: tuple[str, ...] = ()
    frame_scope: tuple[int, ...] = ()
    time_scope: str | None = None
    predicate: str | None = None
    expected_outcome: Any = None
    modality: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_type: str
    frame_id: int | None = None
    rollout_id: str | None = None
    program_branch_id: str | None = None
    intervention: Mapping[str, Any] | None = None
    observed_value: Any = None
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimVerdict:
    claim: AtomicClaim
    status: ClaimVerdictStatus
    evidence: tuple[EvidenceRecord, ...] = ()
    confidence: float = 0.0
    verifier_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "verifier_reason": self.verifier_reason,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Execution evidence passed to a language model, without evaluator labels."""

    query_plan: QueryPlan
    direct_result: Mapping[str, Any]
    state_facts: tuple[Mapping[str, Any], ...] = ()
    interventions: tuple[Mapping[str, Any], ...] = ()
    baseline_results: tuple[Mapping[str, Any], ...] = ()
    counterfactual_results: tuple[Mapping[str, Any], ...] = ()
    policy_results: Mapping[str, Any] = field(default_factory=dict)
    program_trace: tuple[Mapping[str, Any], ...] = ()
    disagreement: Mapping[str, Any] = field(default_factory=dict)
    causal_analysis: Mapping[str, Any] = field(default_factory=dict)
    why_not_recourse: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_plan": self.query_plan.to_dict(),
            "direct_result": dict(self.direct_result),
            "state_facts": [dict(item) for item in self.state_facts],
            "interventions": [dict(item) for item in self.interventions],
            "baseline_results": [dict(item) for item in self.baseline_results],
            "counterfactual_results": [
                dict(item) for item in self.counterfactual_results
            ],
            "policy_results": dict(self.policy_results),
            "program_trace": [dict(item) for item in self.program_trace],
            "disagreement": dict(self.disagreement),
            "causal_analysis": dict(self.causal_analysis),
            "why_not_recourse": dict(self.why_not_recourse),
            "uncertainty": dict(self.uncertainty),
            "limitations": list(self.limitations),
        }
