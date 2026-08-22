"""Validated, atomic and auditable scene intervention execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

from backend.adapters.base import EnvironmentAdapter, EnvironmentSnapshot, Intervention
from backend.nlp.schemas import PrimitiveIntervention, PrimitiveOperation, SceneEditPlan


@dataclass(frozen=True)
class InterventionValidation:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class StateChange:
    """One material state change produced by a scene edit."""

    path: str
    before: Any
    after: Any


@dataclass(frozen=True)
class SceneEditResult:
    """An applied edit plus everything required to inspect or roll it back."""

    original_snapshot: EnvironmentSnapshot
    edited_snapshot: EnvironmentSnapshot
    plan: SceneEditPlan
    interventions: tuple[Intervention, ...]
    changes: tuple[StateChange, ...]
    validation: InterventionValidation

    def rollback(self) -> EnvironmentSnapshot:
        """Return the immutable pre-edit checkpoint."""

        return self.original_snapshot


class InterventionEngine:
    def __init__(self, adapter: EnvironmentAdapter) -> None:
        self.adapter = adapter

    def validate(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
    ) -> InterventionValidation:
        valid, errors = self.adapter.validate_intervention(snapshot, interventions)
        return InterventionValidation(valid, errors)

    def apply(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
    ) -> EnvironmentSnapshot:
        """Apply primitive edits atomically to an immutable snapshot."""

        validation = self.validate(snapshot, interventions)
        if not validation.valid:
            raise ValueError("Invalid intervention: " + "; ".join(validation.errors))
        return self.adapter.apply_interventions(snapshot, interventions)

    def compile_plan(
        self,
        snapshot: EnvironmentSnapshot,
        plan: SceneEditPlan,
    ) -> tuple[tuple[Intervention, ...], InterventionValidation]:
        """Compile a domain-neutral plan into adapter primitives.

        Primitive edits are applied to a temporary checkpoint before relational
        constraints are solved, so a relation can refer to positions changed
        earlier in the same plan. No live environment state is mutated.
        """

        errors = list(plan.validate())
        primitives: list[Intervention] = []
        for item in plan.primitive_interventions:
            try:
                primitives.extend(self._compile_primitive(item))
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        if errors:
            return tuple(primitives), InterventionValidation(False, tuple(errors))

        primitive_validation = self.validate(snapshot, primitives)
        if not primitive_validation.valid:
            return tuple(primitives), primitive_validation
        temporary = (
            self.adapter.apply_interventions(snapshot, primitives)
            if primitives
            else self.adapter.refresh_snapshot(snapshot)
        )
        relational, relation_errors = self.adapter.compile_relational_constraints(
            temporary,
            tuple(item.to_dict() for item in plan.relational_constraints),
        )
        errors.extend(relation_errors)
        compiled = (*primitives, *tuple(relational))
        if not errors:
            final_validation = self.validate(snapshot, compiled)
            errors.extend(final_validation.errors)
        return tuple(compiled), InterventionValidation(not errors, tuple(errors))

    def apply_plan(
        self,
        snapshot: EnvironmentSnapshot,
        plan: SceneEditPlan,
    ) -> SceneEditResult:
        """Validate and commit a complete scene edit, or commit nothing."""

        interventions, validation = self.compile_plan(snapshot, plan)
        if not validation.valid:
            raise ValueError("Invalid SceneEditPlan: " + "; ".join(validation.errors))
        edited = (
            self.adapter.apply_interventions(snapshot, interventions)
            if interventions
            else self.adapter.refresh_snapshot(snapshot)
        )
        changes = _state_diff(snapshot.state, edited.state)
        preserved_errors = _preservation_errors(changes, plan.preserved_variables)
        if preserved_errors:
            raise ValueError("SceneEditPlan changed preserved variables: " + "; ".join(preserved_errors))
        return SceneEditResult(
            original_snapshot=snapshot,
            edited_snapshot=edited,
            plan=plan,
            interventions=interventions,
            changes=changes,
            validation=validation,
        )

    @staticmethod
    def _compile_primitive(item: PrimitiveIntervention) -> tuple[Intervention, ...]:
        operation = item.operation
        if operation == PrimitiveOperation.BATCH_INTERVENTION:
            if not isinstance(item.value, Sequence) or isinstance(
                item.value, (str, bytes)
            ):
                raise ValueError(
                    "batch_intervention requires a sequence of interventions"
                )
            compiled: list[Intervention] = []
            for nested in item.value:
                nested_item = (
                    nested
                    if isinstance(nested, PrimitiveIntervention)
                    else PrimitiveIntervention.from_dict(nested)
                )
                compiled.extend(InterventionEngine._compile_primitive(nested_item))
            return tuple(compiled)
        if operation == PrimitiveOperation.SET_ATTRIBUTE:
            assert item.property_name is not None
            return (Intervention(item.entity_id, item.property_name, item.value),)
        if operation == PrimitiveOperation.MOVE_ENTITY:
            return (Intervention(item.entity_id, "position", item.value),)
        if operation == PrimitiveOperation.SET_DIRECTION:
            return (Intervention(item.entity_id, "heading", item.value),)
        if operation == PrimitiveOperation.SET_BATTERY:
            return (Intervention(item.entity_id, "battery", item.value),)
        if operation == PrimitiveOperation.CHANGE_RESOURCE_STATE:
            assert item.property_name is not None
            return (Intervention(item.entity_id, item.property_name, item.value),)
        if operation == PrimitiveOperation.ENABLE_OR_DISABLE_ENTITY:
            return (Intervention(item.entity_id, item.property_name or "active", item.value),)
        if operation == PrimitiveOperation.ASSIGN_TASK:
            if isinstance(item.value, Mapping):
                return tuple(
                    Intervention(item.entity_id, str(property_name), value)
                    for property_name, value in item.value.items()
                )
            return (
                Intervention(
                    item.entity_id,
                    item.property_name or "task_priority",
                    item.value,
                ),
            )
        raise ValueError(f"Unsupported primitive operation: {operation.value}")


def _plain_state(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        # Agent lists are keyed by canonical IDs to make audit paths stable.
        if "agents" in value and isinstance(value["agents"], list):
            value = dict(value)
            value["agents"] = {
                str(item.get("agent_id", index)): item
                for index, item in enumerate(value["agents"])
            }
        return {str(key): _plain_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_state(item) for item in value]
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def _state_diff(before: Any, after: Any) -> tuple[StateChange, ...]:
    before_flat = _flatten(_plain_state(before))
    after_flat = _flatten(_plain_state(after))
    paths = sorted(set(before_flat) | set(after_flat))
    return tuple(
        StateChange(path, before_flat.get(path), after_flat.get(path))
        for path in paths
        if before_flat.get(path) != after_flat.get(path)
    )


def _preservation_errors(
    changes: Sequence[StateChange],
    preserved_variables: Sequence[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    for preserved in preserved_variables:
        normalized = preserved.strip(".")
        for change in changes:
            if change.path == normalized or change.path.endswith("." + normalized):
                errors.append(
                    f"{preserved}: {change.before!r} -> {change.after!r}"
                )
    return tuple(errors)
