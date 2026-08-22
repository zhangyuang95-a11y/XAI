"""Pure sample-splitting and bounded-program tree utilities for RCPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .program import ProgramNode
from .rcpd_config import OracleOutput


State = Any
FeatureVector = Mapping[str, float]
PredicateGroupContract = Mapping[str, Sequence[str]]


@dataclass
class _Sample:
    state: State
    features: dict[str, float]
    oracle: OracleOutput
    weight: float
    groups: tuple[str, ...] = ("ordinary",)
    pair_id: str | None = None
    split_group: str | None = None


def _counterfactual_changed_training_indices(
    samples: Sequence[_Sample],
    training_indices: Sequence[int] | np.ndarray,
    action_names: Sequence[str],
) -> np.ndarray:
    """Return both endpoints of training pairs whose NN argmax differs."""

    grouped: dict[str, list[int]] = {}
    for raw_index in training_indices:
        index = int(raw_index)
        pair_id = samples[index].pair_id
        if pair_id:
            grouped.setdefault(pair_id, []).append(index)
    selected: list[int] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        preferred_actions = {
            int(
                np.argmax(
                    samples[index].oracle.normalized(action_names)
                )
            )
            for index in indices
        }
        if len(preferred_actions) > 1:
            selected.extend(indices)
    return np.asarray(sorted(set(selected)), dtype=int)


def _linked_sample_groups(
    samples: Sequence[_Sample],
) -> dict[str, list[int]]:
    """Create connected components for episode and counterfactual links."""

    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, sample in enumerate(samples):
        links = []
        if sample.split_group:
            links.append(("split", sample.split_group))
        if sample.pair_id:
            links.append(("pair", sample.pair_id))
        for link in links:
            if link in seen:
                union(index, seen[link])
            else:
                seen[link] = index
    components: dict[str, list[int]] = {}
    for index in range(len(samples)):
        components.setdefault(f"component:{find(index)}", []).append(index)
    return components


def _ensure_required_semantic_training_variation(
    samples: Sequence[_Sample],
    validation: np.ndarray,
    required_groups: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Keep at least one varying example for each required group in training.

    Linked episodes and counterfactual pairs stay indivisible.  Rare semantic
    probes can otherwise all land in held-out validation, leaving the fitting
    partition mathematically unable to place a required split.  Moving a
    complete linked component to training is not leakage; it simply changes
    which intact episode is held out.
    """

    if not required_groups or not len(validation):
        return validation
    components = tuple(_linked_sample_groups(samples).values())
    validation_set = {int(index) for index in validation.tolist()}

    def training_rows(excluding: set[int] | None = None) -> list[_Sample]:
        omitted = validation_set if excluding is None else excluding
        return [
            sample
            for index, sample in enumerate(samples)
            if index not in omitted
        ]

    for features in required_groups.values():
        if any(
            _feature_varies(training_rows(), str(feature))
            for feature in features
        ):
            continue
        for component in sorted(components, key=lambda rows: (len(rows), rows)):
            component_set = {int(index) for index in component}
            if not component_set.issubset(validation_set):
                continue
            proposed_validation = validation_set - component_set
            proposed_training = training_rows(proposed_validation)
            if any(
                _feature_varies(proposed_training, str(feature))
                for feature in features
            ):
                validation_set = proposed_validation
                break

    if not validation_set:
        # Restore the smallest complete component that can be held out without
        # destroying any of the just-established semantic variation.
        for component in sorted(components, key=lambda rows: (len(rows), rows)):
            component_set = {int(index) for index in component}
            proposed_training = [
                sample
                for index, sample in enumerate(samples)
                if index not in component_set
            ]
            if all(
                any(
                    _feature_varies(proposed_training, str(feature))
                    for feature in features
                )
                for features in required_groups.values()
            ):
                validation_set = component_set
                break
    return np.asarray(sorted(validation_set), dtype=int)


def _stratified_validation_indices(
    samples: Sequence[_Sample],
    *,
    validation_size: int,
    random_seed: int,
    interaction_groups: Sequence[str],
    minimum_group_samples: int,
) -> np.ndarray:
    """Choose linked validation components with causal/interaction coverage.

    Episode and counterfactual links remain indivisible.  A plain random
    component split can nevertheless fill the entire validation budget with
    large ordinary-trajectory components, leaving no paired intervention or
    rare interaction evidence.  Candidate trees would then be selected using
    only ordinary one-step fidelity even though they are later used to explain
    conflicts and counterfactuals.

    This splitter first covers available interaction deficits and at least one
    counterfactual pair, preferring components that provide the most needed
    evidence per row.  Remaining capacity is filled deterministically at
    random.  It does not split a linked component and always preserves at
    least one training component when the dataset permits it.
    """

    if not samples:
        return np.asarray([], dtype=int)
    components = _linked_sample_groups(samples)
    keys = list(components)
    rng = np.random.default_rng(int(random_seed))
    rng.shuffle(keys)
    interaction_set = {str(value) for value in interaction_groups}
    total_by_group = {
        group: sum(group in sample.groups for sample in samples)
        for group in interaction_set
    }
    desired_by_group = {
        group: min(
            total,
            max(1, int(minimum_group_samples)),
        )
        for group, total in total_by_group.items()
        if total > 0
    }
    pair_available = any(sample.pair_id for sample in samples)
    selected: list[str] = []
    selected_rows: set[int] = set()
    group_counts = {group: 0 for group in desired_by_group}
    pair_covered = False

    def contribution(key: str) -> tuple[float, int, int]:
        indices = components[key]
        group_gain = sum(
            min(
                max(0, desired_by_group[group] - group_counts[group]),
                sum(group in samples[index].groups for index in indices),
            )
            for group in desired_by_group
        )
        contains_pair = any(samples[index].pair_id for index in indices)
        pair_gain = int(pair_available and not pair_covered and contains_pair)
        # A paired component is essential for measuring causal direction, so
        # give it the same value as one complete interaction quota.
        weighted_gain = group_gain + pair_gain * max(
            1,
            int(minimum_group_samples),
        )
        density = weighted_gain / max(1, len(indices))
        return density, weighted_gain, -len(indices)

    while True:
        unmet_group = any(
            group_counts[group] < desired
            for group, desired in desired_by_group.items()
        )
        unmet_pair = pair_available and not pair_covered
        if not unmet_group and not unmet_pair:
            break
        candidates = [key for key in keys if key not in selected]
        if len(candidates) <= 1 and selected:
            break
        best_key = max(candidates, key=contribution, default=None)
        if best_key is None or contribution(best_key)[1] <= 0:
            break
        best_rows = set(components[best_key])
        if len(selected_rows | best_rows) >= len(samples) and selected:
            break
        selected.append(best_key)
        selected_rows.update(best_rows)
        pair_covered = pair_covered or any(
            samples[index].pair_id for index in best_rows
        )
        for group in group_counts:
            group_counts[group] += sum(
                group in samples[index].groups for index in best_rows
            )

    for key in keys:
        if len(selected_rows) >= max(1, int(validation_size)):
            break
        if key in selected:
            continue
        rows = set(components[key])
        if len(selected_rows | rows) >= len(samples) and selected:
            continue
        selected.append(key)
        selected_rows.update(rows)

    if not selected_rows:
        selected_rows.update(components[keys[0]])
    return np.asarray(sorted(selected_rows), dtype=int)


def _data_split_audit(
    samples: Sequence[_Sample],
    validation_indices: Sequence[int],
) -> dict[str, Any]:
    validation = {int(index) for index in validation_indices}
    training = set(range(len(samples))) - validation

    def values(indices: set[int], attribute: str) -> set[str]:
        return {
            str(value)
            for index in indices
            if (value := getattr(samples[index], attribute))
        }

    train_splits = values(training, "split_group")
    validation_splits = values(validation, "split_group")
    train_pairs = values(training, "pair_id")
    validation_pairs = values(validation, "pair_id")
    validation_group_counts: dict[str, int] = {}
    for index in validation:
        for group in samples[index].groups:
            validation_group_counts[group] = (
                validation_group_counts.get(group, 0) + 1
            )
    return {
        "training_samples": len(training),
        "validation_samples": len(validation),
        "split_group_overlap": sorted(
            train_splits & validation_splits
        ),
        "counterfactual_pair_overlap": sorted(
            train_pairs & validation_pairs
        ),
        "validation_counterfactual_pairs": len(validation_pairs),
        "validation_group_samples": validation_group_counts,
    }


def _feature_family(name: str) -> str:
    if name.startswith("self."):
        return "self_task"
    if name.startswith("goal."):
        return "goal_route"
    if name.startswith("charger."):
        return "charger_resource"
    if name.startswith("other.") or name.startswith("team."):
        return "multiagent_interaction"
    if name.startswith("candidate."):
        if any(
            token in name
            for token in (
                ".legal",
                ".blocked_by_",
                ".predicted_",
                ".blocker_",
                ".nearest_other_",
            )
        ):
            return "candidate_interaction"
        return "candidate_route"
    if name.startswith("time."):
        return "temporal_team"
    return "other"


def _normalize_predicate_group_contract(
    contract: PredicateGroupContract,
    feature_names: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Validate the environment-declared semantic coverage contract.

    The RCPD core deliberately knows nothing about warehouses, batteries, or
    robots.  An adapter may nevertheless require that the final bounded
    program use at least one observable predicate from each named semantic
    group.  Keeping exact feature names here makes the contract auditable and
    prevents a missing/renamed feature from silently weakening an experiment.
    """

    available = {str(name) for name in feature_names}
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_group, raw_features in contract.items():
        group = str(raw_group).strip()
        if not group:
            raise ValueError("Required predicate group names cannot be empty.")
        candidates = tuple(
            dict.fromkeys(
                str(feature).strip()
                for feature in raw_features
                if str(feature).strip()
            )
        )
        if not candidates:
            raise ValueError(
                f"Required semantic predicate group {group!r} is empty."
            )
        unavailable = tuple(
            feature for feature in candidates if feature not in available
        )
        if unavailable:
            raise ValueError(
                f"Required semantic predicate group {group!r} declares "
                "features that are absent from the extraction dataset: "
                + ", ".join(unavailable)
            )
        normalized[group] = candidates
    return normalized


def _program_predicate_group_coverage(
    program: ExecutableProgram,
    required_groups: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    used = program.root.used_predicates()
    return {
        str(group): tuple(
            feature for feature in features if str(feature) in used
        )
        for group, features in required_groups.items()
    }


def _feature_varies(samples: Sequence[_Sample], feature: str) -> bool:
    if len(samples) < 2:
        return False
    values = np.asarray(
        [sample.features.get(feature, 0.0) for sample in samples],
        dtype=float,
    )
    return bool(np.isfinite(values).all() and not np.allclose(values, values[0]))


def _candidate_feature_thresholds(
    samples: Sequence[_Sample],
    feature: str,
    *,
    maximum_thresholds: int = 16,
) -> tuple[float, ...]:
    """Return deterministic split candidates without assuming feature units."""

    values = np.asarray(
        [sample.features.get(feature, 0.0) for sample in samples],
        dtype=float,
    )
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) < 2:
        return ()
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    if len(midpoints) <= max(1, int(maximum_thresholds)):
        return tuple(float(value) for value in midpoints)
    positions = np.linspace(
        0,
        len(midpoints) - 1,
        num=max(1, int(maximum_thresholds)),
        dtype=int,
    )
    return tuple(
        dict.fromkeys(float(midpoints[int(position)]) for position in positions)
    )


def _internal_program_nodes(
    root: ProgramNode,
) -> tuple[tuple[tuple[str, ...], ProgramNode], ...]:
    nodes: list[tuple[tuple[str, ...], ProgramNode]] = []

    def visit(node: ProgramNode, path: tuple[str, ...]) -> None:
        if node.is_leaf:
            return
        nodes.append((path, node))
        if node.left is not None:
            visit(node.left, (*path, "left"))
        if node.right is not None:
            visit(node.right, (*path, "right"))

    visit(root, ())
    return tuple(nodes)


def _leaf_program_nodes(
    root: ProgramNode,
) -> tuple[tuple[tuple[str, ...], ProgramNode], ...]:
    leaves: list[tuple[tuple[str, ...], ProgramNode]] = []

    def visit(node: ProgramNode, path: tuple[str, ...]) -> None:
        if node.is_leaf:
            leaves.append((path, node))
            return
        if node.left is not None:
            visit(node.left, (*path, "left"))
        if node.right is not None:
            visit(node.right, (*path, "right"))

    visit(root, ())
    return tuple(leaves)


def _replace_program_subtree(
    root: ProgramNode,
    path: Sequence[str],
    replacement: ProgramNode,
) -> ProgramNode:
    if not path:
        return replacement
    if root.is_leaf or root.left is None or root.right is None:
        raise ValueError("Program path does not identify an existing node.")
    direction = str(path[0])
    if direction == "left":
        return ProgramNode(
            feature=root.feature,
            threshold=root.threshold,
            left=_replace_program_subtree(root.left, path[1:], replacement),
            right=root.right,
        )
    if direction == "right":
        return ProgramNode(
            feature=root.feature,
            threshold=root.threshold,
            left=root.left,
            right=_replace_program_subtree(root.right, path[1:], replacement),
        )
    raise ValueError(f"Unknown program path direction: {direction!r}")


def _replace_program_node_split(
    root: ProgramNode,
    path: Sequence[str],
    *,
    feature: str,
    threshold: float,
) -> ProgramNode:
    target = root
    for direction in path:
        if target.is_leaf:
            raise ValueError("Program path ended at a leaf.")
        target = target.left if direction == "left" else target.right
        if target is None:
            raise ValueError("Program path references a missing child.")
    if target.is_leaf or target.left is None or target.right is None:
        raise ValueError("Only internal program splits can be replaced.")
    replacement = ProgramNode(
        feature=str(feature),
        threshold=float(threshold),
        left=target.left,
        right=target.right,
    )
    return _replace_program_subtree(root, path, replacement)


def _samples_reaching_path(
    root: ProgramNode,
    samples: Sequence[_Sample],
    path: Sequence[str],
) -> list[_Sample]:
    reached = list(samples)
    node = root
    for expected_direction in path:
        if node.is_leaf or node.feature is None or node.threshold is None:
            return []
        go_left = str(expected_direction) == "left"
        reached = [
            sample
            for sample in reached
            if (
                float(sample.features.get(node.feature, 0.0))
                <= float(node.threshold)
            )
            == go_left
        ]
        node = node.left if go_left else node.right
        if node is None:
            return []
    return reached


def _reestimate_program_leaves(
    root: ProgramNode,
    samples: Sequence[_Sample],
    action_names: Sequence[str],
) -> ProgramNode:
    """Refit leaf distributions from current NN outputs, never invented labels."""

    def rebuild(node: ProgramNode, rows: Sequence[_Sample]) -> ProgramNode:
        if node.is_leaf:
            if not rows:
                return node
            weights = np.asarray(
                [max(1e-8, float(sample.weight)) for sample in rows],
                dtype=float,
            )
            targets = np.asarray(
                [sample.oracle.normalized(action_names) for sample in rows],
                dtype=float,
            )
            probabilities = np.average(targets, axis=0, weights=weights)
            probabilities = probabilities / max(
                1e-12,
                float(probabilities.sum()),
            )
            return ProgramNode(
                probabilities=tuple(float(value) for value in probabilities)
            )
        if (
            node.feature is None
            or node.threshold is None
            or node.left is None
            or node.right is None
        ):
            raise ValueError("Malformed program node during leaf refit.")
        left_rows: list[_Sample] = []
        right_rows: list[_Sample] = []
        for sample in rows:
            destination = (
                left_rows
                if float(sample.features.get(node.feature, 0.0))
                <= float(node.threshold)
                else right_rows
            )
            destination.append(sample)
        return ProgramNode(
            feature=node.feature,
            threshold=node.threshold,
            left=rebuild(node.left, left_rows),
            right=rebuild(node.right, right_rows),
        )

    return rebuild(root, samples)


def _minimum_program_leaf_samples(
    root: ProgramNode,
    samples: Sequence[_Sample],
) -> int:
    counts: list[int] = []

    def route(node: ProgramNode, rows: Sequence[_Sample]) -> None:
        if node.is_leaf:
            counts.append(len(rows))
            return
        if (
            node.feature is None
            or node.threshold is None
            or node.left is None
            or node.right is None
        ):
            counts.append(0)
            return
        left_rows: list[_Sample] = []
        right_rows: list[_Sample] = []
        for sample in rows:
            destination = (
                left_rows
                if float(sample.features.get(node.feature, 0.0))
                <= float(node.threshold)
                else right_rows
            )
            destination.append(sample)
        route(node.left, left_rows)
        route(node.right, right_rows)

    route(root, samples)
    return min(counts) if counts else 0


def _program_feature_allowed(name: str) -> bool:
    """Reject concrete identity selectors while retaining relational roles."""

    normalized = name.lower()
    return not (
        normalized.endswith("identity_index")
        or normalized.endswith("agent_id")
        or ".robot_" in normalized
    )


def _is_relational_predicate(name: str) -> bool:
    """Return whether a predicate actually relates two or more agents.

    ``candidate_interaction`` also contains single-agent facts such as a
    shelf blocking a move, while ``charger_resource`` contains plain geometry
    such as distance to the charger.  Counting those as relational made the
    audit overstate how much multi-agent information a fitted tree really
    used.  Keep the broader feature-family classification for feature
    coverage, but make this reported metric semantically strict.
    """

    if name.startswith(("other.", "team.")):
        return True
    if name.startswith("candidate."):
        return any(
            token in name
            for token in (
                ".blocked_by_robot",
                ".predicted_same_cell_conflict",
                ".predicted_swap_conflict",
                ".predicted_priority_loss",
                ".predicted_conflict_count",
                ".blocker_",
                ".nearest_other_",
            )
        )
    if name.startswith("charger."):
        return any(
            token in name
            for token in (
                "occupied",
                "occupant_",
                "other_",
                "queue_",
                "self_queue_rank",
            )
        )
    return False


def _as_oracle_output(value: OracleOutput | Mapping[str, float]) -> OracleOutput:
    if isinstance(value, OracleOutput):
        return value
    return OracleOutput(probabilities={str(action): float(probability) for action, probability in value.items()})


def _kl_divergence(target: np.ndarray, approximation: np.ndarray) -> float:
    epsilon = 1e-9
    left = np.clip(np.asarray(target, dtype=float), epsilon, 1.0)
    right = np.clip(np.asarray(approximation, dtype=float), epsilon, 1.0)
    left /= left.sum()
    right /= right.sum()
    return float(np.sum(left * np.log(left / right)))


def _temperature_scale_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return ``softmax(log(probabilities) / temperature)`` row-wise."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("probabilities must be a vector or a matrix")
    rows = values.reshape(1, -1) if values.ndim == 1 else values
    rows = np.maximum(rows, 0.0)
    totals = rows.sum(axis=1, keepdims=True)
    normalized = np.divide(
        rows,
        totals,
        out=np.full_like(rows, 1.0 / max(1, rows.shape[1])),
        where=totals > 0.0,
    )
    effective_temperature = max(0.0, float(temperature))
    if effective_temperature <= 0.0:
        scaled = np.zeros_like(normalized)
        scaled[np.arange(len(normalized)), np.argmax(normalized, axis=1)] = 1.0
    elif np.isclose(effective_temperature, 1.0):
        scaled = normalized
    else:
        # Tree leaves store probabilities, so log(p) is their canonical logit
        # representation.  Scaling those logits preserves every argmax.
        logits = np.log(np.clip(normalized, 1e-8, 1.0)) / effective_temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        powered = np.exp(logits)
        scaled = powered / np.clip(
            powered.sum(axis=1, keepdims=True),
            1e-8,
            None,
        )
    return scaled.reshape(-1) if values.ndim == 1 else scaled


def _tree_to_program(
    estimator: DecisionTreeRegressor,
    feature_names: tuple[str, ...],
    action_count: int,
) -> ProgramNode:
    tree = estimator.tree_

    def build(node_id: int) -> ProgramNode:
        feature_index = int(tree.feature[node_id])
        if feature_index < 0:
            raw = np.asarray(tree.value[node_id], dtype=float).reshape(-1)[:action_count]
            raw = np.maximum(raw, 0.0)
            probabilities = raw / raw.sum() if raw.sum() > 0 else np.full(action_count, 1.0 / action_count)
            return ProgramNode(probabilities=tuple(float(value) for value in probabilities))
        return ProgramNode(
            feature=feature_names[feature_index],
            threshold=float(tree.threshold[node_id]),
            left=build(int(tree.children_left[node_id])),
            right=build(int(tree.children_right[node_id])),
        )

    return build(0)
