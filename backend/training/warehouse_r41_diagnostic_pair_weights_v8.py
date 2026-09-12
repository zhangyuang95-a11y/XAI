"""Leakage-safe intervention-pair construction and fit weighting for v8.

The input is the row-array schema emitted by diagnostic RCPD v7.  Scene
families are supplied separately because the v7 archive stores only scene
fingerprints.  Validation action labels are deliberately outside the data
flow: validation rows receive unit weight and validation interventions never
enter pair construction.

Fit sampling has three disjoint occurrence pools: effective intervention
pairs, ordinary rows, and intervention rows that do not belong to an effective
pair.  Before pair emphasis, each pool retains one unit of total mass per
occurrence.  Its mass is balanced hierarchically across family, scene, partner,
and exact critical-group bit cell.  Action factors are estimated from fit
labels only and every leaf cell is renormalized after applying them.  The pair
pool is then multiplied as a whole.  A pair's resulting mass is split equally
between its WAIT and changed-branch endpoints; consequently a WAIT shared by
several pairs accumulates one half of every incident pair.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import fsum, isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "warehouse-r41-diagnostic-pair-weights.v8"
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
WAIT_ACTION = "WAIT"
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
DEFAULT_PAIR_POOL_MULTIPLIER = 8.0
MAX_PAIR_POOL_MULTIPLIER = 32.0

_REQUIRED = frozenset((
    "action_indices",
    "scene_fingerprints",
    "episode_ids",
    "group_bits",
    "kinds",
    "anchor_ids",
    "branch_actions",
    "physical_hashes",
    "split_validation",
))

Cell = tuple[str, str, str, int]


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "input_schema": "warehouse-r41-diagnostic-rcpd.v7 rows arrays",
        "scene_family_input": "explicit fingerprint-to-family mapping",
        "pair_validity": [
            "fit rows only",
            "same nonempty anchor, split, scene, and episode",
            "one WAIT row and one row for the selected branch action",
            "different physical hashes",
            "different frozen-Actor action indices",
        ],
        "pair_occurrence": (
            "one mass unit before balancing; final occurrence mass is divided "
            "equally between WAIT and branch endpoints"
        ),
        "balance_hierarchy": ["family", "scene", "partner", "group_bits"],
        "balance_pools": [
            "effective_pair_occurrences",
            "ordinary_rows",
            "invalid_intervention_rows",
        ],
        "pool_total_mass": "one pre-multiplier unit per occurrence",
        "pair_pool_multiplier": {
            "default": DEFAULT_PAIR_POOL_MULTIPLIER,
            "minimum_exclusive": 0.0,
            "maximum_inclusive": MAX_PAIR_POOL_MULTIPLIER,
            "application": (
                "after pair cell/action normalization and before pool merging; "
                "the complete pair occurrence and both endpoint halves are scaled"
            ),
        },
        "action_factor": (
            "inverse-square-root fit-row action frequency; pair factors use "
            "the geometric mean of their two endpoint factors"
        ),
        "post_action_cell_renormalization": True,
        "validation_weight": 1.0,
        "validation_labels_accessed": False,
        "final_rows_accessed": False,
    }


def _decode(values: np.ndarray, label: str) -> np.ndarray:
    """Decode a one-dimensional fixed-width string array without coercing rows."""
    if values.ndim != 1 or values.dtype.kind not in "SU":
        raise ValueError(label + " must be a one-dimensional byte or unicode array")
    if values.dtype.kind == "U":
        return values.astype(str, copy=True)
    try:
        return np.char.decode(values, "utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(label + " contains invalid UTF-8") from error


def _mapping_text(value: Any, label: str) -> str:
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(label + " contains invalid UTF-8") from error
    if not isinstance(value, str) or not value:
        raise ValueError(label + " must contain nonempty text")
    return value


def _family_rows(
    scenes: np.ndarray,
    fit_indices: np.ndarray,
    scene_families: Mapping[str, str] | Sequence[str],
) -> np.ndarray:
    count = len(scenes)
    result = np.full(count, "", dtype=object)
    if isinstance(scene_families, Mapping):
        lookup: dict[str, str] = {}
        for raw_scene, raw_family in scene_families.items():
            scene = _mapping_text(raw_scene, "scene_families key")
            family = _mapping_text(raw_family, "scene_families value")
            if scene in lookup and lookup[scene] != family:
                raise ValueError("scene_families assigns two families to one scene")
            lookup[scene] = family
        for index in fit_indices:
            scene = str(scenes[index])
            if scene not in lookup:
                raise ValueError("scene_families is missing fit scene " + scene)
            result[index] = lookup[scene]
        return result

    raw = np.asarray(scene_families)
    if raw.shape != (count,) or raw.dtype.kind not in "SU":
        raise ValueError(
            "scene_families must be a mapping or a row-aligned text sequence"
        )
    decoded = _decode(raw, "scene_families")
    by_scene: dict[str, str] = {}
    for index in fit_indices:
        family = _mapping_text(str(decoded[index]), "fit scene family")
        scene = str(scenes[index])
        previous = by_scene.setdefault(scene, family)
        if previous != family:
            raise ValueError("one fit scene has inconsistent family identifiers")
        result[index] = family
    return result


def _partner(episode: str) -> str:
    if ":" not in episode:
        raise ValueError("fit episode id does not encode a partner suffix")
    partner = episode.rsplit(":", 1)[1]
    if not partner:
        raise ValueError("fit episode id has an empty partner suffix")
    return partner


def _group_name(bits: int) -> str:
    if bits == 0:
        return "none"
    return "+".join(name for index, name in enumerate(GROUPS)
                    if bits & (1 << index))


def _cell(
    index: int,
    *,
    families: np.ndarray,
    scenes: np.ndarray,
    episodes: np.ndarray,
    group_bits: np.ndarray,
) -> Cell:
    return (
        str(families[index]),
        str(scenes[index]),
        _partner(str(episodes[index])),
        int(group_bits[index]),
    )


def _pair_group_bits(
    anchor: str,
    wait: int,
    branch: int,
    *,
    kinds: np.ndarray,
    anchors: np.ndarray,
    split: np.ndarray,
    scenes: np.ndarray,
    episodes: np.ndarray,
    group_bits: np.ndarray,
    ordinary_by_anchor: Mapping[str, int] | None = None,
) -> int:
    # The ordinary anchor is the pre-action critical state and is therefore the
    # preferred group identity.  A v7 train-overlap removal can omit that row;
    # in that case the endpoint union is a deterministic public fallback.
    if ordinary_by_anchor is not None:
        source_index = ordinary_by_anchor.get(anchor)
        if (source_index is not None
                and kinds[source_index] == "ordinary"
                and not split[source_index]
                and scenes[source_index] == scenes[wait]
                and episodes[source_index] == episodes[wait]):
            return int(group_bits[source_index])
    else:
        source = np.flatnonzero(
            (kinds == "ordinary")
            & (anchors == anchor)
            & (~split)
            & (scenes == scenes[wait])
            & (episodes == episodes[wait])
        )
        if len(source) == 1:
            return int(group_bits[int(source[0])])
    return int(group_bits[wait] | group_bits[branch])


def _target_by_cell(cells: Sequence[Cell], total_mass: float) -> dict[Cell, float]:
    """Allocate equal mass down family -> scene -> partner -> group leaves."""
    unique = sorted(set(cells))
    if not unique:
        return {}
    families = sorted({cell[0] for cell in unique})
    result: dict[Cell, float] = {}
    for family in families:
        family_cells = [cell for cell in unique if cell[0] == family]
        scenes = sorted({cell[1] for cell in family_cells})
        for scene in scenes:
            scene_cells = [cell for cell in family_cells if cell[1] == scene]
            partners = sorted({cell[2] for cell in scene_cells})
            for partner in partners:
                leaves = [cell for cell in scene_cells if cell[2] == partner]
                share = (
                    total_mass
                    / len(families)
                    / len(scenes)
                    / len(partners)
                    / len(leaves)
                )
                for cell in leaves:
                    result[cell] = share
    return result


def _rollup(
    units: Sequence[dict[str, Any]],
    masses: np.ndarray,
    depth: int,
) -> list[dict[str, Any]]:
    actual: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    targets: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    counts: Counter[tuple[Any, ...]] = Counter()
    for unit, mass in zip(units, masses):
        key = tuple(unit["cell"][:depth])
        actual[key] += float(mass)
        targets[key] += float(unit["target_share"])
        counts[key] += 1
    names = ("family", "scene", "partner", "group_bits")
    result = []
    for key in sorted(actual):
        row = {name: value for name, value in zip(names, key)}
        if depth == 4:
            row["group"] = _group_name(int(key[3]))
        row.update({
            "unit_count": int(counts[key]),
            "target_mass": float(targets[key]),
            "actual_mass": float(actual[key]),
        })
        result.append(row)
    return result


def _balance_units(
    units: list[dict[str, Any]],
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    contribution = np.zeros(row_count, dtype=np.float64)
    if not units:
        empty = {
            "unit_count": 0,
            "target_total_mass": 0.0,
            "actual_total_mass": 0.0,
            "family_totals": [],
            "scene_totals": [],
            "partner_totals": [],
            "cell_totals": [],
            "post_action_cell_renormalized": True,
        }
        return contribution, np.empty(0, dtype=np.float64), empty

    total_mass = float(len(units))
    targets = _target_by_cell([unit["cell"] for unit in units], total_mass)
    grouped: defaultdict[Cell, list[int]] = defaultdict(list)
    for unit_index, unit in enumerate(units):
        grouped[unit["cell"]].append(unit_index)

    masses = np.zeros(len(units), dtype=np.float64)
    cell_details: dict[Cell, dict[str, float | int]] = {}
    for cell, indices in sorted(grouped.items()):
        raw = np.asarray(
            [float(units[index]["action_factor"]) for index in indices],
            dtype=np.float64,
        )
        raw_total = float(raw.sum())
        if not np.isfinite(raw_total) or raw_total <= 0.0:
            raise ValueError("fit action factors produced an invalid cell mass")
        target = float(targets[cell])
        scale = target / raw_total
        masses[np.asarray(indices, dtype=np.int64)] = raw * scale
        cell_details[cell] = {
            "pre_action_factor_mass": raw_total,
            "post_action_normalization": scale,
            "target_mass": target,
            "unit_count": len(indices),
        }

    for unit, mass in zip(units, masses):
        unit["target_share"] = targets[unit["cell"]] / len(grouped[unit["cell"]])
        for row, share in unit["row_shares"]:
            contribution[int(row)] += float(mass) * float(share)

    cell_totals = _rollup(units, masses, 4)
    for row in cell_totals:
        details = cell_details[(
            str(row["family"]), str(row["scene"]), str(row["partner"]),
            int(row["group_bits"]),
        )]
        row["pre_action_factor_mass"] = details["pre_action_factor_mass"]
        row["post_action_normalization"] = details["post_action_normalization"]
    audit = {
        "unit_count": len(units),
        "target_total_mass": total_mass,
        "actual_total_mass": float(masses.sum()),
        "family_totals": _rollup(units, masses, 1),
        "scene_totals": _rollup(units, masses, 2),
        "partner_totals": _rollup(units, masses, 3),
        "cell_totals": cell_totals,
        "post_action_cell_renormalized": all(
            np.isclose(row["target_mass"], row["actual_mass"], rtol=0.0, atol=1e-12)
            for row in cell_totals
        ),
    }
    return contribution, masses, audit


def _aggregate_records(
    pools: Sequence[tuple[str, Sequence[dict[str, Any]], np.ndarray]],
    depth: int,
) -> list[dict[str, Any]]:
    totals: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    components: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    names = ("family", "scene", "partner", "group_bits")
    for pool_name, units, masses in pools:
        for unit, mass in zip(units, masses):
            key = tuple(unit["cell"][:depth])
            totals[key] += float(mass)
            components[key][pool_name] += float(mass)
    rows = []
    for key in sorted(totals):
        row = {name: value for name, value in zip(names, key)}
        if depth == 4:
            row["group"] = _group_name(int(key[3]))
        row["mass"] = float(totals[key])
        row["components"] = {
            name: float(value) for name, value in sorted(components[key].items())
        }
        rows.append(row)
    return rows


def build_pair_weights(
    arrays: Mapping[str, np.ndarray],
    *,
    scene_families: Mapping[str, str] | Sequence[str],
    use_action_factor: bool = True,
    pair_pool_multiplier: float = DEFAULT_PAIR_POOL_MULTIPLIER,
) -> dict[str, Any]:
    """Build v8 fit weights and a complete, JSON-safe audit.

    The result has ``weights`` (float32), ``pairs`` (int64 ``[P, 2]``),
    ``pair_contribution`` (float32 per input row), and ``audit``.  A validation
    action value is never indexed, validated, summarized, or returned.
    """
    if not isinstance(arrays, Mapping) or not _REQUIRED.issubset(arrays):
        missing = sorted(_REQUIRED - set(arrays) if isinstance(arrays, Mapping)
                         else _REQUIRED)
        raise ValueError("v7 row arrays are missing fields: " + ", ".join(missing))
    if type(use_action_factor) is not bool:
        raise ValueError("use_action_factor must be a bool")
    if (isinstance(pair_pool_multiplier, (bool, np.bool_))
            or not np.isscalar(pair_pool_multiplier)):
        raise ValueError("pair_pool_multiplier must be a finite positive scalar")
    pair_pool_multiplier = float(pair_pool_multiplier)
    if (not isfinite(pair_pool_multiplier) or pair_pool_multiplier <= 0.0
            or pair_pool_multiplier > MAX_PAIR_POOL_MULTIPLIER):
        raise ValueError(
            "pair_pool_multiplier must be finite and in the interval (0, 32]"
        )

    raw_split = np.asarray(arrays["split_validation"])
    raw_labels = np.asarray(arrays["action_indices"])
    if raw_split.ndim != 1 or raw_split.dtype != np.dtype(np.bool_):
        raise ValueError("split_validation must be a one-dimensional bool array")
    count = len(raw_split)
    if count == 0 or raw_labels.shape != (count,) or raw_labels.dtype.kind not in "iu":
        raise ValueError("action_indices must be a row-aligned integer array")

    decoded: dict[str, np.ndarray] = {}
    for name in ("scene_fingerprints", "episode_ids", "kinds", "anchor_ids",
                 "branch_actions", "physical_hashes"):
        value = np.asarray(arrays[name])
        if value.shape != (count,):
            raise ValueError(name + " must be row aligned")
        decoded[name] = _decode(value, name)
    raw_groups = np.asarray(arrays["group_bits"])
    if raw_groups.shape != (count,) or raw_groups.dtype.kind not in "iu":
        raise ValueError("group_bits must be a row-aligned integer array")

    split = raw_split.copy()
    fit_indices = np.flatnonzero(~split)
    validation_indices = np.flatnonzero(split)
    # This is the sole label read.  Advanced indexing selects fit rows before
    # any value validation or aggregate is computed.
    fit_labels = raw_labels[fit_indices].astype(np.int64, copy=True)
    if np.any(fit_labels < 0) or np.any(fit_labels >= len(ACTIONS)):
        raise ValueError("fit action_indices are outside the Actor action space")
    fit_label_by_row = {
        int(row): int(label) for row, label in zip(fit_indices, fit_labels)
    }

    groups_fit = raw_groups[fit_indices].astype(np.int64, copy=False)
    if np.any(groups_fit < 0) or np.any(groups_fit >= 1 << len(GROUPS)):
        raise ValueError("fit group_bits are outside the diagnostic group mask")

    scenes = decoded["scene_fingerprints"]
    episodes = decoded["episode_ids"]
    kinds = decoded["kinds"]
    anchors = decoded["anchor_ids"]
    branches = decoded["branch_actions"]
    physical = decoded["physical_hashes"]
    families = _family_rows(scenes, fit_indices, scene_families)

    if np.any(~np.isin(kinds, ("ordinary", "intervention"))):
        raise ValueError("kinds contains a value outside the v7 row schema")
    ordinary_fit = fit_indices[kinds[fit_indices] == "ordinary"]
    intervention_fit = fit_indices[kinds[fit_indices] == "intervention"]
    if (np.any(branches[ordinary_fit] != "")
            or np.any(anchors[intervention_fit] == "")
            or np.any(~np.isin(branches[intervention_fit], ACTIONS))):
        raise ValueError("fit row kind, anchor, or branch metadata is malformed")
    # Force partner parsing for all fit rows before constructing any cell.
    for index in fit_indices:
        _partner(str(episodes[index]))

    action_counts = np.bincount(fit_labels, minlength=len(ACTIONS)).astype(np.float64)
    if use_action_factor:
        action_factors = np.sqrt(
            max(float(len(fit_labels)), 1.0) / np.maximum(action_counts, 1.0)
        )
        used = action_counts > 0
        if np.any(used):
            action_factors /= float(
                np.average(action_factors[used], weights=action_counts[used])
            )
    else:
        action_factors = np.ones(len(ACTIONS), dtype=np.float64)

    by_anchor: defaultdict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(kinds == "intervention"):
        by_anchor[str(anchors[index])].append(int(index))
    # Each critical ordinary row and its intervention endpoints share an
    # anchor.  Index these sources once: scanning the complete archive for
    # every effective pair is quadratic on the full development evidence.
    ordinary_by_anchor: dict[str, int] = {}
    duplicate_ordinary_anchors: set[str] = set()
    for index in ordinary_fit:
        anchor = str(anchors[index])
        if not anchor:
            continue
        if anchor in ordinary_by_anchor:
            duplicate_ordinary_anchors.add(anchor)
        else:
            ordinary_by_anchor[anchor] = int(index)
    for anchor in duplicate_ordinary_anchors:
        ordinary_by_anchor.pop(anchor, None)

    pairs: list[tuple[int, int]] = []
    pair_cells: list[Cell] = []
    invalid_reasons: defaultdict[int, set[str]] = defaultdict(set)
    rejected_candidates: Counter[str] = Counter()

    for anchor in sorted(by_anchor):
        rows = by_anchor[anchor]
        fit_rows = [index for index in rows if not split[index]]
        if not fit_rows:
            # Pure-validation anchors are intentionally ignored before labels.
            continue
        anchor_splits = {bool(split[index]) for index in rows}
        anchor_scenes = {str(scenes[index]) for index in rows}
        anchor_episodes = {str(episodes[index]) for index in rows}
        structural_reason = None
        if len(anchor_splits) != 1:
            structural_reason = "anchor_cross_split"
        elif len(anchor_scenes) != 1:
            structural_reason = "anchor_cross_scene"
        elif len(anchor_episodes) != 1:
            structural_reason = "anchor_cross_episode"
        if structural_reason is not None:
            rejected_candidates[structural_reason] += len(fit_rows)
            for index in fit_rows:
                invalid_reasons[index].add(structural_reason)
            continue

        indexed: defaultdict[str, list[int]] = defaultdict(list)
        for index in fit_rows:
            indexed[str(branches[index])].append(index)
        waits = indexed.get(WAIT_ACTION, [])
        if len(waits) != 1:
            reason = "missing_wait" if not waits else "duplicate_wait"
            rejected_candidates[reason] += max(
                1, sum(len(values) for action, values in indexed.items()
                       if action != WAIT_ACTION)
            )
            for index in fit_rows:
                invalid_reasons[index].add(reason)
            continue
        wait = waits[0]

        for action in ACTIONS[:-1]:
            candidates = indexed.get(action, [])
            if not candidates:
                continue
            if len(candidates) != 1:
                rejected_candidates["duplicate_branch"] += len(candidates)
                for index in candidates:
                    invalid_reasons[index].add("duplicate_branch")
                continue
            branch = candidates[0]
            if str(physical[wait]) == str(physical[branch]):
                rejected_candidates["unchanged_physical_hash"] += 1
                invalid_reasons[branch].add("unchanged_physical_hash")
                continue
            if fit_label_by_row[wait] == fit_label_by_row[branch]:
                rejected_candidates["unchanged_actor_action"] += 1
                invalid_reasons[branch].add("unchanged_actor_action")
                continue
            bits = _pair_group_bits(
                anchor, wait, branch, kinds=kinds, anchors=anchors, split=split,
                scenes=scenes, episodes=episodes, group_bits=raw_groups,
                ordinary_by_anchor=ordinary_by_anchor,
            )
            pair_cell = (
                str(families[wait]), str(scenes[wait]),
                _partner(str(episodes[wait])), bits,
            )
            pairs.append((wait, branch))
            pair_cells.append(pair_cell)

    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    paired_rows = set(map(int, pair_array.reshape(-1))) if len(pair_array) else set()
    invalid_indices = sorted(set(map(int, intervention_fit)) - paired_rows)
    for index in invalid_indices:
        if not invalid_reasons[index]:
            if branches[index] == WAIT_ACTION:
                invalid_reasons[index].add("wait_without_valid_pair")
            else:
                invalid_reasons[index].add("branch_without_valid_pair")

    pair_units: list[dict[str, Any]] = []
    for pair_index, ((wait, branch), pair_cell) in enumerate(zip(pairs, pair_cells)):
        pair_factor = sqrt(
            float(action_factors[fit_label_by_row[wait]])
            * float(action_factors[fit_label_by_row[branch]])
        )
        pair_units.append({
            "pair_index": pair_index,
            "cell": pair_cell,
            "row_shares": ((wait, 0.5), (branch, 0.5)),
            "action_factor": pair_factor,
        })
    ordinary_units = [{
        "cell": _cell(index, families=families, scenes=scenes,
                      episodes=episodes, group_bits=raw_groups),
        "row_shares": ((int(index), 1.0),),
        "action_factor": float(action_factors[fit_label_by_row[int(index)]]),
    } for index in ordinary_fit]
    invalid_units = [{
        "cell": _cell(index, families=families, scenes=scenes,
                      episodes=episodes, group_bits=raw_groups),
        "row_shares": ((int(index), 1.0),),
        "action_factor": float(action_factors[fit_label_by_row[int(index)]]),
    } for index in invalid_indices]

    balanced_pair_contribution, pair_masses, pair_balance = _balance_units(
        pair_units, count
    )
    pair_contribution = balanced_pair_contribution * pair_pool_multiplier
    weighted_pair_masses = pair_masses * pair_pool_multiplier
    pair_balance["pre_balance_unit_mass"] = 1.0
    pair_balance["pre_multiplier_total_mass"] = float(pair_masses.sum())
    pair_balance["pair_pool_multiplier"] = pair_pool_multiplier
    pair_balance["weighted_total_mass"] = float(weighted_pair_masses.sum())
    for rollup in (
        "family_totals", "scene_totals", "partner_totals", "cell_totals",
    ):
        for row in pair_balance[rollup]:
            row["weighted_target_mass"] = (
                float(row["target_mass"]) * pair_pool_multiplier
            )
            row["weighted_actual_mass"] = (
                float(row["actual_mass"]) * pair_pool_multiplier
            )
    ordinary_contribution, ordinary_masses, ordinary_balance = _balance_units(
        ordinary_units, count
    )
    invalid_contribution, invalid_masses, invalid_balance = _balance_units(
        invalid_units, count
    )
    weights64 = pair_contribution + ordinary_contribution + invalid_contribution
    weights64[validation_indices] = 1.0
    if not np.isfinite(weights64).all() or np.any(weights64 <= 0.0):
        raise ValueError("v8 pair weighting did not assign every row positive mass")

    wait_degree: Counter[int] = Counter(int(pair[0]) for pair in pairs)
    occurrences = []
    symmetric = True
    for pair_index, ((wait, branch), balanced_mass, weighted_mass, unit) in enumerate(
        zip(pairs, pair_masses, weighted_pair_masses, pair_units)
    ):
        half = float(weighted_mass) / 2.0
        check = np.isclose(
            half + half, float(weighted_mass), rtol=0.0, atol=1e-15
        )
        symmetric = symmetric and bool(check)
        cell = unit["cell"]
        occurrences.append({
            "pair_index": pair_index,
            "wait_row": int(wait),
            "branch_row": int(branch),
            "branch_action": str(branches[branch]),
            "pre_balance_unit_mass": 1.0,
            "balanced_occurrence_mass": float(balanced_mass),
            "pair_pool_multiplier": pair_pool_multiplier,
            "weighted_occurrence_mass": float(weighted_mass),
            "wait_contribution": half,
            "branch_contribution": half,
            "symmetric": bool(check),
            "family": cell[0],
            "scene": cell[1],
            "partner": cell[2],
            "group_bits": cell[3],
            "group": _group_name(cell[3]),
        })

    pair_mass_total = float(fsum(map(float, weighted_pair_masses)))
    endpoint_mass_total = float(fsum(map(float, pair_contribution)))
    mass_symmetry_tolerance = max(
        1e-12,
        8.0 * np.finfo(np.float64).eps
        * max(1.0, abs(pair_mass_total), abs(endpoint_mass_total)),
    )

    pools = (
        ("effective_pairs", pair_units, weighted_pair_masses),
        ("ordinary", ordinary_units, ordinary_masses),
        ("invalid_intervention", invalid_units, invalid_masses),
    )
    balance_totals = {
        "family_totals": _aggregate_records(pools, 1),
        "scene_totals": _aggregate_records(pools, 2),
        "partner_totals": _aggregate_records(pools, 3),
        "cell_totals": _aggregate_records(pools, 4),
    }
    audit = {
        "version": VERSION,
        "rows": count,
        "fit_rows": int(len(fit_indices)),
        "validation_rows": int(len(validation_indices)),
        "validation_weight": 1.0,
        "validation_labels_accessed": False,
        "validation_labels_used_for_pairs": False,
        "validation_labels_used_for_action_factors": False,
        "action_factor": {
            "enabled": use_action_factor,
            "source": "fit action_indices only",
            "fit_action_counts": {
                action: int(action_counts[index])
                for index, action in enumerate(ACTIONS)
            },
            "factors": {
                action: float(action_factors[index])
                for index, action in enumerate(ACTIONS)
            },
            "pair_factor": "geometric mean of endpoint factors",
            "cell_renormalized_after_factor": True,
        },
        "pair_count": len(pairs),
        "paired_endpoint_rows": len(paired_rows),
        "pair_endpoint_occurrences": 2 * len(pairs),
        "pair_endpoint_contribution_total": endpoint_mass_total,
        "pair_pre_balance_unit_mass": 1.0,
        "pair_pre_multiplier_mass_total": float(pair_masses.sum()),
        "pair_pool_multiplier": pair_pool_multiplier,
        "pair_occurrence_mass_total": pair_mass_total,
        "weighted_pair_total_mass": pair_mass_total,
        "shared_wait_rows": int(sum(degree > 1 for degree in wait_degree.values())),
        "maximum_wait_degree": int(max(wait_degree.values(), default=0)),
        "wait_degrees": {
            str(index): int(degree) for index, degree in sorted(wait_degree.items())
        },
        "pair_occurrences": occurrences,
        "symmetry_checks": {
            "every_pair_has_equal_endpoint_contribution": symmetric,
            "pair_mass_equals_endpoint_contribution_total": bool(
                abs(pair_mass_total - endpoint_mass_total)
                <= mass_symmetry_tolerance
            ),
            "all_pairs_same_anchor_split_scene_episode": all(
                anchors[wait] == anchors[branch]
                and split[wait] == split[branch]
                and scenes[wait] == scenes[branch]
                and episodes[wait] == episodes[branch]
                for wait, branch in pairs
            ),
            "all_pairs_change_physical_hash_and_actor_action": all(
                physical[wait] != physical[branch]
                and fit_label_by_row[wait] != fit_label_by_row[branch]
                for wait, branch in pairs
            ),
        },
        "pair_mass_symmetry_absolute_tolerance": mass_symmetry_tolerance,
        "ordinary_fit_rows": len(ordinary_units),
        "invalid_intervention_fit_rows": len(invalid_units),
        "invalid_intervention_rows": invalid_indices,
        "invalid_reasons_by_row": {
            str(index): sorted(invalid_reasons[index]) for index in invalid_indices
        },
        "rejected_pair_candidates": {
            reason: int(value) for reason, value in sorted(rejected_candidates.items())
        },
        "balance": {
            "hierarchy": ["family", "scene", "partner", "group_bits"],
            "effective_pairs": pair_balance,
            "ordinary": ordinary_balance,
            "invalid_intervention": invalid_balance,
            "combined_training": balance_totals,
        },
        "fit_weight_sum": float(weights64[fit_indices].sum()),
        "validation_weight_sum": float(weights64[validation_indices].sum()),
        "total_weight_sum": float(weights64.sum()),
    }
    if not all(audit["symmetry_checks"].values()):
        raise ValueError("v8 pair weighting symmetry audit failed")
    return {
        "weights": weights64.astype(np.float32),
        "pairs": pair_array,
        "pair_contribution": pair_contribution.astype(np.float32),
        "audit": audit,
    }


# Short alias for callers that treat this module as a pure builder.
build = build_pair_weights


__all__ = [
    "ACTIONS", "DEFAULT_PAIR_POOL_MULTIPLIER", "GROUPS",
    "MAX_PAIR_POOL_MULTIPLIER", "VERSION", "build", "build_pair_weights",
    "contract",
]
