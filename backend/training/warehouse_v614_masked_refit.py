"""Review-only four-purpose refit of a final frozen Actor's tree topology.

This changes Actor weights only during offline BC. It never proposes an
environment action. The separately audited explanation program still fits the
complete frozen NN distribution after self-play.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Mapping, Sequence

import numpy as np

from backend.training.warehouse_v6_core_tree import (
    CoreTreeResult,
    _masked_probabilities, _structural_legal_actions,
)
from backend.training.warehouse_v614_certificate_tree import (
    FeatureSchema, frozen_actor_topology, refit_frozen_topology_fit_only,
    to_current_core_tree_result,
)
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Role, role_sha256
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v614 import (
    TREE_FEATURE_DIM as FEATURE_DIM, TREE_FEATURE_NAMES as FEATURE_NAMES,
)


_STREAK = FEATURE_NAMES.index("same_target_collision_streak_capped")
_DISTANCE = FEATURE_NAMES.index("robots_shortest_distance")
_RETURN = FEATURE_NAMES.index("self_charger_return_margin")
_MASK_INDICES = tuple(FEATURE_NAMES.index(f"legal_{action.lower()}")
                      for action in ACTIONS)
_EPS = 1.0e-9
_EPISODE_PATTERN = re.compile(r"^(public-teacher|nn-visit):([0-9]+)$")


@dataclass(frozen=True)
class RefitConfig:
    teacher_fit_stride: int = 3
    conflict_repeat: int = 3
    max_conflict_rows_per_episode: int = 12
    max_weighted_fit_rows_per_episode: int = 96
    maximum_raw_logit_span: float = 3.5
    temperature_grid: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)
    min_leaf_rows: int = 80
    min_leaf_episodes: int = 8
    max_episode_fraction: float = 0.20
    min_global_agreement: float = 0.96
    max_global_masked_kl: float = 0.25
    min_stratum_rows: int = 8
    min_stratum_episodes: int = 3
    min_stratum_agreement: float = 0.93
    max_stratum_masked_kl: float = 0.35
    min_masked_margin: float = 0.26
    min_internal_core_mass: float = 0.35
    min_internal_expected_actionable_mass: float = 0.30

    def __post_init__(self) -> None:
        if (self.teacher_fit_stride < 1 or self.conflict_repeat < 1
                or self.max_conflict_rows_per_episode < 1
                or self.max_weighted_fit_rows_per_episode < 1
                or self.maximum_raw_logit_span <= 0
                or not self.temperature_grid
                or min(self.temperature_grid) < 1.0
                or self.min_leaf_rows < 1 or self.min_leaf_episodes < 1
                or self.min_stratum_rows < 1 or self.min_stratum_episodes < 1
                or not 0 < self.max_episode_fraction <= 1
                or not 0 < self.min_global_agreement <= 1
                or not 0 < self.min_stratum_agreement <= 1
                or self.max_global_masked_kl <= 0
                or self.max_stratum_masked_kl <= 0
                or self.min_masked_margin <= 0
                or not 0 <= self.min_internal_core_mass <= 1
                or not 0 <= self.min_internal_expected_actionable_mass <= 1):
            raise ValueError("invalid Warehouse v6.13 masked core refit configuration")


@dataclass(frozen=True)
class EpisodeRows:
    encoded: np.ndarray
    masks: np.ndarray
    teacher_actions: np.ndarray
    frozen_nn_masked_probs: np.ndarray
    actual_collision: np.ndarray
    episode_ids: tuple[str, ...]
    scenario_seeds: tuple[int, ...]
    frozen_role_sha256: str

    def __post_init__(self) -> None:
        x = np.asarray(self.encoded)
        m = np.asarray(self.masks)
        y = np.asarray(self.teacher_actions)
        p = np.asarray(self.frozen_nn_masked_probs)
        c = np.asarray(self.actual_collision)
        if (x.ndim != 2 or x.shape[1] != FEATURE_DIM
                or m.shape != (len(x), len(ACTIONS))
                or y.shape != (len(x),)
                or p.shape != (len(x), len(ACTIONS))
                or c.shape != (len(x),)
                or len(self.episode_ids) != len(x)
                or len(self.scenario_seeds) != len(x)
                or len(x) == 0 or not np.isfinite(x).all()
                or not np.isfinite(m).all() or not np.isfinite(c).all()
                or not np.isfinite(p).all()
                or np.any((m != 0) & (m != 1))
                or np.any((c != 0) & (c != 1))
                or np.any(p < 0) or np.any(p[m == 0] > 1e-8)
                or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
                or np.any((y < 0) | (y >= len(ACTIONS)))
                or np.any(m[np.arange(len(m)), y] != 1)
                or not np.array_equal(x[:, _MASK_INDICES], m)
                or any(not value for value in self.episode_ids)
                or any(not isinstance(seed, int) or seed < 0
                       for seed in self.scenario_seeds)
                or re.fullmatch(r"[0-9a-f]{64}", self.frozen_role_sha256) is None):
            raise ValueError("invalid Warehouse public topology episode rows")
        for episode_id, seed in zip(self.episode_ids, self.scenario_seeds,
                                    strict=True):
            match = _EPISODE_PATTERN.fullmatch(episode_id)
            if match is None or int(match.group(2)) != seed:
                raise ValueError("Warehouse episode ID differs from actual reset seed")

    def target_sha256(self) -> str:
        target = np.ascontiguousarray(self.frozen_nn_masked_probs, dtype="<f4")
        return sha256(target.tobytes()).hexdigest()


def _episode_set(rows: EpisodeRows) -> set[str]:
    return set(rows.episode_ids)


def _scenario_set(rows: EpisodeRows) -> set[int]:
    """Reject correlated resets of one scene across different purposes."""
    return set(rows.scenario_seeds)


def _strata(rows: EpisodeRows) -> dict[str, np.ndarray]:
    x = rows.encoded
    # Fidelity strata must be knowable before the Actor proposes this frame's
    # action. The current frame's realized collision is reported separately.
    conflict = x[:, _STREAK] > 0
    near = x[:, _DISTANCE] * 42.0 <= 2.0
    low_return = x[:, _RETURN] * 50.0 <= 4.0
    return {"ordinary": ~(conflict | near | low_return),
            "conflict": conflict, "near_robots": near,
            "low_return_margin": low_return}


def _metrics(result: CoreTreeResult, rows: EpisodeRows, *,
             config: RefitConfig, select_leaves: bool) -> tuple[dict, np.ndarray]:
    leaves = result.leaf_indices(rows.encoded)
    q = _masked_probabilities(result.leaf_logits[leaves], rows.masks)
    target = rows.frozen_nn_masked_probs
    labels = target.argmax(1)
    predicted = q.argmax(1)
    expected_explanation_actionable = target[
        np.arange(len(predicted)), predicted]
    expected_feedback_actionable = expected_explanation_actionable * (
        predicted == labels)
    masked_kl = np.sum(target * (np.log(np.maximum(target, _EPS))
                                 - np.log(np.maximum(q, _EPS))), axis=1)
    masked = np.where(rows.masks > 0, result.leaf_logits[leaves], -1.0e9)
    sorted_logits = np.sort(masked, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    certified, _ = _structural_legal_actions(result)
    groups = _strata(rows)
    details = []
    selected = np.zeros(len(result.core_mask), dtype=np.bool_)
    for leaf in range(result.leaf_count):
        indexes = np.flatnonzero(leaves == leaf)
        raw = int(np.argmax(result.leaf_logits[leaf]))
        if not len(indexes):
            details.append({"leaf": leaf, "rows": 0,
                            "fit_core_candidate": bool(result.core_mask[leaf]),
                            "qualified": False})
            continue
        episodes, counts = np.unique(np.asarray(rows.episode_ids)[indexes],
                                     return_counts=True)
        raw_illegal = int(np.sum(rows.masks[indexes, raw] == 0))
        overall_agreement = float(np.mean(predicted[indexes] == labels[indexes]))
        overall_masked_kl = float(np.mean(masked_kl[indexes]))
        support = (len(indexes) >= config.min_leaf_rows
                   and len(episodes) >= config.min_leaf_episodes
                   and counts.max() / len(indexes) <= config.max_episode_fraction)
        strata = {}
        strata_pass = True
        for name, indicator in groups.items():
            chosen = indexes[indicator[indexes]]
            if not len(chosen):
                continue
            episode_count = len(set(np.asarray(rows.episode_ids)[chosen]))
            agreement = float(np.mean(predicted[chosen] == labels[chosen]))
            stratum_kl = float(np.mean(masked_kl[chosen]))
            adequate = (len(chosen) >= config.min_stratum_rows
                        and episode_count >= config.min_stratum_episodes
                        and agreement >= config.min_stratum_agreement
                        and stratum_kl <= config.max_stratum_masked_kl)
            strata[name] = {"rows": len(chosen), "episodes": episode_count,
                            "agreement": agreement, "masked_kl": stratum_kl,
                            "passes": bool(adequate)}
            strata_pass &= adequate
        qualifies = bool(result.core_mask[leaf] and support and raw_illegal == 0
                         and raw in certified[leaf]
                         and overall_agreement >= config.min_global_agreement
                         and overall_masked_kl <= config.max_global_masked_kl
                         and float(np.min(margin[indexes])) >= config.min_masked_margin
                         and strata_pass)
        rejection_reasons = []
        if not result.core_mask[leaf]:
            rejection_reasons.append("fit_not_core_candidate")
        if not support:
            rejection_reasons.append("episode_or_row_support")
        if raw_illegal or raw not in certified[leaf]:
            rejection_reasons.append("raw_action_not_path_legal")
        if overall_agreement < config.min_global_agreement:
            rejection_reasons.append("full_nn_top1_disagreement")
        if overall_masked_kl > config.max_global_masked_kl:
            rejection_reasons.append("full_nn_masked_kl")
        if float(np.min(margin[indexes])) < config.min_masked_margin:
            rejection_reasons.append("tree_margin")
        if not strata_pass:
            rejection_reasons.append("stratum_support_or_fidelity")
        if select_leaves:
            selected[leaf] = qualifies
        details.append({"leaf": leaf, "rows": len(indexes),
                        "episode_count": len(episodes),
                        "max_episode_fraction": float(counts.max() / len(indexes)),
                        "raw_top": ACTIONS[raw], "raw_illegal": raw_illegal,
                        "structurally_legal": raw in certified[leaf],
                        "full_nn_top1_agreement": overall_agreement,
                        "full_nn_masked_kl": overall_masked_kl,
                        "minimum_masked_margin": float(np.min(margin[indexes])),
                        "realized_collision_rows": int(np.sum(
                            rows.actual_collision[indexes] > 0)),
                        "strata": strata,
                        "fit_core_candidate": bool(result.core_mask[leaf]),
                        "rejection_reasons": rejection_reasons,
                        "qualified": qualifies})
    report = {"rows": len(rows.encoded), "episode_count": len(_episode_set(rows)),
              "core_leaf_rows": int(np.sum(result.core_mask[leaves])),
              "core_leaf_mass": float(np.mean(result.core_mask[leaves])),
              "expected_explanation_actionable_mass": float(np.mean(
                  result.core_mask[leaves] * expected_explanation_actionable)),
              "expected_feedback_actionable_mass": float(np.mean(
                  result.core_mask[leaves] * expected_feedback_actionable)),
              "core_full_nn_top1_agreement": float(np.mean(
                  predicted[result.core_mask[leaves]] ==
                  labels[result.core_mask[leaves]]))
              if np.any(result.core_mask[leaves]) else None,
              "core_full_nn_masked_kl": float(np.mean(
                  masked_kl[result.core_mask[leaves]]))
              if np.any(result.core_mask[leaves]) else None,
              "frozen_role_sha256": rows.frozen_role_sha256,
              "realized_collision_rows": int(np.sum(rows.actual_collision > 0)),
              "target_probs_sha256": rows.target_sha256(),
              "scenario_seeds_sha256": sha256(json.dumps(
                  sorted(_scenario_set(rows)), separators=(",", ":")).encode()).hexdigest(),
              "details": details,
              "strata_summary": {name: {
                  "rows": int(indicator.sum()),
                  "episodes": len(set(np.asarray(rows.episode_ids)[indicator])),
                  "core_rows": int(np.sum(result.core_mask[leaves][indicator])),
                  "core_mass": float(np.mean(result.core_mask[leaves][indicator]))
                  if indicator.any() else None,
              } for name, indicator in groups.items()},
              "scenario_episode_ids_sha256": sha256(json.dumps(
                  sorted(_episode_set(rows)), separators=(",", ":")).encode()).hexdigest()}
    return report, selected


def _replace(result: CoreTreeResult, logits: np.ndarray,
             core: np.ndarray, report: dict) -> CoreTreeResult:
    return CoreTreeResult(result.split_weights, result.split_bias,
                          result.left_refs, result.right_refs,
                          result.node_count, result.leaf_count,
                          logits.astype(np.float32), core.astype(np.bool_),
                          report, result.train_episode_ids)


def _balanced_fit_indices(rows: EpisodeRows, config: RefitConfig
                          ) -> tuple[np.ndarray, dict]:
    """Limit each NN episode so a long collision loop cannot own a split."""
    conflict = ((rows.actual_collision > 0) |
                (rows.encoded[:, _STREAK] > 0))
    identifiers = np.asarray(rows.episode_ids)
    weighted: list[np.ndarray] = []
    per_episode = {}
    for identifier in sorted(set(rows.episode_ids)):
        own = np.flatnonzero(identifiers == identifier)
        ordinary = own[~conflict[own]][::config.teacher_fit_stride]
        risky = own[conflict[own]]
        if len(risky) > config.max_conflict_rows_per_episode:
            positions = np.linspace(0, len(risky) - 1,
                                    config.max_conflict_rows_per_episode,
                                    dtype=np.int64)
            risky = risky[positions]
        repeated = np.repeat(risky, config.conflict_repeat)
        remaining = max(0, config.max_weighted_fit_rows_per_episode - len(repeated))
        if len(ordinary) > remaining:
            positions = np.linspace(0, len(ordinary) - 1, remaining,
                                    dtype=np.int64) if remaining else np.empty(0, dtype=np.int64)
            ordinary = ordinary[positions]
        selected = np.concatenate((ordinary, repeated))
        if len(selected):
            weighted.append(selected)
        per_episode[identifier] = {
            "ordinary_rows": len(ordinary), "distinct_conflict_rows": len(risky),
            "weighted_rows": len(selected),
        }
    indices = np.concatenate(weighted) if weighted else np.empty(0, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("Warehouse core refit lacks fit data")
    return indices, {
        "episode_count": len(per_episode),
        "selected_conflict_rows": sum(value["distinct_conflict_rows"]
                                      for value in per_episode.values()),
        "maximum_weighted_rows_per_episode": max(
            value["weighted_rows"] for value in per_episode.values()),
        "per_episode": per_episode,
    }


def fit_episode_separated_core(groups: Mapping[str, EpisodeRows], *,
                               frozen_role: FrozenWarehouseV6Role,
                               config: RefitConfig = RefitConfig(),
                               expected_topology_sha256: str,
                               ) -> tuple[CoreTreeResult, dict]:
    """Refit fixed Actor topology, calibrate, select core, then validate once.

    Fit rows never select a new topology.  Only the eligibility episodes may
    remove core leaves.  Validation reports pass/fail on the locked result.
    """
    required = ("fit", "calibration", "eligibility", "validation")
    if set(groups) != set(required):
        raise ValueError("Warehouse core refit needs four episode purposes")
    source_hashes = {groups[name].frozen_role_sha256 for name in required}
    if (len(source_hashes) != 1 or source_hashes != {frozen_role.role_sha256}
            or role_sha256(frozen_role.weights) != frozen_role.role_sha256):
        raise ValueError("Warehouse core refit purposes use different frozen NN roles")
    target_parity_max_abs_error = {}
    for name in required:
        rows = groups[name]
        recomputed = _masked_probabilities(
            frozen_role.logits_encoded(rows.encoded), rows.masks)
        error = float(np.max(np.abs(recomputed - rows.frozen_nn_masked_probs)))
        if not np.isfinite(error) or error > 1e-7:
            raise ValueError("Warehouse core target is not the bound frozen NN distribution")
        target_parity_max_abs_error[name] = error
    if role_sha256(frozen_role.weights) != frozen_role.role_sha256:
        raise RuntimeError("Warehouse frozen warmup NN weights changed during target verification")
    sets = {name: _episode_set(groups[name]) for name in required}
    if any(sets[a] & sets[b] for i, a in enumerate(required)
           for b in required[i + 1:]):
        raise ValueError("Warehouse core refit episode purposes overlap")
    scenes = {name: _scenario_set(groups[name]) for name in required}
    if any(scenes[a] & scenes[b] for i, a in enumerate(required)
           for b in required[i + 1:]):
        raise ValueError("Warehouse core refit scenario purposes overlap")
    fit = groups["fit"]
    indices, balancing = _balanced_fit_indices(fit, config)
    target = fit.frozen_nn_masked_probs[indices]
    schema = FeatureSchema(tuple(FEATURE_NAMES), tuple(ACTIONS))
    extracted = frozen_actor_topology(
        frozen_role.weights, schema=schema,
        frozen_actor_sha256=frozen_role.role_sha256)
    refitted = refit_frozen_topology_fit_only(
        extracted, fit.encoded[indices], fit.masks[indices], target,
        np.asarray(fit.episode_ids)[indices],
        frozen_actor_sha256=frozen_role.role_sha256,
        expected_topology_sha256=expected_topology_sha256)
    candidate = to_current_core_tree_result(refitted)
    cal = groups["calibration"]
    cal_leaves = candidate.leaf_indices(cal.encoded)
    temperatures = {}
    for temperature in config.temperature_grid:
        logits = candidate.leaf_logits / temperature
        logits = logits[cal_leaves]
        q = _masked_probabilities(logits, cal.masks)
        kl = np.sum(cal.frozen_nn_masked_probs * (
            np.log(np.maximum(cal.frozen_nn_masked_probs, _EPS))
            - np.log(np.maximum(q, _EPS))), axis=1)
        temperatures[temperature] = float(np.mean(kl))
    temperature = min(temperatures, key=lambda value: (temperatures[value], value))
    logits = candidate.leaf_logits.copy() / temperature
    logits -= logits.mean(axis=1, keepdims=True)
    span = logits.max(1) - logits.min(1)
    scale = np.minimum(1.0, config.maximum_raw_logit_span /
                       np.maximum(span, 1e-8))
    logits *= scale[:, None]
    calibrated = _replace(candidate, logits, candidate.core_mask,
                          dict(candidate.report))
    calibration_report, _ = _metrics(
        calibrated, cal, config=config, select_leaves=False)
    eligibility_report, selected = _metrics(
        calibrated, groups["eligibility"], config=config,
        select_leaves=True)
    locked = _replace(calibrated, logits, selected,
                      dict(candidate.report))
    locked_eligibility_report, _ = _metrics(
        locked, groups["eligibility"], config=config,
        select_leaves=False)
    validation_report, _ = _metrics(
        locked, groups["validation"], config=config,
        select_leaves=False)
    validation_passed = bool(
        validation_report["core_leaf_mass"] >= config.min_internal_core_mass
        and validation_report["expected_feedback_actionable_mass"] >=
        config.min_internal_expected_actionable_mass
        and all(detail["qualified"] for detail in validation_report["details"]
                if detail["fit_core_candidate"] and
                locked.core_mask[detail["leaf"]]))
    report = {
        "method": "episode_separated_v614_frozen_actor_topology_review_only",
        "target_provenance": "one_final_frozen_full_nn_masked_distribution",
        "frozen_role_sha256": next(iter(source_hashes)),
        "frozen_actor_topology_sha256": expected_topology_sha256,
        "target_parity_max_abs_error": target_parity_max_abs_error,
        "target_probs_sha256": {name: groups[name].target_sha256()
                                for name in required},
        "fit": {"rows": len(fit.encoded), "weighted_rows": len(indices),
                "conflict_rows_original": int(np.sum((fit.actual_collision > 0) |
                    (fit.encoded[:, _STREAK] > 0))),
                "episode_balancing": balancing,
                "tree_report": candidate.report},
        "calibration": {"rows": len(cal.encoded),
                        "temperature_grid_masked_kl": {str(k): v for k, v in
                                                 temperatures.items()},
                        "selected_temperature": temperature,
                        "raw_logit_span_cap": config.maximum_raw_logit_span,
                        "calibrated_masked_distribution": calibration_report},
        "eligibility": eligibility_report,
        "locked_eligibility": locked_eligibility_report,
        "validation": validation_report,
        "validation_passed": validation_passed,
        "core_mask_sha256": sha256(locked.core_mask.tobytes()).hexdigest(),
        "tree_content_sha256": locked.content_sha256(),
        "episode_sets_sha256": {name: sha256(json.dumps(
            sorted(sets[name]), separators=(",", ":")).encode()).hexdigest()
            for name in required},
        "scenario_sets_sha256": {name: sha256(json.dumps(
            sorted(scenes[name]), separators=(",", ":")).encode()).hexdigest()
            for name in required},
    }
    return _replace(locked, logits, selected, report), report


__all__ = ["RefitConfig", "EpisodeRows", "fit_episode_separated_core"]
