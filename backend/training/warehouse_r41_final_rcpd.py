"""Post-freeze RCPD extraction for the r4.1 conflict warehouse.

This module is intentionally a thin, environment-specific specialization of
the numerically audited r4 extraction implementation.  It keeps fitting
completely outside PPO, uses disjoint r4.1 train and validation episodes, and
replays every stored neural label against the supplied frozen Actor.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r4_final_rcpd as base
from backend.training.warehouse_native_common import digest
from backend.training.warehouse_native_evaluation import critical_groups
from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_conflict import reset_r41_scenario


VERSION = "warehouse-r41-final-rcpd.v1"
PARTNERS = base.PARTNERS
GROUPS = base.GROUPS
TRAIN_SCENES = 70
VALIDATION_SCENES = 30
TRAIN_SEED = 260_911_410
VALIDATION_SEED = 260_911_510
DEPTHS = base.DEPTHS
LEAVES = base.LEAVES
MINIMUM_OVERALL_FIDELITY = base.MINIMUM_OVERALL_FIDELITY
MINIMUM_CRITICAL_FIDELITY = base.MINIMUM_CRITICAL_FIDELITY


def contract() -> dict[str, Any]:
    """Return the immutable extraction-only experiment contract."""
    return {
        "version": VERSION,
        "source_split": {
            "train": "train:first_70",
            "validation": "conflict_validation:first_30",
        },
        "source_scenes": TRAIN_SCENES + VALIDATION_SCENES,
        "train_scenes": TRAIN_SCENES,
        "validation_scenes": VALIDATION_SCENES,
        "partition": "episode-disjoint frozen r4.1 manifest splits",
        "partners": list(PARTNERS),
        "collection_seeds": {
            "train": TRAIN_SEED,
            "validation": VALIDATION_SEED,
        },
        "evaluated_role": "robot_2",
        "deterministic_actor": True,
        "complete_episodes": True,
        "candidate_depths": list(DEPTHS),
        "candidate_leaves": list(LEAVES),
        "min_samples_leaf": 4,
        "minimum_overall_fidelity": MINIMUM_OVERALL_FIDELITY,
        "minimum_critical_fidelity": MINIMUM_CRITICAL_FIDELITY,
        "critical_groups": list(GROUPS),
        "selection": "minimum structural loss, then KL, fidelity, depth cap, leaf cap",
        "ppo_joint_steps": 0,
        "optimizer_updates": 0,
        "program_feedback_into_actor": False,
        "runtime_action_override": False,
        "independent_intervention_audit_required": True,
    }


def producer_sources() -> dict[str, str]:
    # Reuse the repository's import-closure hasher.  The seed set includes the
    # specialization and the complete audited numerical implementation.
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    return local_source_hashes((Path(__file__), Path(base.__file__)))


def _scenario_contract(scenarios: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """Authenticate the frozen r4.1 split identities used by extraction."""
    from backend.training.warehouse_r41_conflict_scenarios import (
        MANIFEST_VERSION,
        validate_conflict_manifest,
    )

    validate_conflict_manifest(scenarios, replay=True)
    splits = scenarios.get("splits")
    if scenarios.get("version") != MANIFEST_VERSION or not isinstance(splits, dict):
        raise ValueError("Exact frozen r4.1 conflict manifest required")
    train = splits.get("train")
    validation = splits.get("conflict_validation")
    if (not isinstance(train, list) or len(train) < TRAIN_SCENES
            or not isinstance(validation, list) or len(validation) < VALIDATION_SCENES):
        raise ValueError("R4.1 manifest lacks the registered extraction pools")
    train = deepcopy(train[:TRAIN_SCENES])
    validation = deepcopy(validation[:VALIDATION_SCENES])
    train_ids = {(row.get("id"), row.get("fingerprint")) for row in train}
    validation_ids = {(row.get("id"), row.get("fingerprint")) for row in validation}
    if (len(train_ids) != TRAIN_SCENES or len(validation_ids) != VALIDATION_SCENES
            or train_ids & validation_ids):
        raise ValueError("R4.1 extraction episodes are not disjoint")
    return train, validation


def _collect_pool(actor: NumPyNativeActor, scenes: Sequence[Mapping[str, Any]],
                  *, pool: str, seed: int) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {name: [] for name in base._ARRAY_FIELDS}
    for scene_index, scene in enumerate(scenes):
        for partner_index, partner in enumerate(PARTNERS):
            episode = f"{pool}:{scene['id']}:{scene['fingerprint']}:{partner}"
            environment = R41ConflictWarehouseEnv()
            reset_r41_scenario(environment, deepcopy(scene))
            rng = np.random.default_rng(seed + scene_index * 101 + partner_index)
            while not environment.done:
                snapshot = environment.snapshot()
                snapshot_sha256 = digest(snapshot)
                frame = int(environment.state.frame)
                observations = environment.observations()
                if (tuple(environment.feature_names)
                        != tuple(actor.metadata["feature_names"])
                        or observations["robot_2"].shape != (actor.obs_dim,)):
                    raise ValueError("Actor and r4.1 extraction schemas differ")
                actions, probabilities = actor.act(observations, deterministic=True)
                if digest(environment.snapshot()) != snapshot_sha256:
                    raise RuntimeError("Frozen Actor inference changed source state")
                player = partner_action(environment, "robot_1", partner, rng)
                if digest(environment.snapshot()) != snapshot_sha256:
                    raise RuntimeError("Extraction partner changed source state")
                groups = critical_groups(environment, "robot_2")
                action_index = ACTIONS.index(actions["robot_2"])
                _, _, terminated, truncated, info = environment.step({
                    "robot_1": player, "robot_2": actions["robot_2"],
                })
                equal = (info["requested_actions"]["robot_2"]
                         == actions["robot_2"]
                         and actions["robot_2"] == ACTIONS[int(np.argmax(
                             probabilities["robot_2"]))])
                if not equal:
                    raise RuntimeError("Environment command differs from frozen Actor")
                row = {
                    "observations": observations["robot_2"].astype(np.float32, copy=True),
                    "probabilities": probabilities["robot_2"].astype(np.float32, copy=True),
                    "episode_ids": episode,
                    "scenario_fingerprints": scene["fingerprint"],
                    "partners": partner,
                    "frames": frame,
                    "after_frames": int(environment.state.frame),
                    "done": bool(terminated or truncated),
                    "group_bits": base._group_bits(groups),
                    "policy_action_indices": action_index,
                    "player_action_indices": ACTIONS.index(player),
                    "submitted_equal": equal,
                    "source_state_sha256": snapshot_sha256,
                }
                for name, value in row.items():
                    values[name].append(value)
    if not values["observations"]:
        raise ValueError("R4.1 final extraction collected no neural rows")
    return {
        "observations": np.stack(values["observations"]).astype(np.float32),
        "probabilities": np.stack(values["probabilities"]).astype(np.float32),
        "episode_ids": np.asarray(values["episode_ids"], dtype="U180"),
        "scenario_fingerprints": np.asarray(values["scenario_fingerprints"], dtype="U64"),
        "partners": np.asarray(values["partners"], dtype="U16"),
        "frames": np.asarray(values["frames"], dtype=np.int16),
        "after_frames": np.asarray(values["after_frames"], dtype=np.int16),
        "done": np.asarray(values["done"], dtype=np.bool_),
        "group_bits": np.asarray(values["group_bits"], dtype=np.uint8),
        "policy_action_indices": np.asarray(values["policy_action_indices"], dtype=np.uint8),
        "player_action_indices": np.asarray(values["player_action_indices"], dtype=np.uint8),
        "submitted_equal": np.asarray(values["submitted_equal"], dtype=np.bool_),
        "source_state_sha256": np.asarray(values["source_state_sha256"], dtype="U64"),
    }


def _replay_pool(value: Mapping[str, np.ndarray], *, name: str,
                 actor: NumPyNativeActor,
                 expected_scenes: Sequence[Mapping[str, Any]], seed: int) -> None:
    by_episode: dict[str, list[int]] = {}
    for index, episode in enumerate(map(str, value["episode_ids"])):
        by_episode.setdefault(episode, []).append(index)
    for scene_index, scene in enumerate(expected_scenes):
        for partner_index, partner in enumerate(PARTNERS):
            episode = f"{name}:{scene['id']}:{scene['fingerprint']}:{partner}"
            indices = sorted(by_episode[episode], key=lambda row: int(value["frames"][row]))
            environment = R41ConflictWarehouseEnv()
            reset_r41_scenario(environment, deepcopy(scene))
            rng = np.random.default_rng(seed + scene_index * 101 + partner_index)
            for index in indices:
                snapshot_sha256 = digest(environment.snapshot())
                frame = int(environment.state.frame)
                observations = environment.observations()
                actions, probabilities = actor.act(observations, deterministic=True)
                player = partner_action(environment, "robot_1", partner, rng)
                groups = critical_groups(environment, "robot_2")
                if (digest(environment.snapshot()) != snapshot_sha256
                        or frame != int(value["frames"][index])
                        or snapshot_sha256 != str(value["source_state_sha256"][index])
                        or not np.array_equal(observations["robot_2"].astype(
                            np.float32, copy=False), value["observations"][index])
                        or not np.allclose(probabilities["robot_2"],
                                           value["probabilities"][index],
                                           atol=1e-6, rtol=1e-6)
                        or ACTIONS.index(actions["robot_2"])
                            != int(value["policy_action_indices"][index])
                        or ACTIONS.index(player)
                            != int(value["player_action_indices"][index])
                        or base._group_bits(groups) != int(value["group_bits"][index])):
                    raise ValueError(f"{name} row is not from its r4.1 trajectory")
                submitted = {"robot_1": player, "robot_2": actions["robot_2"]}
                _, _, terminated, truncated, info = environment.step(submitted)
                if (info["requested_actions"] != submitted
                        or int(environment.state.frame)
                            != int(value["after_frames"][index])
                        or bool(terminated or truncated) != bool(value["done"][index])):
                    raise ValueError(f"{name} r4.1 replay transition differs")
            if not environment.done:
                raise ValueError(f"{name} r4.1 replay contains a partial episode")


# Reuse the audited, deterministic fitting and evidence-validation machinery.
_validate_pool = base._validate_pool
_softmax = base._softmax
_metrics = base._metrics
_gate = base._gate
_candidate_config = base._candidate_config
_fit_candidates = base._fit_candidates
_selection_key = base._selection_key
_select = base._select
_load_npz = base._load_npz


def _install_specialization():
    """Temporarily bind base globals used by its extraction/read functions."""
    names = {
        "VERSION": VERSION,
        "PARTNERS": PARTNERS,
        "GROUPS": GROUPS,
        "TRAIN_SCENES": TRAIN_SCENES,
        "VALIDATION_SCENES": VALIDATION_SCENES,
        "TRAIN_SEED": TRAIN_SEED,
        "VALIDATION_SEED": VALIDATION_SEED,
        "DEPTHS": DEPTHS,
        "LEAVES": LEAVES,
        "producer_sources": producer_sources,
        "contract": contract,
        "_scenario_contract": _scenario_contract,
        "_collect_pool": _collect_pool,
        "_replay_pool": _replay_pool,
    }
    return {name: getattr(base, name) for name in names}, names


def _call_base(name: str, *args, **kwargs):
    previous, replacements = _install_specialization()
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        return getattr(base, name)(*args, **kwargs)
    finally:
        for key, value in previous.items():
            setattr(base, key, value)


def extract(*, actor_path: str | Path, scenarios_path: str | Path,
            output: str | Path) -> dict[str, Any]:
    return _call_base("extract", actor_path=actor_path,
                      scenarios_path=scenarios_path, output=output)


def read_saved_report(output: str | Path, *, expected_report_sha256: str,
                      actor_path: str | Path, scenarios_path: str | Path,
                      program_path: str | Path | None,
                      require_passed: bool = True) -> dict[str, Any]:
    return _call_base(
        "read_saved_report", output,
        expected_report_sha256=expected_report_sha256,
        actor_path=actor_path, scenarios_path=scenarios_path,
        program_path=program_path, require_passed=require_passed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from backend.training.warehouse_native_common import canonical

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = extract(actor_path=args.actor, scenarios_path=args.scenarios,
                     output=args.output)
    print(canonical({
        "status": report["status"], "selected": report["selected"],
        "candidate_count": report["candidate_count"],
        "ppo_joint_steps": 0, "program_feedback_into_actor": False,
        "report": str((Path(args.output) / "report.json").resolve()),
    }))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "PARTNERS", "GROUPS", "DEPTHS", "LEAVES", "contract",
    "producer_sources", "extract", "read_saved_report", "main",
]
