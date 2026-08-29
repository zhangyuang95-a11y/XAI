"""Build one immutable, pre-formal warehouse candidate artifact bundle.

This command does not train or alter Actor weights.  It normalizes the
already-trained candidate's provenance, distils an explanation-only RCPD tree
from final Actor executions, and calibrates the fixed study artifacts on seed
ranges reserved in the training ledger.  Formal evaluation must run only
after this command and on a disjoint seed family.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from backend.adapters.warehouse import WarehouseAdapter
from backend.artifacts import CollaborativeArtifactPaths, file_sha256
from backend.training.warehouse import (
    _collect_posthoc_rcpd_records,
    _fit_rcpd,
    _program_regularization_summary,
    _rcpd_from_args,
    build_parser,
    write_policy_trajectory,
)
from backend.training.seed_ledger import (
    evaluation_seed_span,
    reserve_evaluation_seed_span,
)
from env.warehouse.contracts import ARTIFACT_NAMESPACE
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.seed_calibration import (
    calibrate_parallel_seed_pairs,
    save_parallel_seed_library,
)
from ui.tutorial import (
    calibrate_reference_trajectory,
    save_reference_trajectory_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def build_candidate_bundle(
    *,
    source: Path,
    base_summary_path: Path,
    base_checkpoint: Path,
    base_metrics: Path,
    development_evaluation: Path,
    root: Path,
    device: str,
    additional_development_seed_ranges: Mapping[str, Any] | None = None,
    rcpd_replay_records: int | None = None,
    rcpd_max_leaf_nodes: int | None = None,
    rcpd_target_temperature: float | None = None,
) -> dict[str, Any]:
    paths = CollaborativeArtifactPaths.under(PROJECT_ROOT, root.name)
    if paths.root != root.resolve():
        raise ValueError("Artifact root must be under output/collaborative.")
    paths.root.mkdir(parents=True, exist_ok=True)

    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    development = json.loads(
        development_evaluation.read_text(encoding="utf-8")
    )
    source_metadata = _mapping(source_payload.get("training_metadata"))
    rejected_formal_evaluations = deepcopy(
        source_metadata.get(
            "rejected_formal_evaluations",
            base_summary.get("rejected_formal_evaluations", []),
        )
    )
    superseded_formal_evaluations = deepcopy(
        source_metadata.get(
            "superseded_formal_evaluations",
            base_summary.get("superseded_formal_evaluations", []),
        )
    )
    source_seed_ledger = _mapping(source_metadata.get("seed_ledger"))
    candidate_seed_ledger = source_seed_ledger or _mapping(
        base_summary.get("seed_ledger")
    )
    if additional_development_seed_ranges:
        additional_seed_start, _ = evaluation_seed_span(
            additional_development_seed_ranges
        )
        candidate_seed_ledger = reserve_evaluation_seed_span(
            candidate_seed_ledger,
            additional_development_seed_ranges,
            name=(
                "additional_final_candidate_development_validation_"
                f"{additional_seed_start}"
            ),
        )
    development_seed_ranges = _mapping(development.get("seed_ranges"))
    development_seed_start, _ = evaluation_seed_span(development_seed_ranges)
    candidate_seed_ledger = reserve_evaluation_seed_span(
        candidate_seed_ledger,
        development_seed_ranges,
        name=f"final_candidate_development_evaluation_{development_seed_start}",
    )
    policy = MAPPOPolicy.load(source, device=device)
    normalized_metadata = {
        "environment": "two_robot_shared_delivery",
        "training_origin": "new_compact_map_from_scratch_then_offline_partner_risk",
        "base_training_summary": str(base_summary_path.resolve()),
        "base_training_checkpoint": str(base_checkpoint.resolve()),
        "seed_ledger": deepcopy(candidate_seed_ledger),
        "rejected_formal_evaluations": rejected_formal_evaluations,
        "superseded_formal_evaluations": superseded_formal_evaluations,
        "source_training_metadata": deepcopy(source_metadata),
        "development_evaluation": {
            "path": str(development_evaluation.resolve()),
            "seed_ranges": deepcopy(development.get("seed_ranges", {})),
            "acceptance_checks": deepcopy(
                development.get("acceptance_checks", {})
            ),
            "formal_seed_use": False,
            "additional_seed_ranges": deepcopy(
                additional_development_seed_ranges or {}
            ),
        },
        "runtime_action_source": "mappo_actor",
        "post_policy_action_interventions": 0,
    }
    policy.save(paths.model, training_metadata=normalized_metadata)
    if base_checkpoint.resolve() != paths.training_checkpoint.resolve():
        shutil.copy2(base_checkpoint, paths.training_checkpoint)
    if base_metrics.resolve() != paths.metrics.resolve():
        shutil.copy2(base_metrics, paths.metrics)
    write_policy_trajectory(
        policy,
        paths.training_trajectory,
        seed=int(policy.environment_config.seed) + 300_000,
    )

    extraction_args = build_parser().parse_args([])
    extraction_args.use_rcpd = True
    extraction_args.seed = int(policy.environment_config.seed)
    if rcpd_replay_records is not None:
        extraction_args.rcpd_replay_records = int(rcpd_replay_records)
    if rcpd_max_leaf_nodes is not None:
        extraction_args.rcpd_max_leaf_nodes = int(rcpd_max_leaf_nodes)
    if rcpd_target_temperature is not None:
        extraction_args.rcpd_target_temperature = float(rcpd_target_temperature)
    rcpd = _rcpd_from_args(extraction_args)
    assert rcpd is not None
    adapter = WarehouseAdapter(WarehouseMultiAgentEnv(policy.environment_config))
    record_count = int(extraction_args.rcpd_replay_records)
    extraction_episodes = max(
        40,
        (record_count + 2 * policy.environment_config.horizon - 1)
        // (2 * policy.environment_config.horizon),
    )
    records = _collect_posthoc_rcpd_records(
        policy,
        adapter,
        episodes=extraction_episodes,
        seed=int(policy.environment_config.seed) + 11_000_000,
    )[-record_count:]
    if not _fit_rcpd(rcpd, records, adapter, step=2_800):
        raise RuntimeError(f"RCPD extraction failed: {rcpd.last_error}")
    assert rcpd.program is not None and rcpd.last_result is not None
    if not rcpd.last_result.metrics.explanation_eligible:
        raise RuntimeError(
            "RCPD explanation gate failed: "
            + ", ".join(
                rcpd.last_result.metrics.explanation_ineligibility_reasons
            )
        )
    rcpd.program.save_json(paths.rcpd_program)
    rcpd.program.export_python(paths.rcpd_python)

    seed_start = int(policy.environment_config.seed) + 400_000
    pairs = calibrate_parallel_seed_pairs(
        policy,
        range(seed_start, seed_start + 256),
    )
    save_parallel_seed_library(paths.parallel_seed_pairs, pairs)
    reference = calibrate_reference_trajectory(policy, maximum_candidates=2_000)
    save_reference_trajectory_manifest(
        paths.reference_trajectory,
        reference,
        policy,
    )

    program_regularization = _program_regularization_summary(
        extraction_args,
        {"neural_policy": development.get("ai_ai", {})},
        rcpd,
    )
    program_regularization.update(
        {
            "training_data_source": "executed_neural_rollout_only",
            "runtime_control_allowed": False,
            "feedback_allowed": False,
            "records": len(records),
            "rcpd_explanation_eligible": True,
            "reference_trajectory_eligible": True,
            "explanation_eligible": True,
        }
    )
    summary = deepcopy(base_summary)
    summary.update(
        {
            "format": "warehouse_collaborative_training_v28_causal_clearance",
            "model_version": policy.model_version,
            "environment_config": asdict(policy.environment_config),
            "algorithm_config": {
                **asdict(policy.algorithm_config),
                "joint_collision_loss_weight": float(
                    _mapping(base_summary.get("algorithm_config")).get(
                        "joint_collision_loss_weight",
                        0.25,
                    )
                ),
            },
            "seed_ledger": deepcopy(candidate_seed_ledger),
            "rejected_formal_evaluations": rejected_formal_evaluations,
            "superseded_formal_evaluations": superseded_formal_evaluations,
            "program_regularization": program_regularization,
            "final_candidate": {
                "source": str(source.resolve()),
                "training_origin": normalized_metadata["training_origin"],
                "development_evaluation": str(
                    development_evaluation.resolve()
                ),
                "formal_evaluation_completed": False,
            },
            "artifacts": {
                "model": str(paths.model),
                "training_checkpoint": str(paths.training_checkpoint),
                "metrics": str(paths.metrics),
                "trajectory": str(paths.training_trajectory),
                "program": str(paths.rcpd_program),
                "parallel_seed_pairs": str(paths.parallel_seed_pairs),
                "reference_trajectory": str(paths.reference_trajectory),
            },
            "reference_trajectory": {
                "eligible": True,
                "seed": int(reference.seed),
                "frame_count": len(reference.frames) + 1,
            },
        }
    )
    paths.training_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = {
        "artifact_namespace": ARTIFACT_NAMESPACE,
        "root": str(paths.root),
        "model_sha256": file_sha256(paths.model),
        "program_sha256": file_sha256(paths.rcpd_program),
        "reference_sha256": file_sha256(paths.reference_trajectory),
        "parallel_seed_sha256": file_sha256(paths.parallel_seed_pairs),
        "training_summary_sha256": file_sha256(paths.training_summary),
        "rcpd_metrics": rcpd.last_result.metrics.to_dict(),
        "reference_seed": int(reference.seed),
        "parallel_task_seeds": [
            seed
            for pair in pairs
            for seed in (pair.task1_seed, pair.task2_seed)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("base_summary", type=Path)
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("base_metrics", type=Path)
    parser.add_argument("development_evaluation", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "output"
            / "collaborative"
            / ARTIFACT_NAMESPACE
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rcpd-replay-records", type=int)
    parser.add_argument("--rcpd-max-leaf-nodes", type=int)
    parser.add_argument("--rcpd-target-temperature", type=float)
    parser.add_argument(
        "--additional-development-seed-range",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "START", "COUNT"),
    )
    args = parser.parse_args()
    additional_ranges = {
        str(name): [int(start), int(count)]
        for name, start, count in args.additional_development_seed_range
    }
    build_candidate_bundle(
        source=args.source,
        base_summary_path=args.base_summary,
        base_checkpoint=args.base_checkpoint,
        base_metrics=args.base_metrics,
        development_evaluation=args.development_evaluation,
        root=args.root.resolve(),
        device=args.device,
        additional_development_seed_ranges=additional_ranges,
        rcpd_replay_records=args.rcpd_replay_records,
        rcpd_max_leaf_nodes=args.rcpd_max_leaf_nodes,
        rcpd_target_temperature=args.rcpd_target_temperature,
    )


if __name__ == "__main__":
    main()
