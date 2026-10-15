"""Create a runtime-loadable r4 Actor/program/protocol evidence set.

The training checkpoint protocol intentionally remains an optimizer protocol.
This builder derives the portable observed197 runtime protocol from the frozen
r3 runtime contract, re-exports identical Actor weights with that protocol
binding, and binds the RCPD program to the resulting Actor artifact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training import warehouse_r4_final_rcpd as final_rcpd
from backend.training import warehouse_r4_production_admission as production_admission
from backend.warehouse_alignment_online_explanation import OnlineAlignmentExplainer
from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r4-active-runtime-bundle.v1"
R3_PROTOCOL = ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/protocol.json"


def _read_json(path: Path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON field in r4 runtime input")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Non-finite JSON value in r4 runtime input: " + value)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def _regular_path(value, label: str) -> Path:
    supplied = Path(value)
    if supplied.is_symlink():
        raise ValueError(label + " cannot be a symlink")
    path = supplied.resolve()
    if not path.is_file():
        raise ValueError(label + " must be a regular file")
    return path


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _export_actor(source_path: Path, destination: Path, metadata: dict) -> None:
    with np.load(source_path, allow_pickle=False) as archive:
        tensors = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, metadata_json=json.dumps(metadata, sort_keys=True), **tensors)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def build_runtime_bundle(*, actor_path, program_path, extraction_report_path,
                         evaluation_path, scenarios_path, output_dir):
    actor_path = _regular_path(actor_path, "R4 training Actor")
    program_path = _regular_path(program_path, "R4 final RCPD program")
    extraction_report_path = _regular_path(
        extraction_report_path, "R4 final RCPD report")
    evaluation_path = _regular_path(evaluation_path, "R4 paired audit report")
    scenarios_path = _regular_path(scenarios_path, "R4 scenario manifest")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("R4 runtime bundle requires a new output directory")
    source_actor = NumPyNativeActor(actor_path)
    training_actor_sha256 = file_hash(actor_path)
    input_hashes = {
        "actor": training_actor_sha256,
        "program": file_hash(program_path),
        "extraction_report": file_hash(extraction_report_path),
        "evaluation": file_hash(evaluation_path),
        "scenarios": file_hash(scenarios_path),
        "base_protocol": file_hash(R3_PROTOCOL),
    }
    actor_parameters_sha256 = source_actor.metadata.get(
        "actor_parameters_sha256", source_actor.metadata.get("parameters_sha256"))
    scenarios = _read_json(scenarios_path)
    extraction = final_rcpd.read_saved_report(
        extraction_report_path.parent,
        expected_report_sha256=file_hash(extraction_report_path),
        actor_path=actor_path, scenarios_path=scenarios_path,
        program_path=program_path, require_passed=True,
    )
    evaluation = _read_json(evaluation_path)
    production_admission._validate_active(evaluation, {
        "training_actor_sha256": training_actor_sha256,
        "validation_entries_sha256": digest(scenarios.get("splits", {}).get("validation")),
        "scenario_manifest_file_sha256": file_hash(scenarios_path),
    })
    production_admission._validate_active_evidence(
        evaluation, evaluation_path, scenarios["splits"]["validation"])
    base_protocol = _read_json(R3_PROTOCOL.resolve())
    feature_names = list(source_actor.metadata["feature_names"])
    protocol = deepcopy(base_protocol)
    protocol.update({
        "version": "warehouse-r4-active-runtime-protocol.v1",
        "public_feedback_mode": "observed",
        "public_feedback_version": "warehouse-native-public-feedback.v1",
        "feature_names": feature_names,
        "feedback_training_enabled": bool(source_actor.metadata.get("feedback_enabled", False)),
        "explanation_qualification_granted": False,
        "r4_active": {
            "bundle_version": VERSION,
            "source_r3_actor_sha256": "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b",
            "unbound_candidate_actor_sha256": file_hash(actor_path),
            "actor_parameters_sha256": actor_parameters_sha256,
            "additional_joint_steps": int(source_actor.metadata["joint_steps"]),
            "total_joint_steps": int(source_actor.metadata["source_counters"]["joint_steps"] + source_actor.metadata["joint_steps"]),
            "partner_mix": {"selfplay": .20, "skilled": .20, "assertive": .50, "noisy": .10},
            "active_shaping": {"productive_progress": .02, "avoidable_wait": -.03,
                               "task_distance_regression": -.02},
            "runtime_action_override": False,
            "evaluation_report_sha256": file_hash(evaluation_path),
            "evaluation_status": evaluation.get("status"),
            "evaluation_selected": bool(evaluation.get("selected", False)),
            "final_rcpd_report_sha256": file_hash(extraction_report_path),
            "final_rcpd_binding_sha256": extraction["final_rcpd_binding_sha256"],
        },
    })
    protocol_sha256 = digest(protocol)
    metadata = deepcopy(source_actor.metadata)
    metadata.update({
        "experiment_version": source_actor.metadata.get("experiment_version", "warehouse-r4-active-trainer.v2"),
        "protocol_sha256": protocol_sha256,
        "feature_names": feature_names,
        "public_feedback_mode": "observed",
        "public_feedback_version": "warehouse-native-public-feedback.v1",
        "action_masks": False,
        "runtime_action_override": False,
        "runtime_controller": "native_neural_actor_only",
        "r4_runtime_bundle_version": VERSION,
        "r4_training_actor_sha256": file_hash(actor_path),
        "r4_evaluation_report_sha256": file_hash(evaluation_path),
        "r4_final_rcpd_report_sha256": file_hash(extraction_report_path),
        "r4_final_rcpd_binding_sha256": extraction["final_rcpd_binding_sha256"],
        "scenario_manifest_sha256": digest(scenarios),
    })
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir.chmod(0o700)
    protocol_path = output_dir / "protocol.json"
    final_actor_path = output_dir / "actor.npz"
    final_program_path = output_dir / "program.json"
    _atomic_json(protocol_path, protocol)
    _export_actor(actor_path, final_actor_path, metadata)
    final_actor_sha256 = file_hash(final_actor_path)

    program = _read_json(program_path)
    source = program.get("program", program)
    source["feature_names"] = feature_names
    source["action_names"] = list(ACTIONS)
    source.setdefault("metadata", {})
    source["metadata"].update({
        "native_source_actor_sha256": final_actor_sha256,
        "source_training_actor_sha256": file_hash(actor_path),
        "source_actor_parameters_sha256": actor_parameters_sha256,
        "source_protocol_sha256": protocol_sha256,
        "final_rcpd_report_sha256": file_hash(extraction_report_path),
        "final_rcpd_binding_sha256": extraction["final_rcpd_binding_sha256"],
        "runtime_controller": "native_neural_actor_only",
        "runtime_action_override": False,
        "feature_order_rebound_to_actor": True,
        "explanation_qualified": False,
        "release_ready": False,
    })
    _atomic_json(final_program_path, program)
    final_program_sha256 = file_hash(final_program_path)

    runtime = OnlineAlignmentRuntime(
        final_actor_path,
        protocol=protocol,
        expected_actor_sha256=final_actor_sha256,
        expected_protocol_sha256=protocol_sha256,
    )
    explainer = OnlineAlignmentExplainer(
        final_program_path,
        expected_program_sha256=final_program_sha256,
        runtime=runtime,
    )
    # Metadata rebinding must not change any weight or numeric inference.
    rebound = NumPyNativeActor(final_actor_path)
    if set(source_actor.weights) != set(rebound.weights) or any(
            not np.array_equal(source_actor.weights[name], rebound.weights[name])
            for name in source_actor.weights):
        raise RuntimeError("Actor weights changed during runtime metadata binding")
    rng = np.random.default_rng(260_910_701)
    observations = rng.normal(size=(64, source_actor.obs_dim)).astype(np.float32)
    source_logits, rebound_logits = source_actor.logits(observations), rebound.logits(observations)
    maximum_error = float(np.max(np.abs(source_logits - rebound_logits)))
    action_equal = bool(np.array_equal(source_logits.argmax(-1), rebound_logits.argmax(-1)))
    if maximum_error > 1e-4 or not action_equal:
        raise RuntimeError("Runtime Actor differs from the trained NumPy Actor")
    scenario = scenarios["splits"]["validation"][0]
    environment = runtime.environment(scenario)
    actual_observations = environment.observations()
    source_actions, _ = source_actor.act(actual_observations, deterministic=True)
    runtime_actions, decision = runtime.decision(environment)
    if source_actions != runtime_actions or decision["post_policy_overrides"] != 0:
        raise RuntimeError("Online runtime action differs on an actual observed197 state")
    transition = runtime.step(environment, "WAIT")
    if (transition["submitted_actions"]["robot_2"] != transition["policy_actions"]["robot_2"]
            or transition["decision"]["post_policy_overrides"] != 0):
        raise RuntimeError("Online runtime changed the Actor command before physics")
    receipt = {
        "version": VERSION,
        "status": "runtime_components_verified",
        "candidate_selected_by_r4_gate": bool(evaluation.get("selected", False)),
        "formal_ready": False,
        "actor": {"path": str(final_actor_path), "sha256": final_actor_sha256,
                  "parameters_sha256": actor_parameters_sha256},
        "program": {"path": str(final_program_path), "sha256": final_program_sha256,
                    "feature_names_equal_actor": tuple(explainer.program.feature_names) == tuple(feature_names),
                    "native_source_actor_sha256": explainer.program.metadata.get("native_source_actor_sha256"),
                    "source_training_actor_sha256": explainer.program.metadata.get(
                        "source_training_actor_sha256"),
                    "final_rcpd_binding_sha256": explainer.program.metadata.get(
                        "final_rcpd_binding_sha256")},
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha256,
                     "file_sha256": file_hash(protocol_path),
                     "source_protocol_sha256": input_hashes["base_protocol"]},
        "evaluation": {"path": str(evaluation_path), "sha256": file_hash(evaluation_path),
                       "status": evaluation.get("status")},
        "final_rcpd": {
            "version": final_rcpd.VERSION,
            "path": str(extraction_report_path),
            "sha256": file_hash(extraction_report_path),
            "status": extraction["status"],
            "binding_sha256": extraction["final_rcpd_binding_sha256"],
            "source_training_actor_path": str(actor_path),
            "source_training_actor_sha256": training_actor_sha256,
            "scenario_manifest_path": str(scenarios_path),
            "scenario_manifest_sha256": file_hash(scenarios_path),
            "program_source_sha256": extraction["program_file_sha256"],
            "candidate_count": extraction["candidate_count"],
            "ppo_joint_steps": extraction["execution"]["ppo_joint_steps"],
            "optimizer_updates": extraction["execution"]["optimizer_updates"],
            "program_feedback_into_actor": extraction["execution"]["program_feedback_into_actor"],
        },
        "runtime": {"signature": runtime.signature, "load_verified": True,
                    "action_masks": False, "post_policy_overrides": 0,
                    "actual_observed197_action_parity": True,
                    "actual_step_policy_equals_submitted": True},
        "explainer": {"signature": explainer.signature, "load_verified": True,
                      "independent_r4_explanation_acceptance_pending": True},
        "numpy_parity": {"observations": len(observations), "maximum_absolute_logit_error": maximum_error,
                         "deterministic_actions_equal": action_equal},
    }
    if input_hashes != {
            "actor": file_hash(actor_path), "program": file_hash(program_path),
            "extraction_report": file_hash(extraction_report_path),
            "evaluation": file_hash(evaluation_path), "scenarios": file_hash(scenarios_path),
            "base_protocol": file_hash(R3_PROTOCOL)}:
        raise RuntimeError("A frozen r4 runtime-bundle input changed during construction")
    _atomic_json(output_dir / "verification.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--extraction-report", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_runtime_bundle(actor_path=args.actor, program_path=args.program,
                                  extraction_report_path=args.extraction_report,
                                  evaluation_path=args.evaluation,
                                  scenarios_path=args.scenarios, output_dir=args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
