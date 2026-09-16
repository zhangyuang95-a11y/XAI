"""Prepare truthful r4.6 deployment artifacts from the selected 800k Actor."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from backend.warehouse_r46_runtime import PROTOCOL_VERSION
from scripts.build_warehouse_r42_delivery_release import migrate_manifest


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")


def _flatten(node, feature_index, arrays):
    index = len(arrays["feature"])
    arrays["children_left"].append(-1)
    arrays["children_right"].append(-1)
    arrays["feature"].append(-2)
    arrays["threshold"].append(-2.0)
    arrays["value"].append([0.0] * 5)
    if "probabilities" in node:
        arrays["value"][index] = [float(item) for item in node["probabilities"]]
        return index
    name = str(node["feature"])
    if name not in feature_index:
        raise ValueError(f"program feature is absent from Actor: {name}")
    left = _flatten(node["left"], feature_index, arrays)
    right = _flatten(node["right"], feature_index, arrays)
    arrays["children_left"][index] = left
    arrays["children_right"][index] = right
    arrays["feature"][index] = feature_index[name]
    arrays["threshold"][index] = float(node["threshold"])
    return index


def build(args):
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest = migrate_manifest(json.loads(Path(args.runtime_manifest).read_text()))
    manifest_path = output / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(
        manifest, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    manifest_semantic = digest(manifest)

    run = json.loads((source / "run.json").read_text())
    validation = json.loads((source / "validation/step_0800000.json").read_text())
    export = json.loads((source / "export_validation.json").read_text())
    rcpd = json.loads((source / "rcpd/final_program.json").read_text())
    if (run.get("best", {}).get("joint_steps") != 800000
            or validation.get("actor_parameters_sha256")
            != run["best"]["actor_parameters_sha256"]
            or export.get("passed") is not True):
        raise ValueError("selected Actor evidence differs")

    protocol = {
        "version": PROTOCOL_VERSION,
        "runtime_action_override": False,
        "feedback": {"runtime_action_override": False,
                     "updates": int(run.get("feedback_updates", 0))},
        "scenario_sampling": {"manifest_semantic_sha256": manifest_semantic},
        "selection": {
            "joint_steps": 800000,
            "actor_parameters_sha256": validation["actor_parameters_sha256"],
            "validation_passed": False,
            "behavior_performance_gate_waived": True,
            "formal_model_qualified": False,
        },
        "provenance": deepcopy(run.get("parent_provenance")),
    }
    protocol_path = output / "training_protocol.json"
    write_json(protocol_path, protocol)
    protocol_sha = digest(protocol)

    with np.load(source / "actor.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update({
        "public_feedback_mode": "observed",
        "protocol_sha256": protocol_sha,
        "conflict_manifest_semantic_sha256": manifest_semantic,
        "runtime_action_override": False,
        "deployment_class": "internal_pilot_behavior_gate_waived",
    })
    arrays["metadata_json"] = np.asarray(json.dumps(
        metadata, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False))
    actor_path = output / "actor.npz"
    np.savez(actor_path, **arrays)
    actor_sha = sha256(actor_path.read_bytes()).hexdigest()

    selected = rcpd["fit_report"]["selected"]
    tree = {name: [] for name in (
        "children_left", "children_right", "feature", "threshold", "value")}
    feature_index = {name: index for index, name in enumerate(metadata["feature_names"])}
    _flatten(rcpd["program"]["root"], feature_index, tree)
    critical = selected.get("critical", {})
    program = {
        "version": "warehouse-r42-decision-program.v1",
        "actor_sha256": actor_sha,
        "actions": list(metadata["actions"]),
        "tree": tree,
        "audit": {
            "overall_fidelity": float(selected["fidelity"]),
            "non_wait_fidelity": float(selected["fidelity"]),
            "critical_fidelity": min(float(row["fidelity"])
                                     for row in critical.values()),
            "effective_intervention_direction_accuracy": 0.0,
            "mean_kl": float(selected["mean_kl"]),
            "explanation_qualified": False,
        },
        "qualification_waived": True,
        "source_actor_parameters_sha256": metadata["actor_parameters_sha256"],
    }
    program["content_sha256"] = digest(program)
    write_json(output / "program.json", program)

    behavior = deepcopy(validation)
    behavior.update({
        "actor_sha256": actor_sha,
        "passed": False,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
    })
    write_json(output / "behavior_report.json", behavior)
    training = deepcopy(run)
    training.update({
        "actor_sha256": actor_sha,
        "selected_joint_steps": 800000,
        "action_authority": {"runtime_overrides": 0,
                             "policy_action_equals_submitted_action": True},
        "formal_model_qualified": False,
        "behavior_performance_gate_waived": True,
    })
    write_json(output / "training_report.json", training)
    receipt = {
        "version": "warehouse-r46-release-preparation.v1",
        "actor_sha256": actor_sha,
        "actor_parameters_sha256": metadata["actor_parameters_sha256"],
        "selected_joint_steps": 800000,
        "program_sha256": sha256((output / "program.json").read_bytes()).hexdigest(),
        "program_qualified": False,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "export_validation": export,
    }
    write_json(output / "preparation_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
