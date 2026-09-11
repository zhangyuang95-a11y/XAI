"""Strict r4.1 training ledger and earliest-passing Actor selection.

Only immutable, atomically committed 50k boundaries count as fresh PPO work.
The ledger has one r3 parent and no r4 parent edges.  It verifies the shared
conflict dynamics, both validation suites, RCPD provenance, and exact neural
action authority before exposing a selected Actor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from . import warehouse_r41_active_evaluation as evaluation
from . import warehouse_r41_active_run as run_module
from .warehouse_native_common import atomic_json, digest, file_hash
from .warehouse_r41_active_trainer import (
    BOUNDARY_INTERVAL, MAXIMUM_ADDITIONAL_JOINT_STEPS, PARTNER_MIX,
    SHAPING, SOURCE_ACTOR_SHA256, SOURCE_CHECKPOINT_SHA256,
    SOURCE_CUMULATIVE_STEPS, VERSION as TRAINER_VERSION,
    _implementation_sources,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-training-ledger.v1"
_BOUNDARY = re.compile(r"step_(\d{7})$")


def _read(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Non-finite JSON value in {path}: {value}")))


def _bound_file(root: Path, relative: Any, expected_sha: Any, label: str) -> Path:
    if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", relative):
        raise ValueError(label + " has an invalid relative path")
    supplied = Path(relative)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError(label + " must stay inside its boundary")
    path = (root / supplied).resolve()
    if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(label + " is not a regular boundary file")
    if not isinstance(expected_sha, str) or file_hash(path) != expected_sha:
        raise ValueError(label + " SHA-256 differs")
    return path


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(label + " must be finite")
    return float(value)


def _segment(path: Path, *, start: int, end: int, environments: int):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("Blank r4.1 training journal line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("R4.1 training journal row is not an object")
            rows.append(row)
    cursor = start
    for row in rows:
        if (row.get("start_joint_step") != cursor
                or type(row.get("end_joint_step")) is not int
                or type(row.get("time_steps")) is not int
                or row["time_steps"] <= 0
                or row["end_joint_step"] - cursor != row["time_steps"] * environments
                or row["end_joint_step"] > end):
            raise ValueError("R4.1 training journal has a gap or invalid increment")
        for key in ("feedback_lambda", "actor_loss", "critic_loss", "feedback_loss"):
            _finite(row.get(key), "training journal " + key)
        if not 0. <= float(row["feedback_lambda"]) <= .01:
            raise ValueError("R4.1 feedback lambda is unbounded")
        cursor = row["end_joint_step"]
    if cursor != end or not rows:
        raise ValueError("R4.1 boundary journal does not cover exactly 50k steps")
    return {"rows": len(rows), "start_step": start, "end_step": end,
            "sha256": file_hash(path)}


def _evaluation(report, *, actor_sha: str, original_binding: dict,
                conflict_binding: dict):
    if (report.get("version") != evaluation.VERSION
            or report.get("candidate_actor_sha256") != actor_sha
            or report.get("baseline_actor_sha256") != SOURCE_ACTOR_SHA256
            or report.get("manifests", {}).get("original", {}).get("file_sha256")
                != original_binding["file_sha256"]
            or report.get("manifests", {}).get("original", {}).get(
                "validation_entries_sha256") != original_binding["validation_entries_sha256"]
            or report.get("manifests", {}).get("conflict", {}).get("file_sha256")
                != conflict_binding["file_sha256"]
            or report.get("manifests", {}).get("conflict", {}).get("semantic_sha256")
                != conflict_binding["semantic_sha256"]
            or report.get("manifests", {}).get("conflict", {}).get("contract_sha256")
                != conflict_binding["contract_sha256"]):
        raise ValueError("R4.1 dual evaluation bindings differ")
    decisions = {}
    for name, suite in (("original_validation", "original"),
                        ("conflict_validation", "conflict")):
        base = report[suite]["baseline"]["summary"]
        candidate = report[suite]["candidate"]["summary"]
        decision = evaluation._suite_decision(base, candidate)
        if report.get("suite_decisions", {}).get(name) != decision:
            raise ValueError("R4.1 stored suite decision differs from registered gates")
        decisions[name] = decision
        if (base.get("action_override_count") != 0
                or candidate.get("action_override_count") != 0
                or base.get("policy_action_equality_rate") != 1.
                or candidate.get("policy_action_equality_rate") != 1.):
            raise ValueError("R4.1 dual evaluation contains action overrides")
    selected = all(item["selected"] for item in decisions.values())
    if (report.get("action_authority_exact") is not True
            or report.get("selected") is not selected
            or report.get("status") != ("passed" if selected else "failed")):
        raise ValueError("R4.1 dual evaluation selection differs")
    return selected


def _boundary(path: Path, *, step: int, prior_step: int, protocol: dict,
              source_receipt: dict):
    summary = _read(path / "summary.json")
    if (summary.get("version") != run_module.RUN_VERSION
            or summary.get("step") != step
            or summary.get("cumulative_joint_steps") != SOURCE_CUMULATIVE_STEPS + step
            or summary.get("runtime_action_override") is not False
            or summary.get("partner_mix") != PARTNER_MIX):
        raise ValueError("R4.1 boundary summary identity differs")
    actor_info = summary.get("actor", {})
    actor_path = _bound_file(path, actor_info.get("path"), actor_info.get("sha256"),
                             "boundary Actor")
    actor = NumPyNativeActor(actor_path)
    if (actor.metadata.get("experiment_version") != TRAINER_VERSION
            or actor.metadata.get("joint_steps") != step
            or actor.metadata.get("source_actor_sha256") != SOURCE_ACTOR_SHA256
            or actor.metadata.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or actor.metadata.get("protocol_sha256") != digest(protocol)
            or actor.metadata.get("conflict_contract_sha256")
                != source_receipt["conflict_manifest"]["contract_sha256"]
            or actor.metadata.get("conflict_contract_version")
                != source_receipt["conflict_manifest"]["contract_version"]
            or actor.metadata.get("conflict_graph_sha256")
                != source_receipt["conflict_manifest"]["graph_sha256"]
            or actor.metadata.get("conflict_manifest_version")
                != source_receipt["conflict_manifest"]["version"]
            or actor.metadata.get("conflict_manifest_semantic_sha256")
                != source_receipt["conflict_manifest"]["semantic_sha256"]
            or actor.metadata.get("runtime_action_override") is not False
            or actor.metadata.get("failed_r4_parent_used") is not False):
        raise ValueError("R4.1 boundary Actor metadata differs")
    if actor.metadata.get("actor_parameters_sha256") != actor_info.get("parameters_sha256"):
        raise ValueError("R4.1 Actor parameter digest differs from its summary")
    checkpoint_info = summary.get("checkpoint", {})
    checkpoint_path = _bound_file(path, checkpoint_info.get("path"),
                                  checkpoint_info.get("sha256"), "boundary checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer = checkpoint.get("trainer", {})
    if (checkpoint.get("run_version") != run_module.RUN_VERSION
            or checkpoint.get("runtime_action_override") is not False
            or checkpoint.get("joint_steps") != step
            or checkpoint.get("protocol_sha256") != digest(protocol)
            or trainer.get("protocol") != protocol
            or trainer.get("source_state_sha256") != source_receipt["source_state_sha256"]
            or trainer.get("action_authority", {}).get("trainable")
                != trainer.get("action_authority", {}).get("equal")):
        raise ValueError("R4.1 boundary checkpoint binding differs")
    segment_info = summary.get("training_segment", {})
    segment_path = _bound_file(path, segment_info.get("path"), segment_info.get("sha256"),
                               "training segment")
    segment = _segment(segment_path, start=prior_step, end=step,
                       environments=int(protocol["training"]["environments"]))
    tree_info = summary.get("RCPD", {})
    tree_path = _bound_file(path, tree_info.get("report_path"),
                            tree_info.get("report_sha256"), "RCPD report")
    program_path = _bound_file(path, tree_info.get("program_path"),
                               tree_info.get("program_sha256"), "RCPD program")
    tree = _read(tree_path)
    program = _read(program_path)
    if (tree.get("r41_conflict_binding") != {
            "contract_sha256": source_receipt["conflict_manifest"]["contract_sha256"],
            "contract_version": source_receipt["conflict_manifest"]["contract_version"],
            "graph_sha256": source_receipt["conflict_manifest"]["graph_sha256"],
            "manifest_version": source_receipt["conflict_manifest"]["version"],
            "manifest_semantic_sha256": source_receipt["conflict_manifest"]["semantic_sha256"],
            "train_entries_sha256": protocol["scenario_sampling"]["train_entries_sha256"],
            "runtime_action_override": False,
            } or program.get("metadata", {}).get("native_source_actor_sha256") != actor_info["sha256"]):
        raise ValueError("R4.1 RCPD does not bind the boundary Actor and conflict dynamics")
    reliable = bool(tree_info.get("reliable_for_training"))
    if (tree.get("reliable") is not reliable
            or tree.get("selected", {}).get("reliable") is not reliable):
        raise ValueError("R4.1 RCPD reliability differs from its fit report")
    eval_info = summary.get("dual_evaluation", {})
    eval_path = _bound_file(path, eval_info.get("path"), eval_info.get("sha256"),
                            "dual evaluation")
    eval_report = _read(eval_path)
    evaluated = _evaluation(
        eval_report, actor_sha=actor_info["sha256"],
        original_binding=source_receipt["original_validation"],
        conflict_binding=source_receipt["conflict_manifest"],
    )
    authority = summary.get("action_authority", {})
    if (authority.get("trainable") != authority.get("equal")
            or authority.get("overrides") != 0):
        raise ValueError("R4.1 training action authority differs")
    selected = bool(evaluated and tree_info.get("reliable_for_training")
                    and authority["overrides"] == 0)
    if (summary.get("candidate_selected") is not selected
            or eval_info.get("selected") is not evaluated):
        raise ValueError("R4.1 boundary selection differs")
    return {
        "step": step, "actor_path": str(actor_path),
        "actor_sha256": actor_info["sha256"],
        "actor_parameters_sha256": actor_info["parameters_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "program_path": str(program_path), "program_sha256": tree_info["program_sha256"],
        "tree_fit_path": str(tree_path), "tree_fit_sha256": tree_info["report_sha256"],
        "evaluation_path": str(eval_path), "evaluation_sha256": eval_info["sha256"],
        "training_segment": segment, "selected": selected,
        "action_authority": authority,
    }


def build(run_root, output_path):
    root = Path(run_root).resolve()
    output_path = Path(output_path).resolve()
    if not root.is_dir() or output_path.exists():
        raise ValueError("R4.1 ledger requires an existing run and new output file")
    protocol = _read(root / "protocol.json")
    receipt = _read(root / "source_receipt.json")
    run = _read(root / "run.json")
    if (protocol.get("version") != TRAINER_VERSION
            or protocol.get("source", {}).get("actor_sha256") != SOURCE_ACTOR_SHA256
            or protocol.get("source", {}).get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or protocol.get("source", {}).get("cumulative_joint_steps") != SOURCE_CUMULATIVE_STEPS
            or protocol.get("source", {}).get("failed_r4_parent_used") is not False
            or protocol.get("partners") != PARTNER_MIX
            or protocol.get("shaping") != SHAPING
            or protocol.get("implementation_sources") != _implementation_sources()
            or protocol.get("maximum_additional_joint_steps") != MAXIMUM_ADDITIONAL_JOINT_STEPS
            or protocol.get("runtime_action_override") is not False):
        raise ValueError("R4.1 protocol differs from the registered fresh-r3 continuation")
    if (receipt.get("version") != run_module.RUN_VERSION
            or receipt.get("source_actor_sha256") != SOURCE_ACTOR_SHA256
            or receipt.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or receipt.get("failed_r4_parent_used") is not False
            or receipt.get("runtime_action_override") is not False):
        raise ValueError("R4.1 source receipt differs")
    conflict_receipt = receipt.get("conflict_manifest", {})
    scenario_sampling = protocol["scenario_sampling"]
    if (conflict_receipt.get("contract_sha256")
            != scenario_sampling["conflict_contract_sha256"]
            or conflict_receipt.get("contract_version")
                != scenario_sampling["conflict_contract_version"]
            or conflict_receipt.get("graph_sha256")
                != scenario_sampling["conflict_graph_sha256"]
            or conflict_receipt.get("version")
                != scenario_sampling["manifest_version"]
            or conflict_receipt.get("file_sha256")
                != scenario_sampling["manifest_file_sha256"]
            or conflict_receipt.get("semantic_sha256")
                != scenario_sampling["manifest_semantic_sha256"]):
        raise ValueError("R4.1 source receipt conflict contract differs")
    boundary_paths = []
    for path in (root / "boundaries").glob("step_*") if (root / "boundaries").exists() else ():
        match = _BOUNDARY.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            boundary_paths.append((int(match.group(1)), path))
    boundary_paths.sort()
    boundaries = []
    prior = 0
    for step, path in boundary_paths:
        if step != prior + BOUNDARY_INTERVAL or step > MAXIMUM_ADDITIONAL_JOINT_STEPS:
            raise ValueError("R4.1 committed boundaries are not contiguous 50k segments")
        boundaries.append(_boundary(path, step=step, prior_step=prior,
                                    protocol=protocol, source_receipt=receipt))
        prior = step
    passing = [item for item in boundaries if item["selected"]]
    if len(passing) > 1 or (passing and passing[0] is not boundaries[-1]):
        raise ValueError("R4.1 did not stop at the earliest passing boundary")
    selected = passing[0] if passing else None
    expected_status = ("selected_early" if selected else
                       "budget_exhausted_no_selection" if prior == MAXIMUM_ADDITIONAL_JOINT_STEPS else
                       "target_completed_no_selection")
    if (run.get("version") != run_module.RUN_VERSION
            or run.get("current_additional_joint_steps") != prior
            or run.get("cumulative_joint_steps") != SOURCE_CUMULATIVE_STEPS + prior
            or run.get("maximum_additional_joint_steps") != MAXIMUM_ADDITIONAL_JOINT_STEPS
            or run.get("runtime_action_override") is not False
            or run.get("status") != expected_status
            or run.get("selected") is not bool(selected)):
        raise ValueError("R4.1 terminal run state differs from committed boundaries")
    selection_path = root / "selection.json"
    if selected:
        top = _read(selection_path)
        if (top.get("status") != "selected"
                or top.get("earliest_passing_step") != selected["step"]
                or top.get("actor_sha256") != selected["actor_sha256"]
                or top.get("runtime_action_override") is not False):
            raise ValueError("R4.1 top-level selection differs")
    elif selection_path.exists():
        raise ValueError("Failed r4.1 run cannot contain a selection artifact")
    report = {
        "version": VERSION,
        "status": "selected" if selected else "failed_no_eligible_actor",
        "admission_eligible": bool(selected),
        "source": {
            "actor_sha256": SOURCE_ACTOR_SHA256,
            "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
            "failed_r4_parent_used": False,
        },
        "maximum_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS,
        "total_actual_additional_joint_steps": prior,
        "boundary_interval": BOUNDARY_INTERVAL,
        "boundaries": boundaries,
        "selected": selected,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "protocol_semantic_sha256": digest(protocol),
        "source_receipt_sha256": file_hash(root / "source_receipt.json"),
        "run_sha256": file_hash(root / "run.json"),
        "conflict_contract_sha256": protocol["scenario_sampling"]["conflict_contract_sha256"],
        "conflict_contract_version": protocol["scenario_sampling"]["conflict_contract_version"],
        "conflict_graph_sha256": protocol["scenario_sampling"]["conflict_graph_sha256"],
        "conflict_manifest_version": protocol["scenario_sampling"]["manifest_version"],
        "conflict_manifest_semantic_sha256": protocol["scenario_sampling"]["manifest_semantic_sha256"],
        "original_validation_entries_sha256": protocol["evaluation"]["original_validation"][
            "validation_entries_sha256"],
        "conflict_validation_entries_sha256": protocol["evaluation"][
            "conflict_validation_entries_sha256"],
        "runtime_action_override": False,
        "budget_semantics": "unique contiguous PPO joint steps from the frozen r3 parent",
    }
    atomic_json(output_path, report)
    return report


def read_saved_ledger(path, *, expected_sha256, require_selected=True):
    path = Path(path).resolve()
    if file_hash(path) != expected_sha256:
        raise ValueError("R4.1 ledger file SHA-256 differs")
    report = _read(path)
    if (report.get("version") != VERSION
            or report.get("runtime_action_override") is not False
            or report.get("source", {}).get("actor_sha256") != SOURCE_ACTOR_SHA256
            or report.get("source", {}).get("failed_r4_parent_used") is not False
            or report.get("maximum_additional_joint_steps") != MAXIMUM_ADDITIONAL_JOINT_STEPS
            or not 0 <= report.get("total_actual_additional_joint_steps", -1)
                <= MAXIMUM_ADDITIONAL_JOINT_STEPS):
        raise ValueError("R4.1 saved ledger identity differs")
    if require_selected and (report.get("status") != "selected"
                             or report.get("admission_eligible") is not True
                             or not isinstance(report.get("selected"), dict)):
        raise ValueError("R4.1 ledger has no eligible Actor")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(args.run_root, args.output)
    print(json.dumps({"status": report["status"],
                      "admission_eligible": report["admission_eligible"],
                      "total_actual_additional_joint_steps": report[
                          "total_actual_additional_joint_steps"],
                      "output": str(Path(args.output).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
