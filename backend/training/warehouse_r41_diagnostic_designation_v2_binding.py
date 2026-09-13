"""Strict downstream loader for the cycle-free r4.1 designation v2.

The designation deliberately does not know any release consumer.  Consumers
use this small one-way adapter to resolve the five repository-relative frozen
artifacts and then invoke the designation's complete strict reader.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from backend.training import warehouse_r41_diagnostic_designation_v2 as designation
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
    read_authenticated_bytes,
)
from backend.training.warehouse_native_common import canonical, digest, file_hash


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DESIGNATION_SHA256 = (
    "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
)


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " cannot be parsed") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def resolve_bound_components_from_bytes(
    raw: bytes, *, original_path: str | Path,
    expected_sha256: str = EXPECTED_DESIGNATION_SHA256,
) -> dict[str, Path]:
    """Resolve component paths from already authenticated designation bytes."""
    if expected_sha256 != EXPECTED_DESIGNATION_SHA256:
        raise ValueError("Diagnostic designation v2 hash cannot be overridden")
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Exact diagnostic Actor designation v2 bytes required")
    candidate = Path(original_path).expanduser().absolute()
    if candidate.is_symlink() or candidate.resolve() != candidate:
        raise ValueError("Diagnostic designation path is noncanonical")
    saved = _strict_json_bytes(raw, "diagnostic designation v2")
    artifacts = saved.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
            designation.ARTIFACT_NAMES):
        raise ValueError("Diagnostic designation v2 artifact registry differs")
    components: dict[str, Path] = {}
    root = ROOT.resolve()
    for name in designation.ARTIFACT_NAMES:
        row = artifacts.get(name)
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Diagnostic designation v2 artifact row differs")
        relative = row.get("path")
        if type(relative) is not str or Path(relative).is_absolute():
            raise ValueError("Diagnostic designation v2 artifact path differs")
        resolved = (root / relative).absolute()
        try:
            normalized = resolved.resolve().relative_to(root)
        except (OSError, ValueError):
            raise ValueError(
                "Diagnostic designation v2 artifact leaves repository") from None
        if normalized.as_posix() != Path(relative).as_posix():
            raise ValueError("Diagnostic designation v2 artifact path is noncanonical")
        components[name] = resolved
    return components


def _validate_snapshot_components(
    snapshots: Mapping[str, str | Path],
    originals: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Validate exact component snapshots while preserving original identities."""
    names = set(designation.ARTIFACT_NAMES)
    if set(snapshots) != names or set(originals) != names:
        raise ValueError("Exact designation snapshot component set required")
    frozen = {name: Path(snapshots[name]).expanduser().absolute()
              for name in names}
    original = {name: Path(originals[name]).expanduser().absolute()
                for name in names}
    for name in names:
        if (not frozen[name].is_file() or frozen[name].is_symlink()
                or frozen[name].resolve() != frozen[name]
                or original[name].is_symlink()
                or original[name].resolve() != original[name]):
            raise ValueError("Designation snapshot path differs: " + name)
    hashes = {name: file_hash(frozen[name]) for name in names}
    expected = {
        "actor": designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "training_ledger": designation.EXPECTED_LEDGER_SHA256,
        "dual_evaluation": designation.EXPECTED_DUAL_EVALUATION_SHA256,
        "failure_closeout": designation.EXPECTED_CLOSEOUT_SHA256,
    }
    if hashes != expected:
        raise ValueError("Designation snapshot component bytes differ")

    protocol = designation._strict_json(
        frozen["protocol"], "snapshot diagnostic protocol")
    if (digest(protocol) != designation.EXPECTED_PROTOCOL_CONTENT_SHA256
            or protocol.get("version") != "warehouse-r41-active-trainer.v1"
            or protocol.get("maximum_additional_joint_steps")
                != designation.DESIGNATED_STEP
            or protocol.get("runtime_action_override") is not False):
        raise ValueError("Designation snapshot protocol differs")
    actor_metadata = designation._actor_metadata(frozen["actor"])
    if (digest(actor_metadata) != designation.EXPECTED_ACTOR_METADATA_SHA256
            or actor_metadata.get("actor_parameters_sha256")
                != designation.EXPECTED_ACTOR_PARAMETERS_SHA256
            or actor_metadata.get("source_actor_sha256")
                != designation.EXPECTED_SOURCE_ACTOR_SHA256
            or actor_metadata.get("joint_steps") != designation.DESIGNATED_STEP
            or actor_metadata.get("cumulative_joint_steps") != 5_950_000
            or actor_metadata.get("runtime_action_override") is not False
            or actor_metadata.get("action_controller") != "neural_actor_only"
            or actor_metadata.get("protocol_sha256") != digest(protocol)):
        raise ValueError("Designation snapshot Actor metadata differs")

    ledger = designation._strict_json(
        frozen["training_ledger"], "snapshot diagnostic training ledger")
    boundaries = ledger.get("boundaries")
    final = boundaries[-1] if isinstance(boundaries, list) and boundaries else {}
    if (ledger.get("version") != "warehouse-r41-training-ledger.v1"
            or ledger.get("status") != "failed_no_eligible_actor"
            or ledger.get("admission_eligible") is not False
            or ledger.get("selected") is not None
            or ledger.get("total_actual_additional_joint_steps")
                != designation.DESIGNATED_STEP
            or ledger.get("runtime_action_override") is not False
            or ledger.get("protocol_sha256") != hashes["protocol"]
            or ledger.get("protocol_semantic_sha256") != digest(protocol)
            or not isinstance(boundaries, list) or len(boundaries) != 40
            or final.get("step") != designation.DESIGNATED_STEP
            or Path(final.get("actor_path", "")).expanduser().resolve()
                != original["actor"]
            or final.get("actor_sha256") != hashes["actor"]
            or final.get("actor_parameters_sha256")
                != designation.EXPECTED_ACTOR_PARAMETERS_SHA256
            or Path(final.get("evaluation_path", "")).expanduser().resolve()
                != original["dual_evaluation"]
            or final.get("evaluation_sha256") != hashes["dual_evaluation"]
            or final.get("selected") is not False
            or final.get("action_authority", {}).get("overrides") != 0
            or final.get("action_authority", {}).get("trainable")
                != final.get("action_authority", {}).get("equal")):
        raise ValueError("Designation snapshot ledger boundary differs")

    dual = designation._strict_json(
        frozen["dual_evaluation"], "snapshot diagnostic dual evaluation")
    if (dual.get("version") != "warehouse-r41-active-dual-evaluation.v1"
            or dual.get("status") != "failed"
            or dual.get("selected") is not False
            or dual.get("candidate_actor_sha256") != hashes["actor"]
            or dual.get("action_authority_exact") is not True
            or any(dual.get("suite_decisions", {}).get(name, {}).get("selected")
                   is not False for name in
                   ("original_validation", "conflict_validation"))):
        raise ValueError("Designation snapshot dual evaluation differs")

    closeout = designation._strict_json(
        frozen["failure_closeout"], "snapshot diagnostic failure closeout")
    training = closeout.get("training", {})
    terminal = training.get("boundaries", [])
    if (closeout.get("version")
            != "warehouse-r41-budget-failure-closeout.v1"
            or closeout.get("status")
                != "closed_budget_exhausted_no_eligible_actor"
            or training.get("ledger_sha256") != hashes["training_ledger"]
            or training.get("actual_additional_joint_steps")
                != designation.DESIGNATED_STEP
            or training.get("all_boundaries_failed_frozen_gate") is not True
            or training.get("all_policy_actions_submitted_exactly") is not True
            or training.get("action_override_count") != 0
            or not terminal or terminal[-1].get("actor_sha256") != hashes["actor"]
            or closeout.get("diagnostic_models_only") is not True
            or closeout.get("formal_ready") is not False
            or closeout.get("release_eligible") is not False):
        raise ValueError("Designation snapshot failure closeout differs")

    root = ROOT.resolve()
    relatives: dict[str, str] = {}
    for name, path in original.items():
        try:
            relatives[name] = path.relative_to(root).as_posix()
        except ValueError:
            raise ValueError(
                "Designation original component leaves repository") from None
    bindings = {
        "actor_sha256": hashes["actor"],
        "actor_parameters_sha256": designation.EXPECTED_ACTOR_PARAMETERS_SHA256,
        "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": digest(protocol),
        "training_ledger_sha256": hashes["training_ledger"],
        "dual_evaluation_sha256": hashes["dual_evaluation"],
        "failure_closeout_sha256": hashes["failure_closeout"],
        "actor_metadata_sha256": digest(actor_metadata),
        "source_actor_sha256": actor_metadata["source_actor_sha256"],
    }
    return {
        "bindings": bindings,
        "artifacts": {
            name: {"path": relatives[name], "sha256": hashes[name]}
            for name in designation.ARTIFACT_NAMES
        },
        "evidence": {
            "training_budget_exhausted": True,
            "all_frozen_checkpoints_failed_behavior_gate": True,
            "terminal_actor_selected_under_frozen_gate": False,
            "terminal_dual_evaluation_status": "failed",
            "action_authority_exact": True,
            "action_override_count": 0,
            "historical_closeout_unchanged": True,
            "designation_authority": (
                "explicit_user_instruction_after_failure_closeout"),
            "v1_source_closure_retired": True,
        },
        "sources": designation.source_closure(),
    }


def read_bound_designation_snapshot(
    snapshot_path: str | Path, *, original_path: str | Path,
    components: Mapping[str, str | Path],
    original_components: Mapping[str, str | Path],
    expected_sha256: str = EXPECTED_DESIGNATION_SHA256,
) -> dict[str, Any]:
    """Strict designation authentication using immutable component snapshots."""
    if expected_sha256 != EXPECTED_DESIGNATION_SHA256:
        raise ValueError("Diagnostic designation v2 hash cannot be overridden")
    snapshot = Path(snapshot_path).expanduser().absolute()
    if (not snapshot.is_file() or snapshot.is_symlink()
            or snapshot.resolve() != snapshot
            or file_hash(snapshot) != expected_sha256):
        raise ValueError("Exact diagnostic Actor designation snapshot required")
    original = Path(original_path).expanduser().absolute()
    try:
        relative = original.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError("Diagnostic designation must stay in repository") from None
    saved = designation._strict_json(snapshot, "diagnostic designation snapshot")
    if (set(saved) != designation.TOP_LEVEL_FIELDS
            or saved.get("version") != designation.VERSION
            or saved.get("status") != designation.STATUS
            or saved.get("designated") is not True
            or saved.get("release_class") != designation.RELEASE_CLASS
            or saved.get("designated_step") != designation.DESIGNATED_STEP
            or saved.get("supersedes_designation_sha256")
                != designation.SUPERSEDES_DESIGNATION_SHA256
            or saved.get("behavior_performance_gate_passed") is not False
            or saved.get("behavior_performance_gate_waived") is not True
            or saved.get("waiver_scope") != ["behavior_performance"]
            or saved.get("formal_ready") is not False
            or saved.get("formal_sample_eligible") is not False
            or saved.get("human_explanation_effect_validated") is not False
            or saved.get("runtime_action_override") is not False
            or saved.get("test_fixture") is not False
            or saved.get("self_path") != relative
            or not isinstance(saved.get("bindings"), dict)
            or set(saved["bindings"]) != designation.BINDING_FIELDS):
        raise ValueError("Exact cycle-free diagnostic Actor designation v2 required")
    checked = _validate_snapshot_components(components, original_components)
    for key in ("bindings", "artifacts", "evidence", "sources"):
        if canonical(saved.get(key)) != canonical(checked[key]):
            raise ValueError("Designation snapshot " + key + " differs")
    return deepcopy(saved)


def resolve_bound_components(
    path: str | Path, *, expected_sha256: str = EXPECTED_DESIGNATION_SHA256,
) -> dict[str, Path]:
    """Resolve every fixed designation artifact without accepting its contents."""
    if expected_sha256 != EXPECTED_DESIGNATION_SHA256:
        raise ValueError("Diagnostic designation v2 hash cannot be overridden")
    candidate = Path(path).expanduser().absolute()
    raw = read_authenticated_bytes(
        candidate, label="diagnostic Actor designation v2",
        expected_sha256=expected_sha256,
    )
    return resolve_bound_components_from_bytes(
        raw, original_path=candidate, expected_sha256=expected_sha256)


def read_bound_designation(
    path: str | Path, *, expected_sha256: str = EXPECTED_DESIGNATION_SHA256,
) -> dict[str, Any]:
    if expected_sha256 != EXPECTED_DESIGNATION_SHA256:
        raise ValueError("Diagnostic designation v2 hash cannot be overridden")
    candidate = Path(path).expanduser().absolute()
    raw = read_authenticated_bytes(
        candidate, label="diagnostic Actor designation v2",
        expected_sha256=expected_sha256,
    )
    components = resolve_bound_components_from_bytes(
        raw, original_path=candidate, expected_sha256=expected_sha256)
    originals = {"designation": candidate}
    originals.update({"component_" + name: component
                      for name, component in components.items()})
    expected = {
        "designation": expected_sha256,
        "component_actor": designation.EXPECTED_ACTOR_SHA256,
        "component_protocol": designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "component_training_ledger": designation.EXPECTED_LEDGER_SHA256,
        "component_dual_evaluation": designation.EXPECTED_DUAL_EVALUATION_SHA256,
        "component_failure_closeout": designation.EXPECTED_CLOSEOUT_SHA256,
    }
    with ImmutableInputSnapshot(
        originals, expected_sha256=expected,
        prefix="warehouse-r41-designation-v2-inputs-",
    ) as frozen:
        saved = read_bound_designation_snapshot(
            frozen.paths["designation"], original_path=candidate,
            components={name: frozen.paths["component_" + name]
                        for name in components},
            original_components=components,
            expected_sha256=expected_sha256,
        )
        frozen.verify()
        return saved


__all__ = [
    "EXPECTED_DESIGNATION_SHA256",
    "resolve_bound_components", "resolve_bound_components_from_bytes",
    "read_bound_designation", "read_bound_designation_snapshot",
]
