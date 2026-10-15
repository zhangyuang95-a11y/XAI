"""Cycle-free refresh of the terminal r4.1 diagnostic Actor designation.

The v1 designation embedded a live recursive closure rooted in production
admission code.  Later release-chain hardening therefore made its strict
reader fail even though the Actor and all frozen training evidence were
unchanged.  This v2 record re-authenticates the same five immutable evidence
files by their frozen bytes and a small set of structural invariants.  Its
source closure contains only this producer, its builder, and release-neutral
hash/closure helpers; it never imports a training, admission, release, or
deployment reader.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-actor-designation.v2"
STATUS = "designated_terminal_actor_for_internal_diagnostic"
RELEASE_CLASS = "internal_diagnostic"
DESIGNATED_STEP = 2_000_000
SUPERSEDES_DESIGNATION_SHA256 = (
    "d80f2736c6d5e359764f6ae4c09ccfa277f26e98a8910c1f18d86d84cb99851f"
)
EXPECTED_ACTOR_SHA256 = (
    "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
)
EXPECTED_ACTOR_PARAMETERS_SHA256 = (
    "fc9095d0c0e230f4edf0004d66be42e0d176be166d0b071a0576d9954c057ea3"
)
EXPECTED_ACTOR_METADATA_SHA256 = (
    "3ecfe4f63d9ec9a9c39c92eeaeed2f2930c3bcd95073664e0cb0f2b846e6011e"
)
EXPECTED_SOURCE_ACTOR_SHA256 = (
    "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b"
)
EXPECTED_PROTOCOL_FILE_SHA256 = (
    "374ae398115672a243bd2917937ff056fdc762499b470bf11ad44f5fdecc13c8"
)
EXPECTED_PROTOCOL_CONTENT_SHA256 = (
    "3430985e9ebf1ca95560cb8e8db88fc977e19f9765a62c64a28283b8471ccf71"
)
EXPECTED_LEDGER_SHA256 = (
    "7ad59b57e959d357086d62c35991d3fefee9d332e108a83be1abc7fd916a447d"
)
EXPECTED_DUAL_EVALUATION_SHA256 = (
    "cb095a654ede8689a95acaa410c42c94c078dc6d4e8e32dc102ef1eca131d0d4"
)
EXPECTED_CLOSEOUT_SHA256 = (
    "a29539e6596e2ea41b15804732e59776809a1ba64dc5647d842ccc6ce1b6abfd"
)

ARTIFACT_NAMES = (
    "actor", "protocol", "training_ledger", "dual_evaluation",
    "failure_closeout",
)
BINDING_FIELDS = frozenset((
    "actor_sha256", "actor_parameters_sha256", "protocol_file_sha256",
    "protocol_content_sha256", "training_ledger_sha256",
    "dual_evaluation_sha256", "failure_closeout_sha256",
    "actor_metadata_sha256", "source_actor_sha256",
))
TOP_LEVEL_FIELDS = frozenset((
    "version", "status", "designated", "release_class", "designated_step",
    "supersedes_designation_sha256", "behavior_performance_gate_passed",
    "behavior_performance_gate_waived", "waiver_scope", "formal_ready",
    "formal_sample_eligible", "human_explanation_effect_validated",
    "runtime_action_override", "test_fixture", "bindings", "artifacts",
    "evidence", "sources", "self_path",
))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_SOURCE = re.compile(
    r"(?:^|/)(?:.*admission.*|.*release.*|.*preflight.*)\.py\Z")


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > 512 * 1024 * 1024):
        raise ValueError(label + " is missing, linked, noncanonical, or oversized")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _repo_file(value: str | Path, label: str) -> tuple[Path, str]:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    if ".." in Path(relative).parts:
        raise ValueError(label + " has an unsafe repository path")
    return path, relative


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def source_closure() -> dict[str, str]:
    sources = local_source_hashes((
        Path(__file__).resolve(),
        ROOT / "scripts/build_warehouse_r41_diagnostic_designation_v2.py",
    ))
    if any(_FORBIDDEN_SOURCE.search(path) for path in sources):
        raise ValueError("Designation v2 source closure entered a release cycle")
    return dict(sorted(sources.items()))


def _actor_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                    "metadata_json", "0.weight", "0.bias", "2.weight", "2.bias",
                    "4.weight", "4.bias"}:
                raise ValueError("Frozen Actor archive schema differs")
            raw = archive["metadata_json"]
            if raw.shape != () or raw.dtype.kind not in "US":
                raise ValueError("Frozen Actor metadata encoding differs")
            metadata = json.loads(str(raw.item()))
            for name in archive.files:
                if name != "metadata_json" and not np.isfinite(archive[name]).all():
                    raise ValueError("Frozen Actor contains a non-finite parameter")
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Frozen Actor archive cannot be authenticated") from error
    if not isinstance(metadata, dict):
        raise ValueError("Frozen Actor metadata must be one JSON object")
    return metadata


def validate_components(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    if not isinstance(paths, Mapping) or set(paths) != set(ARTIFACT_NAMES):
        raise ValueError("Exact r4.1 diagnostic designation artifact set required")
    files: dict[str, Path] = {}
    relatives: dict[str, str] = {}
    for name in ARTIFACT_NAMES:
        files[name], relatives[name] = _repo_file(paths[name], name)
    hashes = {name: file_hash(path) for name, path in files.items()}
    expected = {
        "actor": EXPECTED_ACTOR_SHA256,
        "protocol": EXPECTED_PROTOCOL_FILE_SHA256,
        "training_ledger": EXPECTED_LEDGER_SHA256,
        "dual_evaluation": EXPECTED_DUAL_EVALUATION_SHA256,
        "failure_closeout": EXPECTED_CLOSEOUT_SHA256,
    }
    if hashes != expected:
        changed = sorted(name for name in hashes if hashes[name] != expected[name])
        raise ValueError("Designation v2 input identity differs: " + ", ".join(changed))

    protocol = _strict_json(files["protocol"], "r4.1 diagnostic protocol")
    if (digest(protocol) != EXPECTED_PROTOCOL_CONTENT_SHA256
            or protocol.get("version") != "warehouse-r41-active-trainer.v1"
            or protocol.get("maximum_additional_joint_steps") != DESIGNATED_STEP
            or protocol.get("runtime_action_override") is not False):
        raise ValueError("Designation v2 protocol differs")

    actor_metadata = _actor_metadata(files["actor"])
    if (digest(actor_metadata) != EXPECTED_ACTOR_METADATA_SHA256
            or actor_metadata.get("actor_parameters_sha256")
                != EXPECTED_ACTOR_PARAMETERS_SHA256
            or actor_metadata.get("source_actor_sha256")
                != EXPECTED_SOURCE_ACTOR_SHA256
            or actor_metadata.get("joint_steps") != DESIGNATED_STEP
            or actor_metadata.get("cumulative_joint_steps") != 5_950_000
            or actor_metadata.get("runtime_action_override") is not False
            or actor_metadata.get("action_controller") != "neural_actor_only"
            or actor_metadata.get("protocol_sha256") != digest(protocol)):
        raise ValueError("Designation v2 Actor metadata differs")

    ledger = _strict_json(files["training_ledger"], "r4.1 training ledger")
    boundaries = ledger.get("boundaries")
    final = boundaries[-1] if isinstance(boundaries, list) and boundaries else {}
    if (ledger.get("version") != "warehouse-r41-training-ledger.v1"
            or ledger.get("status") != "failed_no_eligible_actor"
            or ledger.get("admission_eligible") is not False
            or ledger.get("selected") is not None
            or ledger.get("total_actual_additional_joint_steps") != DESIGNATED_STEP
            or ledger.get("runtime_action_override") is not False
            or ledger.get("protocol_sha256") != hashes["protocol"]
            or ledger.get("protocol_semantic_sha256") != digest(protocol)
            or len(boundaries) != 40 or final.get("step") != DESIGNATED_STEP
            or Path(final.get("actor_path", "")).expanduser().resolve()
                != files["actor"]
            or final.get("actor_sha256") != hashes["actor"]
            or final.get("actor_parameters_sha256")
                != EXPECTED_ACTOR_PARAMETERS_SHA256
            or Path(final.get("evaluation_path", "")).expanduser().resolve()
                != files["dual_evaluation"]
            or final.get("evaluation_sha256") != hashes["dual_evaluation"]
            or final.get("selected") is not False
            or final.get("action_authority", {}).get("overrides") != 0
            or final.get("action_authority", {}).get("trainable")
                != final.get("action_authority", {}).get("equal")):
        raise ValueError("Designation v2 ledger boundary differs")

    dual = _strict_json(files["dual_evaluation"], "r4.1 dual evaluation")
    if (dual.get("version") != "warehouse-r41-active-dual-evaluation.v1"
            or dual.get("status") != "failed" or dual.get("selected") is not False
            or dual.get("candidate_actor_sha256") != hashes["actor"]
            or dual.get("action_authority_exact") is not True
            or any(dual.get("suite_decisions", {}).get(name, {}).get("selected")
                   is not False for name in
                   ("original_validation", "conflict_validation"))):
        raise ValueError("Designation v2 dual evaluation differs")

    closeout = _strict_json(files["failure_closeout"], "r4.1 failure closeout")
    training = closeout.get("training", {})
    terminal = training.get("boundaries", [])
    if (closeout.get("version") != "warehouse-r41-budget-failure-closeout.v1"
            or closeout.get("status")
                != "closed_budget_exhausted_no_eligible_actor"
            or training.get("ledger_sha256") != hashes["training_ledger"]
            or training.get("actual_additional_joint_steps") != DESIGNATED_STEP
            or training.get("all_boundaries_failed_frozen_gate") is not True
            or training.get("all_policy_actions_submitted_exactly") is not True
            or training.get("action_override_count") != 0
            or not terminal or terminal[-1].get("actor_sha256") != hashes["actor"]
            or closeout.get("diagnostic_models_only") is not True
            or closeout.get("formal_ready") is not False
            or closeout.get("release_eligible") is not False):
        raise ValueError("Designation v2 failure closeout differs")

    bindings = {
        "actor_sha256": hashes["actor"],
        "actor_parameters_sha256": EXPECTED_ACTOR_PARAMETERS_SHA256,
        "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": digest(protocol),
        "training_ledger_sha256": hashes["training_ledger"],
        "dual_evaluation_sha256": hashes["dual_evaluation"],
        "failure_closeout_sha256": hashes["failure_closeout"],
        "actor_metadata_sha256": digest(actor_metadata),
        "source_actor_sha256": actor_metadata["source_actor_sha256"],
    }
    if set(bindings) != BINDING_FIELDS:
        raise AssertionError("Designation v2 binding schema differs")
    return {
        "bindings": bindings,
        "artifacts": {
            name: {"path": relatives[name], "sha256": hashes[name]}
            for name in ARTIFACT_NAMES
        },
        "evidence": {
            "training_budget_exhausted": True,
            "all_frozen_checkpoints_failed_behavior_gate": True,
            "terminal_actor_selected_under_frozen_gate": False,
            "terminal_dual_evaluation_status": "failed",
            "action_authority_exact": True,
            "action_override_count": 0,
            "historical_closeout_unchanged": True,
            "designation_authority": "explicit_user_instruction_after_failure_closeout",
            "v1_source_closure_retired": True,
        },
        "sources": source_closure(),
    }


def read_superseded_designation(
    path: str | Path, *, components: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Authenticate the exact v1 predecessor without its stale live closure."""
    saved_path, _ = _repo_file(path, "superseded v1 designation")
    if file_hash(saved_path) != SUPERSEDES_DESIGNATION_SHA256:
        raise ValueError("Superseded designation bytes differ")
    saved = _strict_json(saved_path, "superseded v1 designation")
    checked = validate_components(components)
    if (saved.get("version") != "warehouse-r41-diagnostic-actor-designation.v1"
            or saved.get("status") != STATUS
            or saved.get("designated") is not True
            or saved.get("waiver_scope") != ["behavior_performance"]
            or saved.get("runtime_action_override") is not False
            or saved.get("formal_ready") is not False
            or saved.get("formal_sample_eligible") is not False
            or saved.get("bindings") != checked["bindings"]
            or {name: row.get("sha256") for name, row in
                saved.get("artifacts", {}).items()} != {
                    name: checked["artifacts"][name]["sha256"]
                    for name in ARTIFACT_NAMES
                }):
        raise ValueError("Superseded designation evidence differs")
    return deepcopy(saved)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink() or path.resolve() != path
            or path.parent.is_symlink() or path.parent.resolve() != path.parent):
        raise ValueError("Designation v2 output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_designation(
    paths: Mapping[str, str | Path], *, predecessor_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.absolute()
    try:
        relative = output_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError("Designation v2 output must stay inside the repository") from None
    checked = validate_components(paths)
    read_superseded_designation(predecessor_path, components=paths)
    designation = {
        "version": VERSION,
        "status": STATUS,
        "designated": True,
        "release_class": RELEASE_CLASS,
        "designated_step": DESIGNATED_STEP,
        "supersedes_designation_sha256": SUPERSEDES_DESIGNATION_SHA256,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False,
        "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "runtime_action_override": False,
        "test_fixture": False,
        "bindings": checked["bindings"],
        "artifacts": checked["artifacts"],
        "evidence": checked["evidence"],
        "sources": checked["sources"],
        "self_path": relative,
    }
    _write_new(output_path, designation)
    sha256 = file_hash(output_path)
    try:
        read_saved_designation(
            output_path, expected_sha256=sha256, components=paths)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return designation


def read_saved_designation(
    path: str | Path, *, expected_sha256: str,
    components: Mapping[str, str | Path],
) -> dict[str, Any]:
    saved_path, relative = _repo_file(path, "r4.1 diagnostic designation v2")
    if file_hash(saved_path) != _sha(expected_sha256, "diagnostic designation v2"):
        raise ValueError("Designation v2 bytes differ")
    saved = _strict_json(saved_path, "r4.1 diagnostic designation v2")
    if (set(saved) != TOP_LEVEL_FIELDS or saved.get("version") != VERSION
            or saved.get("status") != STATUS or saved.get("designated") is not True
            or saved.get("release_class") != RELEASE_CLASS
            or saved.get("designated_step") != DESIGNATED_STEP
            or saved.get("supersedes_designation_sha256")
                != SUPERSEDES_DESIGNATION_SHA256
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
            or set(saved["bindings"]) != BINDING_FIELDS):
        raise ValueError("Exact cycle-free diagnostic Actor designation v2 required")
    checked = validate_components(components)
    for key in ("bindings", "artifacts", "evidence", "sources"):
        if canonical(saved.get(key)) != canonical(checked[key]):
            raise ValueError("Designation v2 " + key + " differs from live evidence")
    return deepcopy(saved)


__all__ = [
    "VERSION", "STATUS", "RELEASE_CLASS", "DESIGNATED_STEP",
    "SUPERSEDES_DESIGNATION_SHA256", "EXPECTED_ACTOR_SHA256",
    "EXPECTED_ACTOR_PARAMETERS_SHA256", "EXPECTED_ACTOR_METADATA_SHA256",
    "EXPECTED_SOURCE_ACTOR_SHA256", "EXPECTED_PROTOCOL_FILE_SHA256",
    "EXPECTED_PROTOCOL_CONTENT_SHA256", "EXPECTED_LEDGER_SHA256",
    "EXPECTED_DUAL_EVALUATION_SHA256", "EXPECTED_CLOSEOUT_SHA256",
    "ARTIFACT_NAMES", "BINDING_FIELDS", "source_closure",
    "validate_components", "read_superseded_designation",
    "build_designation", "read_saved_designation",
]
