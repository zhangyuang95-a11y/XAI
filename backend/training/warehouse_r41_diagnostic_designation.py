"""Designate the terminal r4.1 Actor for a non-formal diagnostic release.

The frozen r4.1 production gate rejected every checkpoint.  This module does
not revise that result.  It authenticates the terminal 2M-step Actor and the
immutable failure evidence, then records the user's narrower decision to use
that exact Actor in an internal explanation diagnostic.  Only behavioral
performance is waived; action authority remains a hard invariant.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from backend.training import warehouse_r41_failure_closeout as closeout_api
from backend.training import warehouse_r41_training_ledger as ledger_api
from backend.training.warehouse_native_common import canonical, digest, file_hash
from env.warehouse_native.policy import NumPyNativeActor


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-actor-designation.v1"
STATUS = "designated_terminal_actor_for_internal_diagnostic"
RELEASE_CLASS = "internal_diagnostic"
DESIGNATED_STEP = 2_000_000

# These are deliberately constants, not caller-selected values.  A later run
# needs a new designation version rather than silently changing this identity.
EXPECTED_ACTOR_SHA256 = (
    "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
)
EXPECTED_ACTOR_PARAMETERS_SHA256 = (
    "fc9095d0c0e230f4edf0004d66be42e0d176be166d0b071a0576d9954c057ea3"
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
    "behavior_performance_gate_passed", "behavior_performance_gate_waived",
    "waiver_scope", "formal_ready", "formal_sample_eligible",
    "human_explanation_effect_validated", "runtime_action_override",
    "test_fixture", "bindings", "artifacts", "evidence", "sources",
    "self_path",
))
_HEX = re.compile(r"[0-9a-f]{64}\Z")


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
    from backend.training.warehouse_r4_production_admission import local_source_hashes

    return local_source_hashes((
        Path(__file__), ROOT / "scripts/build_warehouse_r41_diagnostic_designation.py",
        Path(ledger_api.__file__), Path(closeout_api.__file__),
        ROOT / "env/warehouse_native/policy.py",
    ))


def _reconstruct_ledger(path: Path, expected_sha256: str) -> dict[str, Any]:
    saved = ledger_api.read_saved_ledger(
        path, expected_sha256=expected_sha256, require_selected=False,
    )
    boundaries = saved.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("Diagnostic designation requires the complete r4.1 ledger")
    actor_path = Path(boundaries[-1].get("actor_path", "")).expanduser().resolve()
    run_root = actor_path.parent.parent.parent
    with tempfile.TemporaryDirectory(prefix="warehouse-r41-diagnostic-ledger-") as tmp:
        rebuilt = ledger_api.build(run_root, Path(tmp) / "training_ledger.json")
    if canonical(rebuilt) != canonical(saved):
        raise ValueError("Diagnostic ledger differs from reconstructed run evidence")
    return saved


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
        raise ValueError("Diagnostic designation input identity differs: " + ", ".join(changed))

    protocol = _strict_json(files["protocol"], "r4.1 diagnostic protocol")
    if (digest(protocol) != EXPECTED_PROTOCOL_CONTENT_SHA256
            or protocol.get("version") != "warehouse-r41-active-trainer.v1"
            or protocol.get("maximum_additional_joint_steps") != DESIGNATED_STEP
            or protocol.get("runtime_action_override") is not False):
        raise ValueError("Diagnostic protocol is not the frozen 2M-step no-override protocol")

    ledger = _reconstruct_ledger(files["training_ledger"], hashes["training_ledger"])
    boundaries = ledger.get("boundaries", [])
    final = boundaries[-1]
    if (ledger.get("status") != "failed_no_eligible_actor"
            or ledger.get("admission_eligible") is not False
            or ledger.get("selected") is not None
            or ledger.get("total_actual_additional_joint_steps") != DESIGNATED_STEP
            or ledger.get("runtime_action_override") is not False
            or ledger.get("protocol_sha256") != hashes["protocol"]
            or ledger.get("protocol_semantic_sha256") != digest(protocol)
            or final.get("step") != DESIGNATED_STEP
            or Path(final.get("actor_path", "")).expanduser().resolve() != files["actor"]
            or final.get("actor_sha256") != hashes["actor"]
            or final.get("actor_parameters_sha256") != EXPECTED_ACTOR_PARAMETERS_SHA256
            or Path(final.get("evaluation_path", "")).expanduser().resolve()
                != files["dual_evaluation"]
            or final.get("evaluation_sha256") != hashes["dual_evaluation"]
            or final.get("selected") is not False
            or final.get("action_authority", {}).get("overrides") != 0
            or final.get("action_authority", {}).get("trainable")
                != final.get("action_authority", {}).get("equal")):
        raise ValueError("Diagnostic designation is not the terminal exact-action boundary")

    dual = _strict_json(files["dual_evaluation"], "r4.1 failed dual evaluation")
    if (dual.get("version") != "warehouse-r41-active-dual-evaluation.v1"
            or dual.get("status") != "failed" or dual.get("selected") is not False
            or dual.get("candidate_actor_sha256") != hashes["actor"]
            or dual.get("action_authority_exact") is not True
            or any(dual.get("suite_decisions", {}).get(name, {}).get("selected") is not False
                   for name in ("original_validation", "conflict_validation"))):
        raise ValueError("Diagnostic designation requires the authentic failed dual report")

    closeout = closeout_api.read_saved_closeout(
        files["failure_closeout"], expected_sha256=hashes["failure_closeout"],
        require_current_sources=False,
    )
    terminal = closeout.get("training", {})
    terminal_rows = terminal.get("boundaries", [])
    if (terminal.get("ledger_sha256") != hashes["training_ledger"]
            or terminal.get("actual_additional_joint_steps") != DESIGNATED_STEP
            or terminal.get("all_policy_actions_submitted_exactly") is not True
            or terminal.get("action_override_count") != 0
            or not terminal_rows or terminal_rows[-1].get("actor_sha256") != hashes["actor"]
            or closeout.get("diagnostic_models_only") is not True
            or closeout.get("formal_ready") is not False
            or closeout.get("release_eligible") is not False):
        raise ValueError("Diagnostic designation differs from the immutable failure closeout")

    actor = NumPyNativeActor(files["actor"])
    if (actor.sha256 != hashes["actor"]
            or actor.metadata.get("actor_parameters_sha256")
                != EXPECTED_ACTOR_PARAMETERS_SHA256
            or actor.metadata.get("joint_steps") != DESIGNATED_STEP
            or actor.metadata.get("cumulative_joint_steps") != 5_950_000
            or actor.metadata.get("runtime_action_override") is not False
            or actor.metadata.get("action_controller") != "neural_actor_only"
            or actor.metadata.get("protocol_sha256") != digest(protocol)):
        raise ValueError("Diagnostic Actor metadata differs from the frozen terminal Actor")
    bindings = {
        "actor_sha256": hashes["actor"],
        "actor_parameters_sha256": EXPECTED_ACTOR_PARAMETERS_SHA256,
        "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": digest(protocol),
        "training_ledger_sha256": hashes["training_ledger"],
        "dual_evaluation_sha256": hashes["dual_evaluation"],
        "failure_closeout_sha256": hashes["failure_closeout"],
        "actor_metadata_sha256": digest(actor.metadata),
        "source_actor_sha256": actor.metadata["source_actor_sha256"],
    }
    if set(bindings) != BINDING_FIELDS:
        raise AssertionError("Diagnostic designation binding schema differs")
    for name, value in bindings.items():
        _sha(value, name)
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
        },
        "sources": source_closure(),
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink() or path.resolve() != path
            or path.parent.is_symlink() or path.parent.resolve() != path.parent):
        raise ValueError("Diagnostic designation output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_designation(paths: Mapping[str, str | Path], *, output: str | Path) -> dict[str, Any]:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.absolute()
    try:
        relative = output_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError("Diagnostic designation output must stay inside the repository") from None
    checked = validate_components(paths)
    designation = {
        "version": VERSION,
        "status": STATUS,
        "designated": True,
        "release_class": RELEASE_CLASS,
        "designated_step": DESIGNATED_STEP,
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
    sha = file_hash(output_path)
    try:
        read_saved_designation(
            output_path, expected_sha256=sha, components=paths,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return designation


def read_saved_designation(path: str | Path, *, expected_sha256: str,
                            components: Mapping[str, str | Path]) -> dict[str, Any]:
    saved_path, relative = _repo_file(path, "r4.1 diagnostic designation")
    if file_hash(saved_path) != _sha(expected_sha256, "diagnostic designation"):
        raise ValueError("Diagnostic designation bytes differ")
    saved = _strict_json(saved_path, "r4.1 diagnostic designation")
    if (set(saved) != TOP_LEVEL_FIELDS or saved.get("version") != VERSION
            or saved.get("status") != STATUS or saved.get("designated") is not True
            or saved.get("release_class") != RELEASE_CLASS
            or saved.get("designated_step") != DESIGNATED_STEP
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
        raise ValueError("Exact non-formal diagnostic Actor designation required")
    checked = validate_components(components)
    for key in ("bindings", "artifacts", "evidence", "sources"):
        if canonical(saved.get(key)) != canonical(checked[key]):
            raise ValueError("Diagnostic designation " + key + " differs from live evidence")
    return deepcopy(saved)


__all__ = [
    "VERSION", "STATUS", "RELEASE_CLASS", "DESIGNATED_STEP",
    "EXPECTED_ACTOR_SHA256", "EXPECTED_ACTOR_PARAMETERS_SHA256",
    "EXPECTED_PROTOCOL_FILE_SHA256", "EXPECTED_PROTOCOL_CONTENT_SHA256",
    "EXPECTED_LEDGER_SHA256", "EXPECTED_DUAL_EVALUATION_SHA256",
    "EXPECTED_CLOSEOUT_SHA256", "ARTIFACT_NAMES", "BINDING_FIELDS",
    "source_closure", "validate_components", "build_designation",
    "read_saved_designation",
]
