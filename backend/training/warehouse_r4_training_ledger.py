"""Build and reopen the fail-closed warehouse-r4 PPO budget ledger.

The r4 continuation counter is lineage cumulative.  Consequently, summing
``run.json.current_additional_joint_steps`` over branches double counts their
parents.  This module instead authenticates the parent Actor/checkpoint edge
of every attempt and inventories each newly produced 50k segment exactly
once.  Evaluation, extraction, and explanation work is recorded separately
and must explicitly attest to zero PPO/optimizer updates.

This command never imports or restores a training checkpoint and performs no
training.  Checkpoints are authenticated as opaque immutable files.

The default path requires one earliest full-pass Actor and writes nothing when
there is none.  ``--diagnostic-failed-ledger`` is a separate closeout mode: it
may record the authenticated budget with ``selection=null``, but marks the
result ineligible for admission and the production reader rejects it.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r4-training-budget-ledger.v2"
TRAINER_PREFIX = "warehouse-r4-active-trainer."
ACTIVE_VERSION = "warehouse-r4-active-evaluation.v1"
R3_ACTOR_SHA256 = "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b"
R3_CHECKPOINT_SHA256 = "c76b77385e2003896ed6d598d2cd8897d42beebe4be62ac4090943d4204b1ec5"
R3_STATE_SHA256 = "d846d76b6b53dea5498ff23965c2e356ececd05a783ee0b412905e1bd64248f4"
R3_CUMULATIVE_JOINT_STEPS = 3_950_000
MAXIMUM_ADDITIONAL_JOINT_STEPS = 1_000_000
BOUNDARY_STEPS = 50_000
ROOT = Path(__file__).resolve().parents[2]
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COUNTERS = frozenset(("ppo_joint_steps", "optimizer_updates", "neural_updates"))
_MAX_JSON = 64 * 1024 * 1024
_MAX_LOG = 32 * 1024 * 1024


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _fail(message: str) -> None:
    raise ValueError(message)


def _strict_object(path: Path, label: str, *, maximum: int = _MAX_JSON) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > maximum):
        _fail(label + " is missing, linked, noncanonical, or oversized")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid JSON in " + label) from error
    if not isinstance(value, dict):
        _fail(label + " must be a JSON object")
    return value


def _strict_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > _MAX_LOG):
        _fail(label + " is missing, linked, noncanonical, or oversized")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError("Invalid UTF-8 in " + label) from error
    if not lines or any(not line for line in lines):
        _fail(label + " must contain non-empty JSONL rows")
    rows = []
    for index, line in enumerate(lines, 1):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    _fail(f"Duplicate JSON field in {label} row {index}")
                result[key] = value
            return result
        try:
            row = json.loads(line, object_pairs_hook=pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"Non-finite JSON value in {label} row {index}")))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(f"Invalid JSON in {label} row {index}") from error
        if not isinstance(row, dict):
            _fail(f"{label} row {index} must be an object")
        rows.append(row)
    return rows


def _repo_path(value: str | Path, label: str, *, file: bool | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        _fail(label + " path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    if path.resolve() != path or path.is_symlink() or (path != ROOT and ROOT not in path.parents):
        _fail(label + " must be a canonical path inside the repository")
    if file is True and not path.is_file():
        _fail(label + " is missing")
    if file is False and not path.is_dir():
        _fail(label + " directory is missing")
    return path


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        _fail("Invalid " + label + " SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, minimum: float = 0.) -> float:
    if (type(value) not in (int, float) or isinstance(value, bool)
            or not math.isfinite(value) or value < minimum):
        _fail(f"{label} must be finite and >= {minimum}")
    return float(value)


def _regular_bound_file(root: Path, relative: Any, expected_sha: Any, label: str) -> Path:
    if type(relative) is not str or not relative:
        _fail(label + " path is missing")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or pure.as_posix() != relative:
        _fail(label + " path is unsafe")
    path = root.joinpath(*pure.parts).absolute()
    if (root not in path.parents or path.is_symlink() or not path.is_file()
            or path.resolve() != path or file_hash(path) != _sha(expected_sha, label)):
        _fail(label + " is missing, linked, or changed")
    return path


def _discover_attempts(inventory_root: Path) -> tuple[Path, ...]:
    attempts = []
    for child in sorted(inventory_root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        run_path = child / "run.json"
        if not run_path.exists():
            continue
        # A directory named as an r4 active attempt cannot hide behind a
        # malformed or different-version run record.
        try:
            run = _strict_object(run_path, "attempt run record")
        except ValueError:
            if child.name.startswith("r4_active"):
                raise
            continue
        version = run.get("version")
        if type(version) is str and version.startswith(TRAINER_PREFIX):
            attempts.append(child.absolute())
        elif child.name.startswith("r4_active"):
            _fail("An r4_active directory has an unregistered trainer version")
    if not attempts:
        _fail("No warehouse-r4 PPO attempts were discovered")
    return tuple(attempts)


def _validate_source_contract(protocol: Mapping[str, Any], receipt: Mapping[str, Any],
                              run: Mapping[str, Any], scenario_semantic_sha: str) -> None:
    version = run.get("version")
    if (protocol.get("version") != version or receipt.get("version") != version
            or protocol.get("source_actor_sha256") != R3_ACTOR_SHA256
            or receipt.get("source_actor_sha256") != R3_ACTOR_SHA256
            or protocol.get("source_checkpoint_sha256") != R3_CHECKPOINT_SHA256
            or receipt.get("source_checkpoint_sha256") != R3_CHECKPOINT_SHA256
            or protocol.get("source_state_sha256") != R3_STATE_SHA256
            or receipt.get("source_state_sha256") != R3_STATE_SHA256
            or protocol.get("source_cumulative_joint_steps") != R3_CUMULATIVE_JOINT_STEPS
            or receipt.get("source_cumulative_joint_steps") != R3_CUMULATIVE_JOINT_STEPS
            or receipt.get("scenario_manifest_sha256") != scenario_semantic_sha
            or protocol.get("maximum_additional_joint_steps") != MAXIMUM_ADDITIONAL_JOINT_STEPS
            or protocol.get("runtime_action_override", False) is not False
            or receipt.get("runtime_action_override") is not False
            or protocol.get("partners") != {
                "selfplay": .2, "skilled": .2, "assertive": .5, "noisy": .1}
            or protocol.get("shaping") != {
                "productive_progress": .02, "avoidable_wait": -.03,
                "task_distance_regression": -.02}):
        _fail("R4 attempt source, budget, partner mix, shaping, or override contract differs")
    training = protocol.get("training")
    feedback = protocol.get("feedback")
    if (not isinstance(training, dict) or training.get("checkpoint_interval") != BOUNDARY_STEPS
            or training.get("environments") != 16 or training.get("rollout_steps") != 128
            or training.get("epochs") != 4 or training.get("gae_lambda") != .95
            or training.get("gamma") != .99 or training.get("clip") != .2
            or not isinstance(feedback, dict)
            or feedback.get("direction") != "KL(nn||program)"
            # v1 predated the explicit role-scope receipt.  Absence is part
            # of that immutable historical protocol; every later producer
            # records the required robot_2 scope.
            or ("role_scope" in feedback and feedback.get("role_scope") != "robot_2")
            or feedback.get("runtime_action_override") is not False):
        _fail("R4 attempt training or program-feedback contract differs")
    if protocol.get("stage_source") != receipt.get("stage_source"):
        _fail("R4 attempt protocol and source receipt disagree on the parent edge")


def _boundary(attempt: Path, version: str, step: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    directory = attempt / "boundaries" / f"step_{step:07d}"
    summary_path = directory / "summary.json"
    summary = _strict_object(summary_path, "r4 boundary summary")
    if (summary.get("version") != version or summary.get("step") != step
            or summary.get("cumulative_joint_steps") != R3_CUMULATIVE_JOINT_STEPS + step
            or summary.get("evaluation_pending") is not True
            or summary.get("candidate_selected") is not False):
        _fail("R4 boundary identity or pre-selection status differs")
    authority = summary.get("action_authority")
    if (not isinstance(authority, dict) or authority.get("overrides") != 0
            or type(authority.get("trainable")) is not int
            or authority.get("trainable") <= 0
            or authority.get("equal") != authority.get("trainable")):
        _fail("R4 boundary contains an action override or incomplete authority record")
    actor_info, checkpoint_info = summary.get("actor"), summary.get("checkpoint")
    if not isinstance(actor_info, dict) or not isinstance(checkpoint_info, dict):
        _fail("R4 boundary Actor/checkpoint record is missing")
    actor_path = _regular_bound_file(attempt, actor_info.get("path"),
                                     actor_info.get("sha256"), "r4 boundary Actor")
    checkpoint_path = _regular_bound_file(attempt, checkpoint_info.get("path"),
                                          checkpoint_info.get("sha256"),
                                          "r4 boundary checkpoint")
    actor = NumPyNativeActor(actor_path)
    metadata = actor.metadata
    if (actor.artifact_sha256 != actor_info.get("sha256")
            or metadata.get("experiment_version") != version
            or metadata.get("joint_steps") != step
            or metadata.get("cumulative_joint_steps") != R3_CUMULATIVE_JOINT_STEPS + step
            or metadata.get("source_actor_sha256") != R3_ACTOR_SHA256
            or metadata.get("source_checkpoint_sha256") != R3_CHECKPOINT_SHA256
            or metadata.get("actor_parameters_sha256") != actor_info.get("parameters_sha256")
            or metadata.get("protocol_sha256") != digest(protocol)
            or metadata.get("runtime_action_override") is not False
            or metadata.get("action_controller") != "neural_actor_only"):
        _fail("R4 boundary Actor metadata differs from its run and protocol")
    return {
        "step": step, "summary_path": _relative(summary_path),
        "summary_sha256": file_hash(summary_path),
        "actor_path": _relative(actor_path), "actor_sha256": actor.artifact_sha256,
        "actor_parameters_sha256": actor_info["parameters_sha256"],
        "checkpoint_path": _relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "elapsed_seconds": _finite(summary.get("elapsed_seconds"),
                                    "boundary elapsed seconds"),
    }


def _attempt_skeleton(path: Path, scenario_semantic_sha: str) -> dict[str, Any]:
    run_path, protocol_path, receipt_path, log_path = (
        path / "run.json", path / "protocol.json", path / "source_receipt.json",
        path / "training.jsonl")
    run = _strict_object(run_path, "r4 run record")
    protocol = _strict_object(protocol_path, "r4 protocol")
    receipt = _strict_object(receipt_path, "r4 source receipt")
    expected_run_fields = {"version", "status", "pid", "target_additional_joint_steps",
        "current_additional_joint_steps", "cumulative_joint_steps", "started_unix",
        "runtime_action_override", "elapsed_seconds"}
    version = run.get("version")
    if (set(run) != expected_run_fields or type(version) is not str
            or not version.startswith(TRAINER_PREFIX)
            or run.get("status") != "training_target_completed"
            or run.get("runtime_action_override") is not False):
        _fail("Every supplied r4 PPO attempt must have a complete terminal run record")
    end = _integer(run.get("current_additional_joint_steps"), "attempt end", minimum=1)
    if (run.get("target_additional_joint_steps") != end or end % BOUNDARY_STEPS
            or run.get("cumulative_joint_steps") != R3_CUMULATIVE_JOINT_STEPS + end
            or end > MAXIMUM_ADDITIONAL_JOINT_STEPS):
        _fail("R4 attempt terminal training clock differs")
    started = _finite(run.get("started_unix"), "attempt start time")
    elapsed = _finite(run.get("elapsed_seconds"), "attempt elapsed time")
    _validate_source_contract(protocol, receipt, run, scenario_semantic_sha)
    stage = receipt.get("stage_source")
    start = 0 if stage is None else _integer(
        stage.get("additional_ppo_joint_steps") if isinstance(stage, dict) else None,
        "parent lineage step")
    if start >= end or start % BOUNDARY_STEPS:
        _fail("R4 attempt parent counter must precede its terminal counter")
    boundaries = [_boundary(path, version, step, protocol)
                  for step in range(start + BOUNDARY_STEPS, end + 1, BOUNDARY_STEPS)]
    actual_summary_names = sorted(item.parent.name for item in path.glob("boundaries/*/summary.json"))
    expected_summary_names = [f"step_{step:07d}"
                              for step in range(start + BOUNDARY_STEPS, end + 1,
                                                BOUNDARY_STEPS)]
    if actual_summary_names != expected_summary_names:
        _fail("R4 attempt boundary set is incomplete or contains an unregistered boundary")
    rows = _strict_jsonl(log_path, "r4 training log")
    previous = start
    maximum_increment = int(protocol["training"]["environments"]) * int(
        protocol["training"]["rollout_steps"])
    for row in rows:
        step = _integer(row.get("joint_steps"), "training-log joint step", minimum=1)
        cumulative = _integer(row.get("cumulative_joint_steps"),
                              "training-log cumulative joint step", minimum=1)
        time_steps = _integer(row.get("time_steps"), "training-log time steps", minimum=1)
        if (step <= previous or step > end or step - previous > maximum_increment
                or cumulative != R3_CUMULATIVE_JOINT_STEPS + step
                or time_steps > int(protocol["training"]["rollout_steps"])):
            _fail("R4 training log counter continuity differs")
        previous = step
    if previous != end:
        _fail("R4 training log does not reach the terminal run counter")
    return {
        "id": path.name, "path": _relative(path), "version": version,
        "run_path": _relative(run_path), "run_sha256": file_hash(run_path),
        "protocol_path": _relative(protocol_path),
        "protocol_sha256": file_hash(protocol_path),
        "protocol_semantic_sha256": digest(protocol),
        "source_receipt_path": _relative(receipt_path),
        "source_receipt_sha256": file_hash(receipt_path),
        "training_log_path": _relative(log_path),
        "training_log_sha256": file_hash(log_path),
        "training_log_rows": len(rows), "started_unix": started,
        "elapsed_seconds": elapsed, "lineage_start_joint_steps": start,
        "lineage_end_joint_steps": end,
        "fresh_ppo_joint_steps": end - start,
        "stage_source": stage, "boundaries": boundaries,
    }


def _bind_dag(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boundary_by_pair: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for attempt in attempts:
        for boundary in attempt["boundaries"]:
            key = (boundary["actor_sha256"], boundary["checkpoint_sha256"])
            if key in boundary_by_pair:
                _fail("Two r4 boundaries claim the same Actor/checkpoint output")
            boundary_by_pair[key] = (attempt["id"], boundary)
    ids = {attempt["id"] for attempt in attempts}
    for attempt in attempts:
        stage = attempt.pop("stage_source")
        if stage is None:
            attempt["parent"] = {
                "kind": "r3", "attempt_id": None, "lineage_step": 0,
                "actor_sha256": R3_ACTOR_SHA256,
                "checkpoint_sha256": R3_CHECKPOINT_SHA256,
            }
            continue
        required_true = ("actor_and_optimizer_preserved", "pre_update_actor_parity",
            "pre_update_optimizer_parity", "rng_restored_before_curriculum_resets")
        if (not isinstance(stage, dict)
                or any(stage.get(key) is not True for key in required_true)
                or stage.get("inflight_episodes_restarted_for_versioned_curriculum") is not True
                or stage.get("runtime_action_override") is not False):
            _fail("R4 continuation parent parity record is incomplete")
        for key in ("program_content_sha256", "parent_actor_parameters_sha256",
                    "parent_optimizer_sha256", "parent_rng_sha256",
                    "parent_owned_rng_sha256"):
            _sha(stage.get(key), "continuation " + key)
        _finite(stage.get("feedback_lambda"), "continuation feedback lambda")
        rebound = stage.get("learning_rates_rebound_after_parity")
        if (rebound is not None and (
                not isinstance(rebound, dict) or set(rebound) != {"actor", "critic"}
                or any(_finite(value, "rebound learning rate", minimum=0.) <= 0
                       for value in rebound.values()))
                or "optimizer_moments_and_steps_preserved" in stage
                and stage["optimizer_moments_and_steps_preserved"] is not True):
            _fail("R4 continuation optimizer-rebind receipt differs")
        pair = (_sha(stage.get("actor_sha256"), "parent Actor"),
                _sha(stage.get("checkpoint_sha256"), "parent checkpoint"))
        parent = boundary_by_pair.get(pair)
        if parent is None:
            _fail("R4 continuation parent Actor/checkpoint is omitted from the attempt inventory")
        parent_id, boundary = parent
        if (parent_id == attempt["id"] or stage.get("trainer_version")
                != next(row for row in attempts if row["id"] == parent_id)["version"]
                or stage.get("additional_ppo_joint_steps") != boundary["step"]
                or boundary["step"] != attempt["lineage_start_joint_steps"]
                or stage.get("parent_actor_parameters_sha256")
                    != boundary["actor_parameters_sha256"]):
            _fail("R4 continuation parent DAG edge differs from the authenticated boundary")
        attempt["parent"] = {
            "kind": "attempt", "attempt_id": parent_id,
            "lineage_step": boundary["step"], "actor_sha256": pair[0],
            "checkpoint_sha256": pair[1],
        }
    # Attempt-level edges must form an acyclic graph.
    by_id = {row["id"]: row for row in attempts}
    for identifier in ids:
        seen = set()
        current = identifier
        while current is not None:
            if current in seen:
                _fail("R4 attempt parent graph contains a cycle")
            seen.add(current)
            current = by_id[current]["parent"]["attempt_id"]
    return attempts


def _segments(attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    segments, identities = [], set()
    for attempt in attempts:
        source_actor = attempt["parent"]["actor_sha256"]
        source_checkpoint = attempt["parent"]["checkpoint_sha256"]
        previous = attempt["lineage_start_joint_steps"]
        for boundary in attempt["boundaries"]:
            identity = digest({
                "attempt_id": attempt["id"], "start": previous,
                "end": boundary["step"], "source_actor_sha256": source_actor,
                "source_checkpoint_sha256": source_checkpoint,
                "actor_sha256": boundary["actor_sha256"],
                "checkpoint_sha256": boundary["checkpoint_sha256"],
            })
            if identity in identities:
                _fail("R4 PPO segment is duplicated")
            identities.add(identity)
            segments.append({
                "id": identity, "attempt_id": attempt["id"], "start": previous,
                "end": boundary["step"], "fresh_ppo_joint_steps": boundary["step"] - previous,
                "source_actor_sha256": source_actor,
                "source_checkpoint_sha256": source_checkpoint,
                "actor_sha256": boundary["actor_sha256"],
                "checkpoint_sha256": boundary["checkpoint_sha256"],
            })
            previous = boundary["step"]
            source_actor, source_checkpoint = (boundary["actor_sha256"],
                                                boundary["checkpoint_sha256"])
    if any(row["fresh_ppo_joint_steps"] != BOUNDARY_STEPS for row in segments):
        _fail("Each authenticated r4 PPO segment must be exactly 50k fresh steps")
    return sorted(segments, key=lambda row: (row["attempt_id"], row["end"]))


def _full_audits(attempt_paths: Sequence[Path], active_report: Path,
                 attempts: Sequence[Mapping[str, Any]], scenario_file_sha: str,
                 scenario_semantic_sha: str, *, allow_no_selection: bool = False
                 ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates = {boundary["actor_sha256"]: (attempt, boundary)
                  for attempt in attempts for boundary in attempt["boundaries"]}
    paths = {active_report}
    for attempt in attempt_paths:
        # Full audits have historically lived under both ``evaluations`` and
        # ``full_paired_audit_300``.  Discover by producer filename throughout
        # every authenticated attempt so an earlier pass cannot be omitted by
        # choosing a different directory convention.
        paths.update(path.absolute() for path in attempt.rglob("paired_report.json"))
    audits_by_sha = {}
    for path in sorted(paths):
        report = _strict_object(path, "r4 paired full audit")
        if report.get("version") != ACTIVE_VERSION:
            _fail("A discovered r4 paired audit has another producer version")
        status, selected = report.get("status"), report.get("selected")
        if status not in ("passed", "failed") or selected is not (status == "passed"):
            _fail("R4 paired audit status and selected flag disagree")
        scenario = report.get("scenario_manifest", {})
        if report.get("baseline", {}).get("actor", {}).get("artifact_sha256") \
                != R3_ACTOR_SHA256:
            _fail("R4 paired audit uses another baseline Actor")
        # Some historical failures used a byte-different serialization of the
        # same registered scenario object.  Authenticate that original file
        # and its semantics; the finally selected report must use the exact
        # final release file below.
        report_scenario_path = _repo_path(scenario.get("path"),
                                          "paired-audit scenario manifest", file=True)
        if (file_hash(report_scenario_path) != scenario.get("sha256")
                or digest(_strict_object(report_scenario_path,
                                         "paired-audit scenario manifest"))
                    != scenario_semantic_sha):
            _fail("R4 paired audit uses another scenario manifest")
        actor_sha = report.get("candidate", {}).get("actor", {}).get("artifact_sha256")
        if actor_sha not in candidates:
            _fail("R4 paired audit candidate is not an authenticated attempt boundary")
        attempt, boundary = candidates[actor_sha]
        if status == "passed" and (
                not isinstance(report.get("absolute_checks"), dict)
                or not report["absolute_checks"]
                or any(value is not True for value in report["absolute_checks"].values())
                or _integer(report.get("relative_checks_passed"),
                            "paired-audit relative checks") < 4):
            _fail("A passing r4 paired audit does not contain all frozen pass checks")
        record = {
            "path": _relative(path), "sha256": file_hash(path), "status": status,
            "actor_sha256": actor_sha, "attempt_id": attempt["id"],
            "lineage_step": boundary["step"],
            "trained_unix": attempt["started_unix"] + boundary["elapsed_seconds"],
        }
        prior = audits_by_sha.get(record["sha256"])
        if prior is not None and (prior["actor_sha256"], prior["status"]) != (
                record["actor_sha256"], record["status"]):
            _fail("Duplicate r4 paired-audit bytes have inconsistent identities")
        audits_by_sha[record["sha256"]] = record
    active_sha = file_hash(active_report)
    active = audits_by_sha[active_sha]
    passed = [record for record in audits_by_sha.values() if record["status"] == "passed"]
    active_object = _strict_object(active_report, "selected r4 paired full audit")
    if not passed:
        if active["status"] != "failed" or not allow_no_selection:
            _fail("The selected r4 Actor lacks a passing full paired audit")
        return sorted(audits_by_sha.values(), key=lambda row: (
            row["trained_unix"], row["path"])), None
    if (active["status"] != "passed"
            or active_object.get("scenario_manifest", {}).get("sha256")
                != scenario_file_sha):
        _fail("The selected r4 Actor lacks a passing full paired audit")
    earliest = min(passed, key=lambda row: (
        row["trained_unix"], row["attempt_id"], row["lineage_step"], row["actor_sha256"]))
    if active["actor_sha256"] != earliest["actor_sha256"]:
        _fail("The selected r4 Actor is not the earliest fully passing audited boundary")
    return sorted(audits_by_sha.values(), key=lambda row: (
        row["trained_unix"], row["path"])), active


def _counter_values(value: Any, *, path: str = "$") -> list[tuple[str, int]]:
    result = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = f"{path}.{key}"
            if key in _COUNTERS:
                result.append((next_path, _integer(item, next_path)))
            result.extend(_counter_values(item, path=next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_counter_values(item, path=f"{path}[{index}]"))
    return result


def _zero_step_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    if not paths:
        _fail("At least one explicit zero-step extraction/diagnostic record is required")
    if len(paths) != len(set(paths)):
        _fail("Zero-step evidence paths are duplicated")
    result = []
    for path in sorted(paths):
        report = _strict_object(path, "zero-step work record")
        counters = _counter_values(report)
        if not counters or any(value != 0 for _, value in counters):
            _fail("Zero-step work must explicitly report zero PPO/optimizer/neural updates")
        version = report.get("version")
        status = report.get("status")
        if type(version) is not str or not version or type(status) is not str or not status:
            _fail("Zero-step work record lacks version or status")
        result.append({
            "path": _relative(path), "sha256": file_hash(path), "version": version,
            "status": status, "counters": {name: value for name, value in counters},
        })
    return result


def _producer_sources() -> dict[str, str]:
    paths = (Path(__file__).absolute(), ROOT / "env/warehouse_native/policy.py")
    return {_relative(path): file_hash(path) for path in paths}


def _assemble(*, inventory_root: Path, attempt_paths: Sequence[Path],
              r3_actor: Path, r3_checkpoint: Path, scenario_manifest: Path,
              active_policy_report: Path,
              zero_step_records: Sequence[Path],
              diagnostic_failed_ledger: bool = False) -> dict[str, Any]:
    if (file_hash(r3_actor) != R3_ACTOR_SHA256
            or file_hash(r3_checkpoint) != R3_CHECKPOINT_SHA256):
        _fail("The supplied r3 Actor/checkpoint is not the frozen r4 parent")
    NumPyNativeActor(r3_actor)
    scenarios = _strict_object(scenario_manifest, "registered scenario manifest")
    scenario_semantic_sha = digest(scenarios)
    discovered = _discover_attempts(inventory_root)
    supplied = tuple(sorted(attempt_paths))
    if supplied != tuple(sorted(set(supplied))) or supplied != discovered:
        _fail("The explicit r4 attempt list does not exactly match discovered run records")
    attempts = [_attempt_skeleton(path, scenario_semantic_sha) for path in supplied]
    if len({row["id"] for row in attempts}) != len(attempts):
        _fail("R4 attempt directory identifiers are duplicated")
    attempts = _bind_dag(attempts)
    segments = _segments(attempts)
    total = sum(row["fresh_ppo_joint_steps"] for row in segments)
    if (total != sum(row["fresh_ppo_joint_steps"] for row in attempts)
            or total <= 0 or total > MAXIMUM_ADDITIONAL_JOINT_STEPS):
        _fail("Authenticated unique r4 PPO steps exceed the approved budget or differ")
    audits, selected = _full_audits(
        supplied, active_policy_report, attempts, file_hash(scenario_manifest),
        scenario_semantic_sha, allow_no_selection=diagnostic_failed_ledger,
    )
    if diagnostic_failed_ledger and selected is not None:
        _fail("Diagnostic failed-ledger mode is only valid when no Actor passed")
    selected_ancestors = set()
    by_id = {row["id"]: row for row in attempts}
    if selected is not None:
        current = by_id[selected["attempt_id"]]["parent"]["attempt_id"]
        while current is not None:
            selected_ancestors.add(current)
            current = by_id[current]["parent"]["attempt_id"]
    for row in attempts:
        row["terminal_status"] = (
            "selected" if selected is not None and row["id"] == selected["attempt_id"]
            else "superseded" if row["id"] in selected_ancestors else "rejected"
        )
        row["selected"] = row["terminal_status"] == "selected"
    selected_boundary = None if selected is None else next(
        boundary for boundary in by_id[selected["attempt_id"]]["boundaries"]
        if boundary["actor_sha256"] == selected["actor_sha256"]
    )
    zero = _zero_step_records(zero_step_records)
    builder_inputs = {
        "inventory_root": _relative(inventory_root),
        "attempts": [_relative(path) for path in supplied],
        "r3_actor": _relative(r3_actor), "r3_checkpoint": _relative(r3_checkpoint),
        "scenario_manifest": _relative(scenario_manifest),
        "active_policy_report": _relative(active_policy_report),
        "zero_step_records": [_relative(path) for path in sorted(zero_step_records)],
        "diagnostic_failed_ledger": diagnostic_failed_ledger,
    }
    selection = None if selected is None else {
        "attempt_id": selected["attempt_id"],
        "lineage_step": selected["lineage_step"],
        "training_actor_path": selected_boundary["actor_path"],
        "training_actor_sha256": selected["actor_sha256"],
        "checkpoint_path": selected_boundary["checkpoint_path"],
        "checkpoint_sha256": selected_boundary["checkpoint_sha256"],
        "active_policy_report_path": _relative(active_policy_report),
        "active_policy_report_sha256": file_hash(active_policy_report),
        "earliest_full_pass": True,
    }
    admission_eligible = selection is not None
    return {
        "version": VERSION,
        "status": "complete" if admission_eligible else "failed_no_eligible_actor",
        "test_fixture": False, "formal_ready": False,
        "admission_eligible": admission_eligible,
        "failure_reason": None if admission_eligible else "no_passing_full_paired_audit",
        "source_r3": {
            "actor_path": _relative(r3_actor), "actor_sha256": R3_ACTOR_SHA256,
            "checkpoint_path": _relative(r3_checkpoint),
            "checkpoint_sha256": R3_CHECKPOINT_SHA256,
            "state_sha256": R3_STATE_SHA256,
            "cumulative_joint_steps": R3_CUMULATIVE_JOINT_STEPS,
        },
        "scenario_manifest": {
            "path": _relative(scenario_manifest), "sha256": file_hash(scenario_manifest),
            "semantic_sha256": scenario_semantic_sha,
        },
        "inventory": {
            "root": _relative(inventory_root),
            "discovered_attempts": [_relative(path) for path in discovered],
            "supplied_attempts": [_relative(path) for path in supplied],
            "exact": True,
        },
        "maximum_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS,
        "total_actual_additional_joint_steps": total,
        "remaining_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS - total,
        "attempts": attempts, "segments": segments,
        "unique_segment_count": len(segments),
        "full_paired_audits": audits,
        "selection": selection,
        "zero_step_work": zero,
        "invariants": {
            "all_attempts_terminal": True, "parent_dag_valid": True,
            "segment_hashes_unique": True, "fresh_steps_counted_once": True,
            "checkpoints_treated_as_opaque": True,
            "training_checkpoints_unsafely_loaded": False,
            "exactly_one_selected_attempt": admission_eligible,
            "selected_actor_is_earliest_full_pass": admission_eligible,
            "ppo_budget_within_cap": True,
        },
        "builder_inputs": builder_inputs,
        "producer_sources": _producer_sources(),
    }


def _paths_from_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"inventory_root", "attempts", "r3_actor", "r3_checkpoint",
        "scenario_manifest", "active_policy_report", "zero_step_records",
        "diagnostic_failed_ledger"}
    if not isinstance(inputs, dict) or set(inputs) != expected:
        _fail("Training-ledger builder input schema differs")
    if (not isinstance(inputs["attempts"], list)
            or not isinstance(inputs["zero_step_records"], list)
            or type(inputs["diagnostic_failed_ledger"]) is not bool):
        _fail("Training-ledger input path lists differ")
    return {
        "inventory_root": _repo_path(inputs["inventory_root"], "inventory root", file=False),
        "attempt_paths": tuple(_repo_path(value, "attempt directory", file=False)
                               for value in inputs["attempts"]),
        "r3_actor": _repo_path(inputs["r3_actor"], "r3 Actor", file=True),
        "r3_checkpoint": _repo_path(inputs["r3_checkpoint"], "r3 checkpoint", file=True),
        "scenario_manifest": _repo_path(inputs["scenario_manifest"], "scenario manifest", file=True),
        "active_policy_report": _repo_path(inputs["active_policy_report"], "active report", file=True),
        "zero_step_records": tuple(_repo_path(value, "zero-step record", file=True)
                                   for value in inputs["zero_step_records"]),
        "diagnostic_failed_ledger": inputs["diagnostic_failed_ledger"],
    }


def read_saved_ledger(path: str | Path, *, expected_report_sha256: str,
                      expected_selected_actor_sha256: str | None,
                      expected_active_report_sha256: str,
                      expected_scenario_manifest_sha256: str,
                      require_admission_eligible: bool = True) -> dict[str, Any]:
    report_path = _repo_path(path, "training budget ledger", file=True)
    if file_hash(report_path) != _sha(expected_report_sha256, "training-ledger report"):
        _fail("Training budget ledger bytes changed")
    report = _strict_object(report_path, "training budget ledger")
    if report.get("version") != VERSION:
        _fail("Training budget ledger producer version differs")
    sources = _producer_sources()
    if report.get("producer_sources") != sources:
        _fail("Training budget ledger producer sources changed")
    before = {key: file_hash(ROOT / key) for key in sources}
    rebuilt = _assemble(**_paths_from_inputs(report.get("builder_inputs")))
    after = {key: file_hash(ROOT / key) for key in sources}
    selection = report.get("selection")
    actual_actor = (selection.get("training_actor_sha256")
                    if isinstance(selection, dict) else None)
    actual_active_report = (selection.get("active_policy_report_sha256")
                            if isinstance(selection, dict)
                            else file_hash(_paths_from_inputs(
                                report.get("builder_inputs"))["active_policy_report"]))
    if (before != after or sources != _producer_sources()
            or digest(rebuilt) != digest(report)
            or actual_actor != expected_selected_actor_sha256
            or actual_active_report != expected_active_report_sha256
            or report.get("scenario_manifest", {}).get("sha256")
                != expected_scenario_manifest_sha256
            or require_admission_eligible and (
                report.get("status") != "complete"
                or report.get("admission_eligible") is not True
                or not isinstance(selection, dict))):
        _fail("Training budget ledger semantic replay or release binding differs")
    return report


def build(*, inventory_root: str | Path, attempts: Sequence[str | Path],
          r3_actor: str | Path, r3_checkpoint: str | Path,
          scenario_manifest: str | Path, active_policy_report: str | Path,
          zero_step_records: Sequence[str | Path], output: str | Path,
          diagnostic_failed_ledger: bool = False) -> dict[str, Any]:
    values = {
        "inventory_root": _repo_path(inventory_root, "inventory root", file=False),
        "attempt_paths": tuple(_repo_path(path, "attempt directory", file=False)
                               for path in attempts),
        "r3_actor": _repo_path(r3_actor, "r3 Actor", file=True),
        "r3_checkpoint": _repo_path(r3_checkpoint, "r3 checkpoint", file=True),
        "scenario_manifest": _repo_path(scenario_manifest, "scenario manifest", file=True),
        "active_policy_report": _repo_path(active_policy_report, "active report", file=True),
        "zero_step_records": tuple(_repo_path(path, "zero-step record", file=True)
                                   for path in zero_step_records),
        "diagnostic_failed_ledger": diagnostic_failed_ledger,
    }
    destination = _repo_path(output, "training-ledger output")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.resolve() != destination.parent.absolute():
        _fail("Training-ledger output parent is unsafe")
    report = _assemble(**values)
    raw = (canonical(report) + "\n").encode("utf-8")
    descriptor = os.open(destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        selection = report.get("selection")
        read_saved_ledger(
            destination, expected_report_sha256=file_hash(destination),
            expected_selected_actor_sha256=(selection["training_actor_sha256"]
                if isinstance(selection, dict) else None),
            expected_active_report_sha256=file_hash(values["active_policy_report"]),
            expected_scenario_manifest_sha256=report["scenario_manifest"]["sha256"],
            require_admission_eligible=not diagnostic_failed_ledger,
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-root", required=True)
    parser.add_argument("--attempt", action="append", required=True)
    parser.add_argument("--r3-actor", required=True)
    parser.add_argument("--r3-checkpoint", required=True)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--active-policy-report", required=True)
    parser.add_argument("--zero-step-record", action="append", required=True)
    parser.add_argument("--diagnostic-failed-ledger", action="store_true",
        help=("write a semantically replayable failed ledger with selection=null "
              "when no Actor passed; this output is never admission eligible"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(inventory_root=args.inventory_root, attempts=args.attempt,
        r3_actor=args.r3_actor, r3_checkpoint=args.r3_checkpoint,
        scenario_manifest=args.scenario_manifest,
        active_policy_report=args.active_policy_report,
        zero_step_records=args.zero_step_record, output=args.output,
        diagnostic_failed_ledger=args.diagnostic_failed_ledger)
    selection = report.get("selection")
    print(canonical({
        "status": report["status"],
        "total_actual_additional_joint_steps": report["total_actual_additional_joint_steps"],
        "remaining_additional_joint_steps": report["remaining_additional_joint_steps"],
        "attempts": len(report["attempts"]), "segments": len(report["segments"]),
        "admission_eligible": report["admission_eligible"],
        "selected_training_actor_sha256": (
            selection["training_actor_sha256"] if isinstance(selection, dict) else None),
        "output": str(_repo_path(args.output, "training-ledger output", file=True)),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "build", "read_saved_ledger"]
