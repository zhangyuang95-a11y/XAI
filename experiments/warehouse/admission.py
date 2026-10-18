"""Warehouse v6 evidence pipeline; no policy action logic lives here.

Each phase writes an immutable receipt.  The sealed audit and heldout claim
files are created *before* their first rollout, so a crash cannot silently
reuse those seeds.  Only a passing acceptance receipt may be packaged into a
release.  This module never installs a policy or opens the pilot.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence
from uuid import uuid4

from . import protocol as p

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAIM_ROOT = PROJECT_ROOT / "output" / "warehouse_v6_sealed_claims"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_json_bytes(value)).hexdigest()


def _file_sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _exclusive_json(path: Path, value: object) -> str:
    """Atomic create without ever changing an existing scientific receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    raw = _json_bytes(value)
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if another process already wrote it.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(raw).hexdigest()


def _immutable_json(path: Path, value: object) -> str:
    """Create a locked input once, or verify its original bytes on retry."""
    expected = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"Warehouse locked input changed: {path}")
        return sha256(expected).hexdigest()
    return _exclusive_json(path, value)


def _read(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"Warehouse receipt is not an object: {path}")
    return value


def _summary(episodes: Sequence[Mapping]) -> dict:
    if not episodes:
        raise ValueError("Warehouse evaluation cannot summarize zero episodes")
    n = len(episodes)
    return {
        "mean_deliveries": sum(float(row["deliveries"]) for row in episodes) / n,
        "mean_collisions": sum(float(row["collisions"]) for row in episodes) / n,
        "shutdown_episode_rate": sum(bool(row["shutdown_episode"]) for row in episodes) / n,
    }


def _quality_trace_valid(value: object, scenario_seed: int, action_seed: int,
                         actor_sha256: str | None = None) -> bool:
    """Check compact trace shape before the independent physics replay gate."""
    try:
        if not isinstance(value, Mapping):
            return False
        if (value["schema"] != "warehouse-quality-trace.v1"
                or value["scenario_seed"] != scenario_seed
                or value["action_seed"] != action_seed
                or type(value["episode_id"]) is not int
                or type(value["frame_count"]) is not int
                or not 1 <= value["frame_count"] <= 120
                or value["action_encoding"] != "base64-u8-a1-times-5-plus-a2"
                or (actor_sha256 is not None and
                    value["actor_sha256"] != actor_sha256)):
            return False
        for key in ("actor_sha256", "initial_pre_state_sha256",
                    "final_post_state_sha256", "chain_sha256"):
            digest = value[key]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                return False
        roles = value["actor_role_sha256"]
        if (not isinstance(roles, Mapping) or set(roles) != set(p.ROLES)
                or any(not isinstance(roles[role], str)
                       or len(roles[role]) != 64
                       or any(char not in "0123456789abcdef"
                              for char in roles[role]) for role in p.ROLES)):
            return False
        encoded = value["actions_b64"]
        if not isinstance(encoded, str):
            return False
        actions = base64.b64decode(encoded, validate=True)
        return (len(actions) == value["frame_count"]
                and all(action < 25 for action in actions)
                and base64.b64encode(actions).decode("ascii") == encoded)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return False


def _validated_episodes(result: Mapping, seeds: Sequence[int], *,
                        require_trace: bool = False,
                        actor_sha256: str | None = None,
                        action_seed_offset: int = 0) -> list[dict]:
    episodes = result.get("episodes")
    if (not isinstance(episodes, list) or len(episodes) != len(seeds)
            or [row.get("seed") for row in episodes] != list(seeds)):
        raise RuntimeError("Warehouse evaluator returned wrong scenario seeds or count")
    normalized = []
    for row in episodes:
        deliveries, collisions = row.get("deliveries"), row.get("collisions")
        if (type(deliveries) is not int or deliveries < 0
                or type(collisions) is not int or collisions < 0
                or type(row.get("shutdown_episode")) is not bool):
            raise RuntimeError("Warehouse evaluator returned invalid episode outcome")
        normalized_row = {"seed": row["seed"], "deliveries": deliveries,
                          "collisions": collisions,
                          "shutdown_episode": row["shutdown_episode"]}
        if require_trace or "quality_trace" in row:
            if not _quality_trace_valid(
                    row.get("quality_trace"), row["seed"],
                    row["seed"] + action_seed_offset, actor_sha256):
                raise RuntimeError("Warehouse evaluator quality trace is invalid")
            normalized_row["quality_trace"] = dict(row["quality_trace"])
        normalized.append(normalized_row)
    measured = _summary(normalized)
    supplied = result.get("summary", {})
    if any(not math.isfinite(float(supplied.get(key, float("nan"))))
           or abs(float(supplied[key]) - value) > 1e-9
           for key, value in measured.items()):
        raise RuntimeError("Warehouse evaluator summary differs from raw episodes")
    return normalized


def _verify_stress(stress: Mapping) -> bool:
    """Recompute every pressure outcome against the preregistered schedule."""
    try:
        expected = p.stress_schedule()
        episodes = stress["episodes"]
        if len(episodes) != len(expected) or stress["schedule_sha256"] != _sha(expected):
            return False
        for row, (task, seed, action_seed) in zip(episodes, expected, strict=True):
            if (row["task"], row["scenario_seed"], row["action_seed"]) != \
                    (task, seed, action_seed):
                return False
            if not _quality_trace_valid(row.get("quality_trace"), seed,
                                        action_seed, stress["actor_sha256"]):
                return False
            if (type(row["deliveries"]) is not int or row["deliveries"] < 0
                    or type(row["collisions"]) is not int or row["collisions"] < 0
                    or type(row["shutdown_episode"]) is not bool):
                return False
        summary = _summary(episodes)
        return all(math.isclose(summary[key], float(stress["summary"][key]),
                                rel_tol=0, abs_tol=1e-9) for key in summary)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _verify_heldout(outcomes: Mapping, release_id: str) -> bool:
    """Check all 600 raw rows before trusting pooled quality summaries."""
    try:
        if set(outcomes) != set(p.VARIANTS):
            return False
        schedule = p.heldout_schedule(release_id)
        for variant in p.VARIANTS:
            row = outcomes[variant]
            if set(row["tasks"]) != set(map(str, p.TASKS)):
                return False
            all_episodes = []
            for task in p.TASKS:
                task_row = row["tasks"][str(task)]
                episodes = task_row["episodes"]
                if (len(episodes) != 100
                        or [item["seed"] for item in episodes] != list(schedule[task])
                        or task_row["episodes_sha256"] != _sha(episodes)):
                    return False
                if any(type(item["deliveries"]) is not int or item["deliveries"] < 0
                       or type(item["collisions"]) is not int or item["collisions"] < 0
                       or type(item["shutdown_episode"]) is not bool
                       for item in episodes):
                    return False
                if any(not _quality_trace_valid(item.get("quality_trace"), item["seed"],
                                                item["seed"], row["actor_sha256"])
                       for item in episodes):
                    return False
                recomputed = _summary(episodes)
                if any(not math.isclose(recomputed[key], float(task_row["summary"][key]),
                                        rel_tol=0, abs_tol=1e-9) for key in recomputed):
                    return False
                all_episodes.extend(episodes)
            pooled = _summary(all_episodes)
            if row["episode_count"] != 300 or any(
                    not math.isclose(pooled[key], float(row["summary"][key]),
                                     rel_tol=0, abs_tol=1e-9) for key in pooled):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _verify_formal_row(row: Mapping, pair: Mapping,
                       source_closure_sha256: str) -> bool:
    try:
        if (row["pair_seed"] != pair["pair_seed"]
                or row["budget"] != pair["joint_steps"]
                or row["pair_report_sha256"] != _sha(pair)
                or row["source_closure_sha256"] != source_closure_sha256
                or row["selection_seeds"] != list(p.FORMAL_SELECTION_SEEDS)
                or set(row["variants"]) != set(p.VARIANTS)):
            return False
        for variant in p.VARIANTS:
            entry = row["variants"][variant]
            trained = pair["variants"][variant]
            if (entry["actor_sha256"] != trained["actor_artifact"]["sha256"]
                    or entry["training_report_sha256"] !=
                    trained["training_report_sha256"]):
                return False
            _validated_episodes({"episodes": entry["episodes"],
                                 "summary": entry["quality"]},
                                p.FORMAL_SELECTION_SEEDS, require_trace=True,
                                actor_sha256=entry["actor_sha256"])
        return row["passed"] is all(
            p.quality_pass(row["variants"][variant]["quality"], development=False)
            for variant in p.VARIANTS)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _audit_valid(audits: Mapping, actor, programs: Mapping,
                 seeds: Sequence[int], *, development: bool) -> bool:
    if not isinstance(audits, Mapping) or set(audits) != set(p.ROLES):
        return False
    for role in p.ROLES:
        audit = audits[role]
        trace_path = audit.get("trace_path") if isinstance(audit, Mapping) else None
        if (not isinstance(trace_path, str) or not trace_path.endswith(".jsonl")
                or type(audit.get("trace_row_count")) is not int
                or audit["trace_row_count"] != audit.get("sample_count")
                or not Path(trace_path).is_file()
                or audit.get("trace_sha256") != _file_sha(trace_path)):
            return False
        raw_trace = Path(trace_path).read_bytes()
        if (raw_trace.count(b"\n") != audit["trace_row_count"]
                or not raw_trace.endswith(b"\n")):
            return False
        if (not isinstance(audit, Mapping)
                or audit.get("role") != role
                or audit.get("actor_role_sha256") != actor.role_sha256[role]
                or audit.get("program_sha256") != programs[role].sha256()
                or audit.get("episode_ids") != sorted(map(str, seeds))
                or audit.get("episode_count") != len(seeds)
                or type(audit.get("sample_count")) is not int
                or audit["sample_count"] < 1000
                or type(audit.get("max_depth")) is not int
                or not 0 <= audit["max_depth"] <= 5
                or type(audit.get("leaf_count")) is not int
                or not 1 <= audit["leaf_count"] <= 16
                or type(audit.get("max_features_per_node")) is not int
                or not 1 <= audit["max_features_per_node"] <= 3):
            return False
        if (audit.get("feedback_eligible") is not True
                or ((development or role == "robot_2") and
                    audit.get("explanation_eligible") is not True)):
            return False
        purposes = (("feedback", "explanation") if development or role == "robot_2"
                    else ("feedback",))
        for purpose in purposes:
            metrics = audit.get(purpose, {})
            if not isinstance(metrics, Mapping):
                return False
            sample_count = audit["sample_count"]
            rows = metrics.get("rows")
            actionable_rows = metrics.get("actionable_rows")
            certified = metrics.get("certified_rows")
            try:
                coverage = float(metrics["coverage"])
                actionable_coverage = float(metrics["actionable_coverage"])
            except (KeyError, TypeError, ValueError):
                return False
            if (metrics.get("sample_count") != sample_count
                    or type(rows) is not int or not 0 <= rows <= sample_count
                    or type(actionable_rows) is not int
                    or not 0 <= actionable_rows <= rows
                    or not math.isfinite(coverage)
                    or not math.isfinite(actionable_coverage)
                    or not math.isclose(coverage, rows / sample_count,
                                        rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(actionable_coverage,
                                        actionable_rows / sample_count,
                                        rel_tol=0, abs_tol=1e-9)
                    or (certified is not None and
                        (type(certified) is not int or not 0 <= certified <= rows))
                    or not p.program_pass(metrics, purpose=purpose,
                                          development=development)):
                return False
    return True


class TrainingGateFailure(RuntimeError):
    """A completed paired run exists, but its preregistered training gate failed."""

    def __init__(self, report: Mapping):
        self.report = report
        super().__init__("Warehouse paired training provenance gate failed")


class BCPrecheckFailure(RuntimeError):
    """A locked development pair ended at the recorded final BC gate."""

    def __init__(self, evidence: Mapping):
        self.evidence = dict(evidence)
        super().__init__("Warehouse final BC internal precheck failed before PPO")


class NativeBackend:
    """Late imports keep pure receipt tests free of Torch and environment setup."""

    def _failed_bc_precheck_evidence(
            self, *, run_spec: Mapping, authorization_path: Path,
            artifact_paths: Mapping[str, str], audit_path: Path,
            journal_path: Path, log_path: Path, returncode: int) -> dict | None:
        """Recognize only a completed, SHA-bound BC failure, never a crash."""
        from backend.training import warehouse_v6_runner as runner
        from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor

        pair_dir = (Path(run_spec["output_dir"]).resolve() /
                    run_spec["candidate_id"] /
                    f"pair_{run_spec['pair_seed']}_steps_{run_spec['joint_steps']}")
        precheck_path = Path(artifact_paths["final_bc_precheck_receipt_path"])
        actor_path = Path(artifact_paths["final_bc_actor_path"])
        failure_message = "Warehouse final BC four-purpose internal precheck failed"
        try:
            if (returncode == 0 or not pair_dir.is_dir()
                    or any((pair_dir / name).exists() for name in (
                        "pair.training.json", "paired_bc.checkpoint.pt",
                        "raw.actor.npz", "regularized.actor.npz",
                        "raw.training.json", "regularized.training.json"))
                    or any(pair_dir.glob("raw.*"))
                    or any(pair_dir.glob("regularized.*"))
                    or not all(path.is_file() for path in (
                        precheck_path, actor_path, authorization_path,
                        audit_path, journal_path, log_path))
                    or failure_message.encode() not in log_path.read_bytes()):
                return None
            raw_precheck = precheck_path.read_bytes()
            precheck = json.loads(raw_precheck)
            if (not isinstance(precheck, Mapping)
                    or raw_precheck != (json.dumps(
                        precheck, sort_keys=True, allow_nan=False) + "\n").encode()
                    or precheck.get("purpose") !=
                    "development_internal_precheck_after_complete_bc"
                    or precheck.get("passed") is not False
                    or precheck.get("action_physics_config_sha256") !=
                    p.REGISTERED_BC_SPEC["action_physics_config_sha256"]):
                return None
            actor_receipt = precheck.get("final_bc_actor")
            if (not isinstance(actor_receipt, Mapping)
                    or actor_receipt.get("path") != str(actor_path)
                    or actor_receipt.get("sha256") != _file_sha(actor_path)
                    or actor_receipt.get("model_version") !=
                    p.REGISTERED_BC_SPEC["actor_model_version"]
                    or actor_receipt.get("feature_schema_sha256") !=
                    p.REGISTERED_BC_SPEC["feature_schema_sha256"]
                    or actor_receipt.get("action_physics_config_sha256") !=
                    p.REGISTERED_BC_SPEC["action_physics_config_sha256"]):
                return None
            actor = FrozenWarehouseV6Actor.load(
                actor_path, expected_sha256=actor_receipt["sha256"],
                expected_role_sha256=actor_receipt.get("role_sha256"))
            if actor.model_version != p.REGISTERED_BC_SPEC["actor_model_version"]:
                return None
            schedule = p.bc_data_seed_schedule(run_spec["pair_seed"])
            final_schedule = {name.removeprefix("final_bc_precheck_"): seeds
                              for name, seeds in schedule.items()
                              if name.startswith("final_bc_precheck_")}
            visits = precheck.get("final_bc_visits")
            if (precheck.get("final_bc_seed_schedule_sha256") != _sha(final_schedule)
                    or precheck.get("final_bc_stage_counts") != {
                        name: list(stage) for name, stage in
                        runner.BC_FINAL_CORE_STAGE_COUNTS.items()}
                    or not isinstance(visits, Mapping)
                    or set(visits) != set(final_schedule)
                    or any(not isinstance(visits[name], Mapping)
                           or visits[name].get("scenario_seeds") != seeds
                           or type(visits[name].get("rows")) is not int
                           or visits[name]["rows"] < 1
                           or visits[name].get("physical_frames") != visits[name]["rows"]
                           or visits[name].get("excluded_frames") != 0
                           for name, seeds in final_schedule.items())):
                return None
            roles = precheck.get("roles")
            if not isinstance(roles, Mapping) or set(roles) != set(p.ROLES):
                return None
            computed_pass = True
            failed_roles = []
            for role in p.ROLES:
                row = roles[role]
                if not isinstance(row, Mapping):
                    return None
                qualification = row.get("program_qualification")
                refit = row.get("refit_report")
                mass = row.get("qualified_validation_mass")
                actionable = row.get("expected_feedback_actionable_mass")
                installed = row.get("qualification_changes_actor") is False
                if (row.get("actor_role_sha256") != actor.role_sha256[role]
                        or not isinstance(refit, Mapping)
                        or not isinstance(qualification, Mapping)
                        or qualification.get("method") !=
                        p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD
                        or type(qualification.get("validation_passed")) is not bool
                        or type(installed) is not bool
                        or type(mass) not in (int, float)
                        or type(actionable) not in (int, float)
                        or not all(math.isfinite(float(value)) for value in
                                   (mass, actionable))
                        or not 0 <= actionable <= mass <= 1):
                    return None
                role_pass = bool(qualification["validation_passed"]
                                 and refit.get("validation_passed") is True
                                 and installed and mass >= .35
                                 and actionable >= .30)
                computed_pass &= role_pass
                if not role_pass:
                    failed_roles.append(role)
            if precheck.get("quality_screen") is not None:
                quality = precheck.get("quality_screen")
                if (not isinstance(quality, Mapping) or
                        precheck.get("quality_screen_seeds") != list(run_spec["checkpoint_seeds"])):
                    return None
                _validated_episodes(quality, run_spec["checkpoint_seeds"], require_trace=True,
                                    actor_sha256=actor_receipt["sha256"])
                quality_passed = p.quality_pass(quality["summary"], development=True)
                if computed_pass and quality_passed:
                    return None
                if not quality_passed:
                    failed_roles.extend(p.ROLES)
            authorization = _read(authorization_path)
            audit = _read(audit_path)
            raw_journal = journal_path.read_bytes()
            journal_rows = [json.loads(line) for line in
                            raw_journal.splitlines()]
            actual = audit.get("actual_reset_rows")
            # The locked runner canonicalizes the BC schedule before issuing
            # resets. Its input JSON is key-sorted, so iterating that raw
            # mapping would give a different (alphabetical) reset order.
            expected_resets = runner._formal_bc_reset_sequence(
                runner._checked_spec(run_spec), include_quality_screen="quality_screen" in precheck)
            if (authorization.get("authorization_kind") !=
                    "warehouse-v618-registered-pair-locked-run-v1"
                    or authorization.get("run_spec_sha256") !=
                    runner._scientific_run_spec_sha256(run_spec)
                    or authorization.get("training_config_sha256") !=
                    runner._training_config_sha256(run_spec)
                    or authorization.get("source_closure_sha256") !=
                    runner.source_closure_sha256()
                    or audit.get("status") != "failed"
                    or audit.get("error") != f"RuntimeError: {failure_message}"
                    or audit.get("candidate_id") != run_spec["candidate_id"]
                    or audit.get("pair_seed") != run_spec["pair_seed"]
                    or audit.get("joint_steps") != run_spec["joint_steps"]
                    or audit.get("authorization_receipt_sha256") !=
                    _file_sha(authorization_path)
                    or audit.get("run_spec_sha256") !=
                    authorization["run_spec_sha256"]
                    or audit.get("training_config_sha256") !=
                    authorization["training_config_sha256"]
                    or audit.get("source_closure_sha256") !=
                    authorization["source_closure_sha256"]
                    or audit.get("bc_recipe_sha256") !=
                    p.REGISTERED_BC_SPEC_SHA256
                    or audit.get("action_physics_config_sha256") !=
                    p.REGISTERED_BC_SPEC["action_physics_config_sha256"]
                    or audit.get("pair_report_sha256") is not None
                    or audit.get("selected_actor_role_sha256") is not None
                    or audit.get("actual_reset_rows") != journal_rows
                    or audit.get("actual_reset_count") != len(expected_resets)
                    or audit.get("actual_reset_sequence_sha256") !=
                    _sha(expected_resets)
                    or audit.get("declared_prefix_matches") is not True
                    or audit.get("reset_journal_path") != str(journal_path)
                    or audit.get("reset_journal_sha256") !=
                    sha256(raw_journal).hexdigest()
                    or not isinstance(actual, list)
                    or [{"phase": row.get("phase"), "seed": row.get("seed")}
                        for row in actual] != expected_resets):
                return None
            return {
                "candidate_id": run_spec["candidate_id"],
                "release_id": run_spec["release_id"],
                "pair_seed": run_spec["pair_seed"],
                "joint_steps": run_spec["joint_steps"],
                "run_spec_sha256": authorization["run_spec_sha256"],
                "training_config_sha256": authorization[
                    "training_config_sha256"],
                "source_closure_sha256": authorization["source_closure_sha256"],
                "authorization_receipt_path": str(authorization_path),
                "authorization_receipt_sha256": _file_sha(authorization_path),
                "bc_precheck_receipt_path": str(precheck_path),
                "bc_precheck_receipt_sha256": sha256(raw_precheck).hexdigest(),
                "final_bc_actor_path": str(actor_path),
                "final_bc_actor_sha256": actor_receipt["sha256"],
                "reset_audit_path": str(audit_path),
                "reset_audit_sha256": _file_sha(audit_path),
                "reset_journal_path": str(journal_path),
                "reset_journal_sha256": sha256(raw_journal).hexdigest(),
                "training_log_path": str(log_path),
                "training_log_sha256": _file_sha(log_path),
                "subprocess_returncode": returncode,
                "failed_roles": sorted(set(failed_roles)),
                "quality_screen_failed": bool(precheck.get("quality_screen") is not None
                    and not p.quality_pass(precheck["quality_screen"]["summary"], development=True)),
            }
        except (KeyError, TypeError, ValueError, OSError, RuntimeError,
                OverflowError, UnicodeError):
            return None

    def train_pair(self, spec: Mapping) -> dict:
        """Launch one registered pair through the SHA-bound, one-call gate."""
        import torch
        from backend.training import warehouse_v6_runner as runner
        from backend.training.warehouse_v6_actor import (
            WarehouseV6Actor, role_hashes)

        pair_seed, budget = spec["pair_seed"], spec["joint_steps"]
        registered = p.PairRunSpec(
            spec["candidate_id"], pair_seed, budget, spec["output_dir"],
            spec["program_fit_method"], release_id=spec["release_id"],
            scenario_namespace=spec["scenario_namespace"],
            checkpoint_seeds=tuple(spec["checkpoint_seeds"]))
        expected = registered.to_mapping()
        if any(spec.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Warehouse locked pair spec differs from registration")
        output_root = Path(spec["output_dir"]).resolve()
        candidate_root = output_root / spec["candidate_id"]
        pair_dir = candidate_root / f"pair_{pair_seed}_steps_{budget}"
        if pair_dir.exists():
            raise RuntimeError("Warehouse locked pair already has an output directory")
        candidate_root.mkdir(parents=True, exist_ok=True)
        if pair_seed in p.FORMAL_PAIR_SEEDS:
            admission = Admission(output_root, spec["candidate_id"],
                                  release_id=spec["release_id"], backend=self,
                                  program_fit_method=spec["program_fit_method"])
            admission._verified_development_lock(
                admission._required("budget.lock.json"))
            if budget != admission._required("budget.lock.json")["locked_budget"]:
                raise RuntimeError("Warehouse formal budget differs from development lock")
        source = runner.source_closure_sha256()
        lock_script = (PROJECT_ROOT / "scripts" /
                       "run_warehouse_v618_locked_pair.py").resolve(strict=True)
        lock_dir = candidate_root / "admission" / "locked-pairs"
        lock_dir.mkdir(parents=True, exist_ok=True)
        stem = f"pair_{pair_seed}_steps_{budget}"
        spec_path = lock_dir / f"{stem}.run-spec.json"
        authorization_path = lock_dir / f"{stem}.authorization.json"
        audit_path = lock_dir / f"{stem}.reset-audit.json"
        journal_path = lock_dir / f"{stem}.reset-audit.jsonl"
        if audit_path.exists() or journal_path.exists():
            raise RuntimeError("Warehouse locked pair reset audit was already claimed")
        artifact_paths = {
            "warmup_actor_path": str((pair_dir / "warmup.actor.npz").resolve()),
            "final_bc_actor_path": str((pair_dir / "final-bc.actor.npz").resolve()),
            "final_bc_precheck_receipt_path": str((
                pair_dir / "final-bc-internal-precheck.json").resolve()),
        }
        run_spec = {**expected, **artifact_paths,
                    "source_closure_sha256": source}
        spec_sha = _immutable_json(spec_path, run_spec)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(pair_seed)
            initial_roles = role_hashes(WarehouseV6Actor())
        authorization = {
            "authorization_kind": "warehouse-v618-registered-pair-locked-run-v1",
            "candidate_id": spec["candidate_id"],
            "pair_seed": pair_seed, "joint_steps": budget,
            "run_spec_sha256": runner._scientific_run_spec_sha256(run_spec),
            "training_config_sha256": runner._training_config_sha256(run_spec),
            "source_closure_sha256": source,
            "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
            "executable_recipe_file_sha256": p.REGISTERED_BC_SPEC[
                "executable_recipe_file_sha256"],
            "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                "action_physics_config_sha256"],
            "bc_data_seed_schedule_sha256": registered.bc_data_seed_schedule_sha256,
            "bc_critic_reset_schedule_sha256":
                registered.bc_critic_reset_schedule_sha256,
            "scenario_schedule_sha256": p.scenario_schedule_sha256(
                registered.scenario_namespace, pair_seed, budget),
            "final_program_fit_seed_schedule_sha256": expected[
                "final_program_fit_seed_schedule_sha256"],
            "initial_role_sha256": initial_roles,
            "output_directory": str(pair_dir),
            "artifact_paths_sha256": _sha(artifact_paths),
            "reset_audit_path": str(audit_path),
            "reset_audit_journal_path": str(journal_path),
            "lock_script_path": str(lock_script),
            "lock_script_sha256": _file_sha(lock_script),
            "run_spec_file_sha256": spec_sha,
        }
        _immutable_json(authorization_path, authorization)
        log_path = lock_dir / f"{stem}.train.log"
        with log_path.open("xb") as log:
            completed = subprocess.run(
                [sys.executable, str(lock_script), "--spec", str(spec_path),
                 "--authorization", str(authorization_path)],
                cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=False)
        if completed.returncode:
            failure = self._failed_bc_precheck_evidence(
                run_spec=run_spec, authorization_path=authorization_path,
                artifact_paths=artifact_paths, audit_path=audit_path,
                journal_path=journal_path, log_path=log_path,
                returncode=completed.returncode)
            if failure is not None:
                raise BCPrecheckFailure(failure)
            raise RuntimeError(
                f"Warehouse locked pair failed ({completed.returncode}); "
                f"see {log_path}")
        pair_report_path = pair_dir / "pair.training.json"
        if not pair_report_path.is_file():
            raise RuntimeError("Warehouse locked pair did not write training receipt")
        return _read(pair_report_path)

    def source_closure_sha256(self) -> str:
        from backend.training.warehouse_v6_runner import source_closure_sha256
        return source_closure_sha256()

    def load_actor(self, artifact: Mapping):
        from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
        return FrozenWarehouseV6Actor.load(
            artifact["path"], expected_sha256=artifact["sha256"],
            expected_role_sha256=artifact["role_sha256"])

    def load_program(self, artifact: Mapping, actor_role_sha256: str):
        from backend.training.warehouse_v6_program import WarehouseV6Program
        return WarehouseV6Program.load(
            artifact["path"], expected_sha256=artifact["sha256"],
            actor_role_sha256=actor_role_sha256)

    def evaluate(self, actor, seeds: Sequence[int], *, action_seed_offset: int = 0,
                 trace: bool = False) -> dict:
        from backend.training.warehouse_v6_runner import evaluate_authorized_frozen
        return evaluate_authorized_frozen(
            actor, seeds, action_seed_offset=action_seed_offset, trace=trace,
            authorization=self._evaluation_authorization(actor))

    def audit_programs(self, actor, programs: Mapping, seeds: Sequence[int], *,
                       trace_dir: str | Path | None = None) -> dict:
        from backend.training.warehouse_v6_runner import audit_authorized_frozen
        return audit_authorized_frozen(
            actor, programs, seeds, trace_dir=trace_dir,
            authorization=self._evaluation_authorization(actor))

    def _evaluation_authorization(self, actor) -> dict:
        from backend.training.warehouse_v6_evaluation_authorization import (
            development_evaluation_authorization, sealed_evaluation_authorization,
        )
        claim = getattr(self, "_sealed_evaluation_claim", None)
        if claim is None:
            return development_evaluation_authorization(actor)
        return sealed_evaluation_authorization(
            actor, claim_path=claim[0], expected_claim_sha256=claim[1])

    @contextmanager
    def sealed_evaluation(self, *, claim_path: str | Path,
                          expected_claim_sha256: str):
        """Bind an existing one-use claim; this never creates or spends seeds."""
        if getattr(self, "_sealed_evaluation_claim", None) is not None:
            raise RuntimeError("Warehouse sealed evaluation context is already active")
        self._sealed_evaluation_claim = (Path(claim_path), expected_claim_sha256)
        try:
            yield
        finally:
            self._sealed_evaluation_claim = None

    def verify_program_trace(self, *, actor, programs: Mapping,
                             trace_bytes_by_role: Mapping[str, bytes],
                             audits: Mapping, seeds: Sequence[int]) -> bool:
        from experiments.warehouse.acceptance import _verify_program_trace_rows
        return _verify_program_trace_rows(
            actor=actor, programs=programs,
            trace_bytes_by_role=trace_bytes_by_role, audits=audits,
            expected_seeds=seeds)


class Admission:
    """State machine for a single candidate and release namespace."""

    def __init__(self, output_root: str | Path, candidate_id: str, *,
                 release_id: str = p.INITIAL_RELEASE_ID, backend=None,
                 claim_root: str | Path | None = None,
                 program_fit_method: str | None = None):
        if (not candidate_id or "/" in candidate_id or ".." in candidate_id
                or "\\" in candidate_id):
            raise ValueError("candidate_id must be a simple directory name")
        p.sealed_seed_blocks(release_id)  # Validate version before any work.
        if program_fit_method not in p.PROGRAM_FIT_METHOD_CANDIDATES:
            raise ValueError("Warehouse candidate requires an explicit supported program method")
        minor_text = release_id.rsplit(".v6.", 1)[1]
        minor = int(minor_text)
        if minor_text != str(minor):
            raise ValueError("Warehouse release version must use a canonical minor number")
        self.output_root = Path(output_root).resolve()
        self.candidate_id = candidate_id
        self.release_id = release_id
        self.program_fit_method = program_fit_method
        policy_version = "v6" if minor == 0 else f"v6.{minor}"
        self.policy_ids = {variant: f"warehouse-{variant}-{policy_version}"
                           for variant in p.VARIANTS}
        self.directory = self.output_root / candidate_id
        self.receipts = self.directory / "admission"
        self.claim_root = Path(claim_root if claim_root is not None
                               else DEFAULT_CLAIM_ROOT).resolve()
        self.backend = backend if backend is not None else NativeBackend()

    def _path(self, name: str) -> Path:
        return self.receipts / name

    def _claim_path(self, phase: str) -> Path:
        return self.claim_root / self.release_id / f"{phase}.claim.json"

    def _required(self, name: str) -> dict:
        path = self._path(name)
        if not path.is_file():
            raise RuntimeError(f"Warehouse prerequisite receipt missing: {name}")
        receipt = _read(path)
        if (receipt.get("candidate_id") != self.candidate_id
                or receipt.get("release_id") != self.release_id):
            raise RuntimeError(f"Warehouse prerequisite identity mismatch: {name}")
        return receipt

    def _new(self, name: str, value: Mapping) -> dict:
        row = {"candidate_id": self.candidate_id,
               "release_id": self.release_id, **value}
        _exclusive_json(self._path(name), row)
        return row

    def _record_failure(self, phase: str, error: BaseException) -> None:
        path = self._path(f"{phase}.failure.json")
        if not path.exists():
            _exclusive_json(path, {"candidate_id": self.candidate_id,
                                   "release_id": self.release_id,
                                   "phase": phase, "error_type": type(error).__name__,
                                   "error": str(error), "passed": False})

    def _ensure_open(self) -> None:
        if self._path("candidate.closed.json").exists():
            raise RuntimeError("Warehouse candidate is closed; use a new candidate and version")

    def _close(self, phase: str, reason: str, **evidence: object) -> None:
        if not self._path("candidate.closed.json").exists():
            self._new("candidate.closed.json", {"phase": phase, "reason": reason,
                                                 "passed": False, **evidence})

    def _pair_spec(self, pair_seed: int, budget: int) -> p.PairRunSpec:
        selection = (p.DEVELOPMENT_SELECTION_SEEDS if
                     pair_seed in p.DEVELOPMENT_PAIR_SEEDS else
                     p.FORMAL_SELECTION_SEEDS)
        return p.PairRunSpec(self.candidate_id, pair_seed, budget,
                             str(self.output_root), self.program_fit_method,
                             release_id=self.release_id,
                             checkpoint_seeds=selection)

    def _verified_bc_failure_evidence(self, evidence: Mapping,
                                      pair_seed: int, budget: int) -> None:
        """Rebind a failed locked BC receipt to this candidate and source."""
        source = self.backend.source_closure_sha256()
        spec = self._pair_spec(pair_seed, budget)
        pair_dir = self.directory / f"pair_{pair_seed}_steps_{budget}"
        lock_dir = self.receipts / "locked-pairs"
        stem = f"pair_{pair_seed}_steps_{budget}"
        paths = {
            "authorization_receipt_path": lock_dir / f"{stem}.authorization.json",
            "bc_precheck_receipt_path": pair_dir / "final-bc-internal-precheck.json",
            "final_bc_actor_path": pair_dir / "final-bc.actor.npz",
            "reset_audit_path": lock_dir / f"{stem}.reset-audit.json",
            "reset_journal_path": lock_dir / f"{stem}.reset-audit.jsonl",
            "training_log_path": lock_dir / f"{stem}.train.log",
        }
        if (not isinstance(evidence, Mapping)
                or evidence.get("candidate_id") != self.candidate_id
                or evidence.get("release_id") != self.release_id
                or evidence.get("pair_seed") != pair_seed
                or evidence.get("joint_steps") != budget
                or evidence.get("source_closure_sha256") != source
                or evidence.get("run_spec_sha256") != spec.sha256()
                or evidence.get("training_config_sha256") !=
                   spec.training_config_sha256(source)
                or type(evidence.get("subprocess_returncode")) is not int
                or evidence["subprocess_returncode"] == 0
                or not isinstance(evidence.get("failed_roles"), list)
                or not evidence["failed_roles"]
                or not set(evidence["failed_roles"]) <= set(p.ROLES)):
            raise RuntimeError("Warehouse failed BC receipt identity changed")
        for key, expected in paths.items():
            digest_key = key.removesuffix("_path") + "_sha256"
            if (evidence.get(key) != str(expected.resolve())
                    or not expected.is_file()
                    or evidence.get(digest_key) != _file_sha(expected)):
                raise RuntimeError("Warehouse failed BC receipt asset changed")
        authorization = _read(paths["authorization_receipt_path"])
        precheck = _read(paths["bc_precheck_receipt_path"])
        audit = _read(paths["reset_audit_path"])
        if (authorization.get("source_closure_sha256") != source
                or authorization.get("run_spec_sha256") != spec.sha256()
                or authorization.get("training_config_sha256") !=
                   spec.training_config_sha256(source)
                or precheck.get("passed") is not False
                or precheck.get("purpose") !=
                   "development_internal_precheck_after_complete_bc"
                or audit.get("status") != "failed"
                or audit.get("authorization_receipt_sha256") !=
                   evidence["authorization_receipt_sha256"]
                or audit.get("source_closure_sha256") != source
                or audit.get("pair_report_sha256") is not None
                or (pair_dir / "pair.training.json").exists()):
            raise RuntimeError("Warehouse failed BC receipts do not agree")

    def _finish_failed_bc_precheck(self, *, pair_seed: int, budget: int,
                                   evidence: Mapping) -> dict:
        """Write a permanent failed development decision and close the run."""
        self._verified_bc_failure_evidence(evidence, pair_seed, budget)
        filename = f"development.{budget}.{pair_seed}.json"
        row_data = {
            "pair_seed": pair_seed, "budget": budget,
            "failure_kind": "final_bc_internal_precheck",
            "bc_precheck_failure": dict(evidence),
            "source_closure_sha256": evidence["source_closure_sha256"],
            "training": {}, "evaluation": {}, "program_audit": {},
            "stress": {}, "stress_episodes": 0,
            "passed": False,
        }
        if self._path(filename).exists():
            row = self._required(filename)
            if row != {"candidate_id": self.candidate_id,
                       "release_id": self.release_id, **row_data}:
                raise RuntimeError("Warehouse failed BC development receipt changed")
        else:
            row = self._new(filename, row_data)
        row_sha = _file_sha(self._path(filename))
        failed_data = {
            "failure_kind": "final_bc_internal_precheck",
            "failed_pair_seed": pair_seed,
            "failed_budget": budget,
            "failed_pair_receipt_sha256": row_sha,
            "bc_precheck_receipt_sha256": evidence[
                "bc_precheck_receipt_sha256"],
            "source_closure_sha256": evidence["source_closure_sha256"],
            "budgets_tried": list(p.BUDGETS[:p.BUDGETS.index(budget) + 1]),
            "remaining_budgets_skipped": list(
                p.BUDGETS[p.BUDGETS.index(budget) + 1:]),
            "reason": "registered final BC precheck failed before PPO",
            "passed": False,
        }
        if self._path("development.failed.json").exists():
            failed = self._required("development.failed.json")
            if failed != {"candidate_id": self.candidate_id,
                          "release_id": self.release_id, **failed_data}:
                raise RuntimeError("Warehouse failed development summary changed")
        else:
            failed = self._new("development.failed.json", failed_data)
        self._close("development", failed_data["reason"],
                    failed_pair_receipt_sha256=row_sha,
                    bc_precheck_receipt_sha256=evidence[
                        "bc_precheck_receipt_sha256"],
                    source_closure_sha256=evidence["source_closure_sha256"])
        return failed

    def _train(self, pair_seed: int, budget: int) -> dict:
        spec = self._pair_spec(pair_seed, budget)
        expected_spec = spec.to_mapping()
        report = self.backend.train_pair(expected_spec)
        source = self.backend.source_closure_sha256()
        if (report.get("candidate_id") != self.candidate_id
                or report.get("release_id") != self.release_id
                or report.get("pair_seed") != pair_seed
                or report.get("joint_steps") != budget
                or report.get("architecture_mode") != spec.architecture_mode
                or report.get("program_fit_method") != spec.program_fit_method
                or report.get("bc_recipe_version") != spec.bc_recipe_version
                or report.get("bc_recipe_sha256") != spec.bc_recipe_sha256
                or report.get("risk_dual_config_sha256") !=
                spec.risk_dual_config_sha256
                or report.get("bc_data_seed_schedule") !=
                expected_spec["bc_data_seed_schedule"]
                or report.get("bc_data_seed_schedule_sha256") !=
                spec.bc_data_seed_schedule_sha256
                or report.get("bc_critic_reset_schedule") !=
                expected_spec["bc_critic_reset_schedule"]
                or report.get("bc_critic_reset_schedule_sha256") !=
                spec.bc_critic_reset_schedule_sha256
                or report.get("bc", {}).get("bc_data_seed_blocks") !=
                expected_spec["bc_data_seed_schedule"]
                or report.get("bc", {}).get("bc_critic_reset_schedule") !=
                expected_spec["bc_critic_reset_schedule"]
                or report.get("bc", {}).get("bc_recipe_sha256") !=
                spec.bc_recipe_sha256
                or not p.bc_report_valid(report.get("bc", {}), pair_seed, source)
                or report.get("checkpoint_seeds") != list(spec.checkpoint_seeds)
                or report.get("online_program_extraction_steps") !=
                list(spec.online_program_extraction_steps)
                or report.get("topology_refresh_steps") !=
                list(spec.topology_refresh_steps)
                or report.get("topology_frozen_at_step") !=
                spec.topology_frozen_at_step
                or report.get("final_program_fit_seed_schedule") !=
                expected_spec["final_program_fit_seed_schedule"]
                or report.get("final_program_fit_seed_schedule_sha256") !=
                expected_spec["final_program_fit_seed_schedule_sha256"]
                or not source or report.get("source_closure_sha256") != source
                or any(row.get("source_closure_sha256") != source
                       for row in report.get("variants", {}).values())):
            raise RuntimeError("Warehouse paired training identity or source mismatch")
        if not p.paired_training_pass(report.get("variants", {}), joint_steps=budget):
            raise TrainingGateFailure(report)
        return report

    def _artifact_path(self, artifact: Mapping) -> Path:
        path = Path(artifact["path"]).resolve()
        if not path.is_relative_to(self.directory):
            raise RuntimeError("Warehouse artifact is outside candidate run directory")
        if _file_sha(path) != artifact["sha256"]:
            raise RuntimeError("Warehouse artifact file hash mismatch")
        return path

    def _load_variant(self, report: Mapping, variant: str):
        entry = report["variants"][variant]
        artifact = entry["actor_artifact"]
        self._artifact_path(artifact)
        actor = self.backend.load_actor(artifact)
        if (actor.artifact_sha256 != artifact["sha256"]
                or actor.role_sha256 != artifact["role_sha256"]
                or actor.role_sha256 != entry["selected_role_sha256"]):
            raise RuntimeError("Warehouse Actor role hash differs from training report")
        return actor

    def _verify_selected_assets(self, pair: Mapping) -> None:
        """Check both frozen Actors and training reports before a sealed claim."""
        artifacts = {variant: pair["variants"][variant]["actor_artifact"]
                     for variant in p.VARIANTS}
        if (artifacts["raw"].get("model_version") !=
                artifacts["regularized"].get("model_version")
                or artifacts["raw"].get("feature_schema_sha256") !=
                artifacts["regularized"].get("feature_schema_sha256")
                or artifacts["raw"].get("action_physics_config_sha256") !=
                p.REGISTERED_BC_SPEC["action_physics_config_sha256"]
                or artifacts["regularized"].get("action_physics_config_sha256") !=
                p.REGISTERED_BC_SPEC["action_physics_config_sha256"]):
            raise RuntimeError("Warehouse paired Actor architecture differs")
        for variant in p.VARIANTS:
            self._load_variant(pair, variant)
            entry = pair["variants"][variant]
            path = Path(entry["training_report_path"]).resolve()
            if (not path.is_relative_to(self.directory)
                    or _file_sha(path) != entry["training_report_sha256"]
                    or _read(path) != {key: value for key, value in entry.items()
                                       if key not in ("training_report_path",
                                                      "training_report_sha256")}):
                raise RuntimeError("Warehouse selected training report hash mismatch")

    def _load_programs(self, report: Mapping, actor) -> dict:
        artifacts = report["variants"]["regularized"].get("program_artifacts", {})
        if set(artifacts) != set(p.ROLES):
            raise RuntimeError("Warehouse regularized Actor lacks both role programs")
        programs = {}
        for role in p.ROLES:
            artifact = artifacts[role]
            self._artifact_path(artifact)
            programs[role] = self.backend.load_program(
                artifact, actor.role_sha256[role])
            if (programs[role].role != role
                    or programs[role].sha256() != artifact["sha256"]
                    or programs[role].report.get("fit_method") !=
                    self.program_fit_method):
                raise RuntimeError("Warehouse program role or content hash mismatch")
            try:
                split = programs[role].report["split_episode_ids"]
                expected_counts = {"fit": 50, "calibration": 25,
                                   "eligibility": 25, "validation": 24}
                if set(split) != set(expected_counts):
                    raise ValueError("wrong split names")
                groups = {name: set(ids) for name, ids in split.items()}
                if (any(len(groups[name]) != count or len(split[name]) != count
                        for name, count in expected_counts.items())
                        or any(groups[left] & groups[right]
                               for left in groups for right in groups if left < right)):
                    raise ValueError("program training episode splits overlap or shrink")
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise RuntimeError(
                    "Warehouse program training episode split certificate invalid") from exc
        return programs

    def _stress(self, actor) -> dict:
        episodes = []
        for task, scenario_seed, action_seed in p.stress_schedule():
            result = self.backend.evaluate(actor, [scenario_seed],
                                           action_seed_offset=action_seed-scenario_seed,
                                           trace=True)
            row = _validated_episodes(
                result, [scenario_seed], require_trace=True,
                actor_sha256=actor.artifact_sha256,
                action_seed_offset=action_seed-scenario_seed)[0]
            episodes.append({"task": task, "seed": scenario_seed,
                             "scenario_seed": scenario_seed,
                             "action_seed": action_seed,
                             "deliveries": row["deliveries"],
                             "collisions": row["collisions"],
                             "shutdown_episode": row["shutdown_episode"],
                             "quality_trace": row["quality_trace"]})
        return {"summary": _summary(episodes), "episodes": episodes,
                "schedule_sha256": _sha(p.stress_schedule()),
                "actor_sha256": actor.artifact_sha256}

    def _verified_development_pair(self, budget: int, pair_seed: int) -> dict:
        """Rebuild a development decision from its saved training and trace bytes.

        This is used both while developing and before the first formal seed.
        A budget lock alone is never evidence that the development gates ran.
        """
        row = self._required(f"development.{budget}.{pair_seed}.json")
        pair = _read(self.directory / f"pair_{pair_seed}_steps_{budget}" /
                     "pair.training.json")
        spec = self._pair_spec(pair_seed, budget)
        source = self.backend.source_closure_sha256()
        expected = spec.to_mapping()
        if (row.get("pair_seed") != pair_seed
                or row.get("budget") != budget
                or row.get("pair_report_sha256") != _sha(pair)
                or row.get("source_closure_sha256") != source
                or row.get("training") != pair.get("variants")
                or row.get("program_audit_seeds") !=
                list(p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS)
                or row.get("stress_schedule_sha256") != _sha(p.stress_schedule())
                or pair.get("candidate_id") != self.candidate_id
                or pair.get("release_id") != self.release_id
                or pair.get("pair_seed") != pair_seed
                or pair.get("joint_steps") != budget
                or pair.get("source_closure_sha256") != source
                or pair.get("run_spec_sha256") != spec.sha256()
                or pair.get("training_config_sha256") !=
                spec.training_config_sha256(source)
                or pair.get("architecture_mode") != spec.architecture_mode
                or pair.get("program_fit_method") != self.program_fit_method
                or pair.get("bc_recipe_version") != spec.bc_recipe_version
                or pair.get("bc_recipe_sha256") != spec.bc_recipe_sha256
                or pair.get("risk_dual_config_sha256") !=
                spec.risk_dual_config_sha256
                or pair.get("bc_data_seed_schedule") !=
                expected["bc_data_seed_schedule"]
                or pair.get("bc_data_seed_schedule_sha256") !=
                spec.bc_data_seed_schedule_sha256
                or pair.get("bc_critic_reset_schedule") !=
                expected["bc_critic_reset_schedule"]
                or pair.get("bc_critic_reset_schedule_sha256") !=
                spec.bc_critic_reset_schedule_sha256
                or pair.get("checkpoint_seeds") != list(spec.checkpoint_seeds)
                or pair.get("online_program_extraction_steps") !=
                list(spec.online_program_extraction_steps)
                or pair.get("topology_refresh_steps") !=
                list(spec.topology_refresh_steps)
                or pair.get("topology_frozen_at_step") !=
                spec.topology_frozen_at_step
                or pair.get("final_program_fit_seed_schedule") !=
                expected["final_program_fit_seed_schedule"]
                or pair.get("final_program_fit_seed_schedule_sha256") !=
                expected["final_program_fit_seed_schedule_sha256"]
                or not p.bc_report_valid(pair.get("bc"), pair_seed, source)):
            raise RuntimeError("Warehouse development pair or BC evidence changed")
        if source in p.RETIRED_119_SOURCE_CLOSURE_SHA256:
            raise RuntimeError("Warehouse failed 119 source identity is retired")
        variants = pair.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(p.VARIANTS):
            raise RuntimeError("Warehouse development paired variants missing")
        for variant in p.VARIANTS:
            trained = variants[variant]
            actor = trained.get("actor_artifact", {})
            if (trained.get("program_fit_method") != self.program_fit_method
                    or trained.get("source_closure_sha256") != source
                    or trained.get("run_spec_sha256") != spec.sha256()
                    or trained.get("training_config_sha256") !=
                    spec.training_config_sha256(source)
                    or actor.get("sha256") == p.RETIRED_119_ACTOR_SHA256
                    or actor.get("model_version") ==
                    p.RETIRED_119_MODEL_VERSION):
                raise RuntimeError("Warehouse failed 119 Actor or training identity is retired")
        self._verify_selected_assets(pair)
        training_passed = p.paired_training_pass(variants, joint_steps=budget)
        if not training_passed:
            if (row.get("passed") is not False
                    or not row.get("gate_failure")
                    or row.get("stress") != {}
                    or row.get("program_audit") != {}
                    or row.get("evaluation") != {}
                    or row.get("stress_episodes") != 0):
                raise RuntimeError("Warehouse failed training gate receipt changed")
            return row
        if row.get("gate_failure") is not None:
            raise RuntimeError("Warehouse passing training has a failure receipt")
        stress = row.get("stress")
        if (not isinstance(stress, Mapping)
                or set(stress) != set(p.VARIANTS)
                or row.get("stress_episodes") != len(p.stress_schedule())
                or not isinstance(row.get("evaluation"), Mapping)):
            raise RuntimeError("Warehouse development stress evidence missing")
        for variant in p.VARIANTS:
            if (not _verify_stress(stress[variant])
                    or stress[variant].get("actor_sha256") !=
                    variants[variant]["actor_artifact"]["sha256"]
                    or stress[variant].get("summary") !=
                    row["evaluation"].get(variant)):
                raise RuntimeError("Warehouse development stress evidence changed")
        actor = self._load_variant(pair, "regularized")
        programs = {}
        try:
            programs = self._load_programs(pair, actor)
        except RuntimeError as exc:
            if ("lacks both role programs" not in str(exc)
                    or row.get("program_error") != str(exc)
                    or row.get("program_audit") != {}):
                raise RuntimeError("Warehouse development program evidence changed") from exc
        else:
            if row.get("program_error") is not None:
                raise RuntimeError("Warehouse development program failure was fabricated")
        audit_valid = False
        if programs:
            audits = row.get("program_audit")
            trace_root = (self.directory / "traces" /
                          f"development.{budget}.{pair_seed}").resolve()
            if not isinstance(audits, Mapping) or set(audits) != set(p.ROLES):
                raise RuntimeError("Warehouse development program audit missing")
            trace_bytes = {}
            for role in p.ROLES:
                trace_path = Path(audits[role]["trace_path"]).resolve()
                if not trace_path.is_relative_to(trace_root):
                    raise RuntimeError("Warehouse development trace escaped run")
                trace_bytes[role] = trace_path.read_bytes()
            if self.backend.verify_program_trace(
                    actor=actor, programs=programs,
                    trace_bytes_by_role=trace_bytes, audits=audits,
                    seeds=p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS) is not True:
                raise RuntimeError("Warehouse development program trace invalid")
            audit_valid = _audit_valid(
                audits, actor, programs, p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS,
                development=True)
        computed = (p.development_pair_pass(row) and bool(programs)
                    and audit_valid)
        if row.get("passed") is not computed:
            raise RuntimeError("Warehouse development gate result changed")
        return row

    def _verified_development_lock(self, lock: Mapping) -> dict[int, list[dict]]:
        """Recheck the complete ordered development history before formal use."""
        budget = lock.get("locked_budget")
        if budget not in p.BUDGETS or lock.get("passed") is not True:
            raise RuntimeError("Warehouse development budget is not locked")
        source = self.backend.source_closure_sha256()
        schedules = {str(seed): p.final_program_fit_seed_schedule(
            self.release_id, seed)
            for seed in (*p.DEVELOPMENT_PAIR_SEEDS, *p.FORMAL_PAIR_SEEDS)}
        if (lock.get("source_closure_sha256") != source
                or source in p.RETIRED_119_SOURCE_CLOSURE_SHA256
                or lock.get("development_pair_seeds") !=
                list(p.DEVELOPMENT_PAIR_SEEDS)
                or lock.get("final_program_fit_seed_schedule") != schedules
                or lock.get("final_program_fit_seed_schedule_sha256") !=
                _sha(schedules)
                or any(lock.get(key) != value for key, value in
                       p.registered_method_lock_fields(
                           self.program_fit_method).items())
                or lock.get("architecture_mode") !=
                p.REGISTERED_ARCHITECTURE_MODE
                or lock.get("bc_recipe_version") !=
                p.REGISTERED_BC_RECIPE_VERSION
                or lock.get("online_program_extraction_steps") !=
                list(p.online_program_extraction_steps(budget))
                or lock.get("topology_refresh_steps") !=
                list(p.topology_refresh_steps(budget))
                or lock.get("topology_frozen_at_step") !=
                p.topology_frozen_at_step(budget)):
            raise RuntimeError("Warehouse development seed lock changed or source retired")
        rows = {}
        for candidate_budget in p.BUDGETS[:p.BUDGETS.index(budget) + 1]:
            rows[candidate_budget] = [self._verified_development_pair(
                candidate_budget, seed) for seed in p.DEVELOPMENT_PAIR_SEEDS]
        selected_hashes = {str(seed): _file_sha(self._path(
            f"development.{budget}.{seed}.json"))
            for seed in p.DEVELOPMENT_PAIR_SEEDS}
        if (lock.get("development_receipt_sha256") != selected_hashes
                or p.choose_development_budget(rows) != budget
                or not all(row["passed"] for row in rows[budget])):
            raise RuntimeError("Warehouse development lock lacks a passing minimal budget")
        return rows

    def develop(self) -> dict:
        """Try registered budgets in order; lock the smallest fully passing one."""
        self._ensure_open()
        lock_path = self._path("budget.lock.json")
        if lock_path.exists():
            lock = self._required("budget.lock.json")
            self._verified_development_lock(lock)
            return lock
        if self._path("development.failed.json").exists():
            failed = self._required("development.failed.json")
            if failed.get("failure_kind") == "final_bc_internal_precheck":
                seed, budget = failed.get("failed_pair_seed"), failed.get(
                    "failed_budget")
                if (seed not in p.DEVELOPMENT_PAIR_SEEDS
                        or budget not in p.BUDGETS):
                    raise RuntimeError("Warehouse failed BC pair identity changed")
                row = self._required(f"development.{budget}.{seed}.json")
                return self._finish_failed_bc_precheck(
                    pair_seed=seed, budget=budget,
                    evidence=row.get("bc_precheck_failure", {}))
            return failed
        reports_by_budget = {}
        for budget in p.BUDGETS:
            rows = []
            for pair_seed in p.DEVELOPMENT_PAIR_SEEDS:
                filename = f"development.{budget}.{pair_seed}.json"
                if self._path(filename).exists():
                    row = self._required(filename)
                    if (row.get("source_closure_sha256") !=
                            self.backend.source_closure_sha256()):
                        raise RuntimeError("Warehouse development source closure changed")
                    if row.get("failure_kind") == "final_bc_internal_precheck":
                        return self._finish_failed_bc_precheck(
                            pair_seed=pair_seed, budget=budget,
                            evidence=row.get("bc_precheck_failure", {}))
                    rows.append(row)
                    continue
                phase = f"development.{budget}.{pair_seed}"
                try:
                    try:
                        pair = self._train(pair_seed, budget)
                    except BCPrecheckFailure as gate:
                        return self._finish_failed_bc_precheck(
                            pair_seed=pair_seed, budget=budget,
                            evidence=gate.evidence)
                    except TrainingGateFailure as gate:
                        row = self._new(filename, {
                            "pair_seed": pair_seed, "budget": budget,
                            "pair_report_sha256": _sha(gate.report),
                            "source_closure_sha256": gate.report[
                                "source_closure_sha256"],
                            "training": gate.report.get("variants", {}),
                            "evaluation": {}, "program_audit": {},
                            "stress": {}, "stress_episodes": 0,
                            "stress_schedule_sha256": _sha(p.stress_schedule()),
                            "program_audit_seeds": list(
                                p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS),
                            "gate_failure": str(gate), "passed": False,
                        })
                        rows.append(row)
                        continue
                    stress = {variant: self._stress(self._load_variant(pair, variant))
                              for variant in p.VARIANTS}
                    actor = self._load_variant(pair, "regularized")
                    program_error = None
                    try:
                        programs = self._load_programs(pair, actor)
                    except RuntimeError as exc:
                        if "lacks both role programs" not in str(exc):
                            raise
                        programs, program_error = {}, str(exc)
                    audits = (self.backend.audit_programs(
                        actor, programs, p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS,
                        trace_dir=self.directory / "traces" /
                        f"development.{budget}.{pair_seed}")
                        if programs else {})
                    facts = {
                        "pair_seed": pair_seed, "budget": budget,
                        "pair_report_sha256": _sha(pair),
                        "source_closure_sha256": pair["source_closure_sha256"],
                        "training": pair["variants"],
                        "evaluation": {variant: stress[variant]["summary"]
                                       for variant in p.VARIANTS},
                        "stress_episodes": len(p.stress_schedule()),
                        "stress_schedule_sha256": _sha(p.stress_schedule()),
                        "stress": stress, "program_audit": audits,
                        "program_audit_seeds": list(p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS),
                    }
                    if program_error is not None:
                        facts["program_error"] = program_error
                    facts["passed"] = (p.development_pair_pass(facts)
                                       and bool(programs) and _audit_valid(
                                           audits, actor, programs,
                                           p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS,
                                           development=True))
                    row = self._new(filename, facts)
                    rows.append(row)
                except Exception as exc:
                    self._record_failure(phase, exc)
                    raise
            reports_by_budget[budget] = rows
            verified = [self._verified_development_pair(budget, seed)
                        for seed in p.DEVELOPMENT_PAIR_SEEDS]
            if verified != rows:
                raise RuntimeError("Warehouse development receipt changed during verification")
            proposed = p.choose_development_budget(reports_by_budget)
            chosen = (proposed if proposed is not None and all(
                row["passed"] for row in reports_by_budget[proposed]) else None)
            if chosen is not None:
                final_program_schedules = {
                    str(seed): p.final_program_fit_seed_schedule(
                        self.release_id, seed)
                    for seed in (*p.DEVELOPMENT_PAIR_SEEDS, *p.FORMAL_PAIR_SEEDS)}
                return self._new("budget.lock.json", {
                    "locked_budget": chosen, "development_pair_seeds":
                    list(p.DEVELOPMENT_PAIR_SEEDS),
                    **p.registered_method_lock_fields(self.program_fit_method),
                    "architecture_mode": p.REGISTERED_ARCHITECTURE_MODE,
                    "bc_recipe_version": p.REGISTERED_BC_RECIPE_VERSION,
                    "online_program_extraction_steps": list(
                        p.online_program_extraction_steps(chosen)),
                    "topology_refresh_steps": list(
                        p.topology_refresh_steps(chosen)),
                    "topology_frozen_at_step": p.topology_frozen_at_step(chosen),
                    "final_program_fit_seed_schedule": final_program_schedules,
                    "final_program_fit_seed_schedule_sha256": _sha(
                        final_program_schedules),
                    "source_closure_sha256": self.backend.source_closure_sha256(),
                    "development_receipt_sha256": {
                        str(seed): _file_sha(self._path(f"development.{budget}.{seed}.json"))
                        for seed in p.DEVELOPMENT_PAIR_SEEDS},
                    "passed": True})
        return self._new("development.failed.json", {
            "budgets_tried": list(p.BUDGETS), "passed": False,
            "reason": "no registered budget passed all development pairs"})

    def formal_train(self) -> dict:
        """Select one pair by fixed development quality, before any preaudit."""
        self._ensure_open()
        if self._path("formal.selection.json").exists():
            selection = self._required("formal.selection.json")
            self._selected_pair()
            return selection
        lock = self._required("budget.lock.json")
        self._verified_development_lock(lock)
        budget = lock["locked_budget"]
        rows = []
        for pair_seed in p.FORMAL_PAIR_SEEDS:
            phase = f"formal.training.{pair_seed}"
            filename = f"{phase}.json"
            if self._path(filename).exists():
                row = self._required(filename)
                pair = _read(self.directory /
                             f"pair_{pair_seed}_steps_{budget}" / "pair.training.json")
                if not _verify_formal_row(row, pair, lock["source_closure_sha256"]):
                    raise RuntimeError("Warehouse formal pair receipt or source changed")
                rows.append(row)
                continue
            try:
                pair = self._train(pair_seed, budget)
                variants = {}
                for variant in p.VARIANTS:
                    actor = self._load_variant(pair, variant)
                    evaluated = self.backend.evaluate(
                        actor, p.FORMAL_SELECTION_SEEDS, trace=True)
                    episodes = _validated_episodes(
                        evaluated, p.FORMAL_SELECTION_SEEDS,
                        require_trace=True, actor_sha256=actor.artifact_sha256)
                    variants[variant] = {
                        "quality": _summary(episodes),
                        "episodes": episodes,
                        "training_report_sha256": pair["variants"][variant][
                            "training_report_sha256"],
                        "actor_sha256": pair["variants"][variant][
                            "actor_artifact"]["sha256"],
                    }
                facts = {
                    "pair_seed": pair_seed, "budget": budget,
                    "pair_report_sha256": _sha(pair),
                    "source_closure_sha256": pair["source_closure_sha256"],
                    "selection_seeds": list(p.FORMAL_SELECTION_SEEDS),
                    "variants": variants, "passed": all(
                        p.quality_pass(variants[v]["quality"], development=False)
                        for v in p.VARIANTS),
                }
                if not _verify_formal_row(facts, pair, lock["source_closure_sha256"]):
                    raise RuntimeError("Warehouse formal evaluation receipt invalid")
                rows.append(self._new(filename, facts))
            except BaseException as exc:
                self._record_failure(phase, exc)
                pair_dir = self.directory / f"pair_{pair_seed}_steps_{budget}"
                lock_dir = self.receipts / "locked-pairs"
                stem = f"pair_{pair_seed}_steps_{budget}"
                claim_files = {
                    "authorization": lock_dir / f"{stem}.authorization.json",
                    "reset_audit": lock_dir / f"{stem}.reset-audit.json",
                    "reset_journal": lock_dir / f"{stem}.reset-audit.jsonl",
                    "training_log": lock_dir / f"{stem}.train.log",
                    "pair_training": pair_dir / "pair.training.json",
                }
                claimed = (pair_dir.exists()
                           or any(path.exists() for path in claim_files.values())
                           or isinstance(exc, TrainingGateFailure))
                if claimed:
                    self._close(
                        phase, "formal locked pair failed after output claim",
                        source_closure_sha256=lock["source_closure_sha256"],
                        budget_lock_sha256=_file_sha(self._path(
                            "budget.lock.json")),
                        failure_receipt_sha256=_file_sha(self._path(
                            f"{phase}.failure.json")),
                        claimed_pair_output_path=str(pair_dir.resolve()),
                        claimed_pair_output_exists=pair_dir.exists(),
                        claimed_artifact_sha256={
                            name: _file_sha(path)
                            for name, path in claim_files.items()
                            if path.is_file()},
                    )
                raise
        try:
            selected = p.select_formal_pair(rows)
        except ValueError as exc:
            self._record_failure("formal.selection", exc)
            self._close("formal.selection", "no pair passed both quality gates")
            raise
        result = self._new("formal.selection.json", {
            "locked_budget": budget,
            "pair_seeds": list(p.FORMAL_PAIR_SEEDS),
            "selected_pair_seed": selected["pair_seed"],
            "selection_seeds": list(p.FORMAL_SELECTION_SEEDS),
            "selection_rule": "maximin_deliveries_minus_0.25_collisions_minus_5_shutdown",
            "pair_quality": {str(row["pair_seed"]): {
                variant: row["variants"][variant]["quality"]
                for variant in p.VARIANTS} for row in rows},
            "pair_receipt_sha256": {
                str(row["pair_seed"]): _file_sha(self._path(
                    f"formal.training.{row['pair_seed']}.json")) for row in rows},
            "passed": True})
        return result

    def _selected_pair(self) -> tuple[dict, dict]:
        selection = self._required("formal.selection.json")
        if selection.get("passed") is not True:
            raise RuntimeError("Warehouse formal selection did not pass")
        lock = self._required("budget.lock.json")
        self._verified_development_lock(lock)
        budget = lock.get("locked_budget")
        source = self.backend.source_closure_sha256()
        if lock.get("source_closure_sha256") != source:
            raise RuntimeError("Warehouse selected source closure changed")
        if (lock.get("passed") is not True
                or budget not in p.BUDGETS
                or selection.get("locked_budget") != budget
                or selection.get("pair_seeds") != list(p.FORMAL_PAIR_SEEDS)
                or selection.get("selection_seeds") != list(p.FORMAL_SELECTION_SEEDS)
                or selection.get("selection_rule") !=
                "maximin_deliveries_minus_0.25_collisions_minus_5_shutdown"):
            raise RuntimeError("Warehouse formal selection or budget lock changed")
        rows = []
        for pair_seed in p.FORMAL_PAIR_SEEDS:
            candidate = _read(self.directory /
                              f"pair_{pair_seed}_steps_{budget}" /
                              "pair.training.json")
            row = self._required(f"formal.training.{pair_seed}.json")
            expected_schedule = p.final_program_fit_seed_schedule(
                self.release_id, pair_seed)
            if (not _verify_formal_row(row, candidate, source)
                    or candidate.get("candidate_id") != self.candidate_id
                    or candidate.get("release_id") != self.release_id
                    or candidate.get("pair_seed") != pair_seed
                    or candidate.get("joint_steps") != budget
                    or candidate.get("final_program_fit_seed_schedule") !=
                    expected_schedule
                    or candidate.get("final_program_fit_seed_schedule_sha256") !=
                    _sha(expected_schedule)
                    or not p.paired_training_pass(candidate.get("variants", {}),
                                                  joint_steps=budget)):
                raise RuntimeError("Warehouse formal pair evidence changed")
            rows.append(row)
        expected_quality = {str(row["pair_seed"]): {
            variant: row["variants"][variant]["quality"]
            for variant in p.VARIANTS} for row in rows}
        expected_hashes = {str(row["pair_seed"]): _file_sha(self._path(
            f"formal.training.{row['pair_seed']}.json")) for row in rows}
        if (selection.get("pair_quality") != expected_quality
                or selection.get("pair_receipt_sha256") != expected_hashes
                or p.select_formal_pair(rows)["pair_seed"] !=
                selection.get("selected_pair_seed")):
            raise RuntimeError("Warehouse formal quality selection changed")
        seed, budget = selection["selected_pair_seed"], selection["locked_budget"]
        path = (self.directory / f"pair_{seed}_steps_{budget}" /
                "pair.training.json")
        pair = _read(path)
        if pair.get("source_closure_sha256") != self.backend.source_closure_sha256():
            raise RuntimeError("Warehouse selected source closure changed")
        if (pair.get("candidate_id") != self.candidate_id
                or pair.get("pair_seed") != seed
                or pair.get("joint_steps") != budget
                or not p.paired_training_pass(pair.get("variants", {}),
                                              joint_steps=budget)):
            raise RuntimeError("Warehouse selected paired training report mismatch")
        selected_receipt = self._required(f"formal.training.{seed}.json")
        if not _verify_formal_row(selected_receipt, pair,
                                  self.backend.source_closure_sha256()):
            raise RuntimeError("Warehouse selected paired training report changed")
        return selection, pair

    def _verified_program_prerequisite(self, name: str, selection: Mapping,
                                       pair: Mapping, actor, programs: Mapping,
                                       seeds: Sequence[int], *,
                                       sealed: bool) -> dict:
        """Recheck the prior audit before spending a fresh sealed seed block."""
        receipt = self._required(name)
        program_hashes = {role: programs[role].sha256() for role in p.ROLES}
        if (receipt.get("passed") is not True
                or receipt.get("selected_pair_seed") !=
                selection["selected_pair_seed"]
                or receipt.get("actor_sha256") != actor.artifact_sha256
                or receipt.get("program_sha256") != program_hashes
                or receipt.get("seeds") != list(seeds)
                or not _audit_valid(receipt.get("audit"), actor, programs,
                                    seeds, development=False)):
            raise RuntimeError(f"Warehouse prerequisite program audit changed: {name}")
        trace_root = (self.directory / "traces" /
                      name.removesuffix(".json")).resolve()
        trace_bytes = {}
        for role in p.ROLES:
            trace_path = Path(receipt["audit"][role]["trace_path"]).resolve()
            if not trace_path.is_relative_to(trace_root):
                raise RuntimeError(f"Warehouse prerequisite trace escaped run: {name}")
            trace_bytes[role] = trace_path.read_bytes()
        try:
            trace_valid = self.backend.verify_program_trace(
                actor=actor, programs=programs,
                trace_bytes_by_role=trace_bytes, audits=receipt["audit"],
                seeds=seeds)
        except Exception as exc:
            raise RuntimeError(
                f"Warehouse prerequisite trace could not be verified: {name}") from exc
        if trace_valid is not True:
            raise RuntimeError(f"Warehouse prerequisite trace invalid: {name}")
        if sealed:
            claim_path = self._claim_path("final_program_audit")
            claim = _read(claim_path)
            expected_actor_hashes = {variant: pair["variants"][variant][
                "actor_artifact"]["sha256"] for variant in p.VARIANTS}
            if (claim.get("phase") != "final_program_audit"
                    or claim.get("candidate_id") != self.candidate_id
                    or claim.get("release_id") != self.release_id
                    or claim.get("actor_hashes") != expected_actor_hashes
                    or claim.get("seeds") != list(seeds)
                    or claim.get("seed_sha256") != _sha(list(seeds))
                    or receipt.get("one_use_claim_sha256") !=
                    _file_sha(claim_path)):
                raise RuntimeError("Warehouse final audit one-use claim changed")
        elif receipt.get("one_use_claim_sha256") is not None:
            raise RuntimeError("Warehouse preaudit unexpectedly has a sealed claim")
        return receipt

    def _run_program_audit(self, phase: str, seeds: Sequence[int],
                           *, sealed: bool) -> dict:
        self._ensure_open()
        name = f"{phase}.json"
        if self._path(name).exists():
            return self._required(name)
        selection, pair = self._selected_pair()
        if phase == "program.preaudit":
            if tuple(seeds) != p.FORMAL_PROGRAM_PREAUDIT_SEEDS:
                raise ValueError("Warehouse preaudit seed mismatch")
        actor = self._load_variant(pair, "regularized")
        programs = self._load_programs(pair, actor)
        self._verify_selected_assets(pair)
        if phase != "program.preaudit":
            self._verified_program_prerequisite(
                "program.preaudit.json", selection, pair, actor, programs,
                p.FORMAL_PROGRAM_PREAUDIT_SEEDS, sealed=False)
        claim_sha = None
        if sealed:
            claim_path = self._claim_path("final_program_audit")
            try:
                p.claim_once(claim_path, phase="final_program_audit",
                             candidate_id=self.candidate_id,
                             actor_hashes={variant: pair["variants"][variant][
                                 "actor_artifact"]["sha256"] for variant in p.VARIANTS},
                             seed_rows=seeds, release_id=self.release_id)
            except FileExistsError as exc:
                self._record_failure(phase, exc)
                self._close(phase, "sealed audit seeds were already claimed")
                raise
            claim_sha = _file_sha(claim_path)
        try:
            authorization_scope = (self.backend.sealed_evaluation(
                claim_path=claim_path, expected_claim_sha256=claim_sha)
                if sealed and isinstance(self.backend, NativeBackend) else nullcontext())
            with authorization_scope:
                audits = self.backend.audit_programs(
                    actor, programs, seeds,
                    trace_dir=self.directory / "traces" / phase)
            passed = _audit_valid(audits, actor, programs, seeds,
                                  development=False)
            receipt = self._new(name, {
                "selected_pair_seed": selection["selected_pair_seed"],
                "actor_sha256": actor.artifact_sha256,
                "program_sha256": {role: programs[role].sha256()
                                   for role in p.ROLES},
                "seeds": list(seeds), "audit": audits,
                "one_use_claim_sha256": claim_sha,
                "passed": passed})
            if not passed:
                self._close(phase, "program audit did not meet both-role gates")
            return receipt
        except Exception as exc:
            self._record_failure(phase, exc)
            if sealed:
                self._close(phase, "sealed program audit raised after claim")
            raise

    def preaudit(self) -> dict:
        return self._run_program_audit("program.preaudit",
                                       p.FORMAL_PROGRAM_PREAUDIT_SEEDS,
                                       sealed=False)

    def final_audit(self) -> dict:
        return self._run_program_audit("final_program_audit",
                                       p.final_audit_seeds(self.release_id),
                                       sealed=True)

    def heldout(self) -> dict:
        """Run exactly 100 fresh scenario seeds per task after a passing audit."""
        self._ensure_open()
        if self._path("heldout.json").exists():
            return self._required("heldout.json")
        selection, pair = self._selected_pair()
        actor = self._load_variant(pair, "regularized")
        programs = self._load_programs(pair, actor)
        self._verify_selected_assets(pair)
        self._verified_program_prerequisite(
            "program.preaudit.json", selection, pair, actor, programs,
            p.FORMAL_PROGRAM_PREAUDIT_SEEDS, sealed=False)
        self._verified_program_prerequisite(
            "final_program_audit.json", selection, pair, actor, programs,
            p.final_audit_seeds(self.release_id), sealed=True)
        schedule = p.heldout_schedule(self.release_id)
        claim_path = self._claim_path("heldout")
        try:
            p.claim_once(claim_path, phase="heldout", candidate_id=self.candidate_id,
                         actor_hashes={variant: pair["variants"][variant][
                             "actor_artifact"]["sha256"] for variant in p.VARIANTS},
                         seed_rows=p.heldout_base_seeds(self.release_id),
                         release_id=self.release_id)
        except FileExistsError as exc:
            self._record_failure("heldout", exc)
            self._close("heldout", "sealed heldout seeds were already claimed")
            raise
        claim_sha = _file_sha(claim_path)
        try:
            outcomes = {}
            authorization_scope = (self.backend.sealed_evaluation(
                claim_path=claim_path, expected_claim_sha256=claim_sha)
                if isinstance(self.backend, NativeBackend) else nullcontext())
            with authorization_scope:
                for variant in p.VARIANTS:
                    actor = self._load_variant(pair, variant)
                    tasks, all_episodes = {}, []
                    for task in p.TASKS:
                        seeds = schedule[task]
                        result = self.backend.evaluate(actor, seeds, trace=True)
                        episodes = _validated_episodes(
                            result, seeds, require_trace=True,
                            actor_sha256=actor.artifact_sha256)
                        tasks[str(task)] = {"summary": _summary(episodes),
                                            "episodes": episodes,
                                            "episodes_sha256": _sha(episodes)}
                        all_episodes.extend(episodes)
                    outcomes[variant] = {"summary": _summary(all_episodes),
                                         "actor_sha256": actor.artifact_sha256,
                                         "tasks": tasks,
                                         "episode_count": len(all_episodes)}
            passed = all(p.quality_pass(outcomes[variant]["summary"],
                                        development=False)
                         for variant in p.VARIANTS)
            receipt = self._new("heldout.json", {
                "selected_pair_seed": selection["selected_pair_seed"],
                "seed_rule": "base_seed_times_10_plus_task",
                "base_seeds": list(p.heldout_base_seeds(self.release_id)),
                "one_use_claim_sha256": claim_sha,
                "outcomes": outcomes, "passed": passed})
            if not passed:
                self._close("heldout", "heldout pooled quality gate failed")
            return receipt
        except Exception as exc:
            self._record_failure("heldout", exc)
            self._close("heldout", "sealed heldout raised after claim")
            raise

    def acceptance(self) -> dict:
        """Materialize a *passing* acceptance only from completed phase evidence."""
        self._ensure_open()
        if self._path("warehouse_v6_acceptance.json").exists():
            return self._required("warehouse_v6_acceptance.json")
        lock = self._required("budget.lock.json")
        selection, pair = self._selected_pair()
        preaudit = self._required("program.preaudit.json")
        final = self._required("final_program_audit.json")
        heldout = self._required("heldout.json")
        if not all(receipt.get("passed") is True
                   for receipt in (lock, selection, preaudit, final, heldout)):
            raise RuntimeError("Warehouse acceptance requires every phase to pass")
        if lock.get("source_closure_sha256") != self.backend.source_closure_sha256():
            raise RuntimeError("Warehouse acceptance source closure changed")
        final_schedules = {
            str(seed): p.final_program_fit_seed_schedule(self.release_id, seed)
            for seed in (*p.DEVELOPMENT_PAIR_SEEDS, *p.FORMAL_PAIR_SEEDS)}
        if (lock.get("final_program_fit_seed_schedule") != final_schedules
                or lock.get("final_program_fit_seed_schedule_sha256") !=
                _sha(final_schedules)
                or any(lock.get(key) != value for key, value in
                       p.registered_method_lock_fields(
                           self.program_fit_method).items())
                or lock.get("architecture_mode") !=
                p.REGISTERED_ARCHITECTURE_MODE
                or lock.get("bc_recipe_version") !=
                p.REGISTERED_BC_RECIPE_VERSION
                or lock.get("online_program_extraction_steps") !=
                list(p.online_program_extraction_steps(lock["locked_budget"]))
                or lock.get("topology_refresh_steps") !=
                list(p.topology_refresh_steps(lock["locked_budget"]))
                or lock.get("topology_frozen_at_step") !=
                p.topology_frozen_at_step(lock["locked_budget"])):
            raise RuntimeError("Warehouse frozen final-program seed schedule changed")
        development = {}
        for budget in p.BUDGETS:
            if budget > lock["locked_budget"]:
                break
            development[budget] = []
            for seed in p.DEVELOPMENT_PAIR_SEEDS:
                name = f"development.{budget}.{seed}.json"
                row = self._required(name)
                pair_path = (self.directory / f"pair_{seed}_steps_{budget}" /
                             "pair.training.json")
                pair_row = _read(pair_path)
                if (row.get("pair_seed") != seed or row.get("budget") != budget
                        or row.get("source_closure_sha256") !=
                        lock["source_closure_sha256"]
                        or row.get("pair_report_sha256") != _sha(pair_row)
                        or pair_row.get("release_id") != self.release_id
                        or pair_row.get("final_program_fit_seed_schedule") !=
                        final_schedules[str(seed)]
                        or pair_row.get("final_program_fit_seed_schedule_sha256") !=
                        _sha(final_schedules[str(seed)])
                        or row.get("training") != pair_row.get("variants")):
                    raise RuntimeError("Warehouse development receipt or source changed")
                if row.get("stress") and (set(row["stress"]) != set(p.VARIANTS)
                                          or not all(_verify_stress(
                                              row["stress"][variant])
                                              for variant in p.VARIANTS)):
                    raise RuntimeError("Warehouse development pressure rows changed")
                computed = p.development_pair_pass(row)
                if computed:
                    actor_row = self._load_variant(pair_row, "regularized")
                    try:
                        programs_row = self._load_programs(pair_row, actor_row)
                    except RuntimeError as exc:
                        if "lacks both role programs" not in str(exc):
                            raise
                        computed = False
                    else:
                        computed = _audit_valid(
                            row["program_audit"], actor_row, programs_row,
                            p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS,
                            development=True)
                if row.get("passed") is not computed:
                    raise RuntimeError("Warehouse development gate result changed")
                development[budget].append(row)
        first_passing = next((budget for budget in p.BUDGETS
                              if budget in development
                              and len(development[budget]) == len(
                                  p.DEVELOPMENT_PAIR_SEEDS)
                              and all(row["passed"] for row in development[budget])),
                             None)
        if first_passing != lock["locked_budget"]:
            raise RuntimeError("Warehouse locked budget is not the smallest passing budget")
        if (lock.get("development_receipt_sha256") != {
                str(seed): _file_sha(self._path(
                    f"development.{lock['locked_budget']}.{seed}.json"))
                for seed in p.DEVELOPMENT_PAIR_SEEDS}):
            raise RuntimeError("Warehouse budget lock development hashes changed")
        training = []
        formal_rows = []
        for pair_seed in p.FORMAL_PAIR_SEEDS:
            path = (self.directory /
                    f"pair_{pair_seed}_steps_{lock['locked_budget']}" /
                    "pair.training.json")
            candidate = _read(path)
            receipt = self._required(f"formal.training.{pair_seed}.json")
            if (not _verify_formal_row(receipt, candidate,
                                       lock["source_closure_sha256"])
                    or candidate.get("release_id") != self.release_id
                    or candidate.get("final_program_fit_seed_schedule") !=
                    final_schedules[str(pair_seed)]
                    or candidate.get("final_program_fit_seed_schedule_sha256") !=
                    _sha(final_schedules[str(pair_seed)])
                    or candidate.get("source_closure_sha256") !=
                    lock["source_closure_sha256"]
                    or not p.paired_training_pass(candidate["variants"],
                                                  joint_steps=lock["locked_budget"])):
                raise RuntimeError("Warehouse formal pair training evidence changed")
            formal_rows.append(receipt)
            training.append({
                "pair_seed": pair_seed, "joint_steps": lock["locked_budget"],
                "pair_report_path": str(path),
                "pair_report_sha256": _file_sha(path),
                "initial_role_sha256": candidate["initial_role_sha256"],
                "variants": {variant: {
                    key: candidate["variants"][variant][key] for key in (
                        "joint_steps", "actor_decisions", "scenario_schedule_sha256",
                        "initial_role_sha256", "selected_role_sha256",
                        "role_weight_change_norm", "feedback_rows",
                        "feedback_gradient_norm", "post_policy_interventions",
                        "physical_no_legal_action_rows", "training_report_sha256")}
                    | {"training_report_path": candidate["variants"][variant][
                        "training_report_path"]}
                    for variant in p.VARIANTS},
            })
        if p.select_formal_pair(formal_rows)["pair_seed"] != selection[
                "selected_pair_seed"]:
            raise RuntimeError("Warehouse pair choice differs from fixed quality selection")
        if (selection.get("locked_budget") != lock["locked_budget"]
                or selection.get("pair_seeds") != list(p.FORMAL_PAIR_SEEDS)
                or selection.get("selection_seeds") != list(p.FORMAL_SELECTION_SEEDS)
                or selection.get("pair_quality") != {
                    str(row["pair_seed"]): {variant: row["variants"][variant]["quality"]
                                             for variant in p.VARIANTS}
                    for row in formal_rows}
                or selection.get("pair_receipt_sha256") != {
                    str(row["pair_seed"]): _file_sha(self._path(
                        f"formal.training.{row['pair_seed']}.json"))
                    for row in formal_rows}):
            raise RuntimeError("Warehouse formal selection receipt changed")
        artifacts = {variant: pair["variants"][variant]["actor_artifact"]
                     for variant in p.VARIANTS}
        if (artifacts["raw"].get("model_version") !=
                artifacts["regularized"].get("model_version")
                or artifacts["raw"].get("feature_schema_sha256") !=
                artifacts["regularized"].get("feature_schema_sha256")
                or artifacts["raw"].get("action_physics_config_sha256") !=
                p.REGISTERED_BC_SPEC["action_physics_config_sha256"]
                or artifacts["regularized"].get("action_physics_config_sha256") !=
                p.REGISTERED_BC_SPEC["action_physics_config_sha256"]):
            raise RuntimeError("Warehouse paired Actor architecture differs")
        for variant in p.VARIANTS:
            self._load_variant(pair, variant)
            report_path = pair["variants"][variant]["training_report_path"]
            if (not Path(report_path).resolve().is_relative_to(self.directory)
                    or _file_sha(report_path) != pair["variants"][variant][
                        "training_report_sha256"]):
                raise RuntimeError("Warehouse selected training report hash mismatch")
        actor = self._load_variant(pair, "regularized")
        programs = self._load_programs(pair, actor)
        if (preaudit.get("selected_pair_seed") != selection["selected_pair_seed"]
                or final.get("selected_pair_seed") != selection["selected_pair_seed"]
                or preaudit.get("actor_sha256") != actor.artifact_sha256
                or final.get("actor_sha256") != actor.artifact_sha256
                or preaudit.get("seeds") != list(p.FORMAL_PROGRAM_PREAUDIT_SEEDS)
                or final.get("seeds") != list(p.final_audit_seeds(self.release_id))
                or any(receipt.get("program_sha256") != {
                    role: programs[role].sha256() for role in p.ROLES}
                    for receipt in (preaudit, final))
                or not _audit_valid(final["audit"], actor, programs,
                             p.final_audit_seeds(self.release_id), development=False)
                or not _audit_valid(preaudit["audit"], actor, programs,
                                    p.FORMAL_PROGRAM_PREAUDIT_SEEDS,
                                    development=False)):
            raise RuntimeError("Warehouse acceptance program evidence mismatch")
        claim_sha = {}
        for phase in ("final_program_audit", "heldout"):
            path = self._claim_path(phase)
            claim = _read(path)
            expected_seeds = (p.final_audit_seeds(self.release_id)
                              if phase == "final_program_audit" else
                              p.heldout_base_seeds(self.release_id))
            if (claim.get("candidate_id") != self.candidate_id
                    or claim.get("release_id") != self.release_id
                    or claim.get("phase") != phase
                    or claim.get("seeds") != list(expected_seeds)
                    or claim.get("seed_sha256") != _sha(list(expected_seeds))
                    or claim.get("actor_hashes") != {
                        variant: artifacts[variant]["sha256"] for variant in p.VARIANTS}
                    or _file_sha(path) != self._required(
                        f"{phase}.json")["one_use_claim_sha256"]):
                raise RuntimeError("Warehouse one-use claim differs from phase receipt")
            claim_sha[phase] = _file_sha(path)
        if (heldout.get("selected_pair_seed") != selection["selected_pair_seed"]
                or heldout.get("base_seeds") !=
                list(p.heldout_base_seeds(self.release_id))
                or heldout.get("seed_rule") != "base_seed_times_10_plus_task"
                or not _verify_heldout(heldout["outcomes"], self.release_id)):
            raise RuntimeError("Warehouse heldout raw rows or seed schedule changed")
        if any(not p.quality_pass(heldout["outcomes"][variant]["summary"],
                                  development=False) for variant in p.VARIANTS):
            raise RuntimeError("Warehouse heldout outcome no longer passes")
        acceptance = {
            "release": self.release_id, "release_id": self.release_id,
            "domain": "warehouse",
            "locked_budget": lock["locked_budget"],
            "topology_refresh_steps": lock["topology_refresh_steps"],
            "topology_frozen_at_step": lock["topology_frozen_at_step"],
            "program_fit_method": lock["program_fit_method"],
            "source_closure_sha256": lock["source_closure_sha256"],
            "development_receipt_path": {
                str(budget): {str(seed): str(self._path(
                    f"development.{budget}.{seed}.json"))
                    for seed in p.DEVELOPMENT_PAIR_SEEDS}
                for budget in p.BUDGETS if budget <= lock["locked_budget"]},
            "development_receipt_sha256": {
                str(budget): {str(seed): _file_sha(self._path(
                    f"development.{budget}.{seed}.json"))
                    for seed in p.DEVELOPMENT_PAIR_SEEDS}
                for budget in p.BUDGETS if budget <= lock["locked_budget"]},
            "pair_seeds": list(p.FORMAL_PAIR_SEEDS),
            "selected_pair_seed": selection["selected_pair_seed"],
            "candidate_training": training,
            "joint_steps": {variant: lock["locked_budget"]
                            for variant in p.VARIANTS},
            "artifact_sha256": {variant: artifacts[variant]["sha256"]
                                for variant in p.VARIANTS},
            "role_sha256": {variant: artifacts[variant]["role_sha256"]
                            for variant in p.VARIANTS},
            "program_sha256": {"regularized": {
                role: programs[role].sha256() for role in p.ROLES}},
            "training_report_sha256": {variant: pair["variants"][variant][
                "training_report_sha256"] for variant in p.VARIANTS},
            "raw_program_feedback_rows": pair["variants"]["raw"]["feedback_rows"],
            "regularized_program_feedback_rows": pair["variants"][
                "regularized"]["feedback_rows"],
            "all_candidate_regularized_feedback_passed": True,
            "post_policy_interventions": 0,
            "post_policy_override_count": 0,
            "final_program_audit": final["audit"],
            "heldout": heldout["outcomes"],
            "heldout_evaluated": True,
            "heldout_scenario_seed_rule": "base_seed_times_10_plus_task",
            "heldout_seed_ranges": {str(task): [
                p.heldout_base_seeds(self.release_id)[0],
                p.heldout_base_seeds(self.release_id)[-1]] for task in p.TASKS},
            "one_use_claim_sha256": claim_sha,
            "one_use_claim_path": {
                phase: str(self._claim_path(phase))
                for phase in ("final_program_audit", "heldout")},
            "phase_receipt_sha256": {name: _file_sha(self._path(name))
                                     for name in ("budget.lock.json",
                                                  "formal.selection.json",
                                                  "program.preaudit.json",
                                                  "final_program_audit.json",
                                                  "heldout.json",
                                                  *(f"formal.training.{seed}.json"
                                                    for seed in p.FORMAL_PAIR_SEEDS))},
            "phase_receipt_path": {name: str(self._path(name))
                                   for name in ("budget.lock.json",
                                                "formal.selection.json",
                                                "program.preaudit.json",
                                                "final_program_audit.json",
                                                "heldout.json",
                                                *(f"formal.training.{seed}.json"
                                                  for seed in p.FORMAL_PAIR_SEEDS))},
            "policy_id": self.policy_ids,
            "actor_model_version": artifacts["regularized"]["model_version"],
            "feature_schema_sha256": artifacts["regularized"][
                "feature_schema_sha256"],
            "action_physics_config_sha256": artifacts["regularized"][
                "action_physics_config_sha256"],
            "passed": True,
        }
        return self._new("warehouse_v6_acceptance.json", acceptance)


def main() -> None:
    parser = argparse.ArgumentParser(description="Warehouse v6 gated evidence phases")
    parser.add_argument("phase", choices=("develop", "formal-train", "preaudit",
                                         "final-audit", "heldout", "acceptance"))
    parser.add_argument("--output-root", required=True,
                        help="isolated run root; preserve it with the release package")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--release-id", default=p.INITIAL_RELEASE_ID)
    parser.add_argument("--program-fit-method", required=True,
                        choices=p.PROGRAM_FIT_METHOD_CANDIDATES)
    args = parser.parse_args()
    admission = Admission(args.output_root, args.candidate_id,
                          release_id=args.release_id,
                          program_fit_method=args.program_fit_method)
    phase = args.phase.replace("-", "_")
    result = getattr(admission, phase)()
    print(json.dumps({"phase": args.phase,
                      "release_id": args.release_id,
                      "candidate_id": args.candidate_id,
                      "passed": result.get("passed", False),
                      "receipt": str(admission._path({
                          "develop": "budget.lock.json" if result.get("passed") else
                                     "development.failed.json",
                          "formal-train": "formal.selection.json",
                          "preaudit": "program.preaudit.json",
                          "final-audit": "final_program_audit.json",
                          "heldout": "heldout.json",
                          "acceptance": "warehouse_v6_acceptance.json",
                      }[args.phase]))}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
