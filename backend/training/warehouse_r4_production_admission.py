"""Fail-closed semantic admission for the warehouse r4 internal pilot.

This module does not run training, fit a program, or manufacture acceptance
measurements.  It only recomputes the release gates from immutable reports
written by the real r4 producers.  A report's own ``passed`` boolean is never
enough: the underlying counts, thresholds, component identities, and action
authority invariants are checked again here.

The resulting admission is intentionally limited to an internal pilot.  The
post-package HTTP, browser, restart, and Render checks remain deployment
checks; they cannot be used to turn this record into formal-study evidence.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Mapping, Sequence


VERSION = "warehouse-r4-production-admission-validator.v1"
ADMISSION_VERSION = "warehouse-r4-local-pilot-admission.v1"
R3_ACTOR_SHA256 = "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b"
TRAINING_LEDGER_VERSION = "warehouse-r4-training-budget-ledger.v2"
ACTIVE_VERSION = "warehouse-r4-active-evaluation.v1"
RUNTIME_VERSION = "warehouse-r4-active-runtime-bundle.v1"
FINAL_RCPD_VERSION = "warehouse-r4-final-rcpd.v1"
SCENE_VERSION = "warehouse-r4-high-conflict-play-selection.v3"
EXPLANATION_VERSION = "warehouse-r4-online-explanation-audit.v1"
QUESTION_VERSION = "warehouse-r4-frozen-question-bank.v1"
QUESTION_PAYLOAD_VERSION = "warehouse-alignment-portable-question-projection.v1"

REPORT_NAMES = (
    "training_budget",
    "validation_scenarios",
    "active_policy",
    "runtime_components",
    "high_conflict_scenes",
    "explanation_program",
    "questionnaire",
)
GATE_NAMES = (
    "active_policy", "action_authority", "high_conflict_scenes",
    "explanation_program", "questionnaire", "online_runtime",
)
PARTNERS = ("skilled", "assertive", "noisy", "fixed_yield", "fixed_region", "fixed_task")
ABSOLUTE_CHECKS = frozenset((
    "active_non_wait", "productive_action", "full_battery_non_charger_wait",
    "first_productive_median", "first_productive_p90", "ai_delivery_share",
    "no_task_progress_streak_p95", "collision_increase",
    "mean_longest_collision_streak", "static_wall_command", "shutdowns",
    "action_authority",
))
RELATIVE_CHECKS = frozenset((
    "active_non_wait_improved", "productive_action_improved_8pp",
    "ai_deliveries_improved_15pct", "ai_deliveries_improved_one",
    "no_progress_p95_improved",
))
CRITICAL_GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
SCENE_GEOMETRY_CHECKS = frozenset((
    "route_overlap_band", "shared_bottleneck_or_intersection",
    "opposing_and_same_direction_opportunities",
))
SCENE_DYNAMIC_CHECKS = frozenset((
    "conflict_opportunity_25_to_40_percent",
    "player_risky_action_mass_10_to_20_percent",
    "collision_cancellation_8_to_18_percent", "simple_baseline_gap",
    "collision_recovery_at_least_90_percent",
    "no_pathological_zero_delivery_collision_episode",
    "no_persistent_collision_deadlock", "no_persistent_no_progress_deadlock",
    "actor_actions_submitted_unchanged",
))
SCENE_SIMPLE_BASELINES = (
    "right_of_way", "region_spillover", "task_spillover",
    "leader_follower", "fixed_yield",
)
ABSOLUTE_GATE = {
    "active_non_wait_rate_min": .85,
    "productive_action_rate_min": .65,
    "full_battery_non_charger_wait_rate_max": .05,
    "first_productive_latency_median_max": 2.,
    "first_productive_latency_p90_max": 5.,
    "ai_delivery_share_min": .35,
    "no_task_progress_streak_p95_max": 18.,
    "mean_longest_collision_streak_max": 3.,
    "static_wall_command_rate_max": .01,
    "shutdown_count_max": 0,
    "action_override_count_max": 0,
}
RELATIVE_GATE = {
    "productive_action_rate_improvement_min": .08,
    "ai_deliveries_relative_improvement_min": .15,
    "ai_deliveries_absolute_improvement_min": 1.,
    "collision_cancellation_rate_increase_max": .02,
    "minimum_checks_passed": 4,
}
ACTIVE_EPISODE_FIELDS = frozenset((
    "scenario_id", "scenario_fingerprint", "partner", "seed", "steps",
    "active_frames", "active_non_wait", "productive_actions", "action_counts",
    "charge_needed_frames", "charge_needed_waits", "charge_needed_nonproductive",
    "full_noncharger_frames", "full_noncharger_waits", "first_productive_latency",
    "ai_deliveries", "team_deliveries", "ai_delivery_share", "collision_steps",
    "collision_cancellations", "longest_collision_streak",
    "longest_no_task_progress_streak", "static_wall_commands", "shutdowns",
    "ai_shutdown", "player_shutdown", "ai_active_end", "policy_actions",
    "submitted_actions", "action_equal", "action_overrides", "terminal_reason",
))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
ROOT = Path(__file__).resolve().parents[2]


def _module_files(parts: Sequence[str]) -> list[Path]:
    if not parts or any(not part or part in (".", "..") for part in parts):
        return []
    base = ROOT.joinpath(*parts)
    result = []
    source = base.with_suffix(".py")
    package = base / "__init__.py"
    if source.is_file():
        result.append(source)
    if package.is_file():
        result.append(package)
    return result


def local_source_hashes(seed_paths: Sequence[str | Path], *,
                        exclude_paths: Sequence[str | Path] = ()) -> dict[str, str]:
    """Hash the exact recursive local-Python import closure plus explicit assets.

    Static imports are resolved inside the repository only.  Explicit non-Python
    paths are included as leaf assets.  Package ``__init__`` files are included
    because Python executes them while importing a child module.
    """
    excluded = {Path(path).resolve() for path in exclude_paths}
    queue = []
    for value in seed_paths:
        supplied = Path(value)
        if supplied.is_symlink():
            raise ValueError("R4 source closure cannot contain a symlink")
        queue.append(supplied.resolve())
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen or path in excluded:
            continue
        if (path.is_symlink() or not path.is_file()
                or (path != ROOT and ROOT not in path.parents)):
            raise ValueError("R4 source closure contains a missing or external path")
        seen.add(path)
        if path.suffix != ".py":
            continue
        relative = path.relative_to(ROOT)
        module_parts = list(relative.with_suffix("").parts)
        package_parts = module_parts[:-1]
        if module_parts[-1] == "__init__":
            package_parts = module_parts[:-1]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (SyntaxError, UnicodeError) as exc:
            raise ValueError("Cannot parse r4 source closure file: " + str(relative)) from exc
        imported: list[list[str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = len(package_parts) - (node.level - 1)
                    if keep < 0:
                        continue
                    base = package_parts[:keep]
                else:
                    base = []
                if node.module:
                    base = [*base, *node.module.split(".")]
                if base:
                    imported.append(base)
                for alias in node.names:
                    if alias.name != "*":
                        imported.append([*base, *alias.name.split(".")])
        for parts in imported:
            files = _module_files(parts)
            for imported_path in files:
                if imported_path.is_symlink():
                    raise ValueError("R4 source closure cannot contain a symlink")
                if imported_path not in seen and imported_path not in excluded:
                    queue.append(imported_path)
            # Importing a child executes every local package initializer.
            for index in range(1, len(parts)):
                initializer = ROOT.joinpath(*parts[:index], "__init__.py")
                if initializer.is_symlink():
                    raise ValueError("R4 source closure cannot contain a symlink")
                if (initializer.is_file() and initializer not in seen
                        and initializer not in excluded):
                    queue.append(initializer)
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(seen)}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _strict_json_object(raw: str, label: str) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                _fail("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid JSON in " + label) from error
    if not isinstance(value, dict):
        _fail("JSON object required in " + label)
    return value


def _fail(message: str) -> None:
    raise ValueError(message)


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        _fail(f"Invalid {name} SHA-256")
    return value


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        _fail(f"{name} must be finite")
    return float(value)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _all_true(value: Any, expected: set[str] | frozenset[str] | None, name: str) -> bool:
    if not isinstance(value, dict) or (expected is not None and set(value) != set(expected)):
        _fail(f"{name} check matrix differs")
    if not value or any(item is not True for item in value.values()):
        _fail(f"{name} contains a failed or non-boolean check")
    return True


def _stat(value: Any, name: str, threshold: float, minimum_scenes: int = 1) -> None:
    if not isinstance(value, dict) or set(value) != {"rows", "scenes", "fidelity"}:
        _fail(f"{name} statistic schema differs")
    rows = _positive_int(value["rows"], f"{name}.rows")
    scenes = _positive_int(value["scenes"], f"{name}.scenes")
    fidelity = _finite(value["fidelity"], f"{name}.fidelity")
    if scenes > rows or scenes < minimum_scenes or not threshold <= fidelity <= 1.0:
        _fail(f"{name} does not meet its registered threshold")


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / ((left + right) / 2.) if left + right else 0.


def _validate_scenarios(scenarios: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    if not isinstance(scenarios, dict) or scenarios.get("version") != "warehouse-native-physical-splits-v1":
        _fail("Registered warehouse scenario manifest required")
    splits, counts = scenarios.get("splits"), scenarios.get("counts")
    required = {"train": 512, "calibration": 100, "validation": 50,
                "extraction": 100, "explanation_test": 100,
                "final_test": 100, "play": 12}
    if not isinstance(splits, dict) or set(splits) != set(required):
        _fail("Warehouse scenario split set differs")
    actual = {name: len(rows) if isinstance(rows, list) else -1 for name, rows in splits.items()}
    if actual != required or counts != required:
        _fail("Warehouse scenario split counts differ")
    fingerprints: set[str] = set()
    for name, rows in splits.items():
        ids = set()
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("id")) is not str
                    or row["id"] in ids or _HEX.fullmatch(str(row.get("fingerprint", ""))) is None
                    or not isinstance(row.get("snapshot"), dict)):
                _fail(f"Invalid or duplicate scenario in {name}")
            ids.add(row["id"])
            if row["fingerprint"] in fingerprints:
                _fail("Scenario physical fingerprints overlap across registered splits")
            fingerprints.add(row["fingerprint"])
    if context["scenario_manifest_semantic_sha256"] != context["actor_scenario_manifest_sha256"]:
        _fail("Final Actor belongs to another scenario manifest")


def _validate_training_ledger(report: Mapping[str, Any], context: Mapping[str, Any],
                              report_path: str | Path, report_sha256: str,
                              active_report_sha256: str) -> None:
    """Reopen every attempt, boundary, checkpoint hash, log, and DAG edge.

    Continuation counters include their parent lineage, so the admission must
    never trust a caller-written ``fresh`` number.  The v2 reader rebuilds the
    ledger from the complete run inventory and counts authenticated 50k edge
    deltas by unique segment hash.
    """
    if (not isinstance(report, dict) or report.get("version") != TRAINING_LEDGER_VERSION
            or report.get("status") != "complete" or report.get("test_fixture") is not False
            or report.get("formal_ready") is not False
            or report.get("admission_eligible") is not True
            or report.get("failure_reason") is not None):
        _fail("Exact complete r4 PPO budget ledger required")
    source, selection, invariants = (report.get("source_r3"), report.get("selection"),
                                     report.get("invariants"))
    total = _positive_int(report.get("total_actual_additional_joint_steps"),
                          "authenticated r4 PPO steps")
    if (not isinstance(source, dict) or source.get("actor_sha256") != R3_ACTOR_SHA256
            or report.get("maximum_additional_joint_steps") != 1_000_000
            or total > 1_000_000
            or report.get("remaining_additional_joint_steps") != 1_000_000 - total
            or not isinstance(report.get("attempts"), list) or not report["attempts"]
            or not isinstance(selection, dict)
            or selection.get("training_actor_sha256") != context["training_actor_sha256"]
            or selection.get("active_policy_report_sha256") != active_report_sha256
            or selection.get("earliest_full_pass") is not True
            or not isinstance(invariants, dict) or not invariants
            or any(value is not True for value in invariants.values())):
        _fail("R4 PPO budget, source, DAG, or selected-Actor summary differs")
    from backend.training import warehouse_r4_training_ledger as producer
    recomputed = producer.read_saved_ledger(
        report_path, expected_report_sha256=report_sha256,
        expected_selected_actor_sha256=context["training_actor_sha256"],
        expected_active_report_sha256=active_report_sha256,
        expected_scenario_manifest_sha256=context["scenario_manifest_file_sha256"],
    )
    if digest(recomputed) != digest(report):
        _fail("R4 PPO budget ledger differs from immutable attempt evidence")


def _validate_active(report: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    fields = {"version", "status", "selected", "absolute_gate", "relative_gate",
              "absolute_checks", "relative_checks", "relative_checks_passed",
              "baseline", "candidate", "scenario_manifest", "definitions"}
    if (not isinstance(report, dict) or set(report) != fields
            or report.get("version") != ACTIVE_VERSION
            or report.get("status") != "passed" or report.get("selected") is not True):
        _fail("R4 paired active-policy audit did not pass")
    if report.get("absolute_gate") != ABSOLUTE_GATE or report.get("relative_gate") != RELATIVE_GATE:
        _fail("R4 active-policy registered thresholds changed")
    baseline, candidate = report.get("baseline"), report.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        _fail("R4 paired audit lacks both Actor results")
    if (baseline.get("actor", {}).get("artifact_sha256") != R3_ACTOR_SHA256
            or candidate.get("actor", {}).get("artifact_sha256") != context["training_actor_sha256"]
            or baseline.get("actor", {}).get("runtime_action_override") is not False
            or candidate.get("actor", {}).get("runtime_action_override") is not False
            or baseline.get("partners") != list(PARTNERS)
            or candidate.get("partners") != list(PARTNERS)):
        _fail("R4 paired audit Actor or partner binding differs")
    validation_sha = context["validation_entries_sha256"]
    for name, result in (("baseline", baseline), ("candidate", candidate)):
        summary = result.get("summary")
        if (result.get("validation_entries_sha256") != validation_sha
                or not isinstance(summary, dict) or summary.get("episodes") != 300
                or summary.get("action_override_count") != 0
                or summary.get("policy_action_equality_rate") != 1.0):
            _fail(f"R4 {name} audit is incomplete or contains an action override")
        for key, value in summary.items():
            if type(value) in (int, float) and not isinstance(value, bool):
                _finite(value, f"{name}.{key}")
    before, after = baseline["summary"], candidate["summary"]
    absolute = {
        "active_non_wait": after.get("active_non_wait_rate", -1) >= .85,
        "productive_action": after.get("productive_action_rate", -1) >= .65,
        "full_battery_non_charger_wait": after.get("full_battery_non_charger_wait_rate", 2) <= .05,
        "first_productive_median": after.get("first_productive_latency_median", math.inf) <= 2.,
        "first_productive_p90": after.get("first_productive_latency_p90", math.inf) <= 5.,
        "ai_delivery_share": after.get("ai_delivery_share", -1) >= .35,
        "no_task_progress_streak_p95": after.get("no_task_progress_streak_p95", math.inf) <= 18.,
        "collision_increase": after.get("collision_cancellation_rate", math.inf)
            <= before.get("collision_cancellation_rate", -math.inf) + .02,
        "mean_longest_collision_streak": after.get("mean_longest_collision_streak", math.inf) <= 3.,
        "static_wall_command": after.get("static_wall_command_rate", math.inf) <= .01,
        "shutdowns": after.get("ai_shutdown_count", math.inf) <= 0,
        "action_authority": after.get("action_override_count", math.inf) <= 0,
    }
    relative = {
        "active_non_wait_improved": after.get("active_non_wait_rate", -math.inf)
            > before.get("active_non_wait_rate", math.inf),
        "productive_action_improved_8pp": after.get("productive_action_rate", -math.inf)
            >= before.get("productive_action_rate", math.inf) + .08,
        "ai_deliveries_improved_15pct": after.get("mean_ai_deliveries", -math.inf)
            >= before.get("mean_ai_deliveries", math.inf) * 1.15,
        "ai_deliveries_improved_one": after.get("mean_ai_deliveries", -math.inf)
            >= before.get("mean_ai_deliveries", math.inf) + 1.,
        "no_progress_p95_improved": after.get("no_task_progress_streak_p95", math.inf)
            < before.get("no_task_progress_streak_p95", -math.inf),
    }
    if report.get("absolute_checks") != absolute or not all(absolute.values()):
        _fail("R4 active-policy absolute checks differ from reported metrics")
    if (report.get("relative_checks") != relative
            or report.get("relative_checks_passed") != sum(relative.values())
            or sum(relative.values()) < 4):
        _fail("R4 active-policy relative gate did not pass four of five recomputed checks")
    scenario = report.get("scenario_manifest")
    if (not isinstance(scenario, dict)
            or scenario.get("sha256") != context["scenario_manifest_file_sha256"]
            or scenario.get("validation_entries_sha256") != validation_sha):
        _fail("R4 paired audit used another validation manifest")


def _active_rows(path: Path) -> list[dict[str, Any]]:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > 4 * 1024 * 1024):
        _fail("R4 paired audit row evidence is missing or oversized")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = _strict_json_object(line, "paired audit evidence")
        if not isinstance(value, dict) or set(value) != ACTIVE_EPISODE_FIELDS:
            _fail("R4 paired audit episode schema differs")
        rows.append(value)
    return rows


def _validate_active_evidence(report: Mapping[str, Any], report_path: str | Path,
                              validation_scenes: Sequence[Mapping[str, Any]]) -> None:
    """Recompute both 300-episode summaries from the producer's raw rows."""
    from backend.training import warehouse_r4_active_evaluation as producer

    supplied = Path(report_path)
    if supplied.is_symlink():
        _fail("R4 paired audit report cannot be a symlink")
    report_path = supplied.resolve()
    root = report_path.parent
    if report_path.name != "paired_report.json" or not report_path.is_file():
        _fail("R4 paired audit requires its canonical evidence directory")
    scene_map = {scene.get("fingerprint"): scene.get("id") for scene in validation_scenes}
    if len(scene_map) != 50 or any(_HEX.fullmatch(str(key)) is None for key in scene_map):
        _fail("R4 paired audit validation scene matrix differs")
    for key, directory in (("baseline", "r3"), ("candidate", "candidate")):
        result = report[key]
        row_path, nested_path = root / directory / "episodes.jsonl", root / directory / "report.json"
        if (row_path.parent != root / directory or nested_path.parent != root / directory
                or not nested_path.is_file() or nested_path.is_symlink()):
            _fail("R4 paired audit evidence paths differ")
        nested = _strict_json_object(
            nested_path.read_text(encoding="utf-8"), "paired audit report")
        if digest(nested) != digest(result):
            _fail("R4 paired audit embedded Actor report differs from its saved report")
        rows = _active_rows(row_path)
        if len(rows) != 300:
            _fail("R4 paired audit requires 300 complete rows per Actor")
        expected = {
            (fingerprint, partner): (identifier, result["seed"] + partner_index * 10_000 + scene_index)
            for partner_index, partner in enumerate(PARTNERS)
            for scene_index, (fingerprint, identifier) in enumerate(scene_map.items())
        }
        actual = {(row["scenario_fingerprint"], row["partner"]): row for row in rows}
        if len(actual) != 300 or set(actual) != set(expected):
            _fail("R4 paired audit scene/partner episode matrix differs")
        for identity, row in actual.items():
            identifier, seed = expected[identity]
            integer_fields = ACTIVE_EPISODE_FIELDS - {
                "scenario_id", "scenario_fingerprint", "partner", "action_counts",
                "ai_delivery_share", "ai_active_end", "terminal_reason",
            }
            if (row["scenario_id"] != identifier or row["seed"] != seed
                    or row["partner"] not in PARTNERS
                    or any(type(row[name]) is not int or row[name] < 0 for name in integer_fields)
                    or not 1 <= row["steps"] <= 120
                    or set(row["action_counts"]) != {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}
                    or any(type(value) is not int or value < 0
                           for value in row["action_counts"].values())
                    or sum(row["action_counts"].values()) != row["active_frames"]
                    or not (0 <= row["active_non_wait"] <= row["active_frames"])
                    or not (0 <= row["productive_actions"] <= row["active_frames"])
                    or not (0 <= row["full_noncharger_waits"] <= row["full_noncharger_frames"])
                    or row["policy_actions"] != row["steps"]
                    or row["submitted_actions"] != row["steps"]
                    or row["action_equal"] != row["steps"]
                    or row["action_overrides"] != 0
                    or type(row["ai_active_end"]) is not bool
                    or row["ai_shutdown"] != int(not row["ai_active_end"])
                    or not math.isclose(float(row["ai_delivery_share"]),
                        row["ai_deliveries"] / max(1, row["team_deliveries"]),
                        rel_tol=0., abs_tol=1e-12)):
                _fail("R4 paired audit episode arithmetic or authority differs")
        recomputed = producer.summarize(rows)
        if digest(recomputed) != digest(result.get("summary")):
            _fail("R4 paired audit summary differs from immutable episode rows")


def _bound_r4_evidence_path(value: Any, label: str, expected_sha256: str) -> Path:
    if type(value) is not str or not value:
        _fail(label + " path is missing")
    supplied = Path(value)
    if supplied.is_symlink():
        _fail(label + " cannot be a symlink")
    path = supplied.resolve()
    if (not path.is_file() or ROOT not in path.parents
            or _HEX.fullmatch(str(expected_sha256)) is None
            or file_hash(path) != expected_sha256):
        _fail(label + " is missing, outside the repository, or changed")
    return path


def _validate_final_rcpd_evidence(final: Mapping[str, Any],
                                  context: Mapping[str, Any]) -> None:
    """Reopen the post-freeze rows and recompute the selected program.

    The runtime receipt is a useful producer record, but its passing fields
    are not an admission boundary by themselves.  Keep the original training
    Actor, registered scenario manifest, candidate programs and NPZ rows live
    through admission and replay them with the final-RCPD semantic reader.
    """
    actor_path = _bound_r4_evidence_path(
        final.get("source_training_actor_path"), "Final-RCPD training Actor",
        final.get("source_training_actor_sha256"))
    scenarios_path = _bound_r4_evidence_path(
        final.get("scenario_manifest_path"), "Final-RCPD scenario manifest",
        final.get("scenario_manifest_sha256"))
    report_path = _bound_r4_evidence_path(
        final.get("path"), "Final-RCPD report", final.get("sha256"))
    if (report_path.name != "report.json"
            or final.get("scenario_manifest_sha256")
                != context["scenario_manifest_file_sha256"]):
        _fail("Final-RCPD evidence layout or scenario binding differs")
    source_program = report_path.parent / "program.json"
    if (source_program.is_symlink() or not source_program.is_file()
            or file_hash(source_program) != final.get("program_source_sha256")):
        _fail("Final-RCPD selected source program changed")
    before = {
        "actor": file_hash(actor_path), "scenarios": file_hash(scenarios_path),
        "report": file_hash(report_path), "program": file_hash(source_program),
    }
    from backend.training import warehouse_r4_final_rcpd as producer
    recomputed = producer.read_saved_report(
        report_path.parent, expected_report_sha256=final["sha256"],
        actor_path=actor_path, scenarios_path=scenarios_path,
        program_path=source_program, require_passed=True,
    )
    after = {
        "actor": file_hash(actor_path), "scenarios": file_hash(scenarios_path),
        "report": file_hash(report_path), "program": file_hash(source_program),
    }
    if (before != after
            or recomputed.get("final_rcpd_binding_sha256") != final.get("binding_sha256")
            or recomputed.get("program_file_sha256") != final.get("program_source_sha256")
            or recomputed.get("bindings", {}).get("actor_file_sha256")
                != final.get("source_training_actor_sha256")):
        _fail("Final-RCPD semantic evidence or immutable binding differs")


def _validate_runtime(report: Mapping[str, Any], context: Mapping[str, Any], report_hashes: Mapping[str, str]) -> None:
    if (not isinstance(report, dict) or report.get("version") != RUNTIME_VERSION
            or report.get("status") != "runtime_components_verified"
            or report.get("candidate_selected_by_r4_gate") is not True
            or report.get("formal_ready") is not False):
        _fail("Final r4 runtime bundle was not verified from a selected candidate")
    actor, program, protocol = report.get("actor", {}), report.get("program", {}), report.get("protocol", {})
    if (actor.get("sha256") != context["actor_sha256"]
            or actor.get("parameters_sha256") != context["actor_parameters_sha256"]
            or program.get("sha256") != context["program_sha256"]
            or program.get("feature_names_equal_actor") is not True
            or program.get("native_source_actor_sha256") != context["actor_sha256"]
            or program.get("source_training_actor_sha256") != context["training_actor_sha256"]
            or protocol.get("sha256") != context["protocol_sha256"]):
        _fail("Final r4 runtime artifact binding differs")
    evaluation = report.get("evaluation", {})
    if (evaluation.get("sha256") != report_hashes["active_policy"]
            or evaluation.get("status") != "passed"):
        _fail("Final r4 runtime does not bind the passing paired audit")
    final_rcpd = report.get("final_rcpd")
    if (not isinstance(final_rcpd, dict)
            or set(final_rcpd) != {"version", "path", "sha256", "status",
                "binding_sha256", "source_training_actor_sha256",
                "source_training_actor_path", "scenario_manifest_path",
                "scenario_manifest_sha256",
                "program_source_sha256", "candidate_count", "ppo_joint_steps",
                "optimizer_updates", "program_feedback_into_actor"}
            or final_rcpd.get("version") != FINAL_RCPD_VERSION
            or final_rcpd.get("status") != "passed"
            or final_rcpd.get("source_training_actor_sha256") != context["training_actor_sha256"]
            or final_rcpd.get("scenario_manifest_sha256")
                != context["scenario_manifest_file_sha256"]
            or final_rcpd.get("candidate_count") != 25
            or final_rcpd.get("ppo_joint_steps") != 0
            or final_rcpd.get("optimizer_updates") != 0
            or final_rcpd.get("program_feedback_into_actor") is not False
            or _HEX.fullmatch(str(final_rcpd.get("sha256", ""))) is None
            or _HEX.fullmatch(str(final_rcpd.get("binding_sha256", ""))) is None
            or _HEX.fullmatch(str(final_rcpd.get("program_source_sha256", ""))) is None
            or program.get("final_rcpd_binding_sha256") != final_rcpd.get("binding_sha256")):
        _fail("Final r4 runtime does not bind a passing post-freeze RCPD extraction")
    _validate_final_rcpd_evidence(final_rcpd, context)
    runtime, explainer, parity = report.get("runtime", {}), report.get("explainer", {}), report.get("numpy_parity", {})
    if (runtime != {"signature": context["runtime_signature"], "load_verified": True,
                    "action_masks": False, "post_policy_overrides": 0,
                    "actual_observed197_action_parity": True,
                    "actual_step_policy_equals_submitted": True}
            or explainer != {"signature": context["explainer_signature"], "load_verified": True,
                             "independent_r4_explanation_acceptance_pending": True}
            or parity.get("deterministic_actions_equal") is not True
            or _positive_int(parity.get("observations"), "NumPy parity observations") < 1
            or not 0 <= _finite(parity.get("maximum_absolute_logit_error"), "NumPy parity error") <= 1e-4):
        _fail("Final r4 runtime parity, authority, or explainer load failed")


def _validate_conflicts(report: Mapping[str, Any], selection: Mapping[str, Any],
                        context: Mapping[str, Any]) -> None:
    if (not isinstance(report, dict) or report.get("version") != SCENE_VERSION
            or report.get("status") != "accepted_six_scene_selection"
            or report.get("participant_data_read") is not False
            or report.get("final_test_rollouts") != 0):
        _fail("R4 high-conflict scene audit did not produce an accepted development selection")
    actor = report.get("actor")
    if (not isinstance(actor, dict) or actor.get("sha256") != context["actor_sha256"]
            or actor.get("deterministic") is not True or actor.get("post_policy_overrides") != 0):
        _fail("R4 high-conflict audit used another or overridden Actor")
    actor_protocol = report.get("actor_protocol")
    if (not isinstance(actor_protocol, dict)
            or actor_protocol.get("semantic_sha256") != context["protocol_sha256"]
            or report.get("source_scenario_manifest_sha256") != context["scenario_manifest_file_sha256"]
            or report.get("actor_action_override_frames") != 0
            or _positive_int(report.get("actor_submission_frames"), "scene Actor submissions") < 1
            or _positive_int(report.get("geometry_evaluated"), "geometry evaluations") < 60):
        _fail("R4 high-conflict input binding or authority count differs")
    rows = report.get("scene_metrics")
    if (not isinstance(rows, list) or report.get("dynamic_evaluated") != len(rows)
            or len(rows) < 6
            or report.get("dynamic_passed") != sum(row.get("dynamic", {}).get("passed") is True for row in rows)):
        _fail("R4 high-conflict dynamic matrix is incomplete")
    by_id = {row.get("id"): row for row in rows}
    selected = [*selection["X"], *selection["Y"]]
    if len(by_id) != len(rows) or any(scene["id"] not in by_id for scene in selected):
        _fail("R4 selected scenes are absent from the dynamic audit")
    for scene in selected:
        row = by_id[scene["id"]]
        geometry, dynamic = row.get("geometry", {}), row.get("dynamic", {})
        if (row.get("fingerprint") != scene["fingerprint"]
                or geometry.get("passed") is not True or dynamic.get("passed") is not True):
            _fail("A selected r4 scene failed geometry or dynamic checks")
        geometry_expected = {
            "route_overlap_band": (
                .25 <= _finite(geometry.get("shortest_task_route_shared_edge_ratio_min"),
                               "scene minimum shared route")
                and _finite(geometry.get("shortest_task_route_shared_edge_ratio_max"),
                            "scene maximum shared route") <= .55
            ),
            "shared_bottleneck_or_intersection": bool(
                geometry.get("shared_bridge_edges") or geometry.get("shared_intersections")),
            "opposing_and_same_direction_opportunities": (
                _positive_int(geometry.get("same_direction_mission_edges"),
                              "scene same-direction edges") > 0
                and _positive_int(geometry.get("opposing_mission_edges"),
                                  "scene opposing edges") > 0
            ),
        }
        if (geometry.get("checks") != geometry_expected
                or set(geometry_expected) != set(SCENE_GEOMETRY_CHECKS)
                or not all(geometry_expected.values())
                or _positive_int(geometry.get("initial_joint_work_steps"),
                                 "scene initial workload") < 1):
            _fail("A selected r4 scene does not meet the frozen geometry thresholds")
        opportunity = _finite(dynamic.get("conflict_opportunity_fraction"),
                              "scene conflict opportunity")
        risky = _finite(dynamic.get("player_risky_action_mass"), "scene risky action mass")
        cancellation = _finite(dynamic.get("collision_cancellation_fraction"),
                               "scene collision cancellation")
        recovery = _finite(dynamic.get("collision_recovery_within_10_rate"),
                           "scene collision recovery")
        reference = _finite(dynamic.get("compatible_reference_mean_deliveries"),
                            "scene reference deliveries")
        best = _finite(dynamic.get("best_simple_mean_deliveries"),
                       "scene best-simple deliveries")
        absolute_gap = _finite(dynamic.get("reference_absolute_delivery_gap"),
                               "scene delivery gap")
        relative_gap = _finite(dynamic.get("reference_relative_delivery_gap"),
                               "scene relative delivery gap")
        profiles = dynamic.get("simple_profile_mean_deliveries")
        best_profile = dynamic.get("best_simple_profile")
        if (not isinstance(profiles, dict) or set(profiles) != set(SCENE_SIMPLE_BASELINES)
                or best_profile not in SCENE_SIMPLE_BASELINES
                or any(not math.isfinite(_finite(value, "scene baseline deliveries"))
                       for value in profiles.values())
                or not math.isclose(best, float(profiles[best_profile]), abs_tol=1e-12)
                or not math.isclose(absolute_gap, reference - best, abs_tol=1e-12)
                or reference <= 0
                or not math.isclose(relative_gap, absolute_gap / reference,
                                    rel_tol=1e-12, abs_tol=1e-12)):
            _fail("A selected r4 scene has inconsistent delivery aggregates")
        dynamic_expected = {
            "conflict_opportunity_25_to_40_percent": .25 <= opportunity <= .40,
            "player_risky_action_mass_10_to_20_percent": .10 <= risky <= .20,
            "collision_cancellation_8_to_18_percent": .08 <= cancellation <= .18,
            "simple_baseline_gap": absolute_gap >= 2. or relative_gap >= .15,
            "collision_recovery_at_least_90_percent": recovery >= .90,
            # This property is available only in the immutable per-episode
            # journal; require the producer's explicit check in addition to
            # recomputing every aggregate check exposed in this report.
            "no_pathological_zero_delivery_collision_episode":
                dynamic.get("checks", {}).get(
                    "no_pathological_zero_delivery_collision_episode") is True,
            "no_persistent_collision_deadlock": (
                _finite(dynamic.get("max_consecutive_collisions"),
                        "scene maximum collision streak") <= 10),
            "no_persistent_no_progress_deadlock": (
                _finite(dynamic.get("max_no_progress_streak"),
                        "scene maximum no-progress streak") <= 40),
            "actor_actions_submitted_unchanged":
                dynamic.get("actor_action_override_frames") == 0,
        }
        if (dynamic.get("checks") != dynamic_expected
                or set(dynamic_expected) != set(SCENE_DYNAMIC_CHECKS)
                or not all(dynamic_expected.values())):
            _fail("A selected r4 scene does not meet the frozen dynamic thresholds")
        if (dynamic.get("actor_action_override_frames") != 0
                or _positive_int(dynamic.get("actor_submission_frames"), "selected-scene Actor submissions") < 1):
            _fail("A selected r4 scene contains an Actor override")
    selected_report = report.get("selection")
    balance = selection.get("balance", {})
    x_rows = [by_id[scene["id"]] for scene in selection["X"]]
    y_rows = [by_id[scene["id"]] for scene in selection["Y"]]
    expected_balance = {
        "X_mean_initial_workload": mean(row["geometry"]["initial_joint_work_steps"]
                                        for row in x_rows),
        "Y_mean_initial_workload": mean(row["geometry"]["initial_joint_work_steps"]
                                        for row in y_rows),
        "X_mean_conflict_opportunity": mean(row["dynamic"]["conflict_opportunity_fraction"]
                                            for row in x_rows),
        "Y_mean_conflict_opportunity": mean(row["dynamic"]["conflict_opportunity_fraction"]
                                            for row in y_rows),
    }
    expected_balance["workload_relative_difference"] = _relative_difference(
        expected_balance["X_mean_initial_workload"], expected_balance["Y_mean_initial_workload"])
    expected_balance["conflict_relative_difference"] = _relative_difference(
        expected_balance["X_mean_conflict_opportunity"],
        expected_balance["Y_mean_conflict_opportunity"])
    if (not isinstance(selected_report, dict)
            or selected_report.get("pairs") != selection.get("pairs")
            or selected_report.get("balance") != balance
            or balance != expected_balance
            or selected_report.get("all_dynamic_pass_count", 0) < 6
            or _finite(balance.get("workload_relative_difference"), "scene workload balance") > .05
            or _finite(balance.get("conflict_relative_difference"), "scene conflict balance") > .05):
        _fail("R4 scene pairing or X/Y balance differs")
    for left, right in zip(x_rows, y_rows):
        if (abs(left["dynamic"]["compatible_reference_mean_deliveries"]
                - right["dynamic"]["compatible_reference_mean_deliveries"]) > 1.
                or abs(left["dynamic"]["conflict_opportunity_fraction"]
                       - right["dynamic"]["conflict_opportunity_fraction"]) > .03):
            _fail("An r4 X/Y pair exceeds the frozen workload or conflict difference")
    online = report.get("deployment_runtime_validation")
    online_rows = online.get("scenes") if isinstance(online, dict) else None
    expected_online_fields = {"id", "seed", "frame", "expected_fingerprint",
        "actual_fingerprint", "fingerprint_equal", "public_history_valid",
        "inference_state_unchanged", "robot_2_policy_action", "step_policy_action",
        "step_submitted_action", "policy_action_submitted_unchanged",
        "post_policy_overrides"}
    if (not isinstance(online, dict) or online.get("passed") is not True
            or online.get("runtime_signature") != context["runtime_signature"]
            or online.get("actor_sha256") != context["actor_sha256"]
            or online.get("protocol_sha256") != context["protocol_sha256"]
            or not isinstance(online_rows, list) or len(online_rows) != 7
            or [row.get("id") for row in online_rows] != [f"play_{index:04d}" for index in range(7)]
            or any(not isinstance(row, dict) or set(row) != expected_online_fields
                   or row.get("frame") != 0 or row.get("public_history_valid") is not False
                   or row.get("fingerprint_equal") is not True
                   or row.get("expected_fingerprint") != row.get("actual_fingerprint")
                   or row.get("inference_state_unchanged") is not True
                   or row.get("policy_action_submitted_unchanged") is not True
                   or row.get("robot_2_policy_action") != row.get("step_policy_action")
                   or row.get("step_policy_action") != row.get("step_submitted_action")
                   or row.get("post_policy_overrides") != 0
                   for row in online_rows)):
        _fail("R4 selected scenes failed genuine online-runtime reset validation")
    evidence = report.get("evidence_artifacts", {})
    if evidence.get("selected_scenes.json") != context["selected_scenes_file_sha256"]:
        _fail("R4 high-conflict report does not bind the packaged scene selection")


def _validate_explanation(report: Mapping[str, Any], context: Mapping[str, Any],
                          report_path: str | Path, report_sha256: str) -> None:
    fields = {"version", "status", "test_fixture", "formal_ready", "bindings",
              "statistics", "base_rows", "branch_anchor_count", "episodes",
              "zero_nn_overrides", "source_state_unchanged", "counterfactual_isolated",
              "program_never_controls_action", "execution", "evidence_artifacts"}
    if (not isinstance(report, dict) or set(report) != fields
            or report.get("version") != EXPLANATION_VERSION or report.get("status") != "passed"
            or report.get("test_fixture") is not False or report.get("formal_ready") is not False):
        _fail("Exact real r4 explanation acceptance report required")
    expected_bindings = {key: context[key] for key in (
        "actor_sha256", "protocol_sha256", "runtime_signature",
        "program_sha256", "explainer_signature",
    )}
    bindings = report.get("bindings")
    if (not isinstance(bindings, dict) or any(bindings.get(key) != value for key, value in expected_bindings.items())
            or bindings.get("scenario_manifest_sha256") != context["scenario_manifest_semantic_sha256"]
            or bindings.get("test_fixture") is not False):
        _fail("R4 explanation audit component or held-out binding differs")
    for key in ("contract_sha256", "producer_sources_sha256", "holdout_fingerprints_sha256"):
        _sha(bindings.get(key), f"explanation {key}")
    if (report.get("zero_nn_overrides") is not True
            or report.get("source_state_unchanged") is not True
            or report.get("counterfactual_isolated") is not True
            or report.get("program_never_controls_action") is not True
            or report.get("episodes") != 300
            or _positive_int(report.get("base_rows"), "explanation base rows") < 1
            or _positive_int(report.get("branch_anchor_count"), "explanation branch anchors") < 1):
        _fail("R4 explanation audit invariants or matrix size differ")
    stats = report.get("statistics")
    required_stats = {"fidelity", "intervention_direction", "checks", "passed",
                      "evaluated_role", "no_effect_pairs_counted_as_success"}
    if (not isinstance(stats, dict) or set(stats) != required_stats
            or stats.get("passed") is not True or stats.get("evaluated_role") != "robot_2"
            or stats.get("no_effect_pairs_counted_as_success") is not False):
        _fail("R4 explanation statistics did not pass for robot_2")
    _all_true(stats.get("checks"), None, "explanation")
    fidelity, direction = stats.get("fidelity"), stats.get("intervention_direction")
    if not isinstance(fidelity, dict) or not isinstance(direction, dict):
        _fail("R4 explanation fidelity matrices are missing")
    _stat(fidelity.get("overall"), "explanation overall", .90)
    _stat(fidelity.get("nonwait"), "explanation non-WAIT", .85)
    by_group, by_group_nonwait = fidelity.get("by_group"), fidelity.get("by_group_nonwait")
    direction_groups = direction.get("by_group")
    if (not isinstance(by_group, dict) or set(by_group) != set(CRITICAL_GROUPS)
            or not isinstance(by_group_nonwait, dict) or set(by_group_nonwait) != set(CRITICAL_GROUPS)
            or not isinstance(direction_groups, dict) or set(direction_groups) != set(CRITICAL_GROUPS)):
        _fail("R4 explanation critical-group matrices differ")
    _stat(direction.get("overall"), "explanation intervention direction", .85, 10)
    for group in CRITICAL_GROUPS:
        _stat(by_group[group], f"explanation {group}", .85, 10)
        _stat(by_group_nonwait[group], f"explanation non-WAIT {group}", .85, 10)
        _stat(direction_groups[group], f"explanation direction {group}", .85, 10)
    execution = report.get("execution")
    if (not isinstance(execution, dict) or execution.get("accounting_complete") is not True
            or execution.get("pending_operation") is not None):
        _fail("R4 explanation execution is incomplete")
    counts = execution.get("counts")
    count_fields = {"base_steps", "counterfactual_steps", "language_runtime_steps",
                    "acknowledged_steps", "neural_updates", "tree_fits", "torch_loads"}
    if (not isinstance(counts, dict) or set(counts) != count_fields
            or _positive_int(counts.get("base_steps"), "explanation base steps") < 1
            or _positive_int(counts.get("counterfactual_steps"), "explanation counterfactual steps") < 1
            or _positive_int(counts.get("language_runtime_steps"), "explanation language runtime steps") < 1
            or counts.get("acknowledged_steps") != (counts.get("base_steps")
                + counts.get("counterfactual_steps") + counts.get("language_runtime_steps"))
            or any(counts.get(key) != 0 for key in ("neural_updates", "tree_fits", "torch_loads"))):
        _fail("R4 explanation execution accounting differs")
    artifacts = report.get("evidence_artifacts")
    expected_artifacts = {"inputs.json", "ordinary_rows.jsonl",
                          "intervention_rows.jsonl", "language_rows.jsonl"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        _fail("R4 explanation audit lacks the exact immutable evidence set")
    for name, value in artifacts.items():
        if type(name) is not str or not name or _HEX.fullmatch(str(value)) is None:
            _fail("R4 explanation evidence artifact hash differs")
    # The r4 producer independently recomputes every reported statistic from
    # its immutable ordinary/intervention/language rows.  This prevents a
    # hand-edited passing summary from entering an admission merely because
    # the edited file has a new hash.
    from backend.training import warehouse_r4_explanation_audit as producer
    recomputed = producer.read_saved_report(
        Path(report_path).resolve().parent,
        expected_report_sha256=report_sha256,
        expected_bindings=bindings,
    )
    if digest(recomputed) != digest(report):
        _fail("R4 explanation saved-row recalculation differs")


def _validate_questionnaire(report: Mapping[str, Any], question: Mapping[str, Any],
                            context: Mapping[str, Any]) -> None:
    if (not isinstance(report, dict) or report.get("version") != QUESTION_VERSION
            or report.get("status") != "candidate_ready" or report.get("formal_ready") is not False
            or report.get("actor_sha256") != context["actor_sha256"]
            or report.get("protocol_sha256") != context["protocol_sha256"]
            or report.get("runtime_signature") != context["runtime_signature"]
            or report.get("source_bank_signature") != context["question_bank_signature"]
            or report.get("payload_sha256") != context["question_bank_sha256"]
            or report.get("selected_scenarios") != 8
            or _positive_int(report.get("candidate_frames"), "questionnaire candidate frames") < 8):
        _fail("Final r4 questionnaire report binding or diversity differs")
    if (not isinstance(question, dict) or question.get("version") != QUESTION_PAYLOAD_VERSION
            or question.get("test_fixture") is not False
            or question.get("actor_sha256") != context["actor_sha256"]
            or question.get("protocol_sha256") != context["protocol_sha256"]
            or question.get("runtime_signature") != context["runtime_signature"]
            or question.get("source_bank_signature") != context["question_bank_signature"]):
        _fail("Final r4 questionnaire payload belongs to another runtime")
    checks = report.get("checks")
    if checks != question.get("checks") or not isinstance(checks, dict) or checks.get("passed") is not True \
            or checks.get("formal_ready") is not False:
        _fail("Final r4 questionnaire replay checks differ")
    categories = checks.get("categories")
    if (not isinstance(categories, dict) or set(categories) != {"next_action", "wait_three"}
            or any(value != {"count": 4, "independent_scenarios": 4, "distinct_outcomes": 4}
                   for value in categories.values())):
        _fail("Final r4 questionnaire is not two diverse four-item groups")
    replay = checks.get("independent_replay")
    if (not isinstance(replay, list) or len(replay) != 8
            or any(set(row) != {"id", "answer_match", "evidence_match", "source_unchanged"}
                   or row["answer_match"] is not True or row["evidence_match"] is not True
                   or row["source_unchanged"] is not True for row in replay)):
        _fail("Final r4 questionnaire independent replay failed")
    items = question.get("items")
    if (not isinstance(items, list) or len(items) != 8
            or len({item.get("id") for item in items}) != 8
            or len({item.get("scenario_id") for item in items}) != 8
            or sum(item.get("kind") == "next_action" for item in items) != 4
            or sum(item.get("kind") == "wait_three" for item in items) != 4):
        _fail("Final r4 questionnaire item matrix differs")


def validate_report_set(*, reports: Mapping[str, Mapping[str, Any]],
                        report_hashes: Mapping[str, str], report_paths: Mapping[str, str | Path],
                        context: Mapping[str, Any],
                        selection: Mapping[str, Any], question: Mapping[str, Any],
                        scenarios: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute every component-level r4 admission gate.

    ``context`` and ``report_hashes`` are supplied by the caller after loading
    the actual frozen components and verifying report bytes.  This function
    accepts no skip flags, fixture mode, or caller-provided gate booleans.
    """
    if (not isinstance(reports, dict) or set(reports) != set(REPORT_NAMES)
            or not isinstance(report_hashes, dict) or set(report_hashes) != set(REPORT_NAMES)
            or not isinstance(report_paths, dict) or set(report_paths) != set(REPORT_NAMES)):
        _fail("Exact r4 production report set required")
    for name, value in report_hashes.items():
        _sha(value, f"{name} report")
    required_context = {"actor_sha256", "actor_parameters_sha256", "actor_scenario_manifest_sha256",
        "training_actor_sha256", "protocol_sha256", "runtime_signature", "program_sha256",
        "explainer_signature", "selected_scenes_sha256", "selected_scenes_file_sha256",
        "question_bank_sha256", "question_bank_signature", "scenario_manifest_semantic_sha256",
        "scenario_manifest_file_sha256", "validation_entries_sha256"}
    if not isinstance(context, dict) or set(context) != required_context:
        _fail("Exact loaded r4 component context required")
    for key, value in context.items():
        _sha(value, key)
    _validate_scenarios(scenarios, context)
    _validate_active(reports["active_policy"], context)
    _validate_active_evidence(reports["active_policy"], report_paths["active_policy"],
                              scenarios["splits"]["validation"])
    _validate_training_ledger(reports["training_budget"], context,
                              report_paths["training_budget"],
                              report_hashes["training_budget"],
                              report_hashes["active_policy"])
    _validate_runtime(reports["runtime_components"], context, report_hashes)
    _validate_conflicts(reports["high_conflict_scenes"], selection, context)
    _validate_explanation(reports["explanation_program"], context,
                          report_paths["explanation_program"],
                          report_hashes["explanation_program"])
    _validate_questionnaire(reports["questionnaire"], question, context)
    gates = {name: True for name in GATE_NAMES}
    # The action-authority and online-runtime gates are derived from several
    # independent producers above; neither has a caller-settable pass flag.
    return gates


def component_context(*, runtime: Any, explainer: Any,
                      selection: Mapping[str, Any], question: Mapping[str, Any],
                      scenarios: Mapping[str, Any], selected_scenes_path: str | Path,
                      scenarios_path: str | Path) -> dict[str, str]:
    """Derive admission identities only from already loaded frozen objects."""
    metadata = runtime.actor.metadata
    parameters = metadata.get("actor_parameters_sha256", metadata.get("parameters_sha256"))
    context = {
        "actor_sha256": runtime.actor_sha256,
        "actor_parameters_sha256": parameters,
        "actor_scenario_manifest_sha256": metadata.get("scenario_manifest_sha256"),
        "training_actor_sha256": metadata.get("r4_training_actor_sha256"),
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature,
        "program_sha256": explainer.program_sha256,
        "explainer_signature": explainer.signature,
        "selected_scenes_sha256": digest(selection),
        "selected_scenes_file_sha256": file_hash(selected_scenes_path),
        "question_bank_sha256": digest(question),
        "question_bank_signature": question.get("source_bank_signature"),
        "scenario_manifest_semantic_sha256": digest(scenarios),
        "scenario_manifest_file_sha256": file_hash(scenarios_path),
        "validation_entries_sha256": digest(scenarios.get("splits", {}).get("validation")),
    }
    if set(context) != {"actor_sha256", "actor_parameters_sha256", "actor_scenario_manifest_sha256",
            "training_actor_sha256", "protocol_sha256", "runtime_signature", "program_sha256",
            "explainer_signature", "selected_scenes_sha256", "selected_scenes_file_sha256",
            "question_bank_sha256", "question_bank_signature", "scenario_manifest_semantic_sha256",
            "scenario_manifest_file_sha256", "validation_entries_sha256"}:
        _fail("Internal r4 context schema differs")
    for key, value in context.items():
        _sha(value, key)
    return context


def report_bindings(paths: Mapping[str, str], hashes: Mapping[str, str]) -> dict[str, dict[str, str]]:
    if set(paths) != set(REPORT_NAMES) or set(hashes) != set(REPORT_NAMES):
        _fail("Exact r4 report paths and hashes required")
    return {name: {"path": paths[name], "sha256": hashes[name]} for name in REPORT_NAMES}


__all__ = ["VERSION", "ADMISSION_VERSION", "REPORT_NAMES", "GATE_NAMES",
           "EXPLANATION_VERSION", "canonical", "digest", "file_hash",
           "local_source_hashes", "component_context", "validate_report_set",
           "report_bindings"]
