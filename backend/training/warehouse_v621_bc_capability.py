"""Opaque, source-bound, single-use capability for finite v6.21 BC development.

Imports and ordinary constructors grant no training authority. Every allowed
reset argument is consumed in order into a durable preauthorization journal.
Nothing in this module authorizes fresh prechecks, PPO, sealing, or a pilot.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from experiments.warehouse import v621_bc_protocol as protocol
from experiments.warehouse import v621_protocol as diagnostics

SCHEMA = "warehouse-v621-bc-development-spec.v1"
CLEARANCE_SCHEMA = "warehouse-v621-bc-seed-clearance.v1"
MODEL_CODE_FILES = (
    "backend/training/warehouse_v621_actor.py",
    "backend/training/warehouse_v621_actor_training.py",
    "backend/training/warehouse_v621_program.py",
)
_CONSTRUCTION_SEAL = object()
_ACTIVE: ContextVar[BCDevelopmentPlan | None] = ContextVar("warehouse_v621_bc_development", default=None)
OPERATIONS = frozenset({"construct", "bc_update", "critic_update", "collect", "export", "program_fit", "gradient_probe"})


def digest(value):
    return protocol.digest(value)


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def model_implementation_sha256(root):
    return digest({name: file_sha(Path(root)/name) for name in MODEL_CODE_FILES})


def reset_schedule(initialization, recipe):
    """The whole sequence, including repeated critic and spent development uses."""
    if type(initialization) is not int or initialization not in protocol.INITIALIZATIONS:
        raise ValueError("unregistered BC initialization")
    if recipe not in (*protocol.RECIPES, "shared_cache"):
        raise ValueError("unregistered BC recipe/scope")
    registered = protocol.seed_plan()["initializations"][str(initialization)]
    selected = []
    for phase, _ in protocol.STAGES:
        teacher = phase in protocol.TEACHER_STAGES
        if teacher != (recipe == "shared_cache"):
            continue
        selected.extend({"phase": phase, "scenario_seed": row["scenario_seed"],
            "action_seed": row["action_seed"]} for row in registered["stages"][phase])
    if recipe == "shared_cache":
        selected.extend({"phase": "critic_setup", "scenario_seed": value, "action_seed": None}
            for value in registered["critic_environment_reset_seeds"])
    else:
        selected.extend({"phase": "spent_development_evaluation", "scenario_seed": row["scenario_seed"],
            "action_seed": row["action_seed"]} for row in protocol.seed_plan()["development_evaluation"]["cases"])
    return selected


def reset_contract():
    return {str(seed): {recipe: reset_schedule(seed, recipe)
        for recipe in ("shared_cache", *protocol.RECIPES)} for seed in protocol.INITIALIZATIONS}


def _pointer(root, value, label):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"malformed BC pointer: {label}")
    path = (root/value["path"]).resolve(strict=True)
    if not path.is_relative_to(root):
        raise RuntimeError(f"BC evidence outside declared checkout: {label}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != value["sha256"]:
        raise RuntimeError(f"BC pointer SHA mismatch: {label}")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError(f"BC pointer is not a JSON object: {label}")
    return path, result


def _strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_strict_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_strict_equal(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def _closed(value, *, name):
    if (value.get("publishable") is not False or value.get("PPO_joint_steps") != 0
            or type(value.get("PPO_joint_steps")) is not int):
        raise RuntimeError(f"BC prerequisite has publication/PPO authority: {name}")


def _checked_prerequisites(root, spec):
    """Bind all eighteen results to the true independent audit, not a flag."""
    summary_path, summary = _pointer(root, spec["diagnostics_summary"], "diagnostics summary")
    audit_path, audit = _pointer(root, spec["diagnostics_independent_audit"], "independent audit")
    _closed(summary, name="diagnostics")
    _closed(audit, name="independent audit")
    if (summary.get("schema") != "warehouse-v621-diagnostics-summary.v1"
            or summary.get("candidate_id") != diagnostics.CANDIDATE_ID
            or summary.get("selected_arm") is not None
            or summary.get("new_data_or_training_authorized") is not False
            or audit.get("schema") != "warehouse-v621-independent-diagnostics-audit.v1"
            or audit.get("passed") is not True or audit.get("jobs_audited") != 18
            or audit.get("selected_arm") is not None or audit.get("summary_sha256") != file_sha(summary_path)
            or audit.get("plan_sha256") != summary.get("plan_sha256")
            or audit.get("production_metric_routing_qualification_oracle_calls") != 0
            or audit.get("environment_resets") != 0 or audit.get("actor_updates") != 0):
        raise RuntimeError("BC requires complete independent audited no-passing-method diagnostics")
    auditor = root/"scripts/audit_warehouse_v621_diagnostics.py"
    if audit.get("auditor_sha256") != file_sha(auditor):
        raise RuntimeError("BC independent auditor implementation hash mismatch")
    plan_path = summary_path.parent/"plan.json"
    plan = json.loads(plan_path.read_bytes())
    if (file_sha(plan_path) != summary["plan_sha256"] or plan.get("jobs") != diagnostics.jobs()
            or plan.get("folds") != diagnostics.grouped_folds()
            or plan.get("identity", {}).get("source_files_sha256") != diagnostics.source_closure_files(root)
            or plan["identity"]["source_closure_sha256"] != audit.get("source_closure_sha256")):
        raise RuntimeError("BC diagnostics source, fold or parent binding differs")
    baseline_actor = Path(plan["identity"]["actor_path"]).resolve(strict=True)
    if (not baseline_actor.is_relative_to(root) or file_sha(baseline_actor) != diagnostics.ACTOR_SHA256
            or plan["identity"].get("actor_role_sha256") != diagnostics.ROLE_SHA256):
        raise RuntimeError("BC baseline dual Actor changed after diagnostics")
    references = summary.get("jobs")
    audited = {r["job"]["job_id"]: r for r in audit.get("jobs", [])}
    expected = {r["job_id"] for r in diagnostics.jobs()}
    if not isinstance(references, dict) or set(references) != expected or set(audited) != expected:
        raise RuntimeError("BC prerequisites do not contain all eighteen audited units")
    bound = {str(summary_path): file_sha(summary_path), str(audit_path): file_sha(audit_path),
        str(plan_path): file_sha(plan_path), str(auditor): file_sha(auditor)}
    for job in diagnostics.jobs():
        path, result = _pointer(root, references[job["job_id"]], "diagnostic job")
        a = audited[job["job_id"]]
        completed = json.loads((path.parent/"completed.json").read_bytes())
        if (result.get("job") != job or a.get("job") != job or a.get("passed") is not True
                or a.get("status") != result.get("status") or a.get("result_sha256") != file_sha(path)
                or result.get("plan_sha256") != file_sha(plan_path)
                or completed.get("result_sha256") != file_sha(path)
                or result.get("status") not in ("completed", "structurally_infeasible")
                or (path.parent/"failed.json").exists()):
            raise RuntimeError("BC audited diagnostics child binding differs")
        bound[str(path)] = file_sha(path)
        bound[str(path.parent/"completed.json")] = file_sha(path.parent/"completed.json")
    if any(arm.get("all_role_fold_gates_passed") is not False for arm in summary["arms"].values()):
        raise RuntimeError("BC route is closed when an extraction method passes")
    for name, expected_sha in plan["identity"]["inputs"]["files_sha256"].items():
        input_path = Path(name).resolve(strict=True)
        if not input_path.is_relative_to(root) or file_sha(input_path) != expected_sha:
            raise RuntimeError("BC spent diagnostic input changed")
    # The old cohort is separately revalidated by the frozen development loader
    # before evaluation; its large traces are not rehashed before each BC reset.
    return bound


def _array_file_identity(path):
    import numpy as np
    result = sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            value = np.ascontiguousarray(archive[name])
            result.update(name.encode())
            result.update(value.dtype.str.encode())
            result.update(np.asarray(value.shape, dtype="<i8").tobytes())
            result.update(value.tobytes())
    return result.hexdigest()


def _checked_shared_cache(root, spec, initialization):
    directory = root/"output/warehouse_v621"/protocol.BATCH/f"init-{initialization}"/"shared_cache"
    manifest_path = directory/"cache.manifest.json"
    completed_path = directory/"capability.completed.json"
    started_path = directory/"capability.started.json"
    started = json.loads(started_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    completed = json.loads(completed_path.read_bytes())
    entry_completed_path = directory/"completed.json"
    entry_completed = json.loads(entry_completed_path.read_bytes())
    if (entry_completed.get("result") != {"path": str(manifest_path), "sha256": file_sha(manifest_path)}
            or entry_completed.get("capability_completed") != {"path": str(completed_path), "sha256": file_sha(completed_path)}
            or entry_completed.get("actual_reset_count") != 542
            or entry_completed.get("source_closure_sha256") != spec["source_closure_sha256"]):
        raise RuntimeError("shared cache entry/capability completion chain differs")
    if (manifest.get("schema") != "warehouse-v621-bc-shared-cache.v1"
            or manifest.get("status") != "completed" or manifest.get("initialization") != initialization
            or manifest.get("source_closure_sha256") != spec["source_closure_sha256"]
            or manifest.get("seed_plan_sha256") != spec["seed_plan"]["sha256"]
            or manifest.get("model_implementation_sha256") != spec["model_implementation_sha256"]
            or type(manifest.get("actual_reset_count")) is not int or manifest["actual_reset_count"] != 542
            or manifest.get("actor_updates") != 0 or manifest.get("PPO_joint_steps") != 0
            or started.get("schema") != "warehouse-v621-bc-development-capability.v1"
            or started.get("source_closure_sha256") != spec["source_closure_sha256"]
            or started.get("initialization") != initialization or started.get("recipe") != "shared_cache"
            or started.get("scheduled_reset_count") != 542
            or completed.get("schema") != "warehouse-v621-bc-capability-completed.v1"
            or completed.get("initialization") != initialization or completed.get("recipe") != "shared_cache"
            or completed.get("preauthorized_resets") != 542):
        raise RuntimeError("model BC requires a complete source/seed/model-bound shared cache")
    bound = {str(manifest_path): file_sha(manifest_path), str(completed_path): file_sha(completed_path),
        str(started_path): file_sha(started_path), str(entry_completed_path): file_sha(entry_completed_path)}
    assets = manifest.get("teacher_assets")
    if (not isinstance(assets, dict) or set(assets) != set(protocol.TEACHER_STAGES)
            or len({item["path"] for item in assets.values()}) != 4):
        raise RuntimeError("shared cache must contain exactly the four teacher stages")
    for name, item in [*assets.items(), ("critic", manifest["critic_asset"])]:
        path = (root/item["path"]).resolve(strict=True)
        expected_identity = item["model_sha256" if name == "critic" else "data_sha256"]
        if (not path.is_relative_to(directory) or file_sha(path) != item["sha256"]
                or (name != "critic" and _array_file_identity(path) != expected_identity)
                or path.stat().st_mode & 0o222):
            raise RuntimeError("shared teacher/critic asset hash or read-only identity differs")
        bound[str(path)] = item["sha256"]
    report_path, report = _pointer(root, manifest["critic_report"], "shared critic report")
    if report.get("model_sha256") != manifest["critic_asset"]["model_sha256"]:
        raise RuntimeError("shared critic report/model identity differs")
    bound[str(report_path)] = file_sha(report_path)
    success = manifest["success_journal"]
    success_path = (root/success["path"]).resolve(strict=True)
    if not success_path.is_relative_to(directory) or file_sha(success_path) != success["sha256"]:
        raise RuntimeError("shared cache native successful reset journal changed")
    actual = [json.loads(line) for line in success_path.read_bytes().splitlines()]
    expected = reset_schedule(initialization, "shared_cache")
    if len(actual) != 542 or [r.get("seed") for r in actual] != [r["scenario_seed"] for r in expected]:
        raise RuntimeError("shared cache native actual reset order/count differs")
    pre = directory/"reset-preauthorization.jsonl"
    if file_sha(pre) != completed["preauthorization_journal_sha256"]:
        raise RuntimeError("shared cache preauthorization journal changed")
    tickets = [json.loads(line) for line in pre.read_bytes().splitlines()]
    if len(tickets) != 542 or any({k: row[k] for k in ("phase", "scenario_seed", "action_seed")} != expected[i]
            or row.get("index") != i for i, row in enumerate(tickets)):
        raise RuntimeError("shared cache native and preauthorized reset schedules differ")
    bound[str(success_path)] = file_sha(success_path)
    bound[str(pre)] = file_sha(pre)
    return bound


def _check_backup(root, spec):
    import math
    refs = spec.get("main_results")
    if not isinstance(refs, Mapping) or set(refs) != {str(s) for s in protocol.INITIALIZATIONS}:
        raise RuntimeError("backup needs three completed immutable main result bindings")
    bound, failures = {}, 0
    for initialization in protocol.INITIALIZATIONS:
        path, result = _pointer(root, refs[str(initialization)], "completed main run")
        folder = (root/"output/warehouse_v621"/protocol.BATCH/f"init-{initialization}"/"shared_b035").resolve()
        if (path != folder/"result.json" or result.get("schema") != "warehouse-v621-bc-development-result.v1"
                or result.get("status") != "completed" or result.get("recipe") != "shared_b035"
                or result.get("initialization") != initialization
                or type(result.get("quality_passed")) is not bool
                or result.get("source_closure_sha256") != spec["source_closure_sha256"]
                or result.get("seed_plan_sha256") != spec["seed_plan"]["sha256"]
                or result.get("model_implementation_sha256") != spec["model_implementation_sha256"]
                or type(result.get("fresh_precheck_episodes")) is not int or result["fresh_precheck_episodes"] != 0
                or (folder/"capability.failed.json").exists()):
            raise RuntimeError("backup main run is incomplete or not source/seed/model-bound")
        _closed(result, name="main run")
        completed_path = folder/"completed.json"
        completed = json.loads(completed_path.read_bytes())
        cap_path, capability = _pointer(root, completed["capability_completed"], "main capability completed")
        schedule = reset_schedule(initialization, "shared_b035")
        if (completed.get("result") != {"path": str(path), "sha256": file_sha(path)}
                or cap_path != folder/"capability.completed.json"
                or result.get("actual_reset_count") != len(schedule)
                or completed.get("actual_reset_count") != len(schedule)
                or capability.get("preauthorized_resets") != len(schedule)
                or capability.get("recipe") != "shared_b035" or capability.get("initialization") != initialization):
            raise RuntimeError("backup main completion/reset capability chain differs")
        for name, filename in (("success_journal", "native-successful-resets.jsonl"),
                               ("preauthorization_journal", "reset-preauthorization.jsonl")):
            pointer = result[name]
            journal = (root/pointer["path"]).resolve(strict=True)
            if journal != folder/filename or file_sha(journal) != pointer["sha256"]:
                raise RuntimeError("backup main native/preauthorization journal changed")
            rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
            if name == "success_journal":
                valid = len(rows) == len(schedule) and all(row.get("phase") == "bc" for row in rows) and [row.get("seed") for row in rows] == [r["scenario_seed"] for r in schedule]
            else:
                valid = (capability.get("preauthorization_journal_sha256") == pointer["sha256"]
                    and len(rows) == len(schedule) and all(row.get("index") == i and
                        {k: row[k] for k in ("phase", "scenario_seed", "action_seed")} == schedule[i]
                        for i, row in enumerate(rows)))
            if not valid:
                raise RuntimeError("backup main exact actual/preauthorized reset schedule differs")
            bound[str(journal)] = file_sha(journal)
        collection_path, collection = _pointer(root, result["development_collection"], "main development outcomes")
        cases = diagnostics.source_cases()
        outcomes = collection.get("outcomes", [])
        if collection.get("cases") != cases or len(outcomes) != len(cases):
            raise RuntimeError("backup main development cohort is not all spent 275 cases")
        for case, row in zip(cases, outcomes, strict=True):
            if (row.get("case") != case or type(row.get("deliveries")) is not int or row["deliveries"] < 0
                    or type(row.get("robot_collisions")) is not int or row["robot_collisions"] < 0
                    or type(row.get("shutdown")) is not bool
                    or type(row.get("physical_frames")) is not int or not 1 <= row["physical_frames"] <= 120):
                raise RuntimeError("backup main outcome is missing, invalid or assigned to another case")
        validation = [r for r in outcomes if r["case"]["original_split"] == "validation"]
        computed = {"mean_deliveries": sum(r["deliveries"] for r in validation)/75,
            "mean_collisions": sum(r["robot_collisions"] for r in validation)/75,
            "shutdown_episode_rate": sum(r["shutdown"] for r in validation)/75}
        quality = result.get("quality")
        if (len(validation) != 75 or not isinstance(quality, dict) or set(quality) != set(computed)
                or any(type(quality[k]) not in (int, float) or not math.isfinite(quality[k])
                       or quality[k] != computed[k] for k in computed)):
            raise RuntimeError("backup main finite quality metrics do not match all 75 recorded outcomes")
        passed = (computed["mean_deliveries"] >= 5.5 and computed["mean_collisions"] <= 6.
                  and computed["shutdown_episode_rate"] <= .05)
        if result["quality_passed"] is not passed:
            raise RuntimeError("backup main quality flag does not recompute from real outcomes")
        failures += not passed
        for target in (path, completed_path, cap_path, collection_path):
            bound[str(target)] = file_sha(target)
    if not failures:
        raise RuntimeError("program-only failures do not unlock the larger residual backup")
    return bound


@dataclass(frozen=True)
class BCDevelopmentPlan:
    root: Path
    specification_path: Path
    initialization: int
    recipe: str
    scope: str
    output_directory: Path
    journal_path: Path
    schedule: tuple
    source_closure_sha256: str
    seed_plan_sha256: str
    model_implementation_sha256: str
    _source_files: Mapping = field(repr=False)
    _bound_files: Mapping = field(repr=False)
    _seal: object = field(repr=False)
    _cursor: list = field(default_factory=lambda: [0], repr=False, compare=False)
    _invalid: list = field(default_factory=lambda: [False], repr=False, compare=False)

    def __post_init__(self):
        if self._seal is not _CONSTRUCTION_SEAL:
            raise RuntimeError("BCDevelopmentPlan cannot be constructed by ordinary callers")

    @property
    def init(self):
        return self.initialization

    @property
    def reset_count(self):
        return self._cursor[0]

    def assert_immutable(self):
        if self._seal is not _CONSTRUCTION_SEAL:
            raise RuntimeError("unsealed BC development plan")
        if self._invalid[0]:
            raise RuntimeError("BC development attempt was invalidated and cannot resume")
        if protocol.source_files(self.root) != dict(self._source_files):
            self._invalid[0] = True
            raise RuntimeError("BC development source changed after authorization")
        for name, expected in self._bound_files.items():
            if file_sha(name) != expected:
                self._invalid[0] = True
                raise RuntimeError(f"BC development bound evidence changed: {name}")

    def authorize_reset(self, phase, scenario_seed, action_seed):
        require_active_plan(self, operation="collect")
        self.assert_immutable()
        index = self._cursor[0]
        actual = {"phase": phase, "scenario_seed": scenario_seed, "action_seed": action_seed}
        if (type(scenario_seed) is not int or (action_seed is not None and type(action_seed) is not int)
                or index >= len(self.schedule) or actual != dict(self.schedule[index])):
            self._invalid[0] = True
            raise RuntimeError("BC reset is outside the exact ordered development schedule")
        row = {"schema": "warehouse-v621-bc-reset-preauthorization.v1", "index": index,
            "initialization": self.initialization, "recipe": self.recipe, **actual,
            "source_closure_sha256": self.source_closure_sha256,
            "seed_plan_sha256": self.seed_plan_sha256,
            "consumed_before_actual_reset": True, "not_an_actual_reset_receipt": True}
        # A ticket is spent before reset. Failure/interruption never permits a
        # second execution, even if the native environment was not reached.
        with self.journal_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._cursor[0] += 1
        return row

    def assert_completed(self):
        self.assert_immutable()
        if self._cursor[0] != len(self.schedule):
            raise RuntimeError("BC development did not consume its complete ordered reset schedule")
        rows = [json.loads(line) for line in self.journal_path.read_bytes().splitlines()]
        if len(rows) != self._cursor[0] or any(r.get("index") != i or
                {k: r[k] for k in ("phase", "scenario_seed", "action_seed")} != dict(self.schedule[i])
                for i, r in enumerate(rows)):
            raise RuntimeError("BC append-only reset journal changed")


def require_active_plan(plan=None, *, operation):
    if operation not in OPERATIONS:
        raise RuntimeError("operation is not in the finite BC development capability")
    active = _ACTIVE.get()
    if (active is None or active._seal is not _CONSTRUCTION_SEAL or active._invalid[0]
            or (plan is not None and plan is not active)):
        raise RuntimeError("ordinary collection, construction and training have no BC development authority")
    if active.scope == "shared_cache" and operation in {"construct", "bc_update", "program_fit", "gradient_probe"}:
        raise RuntimeError("shared cache cannot construct or update a policy Actor")
    return active


def _write_new(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


@contextmanager
def load_and_authorize(root, spec, recipe, initialization):
    """Verify prerequisites, claim one child directory, then expose one plan."""
    root = Path(root).resolve(strict=True)
    path = (root/spec).resolve(strict=True)
    if not path.is_relative_to(root) or _ACTIVE.get() is not None:
        raise RuntimeError("BC spec must be in the checkout and capabilities cannot nest")
    raw = path.read_bytes()
    value = json.loads(raw)
    protocol.checked_configuration(root)
    if (value.get("schema") != SCHEMA or value.get("candidate_id") != diagnostics.CANDIDATE_ID
            or value.get("batch_id") != protocol.BATCH or type(initialization) is not int
            or initialization not in protocol.INITIALIZATIONS or recipe not in (*protocol.RECIPES, "shared_cache")):
        raise RuntimeError("unregistered BC development specification or operation")
    sources = protocol.source_files(root)
    if (value.get("source_closure_sha256") != digest(sources)
            or value.get("model_implementation_sha256") != model_implementation_sha256(root)):
        raise RuntimeError("BC development source/model identity mismatch")
    recipe_path, declared_recipe = _pointer(root, value["recipe"], "recipe")
    seeds_path, declared_seeds = _pointer(root, value["seed_plan"], "seed plan")
    if declared_recipe != protocol.recipe() or declared_seeds != protocol.seed_plan():
        raise RuntimeError("BC development recipe or seed declaration changed")
    bound = _checked_prerequisites(root, value)
    clearance_path, clearance = _pointer(root, value["seed_clearance"], "seed clearance")
    required_clearance = {"schema": CLEARANCE_SCHEMA, "candidate_id": diagnostics.CANDIDATE_ID,
        "batch_id": protocol.BATCH, "approved_for_bc_development": True,
        "source_closure_sha256": value["source_closure_sha256"],
        "model_implementation_sha256": value["model_implementation_sha256"],
        "baseline_actor_sha256": diagnostics.ACTOR_SHA256,
        "recipe_sha256": value["recipe"]["sha256"], "seed_plan_sha256": value["seed_plan"]["sha256"],
        "authorized_reset_contract_sha256": digest(reset_contract()),
        "initializations": list(protocol.INITIALIZATIONS), "unresolved": [], "remaining_conflicts": [],
        "fresh_precheck_authorized": False, "PPO_joint_steps": 0, "pilot_opened": False, "publishable": False}
    if any(not _strict_equal(clearance.get(key), expected) for key, expected in required_clearance.items()):
        raise RuntimeError("BC development requires model/source/recipe/exact-reset-bound seed clearance")
    if recipe != "shared_cache":
        bound.update(_checked_shared_cache(root, value, initialization))
    if recipe == "shared_b050":
        bound.update(_check_backup(root, value))
    expected_output = (root/"output/warehouse_v621"/protocol.BATCH).resolve()
    if Path(value.get("output_directory", "")).resolve() != expected_output:
        raise RuntimeError("BC development output is not the registered isolated batch")
    for p in (path, recipe_path, seeds_path, clearance_path):
        bound[str(p)] = file_sha(p)
    if sha256(raw).hexdigest() != file_sha(path):
        raise RuntimeError("BC specification changed while checking prerequisites")
    output = expected_output/f"init-{initialization}"/recipe
    output.mkdir(parents=True, exist_ok=False)
    journal = output/"reset-preauthorization.jsonl"
    journal.open("x").close()
    plan = BCDevelopmentPlan(root, path, initialization, recipe,
        "shared_cache" if recipe == "shared_cache" else "model_bc_and_spent_development",
        output, journal, tuple(MappingProxyType(r) for r in reset_schedule(initialization, recipe)),
        value["source_closure_sha256"], value["seed_plan"]["sha256"], value["model_implementation_sha256"],
        MappingProxyType(sources), MappingProxyType(bound), _CONSTRUCTION_SEAL)
    _write_new(output/"capability.started.json", {"schema": "warehouse-v621-bc-development-capability.v1",
        "initialization": initialization, "recipe": recipe, "source_closure_sha256": plan.source_closure_sha256,
        "specification_sha256": file_sha(path), "single_use_even_if_interrupted": True,
        "scheduled_reset_count": len(plan.schedule), "PPO_joint_steps": 0, "publishable": False})
    token = _ACTIVE.set(plan)
    try:
        yield plan
        plan.assert_completed()
        _write_new(output/"capability.completed.json", {"schema": "warehouse-v621-bc-capability-completed.v1",
            "initialization": initialization, "recipe": recipe, "preauthorized_resets": plan.reset_count,
            "preauthorization_journal_sha256": file_sha(journal), "requires_native_actual_reset_reconciliation": True,
            "PPO_joint_steps": 0, "fresh_precheck_authorized": False, "publishable": False})
    except BaseException as error:
        _write_new(output/"capability.failed.json", {"schema": "warehouse-v621-bc-capability-failed.v1",
            "initialization": initialization, "recipe": recipe, "preauthorized_resets": plan.reset_count,
            "preauthorization_journal_sha256": file_sha(journal), "error_type": type(error).__name__,
            "error": str(error), "scientific_retry_permitted": False, "PPO_joint_steps": 0, "publishable": False})
        raise
    finally:
        _ACTIVE.reset(token)
