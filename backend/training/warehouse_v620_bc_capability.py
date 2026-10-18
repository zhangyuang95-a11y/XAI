"""Source-bound, single-attempt shared-BC development capability.

No training is authorized by importing this module. The saved entry script
must verify a completed CV-failure decision and an independently cleared exact
reset/RNG schedule. It grants no fresh validation, PPO or publication authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .warehouse_bc_alignment import BCAlignmentRecipe

SCHEMA = "warehouse-v620-shared-bc-pilot-spec.v1"
CANDIDATE_ID = "warehouse-v620-01-shared-bc-pilot"
PILOT_PAIR_SEED = 6160
CV_LOCK_PATH = "configs/warehouse_v620_shared_bc_approved_cv_lock.json"
# Used only when generating the review bundle/default approved-lock record.
# Runtime authorization reads and binds that separate source-closed record.
CV_SOURCE_SHA256 = "5d8236dee814f6d8b83343a3d2d9ddd453d4aee9dfe4d0376448b424a8b085ed"
# Order, frame sampling and BC epoch counts are immutable; this preserves the
# 1,270 existing training episodes, with no intermediate tree installation.
STAGE_COUNTS = (
    ("initial_public_teacher", 100),
    ("initial_nn_dagger_1", 40), ("initial_nn_dagger_2", 40),
    ("joint_augmentation_sampled", 25), ("joint_augmentation_argmax", 25),
    ("risk_public_teacher", 40), ("risk_nn_visitation", 80),
    ("large_public_teacher", 200), ("large_nn_visitation", 120),
    ("deep_public_teacher", 200),
    *((f"deep_{round_index}_{kind}", 50)
      for round_index in range(1, 5) for kind in ("sampled", "argmax")),
)
assert sum(count for _, count in STAGE_COUNTS) == 1270
ALIASES = {"topology_fit_sampled": "joint_augmentation_sampled",
           "topology_fit_argmax": "joint_augmentation_argmax"}
_CONSTRUCTION_SEAL = object()
_ACTIVE_PLAN: ContextVar[BCPilotPlan | None] = ContextVar("warehouse_v620_bc_pilot_plan", default=None)


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def approved_cv_lock_record() -> dict:
    """Review default for the explicitly approved corrected CV batch."""
    return {
        "schema": "warehouse-v620-shared-bc-approved-cv-lock.v1",
        "candidate_id": "warehouse-v620-01", "batch_id": "cv-02",
        "plan_schema": "warehouse-development-cv-plan.v2",
        "recipe_schema": "warehouse-v620-development-recipe.v2",
        "source_closure_sha256": CV_SOURCE_SHA256,
        "recipe_sha256": "89b07e9681ecbcaa2d6e5a5e73e8e040e7a5d5a00831a76cffcc17ad62bd83bd",
        "script_sha256": "1c74034898097490776029201e12abaff4f11be652d62418c3e39a2ab0569d89",
        "closed_engineering_batches": ["cv-01"],
        "round_budget_contract": {
            "M0": {"scope": "historical_baseline_per_initialization_unchanged",
                   "maximum_rounds_per_initialization": 30, "maximum_total_rounds": 240,
                   "maximum_initializations": 8, "maximum_leaf_api_calls_per_role_fit": 20000},
            **{arm: {"scope": "per_role_fit_across_all_initializations",
                     "maximum_total_rounds": 30, "maximum_initializations": 8,
                     "maximum_leaf_api_calls_per_role_fit": 20000} for arm in ("M1", "M2")},
        },
    }


def recipe(gamma: float) -> dict:
    return {
        "schema": "warehouse-v620-shared-bc-pilot-recipe.v1",
        "candidate_id": CANDIDATE_ID,
        "alignment": BCAlignmentRecipe(gamma).as_dict(),
        "training_stage_counts": [[name, count] for name, count in STAGE_COUNTS],
        "bc_training_episodes": 1270, "critic_environment_resets": 2,
        "initial_public_tree_fit": True,
        "tree_gate_trainable_from_initial_bc": True,
        "mid_bc_extracted_tree_installation": False,
        "joint_augmentation_epochs": 8,
        "initial_bc": {"epochs": 20, "batch_size": 256, "learning_rate": .0003},
        "initial_dagger": {"rounds": 2, "epochs": 10},
        "risk_large_bc": {"epochs": 8, "batch_size": 128, "learning_rate": .0001},
        "deep_bc": {"rounds": 4, "epochs": 6, "final_epochs": 6},
        "gradient_clip": .5,
        "offline_collision_risk_weight": 2., "margin_weight": 0.,
        "risk_near_increment": 0., "later_near_increment": 2.,
        "low_return_increment": 8.,
        "sampled_collision_increment": 3., "argmax_collision_increment": 5.,
        "duplicate_collision_signature_cap": 8,
        "rng_schedule_binding": "paired_seed_plan.seed_binding.training_rng_schedule",
        "critic": {"epochs": 80, "batch_size": 256, "learning_rate": .001,
                   "positive_weight": 3.},
        "support_lock": "end_of_bc",
        "fresh_275_episode_collection": False, "quality_rollouts": False,
        "external_program_fit_calls": 0, "PPO_joint_steps": 0,
        "publishable": False, "admission_evidence": False,
    }


def training_rng_schedule(critic_seed: int) -> dict:
    """Declared RNG uses; numbers in distinct namespaces are not new scenes."""
    return {
        "actor_initialization_torch": PILOT_PAIR_SEED,
        "bc_global_torch_seed_reset": PILOT_PAIR_SEED,
        "initial_dagger_augmentation_torch_shuffle_shared_generator": PILOT_PAIR_SEED + 31,
        "risk_torch_shuffle": PILOT_PAIR_SEED + 201,
        "large_torch_shuffle": PILOT_PAIR_SEED + 301,
        "deep_torch_shuffle": [PILOT_PAIR_SEED + 400 + i for i in range(1, 5)],
        "final_torch_shuffle": PILOT_PAIR_SEED + 499,
        "initial_public_tree_classifier_random_states": [19, 20],
        "critic_initialization_torch": critic_seed,
        "critic_unordered_pair_split_hash_argument": critic_seed,
        "critic_minibatch_torch_shuffle": critic_seed + 1,
    }


def seed_binding(schedule: Mapping, critic_seed: int) -> dict:
    return {"initialization_pair_seed": PILOT_PAIR_SEED,
            "bc_seed_schedule": schedule, "critic_seed": critic_seed,
            "critic_environment_reset_seeds": [critic_seed, critic_seed],
            "training_rng_schedule": training_rng_schedule(critic_seed)}


def _positive_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"v6.20 BC requires a nonnegative integer {name}")
    return value


def _checked_pointer(root: Path, pointer: Mapping, name: str) -> tuple[Path, dict]:
    if not isinstance(pointer, Mapping) or set(pointer) != {"path", "sha256"}:
        raise ValueError(f"v6.20 BC {name} pointer malformed")
    path = (root / pointer["path"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or file_sha(path) != pointer["sha256"]:
        raise RuntimeError(f"v6.20 BC {name} pointer hash or location mismatch")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"v6.20 BC {name} must contain a JSON object")
    return path, value


def _checked_native_cv(root: Path, pointer: Mapping) -> tuple[tuple[str, str], ...]:
    """Replay the old source-pinned summary chain without calling load_plan.

    Its live-source check must reject the later BC source. Instead pin the
    original CV closure, verify native plan/jobs/program bytes and recompute
    every gate and final selection with the original unchanged CV script.
    """
    from experiments.warehouse import v620_protocol as v
    from scripts.run_warehouse_development_cv import (
        ROLE_SHA256, development_gate, search_budget_audit, summarize_results,
    )

    summary_path, summary = _checked_pointer(root, pointer, "native CV summary")
    directory = summary_path.parent
    approved_path = root / CV_LOCK_PATH
    approved = json.loads(approved_path.read_bytes())
    if (approved.get("schema") != "warehouse-v620-shared-bc-approved-cv-lock.v1"
            or approved.get("candidate_id") != v.CANDIDATE_ID
            or approved.get("batch_id") != v.CV_BATCH_ID
            or approved.get("batch_id") in approved.get("closed_engineering_batches", [])
            or "cv-01" not in approved.get("closed_engineering_batches", [])
            or approved.get("plan_schema") != v.CV_PLAN_SCHEMA
            or approved.get("round_budget_contract") != v.round_budget_contract()):
        raise RuntimeError("v6.20 BC approved corrected CV lock record is invalid")
    if (summary_path.name != "summary.json"
            or directory != (root / v.OUTPUT_ROOT / approved["batch_id"]).resolve()):
        raise RuntimeError("v6.20 BC requires the canonical native CV summary location")
    plan_path, plan = _checked_pointer(root,
        {"path": str(directory / "plan.json"), "sha256": summary.get("plan_sha256")}, "native CV plan")
    identity = plan.get("identity", {})
    script_path = root / "scripts/run_warehouse_development_cv.py"
    recipe_path = root / v.RECIPE_PATH
    recipe_data = json.loads(recipe_path.read_bytes())
    if (plan.get("schema") != approved["plan_schema"]
            or plan.get("batch_id") != approved["batch_id"]
            or plan.get("round_budget_contract") != approved["round_budget_contract"]
            or plan.get("candidate_id") != v.CANDIDATE_ID
            or plan.get("output_directory") != str(directory)
            or plan.get("jobs") != v.jobs() or plan.get("folds") != v.grouped_folds()
            or plan.get("fold_assignment_sha256") != v.digest(v.grouped_folds())
            or plan.get("gates") != v.GATES or plan.get("selection_arms") != ["M1", "M2"]
            or identity.get("source_closure_sha256") != approved["source_closure_sha256"]
            or identity.get("script_sha256") != file_sha(script_path)
            or identity.get("script_sha256") != approved["script_sha256"]
            or identity.get("recipe_sha256") != file_sha(recipe_path)
            or identity.get("recipe_sha256") != approved["recipe_sha256"]
            or recipe_data.get("schema") != approved["recipe_schema"]
            or recipe_data.get("cv", {}).get("batch_id") != approved["batch_id"]
            or recipe_data.get("cv", {}).get("round_budget_contract") != approved["round_budget_contract"]
            or identity.get("actor_sha256") != v.ACTOR_SHA256
            or identity.get("actor_role_sha256") != ROLE_SHA256
            or any(plan.get(key) is not False for key in
                   ("publishable", "admission_evidence", "fresh_validation_evidence"))
            or any(type(plan.get(key)) is not int or plan[key] != 0 for key in
                   ("environment_resets", "actor_updates", "PPO_joint_steps"))):
        raise RuntimeError("v6.20 BC historical CV plan/source/recipe contract mismatch")
    bound_paths = [summary_path, plan_path, script_path, recipe_path, approved_path]
    actor_path = Path(identity["actor_path"]).resolve(strict=True)
    if not actor_path.is_relative_to(root) or file_sha(actor_path) != v.ACTOR_SHA256:
        raise RuntimeError("v6.20 BC historical CV Actor hash mismatch")
    bound_paths.append(actor_path)
    for path, expected in identity["inputs"]["files_sha256"].items():
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(root) or file_sha(path) != expected:
            raise RuntimeError("v6.20 BC historical CV input hash mismatch")
        bound_paths.append(path)
    references = summary.get("job_results")
    if not isinstance(references, dict) or set(references) != {j["job_id"] for j in v.jobs()}:
        raise RuntimeError("v6.20 BC native CV summary requires all eighteen jobs")
    results = []
    for job in plan["jobs"]:
        job_dir = directory / "jobs" / job["job_id"]
        result_path, result = _checked_pointer(root, references[job["job_id"]], "native CV job result")
        completed_path = job_dir / "completed.json"
        completed = json.loads(completed_path.read_bytes())
        program_path = Path(result["program_path"]).resolve(strict=True)
        if (program_path != job_dir / "program.json"
                or file_sha(program_path) != result.get("program_sha256")
                or file_sha(program_path) != completed.get("program_sha256")):
            raise RuntimeError("v6.20 BC native CV program location or hash mismatch")
        program_data = json.loads(program_path.read_bytes())
        actual_budget = search_budget_audit(program_data.get("report", {}), job["arm"])
        budget_path = job_dir / "search-budget-audit.json"
        recorded_budget = json.loads(budget_path.read_bytes())
        if (result_path != job_dir / "result.json" or program_path != job_dir / "program.json"
                or result.get("schema") != "warehouse-development-cv-job-result.v1"
                or result.get("job") != job or result.get("plan_sha256") != file_sha(plan_path)
                or completed.get("plan_sha256") != file_sha(plan_path)
                or result.get("identity_sha256") != v.digest(identity)
                or result.get("fold_assignment_sha256") != plan["folds"][job["fold"]]["assignment_sha256"]
                or completed.get("result_sha256") != references[job["job_id"]]["sha256"]
                or result.get("program_sha256") != file_sha(program_path)
                or completed.get("program_sha256") != file_sha(program_path)
                or result.get("actor_weights_unchanged") is not True
                or result.get("excluded_frames") != 0
                or any(result.get(key) is not False for key in
                       ("publishable", "admission_evidence", "fresh_validation_evidence"))
                or any(type(result.get(key)) is not int or result[key] != 0 for key in
                       ("environment_resets", "PPO_joint_steps"))
                or result.get("development_gate") != development_gate(result["metrics"]["validation"])
                or actual_budget.get("passed") is not True
                or result.get("search_budget_audit") != actual_budget
                or recorded_budget != actual_budget
                or completed.get("passed") is not result["development_gate"]["passed"]):
            raise RuntimeError("v6.20 BC native CV job completion/program/gate chain mismatch")
        results.append(result)
        bound_paths.extend([result_path, completed_path, program_path, budget_path])
    recomputed = {"schema": "warehouse-development-cv-summary.v1", "candidate_id": v.CANDIDATE_ID,
        "plan_sha256": file_sha(plan_path), "job_results": references, **summarize_results(results)}
    if (summary != recomputed or summary["chosen_method_arm"] is not None
            or summary["next_stage"] != "common_bc_structure_learning_before_fresh_validation"):
        raise RuntimeError("v6.20 BC native CV results do not require common BC")
    return tuple((str(path), file_sha(path)) for path in dict.fromkeys(bound_paths))


@dataclass(frozen=True)
class BCPilotPlan:
    spec_sha256: str
    candidate_id: str
    pair_seed: int
    gamma: float
    run_id: str
    stage_cases: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    critic_seed: int
    output_directory: Path
    recipe_file_sha256: str
    expected_source_closure_sha256: str
    bound_files: tuple[tuple[str, str], ...]
    _seal: object = field(repr=False, compare=False)

    def canonical_stage_name(self, name: str) -> str:
        return ALIASES.get(name, name)

    def cases(self, name: str) -> tuple[tuple[int, int], ...]:
        require_active_plan(self)
        return dict(self.stage_cases)[self.canonical_stage_name(name)]

    def scenarios(self, name: str) -> list[int]:
        return [scene for scene, _ in self.cases(name)]

    def action_seeds(self, name: str) -> dict[int, int]:
        return dict(self.cases(name))

    def rng(self, purpose: str):
        require_active_plan(self)
        return training_rng_schedule(self.critic_seed)[purpose]

    def assert_immutable(self, runner, spec: Mapping) -> None:
        require_active_plan(self)
        if (digest(spec) != self.spec_sha256 or
                runner.source_closure_sha256() != self.expected_source_closure_sha256):
            raise RuntimeError("v6.20 BC spec or source changed during execution")
        for path, expected in self.bound_files:
            if file_sha(Path(path)) != expected:
                raise RuntimeError("v6.20 BC bound file changed during execution")


def require_active_plan(plan: BCPilotPlan) -> None:
    if (not isinstance(plan, BCPilotPlan) or plan._seal is not _CONSTRUCTION_SEAL
            or _ACTIVE_PLAN.get() is not plan):
        raise RuntimeError("v6.20 BC requires its verified opaque active capability")


@contextmanager
def _activate_verified_plan(plan: BCPilotPlan):
    if plan._seal is not _CONSTRUCTION_SEAL or _ACTIVE_PLAN.get() is not None:
        raise RuntimeError("v6.20 BC capability is invalid or already active")
    token = _ACTIVE_PLAN.set(plan)
    try:
        yield plan
    finally:
        _ACTIVE_PLAN.reset(token)


def validate_spec(spec: Mapping, *, runner, main_path: Path) -> BCPilotPlan:
    """Read and verify all external authorization evidence before any reset."""
    root = runner._SOURCE_ROOT.resolve()
    required_keys = {"schema", "candidate_id", "pair_seed", "gamma", "run_id",
        "bc_seed_schedule", "critic_seed", "source_closure_sha256", "recipe",
        "cv_summary", "seed_clearance", "paired_seed_plan", "output_directory"}
    if not isinstance(spec, Mapping) or set(spec) != required_keys:
        raise ValueError("v6.20 BC spec keys differ from the immutable pilot contract")
    gamma = float(BCAlignmentRecipe(spec["gamma"]).gamma)
    expected_run = "pair-6160-gamma-001" if gamma == .01 else "pair-6160-gamma-003"
    if (spec["schema"] != SCHEMA or spec["candidate_id"] != CANDIDATE_ID
            or type(spec["pair_seed"]) is not int or spec["pair_seed"] != PILOT_PAIR_SEED
            or spec["run_id"] != expected_run):
        raise RuntimeError("v6.20 BC pilot identity is not registered")
    if main_path.resolve(strict=True) != (root / "scripts/run_warehouse_v620_shared_bc_pilot.py").resolve(strict=True):
        raise RuntimeError("v6.20 BC requires the saved registered pilot script")
    if spec["source_closure_sha256"] != runner.source_closure_sha256():
        raise RuntimeError("v6.20 BC source closure mismatch")
    schedule = spec["bc_seed_schedule"]
    if not isinstance(schedule, Mapping) or set(schedule) != {name for name, _ in STAGE_COUNTS}:
        raise ValueError("v6.20 BC stage schedule differs from registered purposes")
    stage_cases = []
    for name, count in STAGE_COUNTS:
        cases = schedule[name]
        if not isinstance(cases, list) or len(cases) != count:
            raise ValueError("v6.20 BC stage episode count mismatch")
        rows = []
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"scenario_seed", "action_seed"}:
                raise ValueError("v6.20 BC scene/action seed record malformed")
            rows.append((_positive_integer(case["scenario_seed"], "scenario_seed"),
                         _positive_integer(case["action_seed"], "action_seed")))
        stage_cases.append((name, tuple(rows)))
    scenes = [s for _, cases in stage_cases for s, _ in cases]
    rngs = [r for _, cases in stage_cases for _, r in cases]
    critic_seed = _positive_integer(spec["critic_seed"], "critic_seed")
    training_rng = training_rng_schedule(critic_seed)
    training_rng_values = {item for value in training_rng.values()
                           for item in (value if isinstance(value, list) else [value])}
    if (len(set(scenes)) != 1270 or len(set(rngs)) != 1270
            or set(scenes) & set(rngs)
            or (set(scenes) | set(rngs)) & training_rng_values
            or PILOT_PAIR_SEED in {critic_seed, critic_seed + 1}):
        raise RuntimeError("v6.20 BC initialization, scene, action or critic seeds overlap")
    paired_path, paired = _checked_pointer(root, spec["paired_seed_plan"], "paired seed plan")
    if (paired_path != (root / "configs/warehouse_v620_shared_bc_seed_plan.json").resolve()
            or paired != {"schema": "warehouse-v620-paired-bc-seeds.v1",
                          "candidate_id": CANDIDATE_ID, "pair_seed": PILOT_PAIR_SEED,
                          "gamma_candidates": [.01, .03],
                          "seed_binding": seed_binding(schedule, critic_seed)}):
        raise RuntimeError("v6.20 BC gamma recipes require the same source-bound paired seed plan")
    recipe_path, declared_recipe = _checked_pointer(root, spec["recipe"], "recipe")
    if declared_recipe != recipe(gamma):
        raise RuntimeError("v6.20 BC executable recipe differs from fixed implementation")
    cv_bound_files = _checked_native_cv(root, spec["cv_summary"])
    clearance_path, clearance = _checked_pointer(root, spec["seed_clearance"], "seed clearance")
    if (clearance.get("schema") != "warehouse-v620-bc-seed-clearance.v1"
            or clearance.get("approved_for_shared_bc") is not True
            or clearance.get("seed_binding_sha256") != digest(seed_binding(schedule, critic_seed))
            or type(clearance.get("conflict_count")) is not int
            or clearance.get("conflict_count") != 0):
        raise RuntimeError("v6.20 BC exact scene/RNG/critic schedule is not independently cleared")
    output = Path(spec["output_directory"]).resolve()
    expected_output = root / "output/warehouse_v620_bc_pilot" / CANDIDATE_ID / expected_run
    if output != expected_output.resolve() or output.exists():
        raise RuntimeError("v6.20 BC output must be an isolated new pilot directory")
    bound = tuple((str(path), file_sha(path)) for path in (
        main_path, recipe_path, paired_path, clearance_path)) + cv_bound_files
    return BCPilotPlan(digest(spec), CANDIDATE_ID, PILOT_PAIR_SEED, gamma,
        expected_run, tuple(stage_cases), critic_seed, output, file_sha(recipe_path),
        spec["source_closure_sha256"], bound, _CONSTRUCTION_SEAL)


def expected_resets(plan: BCPilotPlan) -> list[dict]:
    """1,270 real BC episodes plus the two real critic environment resets."""
    result = []
    for name, cases in plan.stage_cases:
        result.extend({"phase": "formal_bc", "seed": scene} for scene, _ in cases)
        if name == "joint_augmentation_argmax":
            result.extend({"phase": "formal_bc", "seed": plan.critic_seed} for _ in range(2))
    if len(result) != 1272:
        raise RuntimeError("v6.20 BC reset budget drifted")
    return result


def runner_spec(plan: BCPilotPlan, original: Mapping) -> dict:
    """Only data required by the existing BC engine; no route to PPO specs."""
    return {
        "candidate_id": plan.candidate_id, "pair_seed": plan.pair_seed,
        "source_closure_sha256": plan.expected_source_closure_sha256,
        "warmup_actor_path": str(plan.output_directory / "warmup.actor.npz"),
        "final_bc_actor_path": str(plan.output_directory / "final-bc.actor.npz"),
        "bc_data_seed_schedule": {name: [s for s, _ in cases] for name, cases in plan.stage_cases},
        "bc_data_seed_schedule_sha256": digest({name: [s for s, _ in cases] for name, cases in plan.stage_cases}),
        "bc_critic_reset_schedule": {"seed": plan.critic_seed, "environment_reset_count": 2},
        "bc_critic_reset_schedule_sha256": digest({"seed": plan.critic_seed, "environment_reset_count": 2}),
        "v620_pilot_original_spec": dict(original),
    }


def authorization_fields(plan: BCPilotPlan, *, runner, initial_roles: Mapping) -> dict:
    return {
        "authorization_kind": "warehouse-v620-shared-bc-pilot-only-v1",
        "candidate_id": plan.candidate_id, "pair_seed": plan.pair_seed,
        "gamma": plan.gamma, "run_id": plan.run_id,
        "original_spec_sha256": plan.spec_sha256,
        "source_closure_sha256": plan.expected_source_closure_sha256,
        "recipe_file_sha256": plan.recipe_file_sha256,
        "bound_files_sha256": digest(plan.bound_files),
        "initial_role_sha256": dict(initial_roles),
        "expected_reset_sequence_sha256": digest(expected_resets(plan)),
        "output_directory": str(plan.output_directory),
        "action_physics_config_sha256": runner.ACTION_PHYSICS_CONFIG_SHA256,
        "PPO_joint_steps": 0, "publishable": False,
        "fresh_validation_episodes": 0,
    }


def execute(spec: Mapping, *, authorization_receipt_path: str | Path, runner,
            main_path: Path) -> dict:
    """Saved-entry-only execution; writes an exclusive receipt on all attempts."""
    import torch
    import sys

    if not runner.FORMAL_TRAINING_DISABLED or runner._LOCKED_EXECUTION_USED_IN_PROCESS:
        raise RuntimeError("v6.20 BC pilot capability unavailable")
    executing_main = getattr(sys.modules["__main__"], "__file__", None)
    if (not isinstance(executing_main, str)
            or Path(executing_main).resolve(strict=True) != main_path.resolve(strict=True)):
        raise RuntimeError("v6.20 BC capability must execute in its authorized saved main script")
    plan = validate_spec(spec, runner=runner, main_path=main_path)
    authorization_path = Path(authorization_receipt_path).resolve(strict=True)
    authorization_sha = file_sha(authorization_path)
    authorization = json.loads(authorization_path.read_bytes())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.pair_seed)
        initial_roles = runner.role_hashes(runner.WarehouseV6Actor())
    required = authorization_fields(plan, runner=runner, initial_roles=initial_roles)
    if (not isinstance(authorization, dict) or set(authorization) != set(required)
            or authorization != required):
        raise RuntimeError("v6.20 BC pilot authorization identity mismatch")
    runner._LOCKED_EXECUTION_USED_IN_PROCESS = True
    plan.output_directory.mkdir(parents=True, exist_ok=False)
    journal = plan.output_directory / "bc-pilot.resets.jsonl"
    receipt_path = plan.output_directory / "bc-pilot.receipt.json"
    report_path = plan.output_directory / "bc-pilot.training.json"
    training_spec = runner_spec(plan, spec)
    expected = expected_resets(plan)
    actual, report, error, status = [], None, None, "failed"
    try:
        torch.manual_seed(plan.pair_seed)
        model = runner.WarehouseV6Actor()
        with runner.capture_stage1_resets(journal) as actual:
            with _activate_verified_plan(plan):
                token = runner._ACTIVE_LOCKED_EXECUTION.set(runner._LOCKED_BC_MARKER)
                try:
                    with runner.stage1_reset_phase("formal_bc"):
                        report = runner._pretrain_pair_production(
                            model, pair_seed=plan.pair_seed, spec=training_spec,
                            _bc_plan=plan)
                    plan.assert_immutable(runner, spec)
                    if (file_sha(authorization_path) != authorization_sha
                            or report["initial_role_sha256"] != initial_roles
                            or runner.role_hashes(model) != report["pretrained_role_sha256"]
                            or report["PPO_joint_steps"] != 0
                            or report["publishable"] is not False):
                        raise RuntimeError("v6.20 BC returned model or authority drifted")
                    projected = [{"phase": r["phase"], "seed": r["seed"]} for r in actual]
                    if projected != expected:
                        raise RuntimeError("v6.20 BC actual reset sequence differs from authorization")
                finally:
                    runner._ACTIVE_LOCKED_EXECUTION.reset(token)
        with report_path.open("x") as handle:
            json.dump(report, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        status = "bc_completed_not_admitted"
        return {"report_path": str(report_path), "report_sha256": file_sha(report_path),
                "receipt_path": str(receipt_path), "status": status}
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        projection = [{"phase": r["phase"], "seed": r["seed"]} for r in actual]
        authorization_unchanged, authorization_hash_error = False, None
        try:
            authorization_unchanged = file_sha(authorization_path) == authorization_sha
        except OSError as exc:
            # Losing the authorization file cannot erase consumed-scene evidence
            # or replace an interruption with a secondary hash exception.
            authorization_hash_error = f"{type(exc).__name__}: {exc}"
        receipt = {
            **required, "status": status, "error": error,
            "authorization_receipt_sha256": authorization_sha,
            "authorization_unchanged": authorization_unchanged,
            "authorization_hash_error": authorization_hash_error,
            "actual_resets": list(actual), "actual_reset_count": len(actual),
            "expected_reset_count": len(expected),
            "reset_prefix_matches": projection == expected[:len(projection)],
            "reset_full_sequence_matches": projection == expected,
            "journal_path": str(journal), "journal_sha256": file_sha(journal) if journal.is_file() else None,
            "bc_report_sha256": file_sha(report_path) if report_path.is_file() else None,
        }
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
