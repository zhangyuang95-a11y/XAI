"""Synthetic engineering recovery receipts; no real models, fit or rollouts."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.warehouse import v622_recovery_protocol as v
from scripts import run_warehouse_v622_capacity_recovery as runner
from scripts import audit_warehouse_v622_capacity_recovery as auditor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "sha256": sha256(raw).hexdigest()}


@pytest.fixture
def old_batch(tmp_path, monkeypatch):
    root = tmp_path
    old = root / v.ORIGINAL_BATCH_PATH
    old_source = {"old_science.py": sha256(b"original scientific source\n").hexdigest()}
    write(root / "old_science.py", b"original scientific source\n")
    old_source_sha = v.digest(old_source)
    monkeypatch.setattr(v, "ORIGINAL_SOURCE_SHA256", old_source_sha)
    monkeypatch.setattr(v.original, "source_closure_files", lambda r: {
        "old_science.py": sha256((Path(r) / "old_science.py").read_bytes()).hexdigest()})
    write(root / v.ORIGINAL_RECIPE_PATH, v.original.executable_recipe())
    plan = {"schema": "warehouse-v622-capacity-plan.v1", "stage": "primary",
        "candidate_id": v.CANDIDATE_ID, "jobs": v.jobs("primary"),
        "scientific_retry_permitted": False,
        "identity": {"source_closure_sha256": old_source_sha, "source_files_sha256": old_source}}
    plan_ptr = write(old / "plan.json", plan)
    monkeypatch.setattr(v, "ORIGINAL_PLAN_SHA256", plan_ptr["sha256"])
    manifest_ptr = write(old / "source-manifest.json", {
        "source_closure_sha256": old_source_sha, "files_sha256": old_source})
    monkeypatch.setattr(v, "ORIGINAL_MANIFEST_SHA256", manifest_ptr["sha256"])
    interrupted_ids = {"fold-1-robot_2-A32", "fold-2-robot_1-A32", "fold-2-robot_2-A32"}
    records = []
    for job in v.jobs("primary"):
        folder = old / "jobs" / job["job_id"]
        row = {"job": job,
            "started": write(folder / "started.json", {"job": job, "plan_sha256": plan_ptr["sha256"]}),
            "log": write(old / "logs" / (job["job_id"] + ".log"), b"synthetic historical lower bound\n")}
        if job["job_id"] in interrupted_ids:
            row.update(status="interrupted_without_search_checkpoint", result=None, completion=None,
                last_logged_progress={"convex_solver_calls": 11, "global_rounds": 2, "leaf_api_calls": 5},
                unlogged_in_flight_work="unknown; logged counters are lower bounds only")
        else:
            result = write(folder / "result.json", {"job": job, "plan_sha256": plan_ptr["sha256"],
                "status": "completed", "development_gate": {"passed": False}})
            row.update(status="completed_before_interruption", result=result,
                completion=write(folder / "completed.json", {"plan_sha256": plan_ptr["sha256"],
                    "result_sha256": result["sha256"], "passed": False}))
        records.append(row)
    receipt = {"schema": "warehouse-v622-engineering-interruption.v1", "candidate_id": v.CANDIDATE_ID,
        "batch_id": "capacity-01", "complete_batch": False, "completed_jobs": 9, "interrupted_jobs": 3,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "heldout_used": False, "sealed_audit_used": False, "permits_training": False,
        "pilot_opened": False, "scientific_method_failure_claimed": False,
        "scientific_timeout_claimed": False, "selected_arm": None, "source_closure_sha256": old_source_sha,
        "plan": plan_ptr, "source_manifest": manifest_ptr,
        "process_inspection": {"checkpoint_resume_available": False, "matching_scientific_processes": 0},
        "recovery_policy": "new full 12-job batch with new source identity; no cross-batch selection or stitching; old compute reported separately",
        "jobs": records}
    def resign(value):
        ptr = write(root / v.INTERRUPTION_PATH, value)
        monkeypatch.setattr(v, "INTERRUPTION_SHA256", ptr["sha256"])
    resign(receipt)
    monkeypatch.setattr(v, "ROOT", root)
    for relative in v.RECOVERY_SOURCE_FILES:
        if relative != v.RECIPE_PATH:
            write(root / relative, b"synthetic new recovery source\n")
    return root, receipt, resign


def test_recipe_is_full_replacement_and_old_scientific_api_unchanged(old_batch):
    root, receipt, _ = old_batch
    assert v.checked_interruption(root) == receipt
    recipe = v.executable_recovery_recipe(root)
    assert recipe["replacement"]["full_primary_jobs"] == v.original.jobs("primary")
    assert len(recipe["replacement"]["full_primary_jobs"]) == 12
    assert len(recipe["replacement"]["conditional_fallback_jobs"]) == 6
    assert not recipe["replacement"]["reuse_old_completed_jobs"]
    assert not recipe["replacement"]["resume_partial_search"]
    assert not recipe["replacement"]["cross_batch_selection"]
    assert not recipe["replacement"]["cross_batch_stitching"]
    assert recipe["replacement"]["report_old_compute_separately"]
    assert v.grouped_folds() == v.original.grouped_folds()
    assert v.GATES == v.original.GATES and v.LIMITS == v.original.LIMITS
    assert v.round_budget_contract() == v.original.round_budget_contract()
    assert v.output_directory("primary") == root / v.OUTPUT_ROOT / "capacity-02"
    assert v.output_directory("fallback") == root / v.OUTPUT_ROOT / "fallback-02"
    with pytest.raises(ValueError):
        v.output_directory("other")


def test_recipe_write_is_exclusive_and_combines_both_recipe_hashes(old_batch):
    root, _, _ = old_batch
    path = v.write_recipe(root)
    raw_sha = sha256(path.read_bytes()).hexdigest()
    expected = v.digest({"original_recipe_sha256": v.original.validate_recipe(root),
                         "recovery_recipe_sha256": raw_sha})
    assert v.validate_recipe(root) == expected
    with pytest.raises(FileExistsError):
        v.write_recipe(root)
    value = json.loads(path.read_bytes())
    value["replacement"]["cross_batch_stitching"] = True
    write(path, value)
    with pytest.raises(RuntimeError, match="recipe changed"):
        v.validate_recipe(root)


def test_original_source_must_match_and_new_executor_is_in_closure(old_batch):
    root, _, _ = old_batch
    v.write_recipe(root)
    files = v.source_closure_files(root)
    assert set(files) == {"old_science.py", *v.RECOVERY_SOURCE_FILES}
    assert "scripts/execute_warehouse_v622_recovery_batch.py" in files
    before = v.source_closure_sha256(root)
    (root / "scripts/execute_warehouse_v622_recovery_batch.py").write_bytes(b"changed new executor")
    assert v.source_closure_sha256(root) != before
    (root / "old_science.py").write_bytes(b"changed frozen science")
    with pytest.raises(RuntimeError, match="frozen scientific source"):
        v.source_closure_files(root)


def test_receipt_raw_hash_tampering_is_rejected(old_batch):
    root, _, _ = old_batch
    path = root / v.INTERRUPTION_PATH
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        v.checked_interruption(root)


@pytest.mark.parametrize("field,value", [
    ("completed_jobs", 10), ("completed_jobs", True), ("interrupted_jobs", 2),
    ("scientific_timeout_claimed", True), ("scientific_method_failure_claimed", True),
    ("heldout_used", True), ("sealed_audit_used", True), ("selected_arm", "A32"),
    ("environment_resets", False), ("permits_training", True),
])
def test_semantically_wrong_disposition_cannot_be_resigned(old_batch, field, value):
    root, receipt, resign = old_batch
    receipt[field] = value
    resign(receipt)
    with pytest.raises(RuntimeError):
        v.checked_interruption(root)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reordered", "checkpoint", "completion", "failed", "summary", "old_audit", "wrong_pointer", "log_tamper"])
def test_no_partial_or_completed_old_result_can_enter_replacement(old_batch, mutation):
    root, receipt, resign = old_batch
    interrupted = next(r for r in receipt["jobs"] if r["status"].startswith("interrupted"))
    folder = root / v.ORIGINAL_BATCH_PATH / "jobs" / interrupted["job"]["job_id"]
    if mutation == "duplicate":
        receipt["jobs"][-1] = deepcopy(receipt["jobs"][0])
    elif mutation == "missing":
        receipt["jobs"].pop()
    elif mutation == "reordered":
        receipt["jobs"] = list(reversed(receipt["jobs"]))
    elif mutation == "checkpoint":
        write(folder / "search-checkpoint.npz", b"synthetic checkpoint")
    elif mutation == "completion":
        write(folder / "completed.json", {"passed": False})
    elif mutation == "failed":
        write(folder / "failed.json", {"error": "not an interruption"})
    elif mutation in ("summary", "old_audit"):
        write(root / v.ORIGINAL_BATCH_PATH / ("summary.json" if mutation == "summary" else "independent-audit.json"), {})
    elif mutation == "wrong_pointer":
        interrupted["log"] = deepcopy(receipt["jobs"][0]["log"])
    elif mutation == "log_tamper":
        Path(interrupted["log"]["path"]).write_bytes(b"tampered")
    resign(receipt)
    with pytest.raises(RuntimeError):
        v.checked_interruption(root)


def test_runner_adapts_only_explicit_scope_and_restores_on_failure(monkeypatch):
    prior_v, prior_output = runner.base.v, runner.base.output_directory
    calls = []
    monkeypatch.setattr(runner.recovery, "validate_recipe", lambda root: calls.append("recipe"))
    def fake_main(argv):
        assert runner.base.v is v
        assert runner.base.output_directory is v.output_directory
        calls.append(argv)
        raise RuntimeError("synthetic dispatch failure")
    monkeypatch.setattr(runner.base, "main", fake_main)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.main(["--prepare", "--expected-source-sha256", "synthetic"])
    assert runner.base.v is prior_v and runner.base.output_directory is prior_output
    assert calls == ["recipe", ["--prepare", "--expected-source-sha256", "synthetic"]]


def test_write_recipe_mode_cannot_dispatch_or_combine_with_scientific_mode(old_batch, monkeypatch, capsys):
    root, _, _ = old_batch
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner.base, "main", lambda *args: pytest.fail("metadata mode dispatched science"))
    monkeypatch.setattr(v, "checked_actor", lambda *args: pytest.fail("metadata mode loaded Actor"))
    monkeypatch.setattr(v, "checked_inputs", lambda *args: pytest.fail("metadata mode loaded traces"))
    with pytest.raises(ValueError, match="exclusive"):
        runner.main(["--write-recipe", "--prepare"])
    assert runner.main(["--write-recipe"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scientific_jobs_started"] == 0 and report["environment_resets"] == 0


def test_auditor_validates_plan_under_recovery_scope_but_math_remains_independent(monkeypatch, tmp_path):
    events = []
    prior_runner_v, prior_output, prior_auditor_v = runner.base.v, runner.base.output_directory, auditor.base.v
    monkeypatch.setattr(v, "validate_recipe", lambda root: events.append("recipe"))
    def load(path):
        assert runner.base.v is v and runner.base.output_directory is v.output_directory
        events.append("immutable-plan-only")
        return Path(path), {}
    def audit(path):
        assert auditor.base.v is v
        assert runner.base.v is prior_runner_v
        events.append("independent-math")
        return {"passed": True}
    monkeypatch.setattr(runner.base, "load_plan", load)
    monkeypatch.setattr(auditor.base, "audit", audit)
    assert auditor.audit(tmp_path / "plan.json") == {"passed": True}
    assert events == ["recipe", "immutable-plan-only", "independent-math"]
    assert runner.base.v is prior_runner_v and runner.base.output_directory is prior_output
    assert auditor.base.v is prior_auditor_v


def test_incomplete_replacement_still_cannot_select():
    rows = [{"job": job, "status": "completed", "budget_audit": {"passed": True},
        "metrics": {"validation": {}}, "leaf_count": 1} for job in v.jobs("primary")]
    with pytest.raises(RuntimeError, match="every registered"):
        v.summarize_units(rows[:-1])


def test_original_prepare_loader_with_synthetic_inputs_uses_only_new_directory(old_batch, monkeypatch):
    root, _, _ = old_batch
    actor_path = root / "synthetic-actor.npz"
    actor_ptr = write(actor_path, b"synthetic actor bytes; not an NPZ or real model")
    monkeypatch.setattr(v, "ACTOR_SHA256", actor_ptr["sha256"])
    monkeypatch.setattr(v, "checked_actor", lambda r: (object(), actor_path))
    monkeypatch.setattr(v, "checked_inputs", lambda r: ({}, {"files_sha256": {}}))
    monkeypatch.setattr(v, "target_identity", lambda actor, episodes: {"synthetic": True})
    monkeypatch.setattr(runner.base, "ROOT", root)
    monkeypatch.setattr(runner.base, "runtime_identity", lambda: {"synthetic": True})
    monkeypatch.setattr(runner.base, "baseline_binding", lambda: {"synthetic_A16": True})
    v.write_recipe(root)
    expected = v.source_closure_sha256(root)
    with runner.recovery_scope():
        path = runner.base.prepare("primary", expected)
        assert path == root / v.OUTPUT_ROOT / "capacity-02" / "plan.json"
        _, plan = runner.base.load_plan(path)
        assert plan["jobs"] == v.jobs("primary")
        assert len(plan["jobs"]) == 12 and plan["identity"]["recipe_sha256"] == v.validate_recipe(root)
        assert plan["identity"]["source_closure_sha256"] == expected
        assert set(plan["identity"]["source_files_sha256"]) == {"old_science.py", *v.RECOVERY_SOURCE_FILES}
        with pytest.raises(FileExistsError):
            runner.base.prepare("primary", expected)
        with pytest.raises(FileNotFoundError):
            runner.base.prepare("fallback", expected)
        old_plan = root / v.ORIGINAL_BATCH_PATH / "plan.json"
        write(old_plan.parent / "plan.claim.json", {"sha256": v.ORIGINAL_PLAN_SHA256, "single_use": True})
        with pytest.raises(RuntimeError, match="immutable capacity plan"):
            runner.base.load_plan(old_plan)
