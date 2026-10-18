from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_warehouse_v620_frozen_gate as g


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def seed_plan():
    parts, offset = {}, 0
    for split, count in (("fit", 100), ("calibration", 25), ("eligibility", 75), ("validation", 75)):
        parts[split] = [{"base": 10000+offset+i, "task": (offset+i)%3+1,
            "scenario_seed": (10000+offset+i)*10+(offset+i)%3+1,
            "action_seed": 2000000000+offset+i,
            "sampling_mode": "argmax" if split == "fit" and i >= 50 else "sampled"}
            for i in range(count)]
        offset += count
    independent = [{"base": base, "task": task, "stream": stream,
        "scenario_seed": base*10+task, "action_seed": 2000010000+(base-20000)*9+(task-1)*3+stream,
        "sampling_mode": "sampled"} for base in range(20000, 20024) for task in (1, 2, 3) for stream in (0, 1, 2)]
    return {"splits": parts, "independent": independent}


def metric(n, coverage=1.):
    rows = int(n*coverage)
    common = {"coverage": coverage, "agreement": 1., "kl": .05,
        "expected_actionable_coverage": coverage*.99, "actionable_coverage": coverage,
        "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0,
        "rows": rows, "sample_count": n, "actionable_rows": rows,
        "expected_actionable_rows": rows*.99}
    return {"feedback": dict(common), "explanation": dict(common)}


class FakeRuntime:
    def __init__(self, *, internal_pass=True, interrupt_at=None, source_drift_at=None, replay_interrupt=False):
        self.source_value, self.calls, self.fit_calls, self.replay_calls = "synthetic-source", [], [], 0
        self.internal_pass = internal_pass
        self.interrupt_at, self.source_drift_at = interrupt_at, source_drift_at
        self.replay_interrupt = replay_interrupt

    def source(self):
        return self.source_value

    def load(self, artifact):
        return SimpleNamespace(artifact_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"])

    def identity(self, actor):
        return actor.role_sha256

    def collect(self, actor, case, journal):
        self.calls.append(case)
        records = [{"seed": case["scenario_seed"]}]
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps(records[0])+"\n")
        if len(self.calls) == self.interrupt_at:
            exc = KeyboardInterrupt("synthetic interrupted collection")
            exc.warehouse_reset_audit = {"resets": records, "actual_reset_count": 1,
                "journal_path": str(journal), "journal_sha256": g.file_sha(journal)}
            raise exc
        if len(self.calls) == self.source_drift_at:
            self.source_value = "changed-after-collection"
        rows = {role: [{"obs": np.zeros(2), "mask": np.ones(5), "base_seed": case["scenario_seed"],
            "episode_id": 1, "frame": 0, "sampled_action": "WAIT", "episode": str(case["scenario_seed"])}] for role in g.ROLES}
        return rows, {"reset_audit": {"resets": records}, "episode_outcomes": [{"seed": case["scenario_seed"],
            "action_seed": case["action_seed"], "deliveries": 6, "collisions": 0,
            "shutdown_episode": False, "frame_count": 1}]}

    def trace(self, rows, split, cases, directory, actor):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory/f"{split}.json"
        g.write_new(target, {"synthetic": True, "rows": len(rows[g.ROLES[0]])})
        return g.pointer(target)

    def fit(self, role, actor, parts, method):
        self.fit_calls.append((role, method))
        return SimpleNamespace(role=role)

    def save(self, program, path, role_sha):
        g.write_new(path, {"role": program.role, "synthetic": True})
        return program

    def internal_metrics(self, program, actor, rows):
        return metric(len(rows), coverage=1. if self.internal_pass else .1)

    def metrics(self, program, actor, rows):
        return metric(len(rows))

    def quality(self, rows, outcome, actor):
        return {"seed": outcome["seed"], "deliveries": outcome["deliveries"],
            "collisions": outcome["collisions"], "shutdown_episode": outcome["shutdown_episode"],
            "quality_trace": {"synthetic": True}}

    def cluster_quality(self, episodes):
        assert len({r["base"] for r in episodes}) == 24
        return {"base_scene_count": 24, "synthetic": True}

    def replay(self, artifact, episodes, directory, reset_sink):
        self.replay_calls += 1
        if self.replay_interrupt:
            reset_sink.append({"seed": episodes[0]["seed"]})
            raise KeyboardInterrupt("synthetic interrupted physical replay")
        reset_sink.extend({"seed": row["seed"]} for row in episodes)
        return {"passed": True, "synthetic": True, "episodes": len(episodes)}


def inputs(tmp_path, runtime):
    actor = tmp_path/"not-a-real-model.npz"
    actor.write_bytes(b"synthetic fixture, not NN weights")
    recipe, selection = tmp_path/"recipe.json", tmp_path/"selection.json"
    put(recipe, {"synthetic": True})
    put(selection, {"locked": True, "program_method": g.METHODS[0], "actor_sha256": g.file_sha(actor)})
    model_path = tmp_path/"model-lock.json"
    put(model_path, {"schema": "warehouse-v620-model-method-lock.v1", "locked": True,
        "candidate_id": "warehouse-v620-synthetic-only", "program_method": g.METHODS[0],
        "source_closure_sha256": runtime.source(), "recipe": g.pointer(recipe),
        "method_selection_receipt": g.pointer(selection),
        "actor": {"path": str(actor), "sha256": g.file_sha(actor),
                  "role_sha256": {"robot_1": "synthetic-one", "robot_2": "synthetic-two"}}})
    seeds_path = tmp_path/"seeds.json"
    seeds = seed_plan()
    put(seeds_path, seeds)
    ledger = tmp_path/"ledger.json"
    output = tmp_path/"exclusive-run"
    model = json.loads(model_path.read_bytes())
    put(ledger, {"schema": "warehouse-frozen-seed-clearance.v1", "approved_for_frozen_validation": True,
        "purpose": g.PURPOSE, "candidate_id": model["candidate_id"], "seed_plan_sha256": g.digest(seeds),
        "authorized_reset_schedule_sha256": g.digest(g.reset_contract(seeds)),
        "model_method_lock_sha256": g.file_sha(model_path), "source_closure_sha256": runtime.source(),
        "recipe_sha256": model["recipe"]["sha256"], "output_directory": str(output),
        "unresolved": [], "remaining_conflicts": [], "PPO_joint_steps": 0, "publishable": False})
    return model_path, seeds_path, ledger, output


def test_exact_275_and_216_schedule_no_derived_authorization():
    plan = seed_plan()
    assert g.validate_seed_plan(plan) is plan
    assert len(g.reset_contract(plan)["collection"]) == 491
    assert len(g.reset_contract(plan)["physical_integrity_replay"]) == 216


@pytest.mark.parametrize("mutation", ["count", "argmax", "base_leak", "action_leak", "cluster", "streams", "bool"])
def test_seed_contract_rejects_leakage_and_incomplete_shapes(mutation):
    p = seed_plan()
    if mutation == "count": p["splits"]["validation"].pop()
    elif mutation == "argmax": p["splits"]["eligibility"][0]["sampling_mode"] = "argmax"
    elif mutation == "base_leak": p["splits"]["calibration"][0] = dict(p["splits"]["fit"][0])
    elif mutation == "action_leak": p["independent"][0]["action_seed"] = p["splits"]["fit"][0]["action_seed"]
    elif mutation == "cluster": p["independent"][0]["base"] += 100
    elif mutation == "streams": p["independent"][0]["stream"] = 1
    else: p["independent"][0]["stream"] = False
    with pytest.raises(RuntimeError): g.validate_seed_plan(p)


def test_failed_bc_gate_never_collects_independent_or_replay_and_is_single_use(tmp_path):
    runtime = FakeRuntime(internal_pass=False)
    paths = inputs(tmp_path, runtime)
    plan = g.prepare(*paths, runtime)
    result = g.execute(plan, runtime)
    assert len(runtime.calls) == 275 and len(runtime.fit_calls) == 2
    assert not result["passed"] and not result["internal_passed"]
    assert result["independent"] is None and runtime.replay_calls == 0
    assert not (plan.parent/"independent.started.json").exists()
    assert result["PPO_joint_steps"] == result["actor_updates"] == 0
    assert not result["publishable"] and not result["pilot_opened"]
    assert json.loads((plan.parent/"result.json").read_bytes())["development_only"]
    with pytest.raises(FileExistsError): g.execute(plan, runtime)
    with pytest.raises(FileExistsError): g.prepare(*paths, runtime)


def test_full_stub_confirmation_uses_same_216_quality_rollouts_and_clustered_program_ci(tmp_path):
    runtime = FakeRuntime()
    plan = g.prepare(*inputs(tmp_path, runtime), runtime)
    result = g.execute(plan, runtime)
    assert result["passed"] and len(runtime.calls) == 491
    assert runtime.replay_calls == 1 and len(result["physical_replay_resets"]) == 216
    independent = result["independent"]
    assert independent["quality"]["mean_deliveries"] == 6
    assert independent["clustered_scene_quality"]["base_scene_count"] == 24
    assert all(row["base_scene_count"] == 24 for row in independent["clustered_scene_program"].values())
    assert independent["integrity_replay_is_not_additional_quality_evaluation"]
    assert not result["admission_evidence"] and not result["publishable"]


@pytest.mark.parametrize("where", ["collection", "source", "replay"])
def test_failures_and_keyboard_interrupt_keep_consumed_reset_evidence(tmp_path, where):
    runtime = FakeRuntime(interrupt_at=3 if where == "collection" else None,
        source_drift_at=2 if where == "source" else None, replay_interrupt=where == "replay")
    plan = g.prepare(*inputs(tmp_path, runtime), runtime)
    with pytest.raises(BaseException): g.execute(plan, runtime)
    failure = json.loads((plan.parent/"failure.json").read_bytes())
    expected = {"collection": 3, "source": 2, "replay": 491}[where]
    assert len(failure["collection_resets"]) == expected
    assert len(failure["physical_replay_resets"]) == (1 if where == "replay" else 0)
    assert failure["scientific_retry_permitted"] is False and failure["development_only"]
    assert not (plan.parent/"result.json").exists()


def test_deleted_plan_cannot_erase_consumed_scene_failure_receipt(tmp_path):
    runtime = FakeRuntime()
    plan = g.prepare(*inputs(tmp_path, runtime), runtime)
    expected_sha = g.file_sha(plan)
    collect = runtime.collect

    def remove_plan_after_reset(*args, **kwargs):
        result = collect(*args, **kwargs)
        plan.unlink()
        return result

    runtime.collect = remove_plan_after_reset
    with pytest.raises(FileNotFoundError):
        g.execute(plan, runtime)
    failure = json.loads((plan.parent/"failure.json").read_bytes())
    assert failure["plan_sha256"] == expected_sha
    assert len(failure["collection_resets"]) == 1
    assert failure["scientific_retry_permitted"] is False
    assert not (plan.parent/"result.json").exists()


@pytest.mark.parametrize("mutation", ["proposal", "reset_schedule", "source", "method", "actor", "recipe", "permission"])
def test_clearance_and_bound_artifact_mismatch_reject_before_collection(tmp_path, mutation):
    runtime = FakeRuntime()
    model, seeds, ledger, output = inputs(tmp_path, runtime)
    if mutation in ("proposal", "reset_schedule", "permission"):
        value = json.loads(ledger.read_bytes())
        if mutation == "proposal": value["schema"] = "warehouse-seed-audit-draft.v1"
        elif mutation == "reset_schedule": value["authorized_reset_schedule_sha256"] = "wrong"
        else: value["approved_for_frozen_validation"] = False
        put(ledger, value)
    elif mutation == "source": runtime.source_value = "different-source"
    elif mutation == "method":
        value = json.loads(model.read_bytes());value["program_method"] = "legacy";put(model, value)
    elif mutation == "actor": (tmp_path/"not-a-real-model.npz").write_bytes(b"changed")
    else: (tmp_path/"recipe.json").write_text('{}')
    with pytest.raises(RuntimeError): g.prepare(model, seeds, ledger, output, runtime)
    assert not runtime.calls and not output.exists()


def test_program_ci_clusters_base_scenes_instead_of_216_draws():
    rows = {base: metric(9) for base in range(24)}
    ci = g.program_cluster_intervals(rows, resamples=100)
    assert ci["base_scene_count"] == 24
    assert ci["metrics"]["explanation"]["coverage"]["base_scene_bootstrap_95_interval"] == [1., 1.]
    with pytest.raises(RuntimeError, match="24"): g.program_cluster_intervals({i: metric(9) for i in range(216)})


@pytest.mark.parametrize("field,value", [("whole_tree_raw_illegal_count", 1),
    ("whole_tree_path_uncertified_count", 1), ("agreement", .92), ("kl", .251),
    ("coverage", .349), ("expected_actionable_coverage", .299)])
def test_bc_gate_requires_every_fidelity_legality_and_coverage_limit(field, value):
    row = metric(100)
    row["feedback"][field] = value
    assert not g.gate(row)["passed"]


def test_independent_actual_action_coverage_and_quality_failure_remain_failed(tmp_path):
    row = metric(100)
    row["explanation"]["actionable_coverage"] = .299
    assert not g.gate(row, independent=True)["passed"]
    runtime = FakeRuntime()
    original = runtime.quality
    runtime.quality = lambda rows, outcome, actor: {**original(rows, outcome, actor), "deliveries": 5}
    plan = g.prepare(*inputs(tmp_path, runtime), runtime)
    result = g.execute(plan, runtime)
    assert result["internal_passed"] and not result["passed"]
    assert result["independent"]["quality"]["mean_deliveries"] == 5
    assert len(runtime.calls) == 491 and len(result["physical_replay_resets"]) == 216


def test_collection_and_action_rng_arguments_cannot_share_environment_namespace():
    plan = seed_plan()
    plan["splits"]["fit"][0]["action_seed"] = plan["splits"]["fit"][0]["scenario_seed"]
    with pytest.raises(RuntimeError, match="disjoint cleared"):
        g.validate_seed_plan(plan)
