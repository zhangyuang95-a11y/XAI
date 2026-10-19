"""Exact generic seed expansion only; no model, environment, fit or training."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from scripts import audit_warehouse_v621_bc_seeds as a
from experiments.warehouse import exact_seed_audit as exact
from experiments.warehouse import v621_bc_protocol as p
from experiments.warehouse import v621_protocol as d


def test_generic_three_initializer_scope_expands_every_new_actual_argument():
    plan = a.exact_plan()
    counts = Counter(row["kind"] for row in plan["uses"])
    assert counts["env"] == counts["action_rng"] == 8190
    assert counts["critic"] == 6
    assert counts["init"] == 15
    assert counts["classifier_rng"] == 18
    action = [r for r in plan["uses"] if r["kind"] == "action_rng"]
    assert sum(r["consumed_as_random_draw"] for r in action) == 4545
    assert sum(not r["consumed_as_random_draw"] for r in action) == 3645
    classifiers = [r for r in plan["uses"] if r["kind"] == "classifier_rng"]
    for initialization in p.INITIALIZATIONS:
        for recipe in p.RECIPES:
            assert {r["value"] for r in classifiers if r["group"].startswith(f'init-{initialization}/{recipe}/')} == {19, 20}
    cache_loads = [r for r in plan["uses"] if r["purpose"] == "critic_cache_load_torch"]
    assert len(cache_loads) == 9
    assert all(r["argument"] == "torch.manual_seed" and "/shared_cache/" not in r["group"] for r in cache_loads)
    assert plan["authorization_granted"] is False
    assert plan["environment_resets"] == plan["fit_calls"] == plan["PPO_joint_steps"] == 0


def test_spent_275_dev_reuse_is_explicitly_separate_from_fresh_absence_claim():
    plan = a.exact_plan()
    old = plan["spent_development_reuse"]
    assert old["cases"] == d.source_cases()
    assert old["freshness_claimed"] is False
    assert old["excluded_from_fresh_absence_claim"] is True
    fresh = {r["value"] for r in plan["uses"] if r["kind"] in exact.FRESH_KINDS}
    assert not fresh & {c["scenario_seed"] for c in old["cases"]}
    assert not fresh & {c["action_seed"] for c in old["cases"]}


def test_generic_exact_validation_does_not_invoke_6160_expander(tmp_path, monkeypatch):
    path = tmp_path/"exact.json"
    path.write_text(json.dumps(a.exact_plan()))
    def forbidden(*args, **kwargs):
        raise AssertionError("old 6160-specific expander called")
    monkeypatch.setattr(exact, "validate_bc_pilot_plan", forbidden)
    result = a.validate_exact(path)
    assert result["conditional_maximum_all_scopes_resets"] == 10671
    assert result["authorization_granted"] is False
    bad = a.exact_plan()
    bad["uses"][0]["value"] += 1
    path.write_text(json.dumps(bad))
    with pytest.raises((RuntimeError, ValueError)):
        a.validate_exact(path)


def test_prepare_declares_metadata_scope_and_no_weight_reads_or_clearance(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "_registered_traces", lambda *_: [])
    output = tmp_path/"seed-audit"
    a.prepare(tmp_path, output)
    proposal, families, _ = a.metadata.read_proposal(output/"proposal.json")
    assert proposal["authorization_granted"] is False
    assert proposal["exclude_paths"] == [str(output)]
    assert len(families) == 6
    assert all(not any(lo <= value < hi for lo, hi in r["intervals"] for value in (19, 20)) for r in families)
    for use in a.exact_plan()["uses"]:
        if use["kind"] in exact.FRESH_KINDS:
            assert any((r["kind"] == "init") == (use["kind"] == "init") and
                any(lo <= use["value"] < hi for lo, hi in r["intervals"]) for r in families)
    exact.read_exact_plan(output/"exact-uses.json")
    assert not (output/"clearance.json").exists()
    with pytest.raises(FileExistsError):
        a.prepare(tmp_path, output)
