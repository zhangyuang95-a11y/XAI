"""Native benchmark contracts with solver stubs; never fits an actual model."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import verify_warehouse_v625_native as n


def test_analytical_prefix_is_maximum_guaranteed_under_per_condition_budget():
    recipe = n.expected_recipe()
    assert recipe["candidate_prefix"] == 15
    assert recipe["pause_after_candidates"] == 7
    assert (15 + 1) * 32 * 8 == recipe["maximum_api_calls_per_condition"]
    assert (16 + 1) * 32 * 8 > recipe["maximum_api_calls_per_condition"]
    assert recipe["outer_reserved_api_calls"] == 16384
    assert recipe["condition_order"] == ["baseline-P0", "checkpoint-P0", "checkpoint-P1", "baseline-P1"]


def test_no_direct_native_fit_without_actual_plan_or_outer_reservation(tmp_path):
    with pytest.raises(RuntimeError, match="outer reservation"):
        n.run_condition("baseline-P0", {}, {}, tmp_path / "conditions" / "baseline-P0", {})
    identity = {"plan_sha256": "a" * 64, "source_sha256": "b" * 64}
    pointer = n.write_new(tmp_path / "attempt-start.json", {"schema": "warehouse-v625-native-attempt.v1",
        "recipe": n.expected_recipe(), **identity,
        "reservations": [{"condition_id": name, "charged_call_slots": 4096} for name in n.ORDER],
        "total_charged_call_slots": 16384, "unused_slots_refundable": False})
    with pytest.raises(FileNotFoundError):
        n.run_condition("baseline-P0", {}, {}, tmp_path / "conditions" / "baseline-P0", identity, reservation=pointer)
    assert not (tmp_path / "conditions").exists()


def test_baseline_every_call_is_durable_before_solver_and_no_state_objects(tmp_path):
    ledger = n.BaselineLedger(tmp_path / "calls.jsonl", {"test": True}, started_at=n.time.monotonic(), max_calls=1)
    token = ledger.consume({"phase": "initial"})
    rows = [json.loads(line) for line in ledger.path.read_bytes().splitlines()]
    assert rows[-1]["event"] == "call_started" and rows[-1]["data"]["call"] == token
    ledger.finish(token, detail={"solver": "synthetic_no_optimizer"})
    with pytest.raises(n.BudgetExhausted):
        ledger.consume()
    assert ledger.store.put_json({"x": list(range(100))})["kind"] == "json"
    assert list(tmp_path.iterdir()) == [ledger.path]
    assert ledger.events()[-1]["event"] == "call_completed"


def test_direct_checkpoint_union_includes_cpu_without_double_counting_io(monkeypatch):
    now = [0.]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    session = SimpleNamespace(store=SimpleNamespace(io_seconds=0.), ledger_io_seconds=0., manifest_io_seconds=0.)
    timer = n.DirectCheckpointOverhead([session])
    with timer.measure("codec_cpu_and_io"):
        now[0] = 5.
        session.store.io_seconds = 3.
    session.ledger_io_seconds = 2.
    result = timer.report()
    assert result["wrapped_CP_only_seconds"] == 5
    assert result["total_actual_CP_IO_seconds"] == 5
    assert result["IO_seconds_outside_wrappers"] == 2
    assert result["direct_full_checkpoint_overhead_seconds"] == 7


def comparison_fixture():
    results = []
    for name in n.ORDER:
        checkpoint = name.startswith("checkpoint")
        results.append({"condition_id": name, "candidate_prefix_completed": 15,
            "full_queue_count": 800, "full_queue_sha256": "a" * 64, "prefix_sha256": "b" * 64,
            "actual_leaf_api_calls": 40, "actual_solver_calls": 40, "settled_api_calls": 40,
            "cache_hits": 20, "cache_evictions": 0, "leaf_cache_entries": 40, "state_cache_entries": 16,
            "candidate_trace_sha256": "c" * 64, "numerical_trace_sha256": "d" * 64,
            "wall_seconds": 100 if checkpoint else 95, "actual_measured_IO_seconds": 3,
            "direct_checkpoint_overhead": {"direct_full_checkpoint_overhead_seconds": 8}})
    return results


def test_paired_gate_uses_direct_cpu_cost_and_both_arms_not_pooled():
    results = comparison_fixture()
    assert n.paired_comparison(results)["passed"]
    results[1]["direct_checkpoint_overhead"]["direct_full_checkpoint_overhead_seconds"] = 11
    assert not n.paired_comparison(results)["passed"]
    assert n.paired_comparison(results)["pairs"]["P0"]["checkpoint_fraction"] == .11


def test_paired_gate_rejects_nonidentical_models_calls_or_order():
    results = comparison_fixture()
    results[1]["numerical_trace_sha256"] = "x" * 64
    assert not n.paired_comparison(results)["passed"]
    with pytest.raises(RuntimeError, match="balanced-order"):
        n.paired_comparison(results[::-1])


def synthetic_inputs():
    from backend.training.warehouse_v6_actor import FEATURE_NAMES
    names = tuple(FEATURE_NAMES)
    x = np.zeros((60, len(names)), np.float32)
    x[:, names.index("self_row")] = np.linspace(0, 1, len(x), dtype=np.float32)
    masks = np.ones((len(x), 5), np.float32)
    for i, name in enumerate(names):
        if name.startswith("legal_"):
            x[:, i] = 1
    bias = np.asarray([-3, -3, -3, -3, 3], np.float32)
    from backend.training.warehouse_v622_program import masked_probabilities
    targets = masked_probabilities(np.broadcast_to(bias, masks.shape), masks)
    return {"encoded": x, "masks": masks, "targets": targets,
        "episodes": np.asarray([f"fit-only-{i // 6}" for i in range(len(x))]),
        "names": names, "indices": tuple(names.index("legal_" + a.lower()) for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")),
        "structure": {"root_ref": -1, "split_weights": np.empty((0, len(names)), np.float32),
            "split_bias": [], "left_refs": [], "right_refs": [], "leaf_count": 1, "feature_dim": len(names)}}


def install_synthetic_solver_and_menu(monkeypatch, inputs):
    from backend.training import warehouse_v625_program as program
    from backend.training.warehouse_program_search import _BeamEntry
    from backend.training.warehouse_v619_tree import _Decision
    actual_solver_calls = []
    def stub(x, masks, targets, names, legal, mode, budget, **kwargs):
        token = budget.consume()
        # This is a declared toy fixture, not scipy or a Warehouse fit.
        actual_solver_calls.append((len(x), mode))
        leaf = {"version": program.LEAF_VERSION, "arithmetic": program.ARITHMETIC,
            "feature_names": list(names), "support": [], "support_names": [],
            "bias": [-3., -3., -3., -3., 3.], "weights": [[] for _ in range(5)],
            "witness": 4, "mode": mode, "report": {"elapsed_seconds": len(actual_solver_calls)}}
        leaf["certificate"] = program.certify_leaf(leaf, names, legal)
        budget.finish(token, success=True, detail={"solver": "synthetic_stub_no_optimizer", "elapsed_seconds": 0})
        return leaf
    def menu(ctx, start):
        parent = _BeamEntry(start)
        result = []
        for i in range(1, 21):
            weight = np.zeros(len(ctx.names), np.float32)
            weight[ctx.names.index("self_row")] = 1
            tree = _Decision(tuple(map(float, weight)), float(np.float32(-i / 21)))
            result.append((tree, "node", f"toy-{i}", parent))
        return result
    monkeypatch.setattr(program, "fit_leaf", stub)
    monkeypatch.setattr(n, "_queue", menu)
    monkeypatch.setattr(n, "checked_reservation", lambda *args, **kwargs: ({}, inputs))
    proposal = n._proposal_binding(inputs)
    return actual_solver_calls, proposal


@pytest.mark.parametrize("mode", ["P0", "P1"])
def test_native_prefix_actual_helpers_resume_same_cache_models_and_scores(monkeypatch, tmp_path, mode):
    inputs = synthetic_inputs()
    calls, proposal = install_synthetic_solver_and_menu(monkeypatch, inputs)
    baseline = n.run_condition("baseline-" + mode, inputs, proposal, tmp_path / ("base-" + mode), {})
    checkpoint = n.run_condition("checkpoint-" + mode, inputs, proposal, tmp_path / ("cp-" + mode), {})
    assert baseline["candidate_trace_sha256"] == checkpoint["candidate_trace_sha256"]
    assert baseline["numerical_trace_sha256"] == checkpoint["numerical_trace_sha256"]
    assert baseline["actual_leaf_api_calls"] == checkpoint["actual_leaf_api_calls"]
    assert baseline["cache_hits"] == checkpoint["cache_hits"]
    assert baseline["actual_solver_calls"] == checkpoint["actual_solver_calls"] == 0
    assert len(calls) == baseline["actual_leaf_api_calls"] + checkpoint["actual_leaf_api_calls"]
    assert checkpoint["full_queue_count"] == 20  # Preserve the entire queue, not only the 15 evaluated trees.
    assert checkpoint["candidate_prefix_completed"] == 15 and checkpoint["pause_cursor"] == 7
    assert len(checkpoint["numerical_trace"]) == 16
    labels = [item["label"] for item in checkpoint["direct_checkpoint_overhead"]["intervals"]]
    assert "full_queue_codec" in labels and "safe_resume_full_state_cache_queue_decode" in labels
    assert not list((tmp_path / ("base-" + mode)).glob("*.objects"))
    assert checkpoint["post_timing_equivalence_seconds"] >= 0


def test_prefit_queue_binding_never_calls_a_leaf_optimizer(monkeypatch):
    inputs = synthetic_inputs()
    calls, proposal = install_synthetic_solver_and_menu(monkeypatch, inputs)
    assert calls == []
    assert proposal["prefix_count"] == 15 and proposal["full_queue_count"] == 20
    assert proposal["proposal_uses_fitted_leaf_outputs"] is False


def test_budgets_stop_fixed_prefix_without_adding_work(monkeypatch, tmp_path):
    inputs = synthetic_inputs()
    calls, proposal = install_synthetic_solver_and_menu(monkeypatch, inputs)
    original = n.BaselineLedger.consume
    def stop(self, metadata=None):
        if self.calls >= 3:
            raise n.BudgetExhausted("synthetic cap")
        return original(self, metadata)
    monkeypatch.setattr(n.BaselineLedger, "consume", stop)
    with pytest.raises(n.BudgetExhausted):
        n.run_condition("baseline-P0", inputs, proposal, tmp_path / "failed", {})
    assert len(calls) == 3
    assert json.loads((tmp_path / "failed" / "failed.json").read_bytes())["actual_leaf_api_calls"] == 3
    assert not (tmp_path / "failed" / "phase-completed.json").exists()


def test_external_supervisor_kills_and_waits_its_owned_group(monkeypatch, tmp_path):
    name = "baseline-P0"
    directory = tmp_path / "conditions" / name
    directory.mkdir(parents=True)
    n.write_new(directory / "phase-started.json", {"condition_id": name, "pid": 62504,
        "started_monotonic": 0, "hard_deadline_monotonic": 1, "wall_seconds": 1200})
    class Process:
        pid, code, waits = 62504, None, 0
        def poll(self):
            return self.code
        def wait(self, timeout=None):
            self.waits += 1
            return self.code
    process = Process()
    captured = {}
    def launch(*args, **kwargs):
        captured.update(kwargs)
        return process
    monkeypatch.setattr(n.subprocess, "Popen", launch)
    monkeypatch.setattr(n.os, "killpg", lambda pid, sig: setattr(process, "code", -15))
    with pytest.raises(n.DeadlineExceeded, match="external native"):
        n.supervise_condition(tmp_path / "plan.json", name, "a" * 64)
    assert captured["start_new_session"] is True
    assert process.waits > 0
    stopped = json.loads((directory / "external-controller-stopped.json").read_bytes())
    assert stopped["all_owned_processes_waited"] and not stopped["retry_authorized"]


def test_old_generic_passed_receipt_cannot_grant_native_authority(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"passed":true,"schema":"warehouse-v625-engineering-preflight.v1"}')
    with pytest.raises(RuntimeError, match="native-04"):
        n.checked_receipt(path)
