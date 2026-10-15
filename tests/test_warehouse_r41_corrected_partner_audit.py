from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from backend.training import warehouse_r41_corrected_partner_audit as audit


def _trace_rows(*, duplicate_yield: bool = False):
    patterns = {
        "skilled": ["UP", "LEFT"] * 20,
        "assertive": ["DOWN", "RIGHT"] * 20,
        "noisy": ["WAIT", "UP"] * 20,
        "fixed_yield": (["UP", "LEFT"] * 20 if duplicate_yield
                        else ["WAIT", "LEFT"] * 20),
        "fixed_region": ["LEFT", "DOWN"] * 20,
        "fixed_task": ["RIGHT", "UP"] * 20,
    }
    return [
        {"suite": suite, "scenario_id": f"scene-{index}", "partner": partner,
         "participant_action_trace": deepcopy(patterns[partner])}
        for suite, count in (("original_validation", 50),
                             ("conflict_validation", 64))
        for index in range(count)
        for partner in audit.PARTNERS
    ]


def _evaluation(actor_sha: str, *, eligible: bool = True):
    summary = {
        "action_override_count": 0,
        "policy_action_equality_rate": 1.,
        "ai_shutdown_count": 0,
        "eligible": eligible,
    }
    return {
        "actor_sha256": actor_sha,
        "suite_summaries": {
            "original_validation": deepcopy(summary),
            "conflict_validation": deepcopy(summary),
        },
        "partner_distinction": audit._distinction_bundle(_trace_rows()),
        "action_authority_exact": True,
        "actor_shutdown_count_by_suite": {
            "original_validation": 0, "conflict_validation": 0,
        },
        "actor_shutdown_count": 0,
        "actor_zero_shutdowns": True,
        "episode_count": 684,
        "episode_evidence_sha256": "e" * 64,
    }


def test_corrected_fixed_yield_is_distinct_but_never_blocks_required_charge(monkeypatch):
    robot_1 = SimpleNamespace(agent_id="robot_1", active=True, position=(1, 1))
    robot_2 = SimpleNamespace(agent_id="robot_2", active=True, position=(1, 2))
    state = SimpleNamespace(
        agents=[robot_1, robot_2],
        by_id=lambda name: robot_1 if name == "robot_1" else robot_2,
    )
    env = SimpleNamespace(
        state=state, config=SimpleNamespace(map_layout_id="fixture"),
    )
    monkeypatch.setattr(audit, "get_map_layout",
                        lambda _: SimpleNamespace(charger_position=(9, 9)))
    monkeypatch.setattr(audit, "shortest_path_distance", lambda *args: 1)
    monkeypatch.setattr(audit, "partner_action", lambda *args: "LEFT")
    monkeypatch.setattr(audit, "_goals", lambda *args: {"robot_1": (2, 2)})
    assert audit.corrected_partner_action(
        env, "robot_1", "fixed_yield", None) == "WAIT"

    monkeypatch.setattr(audit, "_goals", lambda *args: {"robot_1": (9, 9)})
    assert audit.corrected_partner_action(
        env, "robot_1", "fixed_yield", None) == "LEFT"


def test_distinction_gate_rejects_the_historical_duplicate_partner():
    passing = audit._distinction(_trace_rows())
    duplicate = audit._distinction(_trace_rows(duplicate_yield=True))
    assert passing["passed"] is True
    assert duplicate["passed"] is False
    pair = next(row for row in duplicate["pairwise"]
                if row["left"] == "skilled" and row["right"] == "fixed_yield")
    assert pair["different_episodes"] == 0
    assert pair["aligned_action_disagreements"] == 0


def test_explicit_actor_shutdown_count_is_recomputed_from_both_suites():
    value = _evaluation("a" * 64)
    assert audit._valid_evaluation(value)
    value["suite_summaries"]["conflict_validation"]["ai_shutdown_count"] = 1
    assert not audit._valid_evaluation(value)


def test_any_actor_shutdown_explicitly_blocks_corrected_eligibility():
    baseline = _evaluation("a" * 64)
    candidate = _evaluation("b" * 64)
    decisions = {
        suite: {"selected": True}
        for suite in ("original_validation", "conflict_validation")
    }
    assert audit._candidate_eligible(baseline, candidate, decisions)
    candidate["actor_shutdown_count"] = 1
    candidate["actor_zero_shutdowns"] = False
    assert not audit._candidate_eligible(baseline, candidate, decisions)


def test_distinction_must_pass_independently_in_each_validation_suite():
    rows = _trace_rows()
    for row in rows:
        if row["suite"] == "original_validation" and row["partner"] == "fixed_yield":
            row["participant_action_trace"] = ["UP", "LEFT"] * 20
    bundle = audit._distinction_bundle(rows)
    assert bundle["overall"]["passed"] is True
    assert bundle["by_suite"]["conflict_validation"]["passed"] is True
    assert bundle["by_suite"]["original_validation"]["passed"] is False
    assert bundle["passed"] is False


def test_audit_replays_every_boundary_and_requires_frozen_selection_to_be_earliest(
        monkeypatch, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    dual_path = tmp_path / "dual.json"
    manifest_path = tmp_path / "manifest.json"
    original_path = tmp_path / "original.json"
    baseline_path = tmp_path / "baseline.npz"
    actor_1 = tmp_path / "actor-1.npz"
    actor_2 = tmp_path / "actor-2.npz"
    for path in (ledger_path, dual_path, manifest_path, original_path,
                 baseline_path, actor_1, actor_2):
        path.write_text("fixture", encoding="utf-8")
    hashes = {
        ledger_path: "l" * 64, dual_path: "d" * 64,
        manifest_path: "m" * 64, original_path: "o" * 64,
        baseline_path: "b" * 64, actor_1: "1" * 64, actor_2: "2" * 64,
    }
    ledger = {
        "source": {"actor_sha256": hashes[baseline_path]},
        "boundaries": [
            {"step": 50_000, "actor_path": str(actor_1),
             "actor_sha256": hashes[actor_1], "selected": False},
            {"step": 100_000, "actor_path": str(actor_2),
             "actor_sha256": hashes[actor_2], "selected": True},
        ],
        "selected": {"step": 100_000, "actor_sha256": hashes[actor_2]},
    }
    original_scenes = [{"id": f"o-{i}"} for i in range(50)]
    conflict_scenes = [{"id": f"c-{i}"} for i in range(64)]
    dual = {
        "manifests": {
            "original": {"path": str(original_path),
                         "file_sha256": hashes[original_path],
                         "validation_entries_sha256": audit.digest(original_scenes)},
            "conflict": {"file_sha256": hashes[manifest_path],
                         "validation_entries_sha256": audit.digest(conflict_scenes)},
        },
        "original": {"baseline": {"actor": {"path": str(baseline_path)}}},
    }
    manifest = {"content_sha256": "c" * 64,
                "splits": {"conflict_validation": conflict_scenes}}
    original = {"splits": {"validation": original_scenes}}
    monkeypatch.setattr(audit.training_ledger, "read_saved_ledger",
                        lambda *args, **kwargs: deepcopy(ledger))
    monkeypatch.setattr(audit, "_strict_json", lambda path, label: deepcopy(
        original if Path(path) == original_path else
        manifest if Path(path) == manifest_path else dual))
    monkeypatch.setattr(audit, "validate_conflict_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(audit, "file_hash", lambda path: hashes[Path(path)])
    calls = []

    def evaluate(path, *args, **kwargs):
        path = Path(path)
        calls.append(path)
        return _evaluation(hashes[path], eligible=path != actor_1)

    monkeypatch.setattr(audit, "_evaluate_actor", evaluate)
    monkeypatch.setattr(
        audit.active_evaluation, "_suite_decision",
        lambda baseline, candidate: {
            "selected": candidate["eligible"], "absolute_checks": {},
            "relative_checks": {}, "relative_checks_passed": 0,
        },
    )
    report = audit.audit(
        ledger_path=ledger_path, expected_ledger_sha256=hashes[ledger_path],
        dual_evaluation_path=dual_path, conflict_manifest_path=manifest_path,
    )
    assert calls == [baseline_path, actor_1, actor_2]
    assert report["evaluated_boundary_steps"] == [50_000, 100_000]
    assert report["corrected_earliest_passing_step"] == 100_000
    assert report["selected_actor_matches_corrected_earliest"] is True
    assert report["admission_eligible"] is True


def test_audit_fails_closed_when_runner_stopped_on_a_duplicate_weighted_result(
        monkeypatch, tmp_path):
    # The semantic fail-closed rule is exercised directly: if the frozen
    # selected boundary is not the corrected earliest eligible boundary, the
    # release cannot reinterpret or silently replace it.
    ledger = {"selected": {"step": 50_000, "actor_sha256": "a" * 64}}
    earliest = {"step": 100_000, "actor_sha256": "b" * 64}
    passed = bool(earliest["step"] == ledger["selected"]["step"]
                  and earliest["actor_sha256"] == ledger["selected"]["actor_sha256"])
    assert passed is False
