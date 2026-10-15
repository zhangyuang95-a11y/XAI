from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_conflict_play_selection as selection
from backend.training.warehouse_r41_diagnostic_conflict_play_selection import (
    PROTOCOL,
    _public_scene,
    _strict_jsonl,
    _validate_prefilter_journal,
    run_actor_episode,
    select_balanced_six,
)
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS,
    generate_candidate_batch,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.r41_diagnostic_conflict import (
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    reset_diagnostic_scenario,
)


class WaitActor:
    obs_dim = 197

    def __init__(self):
        self.metadata = {
            "feature_names": list(R41DiagnosticConflictWarehouseEnv().feature_names)
        }

    def act(self, observations, deterministic=True):
        probabilities = np.zeros(len(ACTIONS), dtype=np.float32)
        probabilities[ACTIONS.index("WAIT")] = 1.0
        return (
            {agent: "WAIT" for agent in observations},
            {agent: probabilities.copy() for agent in observations},
        )


def _passing_dynamic(index: int):
    return {
        "passed": True,
        "conflict_opportunity_fraction": 0.321 + index * 0.0005,
        "player_risky_action_mass": 0.15,
        "collision_cancellation_fraction": 0.13,
        "compatible_reference_mean_deliveries": 5.0,
        "reference_absolute_delivery_gap": 2.0,
    }


def test_short_episode_submits_wait_actor_unchanged():
    scene = generate_candidate_batch(0, per_family=1, maximum_draws=1_000)[0][0]
    row = run_actor_episode(scene, WaitActor(), "fixed_yield", 774_100, maximum_steps=8)
    assert row["actor_submission_frames"] == row["steps"]
    assert row["actor_action_override_frames"] == 0
    assert row["new_pickup_on_agent"] == 0
    assert row["new_delivery_on_agent"] == 0
    assert row["new_endpoint_on_agent"] == 0
    assert row["immediate_task_recreation"] == 0


def test_selection_covers_each_family_once_and_balances_xy():
    scenes = generate_candidate_batch(0, per_family=1, maximum_draws=1_000)[0]
    for index, scene in enumerate(scenes):
        scene["dynamic"] = _passing_dynamic(index)
    selected = select_balanced_six(scenes)
    assert selected is not None
    chosen = [*selected["X"], *selected["Y"]]
    assert {row["family_id"] for row in chosen} == set(FAMILY_IDS)
    assert len({row["family_id"] for row in chosen}) == 6
    assert selected["balance"]["workload_relative_difference"] <= 0.05
    assert selected["balance"]["conflict_relative_difference"] <= 0.05


def test_public_scene_preserves_robust_graph_binding_and_resets():
    scene = generate_candidate_batch(0, per_family=1, maximum_draws=1_000)[0][0]
    public = _public_scene(scene)
    assert public["diagnostic_conflict_graph_sha256"] == (
        DIAGNOSTIC_CONFLICT_GRAPH_SHA256
    )
    env = R41DiagnosticConflictWarehouseEnv()
    reset_diagnostic_scenario(env, public)
    assert selection.canonical(env.snapshot()) == selection.canonical(scene["snapshot"])


def test_protocol_freezes_batches_prefilter_and_raw_actor():
    assert PROTOCOL["candidate_generation"]["per_family_per_batch"] == 120
    assert PROTOCOL["candidate_generation"]["maximum_batches"] == 3
    assert PROTOCOL["cheap_prefilter"]["shortlist_per_family"] == 20
    assert PROTOCOL["actor_role"].endswith("zero action overrides")
    assert PROTOCOL["strict_successors"].startswith("neither replacement endpoint")
    assert "no immediate recreation" in PROTOCOL["strict_successors"]
    assert "no ordinary fallback" in PROTOCOL["strict_successors"]


def test_saved_episode_reader_rejects_duplicate_fields(tmp_path):
    journal = tmp_path / "episodes.jsonl"
    journal.write_text('{"scene_id":"one","scene_id":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate field"):
        _strict_jsonl(journal)


def test_prefilter_recompute_rejects_tampered_score():
    row = {
        "scene_id": "scene", "batch_index": 0,
        "family_id": "conflict_family_01", "seed": 10,
        "score": 0.0, "conflict_opportunity_fraction": 0.325,
        "player_risky_action_mass": 0.15,
        "collision_cancellation_fraction": 0.13,
        "compatible_reference_mean_deliveries": 5.0,
        "best_simple_mean_deliveries": 3.0,
        "reference_absolute_delivery_gap": 2.0,
        "actor_action_override_frames": 0, "new_pickup_on_agent": 0,
        "new_delivery_on_agent": 0, "new_endpoint_on_agent": 0,
        "immediate_task_recreation": 0, "successor_sampling_failures": 0,
    }
    row["score"] = -0.5
    _validate_prefilter_journal([row])
    row["score"] += 0.01
    with pytest.raises(ValueError, match="score was not recomputed"):
        _validate_prefilter_journal([row])


def test_reissue_source_rejects_hash_bound_journal_tamper(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(selection, "ROOT", tmp_path)
    files = {
        "protocol.json": b"{}\n",
        "prefilter.jsonl": b'{"scene_id":"one"}\n',
        "episodes.jsonl": b'{"scene_id":"one"}\n',
        "selected_scenes.json": b"{}\n",
    }
    artifacts = {}
    for name, raw in files.items():
        (source / name).write_bytes(raw)
        artifacts[name] = sha256(raw).hexdigest()
    report = {"evidence_artifacts": artifacts}
    report_path = source / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = sha256(report_path.read_bytes()).hexdigest()
    with (source / "episodes.jsonl").open("ab") as stream:
        stream.write(b'{"scene_id":"tampered"}\n')
    with pytest.raises(ValueError, match="artifact bytes differ: episodes.jsonl"):
        selection._reissue_source_evidence(
            source,
            expected_report_sha256=report_sha,
            actor_path=tmp_path / "missing_actor.npz",
            manifest_path=tmp_path / "missing_manifest.json",
        )


def _scene_stub(scene_id, family):
    return {
        "id": scene_id, "split": "play", "seed": 1, "batch_index": 0,
        "fingerprint": "1" * 64,
        "diagnostic_contract_sha256": "2" * 64,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": "3" * 64, "family_id": family,
        "initial_edge_id": "edge", "task_geometry_signature": "geometry",
        "initial_conflict": {}, "initial_public_joint_work_steps": 10,
        "initial_robot_positions": [[1, 1], [2, 2]],
        "successor_stream_seed": 5, "snapshot": {},
    }


def test_reissue_is_atomic_and_publishes_corrected_selected_rows(
    tmp_path, monkeypatch
):
    rows = [
        _scene_stub(f"scene_{index}", family)
        for index, family in enumerate(selection.FAMILY_IDS)
    ]
    tutorial = {**_scene_stub("tutorial", selection.FAMILY_IDS[0]),
                "split": "tutorial"}

    class Actor:
        sha256 = selection.FROZEN_ACTOR_SHA256

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setattr(selection, "ROOT", tmp_path)
    monkeypatch.setattr(selection, "_current_source_sha256", lambda: {
        "backend/training/warehouse_r41_diagnostic_conflict_play_selection.py":
            "4" * 64,
    })
    monkeypatch.setattr(selection, "_reissue_source_evidence", lambda *a, **k: {
        "source": source_dir, "source_relative": "source",
        "report": {
            "version": selection.VERSION,
            "status": "accepted_diagnostic_dynamic_selection",
            "release_eligible": True, "dynamic_passed": 6,
            "batches_considered": 1,
        },
        "protocol": {
            "version": selection.VERSION, "protocol": selection.PROTOCOL,
            "protocol_sha256": selection.digest(selection.PROTOCOL),
        },
        "selected": {
            "X": [{"id": row["id"]} for row in rows[:3]],
            "Y": [{"id": row["id"]} for row in rows[3:]],
            "pairs": [], "balance": {}, "selection_score": 0.0,
        },
        "manifest": {
            "content_sha256": "5" * 64,
            "candidate_batches": [rows], "splits": {"tutorial": [tutorial]},
        },
        "actor": Actor(), "artifact_sha256": {
            "protocol.json": "6" * 64, "prefilter.jsonl": "7" * 64,
            "episodes.jsonl": "8" * 64, "selected_scenes.json": "9" * 64,
        },
        "prefilter_raw": b'{"canonical":true}\n',
        "episodes_raw": b'{"canonical":true}\n',
        "source_producer_sha256": {"selector.py": "a" * 64},
    })
    calls = []

    def replay(output, **kwargs):
        calls.append(Path(output))
        selected = json.loads((Path(output) / "selected_scenes.json").read_text())
        assert all(
            row["diagnostic_conflict_graph_sha256"]
            == DIAGNOSTIC_CONFLICT_GRAPH_SHA256
            for row in (*selected["X"], *selected["Y"])
        )
        return {
            "passed": True, "selected_physical_replay_passed": True,
            "selected_physical_replay_episodes": 720,
        }

    monkeypatch.setattr(selection, "read_saved_diagnostic_selection", replay)
    actor_path = tmp_path / "actor.npz"
    manifest_path = tmp_path / "manifest.json"
    actor_path.write_bytes(b"actor")
    manifest_path.write_text("{}")
    output = tmp_path / "reissued"
    result = selection.reissue_from_verified_journals(
        source_dir, expected_source_report_sha256="b" * 64,
        actor_path=actor_path, manifest_path=manifest_path, output=output,
    )
    assert result["selected_physical_replay_episodes"] == 720
    assert output.is_dir() and len(calls) == 1
    assert (output / "episodes.jsonl").read_bytes() == b'{"canonical":true}\n'


def test_failed_reissue_reader_leaves_no_published_directory(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = [
        _scene_stub(f"scene_{index}", family)
        for index, family in enumerate(selection.FAMILY_IDS)
    ]
    tutorial = {
        **_scene_stub("tutorial", selection.FAMILY_IDS[0]),
        "split": "tutorial",
    }
    monkeypatch.setattr(selection, "ROOT", tmp_path)
    monkeypatch.setattr(selection, "_current_source_sha256", lambda: {"selector": "4" * 64})

    class Actor:
        sha256 = selection.FROZEN_ACTOR_SHA256

    monkeypatch.setattr(selection, "_reissue_source_evidence", lambda *a, **k: {
        "source": source_dir, "source_relative": "source",
        "report": {"version": selection.VERSION, "status": "accepted",
                   "release_eligible": True, "dynamic_passed": 1,
                   "batches_considered": 1},
        "protocol": {"version": selection.VERSION, "protocol": selection.PROTOCOL,
                     "protocol_sha256": selection.digest(selection.PROTOCOL)},
        "selected": {"X": [{"id": row["id"]} for row in rows[:3]],
                     "Y": [{"id": row["id"]} for row in rows[3:]],
                     "pairs": [], "balance": {}, "selection_score": 0.0},
        "manifest": {"content_sha256": "5" * 64,
                     "candidate_batches": [rows],
                     "splits": {"tutorial": [tutorial]}},
        "actor": Actor(), "artifact_sha256": {
            "protocol.json": "6" * 64, "prefilter.jsonl": "7" * 64,
            "episodes.jsonl": "8" * 64, "selected_scenes.json": "9" * 64},
        "prefilter_raw": b"{}\n", "episodes_raw": b"{}\n",
        "source_producer_sha256": {"selector": "a" * 64},
    })
    monkeypatch.setattr(
        selection, "read_saved_diagnostic_selection",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("replay failed")),
    )
    actor_path = tmp_path / "actor.npz"; actor_path.write_bytes(b"actor")
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text("{}")
    output = tmp_path / "reissued"
    with pytest.raises(ValueError):
        selection.reissue_from_verified_journals(
            source_dir, expected_source_report_sha256="b" * 64,
            actor_path=actor_path, manifest_path=manifest_path, output=output,
        )
    assert not output.exists()
