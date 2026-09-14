from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace

import numpy as np

from backend.training import warehouse_r41_diagnostic_final_observation_projection_v13 as subject
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api


class _Environment:
    def __init__(self, snapshot: dict):
        self.state = SimpleNamespace(frame=int(snapshot["frame"]))
        self.value = int(snapshot["value"])
        self.done = bool(snapshot["done"])

    def snapshot(self) -> dict:
        return {
            "frame": self.state.frame, "value": self.value, "done": self.done,
            "state": {
                "frame": self.state.frame,
                "agents": {"robot_1": {"value": self.value},
                           "robot_2": {"value": self.value}},
                "tasks": [], "completed_tasks": [], "total_deliveries": 0,
                "terminated": self.done, "truncated": False,
                "terminal_reason": "done" if self.done else None,
            },
        }

    def observations(self) -> dict[str, np.ndarray]:
        return {"robot_2": np.asarray(
            [self.state.frame, self.value, int(self.done)], dtype=np.float32)}


class _Runtime:
    def __init__(self):
        self.decision_reads = 0

    def environment(self, scene: dict) -> _Environment:
        return _Environment({"frame": 0, "value": scene["initial"], "done": False})

    def from_snapshot(self, snapshot: dict) -> _Environment:
        return _Environment(deepcopy(snapshot))

    def decision(self, env: _Environment):
        self.decision_reads += 1
        return ({"robot_2": "WAIT"}, {"probabilities": {
            "robot_2": np.asarray([0, 0, 0, 0, 1], dtype=np.float32)}})

    def step(self, env: _Environment, player_action: str) -> dict:
        env.value += rows_api.ACTIONS.index(player_action) + 1
        env.state.frame += 1
        env.done = env.state.frame >= 4
        return {
            "submitted_actions": {"robot_2": "WAIT"},
            "policy_actions": {"robot_2": "WAIT"},
            "done": env.done,
            "after": env.snapshot(),
        }


def test_projection_exactly_matches_collector_observation_rows(monkeypatch) -> None:
    def groups(env, robot):
        return ("shared_charger",) if env.state.frame in {0, 2} else ()

    def partner(env, robot, name, rng):
        return rows_api.ACTIONS[int(rng.integers(0, len(rows_api.ACTIONS)))]

    monkeypatch.setattr(rows_api, "critical_groups", groups)
    monkeypatch.setattr(rows_api, "partner_action", partner)
    scenes = [{"id": "scene", "fingerprint": "a" * 64, "initial": 7}]
    collector_runtime = _Runtime()
    rows, collector_steps = rows_api._collect(
        collector_runtime, scenes, scene_offset=900_000,
        dense_critical=False, progress_label=None)
    expected = [rows_api.legacy._obs_hash(row["observation"]) for row in rows]

    projection_runtime = _Runtime()
    projected = subject.project_observation_hashes(
        projection_runtime, scenes, scene_offset=900_000,
        dense_critical=False)
    assert projected["ordered_observation_hashes"] == expected
    assert projected["environment_steps"] == collector_steps
    assert projection_runtime.decision_reads == 0
    assert collector_runtime.decision_reads > 0
    subject.validate_projection(projected)


def test_projection_contract_returns_no_targets_or_raw_observations() -> None:
    source = inspect.getsource(subject.project_observation_hashes)
    assert ".decision(" not in source
    assert "submitted_actions" not in source
    assert "policy_actions" not in source
    assert '["probabilities"]' not in source
    assert "program.predict" not in source
    contract = subject.contract()
    assert contract["runtime_decision_output_read"] is False
    assert contract["raw_observations_returned"] is False
    assert contract["actor_actions_returned"] is False
    assert contract["actor_probabilities_returned"] is False
    assert contract["labels_returned"] is False


def test_overlap_filter_regression_counts_all_public_projection_hashes() -> None:
    # The burned-v12 diagnosis found 3 retained-development collisions and 23
    # outer collisions.  A complete projection rejects on their union instead
    # of separately accepting a scene after an incomplete workload replay.
    development = {f"d{index:063x}" for index in range(3)}
    outer = {f"e{index:063x}" for index in range(23)}
    candidate = development | outer | {"f" * 64}
    assert len(candidate & development) == 3
    assert len(candidate & outer) == 23
    assert len(candidate & (development | outer)) == 26
    assert bool(candidate & (development | outer)) is True
