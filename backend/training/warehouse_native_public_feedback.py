"""One-step public action feedback, with a matched zero-feature control.

This is an observation adapter only. All five submitted actions reach the frozen
physics unchanged. Both conditions expose the same confirmed public history;
only the Actor's additional 20 numbers differ. The centralized Critic always
retains the original 354 numbers. No policy, goal or future command is inferred.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from backend.training.warehouse_native_revision_reward import RevisedRewardEnvironment
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.environment import NativeWarehouseEnv

VERSION = "warehouse-native-public-feedback.v1"
BASE_OBSERVATION_SIZE = 177
OBSERVATION_SIZE = 197
GLOBAL_STATE_SIZE = 354
MODES = ("observed", "control")
COLLISION_KINDS = ("none", "same_target", "swap", "occupied_stationary")
HISTORY_FEATURE_NAMES = (
    "history.valid",
    *(f"history.{who}.submitted.{action}" for who in ("self", "other") for action in ACTIONS),
    "history.self.move_canceled", "history.other.move_canceled",
    *(f"history.collision.{kind}" for kind in COLLISION_KINDS),
    "history.self.consecutive_move_canceled", "history.other.consecutive_move_canceled",
    "history.joint.consecutive_collision",
)
_METADATA = frozenset({"public_feedback_version", "public_feedback_mode", "public_feedback_history"})
_HISTORY_KEYS = frozenset({"valid", "frame", "previous_frame", "submitted_actions", "executed_actions",
    "move_canceled", "collision_kind", "post_state_sha256", "unknown_reason",
    "consecutive_move_canceled", "consecutive_collision", "previous_counts"})


def _state_hash(state):
    return hashlib.sha256(json.dumps(asdict(state), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _frame(state):
    if type(state.frame) is not int or state.frame < 0:
        raise ValueError("Public feedback requires a nonnegative integer frame")
    return state.frame


def _unknown(state):
    return {"valid": False, "frame": _frame(state), "previous_frame": None,
        "submitted_actions": None, "executed_actions": None, "move_canceled": None,
        "collision_kind": None, "post_state_sha256": _state_hash(state),
        "unknown_reason": "no_verified_previous_transition",
        "consecutive_move_canceled": {agent.agent_id: 0 for agent in state.agents},
        "consecutive_collision": 0, "previous_counts": None}


def _validate_history(history, state, agent_ids, horizon):
    if not isinstance(history, dict) or set(history) != _HISTORY_KEYS or type(history["valid"]) is not bool:
        raise ValueError("Public feedback history schema is incomplete or damaged")
    current = _frame(state)
    if type(history["frame"]) is not int or history["frame"] != current or history["post_state_sha256"] != _state_hash(state):
        raise ValueError("Public feedback history belongs to a different confirmed state/frame")
    if not history["valid"]:
        if history != _unknown(state):
            raise ValueError("Unknown history cannot contain reconstructed actions or events")
        return
    if (current == 0 or type(history["previous_frame"]) is not int
            or history["previous_frame"] != current - 1 or history["unknown_reason"] is not None):
        raise ValueError("Public feedback history is not the immediately preceding joint step")
    for field in ("submitted_actions", "executed_actions", "move_canceled"):
        if not isinstance(history[field], dict) or set(history[field]) != set(agent_ids):
            raise ValueError("Public feedback history has different robot identities")
    for agent in state.agents:
        key = agent.agent_id
        submitted, executed = history["submitted_actions"][key], history["executed_actions"][key]
        if (submitted not in ACTIONS or executed not in ACTIONS or submitted != agent.last_action
                or executed != agent.last_executed_action or executed not in (submitted, "WAIT")):
            raise ValueError("Public feedback actions differ from the recorded physical transition")
        canceled = history["move_canceled"][key]
        if type(canceled) is not bool or canceled != (submitted != "WAIT" and executed == "WAIT"):
            raise ValueError("Public feedback movement-cancellation flag is inconsistent")
    kind = history["collision_kind"]
    expected_kind = state.last_robot_collision_kind if state.last_robot_collision_event else "none"
    if (kind not in COLLISION_KINDS or kind != expected_kind
            or bool(state.last_robot_collision_event) != (kind != "none")):
        raise ValueError("Public feedback collision differs from the confirmed event")
    def counts(moves, collision, frame):
        if not isinstance(moves, dict) or set(moves) != set(agent_ids):
            raise ValueError("Consecutive cancellation counts have different robot identities")
        for value in (*moves.values(), collision):
            if type(value) is not int or not 0 <= value <= min(frame, horizon):
                raise ValueError("Consecutive counts must be integers bounded by confirmed frames/horizon")
    previous = history["previous_counts"]
    if (not isinstance(previous, dict) or set(previous) != {"valid", "frame", "move_canceled", "collision"}
            or type(previous["valid"]) is not bool or type(previous["frame"]) is not int
            or previous["frame"] != current - 1 or (previous["valid"] and previous["frame"] == 0)):
        raise ValueError("Consecutive counts lack a matching previous confirmed boundary")
    counts(previous["move_canceled"], previous["collision"], current - 1)
    counts(history["consecutive_move_canceled"], history["consecutive_collision"], current)
    if not previous["valid"] and (any(previous["move_canceled"].values()) or previous["collision"]):
        raise ValueError("Unknown prior history cannot contribute consecutive counts")
    for key in agent_ids:
        expected = previous["move_canceled"][key] + 1 if history["move_canceled"][key] else 0
        if history["consecutive_move_canceled"][key] != expected:
            raise ValueError("Consecutive cancellation count differs from confirmed transition")
    expected = previous["collision"] + 1 if kind != "none" else 0
    if history["consecutive_collision"] != expected:
        raise ValueError("Consecutive collision count differs from confirmed transition")


class PublicFeedbackEnvironment(RevisedRewardEnvironment):
    """197-feature Actor input, unchanged 354-feature centralized Critic input.

    A raw physical/reward snapshot is a scenario start, not evidence of its last
    transition; it always initializes unknown feedback, even at nonzero frames.
    Training adapters should use ``restore(..., require_feedback=True)`` for
    actual adapter checkpoints, so stripping all metadata cannot silently turn
    a saved trajectory into an unknown-history start.
    """

    def __init__(self, config=None, reward_config=None, collision_cost=0.05, *, mode="observed"):
        if type(mode) is not str or mode not in MODES:
            raise ValueError("Public feedback mode must be observed or control")
        self._mode, self._history = mode, None
        super().__init__(config, reward_config, collision_cost=collision_cost)
        if (NativeWarehouseEnv.observation_size.fget(self) != BASE_OBSERVATION_SIZE
                or tuple(ACTIONS) != ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
                or type(self.config.horizon) is not int):
            raise ValueError("Public feedback adapter requires the frozen 177-feature five-action contract")

    @property
    def mode(self): return self._mode

    @property
    def observation_size(self): return OBSERVATION_SIZE

    @property
    def feature_names(self):
        return tuple(NativeWarehouseEnv.feature_names.fget(self)) + HISTORY_FEATURE_NAMES

    def _current_history(self):
        self._require_state()
        # During the base step, observations are constructed before this adapter
        # receives its completed info. That intermediate return has no new history.
        # Reads never mutate history or infer it from existing last_action fields.
        if (self._history is None or not self._history["valid"]
                or self._history["frame"] != self.state.frame
                or self._history["post_state_sha256"] != _state_hash(self.state)):
            return _unknown(self.state)
        return deepcopy(self._history)

    def public_history(self):
        history = self._current_history()
        return {key: deepcopy(value) for key, value in history.items()
                if key not in ("post_state_sha256", "unknown_reason", "previous_counts")}

    def observations(self):
        base = NativeWarehouseEnv.observations(self)
        history = self._current_history()
        result = {}
        for role, key in enumerate(self.agent_ids):
            extra = np.zeros(len(HISTORY_FEATURE_NAMES), dtype=np.float32)
            if self.mode == "observed" and history["valid"]:
                other = self.agent_ids[1 - role]
                values = [1.]
                for person in (key, other):
                    values.extend(float(history["submitted_actions"][person] == action) for action in ACTIONS)
                values.extend(float(history["move_canceled"][person]) for person in (key, other))
                values.extend(float(history["collision_kind"] == kind) for kind in COLLISION_KINDS)
                scale = math.log1p(self.config.horizon)
                values.extend(math.log1p(history["consecutive_move_canceled"][person]) / scale for person in (key, other))
                values.append(math.log1p(history["consecutive_collision"]) / scale)
                extra[:] = values
            result[key] = np.concatenate((base[key], extra)).astype(np.float32)
        return result

    def global_state(self):
        base = NativeWarehouseEnv.observations(self)
        return np.concatenate([base[key] for key in self.agent_ids]).astype(np.float32)

    def reset(self, *, seed=None):
        self._history = None
        _, info = super().reset(seed=seed)
        self._history = _unknown(self.state)
        info["public_feedback"] = self.public_history()
        return self.observations(), info

    def step(self, actions, *, decision_metadata=None):
        previous = self._current_history()
        _, rewards, terminated, truncated, info = super().step(actions, decision_metadata=decision_metadata)
        requested, executed = deepcopy(info["requested_actions"]), deepcopy(info["executed_actions"])
        canceled = {key: requested[key] != "WAIT" and executed[key] == "WAIT" for key in self.agent_ids}
        history = {"valid": True, "frame": info["frame"], "previous_frame": info["frame"] - 1,
            "submitted_actions": requested, "executed_actions": executed,
            "move_canceled": canceled,
            "collision_kind": info["collision_kind"] or "none", "post_state_sha256": _state_hash(self.state),
            "unknown_reason": None,
            "previous_counts": {"valid": previous["valid"], "frame": previous["frame"],
                "move_canceled": deepcopy(previous["consecutive_move_canceled"]), "collision": previous["consecutive_collision"]},
            "consecutive_move_canceled": {key: previous["consecutive_move_canceled"][key] + 1 if canceled[key] else 0 for key in self.agent_ids},
            "consecutive_collision": previous["consecutive_collision"] + 1 if info["robot_collision"] else 0}
        _validate_history(history, self.state, self.agent_ids, self.config.horizon)
        self._history = history
        info["public_feedback"] = self.public_history()
        return self.observations(), rewards, terminated, truncated, info

    def public_view(self):
        view = super().public_view()
        # Identical in both arms: no mode, model treatment or strategy annotation.
        view["public_feedback"] = self.public_history()
        return view

    def set_state(self, state):
        super().set_state(state)
        self._history = None

    def snapshot(self):
        payload = super().snapshot()
        payload.update(public_feedback_version=VERSION, public_feedback_mode=self.mode,
                       public_feedback_history=self._current_history())
        return payload

    def restore(self, payload, *, require_feedback=False):
        if (type(require_feedback) is not bool or not isinstance(payload, dict)
                or any(type(key) is not str for key in payload)):
            raise ValueError("Snapshot and require_feedback flag must have their declared types")
        metadata = {key for key in payload if key.startswith("public_feedback")}
        if require_feedback and not metadata:
            raise ValueError("Adapter checkpoint must contain complete public feedback history")
        if metadata and (metadata != _METADATA or payload.get("public_feedback_version") != VERSION
                         or payload.get("public_feedback_mode") != self.mode):
            raise ValueError("Public feedback snapshot version/mode/metadata mismatch")
        # Frozen Native.restore changes state before validating RNG. Stage the
        # complete restore in a separate environment so any rejection is atomic.
        candidate = RevisedRewardEnvironment(self.config, deepcopy(self.reward_config), collision_cost=self.collision_cost)
        candidate.restore(deepcopy({key: value for key, value in payload.items() if key not in _METADATA}))
        history = deepcopy(payload["public_feedback_history"]) if metadata else _unknown(candidate.state)
        _validate_history(history, candidate.state, candidate.agent_ids, self.config.horizon)
        self.state = candidate.state
        self.set_rng_state(candidate.get_rng_state())
        self._episode_counter = candidate._episode_counter
        self._history = history

    def branch(self):
        result = type(self)(self.config, deepcopy(self.reward_config), collision_cost=self.collision_cost, mode=self.mode)
        result.restore(self.snapshot(), require_feedback=True)
        return result
