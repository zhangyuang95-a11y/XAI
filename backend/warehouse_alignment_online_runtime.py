"""Portable observed197 warehouse runtime for the frozen alignment Actor.

This module deliberately has no dependency on PyTorch, scikit-learn, or any
training package.  The neural command submitted for robot_2 is the exact
argmax of the exported NumPy Actor; physics may cancel it but never replaces it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import sys
import types
from typing import Any, Mapping

import numpy as np

# ``core.__init__`` exposes offline RCPD fitting and consequently imports
# scikit-learn.  Runtime physics only needs leaf modules such as
# ``core.policy_contracts``.  Register a normal namespace package so Python can
# load those leaves without executing the offline package initializer.
if "core" not in sys.modules:
    _core = types.ModuleType("core")
    _core.__path__ = [str(Path(__file__).resolve().parents[1] / "core")]
    _core.__package__ = "core"
    sys.modules["core"] = _core

from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.observations import observation_names
from env.warehouse_native.scenarios import reset_scenario

RUNTIME_VERSION = "warehouse-alignment-online-numpy-runtime.v1"
FAMILY = "alignment_feedback197"
PUBLIC_FEEDBACK_VERSION = "warehouse-native-public-feedback.v1"
REWARD_VERSION = "warehouse-native-score-pbrs.v2"
REWARD_REVISION = "warehouse-native-score-pbrs.collision-r1"
BASE_OBSERVATION_SIZE = 177
OBSERVATION_SIZE = 197
GLOBAL_STATE_SIZE = 354
COLLISION_KINDS = ("none", "same_target", "swap", "occupied_stationary")
HISTORY_FEATURE_NAMES = (
    "history.valid",
    *(f"history.{who}.submitted.{action}" for who in ("self", "other") for action in ACTIONS),
    "history.self.move_canceled", "history.other.move_canceled",
    *(f"history.collision.{kind}" for kind in COLLISION_KINDS),
    "history.self.consecutive_move_canceled", "history.other.consecutive_move_canceled",
    "history.joint.consecutive_collision",
)
_FEEDBACK_METADATA = frozenset({"public_feedback_version", "public_feedback_mode", "public_feedback_history"})
_REWARD_METADATA = frozenset({"training_reward_version", "training_reward_config", "training_reward_revision"})
_HISTORY_KEYS = frozenset({"valid", "frame", "previous_frame", "submitted_actions", "executed_actions",
    "move_canceled", "collision_kind", "post_state_sha256", "unknown_reason",
    "consecutive_move_canceled", "consecutive_collision", "previous_counts"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError("Invalid external " + name)
    return value


def runtime_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "env/warehouse_native/environment.py",
             root / "env/warehouse_native/observations.py", root / "env/warehouse_native/scenarios.py",
             root / "env/warehouse/domain.py", root / "env/warehouse/navigation.py"]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def _state_hash(state) -> str:
    return digest(asdict(state))


def _unknown(state) -> dict:
    if type(state.frame) is not int or state.frame < 0:
        raise ValueError("Public feedback requires a nonnegative integer frame")
    return {"valid": False, "frame": state.frame, "previous_frame": None,
        "submitted_actions": None, "executed_actions": None, "move_canceled": None,
        "collision_kind": None, "post_state_sha256": _state_hash(state),
        "unknown_reason": "no_verified_previous_transition",
        "consecutive_move_canceled": {agent.agent_id: 0 for agent in state.agents},
        "consecutive_collision": 0, "previous_counts": None}


def _validate_history(history, state, agent_ids, horizon) -> None:
    if not isinstance(history, dict) or set(history) != _HISTORY_KEYS or type(history["valid"]) is not bool:
        raise ValueError("Public feedback history schema is incomplete or damaged")
    current = state.frame
    if type(history["frame"]) is not int or history["frame"] != current or history["post_state_sha256"] != _state_hash(state):
        raise ValueError("Public feedback history belongs to a different confirmed state/frame")
    if not history["valid"]:
        if history != _unknown(state):
            raise ValueError("Unknown history cannot contain reconstructed actions or events")
        return
    if current == 0 or history["previous_frame"] != current - 1 or history["unknown_reason"] is not None:
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
    if kind not in COLLISION_KINDS or kind != expected_kind or bool(state.last_robot_collision_event) != (kind != "none"):
        raise ValueError("Public feedback collision differs from the confirmed event")
    previous = history["previous_counts"]
    if not isinstance(previous, dict) or set(previous) != {"valid", "frame", "move_canceled", "collision"}:
        raise ValueError("Consecutive counts lack a matching previous confirmed boundary")
    for key in agent_ids:
        expected = previous["move_canceled"][key] + 1 if history["move_canceled"][key] else 0
        if history["consecutive_move_canceled"][key] != expected:
            raise ValueError("Consecutive cancellation count differs from confirmed transition")
    expected = previous["collision"] + 1 if kind != "none" else 0
    if history["consecutive_collision"] != expected:
        raise ValueError("Consecutive collision count differs from confirmed transition")


class OnlineNumPyActor:
    """Exact portable loader/inference for ``warehouse_native_actor_v1``."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.artifact_sha256 = self.sha256 = file_hash(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata_json"].item()))
            self.weights = {name: archive[name].astype(np.float32, copy=True)
                            for name in archive.files if name != "metadata_json"}
        expected = {"format": "warehouse_native_actor_v1", "policy_version": "warehouse_native_plain_mlp_v1",
            "actions": list(ACTIONS), "architecture": "two_hidden_layer_tanh",
            "action_masks": False, "runtime_action_override": False}
        if any(self.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible native Actor metadata")
        self.obs_dim, self.state_dim, self.hidden = (int(self.metadata[key]) for key in ("obs_dim", "state_dim", "hidden"))
        shapes = {"0.weight": (self.hidden, self.obs_dim), "0.bias": (self.hidden,),
            "2.weight": (self.hidden, self.hidden), "2.bias": (self.hidden,),
            "4.weight": (len(ACTIONS), self.hidden), "4.bias": (len(ACTIONS),)}
        if set(self.weights) != set(shapes):
            raise ValueError("Native Actor export must contain only its six MLP tensors")
        for name, shape in shapes.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError("Invalid native Actor tensor: " + name)
            self.weights[name].setflags(write=False)

    def logits(self, observations: Any) -> np.ndarray:
        value = np.asarray(observations, dtype=np.float32)
        if value.ndim < 1 or value.shape[-1] != self.obs_dim or not np.isfinite(value).all():
            raise ValueError("Invalid native Actor observations")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for index in (0, 2, 4):
                value = value @ self.weights[f"{index}.weight"].T + self.weights[f"{index}.bias"]
                if not np.isfinite(value).all():
                    raise FloatingPointError("Native neural inference produced non-finite logits")
                if index != 4:
                    value = np.tanh(value)
        return np.asarray(value, dtype=np.float32)

    def act(self, observations: Mapping[str, Any], deterministic: bool = True,
            rng: np.random.Generator | None = None):
        if len(observations) != 2 or (not deterministic and rng is None):
            raise ValueError("Invalid Actor call")
        agents = sorted(observations)
        logits = self.logits(np.stack([observations[a] for a in agents]))
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        actions, distributions = {}, {}
        for row, agent in enumerate(agents):
            index = int(np.argmax(probabilities[row])) if deterministic else int(rng.choice(len(ACTIONS), p=probabilities[row]))
            actions[agent] = ACTIONS[index]
            distributions[agent] = probabilities[row].copy()
        return actions, distributions


class OnlinePublicFeedbackEnvironment(NativeWarehouseEnv):
    def __init__(self, config=None, reward_config=None, collision_cost=.05, *, mode="observed"):
        self._mode, self._history = mode, None
        super().__init__(config)
        self.reward_config = deepcopy(reward_config)
        self.collision_cost = float(collision_cost)
        self.reward_config.update(revision=REWARD_REVISION, collision_training_cost=self.collision_cost)
        if mode != "observed" or NativeWarehouseEnv.observation_size.fget(self) != BASE_OBSERVATION_SIZE:
            raise ValueError("Online runtime requires the observed197 contract")

    @property
    def mode(self): return self._mode
    @property
    def observation_size(self): return OBSERVATION_SIZE
    @property
    def feature_names(self): return tuple(NativeWarehouseEnv.feature_names.fget(self)) + HISTORY_FEATURE_NAMES

    def _current_history(self):
        self._require_state()
        if self._history is None or not self._history["valid"] or self._history["frame"] != self.state.frame or self._history["post_state_sha256"] != _state_hash(self.state):
            return _unknown(self.state)
        return deepcopy(self._history)

    def public_history(self):
        return {k: deepcopy(v) for k, v in self._current_history().items()
                if k not in ("post_state_sha256", "unknown_reason", "previous_counts")}

    def observations(self):
        base, history, result = NativeWarehouseEnv.observations(self), self._current_history(), {}
        for role, key in enumerate(self.agent_ids):
            extra = np.zeros(len(HISTORY_FEATURE_NAMES), dtype=np.float32)
            if history["valid"]:
                other = self.agent_ids[1-role]
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
        _, info = NativeWarehouseEnv.reset(self, seed=seed)
        self._history = _unknown(self.state)
        info["public_feedback"] = self.public_history()
        return self.observations(), info

    def step(self, actions, *, decision_metadata=None):
        previous, before_score = self._current_history(), self.native_score
        _, _, terminated, truncated, info = NativeWarehouseEnv.step(self, actions, decision_metadata=decision_metadata)
        old_components = info["reward_components"]
        cfg = self.reward_config
        components = {"native_score_increment": cfg["native_score_scale"] * (self.native_score-before_score),
            "static_wall_commands": -cfg["static_wall_command_cost"] * len(info["invalid_moves"]),
            "potential": cfg["potential_scale"] * (cfg["gamma"]*info["potential_after"]-info["potential_before"])}
        adjustment = -self.collision_cost - self.config.robot_collision_points*cfg["native_score_scale"]
        components["collision_risk_adjustment"] = adjustment * int(info["robot_collision"])
        reward = float(sum(components.values()))
        rewards = {key: reward for key in self.agent_ids}
        self.state.last_rewards = dict(rewards)
        info.update(reward_version=REWARD_VERSION, reward_components=components,
                    original_foundation_reward_components=old_components, participant_score_unchanged=True,
                    reward_revision=REWARD_REVISION, collision_training_cost=self.collision_cost)
        requested, executed = deepcopy(info["requested_actions"]), deepcopy(info["executed_actions"])
        canceled = {key: requested[key] != "WAIT" and executed[key] == "WAIT" for key in self.agent_ids}
        history = {"valid": True, "frame": info["frame"], "previous_frame": info["frame"]-1,
            "submitted_actions": requested, "executed_actions": executed, "move_canceled": canceled,
            "collision_kind": info["collision_kind"] or "none", "post_state_sha256": _state_hash(self.state),
            "unknown_reason": None, "previous_counts": {"valid": previous["valid"], "frame": previous["frame"],
                "move_canceled": deepcopy(previous["consecutive_move_canceled"]), "collision": previous["consecutive_collision"]},
            "consecutive_move_canceled": {key: previous["consecutive_move_canceled"][key]+1 if canceled[key] else 0 for key in self.agent_ids},
            "consecutive_collision": previous["consecutive_collision"]+1 if info["robot_collision"] else 0}
        _validate_history(history, self.state, self.agent_ids, self.config.horizon)
        self._history = history
        info["public_feedback"] = self.public_history()
        return self.observations(), rewards, terminated, truncated, info

    def public_view(self):
        view = NativeWarehouseEnv.public_view(self); view["public_feedback"] = self.public_history(); return view

    def set_state(self, state):
        NativeWarehouseEnv.set_state(self, state); self._history = None

    def snapshot(self):
        payload = NativeWarehouseEnv.snapshot(self)
        payload.update(training_reward_version=REWARD_VERSION, training_reward_config=deepcopy(self.reward_config),
                       training_reward_revision=REWARD_REVISION, public_feedback_version=PUBLIC_FEEDBACK_VERSION,
                       public_feedback_mode=self.mode, public_feedback_history=self._current_history())
        return payload

    def restore(self, payload, *, require_feedback=False):
        if not isinstance(payload, dict): raise ValueError("Snapshot must be an object")
        feedback = {k for k in payload if k.startswith("public_feedback")}
        if require_feedback and feedback != _FEEDBACK_METADATA: raise ValueError("Adapter checkpoint requires feedback")
        if feedback and (feedback != _FEEDBACK_METADATA or payload.get("public_feedback_version") != PUBLIC_FEEDBACK_VERSION or payload.get("public_feedback_mode") != self.mode):
            raise ValueError("Public feedback snapshot mismatch")
        reward = {k for k in payload if k.startswith("training_reward_")}
        if reward and (reward != _REWARD_METADATA or payload.get("training_reward_version") != REWARD_VERSION
                       or payload.get("training_reward_revision") != REWARD_REVISION
                       or payload.get("training_reward_config") != self.reward_config):
            raise ValueError("Training reward snapshot mismatch")
        candidate = NativeWarehouseEnv(self.config)
        candidate.restore(deepcopy({k:v for k,v in payload.items() if k not in _FEEDBACK_METADATA | _REWARD_METADATA}))
        history = deepcopy(payload["public_feedback_history"]) if feedback else _unknown(candidate.state)
        _validate_history(history, candidate.state, candidate.agent_ids, self.config.horizon)
        self.state = candidate.state; self.set_rng_state(candidate.get_rng_state()); self._episode_counter = candidate._episode_counter
        self._history = history

    def branch(self):
        result = type(self)(self.config, self.reward_config, self.collision_cost, mode=self.mode)
        result.restore(self.snapshot(), require_feedback=True); return result


class OnlineAlignmentRuntime:
    def __init__(self, actor_path, *, protocol, expected_actor_sha256, expected_protocol_sha256,
                 expected_bindings=None, allow_test_fixture=False, config=None):
        _sha(expected_actor_sha256, "Actor hash"); _sha(expected_protocol_sha256, "protocol hash")
        if digest(protocol) != expected_protocol_sha256: raise ValueError("Protocol differs from external anchor")
        self._actor_path = Path(actor_path).expanduser().resolve()
        if file_hash(self._actor_path) != expected_actor_sha256: raise ValueError("Actor bytes differ")
        self.actor = OnlineNumPyActor(self._actor_path)
        self.config = config or collaborative_study_config()
        if self.config != collaborative_study_config(): raise ValueError("Online runtime requires public configuration")
        self.protocol, self.actor_sha256, self.protocol_sha256 = deepcopy(protocol), expected_actor_sha256, expected_protocol_sha256
        self.test_fixture = bool(allow_test_fixture)
        metadata = self.actor.metadata
        expected = {"obs_dim": 197, "state_dim": 354, "hidden": 128,
            "feature_names": list(observation_names(self.config))+list(HISTORY_FEATURE_NAMES),
            "public_feedback_version": PUBLIC_FEEDBACK_VERSION, "public_feedback_mode": "observed",
            "action_masks": False, "runtime_action_override": False}
        if any(metadata.get(k) != v for k,v in expected.items()): raise ValueError("Actor observed197 contract differs")
        if metadata.get("protocol_sha256") != expected_protocol_sha256: raise ValueError("Actor protocol binding differs")
        if protocol.get("public_feedback_mode") != "observed" or protocol.get("public_feedback_version") != PUBLIC_FEEDBACK_VERSION:
            raise ValueError("Protocol observed197 contract differs")
        if expected_bindings:
            for key,value in expected_bindings.items():
                actual = self.actor_sha256 if key == "actor_sha256" else metadata.get(key)
                if actual != value: raise ValueError("Actor binding differs: "+key)
        self._reward_config = {**deepcopy(protocol["reward"]), "revision": REWARD_REVISION,
            "collision_training_cost": float(protocol["collision_training_cost"])}
        self.sources = runtime_sources(); self._metadata_sha256 = digest(metadata); self._weights_sha256 = self._weight_digest()
        self.signature = digest({"version": RUNTIME_VERSION, "actor_sha256": self.actor_sha256,
            "protocol_sha256": self.protocol_sha256, "actor_metadata_sha256": self._metadata_sha256,
            "configuration": asdict(self.config), "sources": self.sources})
        self.contract_report = {"version": RUNTIME_VERSION, "signature": self.signature,
            "actor_sha256": self.actor_sha256, "protocol_sha256": self.protocol_sha256,
            "runtime_sources_sha256": digest(self.sources), "obs_dim":197, "state_dim":354,
            "qualification_evaluated":False, "release_ready":False, "explanation_qualified":False,
            "test_fixture":self.test_fixture}
        self.verify_binding()

    def _weight_digest(self):
        return digest({k:{"shape":list(v.shape),"dtype":v.dtype.str,"sha256":sha256(v.tobytes()).hexdigest()}
                       for k,v in self.actor.weights.items()})

    def verify_binding(self):
        if (file_hash(self._actor_path) != self.actor_sha256 or digest(self.protocol) != self.protocol_sha256
                or digest(self.actor.metadata) != self._metadata_sha256 or self._weight_digest() != self._weights_sha256
                or runtime_sources() != self.sources): raise ValueError("Online runtime binding changed")
        return self.signature

    def _new_environment(self):
        return OnlinePublicFeedbackEnvironment(self.config, deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"], mode="observed")

    def _check_environment(self, env):
        if type(env) is not OnlinePublicFeedbackEnvironment or env.config != self.config or env.reward_config != self._reward_config:
            raise ValueError("Online runtime environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        if not isinstance(scenario,dict) or not isinstance(scenario.get("snapshot"),dict): raise ValueError("A raw scenario is required")
        env=self._new_environment(); reset_scenario(env,deepcopy(scenario)); return env

    def from_snapshot(self,snapshot):
        self.verify_binding(); env=self._new_environment(); env.restore(deepcopy(snapshot),require_feedback=True); return env

    def clone(self,env): self._check_environment(env); return self.from_snapshot(env.snapshot())

    def decision(self,env):
        self.verify_binding(); self._check_environment(env)
        if env.done: raise ValueError("round_ended")
        before=digest(env.snapshot()); observations=env.observations(); proposals,probabilities=self.actor.act(observations,deterministic=True)
        if digest(env.snapshot()) != before: raise ValueError("Inference changed environment")
        decision={"version":RUNTIME_VERSION,"runtime_signature":self.signature,"actor_sha256":self.actor_sha256,
            "protocol_sha256":self.protocol_sha256,"frame":env.state.frame,"policy_actions":deepcopy(proposals),
            "proposed_actions":deepcopy(proposals),"probabilities":{k:v.tolist() for k,v in probabilities.items()},
            "observation_hashes":{k:sha256(np.asarray(v,dtype=np.float32).tobytes()).hexdigest() for k,v in observations.items()},
            "masks":False,"post_policy_overrides":0,"robot_1_policy_is_not_participant_input":True}
        return proposals,decision

    def step(self,env,player_action):
        if player_action not in ACTIONS: raise ValueError("invalid_action")
        before=env.snapshot(); proposals,decision=self.decision(env); submitted={"robot_1":player_action,"robot_2":proposals["robot_2"]}
        _,rewards,terminated,truncated,info=env.step(submitted)
        if info["requested_actions"] != submitted: raise ValueError("Neural action was overridden")
        return {"before":before,"after":env.snapshot(),"decision":decision,"policy_actions":deepcopy(proposals),
            "proposed_actions":deepcopy(proposals),"participant_action":player_action,"submitted_actions":deepcopy(submitted),
            "executed_actions":deepcopy(info["executed_actions"]),"physical_actions":deepcopy(info["executed_actions"]),
            "events":deepcopy(info["events"]),"rewards":rewards,"info":info,"done":bool(terminated or truncated),
            "runtime_signature":self.signature}

    def counterfactual(self,snapshot,player_actions,*,steps=3):
        if type(steps) is not int or not 1<=steps<=3 or not isinstance(player_actions,(list,tuple)) or len(player_actions)>steps:
            raise ValueError("counterfactual_steps_must_be_1_to_3")
        env=self.from_snapshot(snapshot); assumptions=list(player_actions)+["WAIT"]*(steps-len(player_actions)); transitions=[]
        for action in assumptions:
            if env.done: break
            transitions.append(self.step(env,action))
        return {"frame":snapshot["state"]["frame"],"assumed_player_actions":assumptions,"transitions":transitions,
            "actor_sha256":self.actor_sha256,"runtime_signature":self.signature,
            "kind":"executable_player_action_intervention","steps_requested":steps,"steps_executed":len(transitions)}


# Concise aliases ease server adapters without importing the research runtime.
AlignmentRuntime = OnlineAlignmentRuntime
PublicHistoryRuntime = OnlineAlignmentRuntime
