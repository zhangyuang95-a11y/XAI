"""The same plain Actor and public encoder used by sampling and local play."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from .environment import NativeWarehouseEnv
from .policy import ACTIONS, NumPyNativeActor
from .scenarios import reset_scenario

RUNTIME_VERSION = "warehouse-native-direct-runtime.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def runtime_signature(actor):
    here = Path(__file__).resolve().parent
    sources = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}
    return sha256(canonical({"version": RUNTIME_VERSION, "actor": actor.artifact_sha256,
                             "sources": sources}).encode()).hexdigest()


class NativeRuntime:
    def __init__(self, actor_path):
        self.actor = NumPyNativeActor(actor_path)
        self.signature = runtime_signature(self.actor)
        probe = NativeWarehouseEnv()
        probe.reset(seed=0)
        if self.actor.obs_dim != probe.observation_size:
            raise ValueError("Actor/observation contract mismatch")
        if self.actor.metadata.get("feature_names") != list(probe.feature_names):
            raise ValueError("Actor public feature schema is not the current native schema")

    def environment(self, scenario):
        env = NativeWarehouseEnv()
        reset_scenario(env, scenario)
        return env

    def decision(self, env):
        """No player-command argument; decisions are pure functions of S_t."""
        if env.done:
            raise ValueError("round_ended")
        observations = env.observations()
        proposed, probabilities = self.actor.act(observations, deterministic=True)
        return proposed, {"version": RUNTIME_VERSION, "actor_sha256": self.actor.artifact_sha256,
            "frame": env.state.frame, "policy_actions": proposed,
            "probabilities": {key: value.tolist() for key, value in probabilities.items()},
            "observation_hashes": {key: sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()
                                   for key, value in observations.items()},
            "masks": False, "post_policy_overrides": 0}

    def step(self, env, player_action):
        if player_action not in ACTIONS:
            raise ValueError("invalid_action")
        before = env.snapshot()
        proposals, decision = self.decision(env)
        submitted = {"robot_1": player_action, "robot_2": proposals["robot_2"]}
        _, rewards, terminated, truncated, info = env.step(submitted)
        assert submitted["robot_2"] == proposals["robot_2"]
        return {"before": before, "after": env.snapshot(), "decision": decision,
                "participant_action": player_action, "submitted_actions": submitted,
                "executed_actions": info["executed_actions"], "rewards": rewards,
                "done": bool(terminated or truncated)}

    def counterfactual(self, snapshot, player_actions, *, steps=3):
        """Actual player interventions; unspecified subsequent actions are WAIT."""
        if not 1 <= steps <= 3 or len(player_actions) > steps:
            raise ValueError("counterfactual_steps_must_be_1_to_3")
        if any(action not in ACTIONS for action in player_actions):
            raise ValueError("invalid_counterfactual_action")
        env = NativeWarehouseEnv()
        env.restore(snapshot)
        assumptions = list(player_actions) + ["WAIT"] * (steps-len(player_actions))
        trajectory = []
        for action in assumptions:
            if env.done:
                break
            trajectory.append(self.step(env, action))
        return {"frame": snapshot["state"]["frame"] if "state" in snapshot and isinstance(snapshot["state"], dict) and "frame" in snapshot["state"] else None,
                "assumed_player_actions": assumptions, "transitions": trajectory,
                "actor_sha256": self.actor.artifact_sha256,
                "kind": "executable_player_action_intervention"}
