"""Direct observed197 warehouse runtime; this contract grants no release eligibility.

Only the fixed NumPy neural Actor chooses robot_2 commands. The environment may
cancel a physical move, but no policy action is masked, replaced or reselected.
Training delivery-credit adjustments are not part of public runtime rewards.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_v2 import v2_source_hashes
from backend.training.warehouse_native_public_feedback import (
    PublicFeedbackEnvironment, VERSION as OBSERVER_VERSION, HISTORY_FEATURE_NAMES,
)
from backend.training.warehouse_native_public_feedback_evaluation import REWARD
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import ACTIONS, NumPyNativeActor
from env.warehouse_native.scenarios import reset_scenario

RUNTIME_VERSION = "warehouse-native-observed197-direct-runtime.v1"
ACTOR_PROTOCOLS = {
    "warehouse-native-continuation-trainer.v1": "warehouse-native-continuation-protocol.v1",
    "warehouse-native-continuation-feedback-trainer.v1": "warehouse-native-continuation-feedback-protocol.v1",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def runtime_sources():
    result = v2_source_hashes()
    paths = [Path(__file__), ROOT / "backend/training/warehouse_native_revision_reward.py",
        ROOT / "backend/training/warehouse_native_public_feedback.py",
        ROOT / "backend/training/warehouse_native_public_feedback_evaluation.py"]
    for path in paths: result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _sha(value, name):
    if type(value) is not str or not _SHA.fullmatch(value): raise ValueError("Invalid external " + name)
    return value


class PublicHistoryRuntime:
    def __init__(self, actor_path, *, protocol, expected_actor_sha256, expected_protocol_sha256,
                 expected_bindings=None, allow_test_fixture=False, config=None):
        _sha(expected_actor_sha256, "Actor hash"); _sha(expected_protocol_sha256, "protocol hash")
        if type(allow_test_fixture) is not bool or not isinstance(protocol, dict):
            raise ValueError("Explicit runtime provenance is required")
        if digest(protocol) != expected_protocol_sha256:
            raise ValueError("Runtime protocol differs from its external anchor")
        self._actor_path = Path(actor_path).expanduser().resolve()
        if file_hash(self._actor_path) != expected_actor_sha256:
            raise ValueError("Runtime Actor differs from its external anchor")
        self.actor = NumPyNativeActor(self._actor_path)
        if self.actor.artifact_sha256 != expected_actor_sha256 or file_hash(self._actor_path) != expected_actor_sha256:
            raise ValueError("Runtime Actor bytes changed while loading")
        self.config = config or collaborative_study_config()
        if (not allow_test_fixture and self.config != collaborative_study_config()) or not 1 <= self.config.horizon <= 120:
            raise ValueError("Runtime requires the registered public warehouse configuration")
        self.protocol = deepcopy(protocol)
        self.actor_sha256, self.protocol_sha256 = expected_actor_sha256, expected_protocol_sha256
        self.test_fixture = allow_test_fixture
        metadata = self.actor.metadata
        version = metadata.get("experiment_version")
        if (version not in ACTOR_PROTOCOLS or protocol.get("version") != ACTOR_PROTOCOLS[version]
                or metadata.get("protocol_sha256") != expected_protocol_sha256
                or metadata.get("test_fixture", False) is not allow_test_fixture
                or protocol.get("test_fixture", False) is not allow_test_fixture):
            raise ValueError("Runtime requires a matching actual197 continuation identity")
        expected = {"obs_dim": 197, "state_dim": 354, "hidden": 128,
            "feature_names": list(observation_names(self.config)) + list(HISTORY_FEATURE_NAMES),
            "public_feedback_version": OBSERVER_VERSION, "public_feedback_mode": "observed",
            "action_masks": False, "runtime_action_override": False}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Runtime Actor dimensions, ordered public features or direct-action contract differ")
        if any(type(metadata.get(key)) is not int for key in ("obs_dim", "state_dim", "hidden")):
            raise ValueError("Actor dimensions must be exact integers")
        for key in ("source_sha256", "scenario_manifest_sha256", "initialization_sha256", "source_checkpoint_sha256"):
            _sha(metadata.get(key), key)
        if (protocol.get("public_feedback_mode") != "observed" or protocol.get("public_feedback_version") != OBSERVER_VERSION
                or protocol.get("reward") != REWARD or protocol.get("collision_training_cost") != .05):
            raise ValueError("Runtime retains original shared r1 rewards and observed public history")
        if (metadata.get("cycle_id") != protocol.get("cycle_id") or type(metadata.get("cycle_id")) is not str
                or not metadata["cycle_id"] or metadata.get("branch") != protocol.get("branch")
                or metadata.get("delivery_credit_alpha") != protocol.get("delivery_credit_alpha")):
            raise ValueError("Runtime source cycle or inherited training condition differs")
        source = protocol.get("source", {})
        if (metadata.get("source_checkpoint_sha256") != source.get("checkpoint_sha256")
                or metadata.get("initialization_sha256") != source.get("state_sha256")
                or metadata.get("branch") != source.get("branch")
                or type(metadata.get("joint_steps")) is not int or metadata["joint_steps"] < 0
                or type(metadata.get("source_counters", {}).get("joint_steps")) is not int
                or metadata["source_counters"]["joint_steps"] != source.get("cumulative_joint_steps")
                or not isinstance(metadata.get("source_lineage"), list) or not metadata["source_lineage"]
                or metadata["source_lineage"] != protocol.get("source_lineage")
                or metadata["source_lineage"][-1] != source):
            raise ValueError("Runtime Actor source lineage or cumulative clock differs")
        if version.endswith("feedback-trainer.v1"):
            condition = metadata.get("feedback_branch")
            if condition not in ("control", "feedback") or metadata.get("feedback_enabled") is not (condition == "feedback"):
                raise ValueError("The recorded feedback condition differs")
            _sha(metadata.get("actor_parameters_sha256"), "Actor parameter hash")
        if expected_bindings is not None:
            if not isinstance(expected_bindings, dict): raise ValueError("Expected Actor bindings must be an object")
            for key, value in expected_bindings.items():
                actual = self.actor.artifact_sha256 if key == "actor_sha256" else metadata.get(key)
                if actual != value: raise ValueError("Runtime external Actor binding differs: " + key)
        self.sources = runtime_sources()
        self._metadata_sha256 = digest(metadata)
        self._weights_sha256 = self._weight_digest()
        self.signature = digest({"version": RUNTIME_VERSION, "actor_sha256": self.actor_sha256,
            "protocol_sha256": self.protocol_sha256, "actor_metadata_sha256": self._metadata_sha256,
            "configuration": asdict(self.config), "sources": self.sources})
        self.contract_report = {"version": RUNTIME_VERSION, "signature": self.signature,
            "actor_sha256": self.actor_sha256, "protocol_sha256": self.protocol_sha256,
            "runtime_sources_sha256": digest(self.sources), "obs_dim": 197, "state_dim": 354,
            "qualification_evaluated": False, "release_ready": False, "test_fixture": allow_test_fixture}
        # A config-only construction does not reset, sample or infer any action.
        template = self._new_environment()
        self._reward_config = deepcopy(template.reward_config)

    def _weight_digest(self):
        return digest({key: {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": sha256(value.tobytes()).hexdigest()} for key, value in self.actor.weights.items()})

    def verify_binding(self):
        if (file_hash(self._actor_path) != self.actor_sha256 or digest(self.protocol) != self.protocol_sha256
                or digest(self.actor.metadata) != self._metadata_sha256 or self._weight_digest() != self._weights_sha256
                or runtime_sources() != self.sources):
            raise ValueError("Runtime Actor, protocol or execution source changed")
        return self.signature

    def _new_environment(self):
        return PublicFeedbackEnvironment(self.config, deepcopy(REWARD), collision_cost=.05, mode="observed")

    def _check_environment(self, env):
        if (type(env) is not PublicFeedbackEnvironment or env.mode != "observed"
                or env.config != self.config or env.reward_config != self._reward_config):
            raise ValueError("Runtime requires its original shared-reward observed environment")
        env._require_state()

    def environment(self, scenario):
        """Registered raw scenario starts have unknown history, never reconstructed history."""
        self.verify_binding()
        if not isinstance(scenario, dict) or not isinstance(scenario.get("snapshot"), dict):
            raise ValueError("A registered raw scenario is required")
        if any(key.startswith("public_feedback") for key in scenario["snapshot"]):
            raise ValueError("Use from_snapshot for a confirmed public-history frame")
        env = self._new_environment()
        reset_scenario(env, deepcopy(scenario))
        if env.public_history()["valid"]: raise ValueError("Raw starts cannot invent confirmed history")
        return env

    def from_snapshot(self, snapshot):
        self.verify_binding()
        env = self._new_environment()
        env.restore(deepcopy(snapshot), require_feedback=True)
        return env

    def clone(self, env):
        self._check_environment(env)
        return self.from_snapshot(env.snapshot())

    def decision(self, env):
        """Pure S_t inference. There is deliberately no player-command parameter."""
        self.verify_binding(); self._check_environment(env)
        if env.done: raise ValueError("round_ended")
        before = digest(env.snapshot())
        observations = env.observations()
        proposals, probabilities = self.actor.act(observations, deterministic=True)
        if digest(env.snapshot()) != before: raise ValueError("Neural inference changed the environment")
        decision = {"version": RUNTIME_VERSION, "runtime_signature": self.signature,
            "actor_sha256": self.actor_sha256, "protocol_sha256": self.protocol_sha256,
            "frame": env.state.frame, "policy_actions": deepcopy(proposals), "proposed_actions": deepcopy(proposals),
            "probabilities": {key: value.tolist() for key, value in probabilities.items()},
            "observation_hashes": {key: sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()
                for key, value in observations.items()}, "masks": False, "post_policy_overrides": 0,
            "robot_1_policy_is_not_participant_input": True}
        return proposals, decision

    def step(self, env, player_action):
        if type(player_action) is not str or player_action not in ACTIONS: raise ValueError("invalid_action")
        before = env.snapshot()
        proposals, decision = self.decision(env)
        submitted = {"robot_1": player_action, "robot_2": proposals["robot_2"]}
        _, rewards, terminated, truncated, info = env.step(submitted)
        if info["requested_actions"] != submitted or submitted["robot_2"] != proposals["robot_2"]:
            raise ValueError("Neural submitted action was overridden")
        return {"before": before, "after": env.snapshot(), "decision": decision,
            "policy_actions": deepcopy(proposals), "proposed_actions": deepcopy(proposals),
            "participant_action": player_action, "submitted_actions": deepcopy(submitted),
            "executed_actions": deepcopy(info["executed_actions"]), "physical_actions": deepcopy(info["executed_actions"]),
            "events": deepcopy(info["events"]), "rewards": rewards, "info": info,
            "done": bool(terminated or truncated), "runtime_signature": self.signature}

    def counterfactual(self, snapshot, player_actions, *, steps=3):
        if (type(steps) is not int or not 1 <= steps <= 3 or not isinstance(player_actions, (list, tuple))
                or len(player_actions) > steps):
            raise ValueError("counterfactual_steps_must_be_1_to_3")
        if any(type(action) is not str or action not in ACTIONS for action in player_actions):
            raise ValueError("invalid_counterfactual_action")
        env = self.from_snapshot(snapshot)
        assumptions = list(player_actions) + ["WAIT"] * (steps - len(player_actions))
        trajectory = []
        for action in assumptions:
            if env.done: break
            trajectory.append(self.step(env, action))
        return {"frame": snapshot["state"]["frame"], "assumed_player_actions": assumptions,
            "transitions": trajectory, "actor_sha256": self.actor_sha256, "runtime_signature": self.signature,
            "kind": "executable_player_action_intervention", "steps_requested": steps,
            "steps_executed": len(trajectory)}
