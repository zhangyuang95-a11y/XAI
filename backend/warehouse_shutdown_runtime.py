"""Explicit retained-beta Actor runtime over unchanged public warehouse physics.

Training beta is provenance, never an extra participant penalty. Admission uses
the real new stage contract; no older Actor metadata or protocol is relabelled.
This runtime grants neither model qualification nor experiment permission.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from backend import warehouse_public_history_runtime as physical
from backend.training import warehouse_native_shutdown_stage_evaluation as admission
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.policy import NumPyNativeActor

RUNTIME_VERSION = "warehouse-native-retained-beta-direct-runtime.v1"


def runtime_sources():
    result = physical.runtime_sources()
    result.update(admission.execution_sources())
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


class ShutdownRuntime(physical.PublicHistoryRuntime):
    """Own admission/signature, shared pure public physics and isolation methods."""
    def __init__(self, actor_path, *, protocol, expected_actor_sha256, expected_protocol_sha256,
                 expected_bindings, allow_test_fixture=False, config=None):
        physical._sha(expected_actor_sha256, "Actor hash")
        physical._sha(expected_protocol_sha256, "protocol hash")
        if type(allow_test_fixture) is not bool or not isinstance(expected_bindings, dict):
            raise ValueError("Explicit stage provenance and fixture scope are required")
        if digest(protocol) != expected_protocol_sha256:
            raise ValueError("Stage protocol differs from its external anchor")
        if expected_bindings.get("actor_sha256") != expected_actor_sha256:
            raise ValueError("Stage Actor anchors differ")
        self._actor_path = Path(actor_path).expanduser().resolve()
        if file_hash(self._actor_path) != expected_actor_sha256:
            raise ValueError("Stage Actor differs from its external anchor")
        self.actor = NumPyNativeActor(self._actor_path)
        self.config = config or collaborative_study_config()
        receipt = admission.validate_actor(self.actor, expected_bindings, protocol,
            config=self.config, allow_test_fixture=allow_test_fixture)
        self.protocol = deepcopy(protocol)
        self.actor_sha256, self.protocol_sha256 = expected_actor_sha256, expected_protocol_sha256
        self.test_fixture = allow_test_fixture
        self.sources = runtime_sources()
        self._metadata_sha256 = digest(self.actor.metadata)
        self._weights_sha256 = self._weight_digest()
        self._configuration_sha256 = digest(asdict(self.config))
        self.signature = digest({"version": RUNTIME_VERSION, "actor_sha256": self.actor_sha256,
            "protocol_sha256": self.protocol_sha256, "actor_metadata_sha256": self._metadata_sha256,
            "configuration": asdict(self.config), "sources": self.sources})
        self.contract_report = {"version": RUNTIME_VERSION, "signature": self.signature,
            "actor_sha256": self.actor_sha256, "protocol_sha256": self.protocol_sha256,
            "runtime_sources_sha256": digest(self.sources), "obs_dim": 197, "state_dim": 354,
            "training_stage": receipt["training_stage"], "shutdown_arm": self.actor.metadata["shutdown_arm"],
            "training_own_shutdown_beta": self.actor.metadata["own_shutdown_beta"],
            "public_reward": "original_shared_r1", "qualification_evaluated": False,
            "release_ready": False, "explanation_qualified": False, "test_fixture": allow_test_fixture}
        self._reward_config = deepcopy(self._new_environment().reward_config)
        self.verify_binding()

    def verify_binding(self):
        if (type(self.actor) is not NumPyNativeActor
                or digest(asdict(self.config)) != self._configuration_sha256
                or file_hash(self._actor_path) != self.actor_sha256 or digest(self.protocol) != self.protocol_sha256
                or digest(self.actor.metadata) != self._metadata_sha256 or self._weight_digest() != self._weights_sha256
                or runtime_sources() != self.sources):
            raise ValueError("Stage runtime Actor, protocol or execution source changed")
        expected = digest({"version": RUNTIME_VERSION, "actor_sha256": self.actor_sha256,
            "protocol_sha256": self.protocol_sha256, "actor_metadata_sha256": self._metadata_sha256,
            "configuration": asdict(self.config), "sources": self.sources})
        if self.signature != expected:
            raise ValueError("Stage runtime signature differs from its bound inputs")
        return self.signature

    def decision(self, env):
        """The only action source is the same frozen NN at the pre-action state."""
        self.verify_binding(); self._check_environment(env)
        if env.done:
            raise ValueError("round_ended")
        before = digest(env.snapshot())
        observations = env.observations()
        proposals, probabilities = self.actor.act(observations, deterministic=True)
        if digest(env.snapshot()) != before:
            raise ValueError("Neural inference changed the environment")
        decision = {"version": RUNTIME_VERSION, "runtime_signature": self.signature,
            "actor_sha256": self.actor_sha256, "protocol_sha256": self.protocol_sha256,
            "frame": env.state.frame, "policy_actions": deepcopy(proposals), "proposed_actions": deepcopy(proposals),
            "probabilities": {key: value.tolist() for key, value in probabilities.items()},
            "observation_hashes": {key: sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()
                for key, value in observations.items()}, "masks": False, "post_policy_overrides": 0,
            "robot_1_policy_is_not_participant_input": True}
        return proposals, decision
