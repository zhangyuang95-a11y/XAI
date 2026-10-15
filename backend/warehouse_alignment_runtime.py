"""Genuine alignment Actor admission; inherited verified contexts and public physics.

The historical registry and branch runtime stay unchanged. Only this module's
exact AlignmentRuntime type is accepted by its verifier. New alignment metadata is
validated directly, never rewritten to obtain older-family admission.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from backend import warehouse_public_history_runtime as physical
from backend import warehouse_branch_runtime as inherited
from backend.training import warehouse_family_alignment_evaluation as admission
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_revision_reward import REWARD_REVISION
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.policy import NumPyNativeActor

RUNTIME_VERSION = "warehouse-family-alignment-direct-runtime.v1"
VERIFY_VERSION = "warehouse-family-alignment-runtime-verification.v1"
FAMILY = "alignment_feedback197"


def runtime_sources():
    result = physical.runtime_sources()
    result.update(admission.execution_sources())
    for path in (Path(__file__), Path(inherited.__file__)):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


class AlignmentRuntime(inherited.BranchRuntime):
    """New admission/hash anchors; unchanged original NN, physics and context guard."""
    def __init__(self, actor_path, *, protocol, expected_actor_sha256, expected_protocol_sha256,
                 expected_bindings=None, allow_test_fixture=False, config=None):
        physical._sha(expected_actor_sha256, "Actor hash")
        physical._sha(expected_protocol_sha256, "protocol hash")
        if digest(protocol) != expected_protocol_sha256: raise ValueError("Alignment protocol external anchor differs")
        self._actor_path = Path(actor_path).expanduser().resolve()
        if file_hash(self._actor_path) != expected_actor_sha256: raise ValueError("Alignment Actor bytes differ")
        self.actor = NumPyNativeActor(self._actor_path)
        receipt = admission.validate_actor(self.actor, protocol, expected_actor_sha256,
            allow_test_fixture=allow_test_fixture)
        self.config = config or collaborative_study_config()
        if self.config != collaborative_study_config(): raise ValueError("Alignment runtime keeps the original public configuration")
        if expected_bindings is not None:
            if not isinstance(expected_bindings, dict): raise ValueError("Expected Actor bindings must be an object")
            for key, value in expected_bindings.items():
                actual = self.actor.artifact_sha256 if key == "actor_sha256" else self.actor.metadata.get(key)
                if digest(actual) != digest(value):
                    raise ValueError("Alignment runtime external binding differs: " + key)
        self.protocol = deepcopy(protocol)
        self.actor_sha256, self.protocol_sha256 = expected_actor_sha256, expected_protocol_sha256
        self.test_fixture = allow_test_fixture
        self.sources = runtime_sources()
        self._metadata_sha256, self._weights_sha256 = digest(self.actor.metadata), self._weight_digest()
        self._configuration_sha256 = digest(asdict(self.config))
        self.signature = self._signature()
        self.contract_report = {"version": RUNTIME_VERSION, "signature": self.signature,
            "actor_sha256": self.actor_sha256, "protocol_sha256": self.protocol_sha256,
            "runtime_sources_sha256": digest(self.sources), "obs_dim": 197, "state_dim": 354,
            "training_stage": "alignment_feedback", "shutdown_arm": self.actor.metadata["shutdown_arm"],
            "training_own_shutdown_beta": self.actor.metadata["own_shutdown_beta"],
            "public_reward": "original_shared_r1", "qualification_evaluated": False,
            "release_ready": False, "explanation_qualified": False, "test_fixture": allow_test_fixture}
        # Exactly the reward configuration assembled by the inherited public
        # environment; avoiding a throwaway environment keeps admission pure.
        self._reward_config = {**deepcopy(physical.REWARD), "revision": REWARD_REVISION,
            "collision_training_cost": .05}
        self._context_guard = None
        self.verify_binding()

    def _signature(self):
        return digest({"version": RUNTIME_VERSION, "actor_sha256": self.actor_sha256,
            "protocol_sha256": self.protocol_sha256, "actor_metadata_sha256": self._metadata_sha256,
            "configuration": asdict(self.config), "sources": self.sources})

    def _full_verify_binding(self):
        if (type(self.actor) is not NumPyNativeActor or digest(asdict(self.config)) != self._configuration_sha256
                or self.test_fixture is not self.actor.metadata.get("test_fixture")
                or self.test_fixture is not self.protocol.get("test_fixture")
                or file_hash(self._actor_path) != self.actor_sha256 or digest(self.protocol) != self.protocol_sha256
                or digest(self.actor.metadata) != self._metadata_sha256 or self._weight_digest() != self._weights_sha256
                or runtime_sources() != self.sources or self.signature != self._signature()
                or self._reward_config != {**physical.REWARD, "revision": REWARD_REVISION, "collision_training_cost": .05}):
            raise ValueError("Alignment runtime Actor, configuration, source or signature changed")
        return self.signature

    def decision(self, env):
        proposals, decision = physical.PublicHistoryRuntime.decision(self, env)
        return proposals, {**decision, "version": RUNTIME_VERSION}


def verify(runtime, *, allow_test_fixture=False, expected_family=None, expected_signature=None):
    """Zero-forward verification for this new exact type; never an eligibility grant."""
    if type(runtime) is not AlignmentRuntime or type(allow_test_fixture) is not bool or runtime.test_fixture is not allow_test_fixture:
        raise ValueError("Only the exact new AlignmentRuntime and matching fixture scope are supported")
    if expected_family is not None and expected_family != FAMILY: raise ValueError("Alignment runtime family differs")
    signature = runtime.verify_binding()
    if expected_signature is not None and expected_signature != signature: raise ValueError("Alignment runtime signature differs")
    return {"version": VERIFY_VERSION, "family": FAMILY, "runtime_version": RUNTIME_VERSION,
        "runtime_signature": signature, "runtime_sources": deepcopy(runtime.sources),
        "runtime_sources_sha256": digest(runtime.sources), "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256, "actor_metadata_sha256": digest(runtime.actor.metadata),
        "actor_experiment_version": runtime.actor.metadata["experiment_version"],
        "configuration": asdict(runtime.config), "test_fixture": allow_test_fixture,
        "verification_scope": "context_object_checks" if runtime._context_guard is not None else "full_content_hashes",
        "scope": "runtime_identity_and_source_only", "qualification_evaluated": False,
        "release_ready": False, "explanation_qualified": False}

def fresh_instance(runtime, *, allow_test_fixture=False, expected_family=None, expected_signature=None):
    before = verify(runtime, allow_test_fixture=allow_test_fixture,
        expected_family=expected_family, expected_signature=expected_signature)
    result = AlignmentRuntime(runtime._actor_path, protocol=runtime.protocol,
        expected_actor_sha256=runtime.actor_sha256, expected_protocol_sha256=runtime.protocol_sha256,
        allow_test_fixture=allow_test_fixture, config=runtime.config)
    if verify(result, allow_test_fixture=allow_test_fixture) != before:
        raise ValueError("Private alignment runtime differs from its source")
    return result
