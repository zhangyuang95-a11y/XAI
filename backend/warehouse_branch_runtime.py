"""Independent branch-Actor runtime over the original public warehouse physics.

The closed historical runtime registry is untouched. This module admits its
own real exported version and exposes an explicit verification entry point.
Construction verifies arrays and signatures but creates no environment and
performs no neural inference; physical methods are inherited unchanged.
"""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from backend import warehouse_public_history_runtime as physical
from backend.training import warehouse_family_branch_evaluation as admission
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_revision_reward import REWARD_REVISION
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.policy import NumPyNativeActor

RUNTIME_VERSION = "warehouse-family-branch-direct-runtime.v2"
VERIFY_VERSION = "warehouse-family-branch-runtime-verification.v2"
FAMILY = "branch_feedback197"


def runtime_sources():
    result = physical.runtime_sources()
    result.update(admission.execution_sources())
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


class BranchRuntime(physical.PublicHistoryRuntime):
    """Own admission/signature, original five-action NN and public physics."""
    def __init__(self, actor_path, *, protocol, expected_actor_sha256, expected_protocol_sha256,
                 expected_bindings=None, allow_test_fixture=False, config=None):
        physical._sha(expected_actor_sha256, "Actor hash")
        physical._sha(expected_protocol_sha256, "protocol hash")
        if digest(protocol) != expected_protocol_sha256: raise ValueError("Branch protocol external anchor differs")
        self._actor_path = Path(actor_path).expanduser().resolve()
        if file_hash(self._actor_path) != expected_actor_sha256: raise ValueError("Branch Actor bytes differ")
        self.actor = NumPyNativeActor(self._actor_path)
        receipt = admission.validate_actor(self.actor, protocol, expected_actor_sha256,
            allow_test_fixture=allow_test_fixture)
        self.config = config or collaborative_study_config()
        if self.config != collaborative_study_config(): raise ValueError("Branch runtime keeps the original public configuration")
        if expected_bindings is not None:
            if not isinstance(expected_bindings, dict): raise ValueError("Expected Actor bindings must be an object")
            for key, value in expected_bindings.items():
                actual = self.actor.artifact_sha256 if key == "actor_sha256" else self.actor.metadata.get(key)
                if digest(actual) != digest(value):
                    raise ValueError("Branch runtime external binding differs: " + key)
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
            "training_stage": "branch_feedback", "shutdown_arm": self.actor.metadata["shutdown_arm"],
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
            raise ValueError("Branch runtime Actor, configuration, source or signature changed")
        return self.signature

    def _verify_context_objects(self):
        guard = self._context_guard
        if (guard is None or self.actor is not guard["actor"] or type(self.actor) is not NumPyNativeActor
                or self.actor.weights is not guard["weights"] or self.actor.metadata is not guard["metadata"]
                or self.protocol is not guard["protocol"] or self.config is not guard["config"]
                or self.sources is not guard["sources"] or self._reward_config is not guard["reward"]
                or self.signature != guard["signature"]
                or (self.actor.obs_dim, self.actor.state_dim, self.actor.hidden) != (197, 354, 128)
                or guard["env"].config is not guard["env_config"]
                or guard["env"].reward_config is not guard["env_reward"]
                or tuple(self.actor.weights) != guard["weight_keys"]
                or (self.actor_sha256, self.protocol_sha256, self._metadata_sha256,
                    self._weights_sha256, self._configuration_sha256) != guard["anchors"]):
            raise ValueError("Verified context runtime/Actor/environment object or signature changed")
        for key, original, shape, dtype in guard["arrays"]:
            value = self.actor.weights[key]
            if value is not original or value.shape != shape or value.dtype != dtype or value.flags.writeable:
                raise ValueError("Verified context requires the same six read-only Actor arrays")
        return self.signature

    def verify_binding(self):
        """Full hashes outside a context; object/array checks inside one."""
        if self._context_guard is not None: return self._verify_context_objects()
        return self._full_verify_binding()

    @contextmanager
    def verified_context(self, env):
        """Full content verification at both boundaries, before caller ACK.

        Inner decisions/clones retain their original NN and physical methods.
        Their verification checks object identities and immutable array shape,
        dtype and flags; it deliberately does not claim per-step content hashes.
        Exceptions always clear the context and do not authorize a retry.
        """
        if self._context_guard is not None: raise ValueError("Nested verified contexts are not supported")
        self._full_verify_binding(); self._check_environment(env)
        self._context_guard = {"actor": self.actor, "weights": self.actor.weights,
            "metadata": self.actor.metadata, "protocol": self.protocol, "config": self.config,
            "sources": self.sources, "reward": self._reward_config, "signature": self.signature,
            "env": env, "env_config": env.config, "env_reward": env.reward_config,
            "weight_keys": tuple(self.actor.weights),
            "arrays": [(key, value, value.shape, value.dtype) for key, value in self.actor.weights.items()],
            "anchors": (self.actor_sha256, self.protocol_sha256, self._metadata_sha256,
                self._weights_sha256, self._configuration_sha256)}
        body_error = None
        try:
            self._verify_context_objects()
            yield self
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                self._full_verify_binding()
                self._verify_context_objects()
                self._check_environment(env)
            except BaseException as verification_error:
                if body_error is None: raise
                body_error.add_note("Context exit verification also failed: " + str(verification_error))
            finally:
                self._context_guard = None

    def decision(self, env):
        proposals, decision = super().decision(env)
        # The shared computation returns our runtime's own event envelope.
        return proposals, {**decision, "version": RUNTIME_VERSION}


def verify(runtime, *, allow_test_fixture=False, expected_family=None, expected_signature=None):
    """Zero-forward verification for this new exact type; never an eligibility grant."""
    if type(runtime) is not BranchRuntime or type(allow_test_fixture) is not bool or runtime.test_fixture is not allow_test_fixture:
        raise ValueError("Only the exact new BranchRuntime and matching fixture scope are supported")
    if expected_family is not None and expected_family != FAMILY: raise ValueError("Branch runtime family differs")
    signature = runtime.verify_binding()
    if expected_signature is not None and expected_signature != signature: raise ValueError("Branch runtime signature differs")
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
    result = BranchRuntime(runtime._actor_path, protocol=runtime.protocol,
        expected_actor_sha256=runtime.actor_sha256, expected_protocol_sha256=runtime.protocol_sha256,
        allow_test_fixture=allow_test_fixture, config=runtime.config)
    if verify(result, allow_test_fixture=allow_test_fixture) != before:
        raise ValueError("Private branch runtime differs from its source")
    return result
