"""Frame-bound evidence rendering for the closed warehouse runtime registry.

Only the construction and source boundary differ from the frozen observed197
renderer. Its parsing, actual-transition replay and public answer algorithms
are inherited unchanged. This component never grants participant eligibility.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import re
import struct

import numpy as np

from backend import warehouse_runtime_family as registry
from backend import warehouse_public_history_explanation as evidence
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training import warehouse_native_expanded_rcpd as expanded
from core.program import ExecutableProgram
from env.warehouse_native.policy import ACTIONS

VERSION = "warehouse-native-runtime-family-evidence-answers.v2"


def explanation_sources():
    sources = registry.execution_sources()
    for name, value in evidence.explanation_sources().items():
        if name in sources and sources[name] != value:
            raise ValueError("Explanation and runtime-family source closures disagree")
        sources[name] = value
    sources[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    for name in ("backend/training/warehouse_native_expanded_rcpd.py",
                 "backend/training/warehouse_native_program_batch.py",
                 "backend/training/warehouse_native_public_feedback_initialization.py"):
        value = file_hash(ROOT / name)
        if name in sources and sources[name] != value:
            raise ValueError("Expanded program/parameter encoding source differs")
        sources[name] = value
    return sources


def actor_parameter_sha256(actor):
    """Pure bytes for the existing dense-CPU-float32 Actor state-dict digest.

    This implements the documented T/B tags of initialization_sha256 on the
    six actual NumPy weights; no Torch object, checkpoint or NN is constructed.
    It verifies present parameters, not training history or extraction quality.
    """
    shapes = {"0.weight": (128, 197), "0.bias": (128,),
              "2.weight": (128, 128), "2.bias": (128,),
              "4.weight": (5, 128), "4.bias": (5,)}
    if set(actor.weights) != set(shapes):
        raise ValueError("Expanded evidence requires the six actual Actor arrays")
    result = sha256()
    def part(tag, raw):
        result.update(tag + struct.pack("!Q", len(raw)) + raw)
    part(b"d", str(len(shapes)).encode())
    for name in sorted(shapes):
        value = np.asarray(actor.weights[name])
        if value.dtype != np.float32 or value.shape != shapes[name] or not np.isfinite(value).all():
            raise ValueError("Expanded evidence requires finite float32 Actor parameters")
        part(b"s", name.encode())
        part(b"T", json.dumps(["torch.float32", list(value.shape)]).encode())
        part(b"B", np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def _representation(program, runtime):
    """Expanded capacity is conditional on its true schema and current Actor."""
    meta = program.metadata
    declared = (meta.get("native_feedback_version") == expanded.VERSION
                or meta.get("prediction_semantics") == expanded.program_batch.VERSION)
    if not declared:
        if program.root.depth() > 8 or program.root.leaf_count() > 64:
            raise ValueError("Legacy observed197 trees retain their 8/64 capacity")
        return {"schema": "legacy_observed197", "maximum_depth": 8, "maximum_leaves": 64}
    if (meta.get("native_feedback_version") != expanded.VERSION
            or meta.get("prediction_semantics") != expanded.program_batch.VERSION
            or digest(meta.get("native_feedback_config")) != digest(expanded.contract()["feedback_config"])
            or program.root.depth() > 16 or program.root.leaf_count() > 256):
        raise ValueError("Expanded version, exact predicates, configuration or capacity differs")
    binding = meta.get("observed197_bindings")
    expected_keys = {"config_sha256", "training_data_sha256", "selection_data_sha256",
                     "actor_sha256", "actor_parameters_sha256", "cumulative_fit_step"}
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ValueError("Expanded program needs its complete observed197 evidence binding")
    for name in expected_keys - {"cumulative_fit_step"}:
        if type(binding[name]) is not str or not re.fullmatch(r"[0-9a-f]{64}", binding[name]):
            raise ValueError("Invalid expanded evidence hash")
    metadata = runtime.actor.metadata
    inherited, additional = metadata["source_counters"]["joint_steps"], metadata["joint_steps"]
    if (type(inherited) is not int or type(additional) is not int or min(inherited, additional) < 0
            or type(binding["cumulative_fit_step"]) is not int
            or binding["cumulative_fit_step"] != inherited + additional
            or binding["actor_sha256"] != runtime.actor_sha256
            or binding["config_sha256"] != digest(expanded.contract())):
        raise ValueError("Expanded program belongs to another Actor, fit clock or search contract")
    parameter_sha = actor_parameter_sha256(runtime.actor)
    if (binding["actor_parameters_sha256"] != parameter_sha
            or ("actor_parameters_sha256" in metadata and metadata["actor_parameters_sha256"] != parameter_sha)):
        raise ValueError("Expanded program parameters differ from the actual frozen NN")
    if meta.get("metrics", {}).get("explanation_eligible") is not False:
        raise ValueError("Expanded tree capacity cannot grant explanation qualification")
    return {"schema": expanded.VERSION, "prediction_semantics": expanded.program_batch.VERSION,
            "maximum_depth": 16, "maximum_leaves": 256, "observed197_bindings": deepcopy(binding),
            "actor_parameters_sha256": parameter_sha,
            "qualification_evaluated": False}


class FamilyExplainer(evidence.PublicHistoryExplainer):
    """An identity-checked renderer; qualification is a separate release step."""

    def __init__(self, program_path, *, expected_program_sha256, runtime,
                 allow_test_fixture=False):
        identity = registry.verify(runtime, allow_test_fixture=allow_test_fixture)
        self.program_path = Path(program_path).expanduser().resolve()
        raw = self.program_path.read_bytes()
        if (not isinstance(expected_program_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_program_sha256)
                or sha256(raw).hexdigest() != expected_program_sha256):
            raise ValueError("Program differs from its external hash")
        payload = json.loads(raw)
        source = payload["program"] if payload.get("version") == "warehouse_native_rcpd_feedback_v1" else payload
        self.program = ExecutableProgram.from_dict(source)
        if (tuple(self.program.feature_names) != tuple(runtime.actor.metadata["feature_names"])
                or len(self.program.feature_names) != 197
                or tuple(self.program.action_names) != ACTIONS
                or self.program.metadata.get("native_source_actor_sha256") != runtime.actor_sha256
                or self.program.metadata.get("action_legality_features")
                or self.program.metadata.get("action_constraint_reason_features")):
            raise ValueError("Program source, observed197 schema or unmasked action contract differs")
        stack = [self.program.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                p = np.asarray(node.probabilities)
                if (p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any()
                        or not np.isclose(p.sum(), 1)):
                    raise ValueError("Invalid program leaf")
            else:
                if (node.feature not in self.program.feature_names or node.threshold is None
                        or not np.isfinite(node.threshold) or node.left is None or node.right is None):
                    raise ValueError("Invalid program predicate")
                stack.extend((node.left, node.right))
        self.representation = _representation(self.program, runtime)
        self.actor_sha256 = runtime.actor_sha256
        self.runtime_signature = runtime.signature
        self.runtime_family = identity["family"]
        self.runtime_identity = deepcopy(identity)
        self.program_sha256 = expected_program_sha256
        self.program_content_sha256 = digest(self.program.to_dict())
        self.sources = explanation_sources()
        self.test_fixture = allow_test_fixture
        self.eligible = False
        self.participant_enabled = False
        self.fixture_enabled = allow_test_fixture
        self.signature = self._signature()
        self._contract_report = {"version": VERSION, "signature": self.signature,
            "runtime_family": self.runtime_family, "runtime_signature": self.runtime_signature,
            "actor_sha256": self.actor_sha256, "program_sha256": self.program_sha256,
            "representation": deepcopy(self.representation),
            "source_sha256": digest(self.sources), "test_fixture": allow_test_fixture,
            "eligibility_evaluated": False, "explanation_eligible": False,
            "participant_enabled": False, "release_ready": False,
            "scope": "component_evidence_rendering_only_requires_separate_qualified_release"}
        self._assert_current(runtime)

    def _signature(self):
        return digest({"version": VERSION, "runtime_identity": self.runtime_identity,
            "program": self.program_sha256, "program_content": self.program_content_sha256,
            "representation": self.representation,
            "sources": self.sources})

    @property
    def contract_report(self):
        return deepcopy(self._contract_report)

    def _assert_current(self, runtime):
        identity = registry.verify(runtime, allow_test_fixture=self.test_fixture,
            expected_family=self.runtime_family, expected_signature=self.runtime_signature)
        if (identity != self.runtime_identity or runtime.actor_sha256 != self.actor_sha256
                or explanation_sources() != self.sources or self._signature() != self.signature):
            raise ValueError("explanation_runtime_version_mismatch")
        if (sha256(self.program_path.read_bytes()).hexdigest() != self.program_sha256
                or digest(self.program.to_dict()) != self.program_content_sha256):
            raise ValueError("explanation_program_changed")
        if self.eligible or self.participant_enabled or self.fixture_enabled is not self.test_fixture:
            raise ValueError("This component cannot grant participant qualification")
