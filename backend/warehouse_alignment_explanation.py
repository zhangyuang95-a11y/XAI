"""Component-only evidence answers for the genuine final alignment NN/tree.

The existing public-history answer algorithm, including isolated historical
replay and physical counterfactuals, is inherited without modification. This
module admits the actual branch-teacher schema; it grants no qualification.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np

from backend import warehouse_alignment_runtime as alignment
from backend import warehouse_public_history_explanation as evidence
from backend.warehouse_family_explanation import actor_parameter_sha256
from backend.training import warehouse_family_branch_teacher_fit as teacher
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from core.program import ExecutableProgram
from env.warehouse_native.policy import ACTIONS

VERSION = "warehouse-alignment-evidence-answers.v1"
PRODUCTION_CUMULATIVE_FIT_STEP = 3950000


def explanation_sources():
    sources = {}
    for records in (alignment.runtime_sources(), evidence.explanation_sources(),
                    teacher.execution_sources()):
        for name, value in records.items():
            if name in sources and sources[name] != value:
                raise ValueError("Alignment explanation source closures disagree")
            sources[name] = value
    for path in (Path(__file__), ROOT / "backend/warehouse_family_explanation.py",
                 ROOT / "backend/training/warehouse_native_public_feedback_initialization.py"):
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    return sources


def _identity(runtime, *, allow_test_fixture, expected_signature=None):
    identity = alignment.verify(runtime, allow_test_fixture=allow_test_fixture,
        expected_family=alignment.FAMILY, expected_signature=expected_signature)
    # A verified context has full hashes at entry/exit and object checks within
    # it. That transient verification mode does not change the frozen identity.
    scope = identity.pop("verification_scope")
    if scope not in ("full_content_hashes", "context_object_checks"):
        raise ValueError("Unknown alignment runtime verification scope")
    return identity


def _representation(program, runtime, allow_test_fixture):
    meta = program.metadata
    if (meta.get("native_feedback_version") != teacher.VERSION
            or meta.get("prediction_semantics") != teacher.pure.prediction.VERSION
            or digest(meta.get("native_feedback_config")) != digest(asdict(teacher.feedback_config()))
            or not any(digest(meta.get("config")) == digest(asdict(teacher.config(candidate)))
                       for candidate in teacher.CANDIDATES)
            or program.root.depth() > 12 or program.root.leaf_count() > 256
            or meta.get("role_scope") != list(teacher.ROLES)
            or meta.get("reliability_scope") != "ordinary_executed_nn_trajectories"
            or meta.get("runtime_controller") != "native_neural_actor_only"
            or meta.get("test_fixture") is not allow_test_fixture):
        raise ValueError("Actual branch-teacher schema, predicates, configuration or scope differs")
    binding = meta.get("observed197_bindings")
    keys = {"config_sha256", "training_data_sha256", "selection_data_sha256",
            "actor_sha256", "actor_parameters_sha256", "cumulative_fit_step"}
    if type(binding) is not dict or set(binding) != keys:
        raise ValueError("Complete original branch-teacher evidence binding required")
    for name in keys - {"cumulative_fit_step"}:
        if type(binding[name]) is not str or not re.fullmatch(r"[0-9a-f]{64}", binding[name]):
            raise ValueError("Invalid branch-teacher evidence hash")
    metadata = runtime.actor.metadata
    inherited = metadata["source_counters"]["joint_steps"]
    additional = metadata["joint_steps"]
    step = binding["cumulative_fit_step"]
    if (type(inherited) is not int or type(additional) is not int or min(inherited, additional) < 0
            or type(step) is not int or step != inherited + additional
            or (not allow_test_fixture and step != PRODUCTION_CUMULATIVE_FIT_STEP)
            or binding["actor_sha256"] != runtime.actor_sha256
            or binding["config_sha256"] != digest(teacher.contract())):
        raise ValueError("Branch teacher belongs to another Actor, fit clock or extraction contract")
    parameter_sha = actor_parameter_sha256(runtime.actor)
    if (binding["actor_parameters_sha256"] != parameter_sha
            or metadata.get("actor_parameters_sha256") != parameter_sha):
        raise ValueError("Branch teacher parameters differ from the actual frozen alignment NN")
    if meta.get("metrics", {}).get("explanation_eligible") is not False:
        raise ValueError("Branch-teacher metadata cannot grant explanation qualification")
    return {"schema": teacher.VERSION, "prediction_semantics": teacher.pure.prediction.VERSION,
        "maximum_depth": 12, "maximum_leaves": 256, "role_scope": list(teacher.ROLES),
        "observed197_bindings": deepcopy(binding), "actor_parameters_sha256": parameter_sha,
        "qualification_evaluated": False}


class AlignmentExplainer(evidence.PublicHistoryExplainer):
    """Same-frame NN evidence rendering, separate from a qualified study release."""

    def __init__(self, program_path, *, expected_program_sha256, runtime,
                 allow_test_fixture=False):
        identity = _identity(runtime, allow_test_fixture=allow_test_fixture)
        self.program_path = Path(program_path).expanduser().resolve()
        raw = self.program_path.read_bytes()
        if (type(expected_program_sha256) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", expected_program_sha256)
                or sha256(raw).hexdigest() != expected_program_sha256):
            raise ValueError("Program differs from its external hash")
        payload = json.loads(raw)
        if type(payload) is not dict or payload.get("format") != "rcpd_executable_program_v2":
            raise ValueError("The original standalone branch-teacher executable program is required")
        self.program = ExecutableProgram.from_dict(payload)
        if digest(payload) != digest(self.program.to_dict()):
            raise ValueError("Program does not preserve its canonical executable schema")
        names = self.program.feature_names
        if (len(names) != 197 or len(set(names)) != 197
                or any(type(name) is not str or not name for name in names)
                or tuple(names) != tuple(runtime.actor.metadata["feature_names"])
                or tuple(self.program.action_names) != ACTIONS
                or self.program.metadata.get("native_source_actor_sha256") != runtime.actor_sha256
                or self.program.metadata.get("action_legality_features")
                or self.program.metadata.get("action_constraint_reason_features")):
            raise ValueError("Program source, public197 features or unmasked five-action contract differs")
        stack = [self.program.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                p = np.asarray(node.probabilities)
                if (p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any()
                        or not np.isclose(p.sum(), 1)):
                    raise ValueError("Invalid branch-teacher leaf")
            else:
                if (node.feature not in names or node.threshold is None
                        or not np.isfinite(node.threshold) or node.left is None or node.right is None):
                    raise ValueError("Invalid branch-teacher predicate")
                stack.extend((node.left, node.right))
        self.representation = _representation(self.program, runtime, allow_test_fixture)
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
        self.explanation_qualified = False
        self.release_ready = False
        self.fixture_enabled = allow_test_fixture
        self.signature = self._signature()
        self._contract_report = {"version": VERSION, "signature": self.signature,
            "runtime_family": self.runtime_family, "runtime_signature": self.runtime_signature,
            "actor_sha256": self.actor_sha256, "program_sha256": self.program_sha256,
            "program_content_sha256": self.program_content_sha256,
            "representation": deepcopy(self.representation), "source_sha256": digest(self.sources),
            "test_fixture": allow_test_fixture, "eligibility_evaluated": False,
            "explanation_eligible": False, "explanation_qualified": False,
            "participant_enabled": False, "release_ready": False,
            "scope": "component_evidence_rendering_only_requires_separate_qualified_release"}
        self._assert_current(runtime)

    def _signature(self):
        return digest({"version": VERSION, "runtime_identity": self.runtime_identity,
            "program": self.program_sha256, "program_content": self.program_content_sha256,
            "representation": self.representation, "sources": self.sources})

    @property
    def contract_report(self):
        return deepcopy(self._contract_report)

    def _assert_current(self, runtime):
        identity = _identity(runtime, allow_test_fixture=self.test_fixture,
            expected_signature=self.runtime_signature)
        if (identity != self.runtime_identity or runtime.actor_sha256 != self.actor_sha256
                or explanation_sources() != self.sources or self._signature() != self.signature):
            raise ValueError("explanation_runtime_version_mismatch")
        if (sha256(self.program_path.read_bytes()).hexdigest() != self.program_sha256
                or digest(self.program.to_dict()) != self.program_content_sha256):
            raise ValueError("explanation_program_changed")
        if (any(value is not False for value in (self.eligible, self.participant_enabled,
                self.explanation_qualified, self.release_ready))
                or self.fixture_enabled is not self.test_fixture):
            raise ValueError("This component cannot grant participant qualification")
