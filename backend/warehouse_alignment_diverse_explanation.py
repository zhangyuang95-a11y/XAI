"""Qualified diverse-tree evidence answers for the final alignment runtime.

Question parsing, historical verification, actual-neural decisions, isolated
physical counterfactuals, and disagreement wording are inherited unchanged
from :class:`PublicHistoryExplainer`.  This module only admits the independently
accepted diverse program schema and its larger declared capacity.  Acceptance
of explanation evidence does not enable a study or participants here.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np

from backend import warehouse_alignment_runtime as alignment
from backend import warehouse_public_history_explanation as evidence
from backend.warehouse_family_explanation import actor_parameter_sha256
from backend.training import warehouse_family_alignment_diverse_collection as collector
from backend.training import warehouse_family_alignment_diverse_teacher_fit as fitter
from backend.training import warehouse_family_explanation_acceptance as system_acceptance
from backend.training import warehouse_family_explanation_system_run as system_run
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from core.program import ExecutableProgram
from env.warehouse_native.policy import ACTIONS


VERSION = "warehouse-alignment-diverse-qualified-evidence-answers.v1"
MAXIMUM_DEPTH = 20
MAXIMUM_LEAVES = 4096


def explanation_sources():
    sources = {}
    for records in (
        alignment.runtime_sources(),
        evidence.explanation_sources(),
        system_run.execution_sources(),
    ):
        for name, value in records.items():
            if name in sources and sources[name] != value:
                raise ValueError("Diverse explanation source closures disagree")
            sources[name] = value
    for path in (
        Path(__file__),
        ROOT / "backend/warehouse_family_explanation.py",
        ROOT / "core/program.py",
    ):
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    return sources


def _identity(runtime, *, allow_test_fixture, expected_signature=None):
    identity = alignment.verify(
        runtime,
        allow_test_fixture=allow_test_fixture,
        expected_family=alignment.FAMILY,
        expected_signature=expected_signature,
    )
    scope = identity.pop("verification_scope")
    if scope not in ("full_content_hashes", "context_object_checks"):
        raise ValueError("Unknown alignment runtime verification scope")
    return identity


def _sha(value, name):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid diverse explanation SHA256: " + name)
    return value


def _program_configuration(meta):
    # The executable program stores the four candidate coordinates directly.
    # ``config`` exists in fit audit records, not in the frozen program.  A
    # later qualification wrapper must preserve this actual schema rather than
    # manufacturing an audit-only config object.
    if "config" in meta:
        raise ValueError("Qualified program must preserve the frozen executable schema")
    coordinates = (
        meta.get("candidate_depth"),
        meta.get("candidate_max_leaf_nodes"),
        meta.get("counterfactual_changed_pair_weight"),
        meta.get("action_structure_weight"),
    )
    matches = [candidate for candidate in fitter.CANDIDATES
               if digest(list(coordinates)) == digest(list(candidate))]
    if len(matches) != 1:
        raise ValueError("Qualified program is outside the frozen diverse fit search")
    frozen = fitter.contract()
    if (frozen.get("candidates") != [list(value) for value in fitter.CANDIDATES]
            or frozen.get("candidate_fields") != [
                "max_depth", "max_leaf_nodes",
                "counterfactual_changed_pair_weight", "action_structure_weight",
            ]
            or frozen.get("backend") != fitter.BACKEND_VERSION
            or frozen.get("min_samples_leaf") != 8
            or frozen.get("regularization_lambda") != 0.0
            or frozen.get("importance_weight_scale") != 8.0
            or frozen.get("counterfactual_loss_weight") != 0.2):
        raise ValueError("Frozen diverse fit configuration changed")
    return matches[0]


def _representation(program, runtime, allow_test_fixture):
    meta = program.metadata
    candidate = _program_configuration(meta)
    if (meta.get("native_feedback_version") != fitter.VERSION
            or meta.get("prediction_semantics") != fitter.pure.prediction.VERSION
            or meta.get("runtime_controller") != "native_neural_actor_only"
            or meta.get("role_scope") != list(fitter.original.ROLES)
            or meta.get("fit_pool_only") is not True
            or meta.get("test_fixture") is not allow_test_fixture
            or meta.get("distillation_method") != "regularity_constrained_policy_distillation"
            or meta.get("distillation_backend") != fitter.BACKEND_VERSION
            or meta.get("sklearn_fit_count") != 1
            or program.root.depth() > candidate[0]
            or program.root.leaf_count() > candidate[1]
            or program.root.depth() > MAXIMUM_DEPTH
            or program.root.leaf_count() > MAXIMUM_LEAVES):
        raise ValueError("Qualified diverse schema, role scope, or 20/4096 capacity differs")

    bindings = meta.get("source_actor_bindings")
    binding_keys = {
        "actor_sha256", "actor_parameters_sha256", "protocol_sha256",
        "source_sha256", "runtime_signature",
    }
    if type(bindings) is not dict or set(bindings) != binding_keys:
        raise ValueError("Complete final alignment Actor bindings are required")
    for name in binding_keys:
        _sha(bindings[name], name)
    metadata = runtime.actor.metadata
    inherited = metadata.get("source_counters", {}).get("joint_steps")
    additional = metadata.get("joint_steps")
    parameter_sha = actor_parameter_sha256(runtime.actor)
    if (type(inherited) is not int or type(additional) is not int
            or min(inherited, additional) < 0
            or bindings["actor_sha256"] != runtime.actor_sha256
            or bindings["actor_parameters_sha256"] != parameter_sha
            or bindings["protocol_sha256"] != runtime.protocol_sha256
            or bindings["source_sha256"] != metadata.get("source_sha256")
            or bindings["runtime_signature"] != runtime.signature
            or meta.get("native_source_actor_sha256") != runtime.actor_sha256
            or ("actor_parameters_sha256" in metadata
                and metadata["actor_parameters_sha256"] != parameter_sha)
            or (not allow_test_fixture
                and (runtime.actor_sha256 != collector.PRODUCTION_ACTOR_SHA256
                     or inherited + additional != collector.PRODUCTION_ACTOR_TRAINING_CLOCK))):
        raise ValueError("Qualified program differs from the final frozen alignment Actor")

    for name in ("fit_data_sha256", "source_plan_file_sha256", "source_manifest_file_sha256"):
        _sha(meta.get(name), name)
    metrics = meta.get("metrics")
    qualified = meta.get("explanation_system_acceptance")
    fields = {
        "contract_version", "producer_version", "actor_sha256",
        "input_program_sha256", "ordinary_tree_source_program_sha256",
        "case_plan_sha256", "evidence_file_sha256", "report_file_sha256",
        "sources_sha256", "scope", "free_question_answer_qualified",
        "release_ready",
    }
    if type(qualified) is not dict or set(qualified) != fields:
        raise ValueError("Complete revised explanation-system acceptance is required")
    for name in (
        "input_program_sha256", "ordinary_tree_source_program_sha256",
        "case_plan_sha256", "evidence_file_sha256", "report_file_sha256",
        "sources_sha256",
    ):
        _sha(qualified.get(name), "system " + name)
    system_sources_sha256 = digest(system_run.execution_sources())
    if (meta.get("explanation_system_qualified") is not True
            or qualified.get("contract_version") != system_acceptance.VERSION
            or qualified.get("producer_version") != system_run.VERSION
            or qualified.get("actor_sha256") != runtime.actor_sha256
            or qualified.get("input_program_sha256")
                != qualified.get("ordinary_tree_source_program_sha256")
            or qualified.get("sources_sha256") != system_sources_sha256
            or qualified.get("scope") != "ordinary_tree_and_isolated_nn_engine"
            or qualified.get("free_question_answer_qualified") is not False
            or qualified.get("release_ready") is not False
            # Preserve the input program's closed old-route flags.  The new
            # wrapper records a distinct revised qualification instead of
            # rewriting a failed strict-tree result into an old-style pass.
            or meta.get("explanation_eligible") is not False
            or meta.get("explanation_qualified") is not False
            or meta.get("release_ready") is not False
            or not isinstance(metrics, dict)
            or metrics.get("explanation_eligible") is not False):
        raise ValueError("Revised explanation-system qualification metadata differs")
    return {
        "schema": fitter.VERSION,
        "acceptance_version": system_acceptance.VERSION,
        "acceptance_producer_version": system_run.VERSION,
        "prediction_semantics": fitter.pure.prediction.VERSION,
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_leaves": MAXIMUM_LEAVES,
        "candidate": list(candidate),
        "role_scope": list(fitter.original.ROLES),
        "source_actor_bindings": deepcopy(bindings),
        "actor_parameters_sha256": parameter_sha,
        "acceptance_case_plan_sha256": qualified["case_plan_sha256"],
        "acceptance_evidence_file_sha256": qualified["evidence_file_sha256"],
        "acceptance_report_file_sha256": qualified["report_file_sha256"],
        "acceptance_source_sha256": system_sources_sha256,
        "counterfactual_evidence_source": "isolated_nn_engine",
        "tree_pair_diagnostic_is_gate": False,
        "explanation_qualified": True,
        "study_ready": False,
    }


def _validate_tree(program):
    names = program.feature_names
    stack = [program.root]
    nodes = 0
    while stack:
        node = stack.pop()
        nodes += 1
        if nodes > 2 * MAXIMUM_LEAVES - 1:
            raise ValueError("Qualified diverse tree exceeds its complete node capacity")
        if node.is_leaf:
            probabilities = np.asarray(node.probabilities)
            if (probabilities.shape != (5,) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)):
                raise ValueError("Invalid qualified diverse program leaf")
        else:
            if (node.feature not in names or node.threshold is None
                    or not np.isfinite(node.threshold) or node.left is None or node.right is None):
                raise ValueError("Invalid qualified diverse program predicate")
            stack.extend((node.left, node.right))


class DiverseAlignmentExplainer(evidence.PublicHistoryExplainer):
    """Qualified evidence renderer that remains unable to activate a study."""

    def __init__(self, program_path, *, expected_program_sha256, runtime,
                 allow_test_fixture=False):
        if type(allow_test_fixture) is not bool:
            raise ValueError("Explicit boolean fixture scope required")
        identity = _identity(runtime, allow_test_fixture=allow_test_fixture)
        self.program_path = Path(program_path).expanduser().resolve()
        raw = self.program_path.read_bytes()
        if (type(expected_program_sha256) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", expected_program_sha256)
                or sha256(raw).hexdigest() != expected_program_sha256):
            raise ValueError("Program differs from its external hash")
        payload = json.loads(raw)
        if type(payload) is not dict or payload.get("format") != "rcpd_executable_program_v2":
            raise ValueError("Standalone qualified diverse executable program required")
        self.program = ExecutableProgram.from_dict(payload)
        if digest(payload) != digest(self.program.to_dict()):
            raise ValueError("Program does not preserve its canonical executable schema")
        names = self.program.feature_names
        if (len(names) != 197 or len(set(names)) != 197
                or any(type(name) is not str or not name for name in names)
                or tuple(names) != tuple(runtime.actor.metadata["feature_names"])
                or tuple(self.program.action_names) != ACTIONS
                or self.program.metadata.get("action_legality_features")
                or self.program.metadata.get("action_constraint_reason_features")):
            raise ValueError("Program source, ordered197 features, or unmasked action schema differs")
        _validate_tree(self.program)
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
        self.study_ready = False
        self.explanation_qualified = True
        self.release_ready = False
        self.fixture_enabled = allow_test_fixture
        self.signature = self._signature()
        self._contract_report = {
            "version": VERSION,
            "signature": self.signature,
            "runtime_family": self.runtime_family,
            "runtime_signature": self.runtime_signature,
            "actor_sha256": self.actor_sha256,
            "program_sha256": self.program_sha256,
            "program_content_sha256": self.program_content_sha256,
            "representation": deepcopy(self.representation),
            "source_sha256": digest(self.sources),
            "test_fixture": allow_test_fixture,
            "eligibility_evaluated": True,
            "explanation_eligible": True,
            "explanation_qualified": True,
            "study_ready": False,
            "participant_enabled": False,
            "release_ready": False,
            "scope": "qualified_evidence_component_requires_separate_web_release_integration",
        }
        self._assert_current(runtime)

    def _signature(self):
        return digest({
            "version": VERSION,
            "runtime_identity": self.runtime_identity,
            "program": self.program_sha256,
            "program_content": self.program_content_sha256,
            "representation": self.representation,
            "sources": self.sources,
        })

    @property
    def contract_report(self):
        return deepcopy(self._contract_report)

    def _assert_current(self, runtime):
        identity = _identity(
            runtime,
            allow_test_fixture=self.test_fixture,
            expected_signature=self.runtime_signature,
        )
        if (identity != self.runtime_identity or runtime.actor_sha256 != self.actor_sha256
                or explanation_sources() != self.sources or self._signature() != self.signature):
            raise ValueError("explanation_runtime_version_mismatch")
        if (sha256(self.program_path.read_bytes()).hexdigest() != self.program_sha256
                or digest(self.program.to_dict()) != self.program_content_sha256):
            raise ValueError("explanation_program_changed")
        if (self.eligible is not False or self.participant_enabled is not False
                or self.study_ready is not False or self.explanation_qualified is not True
                or self.release_ready is not False
                or self.fixture_enabled is not self.test_fixture):
            raise ValueError("This evidence component cannot activate a study")


# Alternate natural ordering retained for callers that spell the family first.
AlignmentDiverseExplainer = DiverseAlignmentExplainer
