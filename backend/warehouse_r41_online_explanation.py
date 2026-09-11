"""r4.1-bound wrapper for the dependency-light warehouse explainer.

The historical explainer deliberately accepts the exact historical runtime
type.  This additive wrapper keeps that contract untouched while binding the
same verified answer renderer to :class:`R41OnlineAlignmentRuntime`.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from backend import warehouse_alignment_online_explanation as base
from backend.warehouse_alignment_online_runtime import ACTIONS, digest
from backend.warehouse_r41_online_runtime import R41OnlineAlignmentRuntime
from core.program import ExecutableProgram


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r41-alignment-online-readable-answers.v1"


def explanation_sources() -> dict[str, str]:
    paths = (
        Path(__file__),
        ROOT / "backend/warehouse_alignment_online_explanation.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "core/program.py",
    )
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("r4.1 explanation source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


class R41OnlineAlignmentExplainer(base.OnlineAlignmentExplainer):
    """The existing answer semantics with an exact r4.1 runtime binding."""

    def __init__(self, program_path, *, expected_program_sha256, runtime,
                 allow_test_fixture=False):
        if (type(runtime) is not R41OnlineAlignmentRuntime
                or type(allow_test_fixture) is not bool
                or runtime.test_fixture is not allow_test_fixture):
            raise ValueError("Explicit matching r4.1 observed197 runtime scope is required")
        runtime.verify_binding()
        self.program_path = Path(program_path).expanduser().resolve()
        raw = self.program_path.read_bytes()
        if (not isinstance(expected_program_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_program_sha256)
                or sha256(raw).hexdigest() != expected_program_sha256):
            raise ValueError("Program differs from its external hash")
        payload = json.loads(raw)
        source = payload["program"] if payload.get("version") \
            == "warehouse_native_rcpd_feedback_v1" else payload
        self.program = ExecutableProgram.from_dict(source)
        if (tuple(self.program.feature_names)
                != tuple(runtime.actor.metadata["feature_names"])
                or tuple(self.program.action_names) != ACTIONS
                or self.program.metadata.get("native_source_actor_sha256")
                    != runtime.actor_sha256
                or self.program.metadata.get("action_legality_features")
                or self.program.metadata.get("action_constraint_reason_features")
                or self.program.root.depth() > 12
                or self.program.root.leaf_count() > 256):
            raise ValueError("Program source, r4.1 schema or unmasked action contract differs")
        stack = [self.program.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                probabilities = np.asarray(node.probabilities)
                if (probabilities.shape != (5,) or not np.isfinite(probabilities).all()
                        or (probabilities < 0).any()
                        or not np.isclose(probabilities.sum(), 1)):
                    raise ValueError("Invalid program leaf")
            else:
                if (node.feature not in self.program.feature_names
                        or node.threshold is None or not np.isfinite(node.threshold)
                        or node.left is None or node.right is None):
                    raise ValueError("Invalid program predicate")
                stack.extend((node.left, node.right))
        self.actor_sha256 = runtime.actor_sha256
        self.runtime_signature = runtime.signature
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
        self.signature = digest({
            "version": VERSION,
            "runtime": self.runtime_signature,
            "program": self.program_sha256,
            "sources": self.sources,
        })
        self.contract_report = {
            "version": VERSION, "signature": self.signature,
            "test_fixture": allow_test_fixture, "eligibility_evaluated": True,
            "explanation_eligible": True, "explanation_qualified": True,
            "study_ready": False, "participant_enabled": False,
            "release_ready": False,
            "scope": "component_evidence_rendering_only_requires_separate_qualified_release",
        }

    def _assert_current(self, runtime):
        if (type(runtime) is not R41OnlineAlignmentRuntime
                or runtime.verify_binding() != self.runtime_signature
                or runtime.actor_sha256 != self.actor_sha256
                or runtime.test_fixture is not self.test_fixture
                or explanation_sources() != self.sources):
            raise ValueError("explanation_runtime_version_mismatch")
        if (sha256(self.program_path.read_bytes()).hexdigest()
                != self.program_sha256
                or digest(self.program.to_dict()) != self.program_content_sha256):
            raise ValueError("explanation_program_changed")
        if (self.eligible or self.participant_enabled or self.study_ready
                or not self.explanation_qualified or self.release_ready):
            raise ValueError("This component cannot grant participant qualification")


AlignmentExplainer = R41OnlineAlignmentExplainer

__all__ = ["VERSION", "R41OnlineAlignmentExplainer", "AlignmentExplainer",
           "explanation_sources"]
