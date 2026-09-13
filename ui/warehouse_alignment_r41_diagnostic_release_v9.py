"""Fail-closed release seam for the r4.1 diagnostic v9 study.

The participant protocol and readable v9 explanation renderer are wired in
source before the separately audited Actor, program, scene, tutorial, and
question-bank artifacts exist.  This module deliberately cannot load a study
package yet.  A later admission change must replace the unbound loader and
bind the exact artifact hashes before this release can start.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r41-diagnostic-online-release.v9"
STATUS = "r41_diagnostic_v9_wiring_ready_artifacts_unbound"
PUBLIC_RELEASE_VERSION = "r4.1-diagnostic"
PILOT_CLASS = "internal_diagnostic"
ANIMATION_DURATION_MS = 380
UNBOUND_REASON = "r41_diagnostic_v9_release_artifacts_not_bound"

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_WIRING_PATHS = (
    "ui/warehouse_alignment_r41_diagnostic_release_v9.py",
    "ui/warehouse_alignment_online_server.py",
    "backend/warehouse_r41_diagnostic_online_explanation_v9.py",
    "ui/warehouse_family_feedback_research/index.html",
    "ui/warehouse_family_feedback_research/app.js",
    "ui/warehouse_family_feedback_research/styles.css",
    "ui/warehouse_family_feedback_research/favicon.svg",
)


class V9ReleaseArtifactsNotBound(RuntimeError):
    """Raised until a real v9 admission and artifact loader are installed."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return sha256(canonical(value).encode()).hexdigest()


def release_sources() -> dict[str, str]:
    """Hash the minimal participant-facing v9 wiring source closure."""

    result: dict[str, str] = {}
    for name in _WIRING_PATHS:
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("v9 release wiring source is missing: " + name)
        result[name] = sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def release_projection() -> dict[str, Any]:
    """Describe the wired boundary without claiming artifact readiness."""

    return {
        "status": STATUS,
        "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": PILOT_CLASS,
        "model_ready": False,
        "explanation_ready": False,
        "study_ready": False,
        "formal_ready": False,
        "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "data_persistent": False,
        "runtime_action_override": False,
        "artifact_binding_required": True,
        "animation_duration_ms": ANIMATION_DURATION_MS,
        "wiring_sources_sha256": digest(release_sources()),
    }


def validate_bound_context(
    context: Any, *, expected_package_sha256: str,
    expected_manifest_sha256: str,
) -> Any:
    """Validate a future admitted context before the generic server sees it.

    This validation never creates readiness.  The future package loader must
    first construct and authenticate the context from independently admitted
    artifacts, then pass it through this boundary.
    """

    if (_HEX.fullmatch(str(expected_package_sha256)) is None
            or _HEX.fullmatch(str(expected_manifest_sha256)) is None):
        raise ValueError("exact v9 package and manifest SHA-256 are required")
    release = getattr(context, "release", None)
    provenance = getattr(context, "provenance", None)
    explainer = getattr(context, "explainer", None)
    if (not isinstance(release, Mapping)
            or not isinstance(provenance, Mapping)
            or provenance.get("version") != VERSION
            or provenance.get("package_sha256") != expected_package_sha256
            or provenance.get("manifest_sha256") != expected_manifest_sha256
            or release.get("release_version") != PUBLIC_RELEASE_VERSION
            or release.get("pilot_class") != PILOT_CLASS
            or any(release.get(key) is not True
                   for key in ("model_ready", "explanation_ready", "study_ready"))
            or release.get("formal_ready") is not False
            or release.get("formal_sample_eligible") is not False
            or release.get("human_explanation_effect_validated") is not False
            or release.get("behavior_performance_gate_passed") is not False
            or release.get("behavior_performance_gate_waived") is not True
            or release.get("data_persistent") is not False
            or release.get("runtime_action_override") is not False
            or type(explainer).__module__
                != "backend.warehouse_r41_diagnostic_online_explanation_v9"
            or type(explainer).__name__
                != "R41DiagnosticOnlineAlignmentExplainerV9"
            or not callable(getattr(explainer, "answer_study", None))
            or not isinstance(getattr(explainer, "artifact_binding", None), Mapping)
            or not callable(getattr(context, "close", None))):
        raise ValueError("admitted v9 online context differs from its release boundary")
    return context


def inspect_online_release(*args: Any, **kwargs: Any) -> None:
    """Refuse inspection until the separately admitted v9 archive exists."""

    del args, kwargs
    raise V9ReleaseArtifactsNotBound(UNBOUND_REASON)


def load_online_release(*, expected_package_sha256: str,
                        expected_manifest_sha256: str,
                        package_path: str | Path | None = None,
                        base64_path: str | Path | None = None) -> None:
    """Fail closed instead of treating source wiring as a deployable release."""

    if (_HEX.fullmatch(str(expected_package_sha256)) is None
            or _HEX.fullmatch(str(expected_manifest_sha256)) is None):
        raise ValueError("exact v9 package and manifest SHA-256 are required")
    if bool(package_path) == bool(base64_path):
        raise ValueError("provide exactly one v9 package source")
    # Do not open the caller-supplied path here.  The later, hash-bound loader
    # will own archive authentication after the real admission is produced.
    raise V9ReleaseArtifactsNotBound(UNBOUND_REASON)


__all__ = [
    "VERSION", "STATUS", "PUBLIC_RELEASE_VERSION", "PILOT_CLASS",
    "ANIMATION_DURATION_MS", "UNBOUND_REASON",
    "V9ReleaseArtifactsNotBound", "release_sources", "release_projection",
    "validate_bound_context", "inspect_online_release", "load_online_release",
]
