"""Diagnostic r4.1 runtime bound to the unmodified final neural Actor.

The r4.1 continuation Actor records the hash of its *training* protocol.  The
historical online runtime expected a different, serving-only protocol shape,
so treating the training protocol as a serving protocol made the final Actor
impossible to load.  This adapter keeps the original Actor bytes untouched and
authenticates both the training protocol and the complete conflict manifest as
external inputs.  Environment and action semantics remain those of the r4.1
runtime; the program never participates in action selection.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from backend.warehouse_alignment_online_runtime import (
    HISTORY_FEATURE_NAMES,
    OnlineNumPyActor,
    OnlinePublicFeedbackEnvironment,
    REWARD_REVISION,
    digest,
    file_hash,
)
from backend.warehouse_r41_online_runtime import (
    DEFAULT_REWARD_CONFIG,
    R41OnlineAlignmentRuntime,
    r41_runtime_sources,
)
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
    R41DiagnosticConflictMixin,
    reset_diagnostic_scenario,
)


VERSION = "warehouse-r41-diagnostic-online-runtime.v3"
FULL_SCENE_MANIFEST_VERSION = "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
PORTABLE_RUNTIME_MANIFEST_VERSION = "warehouse-r41-diagnostic-portable-runtime-manifest.v1"
CONFLICT_VALIDATION_VERSION = "warehouse-r41-diagnostic-conflict-scene-validation.v3"


def _regular(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().absolute()
    if value.is_symlink() or not value.is_file() or value.resolve() != value:
        raise ValueError(label + " must be a canonical regular file")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Cheap change detector after constructor-time cryptographic authentication."""
    stat = path.stat(follow_symlinks=False)
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
            int(stat.st_mtime_ns), int(stat.st_ctime_ns))


def diagnostic_runtime_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    result = r41_runtime_sources()
    # The strict conflict environment changes successor-task sampling and reset
    # validation, so its bytes are part of the serving runtime contract rather
    # than merely data referenced by the scene manifest.
    for path in (
        Path(__file__).resolve(),
        root / "env" / "warehouse_native" / "r41_diagnostic_conflict.py",
    ):
        result[str(path.relative_to(root))] = file_hash(path)
    return dict(sorted(result.items()))


class R41DiagnosticConflictWarehouseEnv(
    R41DiagnosticConflictMixin, OnlinePublicFeedbackEnvironment
):
    """Observed197 environment with strict diagnostic successor sampling."""

    def __init__(self, config=None, reward_config=None, collision_cost=.05,
                 mode="observed"):
        super().__init__(
            config,
            deepcopy(DEFAULT_REWARD_CONFIG if reward_config is None else reward_config),
            collision_cost,
            mode=mode,
        )
        self._initialize_r41_conflict()


class R41DiagnosticOnlineAlignmentRuntime(R41OnlineAlignmentRuntime):
    """Serve the exact final r4.1 Actor under explicit external bindings."""

    def __init__(
        self,
        actor_path: str | Path,
        *,
        training_protocol_path: str | Path,
        manifest_path: str | Path,
        expected_actor_sha256: str,
        expected_training_protocol_file_sha256: str,
        expected_training_protocol_content_sha256: str,
        expected_manifest_file_sha256: str,
        expected_manifest_content_sha256: str,
        expected_manifest_semantic_sha256: str,
        allow_test_fixture: bool = False,
    ) -> None:
        self._actor_path = _regular(actor_path, "Diagnostic Actor")
        self._training_protocol_path = _regular(
            training_protocol_path, "Diagnostic training protocol")
        self._manifest_path = _regular(manifest_path, "Diagnostic conflict manifest")
        protocol = _read_object(self._training_protocol_path, "Training protocol")
        manifest = _read_object(self._manifest_path, "Conflict manifest")
        manifest_version = manifest.get("version")
        if manifest_version not in (
                FULL_SCENE_MANIFEST_VERSION, PORTABLE_RUNTIME_MANIFEST_VERSION):
            raise ValueError("Diagnostic runtime manifest version differs")
        source_manifest_version = (
            manifest_version if manifest_version == FULL_SCENE_MANIFEST_VERSION
            else manifest.get("source_full_manifest_version")
        )
        source_contract_version = manifest.get("diagnostic_contract_version")
        source_validation_version = (
            CONFLICT_VALIDATION_VERSION
            if manifest_version == FULL_SCENE_MANIFEST_VERSION
            else manifest.get("source_conflict_validation_version")
        )
        if (source_manifest_version != FULL_SCENE_MANIFEST_VERSION
                or source_contract_version != DIAGNOSTIC_CONTRACT_VERSION
                or source_validation_version != CONFLICT_VALIDATION_VERSION
                or (manifest_version == PORTABLE_RUNTIME_MANIFEST_VERSION
                    and (type(manifest.get("source_conflict_validation_sha256"))
                         is not str
                         or re.fullmatch(
                             r"[0-9a-f]{64}",
                             manifest["source_conflict_validation_sha256"])
                         is None))
                or manifest.get("diagnostic_contract_sha256")
                    != DIAGNOSTIC_CONTRACT_SHA256
                or manifest.get("diagnostic_conflict_graph_sha256")
                    != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
                or manifest.get("conflict_families_sha256")
                    != CONFLICT_FAMILIES_SHA256):
            raise ValueError("Diagnostic runtime source contract differs")
        content = deepcopy(manifest)
        claimed_content = content.pop("content_sha256", None)

        actual = {
            "actor_file_sha256": file_hash(self._actor_path),
            "training_protocol_file_sha256": file_hash(self._training_protocol_path),
            "training_protocol_content_sha256": digest(protocol),
            "manifest_file_sha256": file_hash(self._manifest_path),
            "manifest_content_sha256": claimed_content,
            "manifest_semantic_sha256": digest(manifest),
        }
        expected = {
            "actor_file_sha256": expected_actor_sha256,
            "training_protocol_file_sha256": expected_training_protocol_file_sha256,
            "training_protocol_content_sha256": expected_training_protocol_content_sha256,
            "manifest_file_sha256": expected_manifest_file_sha256,
            "manifest_content_sha256": expected_manifest_content_sha256,
            "manifest_semantic_sha256": expected_manifest_semantic_sha256,
        }
        if actual != expected or claimed_content != digest(content):
            raise ValueError("Diagnostic Actor/protocol/manifest external binding differs")
        source_full = {
            "manifest_version": source_manifest_version,
            "manifest_file_sha256": manifest.get(
                "source_full_manifest_file_sha256", actual["manifest_file_sha256"]),
            "manifest_content_sha256": manifest.get(
                "source_full_manifest_content_sha256", actual["manifest_content_sha256"]),
            "manifest_semantic_sha256": manifest.get(
                "source_full_manifest_semantic_sha256", actual["manifest_semantic_sha256"]),
            "diagnostic_contract_version": source_contract_version,
            "diagnostic_contract_sha256": manifest.get(
                "diagnostic_contract_sha256"),
            "diagnostic_conflict_graph_sha256": manifest.get(
                "diagnostic_conflict_graph_sha256"),
            "conflict_families_sha256": manifest.get("conflict_families_sha256"),
            "conflict_validation_version": source_validation_version,
        }
        hash_bindings = {
            key: value for key, value in source_full.items()
            if key.endswith("_sha256")
        }
        if any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in hash_bindings.values()):
            raise ValueError("Source full diagnostic manifest binding differs")

        self.actor = OnlineNumPyActor(self._actor_path)
        self.config = collaborative_study_config()
        metadata = self.actor.metadata
        metadata_contract = {
            "obs_dim": 197,
            "state_dim": 354,
            "hidden": 128,
            "feature_names": list(observation_names(self.config))
            + list(HISTORY_FEATURE_NAMES),
            "public_feedback_mode": "observed",
            "action_masks": False,
            "runtime_action_override": False,
            "protocol_sha256": expected_training_protocol_content_sha256,
            "conflict_manifest_semantic_sha256": protocol.get(
                "scenario_sampling", {}).get("manifest_semantic_sha256"),
        }
        if any(metadata.get(key) != value for key, value in metadata_contract.items()):
            raise ValueError("Final Actor metadata differs from diagnostic external binding")
        if (protocol.get("version") != "warehouse-r41-active-trainer.v1"
                or protocol.get("runtime_action_override") is not False
                or protocol.get("feedback", {}).get("runtime_action_override") is not False
                or source_contract_version != DIAGNOSTIC_CONTRACT_VERSION):
            raise ValueError("Diagnostic training/environment protocol contract differs")

        self.protocol = deepcopy(protocol)
        self.training_protocol_sha256 = expected_training_protocol_content_sha256
        # Historical methods write ``protocol_sha256`` into decision evidence;
        # here that value truthfully denotes the Actor-bound training protocol.
        self.protocol_sha256 = self.training_protocol_sha256
        self.actor_sha256 = expected_actor_sha256
        self.manifest_file_sha256 = expected_manifest_file_sha256
        self.manifest_content_sha256 = expected_manifest_content_sha256
        self.manifest_semantic_sha256 = expected_manifest_semantic_sha256
        self.test_fixture = bool(allow_test_fixture)
        self._reward_config = {
            **deepcopy(DEFAULT_REWARD_CONFIG),
            "revision": REWARD_REVISION,
            "collision_training_cost": 0.05,
        }
        self._sources = diagnostic_runtime_sources()
        self._metadata_sha256 = digest(metadata)
        self._weights_sha256 = self._weight_digest()
        self._external_bindings = deepcopy(actual)
        self._external_file_identities = {
            "actor": _file_identity(self._actor_path),
            "training_protocol": _file_identity(self._training_protocol_path),
            "manifest": _file_identity(self._manifest_path),
        }
        self.source_full_manifest_bindings = deepcopy(source_full)
        self.runtime_manifest_signature = digest({
            "version": VERSION,
            "runtime_manifest": actual,
            "source_full_manifest": source_full,
        })
        self.signature = digest({
            "version": VERSION,
            "source_full_manifest": source_full,
            "actor_metadata_sha256": self._metadata_sha256,
            "actor_weights_sha256": self._weights_sha256,
            "configuration": asdict(self.config),
            "reward_config": self._reward_config,
            "sources": self._sources,
        })
        self.contract_report = {
            "version": VERSION,
            "signature": self.signature,
            "actor_sha256": self.actor_sha256,
            "training_protocol_sha256": self.training_protocol_sha256,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "runtime_manifest_signature": self.runtime_manifest_signature,
            "source_full_manifest_bindings": deepcopy(source_full),
            "qualification_evaluated": False,
            "release_ready": False,
            "explanation_qualified": False,
            "test_fixture": self.test_fixture,
            "runtime_action_override": False,
        }
        self.verify_binding()

    def _new_environment(self):
        return R41DiagnosticConflictWarehouseEnv(
            self.config,
            deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"],
            mode="observed",
        )

    def _check_environment(self, env):
        if (type(env) is not R41DiagnosticConflictWarehouseEnv
                or env.config != self.config
                or env.reward_config != self._reward_config):
            raise ValueError("Diagnostic online environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        env = self._new_environment()
        reset_diagnostic_scenario(env, deepcopy(scenario))
        return env

    @property
    def external_bindings(self) -> dict[str, str]:
        return deepcopy(self._external_bindings)

    def verify_binding(self):
        identities = {
            "actor": _file_identity(self._actor_path),
            "training_protocol": _file_identity(self._training_protocol_path),
            "manifest": _file_identity(self._manifest_path),
        }
        if identities != self._external_file_identities:
            # A touched or replaced input is accepted only after full bytes and
            # canonical content are re-authenticated.  The common inference
            # path avoids reparsing a 78 MB full manifest at every frame.
            protocol = _read_object(self._training_protocol_path, "Training protocol")
            manifest = _read_object(self._manifest_path, "Conflict manifest")
            content = deepcopy(manifest)
            claimed = content.pop("content_sha256", None)
            current = {
                "actor_file_sha256": file_hash(self._actor_path),
                "training_protocol_file_sha256": file_hash(self._training_protocol_path),
                "training_protocol_content_sha256": digest(protocol),
                "manifest_file_sha256": file_hash(self._manifest_path),
                "manifest_content_sha256": claimed,
                "manifest_semantic_sha256": digest(manifest),
            }
            if current != self._external_bindings or claimed != digest(content):
                raise ValueError("Diagnostic runtime external binding changed")
            self._external_file_identities = identities
        if (digest(self.actor.metadata) != self._metadata_sha256
                or self._weight_digest() != self._weights_sha256
                or diagnostic_runtime_sources() != self._sources):
            raise ValueError("Diagnostic runtime binding changed")
        return self.signature


DiagnosticOnlineRuntime = R41DiagnosticOnlineAlignmentRuntime

__all__ = [
    "VERSION", "FULL_SCENE_MANIFEST_VERSION",
    "PORTABLE_RUNTIME_MANIFEST_VERSION", "CONFLICT_VALIDATION_VERSION",
    "R41DiagnosticConflictWarehouseEnv",
    "R41DiagnosticOnlineAlignmentRuntime",
    "DiagnosticOnlineRuntime",
    "diagnostic_runtime_sources",
]
