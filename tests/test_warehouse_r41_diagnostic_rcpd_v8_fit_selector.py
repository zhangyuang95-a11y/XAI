from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v8 as v8
from backend.training import warehouse_r41_diagnostic_rcpd_v8_fit_selector as subject
from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as outer_api
from backend.training.warehouse_native_common import file_hash


FAMILIES = tuple(subject.INNER_HOLDOUT_FAMILY_QUOTAS)


def _fingerprint(number: int) -> str:
    return f"{number:064x}"


def _scope(
    *, eligible: list[str], exposed: list[str], families: dict[str, str],
    fresh: list[str] | None = None,
) -> dict:
    if fresh is None:
        fresh = [
            _fingerprint(100_000 + index)
            for index in range(outer_api.FRESH_OUTER_SCENE_COUNT)
        ]
    value = {
        "version": subject.SCOPE_VERSION,
        "source_report_sha256": "a" * 64,
        "source_rows_sha256": "b" * 64,
        "eligible_fit_scene_fingerprints": eligible,
        "inner_candidate_scenes": [
            {"fingerprint": scene, "family_id": families[scene]}
            for scene in eligible
        ],
        "previously_exposed_outer_scene_fingerprints": exposed,
        "fresh_outer_scene_fingerprints": fresh,
        "fresh_outer_registry_file_sha256": "c" * 64,
        "fresh_outer_registry_content_sha256": "d" * 64,
        "fresh_outer_report_file_sha256": "e" * 64,
        "fresh_outer_report_content_sha256": "f" * 64,
        "label_blind": True,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
    }
    value["content_sha256"] = subject.digest(value)
    return value


def _rows(specifications: list[tuple[str, bool, int]]) -> dict[str, np.ndarray]:
    count = len(specifications)
    observations = np.zeros((count, v8.BASE_FEATURE_COUNT), dtype=np.float32)
    probabilities = np.full((count, len(v8.ACTIONS)), 0.01, dtype=np.float32)
    for index, (_, _, action) in enumerate(specifications):
        probabilities[index, action] = 0.96
    observation_hashes = np.asarray(
        [_fingerprint(10_000 + index) for index in range(count)], dtype="S64")
    scenes = np.asarray([row[0] for row in specifications], dtype="S64")
    split = np.asarray([row[1] for row in specifications], dtype=np.bool_)
    return {
        "observations": observations,
        "probabilities": probabilities,
        "action_indices": np.asarray(
            [row[2] for row in specifications], dtype=np.uint8),
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": observation_hashes,
        "scene_fingerprints": scenes,
        "episode_ids": np.asarray(
            [f"episode-{index}:skilled" for index in range(count)], dtype="S180"),
        "frames": np.arange(count, dtype=np.int16),
        "group_bits": np.zeros(count, dtype=np.uint8),
        "kinds": np.full(count, "ordinary", dtype="S16"),
        "anchor_ids": np.full(count, "", dtype="S240"),
        "branch_actions": np.full(count, "", dtype="S8"),
        "physical_hashes": np.full(count, "", dtype="S64"),
        "source_state_hashes": np.asarray(
            [_fingerprint(20_000 + index) for index in range(count)], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.zeros(count, dtype=np.bool_),
        "split_validation": split,
    }


def _config() -> dict:
    model = {
        "learning_rate": 0.1,
        "max_iter": 70,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 10,
        "l2_regularization": 0.1,
        "max_depth": None,
        "max_bins": 255,
        "random_state": 941,
    }
    return {
        "version": v8.CONFIG_VERSION,
        "pair_pool_multiplier": 16.0,
        "use_action_factor": True,
        "models": {
            "base": deepcopy(model),
            "narrow_passage": {**model, "max_iter": 1, "max_leaf_nodes": 2,
                               "random_state": 947},
            "shared_pickup": {**model, "max_iter": 1, "max_leaf_nodes": 2,
                              "random_state": 953},
            "shared_charger": {**model, "max_iter": 1, "max_leaf_nodes": 2,
                               "random_state": 967},
        },
        "mix_weights": {group: 0.0 for group in v8.GROUPS},
    }


def _metrics(
    *, charger_direction: float, other: float = 0.92,
    narrow_direction: float | None = None,
) -> dict:
    if narrow_direction is None:
        narrow_direction = other
    return {
        "overall": {"rows": 100, "scenes": 12, "fidelity": other},
        "nonwait": {"rows": 80, "scenes": 12, "fidelity": other},
        "critical": {
            group: {"rows": 30, "scenes": 12, "fidelity": other}
            for group in v8.GROUPS
        },
        "effective_intervention_direction": {
            "pairs": 50,
            "scenes": 12,
            "fidelity": other,
            "by_group": {
                group: {
                    "pairs": 20,
                    "scenes": 12,
                    "fidelity": (
                        charger_direction if group == "shared_charger"
                        else narrow_direction if group == "narrow_passage"
                        else other
                    ),
                }
                for group in v8.GROUPS
            },
        },
        "mean_kl": 0.1,
    }


def _fresh_outer_pair(source_rows_sha256: str) -> tuple[dict, dict, str, str]:
    identities = []
    number = 200_000
    for family, quota in outer_api.FAMILY_QUOTAS.items():
        for offset in range(quota):
            identities.append({
                "batch_index": offset % 3,
                "family_id": family,
                "seed": number,
                "fingerprint": _fingerprint(number),
            })
            number += 1
    scenes = [
        {**identity, "split": "development_validation"}
        for identity in identities
    ]
    sources = outer_api.producer_sources()
    boundaries = {
        "outer_actor_rows_collected": False,
        "outer_candidate_scored": False,
        "actor_loaded_or_inferred": False,
        "observations_generated": False,
        "protected_final_access": False,
    }
    statistics = {
        "fresh_outer_scenes": outer_api.FRESH_OUTER_SCENE_COUNT,
    }
    registry = {
        "version": outer_api.VERSION,
        "status": outer_api.STATUS,
        "contract": outer_api.contract(),
        "bindings": {"source_rows_file_sha256": source_rows_sha256},
        "development_validation": scenes,
        "selected_outer_identities": identities,
        "statistics": statistics,
        "information_boundary": boundaries,
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "runtime_action_override": False,
        "producer_sources": sources,
        "producer_sources_sha256": subject.digest(sources),
        "formal_ready": False,
    }
    registry["content_sha256"] = subject.digest(registry)
    registry_file_sha256 = "7" * 64
    report = {
        "version": outer_api.REPORT_VERSION,
        "status": outer_api.STATUS,
        "registry_file_sha256": registry_file_sha256,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": deepcopy(registry["bindings"]),
        "selection": {
            "salt": outer_api.SELECTION_SALT,
            "family_quotas": deepcopy(outer_api.FAMILY_QUOTAS),
            "selected_identity_sha256": subject.digest(identities),
        },
        "statistics": deepcopy(statistics),
        "information_boundary": deepcopy(boundaries),
        "producer_sources": sources,
        "producer_sources_sha256": subject.digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = subject.digest(report)
    return registry, report, registry_file_sha256, "8" * 64


def _write_json(path, value):
    path.write_text(subject.canonical(value) + "\n", encoding="utf-8")


def _selector_evidence(tmp_path, monkeypatch):
    sources = {"selector.py": "1" * 64}
    monkeypatch.setattr(subject, "producer_sources", lambda: sources)
    evidence = tmp_path / "selector"
    evidence.mkdir()
    source_report_path = evidence / "source_v8_report.json"
    source_report_path.write_bytes(b"fixed-source-report\n")
    source_rows_path = evidence / "source_v8_rows.npz"
    source_rows_path.write_bytes(b"fixed-source-rows\n")
    source_report_sha = file_hash(source_report_path)
    source_rows_sha = file_hash(source_rows_path)
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_REPORT_SHA256", source_report_sha)
    monkeypatch.setattr(subject, "FROZEN_SOURCE_V8_ROWS_SHA256", source_rows_sha)
    monkeypatch.setattr(subject, "FROZEN_SOURCE_V8_CONFIG_FILE_SHA256", "2" * 64)
    monkeypatch.setattr(subject, "FROZEN_SOURCE_V8_PROGRAM_SHA256", "3" * 64)
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256", "4" * 64)
    monkeypatch.setattr(subject, "FROZEN_ACTOR_FILE_SHA256", "5" * 64)

    registry, outer_report, _, _ = _fresh_outer_pair(source_rows_sha)
    registry_path = tmp_path / "development_expansion.json"
    _write_json(registry_path, registry)
    registry_sha = file_hash(registry_path)
    outer_report["registry_file_sha256"] = registry_sha
    outer_report["content_sha256"] = subject.digest({
        key: value for key, value in outer_report.items()
        if key != "content_sha256"
    })
    outer_report_path = tmp_path / "development_expansion_report.json"
    _write_json(outer_report_path, outer_report)
    outer_report_sha = file_hash(outer_report_path)

    eligible = [_fingerprint(index) for index in range(1, 7)]
    families = {scene: FAMILIES[index] for index, scene in enumerate(eligible)}
    scope = _scope(eligible=eligible, exposed=[], families=families)
    scope.update({
        "source_report_sha256": source_report_sha,
        "source_rows_sha256": source_rows_sha,
        "fresh_outer_scene_fingerprints": sorted(
            row["fingerprint"] for row in registry["selected_outer_identities"]),
        "fresh_outer_registry_file_sha256": registry_sha,
        "fresh_outer_registry_content_sha256": registry["content_sha256"],
        "fresh_outer_report_file_sha256": outer_report_sha,
        "fresh_outer_report_content_sha256": outer_report["content_sha256"],
    })
    scope["content_sha256"] = subject.digest({
        key: value for key, value in scope.items() if key != "content_sha256"
    })
    scope_path = evidence / "fit_scope.json"
    _write_json(scope_path, scope)
    config = subject.candidate_configs(_config())[1]
    selected = {
        "version": subject.VERSION,
        "status": subject.STATUS_SELECTED,
        "selected_mix_weight": 0.25,
        "selected_config": config,
        "selected_config_sha256": subject.digest(config),
        "outer_evaluation_performed": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
    }
    selected_path = evidence / "selected_config.json"
    _write_json(selected_path, selected)
    for name in subject.EVIDENCE_ARTIFACT_NAMES:
        path = evidence / name
        if not path.exists():
            path.write_bytes((name + "\n").encode("ascii"))
    artifacts = {
        name: file_hash(evidence / name)
        for name in subject.EVIDENCE_ARTIFACT_NAMES
    }
    monkeypatch.setattr(
        subject, "FROZEN_FIT_SCOPE_SHA256", artifacts["fit_scope.json"])
    source_semantic = "0" * 64
    fit_semantic = "7" * 64
    monkeypatch.setattr(
        subject, "_authenticate_frozen_source_projection",
        lambda **kwargs: {
            "source_rows_semantic_sha256": source_semantic,
            "projected_rows_semantic_sha256": fit_semantic,
            "projection": {},
        })
    report = {
        "version": subject.VERSION,
        "status": subject.STATUS_SELECTED,
        "contract": subject.contract(),
        "development_diagnosis": deepcopy(subject.DEVELOPMENT_DIAGNOSIS),
        "bindings": {
            "source_v8_report_sha256": source_report_sha,
            "source_v8_rows_sha256": source_rows_sha,
            "source_v8_config_sha256": "2" * 64,
            "source_v8_program_sha256": "3" * 64,
            "source_v8_weights_audit_sha256": "4" * 64,
            "actor_file_sha256": "5" * 64,
            "fit_scope_file_sha256": artifacts["fit_scope.json"],
            "fit_scope_content_sha256": scope["content_sha256"],
            "fresh_outer_registry_file_sha256": registry_sha,
            "fresh_outer_registry_content_sha256": registry["content_sha256"],
            "fresh_outer_report_file_sha256": outer_report_sha,
            "fresh_outer_report_content_sha256": outer_report["content_sha256"],
            "selector_binding_sha256": "6" * 64,
            "producer_sources_sha256": subject.digest(sources),
            "source_v8_rows_semantic_sha256": source_semantic,
            "fit_only_rows_semantic_sha256": fit_semantic,
            "config_registry_content_sha256": "8" * 64,
            "inner_split_audit_content_sha256": "9" * 64,
            "inner_selection_content_sha256": "c" * 64,
            "selected_config_content_sha256": subject.digest(selected),
            "inner_fit_program_content_sha256": "d" * 64,
        },
        "projection": {},
        "selection": {
            "status": subject.STATUS_SELECTED,
            "selected_mix_weight": 0.25,
            "outer_evaluation_performed": False,
        },
        "selected_config_sha256": subject.digest(config),
        "outer_evaluation_performed": False,
        "outer_labels_used_for_projection_fit_or_selection": False,
        "outer_probabilities_used_for_projection_fit_or_selection": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
        "runtime_action_override": False,
        "actor_changed": False,
        "formal_ready": False,
        "sources": sources,
        "evidence_artifacts": artifacts,
    }
    report_path = evidence / "report.json"
    _write_json(report_path, report)
    return {
        "evidence": evidence, "report": report_path, "scope": scope_path,
        "selected": selected_path, "source_report": source_report_path,
        "source_rows": source_rows_path,
        "fit_only_rows": evidence / "fit_only_rows.npz",
        "registry": registry_path, "outer_report": outer_report_path,
        "config": config,
    }

def _strict_selector_evidence(tmp_path, monkeypatch):
    sources = {"selector.py": "1" * 64}
    monkeypatch.setattr(subject, "producer_sources", lambda: sources)

    eligible = []
    families = {}
    next_identity = 1
    for family, quota in subject.INNER_HOLDOUT_FAMILY_QUOTAS.items():
        for _ in range(quota + 1):
            scene = _fingerprint(next_identity)
            next_identity += 1
            eligible.append(scene)
            families[scene] = family
    monkeypatch.setattr(subject, "FROZEN_SOURCE_ROW_COUNT", len(eligible))
    monkeypatch.setattr(subject, "FROZEN_SOURCE_SCENE_COUNT", len(eligible))
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_ELIGIBLE_ROW_COUNT", len(eligible))

    actor_path = tmp_path / "actor.npz"
    actor_path.write_bytes(b"frozen-test-actor\n")
    actor_sha = file_hash(actor_path)
    actor_parameters_sha = "5" * 64
    monkeypatch.setattr(subject, "FROZEN_ACTOR_FILE_SHA256", actor_sha)
    monkeypatch.setattr(
        subject, "FROZEN_ACTOR_PARAMETERS_SHA256", actor_parameters_sha)
    monkeypatch.setattr(subject, "FROZEN_SOURCE_V8_CONFIG_FILE_SHA256", "2" * 64)
    monkeypatch.setattr(subject, "FROZEN_SOURCE_V8_PROGRAM_SHA256", "3" * 64)
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256", "4" * 64)
    source_identity = deepcopy(subject.FROZEN_SOURCE_IDENTITY)
    source_identity["native_source_actor_sha256"] = actor_sha
    source_identity["source_actor_parameters_sha256"] = actor_parameters_sha
    monkeypatch.setattr(subject, "FROZEN_SOURCE_IDENTITY", source_identity)

    evidence = tmp_path / "strict_selector"
    evidence.mkdir()
    source_rows = _rows([
        (scene, False, index % len(v8.ACTIONS))
        for index, scene in enumerate(eligible)
    ])
    source_rows_path = evidence / "source_v8_rows.npz"
    subject._write_npz(source_rows_path, source_rows)
    source_rows_sha = file_hash(source_rows_path)
    monkeypatch.setattr(subject, "FROZEN_SOURCE_V8_ROWS_SHA256", source_rows_sha)
    source_report = {
        "version": v8.VERSION,
        "status": v8.STATUS_FAILED,
        "evidence_artifacts": {
            "rows.npz": source_rows_sha,
            "fit_config.json": "2" * 64,
            "program.json": "3" * 64,
            "weights_audit.json": "4" * 64,
        },
        "bindings": {"actor_file_sha256": actor_sha},
    }
    source_report_path = evidence / "source_v8_report.json"
    _write_json(source_report_path, source_report)
    source_report_sha = file_hash(source_report_path)
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_REPORT_SHA256", source_report_sha)

    registry, outer_report, _, _ = _fresh_outer_pair(source_rows_sha)
    registry_path = tmp_path / "development_expansion.json"
    _write_json(registry_path, registry)
    registry_sha = file_hash(registry_path)
    outer_report["registry_file_sha256"] = registry_sha
    outer_report["content_sha256"] = subject.digest({
        key: value for key, value in outer_report.items()
        if key != "content_sha256"
    })
    outer_report_path = tmp_path / "development_expansion_report.json"
    _write_json(outer_report_path, outer_report)
    outer_report_sha = file_hash(outer_report_path)

    scope = _scope(eligible=eligible, exposed=[], families=families)
    scope.update({
        "source_report_sha256": source_report_sha,
        "source_rows_sha256": source_rows_sha,
        "fresh_outer_scene_fingerprints": sorted(
            row["fingerprint"] for row in registry["selected_outer_identities"]),
        "fresh_outer_registry_file_sha256": registry_sha,
        "fresh_outer_registry_content_sha256": registry["content_sha256"],
        "fresh_outer_report_file_sha256": outer_report_sha,
        "fresh_outer_report_content_sha256": outer_report["content_sha256"],
    })
    scope["content_sha256"] = subject.digest({
        key: value for key, value in scope.items() if key != "content_sha256"
    })
    scope_path = evidence / "fit_scope.json"
    _write_json(scope_path, scope)
    monkeypatch.setattr(subject, "FROZEN_FIT_SCOPE_SHA256", file_hash(scope_path))
    rows, source_projection = subject._project_fit_only(source_rows, scope)
    subject._write_npz(evidence / "fit_only_rows.npz", rows)

    configs = subject.candidate_configs(subject.FROZEN_SOURCE_CONFIG)
    config_registry = subject._config_registry(configs)
    _write_json(evidence / "config_registry.json", config_registry)

    class FakeActor:
        artifact_sha256 = actor_sha
        metadata = {
            "actor_parameters_sha256": actor_parameters_sha,
            "feature_names": [],
        }

    class FakeProgram:
        def __init__(self, payload):
            self.payload = deepcopy(payload)

        def to_dict(self):
            return deepcopy(self.payload)

    class FakeProgramReader:
        @classmethod
        def from_dict(cls, payload):
            return FakeProgram(payload)

    monkeypatch.setattr(subject, "NumPyNativeActor", lambda path: FakeActor())
    monkeypatch.setattr(subject.v8, "_validate_base_shapes", lambda *args: None)
    monkeypatch.setattr(
        subject.v8, "R41DiagnosticPublicTreeProgramV8", FakeProgramReader)

    source_rows_semantic = subject._arrays_digest(source_rows)
    rows_semantic = subject._arrays_digest(rows)
    selector_binding = subject._selector_binding(
        scope_file_sha256=file_hash(scope_path),
        fit_only_rows_semantic_sha256=rows_semantic,
        source_rows_semantic_sha256=source_rows_semantic,
        sources=sources, configs=configs)
    program_payload = {
        "kind": "deterministic-test-program",
        "selector_binding_sha256": selector_binding,
    }
    metrics = (
        _metrics(charger_direction=0.84),
        _metrics(charger_direction=0.851, other=0.919),
        _metrics(charger_direction=0.87, other=0.917),
        _metrics(charger_direction=0.88, other=0.919),
    )
    selection = subject.choose_candidate([
        {
            "mix_weight": mix,
            "config_sha256": subject.digest(config),
            "validation_probabilities_sha256": subject.digest({
                "mix": mix, "kind": "probabilities",
            }),
            "validation_predictions_sha256": subject.digest({
                "mix": mix, "kind": "predictions",
            }),
            "metrics": metric,
        }
        for mix, config, metric in zip(subject.MIX_CANDIDATES, configs, metrics)
    ])
    selection.update({
        "fit_diagnostics": {"deterministic": True},
        "weight_audit": {"deterministic": True},
        "fit_pair_group_bits_sha256": "6" * 64,
        "validation_pair_group_bits_sha256": "7" * 64,
    })

    def fake_select(arrays, **kwargs):
        assert subject._arrays_digest(arrays) == rows_semantic
        assert kwargs["source_config"] == subject.FROZEN_SOURCE_CONFIG
        assert kwargs["source_identity"] == source_identity
        assert kwargs["selector_binding"] == selector_binding
        return deepcopy(selection), FakeProgram(program_payload), deepcopy(configs)

    monkeypatch.setattr(subject, "_select_projected", fake_select)
    projection = subject._projected_fit_only_audit(
        rows, scope=scope, actor=FakeActor())
    assert projection == source_projection
    selected = subject._selected_config_record(selection, configs)
    _write_json(evidence / "inner_split_audit.json", projection)
    _write_json(evidence / "inner_selection.json", selection)
    _write_json(evidence / "selected_config.json", selected)
    _write_json(evidence / "inner_fit_program.json", program_payload)
    artifacts = {
        name: file_hash(evidence / name)
        for name in subject.EVIDENCE_ARTIFACT_NAMES
    }
    report = {
        "version": subject.VERSION,
        "status": subject.STATUS_SELECTED,
        "contract": subject.contract(),
        "development_diagnosis": deepcopy(subject.DEVELOPMENT_DIAGNOSIS),
        "bindings": {
            "source_v8_report_sha256": source_report_sha,
            "source_v8_rows_sha256": source_rows_sha,
            "source_v8_config_sha256": "2" * 64,
            "source_v8_program_sha256": "3" * 64,
            "source_v8_weights_audit_sha256": "4" * 64,
            "actor_file_sha256": actor_sha,
            "fit_scope_file_sha256": file_hash(scope_path),
            "fit_scope_content_sha256": scope["content_sha256"],
            "fresh_outer_registry_file_sha256": registry_sha,
            "fresh_outer_registry_content_sha256": registry["content_sha256"],
            "fresh_outer_report_file_sha256": outer_report_sha,
            "fresh_outer_report_content_sha256": outer_report["content_sha256"],
            "selector_binding_sha256": selector_binding,
            "producer_sources_sha256": subject.digest(sources),
            "source_v8_rows_semantic_sha256": source_rows_semantic,
            "fit_only_rows_semantic_sha256": rows_semantic,
            "config_registry_content_sha256": subject.digest(config_registry),
            "inner_split_audit_content_sha256": subject.digest(projection),
            "inner_selection_content_sha256": subject.digest(selection),
            "selected_config_content_sha256": subject.digest(selected),
            "inner_fit_program_content_sha256": subject.digest(program_payload),
        },
        "projection": projection,
        "selection": selection,
        "selected_config_sha256": selected["selected_config_sha256"],
        "outer_evaluation_performed": False,
        "outer_labels_used_for_projection_fit_or_selection": False,
        "outer_probabilities_used_for_projection_fit_or_selection": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
        "runtime_action_override": False,
        "actor_changed": False,
        "formal_ready": False,
        "sources": sources,
        "evidence_artifacts": artifacts,
    }
    report_path = evidence / "report.json"
    _write_json(report_path, report)
    return {
        "evidence": evidence,
        "report": report_path,
        "registry": registry_path,
        "outer_report": outer_report_path,
        "actor": actor_path,
        "source_identity": source_identity,
        "selection": selection,
        "program": program_payload,
        "source_rows": source_rows,
    }

def _refresh_strict_selector_report(paths):
    evidence = paths["evidence"]
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    artifacts = {
        name: file_hash(evidence / name)
        for name in subject.EVIDENCE_ARTIFACT_NAMES
    }
    report["evidence_artifacts"] = artifacts
    projection = json.loads(
        (evidence / "inner_split_audit.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (evidence / "inner_selection.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (evidence / "selected_config.json").read_text(encoding="utf-8"))
    program = json.loads(
        (evidence / "inner_fit_program.json").read_text(encoding="utf-8"))
    registry = json.loads(
        (evidence / "config_registry.json").read_text(encoding="utf-8"))
    report["projection"] = projection
    report["selection"] = selection
    report["selected_config_sha256"] = selected["selected_config_sha256"]
    report["bindings"].update({
        "config_registry_content_sha256": subject.digest(registry),
        "inner_split_audit_content_sha256": subject.digest(projection),
        "inner_selection_content_sha256": subject.digest(selection),
        "selected_config_content_sha256": subject.digest(selected),
        "inner_fit_program_content_sha256": subject.digest(program),
    })
    _write_json(paths["report"], report)


def _strict_authenticate(paths):
    return subject.authenticate_selected_config_snapshot(
        evidence_directory=paths["evidence"],
        expected_report_sha256=file_hash(paths["report"]),
        actor_path=paths["actor"],
        source_full_manifest_bindings=paths["source_identity"][
            "source_full_manifest_bindings"],
        fresh_outer_registry_path=paths["registry"],
        expected_fresh_outer_registry_sha256=file_hash(paths["registry"]),
        fresh_outer_report_path=paths["outer_report"],
        expected_fresh_outer_report_sha256=file_hash(paths["outer_report"]),
    )


def test_contract_freezes_fit_only_search_and_has_no_outer_or_final_metric_input():
    value = subject.contract()
    assert value["source"][
        "previously_exposed_outer_rows_removed_before_label_validation"] is True
    assert value["source"][
        "source_archive_values_decompressed_before_row_projection"] is True
    assert value["source"][
        "outer_labels_used_for_projection_fit_or_selection"] is False
    assert value["source"][
        "outer_probabilities_used_for_projection_fit_or_selection"] is False
    assert "identity-only registry/report" in value["source"][
        "fresh_outer_binding"]
    assert value["source"]["final_rows_accessed"] is False
    assert value["source"]["final_labels_accessed"] is False
    assert value["frozen_model_change"]["shared_charger"][
        "mix_candidates"] == [0.0, 0.25, 0.5, 1.0]
    assert value["frozen_model_change"]["narrow_passage"] == {
        "model": subject.NARROW_MODEL,
        "fit_population": v8.NARROW_PAIR_ENDPOINT_FIT,
        "mix_weight": 1.0,
    }
    assert value["frozen_model_change"]["base"] == {
        "model": subject.BASE_MODEL,
        "depth_bounded_from_frozen_source": True,
    }
    assert value["selection"]["primary"] == (
        "maximum minimum margin across all nine v8 gates")
    assert value["development_diagnosis_sha256"] == subject.digest(
        subject.DEVELOPMENT_DIAGNOSIS)
    assert value["inner_split"]["holdout_scenes"] == 64
    assert sum(value["inner_split"]["family_quotas"].values()) == 64


def test_candidate_registry_freezes_pair_narrow_capacity_and_component_mixes():
    source = _config()
    candidates = subject.candidate_configs(source)
    assert [row["mix_weights"]["shared_charger"] for row in candidates] == [
        0.0, 0.25, 0.5, 1.0]
    for row in candidates:
        assert row["models"]["base"] == subject.BASE_MODEL
        assert row["models"]["narrow_passage"] == subject.NARROW_MODEL
        assert row["models"]["shared_pickup"] == source["models"]["shared_pickup"]
        assert row["models"]["shared_charger"] == subject.CHARGER_MODEL
        assert row["mix_weights"]["narrow_passage"] == 1.0
        assert row["mix_weights"]["shared_pickup"] == 1.0
        assert row["pair_pool_multiplier"] == 16.0
        assert row["use_action_factor"] is True


def test_narrow_pair_fit_allocates_each_fit_occurrence_two_to_one():
    arrays = {
        "split_validation": np.zeros(5, dtype=np.bool_),
        "branch_actions": np.asarray(
            ["WAIT", "LEFT", "DOWN", "WAIT", "RIGHT"], dtype="S8"),
    }
    pairs = np.asarray([[0, 1], [0, 2], [3, 4]], dtype=np.int64)
    pair_bits = np.asarray([1, 3, 2], dtype=np.uint8)
    occurrences = [
        {"wait_row": 0, "branch_row": 1, "weighted_occurrence_mass": 9.0,
         "group_bits": 1},
        {"wait_row": 0, "branch_row": 2, "weighted_occurrence_mass": 3.0,
         "group_bits": 3},
        {"wait_row": 3, "branch_row": 4, "weighted_occurrence_mass": 12.0,
         "group_bits": 2},
    ]
    indices, weights, audit = v8._narrow_pair_endpoint_fit(
        arrays, {"pairs": pairs, "audit": {"pair_occurrences": occurrences}},
        pair_bits)
    assert indices.tolist() == [0, 1, 2]
    # Row zero is the shared WAIT endpoint: 9*(2/3) + 3*(2/3).
    assert weights.tolist() == pytest.approx([8.0, 3.0, 1.0])
    assert sum(weights) == pytest.approx(12.0)
    assert audit["fit_pair_occurrences"] == 2
    assert audit["fit_endpoint_rows"] == 3
    assert audit["fit_wait_endpoint_rows"] == 1
    assert audit["fit_changed_branch_endpoint_rows"] == 2
    assert audit["validation_labels_used"] is False
    assert audit["final_rows_accessed"] is False


def test_four_candidate_probability_matrices_normalize_unscored_actor_rows(
        monkeypatch):
    rows = _rows([
        (_fingerprint(index + 1), index >= 4, index % len(v8.ACTIONS))
        for index in range(8)
    ])
    validation = rows["split_validation"]
    # Frozen Actor rows are float32 and legitimately use the looser source-row
    # normalization tolerance.  Copying them into the float64 candidate matrix
    # was the production failure this test guards against.
    source_sums = rows["probabilities"].astype(np.float64).sum(axis=1)
    assert np.any(np.abs(source_sums - 1.0) > 2e-12)

    class FittedProgram:
        base_feature_names = ("test",)
        base_program = object()
        _specialist_programs = {
            "narrow_passage": object(),
            "shared_pickup": object(),
            "shared_charger": object(),
        }
        metadata = {"kind": "fit-only-test"}

    seen_mixes = []

    def fake_assemble(*args, **kwargs):
        mix = float(kwargs["mix_weights"]["shared_charger"])
        assert kwargs["mix_weights"]["narrow_passage"] == 1.0
        assert kwargs["mix_weights"]["shared_pickup"] == 1.0
        seen_mixes.append(mix)
        return mix

    def fake_predict(mix, observations):
        result = np.zeros(
            (len(observations), len(v8.ACTIONS)), dtype=np.float64)
        result[:, subject.MIX_CANDIDATES.index(mix)] = 1.0
        return result

    monkeypatch.setattr(
        subject, "assemble_public_tree_program_v8", fake_assemble)
    monkeypatch.setattr(subject.v8, "_predict_in_batches", fake_predict)
    pairs = subject.v7._effective_pairs(rows, validation)
    pair_bits = np.empty(len(pairs), dtype=np.uint8)

    for mix in subject.MIX_CANDIDATES:
        probabilities = subject._candidate_probabilities(
            FittedProgram(), rows["observations"], rows["probabilities"],
            validation, _config(), mix)
        assert probabilities.shape == rows["probabilities"].shape
        assert np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-12)
        assert np.array_equal(
            np.argmax(probabilities[~validation], axis=1),
            np.argmax(rows["probabilities"][~validation], axis=1))
        expected_validation = np.zeros(
            (int(np.sum(validation)), len(v8.ACTIONS)), dtype=np.float64)
        expected_validation[:, subject.MIX_CANDIDATES.index(mix)] = 1.0
        assert np.array_equal(
            probabilities[validation], expected_validation)
        metrics = v8._metrics_from_probabilities(
            probabilities, rows, validation,
            pairs=pairs, pair_group_bits=pair_bits)
        assert metrics["overall"]["rows"] == int(np.sum(validation))

    assert seen_mixes == list(subject.MIX_CANDIDATES)


def test_outer_label_and_probability_mutation_cannot_change_fit_only_projection():
    eligible = [_fingerprint(index) for index in range(1, 5)]
    exposed = [_fingerprint(99)]
    family_map = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=exposed, families=family_map)
    rows = _rows([
        (eligible[0], False, 0), (eligible[0], False, 1),
        (eligible[1], False, 2), (eligible[1], False, 3),
        (eligible[2], False, 0), (eligible[2], False, 1),
        (eligible[3], False, 2), (eligible[3], False, 3),
        (exposed[0], True, 4),
    ])
    changed = {name: value.copy() for name, value in rows.items()}
    changed["action_indices"][-1] = 0
    changed["probabilities"][-1] = [0.96, 0.01, 0.01, 0.01, 0.01]
    quotas = {FAMILIES[0]: 1, FAMILIES[1]: 1}
    first, first_audit = subject._project_fit_only(
        rows, scope, quotas=quotas, salt="unit-test-salt")
    second, second_audit = subject._project_fit_only(
        changed, scope, quotas=quotas, salt="unit-test-salt")
    assert subject._arrays_digest(first) == subject._arrays_digest(second)
    assert first_audit == second_audit
    assert exposed[0] not in set(map(str, v8._decode(
        first["scene_fingerprints"], "test scenes")))
    assert first_audit["outer_labels_or_probabilities_used_for_projection"] is False


def test_projection_is_whole_scene_and_validation_wins_exact_observation_overlap():
    eligible = [_fingerprint(index) for index in range(10, 14)]
    exposed = [_fingerprint(199)]
    family_map = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=exposed, families=family_map)
    rows = _rows([
        (scene, False, index % 5)
        for index, scene in enumerate(eligible)
        for _ in range(2)
    ] + [(exposed[0], True, 4)])
    quotas = {FAMILIES[0]: 1, FAMILIES[1]: 1}
    chosen, _ = subject._inner_holdout(
        scope["inner_candidate_scenes"], quotas=quotas, salt="overlap-salt")
    chosen_set = set(chosen)
    decoded = v8._decode(rows["scene_fingerprints"], "fixture scenes")
    validation_index = int(np.flatnonzero(np.isin(decoded, list(chosen_set)))[0])
    training_index = int(np.flatnonzero(~np.isin(decoded, list(chosen_set)))[0])
    rows["observation_hashes"][training_index] = rows["observation_hashes"][
        validation_index]
    projected, audit = subject._project_fit_only(
        rows, scope, quotas=quotas, salt="overlap-salt")
    projected_scenes = v8._decode(projected["scene_fingerprints"], "projected")
    assert set(projected_scenes[projected["split_validation"]]) == chosen_set
    assert not (set(map(bytes, projected["observation_hashes"][
        ~projected["split_validation"]])) & set(map(bytes, projected[
            "observation_hashes"][projected["split_validation"]])))
    assert audit["inner_fit_rows_removed_for_exact_holdout_overlap"] == 1
    assert audit["scene_identity_overlap"] == 0
    assert audit["episode_identity_overlap"] == 0
    assert audit["nonempty_anchor_identity_overlap"] == 0
    for scene in eligible:
        flags = projected["split_validation"][projected_scenes == scene]
        assert len(set(map(bool, flags))) == 1


def test_projection_rejects_episode_or_anchor_identity_crossing_inner_split():
    eligible = [_fingerprint(index) for index in range(20, 24)]
    exposed = [_fingerprint(219)]
    families = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=exposed, families=families)
    quotas = {FAMILIES[0]: 1, FAMILIES[1]: 1}
    rows = _rows([
        (eligible[0], False, 0), (eligible[1], False, 1),
        (eligible[2], False, 2), (eligible[3], False, 3),
        (exposed[0], True, 4),
    ])
    chosen, _ = subject._inner_holdout(
        scope["inner_candidate_scenes"], quotas=quotas, salt="identity-salt")
    decoded = v8._decode(rows["scene_fingerprints"], "identity fixture scenes")
    validation_index = int(np.flatnonzero(np.isin(decoded, chosen))[0])
    fit_index = int(np.flatnonzero(
        (~rows["split_validation"]) & ~np.isin(decoded, chosen))[0])

    crossed_episode = {name: value.copy() for name, value in rows.items()}
    crossed_episode["episode_ids"][fit_index] = crossed_episode["episode_ids"][
        validation_index]
    with pytest.raises(ValueError, match="projection isolation differs"):
        subject._project_fit_only(
            crossed_episode, scope, quotas=quotas, salt="identity-salt")

    crossed_anchor = {name: value.copy() for name, value in rows.items()}
    crossed_anchor["anchor_ids"][[fit_index, validation_index]] = "shared-anchor"
    with pytest.raises(ValueError, match="projection isolation differs"):
        subject._project_fit_only(
            crossed_anchor, scope, quotas=quotas, salt="identity-salt")


def test_fresh_outer_must_be_absent_from_source_rows():
    eligible = [_fingerprint(index) for index in range(30, 34)]
    exposed = [_fingerprint(299)]
    families = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    fresh = [
        _fingerprint(400 + index)
        for index in range(outer_api.FRESH_OUTER_SCENE_COUNT)
    ]
    scope = _scope(
        eligible=eligible, exposed=exposed, families=families,
        fresh=fresh,
    )
    rows = _rows([
        (eligible[0], False, 0), (eligible[1], False, 1),
        (eligible[2], False, 2), (eligible[3], False, 3),
        (exposed[0], True, 4),
    ])
    projected, audit = subject._project_fit_only(
        rows, scope, quotas={FAMILIES[0]: 1, FAMILIES[1]: 1},
        salt="fresh-absent")
    assert audit["fresh_outer_scenes_registered_absent_from_source"] == 64
    contaminated = {name: value.copy() for name, value in rows.items()}
    extra = _rows([(_fingerprint(400), False, 0)])
    contaminated = {
        name: np.concatenate([contaminated[name], extra[name]], axis=0)
        for name in contaminated
    }
    with pytest.raises(ValueError, match="scene identity differs"):
        subject._project_fit_only(
            contaminated, scope,
            quotas={FAMILIES[0]: 1, FAMILIES[1]: 1}, salt="fresh-absent")


def test_fresh_outer_scope_requires_one_exact_authenticated_registry_report_pair():
    source_rows_sha256 = "9" * 64
    registry, report, registry_sha256, report_sha256 = _fresh_outer_pair(
        source_rows_sha256)
    selected = subject._validate_fresh_outer_binding(
        registry,
        report,
        registry_file_sha256=registry_sha256,
        report_file_sha256=report_sha256,
        source_rows_sha256=source_rows_sha256,
    )
    assert selected == {
        row["fingerprint"] for row in registry["selected_outer_identities"]
    }

    mismatched = deepcopy(report)
    mismatched["registry_file_sha256"] = "6" * 64
    mismatched["content_sha256"] = subject.digest({
        key: value for key, value in mismatched.items()
        if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="registry/report differs"):
        subject._validate_fresh_outer_binding(
            registry,
            mismatched,
            registry_file_sha256=registry_sha256,
            report_file_sha256=report_sha256,
            source_rows_sha256=source_rows_sha256,
        )

    changed_scope = _scope(
        eligible=[_fingerprint(1)],
        exposed=[],
        families={_fingerprint(1): FAMILIES[0]},
        fresh=sorted(selected),
    )
    changed_scope.pop("fresh_outer_report_file_sha256")
    changed_scope["content_sha256"] = subject.digest({
        key: value for key, value in changed_scope.items()
        if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="scope schema differs"):
        subject.normalize_scope(changed_scope)


def test_authenticated_selected_config_extracts_wrapper_and_binds_fresh_outer(
        tmp_path, monkeypatch):
    paths = _selector_evidence(tmp_path, monkeypatch)
    result = subject.authenticate_embedded_selected_config_snapshot(
        report_path=paths["report"],
        expected_report_sha256=file_hash(paths["report"]),
        scope_path=paths["scope"], selected_config_path=paths["selected"],
        source_report_path=paths["source_report"],
        source_rows_path=paths["source_rows"],
        fit_only_rows_path=paths["fit_only_rows"],
        actor_file_sha256="5" * 64,
        fresh_outer_registry_path=paths["registry"],
        expected_fresh_outer_registry_sha256=file_hash(paths["registry"]),
        fresh_outer_report_path=paths["outer_report"],
        expected_fresh_outer_report_sha256=file_hash(paths["outer_report"]),
    )
    assert result["config"] == paths["config"]
    assert result["selected_config_record"]["selected_config"] == paths["config"]
    assert result["report_file_sha256"] == file_hash(paths["report"])


def test_authenticated_selected_config_rejects_wrapper_or_scope_substitution(
        tmp_path, monkeypatch):
    paths = _selector_evidence(tmp_path, monkeypatch)
    selected = json.loads(paths["selected"].read_text(encoding="utf-8"))
    selected["selected_mix_weight"] = 1.0
    _write_json(paths["selected"], selected)
    with pytest.raises(ValueError, match="artifact hash differs"):
        subject.authenticate_embedded_selected_config_snapshot(
            report_path=paths["report"],
            expected_report_sha256=file_hash(paths["report"]),
            scope_path=paths["scope"], selected_config_path=paths["selected"],
            source_report_path=paths["source_report"],
            source_rows_path=paths["source_rows"],
            fit_only_rows_path=paths["fit_only_rows"],
            actor_file_sha256="5" * 64,
            fresh_outer_registry_path=paths["registry"],
            expected_fresh_outer_registry_sha256=file_hash(paths["registry"]),
            fresh_outer_report_path=paths["outer_report"],
            expected_fresh_outer_report_sha256=file_hash(paths["outer_report"]),
        )


def test_strict_selector_refits_all_candidates_and_accepts_exact_evidence(
        tmp_path, monkeypatch):
    paths = _strict_selector_evidence(tmp_path, monkeypatch)
    result = _strict_authenticate(paths)
    assert result["strict_refit_performed"] is True
    assert result["selection"] == paths["selection"]
    assert result["config"]["mix_weights"]["shared_charger"] == 1.0
    assert len(result["strict_refit_receipt_sha256"]) == 64


def test_build_atomically_publishes_complete_evidence_then_passes_strict_reader(
        tmp_path, monkeypatch):
    """Exercise real snapshot/publication I/O with a deterministic fake fit."""
    fixture = _strict_selector_evidence(tmp_path, monkeypatch)
    source = tmp_path / "source_candidate"
    source.mkdir()
    source_rows_path = source / "rows.npz"
    source_rows_path.write_bytes(
        (fixture["evidence"] / "source_v8_rows.npz").read_bytes())
    source_rows_sha = file_hash(source_rows_path)

    source_config = deepcopy(subject.FROZEN_SOURCE_CONFIG)
    _write_json(source / "fit_config.json", source_config)
    _write_json(source / "program.json", {"kind": "source-program"})
    scope = json.loads(
        (fixture["evidence"] / "fit_scope.json").read_text(encoding="utf-8"))
    families = {
        row["fingerprint"]: row["family_id"]
        for row in scope["inner_candidate_scenes"]
    }
    weight_audit = {"balance": {"combined_training": {"scene_totals": [
        {"scene": scene, "family": family, "mass": 1.0}
        for scene, family in sorted(families.items())
    ]}}}
    _write_json(source / "weights_audit.json", weight_audit)
    source_artifacts = {
        name: file_hash(source / name)
        for name in (
            "rows.npz", "fit_config.json", "program.json",
            "weights_audit.json",
        )
    }
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_ROWS_SHA256",
        source_artifacts["rows.npz"])
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_CONFIG_FILE_SHA256",
        source_artifacts["fit_config.json"])
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_CONFIG_CONTENT_SHA256",
        subject.digest(source_config))
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_PROGRAM_SHA256",
        source_artifacts["program.json"])
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256",
        source_artifacts["weights_audit.json"])

    source_bindings = {
        "actor_file_sha256": file_hash(fixture["actor"]),
        "actor_parameters_sha256": subject.FROZEN_ACTOR_PARAMETERS_SHA256,
        "source_full_manifest_bindings": deepcopy(
            fixture["source_identity"]["source_full_manifest_bindings"]),
    }
    source_bindings["source_full_manifest_bindings_sha256"] = subject.digest(
        source_bindings["source_full_manifest_bindings"])
    source_bindings["fit_config_content_sha256"] = subject.digest(source_config)
    source_report = {
        "version": v8.VERSION,
        "status": v8.STATUS_FAILED,
        "bindings": source_bindings,
        "execution": {
            "final_rows_accessed": False,
            "final_labels_accessed": False,
        },
        "evidence_artifacts": source_artifacts,
    }
    _write_json(source / "report.json", source_report)
    source_report_sha = file_hash(source / "report.json")
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_REPORT_SHA256", source_report_sha)

    scope["source_report_sha256"] = source_report_sha
    scope["source_rows_sha256"] = source_rows_sha
    scope["content_sha256"] = subject.digest({
        key: value for key, value in scope.items() if key != "content_sha256"
    })
    scope_path = tmp_path / "published_fit_scope.json"
    _write_json(scope_path, scope)
    monkeypatch.setattr(
        subject, "FROZEN_FIT_SCOPE_SHA256", file_hash(scope_path))

    class PublishedProgram:
        def __init__(self, payload):
            self.payload = deepcopy(payload)

        def to_dict(self):
            return deepcopy(self.payload)

    class PublishedProgramReader:
        @classmethod
        def from_dict(cls, payload):
            return PublishedProgram(payload)

    monkeypatch.setattr(
        subject.v8, "R41DiagnosticPublicTreeProgramV8",
        PublishedProgramReader)

    def fake_select(arrays, **kwargs):
        configs = subject.candidate_configs(kwargs["source_config"])
        metrics = (
            _metrics(charger_direction=0.84),
            _metrics(charger_direction=0.851, other=0.919),
            _metrics(charger_direction=0.87, other=0.917),
            _metrics(charger_direction=0.88, other=0.919),
        )
        selection = subject.choose_candidate([
            {
                "mix_weight": mix,
                "config_sha256": subject.digest(config),
                "validation_probabilities_sha256": subject.digest({
                    "mix": mix, "kind": "publication-probabilities",
                }),
                "validation_predictions_sha256": subject.digest({
                    "mix": mix, "kind": "publication-predictions",
                }),
                "metrics": metric,
            }
            for mix, config, metric in zip(
                subject.MIX_CANDIDATES, configs, metrics)
        ])
        selection.update({
            "fit_diagnostics": {"publication_test": True},
            "weight_audit": {"publication_test": True},
            "fit_pair_group_bits_sha256": "6" * 64,
            "validation_pair_group_bits_sha256": "7" * 64,
        })
        program = PublishedProgram({
            "kind": "published-test-program",
            "selector_binding_sha256": kwargs["selector_binding"],
        })
        return selection, program, configs

    monkeypatch.setattr(subject, "_select_projected", fake_select)
    output = tmp_path / "published_selector"
    report = subject.build(
        source_evidence=source,
        expected_source_report_sha256=source_report_sha,
        actor_path=fixture["actor"],
        fit_scope_path=scope_path,
        expected_fit_scope_sha256=file_hash(scope_path),
        fresh_outer_registry_path=fixture["registry"],
        expected_fresh_outer_registry_sha256=file_hash(fixture["registry"]),
        fresh_outer_report_path=fixture["outer_report"],
        expected_fresh_outer_report_sha256=file_hash(fixture["outer_report"]),
        output=output,
    )
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {
        "report.json", *subject.EVIDENCE_ARTIFACT_NAMES,
    }
    assert len(report["evidence_artifacts"]) == 9
    assert (output / "source_v8_report.json").read_bytes() == (
        source / "report.json").read_bytes()
    assert report["evidence_artifacts"]["source_v8_report.json"] == (
        source_report_sha)

    strict = subject.authenticate_selected_config_snapshot(
        evidence_directory=output,
        expected_report_sha256=file_hash(output / "report.json"),
        actor_path=fixture["actor"],
        source_full_manifest_bindings=fixture["source_identity"][
            "source_full_manifest_bindings"],
        fresh_outer_registry_path=fixture["registry"],
        expected_fresh_outer_registry_sha256=file_hash(fixture["registry"]),
        fresh_outer_report_path=fixture["outer_report"],
        expected_fresh_outer_report_sha256=file_hash(fixture["outer_report"]),
    )
    assert strict["strict_refit_performed"] is True
    assert strict["selection"] == report["selection"]


@pytest.mark.parametrize("component", ("base", "narrow_passage", "shared_pickup"))
def test_strict_selector_rejects_preregistered_model_parameter_substitution(
        component, tmp_path, monkeypatch):
    paths = _strict_selector_evidence(tmp_path, monkeypatch)
    registry_path = paths["evidence"] / "config_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    changed = registry["candidates"][0]
    changed["config"]["models"][component]["max_iter"] += 1
    changed["config_sha256"] = subject.digest(changed["config"])
    _write_json(registry_path, registry)
    _refresh_strict_selector_report(paths)
    with pytest.raises(ValueError, match="registry differs from preregistration"):
        _strict_authenticate(paths)


def test_strict_selector_rejects_self_reported_inner_metrics(
        tmp_path, monkeypatch):
    paths = _strict_selector_evidence(tmp_path, monkeypatch)
    selection_path = paths["evidence"] / "inner_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidates"][1]["metric_values"]["overall"] = 0.999
    _write_json(selection_path, selection)
    _refresh_strict_selector_report(paths)
    with pytest.raises(ValueError, match="metrics or selection differ from refit"):
        _strict_authenticate(paths)


def test_strict_selector_rejects_program_substitution_with_rehashed_report(
        tmp_path, monkeypatch):
    paths = _strict_selector_evidence(tmp_path, monkeypatch)
    program_path = paths["evidence"] / "inner_fit_program.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["substitution"] = True
    _write_json(program_path, program)
    _refresh_strict_selector_report(paths)
    with pytest.raises(ValueError, match="program differs from deterministic refit"):
        _strict_authenticate(paths)


def test_strict_selector_rejects_unregistered_artifact_bytes(
        tmp_path, monkeypatch):
    paths = _strict_selector_evidence(tmp_path, monkeypatch)
    (paths["evidence"] / "config_registry.json").write_text(
        "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes required|artifact hash differs"):
        _strict_authenticate(paths)


def test_strict_selector_rejects_rehashed_forged_fit_observation(
        tmp_path, monkeypatch):
    """A self-consistent selector archive must still match fixed source rows."""
    paths = _strict_selector_evidence(tmp_path, monkeypatch)
    fit_path = paths["evidence"] / "fit_only_rows.npz"
    arrays = subject._load_npz(fit_path, "Test fit-only rows")
    arrays["observations"] = arrays["observations"].copy()
    arrays["observations"][0, 0] = np.float32(123.0)
    arrays["observation_hashes"] = arrays["observation_hashes"].copy()
    arrays["observation_hashes"][0] = np.bytes_(_fingerprint(999_999))
    fit_path.unlink()
    subject._write_npz(fit_path, arrays)

    # Model the attacker's strongest self-report: refresh the artifact registry
    # and semantic binding after substituting a public observation belonging to
    # an otherwise eligible scene.  The fixed source projection remains the
    # independent authority and must reject it before any refit or metric use.
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["bindings"]["fit_only_rows_semantic_sha256"] = \
        subject._arrays_digest(arrays)
    _write_json(paths["report"], report)
    _refresh_strict_selector_report(paths)
    with pytest.raises(ValueError, match="exact fixed-source projection"):
        _strict_authenticate(paths)


def test_strict_reader_rejects_evidence_race_after_snapshot(
        tmp_path, monkeypatch):
    paths = _strict_selector_evidence(tmp_path, monkeypatch)

    def mutate_original(**kwargs):
        frozen_config = Path(kwargs["evidence_artifact_paths"][
            "config_registry.json"])
        assert frozen_config.read_bytes() == (
            paths["evidence"] / "config_registry.json").read_bytes()
        (paths["evidence"] / "config_registry.json").write_bytes(b"changed\n")
        return {"unreachable": True}

    from pathlib import Path
    monkeypatch.setattr(
        subject, "_authenticate_selected_config_snapshot_from_paths",
        mutate_original)
    with pytest.raises(RuntimeError, match="changed during transaction"):
        _strict_authenticate(paths)


def test_selection_maximizes_the_minimum_nine_gate_margin():
    baseline = _metrics(charger_direction=0.86, narrow_direction=0.851)
    quarter = _metrics(charger_direction=0.88, narrow_direction=0.854)
    half = _metrics(charger_direction=0.90, narrow_direction=0.852)
    full = _metrics(charger_direction=0.88, narrow_direction=0.856)
    result = subject.choose_candidate([
        {"mix_weight": mix, "metrics": metrics}
        for mix, metrics in zip(subject.MIX_CANDIDATES,
                                (baseline, quarter, half, full))
    ])
    assert result["status"] == subject.STATUS_SELECTED
    assert result["selected_mix_weight"] == 1.0
    by_mix = {row["mix_weight"]: row for row in result["candidates"]}
    assert by_mix[0.0]["eligible"]
    assert by_mix[0.25]["eligible"]
    assert by_mix[0.5]["eligible"]
    assert by_mix[1.0]["eligible"]
    assert by_mix[1.0]["minimum_gate_margin"] == pytest.approx(0.006)
    assert result["outer_evaluation_performed"] is False


def test_weight_audit_recovers_one_true_family_per_fit_scene():
    scenes = {_fingerprint(501), _fingerprint(502)}
    audit = {"balance": {"combined_training": {"scene_totals": [
        {"scene": sorted(scenes)[0], "family": FAMILIES[0], "mass": 10.0},
        {"scene": sorted(scenes)[1], "family": FAMILIES[1], "mass": 20.0},
    ]}}}
    result = subject._scene_families_from_weight_audit(
        audit, expected_scenes=scenes)
    assert result == {
        sorted(scenes)[0]: FAMILIES[0], sorted(scenes)[1]: FAMILIES[1],
    }
    audit["balance"]["combined_training"]["scene_totals"].append({
        "scene": sorted(scenes)[0], "family": FAMILIES[2], "mass": 1.0,
    })
    with pytest.raises(ValueError, match="two scene families"):
        subject._scene_families_from_weight_audit(audit, expected_scenes=scenes)


def test_scope_family_swap_is_rejected_against_authenticated_weight_audit():
    eligible = [_fingerprint(601), _fingerprint(602)]
    families = {
        eligible[0]: FAMILIES[0],
        eligible[1]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=[], families=families)
    assert subject._validate_scope_family_registry(
        scope, authenticated_families=families) == families

    swapped = deepcopy(scope)
    swapped["inner_candidate_scenes"][0]["family_id"] = FAMILIES[1]
    swapped["inner_candidate_scenes"][1]["family_id"] = FAMILIES[0]
    swapped["content_sha256"] = subject.digest({
        key: value for key, value in swapped.items()
        if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="authenticated v8 audit"):
        subject._validate_scope_family_registry(
            swapped, authenticated_families=families)


def test_build_input_snapshot_rejects_original_change_before_publish(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    artifacts = {}
    for name, raw in (
        ("rows.npz", b"rows"),
        ("fit_config.json", b"{}\n"),
        ("program.json", b"{}\n"),
        ("weights_audit.json", b"{}\n"),
    ):
        path = source / name
        path.write_bytes(raw)
        artifacts[name] = file_hash(path)
    actor = tmp_path / "actor.npz"
    actor.write_bytes(b"actor")
    scope = tmp_path / "fit_scope.json"
    scope.write_bytes(b"{}\n")
    registry = tmp_path / "development_expansion.json"
    registry.write_bytes(b"{}\n")
    outer_report = tmp_path / "outer_report.json"
    outer_report.write_bytes(b"{}\n")
    report = {
        "version": v8.VERSION,
        "bindings": {"actor_file_sha256": file_hash(actor)},
        "evidence_artifacts": artifacts,
    }
    report_path = source / "report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_REPORT_SHA256", file_hash(report_path))
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_ROWS_SHA256", artifacts["rows.npz"])
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_CONFIG_FILE_SHA256",
        artifacts["fit_config.json"])
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_PROGRAM_SHA256", artifacts["program.json"])
    monkeypatch.setattr(
        subject, "FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256",
        artifacts["weights_audit.json"])

    with subject._snapshot_build_inputs(
        source_evidence=source,
        expected_source_report_sha256=file_hash(report_path),
        actor_path=actor,
        fit_scope_path=scope,
        expected_fit_scope_sha256=file_hash(scope),
        fresh_outer_registry_path=registry,
        expected_fresh_outer_registry_sha256=file_hash(registry),
        fresh_outer_report_path=outer_report,
        expected_fresh_outer_report_sha256=file_hash(outer_report),
    ) as snapshot:
        assert snapshot.paths["actor"].read_bytes() == b"actor"
        actor.write_bytes(b"changed actor")
        assert snapshot.paths["actor"].read_bytes() == b"actor"
        with pytest.raises(RuntimeError, match="changed during transaction"):
            snapshot.verify()


def test_source_closure_contains_no_release_or_final_evaluator():
    sources = subject.producer_sources()
    assert "scripts/build_warehouse_r41_diagnostic_rcpd_v8_fit_selector.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v8.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_pair_weights_v8.py" in sources
    assert not [path for path in sources if any(token in path for token in (
        "admission", "release", "preflight", "fresh_final_holdout",
        "final_once", "explanation_audit",
    ))]
