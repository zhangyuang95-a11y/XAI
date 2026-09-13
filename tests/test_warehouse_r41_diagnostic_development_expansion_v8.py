from collections import Counter
import inspect
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as subject


def _candidate(family: str, index: int) -> dict:
    # A real SHA-shaped identity keeps this fixture inside the production schema.
    fingerprint = subject.digest({"family": family, "index": index})
    return {
        "id": f"candidate-{family}-{index}",
        "split": "play_candidates",
        "family_id": family,
        "fingerprint": fingerprint,
        "seed": 1_000_000 + subject.FAMILY_IDS.index(family) * 10_000 + index,
    }


def _manifest(per_family: int = 80) -> dict:
    return {
        "candidate_batches": [[
            _candidate(family, index)
            for family in subject.FAMILY_IDS
            for index in range(per_family)
        ]]
    }


def test_contract_freezes_balanced_program_blind_fit_and_validation_splits():
    contract = subject.contract()
    assert subject.VERSION == "warehouse-r41-diagnostic-development-expansion.v8"
    assert sum(contract["fit_family_quotas"].values()) == 128
    assert sum(contract["validation_family_quotas"].values()) == 64
    assert contract["selection_sequence"] == [
        "development_validation", "fit_supplement"
    ]
    assert contract["validation_frozen_before_fit_selection"] is True
    assert contract["program_access"] is False
    assert contract["program_predictions_access"] is False
    assert contract["actor_logits_access"] is False
    assert contract["actor_hidden_state_access"] is False
    assert contract["participant_data_access"] is False
    assert contract["final_audit_rows_access"] is False
    assert contract["final_labels_used_for_selection"] is False
    assert contract["runtime_action_override"] is False


def test_expansion_is_pinned_to_cycle_free_designation_v2():
    assert subject.designation_api.VERSION == (
        "warehouse-r41-diagnostic-actor-designation.v2"
    )
    assert subject.EXPECTED_DESIGNATION_SHA256 == (
        "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
    )
    assert subject.designation_api.__file__.endswith(
        "warehouse_r41_diagnostic_designation_v2.py"
    )
    assert subject.EXPECTED_SELECTED_SCENES_SHA256 == (
        "30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8"
    )
    assert subject.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256 == (
        "8931b74940f1c41940d9fbb73a52f940f527b8f9c67dfd6b94a4a4d1afd977fe"
    )
    assert subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256 == (
        "cdd17b1a8d46b54dfeec70f74fb3acb43dc1b575c54f58cf9be982ad89d5f111"
    )
    assert subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256 == (
        "4188b293f5c0e02cc80ece7fa1c0841134369c740fffea2aecef90653e077db5"
    )
    assert subject.contract()["full_retired_holdout_access"] is False
    assert subject.contract()["retired_selection_trace_access"] is False
    assert subject.contract()["retired_statistics_or_metrics_access"] is False
    assert subject.contract()["retired_snapshot_or_workload_access"] is False


def test_producer_source_closure_has_no_program_or_final_audit_source():
    sources = subject.producer_sources()
    assert sources
    assert all("explanation_audit" not in path for path in sources)
    assert all("model_tree" not in path for path in sources)
    assert all("online_explanation" not in path for path in sources)
    assert "scripts/build_warehouse_r41_diagnostic_designation_v2.py" in sources
    parameters = set(inspect.signature(subject.build).parameters)
    assert not any("program" in name or "audit" in name or "label" in name
                   for name in parameters)
    assert "retired_identity_projection_path" in parameters
    assert "retired_holdout_paths" not in parameters


def test_selection_is_balanced_disjoint_and_uses_frozen_workload_indexes(monkeypatch):
    calls = []

    def fake_screen(scene, *, split, scene_index, actor, train_count=128):
        calls.append((scene["fingerprint"], split, scene_index, train_count, actor))
        return {
            "passed": True,
            "split": split,
            "scene_index": scene_index,
            "receipt_sha256": subject.digest({
                "fingerprint": scene["fingerprint"],
                "split": split,
                "scene_index": scene_index,
                "train_count": train_count,
            }),
        }

    monkeypatch.setattr(subject, "screen_scene", fake_screen)
    manifest = _manifest()
    # Exclude several high-ranked rows without depending on their manifest order.
    first_family = subject.FAMILY_IDS[0]
    ordered = subject._ordered_candidates(
        manifest, first_family, salt=subject.VALIDATION_ORDER_SALT
    )
    excluded = {row["fingerprint"] for row in ordered[:3]}
    excluded_seeds = {row["seed"] for row in ordered[3:5]}

    fit, validation, trace = subject._select(
        manifest=manifest, actor="actor",
        excluded_fingerprints=set(excluded),
        excluded_seeds=set(excluded_seeds),
    )

    assert len(fit) == subject.FIT_SUPPLEMENT_SCENES == 128
    assert len(validation) == subject.VALIDATION_SCENES == 64
    assert Counter(row["family_id"] for row in fit) == Counter(
        subject.FIT_FAMILY_QUOTAS
    )
    assert Counter(row["family_id"] for row in validation) == Counter(
        subject.VALIDATION_FAMILY_QUOTAS
    )
    fit_fp = {row["fingerprint"] for row in fit}
    validation_fp = {row["fingerprint"] for row in validation}
    assert not fit_fp & validation_fp
    assert not (fit_fp | validation_fp) & excluded
    assert not {row["seed"] for row in (*fit, *validation)} & excluded_seeds
    assert len(trace["development_validation"]) >= len(validation)
    assert len(trace["fit_supplement"]) >= len(fit)

    validation_calls = [row for row in calls if row[1] == "conflict_validation"]
    fit_calls = [row for row in calls if row[1] == "train"]
    assert [row[2] for row in validation_calls] == list(range(64))
    assert {row[3] for row in validation_calls} == {
        subject.TOTAL_DEVELOPMENT_FIT_SCENES
    }
    assert [row[2] for row in fit_calls] == list(range(192, 320))
    assert all(row[4] == "actor" for row in calls)
    assert max(index for _, split, index, _, _ in calls
               if split == "conflict_validation") == 63


def test_selection_rejects_a_family_when_public_workload_cannot_fill_quota(monkeypatch):
    monkeypatch.setattr(
        subject,
        "screen_scene",
        lambda *args, **kwargs: {"passed": False},
    )
    with pytest.raises(RuntimeError, match="family quota unavailable"):
        subject._select(
            manifest=_manifest(per_family=40), actor=object(),
            excluded_fingerprints=set(), excluded_seeds=set(),
        )


def test_scene_identity_validation_rejects_duplicate_seed():
    rows = [_candidate(subject.FAMILY_IDS[0], 0), _candidate(subject.FAMILY_IDS[0], 1)]
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError, match="scene identity differs"):
        subject._scene_identities(rows, expected_count=2, label="fixture")


def test_atomic_output_is_exclusive_and_leaves_no_partial_destination(tmp_path: Path):
    output = tmp_path / "registry"
    registry = {"value": 1}
    report = {"value": 2}
    subject._atomic_output(output, registry, report)
    assert (output / "development_expansion.json").is_file()
    assert (output / "report.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        subject._atomic_output(output, registry, report)
    assert not list(tmp_path.glob(".registry.tmp-*"))
    assert not (tmp_path / ".registry.lock").exists()


def test_atomic_output_removes_temporary_directory_after_write_failure(
    tmp_path: Path, monkeypatch,
):
    original = subject._write_exclusive
    calls = 0

    def fail_second(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return original(path, value)

    monkeypatch.setattr(subject, "_write_exclusive", fail_second)
    with pytest.raises(RuntimeError, match="injected failure"):
        subject._atomic_output(
            tmp_path / "registry", {"value": 1}, {"value": 2}
        )
    assert not (tmp_path / "registry").exists()
    assert not list(tmp_path.glob(".registry.tmp-*"))
    assert not (tmp_path / ".registry.lock").exists()


def test_atomic_output_rechecks_integrity_after_staging_before_rename(
        tmp_path: Path):
    output = tmp_path / "registry"

    def reject_drift():
        raise RuntimeError("injected manifest-validation drift")

    with pytest.raises(RuntimeError, match="manifest-validation drift"):
        subject._atomic_output(
            output,
            {"value": 1},
            {"value": 2},
            before_publish=reject_drift,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".registry.tmp-*"))
    assert not (tmp_path / ".registry.lock").exists()


@pytest.mark.parametrize("replacement", [
    "manifest_validation", "selected", "previous", "retired_projection",
])
def test_input_identity_rejects_nonfixed_exclusion_bytes_before_json_parse(
        tmp_path: Path, monkeypatch, replacement: str):
    paths = {
        name: tmp_path / (name + ".json")
        for name in (
            "actor", "manifest", "manifest_validation", "designation",
            "selected", "previous", "retired_projection",
        )
    }
    paths["manifest_validation"] = tmp_path / "validation.json"
    hashes = {
        paths["actor"]: subject.designation_api.EXPECTED_ACTOR_SHA256,
        paths["manifest"]: subject.EXPECTED_MANIFEST_SHA256,
        paths["manifest_validation"]: subject.manifest_binding.EXPECTED_VALIDATION_SHA256,
        paths["designation"]: subject.EXPECTED_DESIGNATION_SHA256,
        paths["selected"]: subject.EXPECTED_SELECTED_SCENES_SHA256,
        paths["previous"]: subject.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256,
        paths["retired_projection"]: (
            subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
    }
    hashes[paths[replacement]] = "0" * 64
    monkeypatch.setattr(subject, "file_hash", lambda path: hashes[Path(path)])
    monkeypatch.setattr(
        subject.manifest_binding, "read_saved_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manifest parsed before fixed exclusion hashes")))
    monkeypatch.setattr(
        subject, "_read", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("JSON parsed before fixed exclusion hashes")))
    monkeypatch.setattr(
        subject.retired_identity_api, "read_saved_projection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("projection parsed before fixed input hashes")))
    monkeypatch.setattr(
        subject, "_read_exact",
        lambda path, label, expected_sha256: (_ for _ in ()).throw(
            ValueError("requires the fixed input")),
    )
    with pytest.raises((ValueError, AssertionError)):
        subject._input_identity(
            actor_path=paths["actor"], manifest_path=paths["manifest"],
            manifest_validation_path=paths["manifest_validation"],
            designation_path=paths["designation"],
            selected_path=paths["selected"],
            previous_development_path=paths["previous"],
            retired_identity_projection_path=paths["retired_projection"],
        )


def test_input_identity_consumes_only_strict_projection_not_full_retired_json(
        tmp_path: Path, monkeypatch):
    paths = {
        name: tmp_path / name
        for name in (
            "actor", "manifest", "validation.json", "designation", "selected",
            "previous", "retired_identity_projection.json",
        )
    }
    hashes = {
        paths["actor"]: subject.designation_api.EXPECTED_ACTOR_SHA256,
        paths["manifest"]: subject.EXPECTED_MANIFEST_SHA256,
        paths["validation.json"]: subject.manifest_binding.EXPECTED_VALIDATION_SHA256,
        paths["designation"]: subject.EXPECTED_DESIGNATION_SHA256,
        paths["selected"]: subject.EXPECTED_SELECTED_SCENES_SHA256,
        paths["previous"]: subject.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256,
        paths["retired_identity_projection.json"]: (
            subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
    }
    monkeypatch.setattr(subject, "file_hash", lambda path: hashes[Path(path)])
    manifest = {
        "version": subject.MANIFEST_VERSION,
        "content_sha256": subject.manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256,
        "frozen_actor": {
            "sha256": subject.designation_api.EXPECTED_ACTOR_SHA256,
            "actor_parameters_sha256": (
                subject.designation_api.EXPECTED_ACTOR_PARAMETERS_SHA256),
        },
        "authentication": {
            "validation_file_sha256": (
                subject.manifest_binding.EXPECTED_VALIDATION_SHA256),
            "candidate_scenes_disjoint_from_all_base_splits_authenticated": True,
        },
        "candidate_batches": [],
        "splits": {name: [] for name in subject.manifest_binding.DEVELOPMENT_REPLAY_SPLITS},
    }
    registered = []
    for index in range(229):
        registered.append({
            "seed": 1000 + index,
            "fingerprint": subject.digest({"registered": index}),
        })
    first = subject.manifest_binding.DEVELOPMENT_REPLAY_SPLITS[0]
    manifest["splits"][first] = registered
    monkeypatch.setattr(
        subject.manifest_binding, "read_saved_manifest",
        lambda *args, **kwargs: manifest)
    designation = {
        "version": subject.designation_api.VERSION,
        "status": subject.designation_api.STATUS,
        "designated": True,
        "release_class": subject.designation_api.RELEASE_CLASS,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "runtime_action_override": False,
        "formal_ready": False,
        "formal_sample_eligible": False,
        "test_fixture": False,
        "bindings": {
            "actor_sha256": subject.designation_api.EXPECTED_ACTOR_SHA256,
            "actor_parameters_sha256": (
                subject.designation_api.EXPECTED_ACTOR_PARAMETERS_SHA256),
        },
    }
    monkeypatch.setattr(
        subject.designation_binding, "read_bound_designation_snapshot",
        lambda *args, **kwargs: designation)
    selected = {
        "version": "warehouse-r41-diagnostic-conflict-dynamic-selection.v3",
        "release_eligible": True,
        "actor_sha256": subject.designation_api.EXPECTED_ACTOR_SHA256,
        "source_manifest_file_sha256": subject.EXPECTED_MANIFEST_SHA256,
        "source_manifest_content_sha256": (
            subject.manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256),
        "six_distinct_conflict_families": True,
        "zero_action_overrides": True,
        "X": [{"seed": 10 + i, "fingerprint": subject.digest({"x": i})}
              for i in range(3)],
        "Y": [{"seed": 20 + i, "fingerprint": subject.digest({"y": i})}
              for i in range(3)],
    }
    previous = {
        "version": "warehouse-r41-diagnostic-development-supplement.v1",
        "status": "passed",
        "program_access": False,
        "final_audit_rows_access": False,
        "bindings": {
            "actor_sha256": subject.designation_api.EXPECTED_ACTOR_SHA256,
            "source_manifest_sha256": subject.EXPECTED_MANIFEST_SHA256,
            "selected_scenes_sha256": subject.EXPECTED_SELECTED_SCENES_SHA256,
        },
        "statistics": {"accepted": 64},
        "scenes": [
            {"seed": 3000 + i, "fingerprint": subject.digest({"previous": i})}
            for i in range(64)
        ],
    }
    previous["content_sha256"] = subject.digest(previous)
    read_labels = []

    def read(path, label):
        read_labels.append(label)
        if path == paths["selected"]:
            return selected
        if path == paths["previous"]:
            return previous
        raise AssertionError("full retired JSON was opened by development expansion")

    monkeypatch.setattr(
        subject, "_read_exact",
        lambda path, label, **kwargs: read(path, label))
    retired_projection = {
        "version": subject.retired_identity_api.VERSION,
        "content_sha256": "a" * 64,
        "sources": [
            {
                "version": version,
                "source_file_sha256": source_sha,
                "accepted_count": 64,
                "exposed_count": count,
            }
            for count, (version, source_sha) in zip(
                (68, 76),
                sorted(subject.retired_identity_api.SOURCE_FILE_SHA256.items()),
            )
        ],
        "exposed_identities": [
            {"seed": 4000 + i,
             "fingerprint": subject.digest({"retired": 4000 + i})}
            for i in range(139)
        ],
    }
    projection_calls = []

    def read_projection(path, **kwargs):
        projection_calls.append((path, kwargs))
        return retired_projection

    monkeypatch.setattr(
        subject.retired_identity_api, "read_saved_projection", read_projection)
    result = subject._input_identity(
        actor_path=paths["actor"], manifest_path=paths["manifest"],
        manifest_validation_path=paths["validation.json"],
        designation_path=paths["designation"], selected_path=paths["selected"],
        previous_development_path=paths["previous"],
        retired_identity_projection_path=(
            paths["retired_identity_projection.json"]),
        designation_original_path=paths["designation"],
        designation_snapshot_components={},
        designation_original_components={},
    )
    assert len(result[4]) == 229 + 6 + 64 + 139
    assert len(result[5]) == 229 + 6 + 64 + 139
    assert read_labels == ["formal X/Y selection", "previous development supplement"]
    assert projection_calls == [(
        paths["retired_identity_projection.json"],
        {
            "expected_projection_sha256": (
                subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
            "expected_report_sha256": (
                subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        },
    )]


def test_input_identity_has_no_live_designation_fallback(tmp_path, monkeypatch):
    paths = {
        name: tmp_path / name
        for name in (
            "actor", "manifest", "validation.json", "designation", "selected",
            "previous", "retired_identity_projection.json",
        )
    }
    hashes = {
        paths["actor"]: subject.designation_api.EXPECTED_ACTOR_SHA256,
        paths["manifest"]: subject.EXPECTED_MANIFEST_SHA256,
        paths["validation.json"]: (
            subject.manifest_binding.EXPECTED_VALIDATION_SHA256),
        paths["designation"]: subject.EXPECTED_DESIGNATION_SHA256,
        paths["selected"]: subject.EXPECTED_SELECTED_SCENES_SHA256,
        paths["previous"]: subject.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256,
        paths["retired_identity_projection.json"]: (
            subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
    }
    monkeypatch.setattr(subject, "file_hash", lambda path: hashes[Path(path)])
    manifest = {
        "version": subject.MANIFEST_VERSION,
        "content_sha256": subject.manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256,
        "frozen_actor": {
            "sha256": subject.designation_api.EXPECTED_ACTOR_SHA256,
            "actor_parameters_sha256": (
                subject.designation_api.EXPECTED_ACTOR_PARAMETERS_SHA256),
        },
        "authentication": {
            "validation_file_sha256": (
                subject.manifest_binding.EXPECTED_VALIDATION_SHA256),
            "candidate_scenes_disjoint_from_all_base_splits_authenticated": True,
        },
        "candidate_batches": [],
        "splits": {
            name: [] for name in subject.manifest_binding.DEVELOPMENT_REPLAY_SPLITS
        },
    }
    monkeypatch.setattr(
        subject.manifest_binding, "read_saved_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        subject.designation_binding, "read_bound_designation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live designation fallback was used")),
    )
    with pytest.raises(ValueError, match="designation snapshot binding"):
        subject._input_identity(
            actor_path=paths["actor"], manifest_path=paths["manifest"],
            manifest_validation_path=paths["validation.json"],
            designation_path=paths["designation"],
            selected_path=paths["selected"],
            previous_development_path=paths["previous"],
            retired_identity_projection_path=(
                paths["retired_identity_projection.json"]),
        )


@pytest.mark.parametrize("drift", [
    "source",
    "manifest_validation",
    "retired_projection",
    "retired_projection_report",
    "training_ledger",
    "dual_evaluation",
    "failure_closeout",
])
def test_build_rejects_source_or_input_drift_before_publication(
        tmp_path: Path, monkeypatch, drift: str):
    files = []
    for name in (
            "actor", "manifest", "validation.json", "designation", "selected",
            "previous", "retired_identity_projection.json", "report.json"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        files.append(path)
    (actor, manifest_path, manifest_validation_path, designation_path,
     selected_path, previous, retired_projection_path,
     retired_projection_report_path) = files
    designation_components = {
        "actor": actor,
        "protocol": tmp_path / "component-protocol",
        "training_ledger": tmp_path / "component-training-ledger",
        "dual_evaluation": tmp_path / "component-dual-evaluation",
        "failure_closeout": tmp_path / "component-failure-closeout",
    }
    for name, path in designation_components.items():
        if name != "actor":
            path.write_text("{}\n", encoding="utf-8")
    source_calls = []

    def sources():
        source_calls.append(1)
        changed = drift == "source" and len(source_calls) > 1
        return {"producer.py": ("b" if changed else "a") * 64}

    monkeypatch.setattr(subject, "producer_sources", sources)
    monkeypatch.setattr(
        subject.designation_binding, "resolve_bound_components_from_bytes",
        lambda *args, **kwargs: designation_components,
    )
    expected_by_path = {
        "actor": actor,
        "manifest": manifest_path,
        "manifest_validation": manifest_validation_path,
        "designation": designation_path,
        "selected": selected_path,
        "previous": previous,
        "retired_projection": retired_projection_path,
        "retired_projection_report": retired_projection_report_path,
        "designation_actor": actor,
        "designation_protocol": designation_components["protocol"],
        "designation_training_ledger": designation_components["training_ledger"],
        "designation_dual_evaluation": designation_components["dual_evaluation"],
        "designation_failure_closeout": designation_components["failure_closeout"],
    }
    monkeypatch.setattr(
        subject, "read_authenticated_bytes",
        lambda path, **kwargs: Path(path).read_bytes())
    original_snapshot = subject.ImmutableInputSnapshot

    def snapshot(paths, **kwargs):
        return original_snapshot(
            paths,
            expected_sha256={name: subject.file_hash(path)
                             for name, path in paths.items()},
            relative_names=kwargs.get("relative_names"),
            prefix=kwargs.get("prefix", "test-expansion-"),
        )

    monkeypatch.setattr(subject, "ImmutableInputSnapshot", snapshot)
    monkeypatch.setattr(subject, "_input_identity", lambda **kwargs: (
        {"candidate_batches": []},
        {"bindings": {
            "actor_parameters_sha256": "1" * 64,
            "protocol_file_sha256": "2" * 64,
            "protocol_content_sha256": "3" * 64,
        }},
        {}, {
            "version": subject.retired_identity_api.VERSION,
            "content_sha256": "6" * 64,
            "sources": [
                {"version": version, "source_file_sha256": source_sha,
                 "accepted_count": 64, "exposed_count": count}
                for count, (version, source_sha) in zip(
                    (68, 76), sorted(
                        subject.retired_identity_api.SOURCE_FILE_SHA256.items()))
            ],
            "exposed_identities": [],
        }, set(), set(), {
            "registered_manifest_scenes": 229,
            "formal_xy_scenes": 6,
            "previous_development_scenes": 64,
            "retired_fresh_final_accepted_scenes": 128,
            "retired_fresh_final_exposed_identities": 139,
            "unique_excluded_fingerprints": 438,
            "unique_excluded_seeds": 438,
            "excluded_fingerprints_sha256": "4" * 64,
            "excluded_seeds_sha256": "5" * 64,
        }, {"content_sha256": "8" * 64},
    ))
    monkeypatch.setattr(subject, "load_frozen_actor", lambda path: object())

    def rows(quotas, prefix, seed_offset):
        result = []
        for family in subject.FAMILY_IDS:
            for index in range(quotas[family]):
                result.append({
                    "family_id": family,
                    "fingerprint": subject.digest(
                        {"prefix": prefix, "family": family, "index": index}),
                    "seed": seed_offset + len(result),
                })
        return result

    fit = rows(subject.FIT_FAMILY_QUOTAS, "fit", 10_000)
    validation = rows(subject.VALIDATION_FAMILY_QUOTAS, "validation", 20_000)
    def select(**kwargs):
        if drift == "manifest_validation":
            manifest_validation_path.write_text(
                '{"changed":true}\n', encoding="utf-8")
        elif drift == "retired_projection":
            retired_projection_path.write_text(
                '{"changed":true}\n', encoding="utf-8")
        elif drift == "retired_projection_report":
            retired_projection_report_path.write_text(
                '{"changed":true}\n', encoding="utf-8")
        elif drift in designation_components and drift != "actor":
            designation_components[drift].write_text(
                '{"changed":true}\n', encoding="utf-8")
        return fit, validation, {
            "fit_supplement": [], "development_validation": [],
        }

    monkeypatch.setattr(subject, "_select", select)
    monkeypatch.setattr(
        subject, "_read",
        lambda path, label: {"content_sha256": "8" * 64})
    monkeypatch.setattr(
        subject.manifest_binding, "runtime_sources",
        lambda: {"sources_sha256": "9" * 64})
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="changed"):
        subject.build(
            actor_path=actor, manifest_path=manifest_path,
            designation_path=designation_path,
            selected_scenes_path=selected_path,
            previous_development_path=previous,
            retired_identity_projection_path=retired_projection_path,
            output=output,
        )
    if drift == "source":
        assert len(source_calls) >= 2
    else:
        assert source_calls
    assert not output.exists()
