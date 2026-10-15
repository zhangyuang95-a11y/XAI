from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as frozen
from backend import warehouse_r41_diagnostic_online_runtime as runtime_module


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/manifest.json"
)
ACTOR = (
    ROOT
    / "output/warehouse_native/r41_active_2m_20260911/boundaries/step_2000000/actor.npz"
)
PROTOCOL = ROOT / "output/warehouse_native/r41_active_2m_20260911/protocol.json"


def test_none_scope_authenticates_bytes_without_exposing_any_scene():
    value = frozen.read_saved_manifest(MANIFEST, replay_scope="none")

    assert value["content_sha256"] == frozen.EXPECTED_MANIFEST_CONTENT_SHA256
    assert "splits" not in value
    assert "candidate_batches" not in value
    authentication = value["authentication"]
    assert authentication["full_manifest_json_parsed"] is False
    assert authentication["validation_file_sha256"] \
        == frozen.EXPECTED_VALIDATION_SHA256
    assert authentication[
        "candidate_scenes_disjoint_from_all_base_splits_authenticated"] is True
    assert authentication["protected_final_identity_sha256"] \
        == frozen.EXPECTED_FINAL_IDENTITY_SHA256
    assert authentication["protected_final_fingerprints_sha256"] \
        == frozen.EXPECTED_FINAL_FINGERPRINTS_SHA256
    assert authentication["protected_final_seeds_sha256"] \
        == frozen.EXPECTED_FINAL_SEEDS_SHA256
    forbidden = {"snapshot", "workload_screen", "metrics", "probabilities"}
    assert not forbidden.intersection(str(value))


def test_refuses_an_expected_hash_override_before_reading(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="cannot be overridden"):
        frozen.read_saved_manifest(missing, expected_sha256="0" * 64)


def test_rejects_non_designated_bytes(tmp_path):
    candidate = tmp_path / "manifest.json"
    candidate.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bytes differ"):
        frozen.read_saved_manifest(candidate)


def test_runtime_source_closure_is_complete_and_fixed():
    binding = frozen.runtime_sources()

    assert binding["sources_sha256"] == frozen.EXPECTED_RUNTIME_SOURCES_SHA256
    assert "env/warehouse/transition_outcome.py" in binding["sources"]
    assert "env/warehouse/state_support.py" in binding["sources"]


def test_runtime_source_drift_fails_closed(monkeypatch):
    monkeypatch.setattr(
        frozen,
        "diagnostic_runtime_sources",
        lambda: {"backend/warehouse_r41_diagnostic_online_runtime.py": "0" * 64},
    )

    with pytest.raises(ValueError, match="source closure differs"):
        frozen.runtime_sources()


def test_development_scope_returns_only_verified_development_projection(monkeypatch):
    projection = {
        "splits": {
            "train": [], "conflict_validation": [],
            "tutorial": [], "question_bank": [],
        },
        "workload_generation_reports": [],
        "candidate_batches": [[], [], []],
        "batch_reports": [],
    }
    monkeypatch.setattr(
        frozen, "_development_projection", lambda actor: projection)

    value = frozen.read_saved_manifest(
        MANIFEST, actor_path=ACTOR, replay_scope="development")

    assert set(value["splits"]) == set(frozen.DEVELOPMENT_REPLAY_SPLITS)
    assert "final_test" not in value["splits"]
    assert value["authentication"]["final_test_rows_present"] is False
    assert value["authentication"]["full_manifest_json_parsed"] is False
    assert value["authentication"]["development_projection_sha256"] \
        == frozen.digest(projection)


def test_none_and_development_do_not_parse_full_manifest(monkeypatch):
    monkeypatch.setattr(
        frozen, "_read_json_object",
        lambda raw: (_ for _ in ()).throw(AssertionError("full JSON parsed")))
    frozen.read_saved_manifest(MANIFEST, replay_scope="none")

    projection = {
        "splits": {name: [] for name in frozen.DEVELOPMENT_REPLAY_SPLITS},
        "workload_generation_reports": [],
        "candidate_batches": [[], [], []],
        "batch_reports": [],
    }
    monkeypatch.setattr(
        frozen, "_development_projection", lambda actor: projection)
    frozen.read_saved_manifest(
        MANIFEST, actor_path=ACTOR, replay_scope="development")


def test_public_all_scope_is_permanently_rejected_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="post-success publication authorization"):
        frozen.read_saved_manifest(
            tmp_path / "missing.json", actor_path=tmp_path / "missing.npz",
            replay_scope="all")


def test_module_has_no_unconditional_full_manifest_loader():
    assert not hasattr(frozen, "_read_saved_manifest_for_authorized_full_replay")
    assert not hasattr(frozen, "_read_saved_manifest_for_post_success")
    assert not hasattr(frozen, "protected_final_identities")
    assert "protected_final_identities" not in frozen.__all__


def test_invalid_replay_scope_fails_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="scope is invalid"):
        frozen.read_saved_manifest(
            tmp_path / "missing.json", replay_scope="everything")


def test_development_regeneration_rejects_final_split_before_actor_access(tmp_path):
    with pytest.raises(ValueError, match="Only fixed development splits"):
        frozen.regenerate_development_splits(
            tmp_path / "missing.npz", splits=("final_test",))


def test_development_view_builds_runtime_from_portable_identity():
    runtime = frozen.build_runtime(
        actor_path=ACTOR, protocol_path=PROTOCOL, manifest_path=MANIFEST)

    assert runtime.source_full_manifest_bindings["manifest_semantic_sha256"] \
        == frozen.EXPECTED_MANIFEST_SEMANTIC_SHA256
    assert runtime.source_full_manifest_bindings["manifest_content_sha256"] \
        == frozen.EXPECTED_MANIFEST_CONTENT_SHA256
    assert runtime.source_full_manifest_bindings["manifest_file_sha256"] \
        == frozen.EXPECTED_MANIFEST_SHA256
    assert runtime.manifest_semantic_sha256 \
        != frozen.EXPECTED_MANIFEST_SEMANTIC_SHA256


def test_development_runtime_never_passes_full_manifest_to_json_reader(monkeypatch):
    original = runtime_module._read_object
    full = MANIFEST.resolve()

    def guarded(path, label):
        if Path(path).resolve() == full:
            raise AssertionError("development runtime parsed full manifest")
        return original(path, label)

    monkeypatch.setattr(runtime_module, "_read_object", guarded)
    runtime = frozen.build_runtime(
        actor_path=ACTOR, protocol_path=PROTOCOL, manifest_path=MANIFEST)
    assert runtime.source_full_manifest_bindings["manifest_file_sha256"] \
        == frozen.EXPECTED_MANIFEST_SHA256


def test_development_projection_cache_key_covers_transitive_producer_sources(
        monkeypatch):
    workload_path = Path(
        frozen.replay_workload_and_compare.__code__.co_filename).resolve()
    monkeypatch.setattr(
        frozen, "file_hash",
        lambda path: (frozen.EXPECTED_WORKLOAD_SOURCE_SHA256
                      if Path(path).resolve() == workload_path else "1" * 64),
    )
    monkeypatch.setattr(
        frozen, "runtime_sources", lambda: {"sources_sha256": "2" * 64})
    closures = iter(({"dependency.py": "3" * 64},
                     {"dependency.py": "4" * 64}))
    monkeypatch.setattr(
        frozen, "local_source_hashes", lambda roots: next(closures))

    first = frozen._development_projection_key(Path("actor.npz"))
    second = frozen._development_projection_key(Path("actor.npz"))

    assert first[:-1] == second[:-1]
    assert first[-1] != second[-1]
