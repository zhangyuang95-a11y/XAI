from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect

from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_split as subject
from backend.training.warehouse_native_common import digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _component(label: str, values: list[str], identities: list[dict] | None = None) -> dict:
    projection = {
        "outer_observation_hashes": values,
        "outer_observation_hashes_sha256": digest(values),
        "source_projection_content_sha256": _fp(label + "-projection"),
        "content_sha256": _fp(label + "-projection"),
    }
    value = {"observation_hash_projection": projection}
    if identities is not None:
        value.update({"identities": identities,
                      "identities_sha256": digest(identities)})
    return value


def _closeout(label: str, values: list[str]) -> dict:
    return {"content_sha256": _fp(label),
            "consumed_outer": _component(label, values)}


def test_remaining_population_excludes_v12_outer_and_burned_final():
    rows = []
    excluded_seeds, excluded_fingerprints = set(), set()
    for family_index, family in enumerate(subject.FAMILY_IDS):
        expected = subject.EXPECTED_REMAINING_FAMILY_COUNTS[family]
        for offset in range(expected + 24):
            seed = family_index * 10_000 + offset
            fingerprint = _fp(f"{family}-{offset}")
            row = {"batch_index": 0, "family_id": family,
                   "family_offset": offset, "seed": seed,
                   "fingerprint": fingerprint}
            rows.append(row)
            if offset >= expected:
                excluded_seeds.add(seed)
                excluded_fingerprints.add(fingerprint)
    # The production population is fixed at 2160. Pad every family with rows
    # already excluded by both identity coordinates.
    while len(rows) < subject.FIXED_CANDIDATE_SCENE_COUNT:
        index = len(rows)
        family = subject.FAMILY_IDS[index % len(subject.FAMILY_IDS)]
        seed = 100_000 + index
        fingerprint = _fp(f"excluded-{index}")
        rows.append({"batch_index": 9, "family_id": family,
                     "family_offset": index, "seed": seed,
                     "fingerprint": fingerprint})
        excluded_seeds.add(seed)
        excluded_fingerprints.add(fingerprint)
    retained, selected, counts = subject._remaining_and_selected(
        rows, excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    assert len(retained) == 1375
    assert counts == subject.EXPECTED_REMAINING_FAMILY_COUNTS
    assert len(selected) == 64


def test_prior_projection_unions_six_exposure_components():
    values = [_fp(letter) for letter in "abcdef"]
    v8 = _closeout("v8", sorted(values[:2]))
    v9 = _closeout("v9", sorted(values[1:3]))
    v10 = _closeout("v10", sorted(values[2:4]))
    v11 = _closeout("v11", sorted(values[3:5]))
    v12_outer = _component("v12-outer", sorted(values[4:]))
    v12_final = _component("v12-final", sorted([values[0], values[5]]))
    burned_closeout = {"content_sha256": _fp("v12-closeout")}
    value = subject._prior_validation_wins_projection(
        old_closeout=v8, attempt_closeout=v9, final_closeout=v10,
        consumed_v11_closeout=v11, consumed_v12_outer=v12_outer,
        burned_v12_final=v12_final,
        burned_v12_closeout=burned_closeout)
    assert value["outer_observation_hashes"] == sorted(values)
    assert value["component_unique_counts"] == {
        "consumed_v8": 2, "consumed_v9": 2,
        "consumed_v10": 2, "consumed_v11": 2,
        "consumed_v12_outer": 2, "burned_v12_final": 2}
    assert value["source_v12_final_closeout_content_sha256"] == _fp(
        "v12-closeout")
    assert value["selector_rule"] == (
        "remove fit rows matching any prior outer observation hash before "
        "reading actions, probabilities, or raw observations")
    assert value["selection_used_this_projection"] is True
    assert value["content_sha256"] == digest({
        key: child for key, child in value.items()
        if key != "content_sha256"})


def test_hash_screen_rejects_any_historical_overlap(monkeypatch):
    family = "only"
    monkeypatch.setattr(subject, "FAMILY_IDS", (family,))
    monkeypatch.setattr(subject, "FAMILY_QUOTAS", {family: 1})
    forbidden = _fp("blocked")
    candidates = [
        {"batch_index": 0, "family_id": family, "family_offset": index,
         "seed": index, "fingerprint": _fp(f"scene-{index}")}
        for index in range(2)]
    candidates.sort(key=lambda row: (subject._rank(row), row["fingerprint"]))
    blocked_seed = candidates[0]["seed"]
    monkeypatch.setattr(subject.v12.v11.v9.manifest_binding, "build_runtime",
                        lambda **_: object())
    monkeypatch.setattr(subject, "_materialize_scene",
                        lambda row, index: {**deepcopy(row), "id": str(index)})

    def collect(_runtime, scenes, **_kwargs):
        value = forbidden if scenes[0]["seed"] == blocked_seed else _fp("clear")
        return [{"observation": value}], 1

    monkeypatch.setattr(subject.v12.v11.projection_v10.rows_v7, "_collect", collect)
    monkeypatch.setattr(subject.v12.v11.projection_v10.rows_v7.legacy,
                        "_obs_hash", lambda value: value)
    closure = {"paths": {"actor": "actor", "protocol": "protocol"},
               "screening_manifest": "manifest"}
    projection = {"outer_observation_hashes": [forbidden]}
    selected, _scenes, audit = subject._hash_screened_selection(
        remaining=candidates, prior_projection=projection, closure=closure)
    assert selected[0]["seed"] != blocked_seed
    assert audit["rejected_historical_observation_overlap_scene_count"] == 1
    assert audit["selected_historical_observation_overlap"] == 0


def test_component_identities_are_strict_and_digest_bound():
    identities = [{"batch_index": index, "family_id": "f",
                   "seed": index, "fingerprint": _fp(str(index))}
                  for index in range(subject.FRESH_OUTER_SCENE_COUNT)]
    component = {"identities": identities,
                 "identities_sha256": digest(identities)}
    assert subject._component_identities(component, "test") == identities
    changed = deepcopy(component)
    changed["identities"][0]["seed"] = 999
    try:
        subject._component_identities(changed, "test")
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("tampered identity list accepted")


def test_information_boundary_has_no_final_or_private_salt_access():
    boundary = subject.RECOVERY_BOUNDARY
    assert boundary["burned_v12_final_closed_before_selection"] is True
    assert boundary["consumed_v12_outer_identities_excluded"] is True
    assert boundary["burned_v12_final_identities_excluded"] is True
    assert boundary["promoted_rows_used_during_identity_selection"] is False
    assert boundary["protected_final_access"] is False
    assert boundary["private_salt_access"] is False
    source = inspect.getsource(subject)
    assert "fresh_final_holdout" not in source
    assert "materializer_output" not in source


def test_saved_registry_binds_closeout_and_combined_promotion(
    tmp_path, monkeypatch,
):
    outer_identities = [
        {"batch_index": index, "family_id": "conflict_family_01",
         "seed": 10_000 + index, "fingerprint": _fp(f"outer-{index}")}
        for index in range(subject.FRESH_OUTER_SCENE_COUNT)
    ]
    final_identities = [
        {"batch_index": index, "family_id": "conflict_family_01",
         "seed": 20_000 + index, "fingerprint": _fp(f"final-{index}")}
        for index in range(subject.FRESH_OUTER_SCENE_COUNT)
    ]
    outer_hashes = sorted([_fp("outer-hash")])
    final_hashes = sorted([_fp("final-hash")])
    combined_hashes = sorted(set(outer_hashes + final_hashes))
    closeout = {
        "content_sha256": "a" * 64,
        "consumed_v12_outer": {
            "identities": outer_identities,
            "identities_sha256": digest(outer_identities),
            "rows_sha256": "b" * 64,
            "rows_semantic_sha256": "c" * 64,
            "observation_hash_projection": {
                "outer_observation_hashes": outer_hashes,
                "outer_observation_hashes_sha256": digest(outer_hashes),
                "content_sha256": "d" * 64,
            },
        },
        "burned_final": {
            "identities": final_identities,
            "identities_sha256": digest(final_identities),
            "observation_hash_projection": {
                "outer_observation_hashes": final_hashes,
                "outer_observation_hashes_sha256": digest(final_hashes),
                "content_sha256": "e" * 64,
            },
        },
        "combined_promoted_development": {
            "rows_sha256": "1" * 64,
            "rows_semantic_sha256": "2" * 64,
            "observation_hash_projection": {
                "outer_observation_hashes": combined_hashes,
                "outer_observation_hashes_sha256": digest(combined_hashes),
                "content_sha256": "3" * 64,
            },
        },
    }
    expected_closeout_sha = "4" * 64
    monkeypatch.setattr(
        subject.burned_v12_api, "read_saved_closeout_public",
        lambda *_args, **_kwargs: deepcopy(closeout))
    identities = [
        {"batch_index": index, "family_id": subject.FAMILY_IDS[
            index % len(subject.FAMILY_IDS)], "seed": 30_000 + index,
         "fingerprint": _fp(f"selected-{index}")}
        for index in range(subject.FRESH_OUTER_SCENE_COUNT)
    ]
    scenes = [
        {"id": f"diagnostic_v13_fresh_outer_{index:04d}",
         **identity}
        for index, identity in enumerate(identities)
    ]
    bindings = {name: "f" * 64 for name in subject.RECOVERY_BINDINGS}
    bindings.update({
        "contract_sha256": digest(subject.contract()),
        "burned_v12_final_closeout_file_sha256": expected_closeout_sha,
        "burned_v12_final_closeout_content_sha256": closeout["content_sha256"],
        "promotion_closeout_file_sha256": expected_closeout_sha,
        "promotion_closeout_content_sha256": closeout["content_sha256"],
        "consumed_v12_outer_rows_sha256": "b" * 64,
        "consumed_v12_outer_rows_semantic_sha256": "c" * 64,
        "consumed_v12_outer_selected_identity_sha256": digest(outer_identities),
        "consumed_v12_outer_observation_hashes_sha256": digest(outer_hashes),
        "burned_v12_final_selected_identity_sha256": digest(final_identities),
        "burned_v12_final_observation_hashes_sha256": digest(final_hashes),
        "combined_promoted_rows_sha256": "1" * 64,
        "combined_promoted_rows_semantic_sha256": "2" * 64,
        "combined_promoted_observation_hashes_sha256": digest(combined_hashes),
    })
    stats = {
        "remaining_candidate_scene_count": subject.EXPECTED_REMAINING_SCENE_COUNT,
        "remaining_family_counts": subject.EXPECTED_REMAINING_FAMILY_COUNTS,
        "consumed_v12_outer_identities": subject.FRESH_OUTER_SCENE_COUNT,
        "burned_v12_final_identities": subject.FRESH_OUTER_SCENE_COUNT,
        "union_candidate_identities_excluded": 785,
        "combined_promoted_unique_observation_count": len(combined_hashes),
        "selected_outer_scene_count": subject.FRESH_OUTER_SCENE_COUNT,
        "selected_outer_family_counts": subject.FAMILY_QUOTAS,
        "selected_exposed_seed_overlap": 0,
        "selected_exposed_fingerprint_overlap": 0,
        "hash_screen": {"selected_historical_observation_overlap": 0},
    }
    exclusions = {"all": "5" * 64}
    sources = {"producer.py": "6" * 64}
    registry = {
        "version": subject.VERSION, "status": subject.STATUS,
        "contract": subject.contract(), "bindings": bindings,
        "development_outer": scenes,
        "selected_outer_identities": identities,
        "exclusion_counts": {}, "exclusion_digests": exclusions,
        "statistics": stats,
        "information_boundary": deepcopy(subject.RECOVERY_BOUNDARY),
        "program_access": False, "program_predictions_access": False,
        "action_labels_access": False, "probabilities_access": False,
        "final_audit_rows_access": False, "formal_ready": False,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
    }
    registry["content_sha256"] = digest(registry)
    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(subject._json_bytes(registry))
    report = {
        "version": subject.REPORT_VERSION, "status": subject.STATUS,
        "registry_file_sha256": file_hash(registry_path),
        "registry_content_sha256": registry["content_sha256"],
        "bindings": bindings,
        "selection": {
            "salt": subject.SELECTION_SALT,
            "family_quotas": subject.FAMILY_QUOTAS,
            "selected_identity_sha256": digest(identities),
            "exclusion_digests": exclusions,
        },
        "statistics": stats,
        "information_boundary": deepcopy(subject.RECOVERY_BOUNDARY),
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    report_path = tmp_path / "report.json"
    report_path.write_bytes(subject._json_bytes(report))
    checked_registry, checked_report = subject.read_saved_registry(
        registry_path, report_path,
        expected_registry_sha256=file_hash(registry_path),
        expected_report_sha256=file_hash(report_path),
        burned_v12_final_closeout_path=tmp_path / "closeout.json",
        expected_burned_v12_final_closeout_sha256=expected_closeout_sha,
        permanent_v12_final_closeout_registry=tmp_path)
    assert checked_registry == registry
    assert checked_report == report


def test_registry_uses_public_only_closeout_reader():
    source = inspect.getsource(subject)
    assert source.count("burned_v12_api.read_saved_closeout_public(") == 2
    assert "burned_v12_api.read_saved_closeout(" not in source
    public_reader_source = inspect.getsource(subject.burned_v12_api)
    assert "load_authenticated_rows" not in public_reader_source
    assert "import numpy" not in public_reader_source.lower()
    assert "import zipfile" not in public_reader_source.lower()
