from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_final_materializer_v9 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _claim(tmp_path: Path, *, sources=None) -> tuple[Path, dict]:
    sources = subject.producer_sources() if sources is None else sources
    inputs = {
        "scheme": subject.FINAL_ONCE_VERSION + ".candidate-and-outer.v1",
        "candidate_lock_sha256": _fp("lock-file"),
        "candidate_lock_content_sha256": _fp("lock-content"),
        "actor_sha256": _fp("actor"),
        "protocol_sha256": _fp("protocol"),
        "runtime_manifest_sha256": _fp("manifest"),
        "designation_sha256": _fp("designation"),
        "program_sha256": _fp("program"),
        "public_feature_contract_sha256": _fp("features"),
        "candidate_source_closure_sha256": _fp("candidate-sources"),
        "outer_result_sha256": _fp("outer-result"),
        "outer_attempt_key": _fp("outer-attempt"),
    }
    attempt_key = digest(inputs)
    campaign = tmp_path / attempt_key
    campaign.mkdir()
    value = {
        "version": subject.ANCHOR_VERSION,
        "status": subject.ANCHOR_STATUS,
        "attempt_key": attempt_key,
        "attempt_key_inputs": inputs,
        "bindings": {
            "outer_result_content_sha256": _fp("outer-content"),
            "candidate_runtime_source_closure_sha256": _fp("runtime-sources"),
            "final_controller_source_closure_sha256": _fp("controller-sources"),
            "final_materializer_source_closure_sha256": digest(sources),
            "actor_feature_names_sha256": _fp("actor-features"),
        },
        "candidate_and_outer_authenticated_before_claim": True,
        "final_identity_or_rows_accessed_before_claim": False,
        "retry_allowed": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    path = campaign / "attempt_anchor.json"
    path.write_text(canonical(value) + "\n", encoding="utf-8")
    return path, value


def _scene(family: str, index: int) -> dict:
    fingerprint = _fp(f"{family}:{index}")
    return {
        "id": f"source-{family}-{index}", "split": "play_candidates",
        "seed": 100_000 + 1_000 * subject.FAMILY_IDS.index(family) + index,
        "fingerprint": fingerprint, "family_id": family,
        "batch_index": index % 3, "snapshot": {"frame": 0},
    }


def _population(extra: int = 2):
    scenes = [
        _scene(family, index)
        for family in subject.FAMILY_IDS
        for index in range(subject.FAMILY_QUOTAS[family] + extra)
    ]
    identities = [{key: scene[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")}
        for scene in scenes]
    return scenes, identities, {scene["fingerprint"]: scene for scene in scenes}


def test_contract_declares_program_blind_actor_output_boundary():
    value = subject.contract()
    assert value["private_salt_read_after_authenticated_claim"] is True
    assert value["actor_executes_only_inside_fixed_public_workload_replay"] is True
    assert value["raw_actor_actions_used_as_ranking_input"] is False
    assert value["actor_logits_or_probabilities_used_as_ranking_input"] is False
    assert value["actor_outputs_written_to_material_artifact"] is False
    assert value["program_access"] is False
    assert value["runtime_action_override"] is False


def test_claim_authentication_accepts_exact_permanent_anchor(tmp_path):
    path, expected = _claim(tmp_path)
    actual_path, actual = subject._authenticate_claim(path)
    assert actual_path == path
    assert actual == expected


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(status="not-claimed"),
    lambda value: value["attempt_key_inputs"].update(actor_sha256=_fp("other")),
    lambda value: value["bindings"].update(
        final_materializer_source_closure_sha256=_fp("other-source")),
    lambda value: value.update(final_identity_or_rows_accessed_before_claim=True),
])
def test_claim_authentication_fails_closed_on_tampering(tmp_path, mutation):
    path, value = _claim(tmp_path)
    mutation(value)
    value["content_sha256"] = digest({
        key: child for key, child in value.items() if key != "content_sha256"
    })
    path.write_text(canonical(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="claim semantics"):
        subject._authenticate_claim(path)


def test_invalid_claim_prevents_configuration_or_salt_access(tmp_path, monkeypatch):
    bad = tmp_path / "attempt_anchor.json"
    bad.write_text("{}\n", encoding="utf-8")
    touched = []
    monkeypatch.setattr(subject, "_load_config",
                        lambda: touched.append("config"))
    monkeypatch.setattr(subject, "_read_committed_salt",
                        lambda _path: touched.append("salt"))
    with pytest.raises(ValueError):
        subject.materialize(claim=bad, output=tmp_path / "materializer_output.json")
    assert touched == []


def test_materialize_opens_salt_after_public_preparation(tmp_path, monkeypatch):
    sources = {"materializer.py": _fp("source")}
    monkeypatch.setattr(subject, "producer_sources", lambda: sources)
    claim_path, anchor = _claim(tmp_path, sources=sources)
    events = []
    paths = {name: tmp_path / name for name in subject._CONFIG_PATH_FIELDS}
    monkeypatch.setattr(
        subject, "_authenticate_claim",
        lambda value: (events.append("claim") or (claim_path, anchor)))
    monkeypatch.setattr(subject, "_load_config",
                        lambda: (events.append("config") or (tmp_path / "c", paths)))
    authenticated = {"development_scenes": set(), "fresh_outer_scenes": set()}
    monkeypatch.setattr(
        subject, "_authenticate_public_inputs",
        lambda **kwargs: (events.append("public") or authenticated))
    prepared = {"ready": True}
    monkeypatch.setattr(subject, "_prepare_selection",
                        lambda **kwargs: (events.append("prepare") or prepared))
    monkeypatch.setattr(subject, "_read_committed_salt",
                        lambda _path: (events.append("salt") or b"test-salt"))
    scenes = [{
        "id": f"final-{index}", "seed": index,
        "fingerprint": _fp(f"final-{index}"),
        "family_id": subject.FAMILY_IDS[index % len(subject.FAMILY_IDS)],
    } for index in range(subject.FINAL_SCENE_COUNT)]

    def build(**kwargs):
        events.append("build")
        assert kwargs["salt"] == b"test-salt"
        value = {
            "version": subject.MATERIAL_VERSION,
            "status": subject.MATERIAL_STATUS,
            "claim": {"attempt_key": anchor["attempt_key"],
                      "attempt_anchor_content_sha256": anchor["content_sha256"]},
            "scenes": scenes,
            "selection": {}, "producer_sources": sources,
            "producer_sources_sha256": digest(sources), "formal_ready": False,
        }
        value["content_sha256"] = digest(value)
        return value

    monkeypatch.setattr(subject, "_build_material", build)
    output = claim_path.parent / "materializer_output.json"
    subject.materialize(claim=claim_path, output=output)
    assert events == ["claim", "config", "public", "prepare", "salt", "build"]
    assert output.is_file()


def test_materialize_rejects_non_claim_local_output(tmp_path, monkeypatch):
    sources = {"materializer.py": _fp("source")}
    monkeypatch.setattr(subject, "producer_sources", lambda: sources)
    claim_path, anchor = _claim(tmp_path, sources=sources)
    monkeypatch.setattr(subject, "_authenticate_claim", lambda _: (claim_path, anchor))
    paths = {name: tmp_path / name for name in subject._CONFIG_PATH_FIELDS}
    monkeypatch.setattr(subject, "_load_config", lambda: (tmp_path / "c", paths))
    monkeypatch.setattr(subject, "_authenticate_public_inputs", lambda **_: {})
    monkeypatch.setattr(subject, "_prepare_selection", lambda **_: {})
    monkeypatch.setattr(subject, "_read_committed_salt", lambda _: b"salt")
    monkeypatch.setattr(subject, "_build_material", lambda **_: {
        "producer_sources": sources,
    })
    with pytest.raises(ValueError, match="claim-local"):
        subject.materialize(
            claim=claim_path, output=tmp_path / "materializer_output.json")


def test_ordering_is_salt_bound_and_family_local():
    _, identities, _ = _population()
    first = subject._ordered_candidates(
        identities, family=subject.FAMILY_IDS[0], salt=b"salt-a")
    again = subject._ordered_candidates(
        list(reversed(identities)), family=subject.FAMILY_IDS[0], salt=b"salt-a")
    second = subject._ordered_candidates(
        identities, family=subject.FAMILY_IDS[0], salt=b"salt-b")
    assert [row["fingerprint"] for row in first] == [
        row["fingerprint"] for row in again]
    assert [row["fingerprint"] for row in first] != [
        row["fingerprint"] for row in second]
    assert {row["family_id"] for row in first} == {subject.FAMILY_IDS[0]}


def test_selection_replays_fixed_actor_but_never_publishes_receipt(
    monkeypatch,
):
    scenes, identities, mapping = _population()
    runtime = SimpleNamespace(actor=object())
    seen = []

    def screen(scene, **kwargs):
        seen.append((scene["fingerprint"], kwargs["scene_index"]))
        return {
            "passed": True,
            "next_actions": ["RIGHT"],
            "actor_probability": [0.0, 0.0, 1.0, 0.0, 0.0],
            "receipt_sha256": _fp("receipt:" + scene["fingerprint"]),
        }

    monkeypatch.setattr(subject, "screen_scene", screen)
    monkeypatch.setattr(
        subject.v4, "_exact_final_workload_observations",
        lambda _runtime, scene, index: {
            _fp(f"observation:{scene['fingerprint']}:{index}")})
    selected, stats = subject._select_and_replay(
        runtime=runtime, identities=identities,
        scene_by_fingerprint=mapping, salt=b"committed-test-salt",
        excluded_seeds=set(), excluded_fingerprints=set(),
        forbidden_observation_hashes=set())
    assert len(selected) == subject.FINAL_SCENE_COUNT
    assert stats["families"] == dict(sorted(subject.FAMILY_QUOTAS.items()))
    assert stats["raw_actor_outputs_exposed_to_selector"] is False
    assert all("workload_screen" not in scene for scene in selected)
    assert all("next_actions" not in canonical(scene) for scene in selected)
    assert all("actor_probability" not in canonical(scene) for scene in selected)
    assert seen


def test_selection_skips_public_observation_overlap_without_action_access(
    monkeypatch,
):
    _, identities, mapping = _population(extra=3)
    family = subject.FAMILY_IDS[0]
    first = subject._ordered_candidates(
        identities, family=family, salt=b"salt")[0]
    blocked = _fp("blocked-observation")
    monkeypatch.setattr(subject, "screen_scene", lambda *_args, **_kwargs: {
        "passed": True, "private_action": "LEFT"})

    def observations(_runtime, scene, index):
        if scene["fingerprint"] == first["fingerprint"]:
            return {blocked}
        return {_fp(f"clean:{scene['fingerprint']}:{index}")}

    monkeypatch.setattr(
        subject.v4, "_exact_final_workload_observations", observations)
    selected, stats = subject._select_and_replay(
        runtime=SimpleNamespace(actor=object()), identities=identities,
        scene_by_fingerprint=mapping, salt=b"salt",
        excluded_seeds=set(), excluded_fingerprints=set(),
        forbidden_observation_hashes={blocked})
    assert first["fingerprint"] not in {
        scene["fingerprint"] for scene in selected}
    assert stats["rejected"]["prior_public_observation_overlap"] == 1


def test_selection_never_replaces_actor_action_or_opens_program(monkeypatch):
    source = inspect.getsource(subject)
    config_fields = subject._CONFIG_PATH_FIELDS
    assert "program" not in config_fields
    assert "runtime.step" not in inspect.getsource(subject._select_and_replay)
    assert ".predict" not in inspect.getsource(subject._select_and_replay)
    assert "program_path" not in source
    assert "submitted_action" not in source
    assert subject.contract()["runtime_action_override"] is False


def test_public_authentication_reproduces_locked_v11_validation_wins():
    development = [_fp("same-a"), _fp("outer"), _fp("same-a"), _fp("same-c")]
    outer = [_fp("outer")]
    keep = ~np.isin(
        np.asarray(development, dtype="U64"), np.asarray(outer, dtype="U64"))
    retained = [value for value, selected in zip(development, keep)
                if bool(selected)]
    packed = np.ascontiguousarray(keep.astype(np.uint8))
    validation_wins = {
        "source_rows": 4,
        "retained_rows": 3,
        "removed_rows": 1,
        "source_unique_observations": 3,
        "retained_unique_observations": 2,
        "fresh_outer_unique_observations": 1,
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": sha256(memoryview(packed).cast("B")).hexdigest(),
        "retained_observation_hashes_sha256": digest(retained),
    }
    assert subject._retained_development_hashes(
        development_ordered=development, outer_unique_hashes=outer,
        validation_wins=validation_wins) == {_fp("same-a"), _fp("same-c")}
    with pytest.raises(ValueError, match="validation-wins projection differs"):
        subject._retained_development_hashes(
            development_ordered=development, outer_unique_hashes=outer,
            validation_wins=dict(validation_wins, retained_rows=4))


def test_source_orders_claim_config_public_preparation_then_salt():
    source = inspect.getsource(subject.materialize)
    positions = [source.index(token) for token in (
        "_authenticate_claim(", "_load_config(", "_authenticate_public_inputs(",
        "_prepare_selection(", "_read_committed_salt(", "_build_material(",
    )]
    assert positions == sorted(positions)


def test_config_is_strict_and_does_not_embed_secret(tmp_path, monkeypatch):
    assert "selector_report" in subject._CONFIG_PATH_FIELDS
    assert "burned_v10_final_closeout" in subject._CONFIG_PATH_FIELDS
    assert "permanent_v10_final_closeout_registry" in (
        subject._CONFIG_PATH_FIELDS)
    paths = {name: str((tmp_path / name).absolute())
             for name in subject._CONFIG_PATH_FIELDS}
    value = {"version": subject.CONFIG_VERSION, "paths": paths}
    value["content_sha256"] = digest(value)
    config = tmp_path / "config.json"
    config.write_text(canonical(value) + "\n", encoding="utf-8")
    monkeypatch.setenv(subject.CONFIG_ENV, str(config))
    path, actual = subject._load_config()
    assert path == config
    assert actual["private_salt"] == tmp_path / "private_salt"
    assert all(isinstance(value, Path) for value in actual.values())


def _prior_projection() -> dict:
    values = sorted({_fp("v8-observation"), _fp("v9-observation"),
                     _fp("v10-observation")})
    value = {
        "version": subject.registry_api.VERSION
            + ".validation-wins-exclusion.v1",
        "source_v8_closeout_content_sha256": _fp("v8-closeout"),
        "source_v8_projection_content_sha256": _fp("v8-projection"),
        "source_v9_closeout_content_sha256": _fp("v9-closeout"),
        "source_v9_projection_content_sha256": _fp("v9-projection"),
        "source_v10_final_closeout_content_sha256": _fp("v10-closeout"),
        "source_v10_projection_content_sha256": _fp("v10-projection"),
        "outer_observation_hashes": values,
        "unique_outer_observation_hash_count": len(values),
        "outer_observation_hashes_sha256": digest(values),
        "component_unique_counts": {
            "consumed_v8": 1, "consumed_v9": 1, "consumed_v10": 1},
        "component_hashes_sha256": {
            "consumed_v8": digest([_fp("v8-observation")]),
            "consumed_v9": digest([_fp("v9-observation")]),
            "consumed_v10": digest([_fp("v10-observation")]),
        },
        "selector_rule": "validation-wins test projection",
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
        "selection_used_this_projection": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def test_prior_outer_projection_requires_exact_v11_three_campaign_union(tmp_path):
    expected = _prior_projection()
    path = tmp_path / "outer_observation_hashes.json"
    path.write_text(canonical(expected) + "\n", encoding="utf-8")
    assert subject._prior_outer_hashes(
        path, expected_file_sha256=file_hash(path),
        expected_content_sha256=expected["content_sha256"],
        expected_projection=expected) == set(expected["outer_observation_hashes"])

    changed = deepcopy(expected)
    changed["source_v10_projection_content_sha256"] = _fp("other-v10")
    changed["content_sha256"] = digest({
        key: child for key, child in changed.items() if key != "content_sha256"})
    path.write_text(canonical(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Prior failed-outer hash projection differs"):
        subject._prior_outer_hashes(
            path, expected_file_sha256=file_hash(path),
            expected_content_sha256=changed["content_sha256"],
            expected_projection=expected)


def test_exposure_closure_accepts_v11_hash_screened_selection(
        tmp_path, monkeypatch):
    selected = []
    index = 0
    for family in subject.FAMILY_IDS:
        for _ in range(subject.FAMILY_QUOTAS[family]):
            selected.append({
                "batch_index": index % 3,
                "family_id": family,
                "seed": 900_000 + index,
                "fingerprint": _fp(f"v11-selected-{index}"),
            })
            index += 1
    remaining = selected + [{
        "batch_index": 0, "family_id": subject.FAMILY_IDS[0],
        "seed": 999_999, "fingerprint": _fp("identity-only-first"),
    }]
    consumed_v10_hash = _fp("consumed-v10-observations")
    closure = {
        "candidates": remaining,
        "excluded_seeds": set(), "excluded_fingerprints": set(),
        "row_fingerprints": set(), "row_candidate_fingerprints": set(),
        "trace_fingerprints": set(), "current_identities": [],
        "v9_consumed_identities": [], "v10_consumed_identities": [],
        "formal_fingerprints": set(), "previous_fingerprints": set(),
        "retired_fingerprints": set(),
        "burned_final_closeout": {"consumed_outer": {
            "observation_hash_projection": {
                "outer_observation_hashes_sha256": consumed_v10_hash}}},
    }
    monkeypatch.setattr(
        subject.registry_api, "replay_exclusion_closure",
        lambda **_kwargs: closure)
    remaining_counts = dict(sorted({
        family: subject.FAMILY_QUOTAS[family]
            + (1 if family == subject.FAMILY_IDS[0] else 0)
        for family in subject.FAMILY_IDS}.items()))
    # The identity-only prefix intentionally differs from the saved set.  V11
    # may skip ranked identities whose replay hashes overlap consumed v10.
    monkeypatch.setattr(
        subject.registry_api, "_remaining_and_selected",
        lambda *_args, **_kwargs: (
            remaining, list(reversed(selected)), remaining_counts))
    hash_screen = {
        "selected": [{key: row[key] for key in (
            "seed", "fingerprint", "family_id")} for row in selected],
        "selected_scene_count": len(selected),
        "selected_v10_observation_overlap": 0,
        "consumed_v10_observation_hashes_sha256": consumed_v10_hash,
    }
    hash_screen["content_sha256"] = digest(hash_screen)
    exclusion_counts = {
        "source_row_scene_fingerprints": 0,
        "source_rows_in_fixed_candidate_population": 0,
        "original_expansion_trace_identities": 0,
        "consumed_outer_identities": 0,
        "consumed_v9_outer_identities": 0,
        "consumed_v10_outer_identities": 0,
        "formal_xy_identities": 0,
        "previous_development_identities": 0,
        "retired_exposed_identities": 0,
        "union_candidate_identities_excluded": 0,
    }
    registry = {
        "bindings": {
            "failure_closeout_file_sha256": _fp("failed"),
            "consumed_v9_attempt_closeout_file_sha256": _fp("v9"),
            "burned_v10_final_closeout_file_sha256": _fp("v10"),
        },
        "selected_outer_identities": selected,
        "statistics": {
            "remaining_candidate_scene_count": len(remaining),
            "remaining_family_counts": remaining_counts,
            "selected_outer_scene_count": len(selected),
            "selected_outer_family_counts": subject.FAMILY_QUOTAS,
            "hash_screen": hash_screen,
        },
        "exclusion_counts": exclusion_counts,
        "exclusion_digests": {
            "excluded_seeds_sha256": digest([]),
            "excluded_scene_fingerprints_sha256": digest([]),
            "source_row_scene_fingerprints_sha256": digest([]),
            "original_expansion_trace_fingerprints_sha256": digest([]),
            "consumed_outer_identities_sha256": digest([]),
            "consumed_v9_outer_identities_sha256": digest([]),
            "consumed_v10_outer_identities_sha256": digest([]),
            "consumed_v10_outer_observation_hashes_sha256": consumed_v10_hash,
            "retired_exposed_fingerprints_sha256": digest([]),
        },
    }
    paths = {name: tmp_path / name for name in subject._CONFIG_PATH_FIELDS}
    result = subject._exposure_closure(paths=paths, registry=registry)
    assert result[0] == remaining
    assert result[4] is closure
    assert {row["fingerprint"] for row in selected} <= result[2]


def test_real_source_closure_is_nonempty_and_self_bound():
    sources = subject.producer_sources()
    relative = "backend/training/warehouse_r41_diagnostic_final_materializer_v9.py"
    assert relative in sources
    assert sources[relative] == sha256(
        Path(subject.__file__).read_bytes()).hexdigest()
    for required in (
        "backend/training/warehouse_r41_diagnostic_outer_collection_v11.py",
        "backend/training/warehouse_r41_diagnostic_outer_hash_projection_v11.py",
        "backend/training/warehouse_r41_diagnostic_rcpd_v11_outer_once.py",
        "backend/training/warehouse_r41_diagnostic_rcpd_v11_outer_split.py",
        "backend/training/warehouse_r41_diagnostic_final_attempt_closeout_v11.py",
    ):
        assert required in sources
