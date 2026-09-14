from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_final_materializer_v14 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _claim(tmp_path: Path, sources=None):
    sources = subject.producer_sources() if sources is None else sources
    inputs = {
        "scheme": subject.FINAL_ONCE_VERSION + ".candidate-and-outer.v1",
        "candidate_lock_sha256": _fp("lock-file"),
        "candidate_lock_content_sha256": _fp("lock-content"),
        "actor_sha256": _fp("actor"), "protocol_sha256": _fp("protocol"),
        "runtime_manifest_sha256": _fp("manifest"),
        "designation_sha256": _fp("designation"),
        "program_sha256": _fp("program"),
        "public_feature_contract_sha256": _fp("features"),
        "candidate_source_closure_sha256": _fp("candidate-sources"),
        "outer_result_sha256": _fp("outer-result"),
        "outer_attempt_key": _fp("outer-attempt"),
        "final_projection_source_closure_sha256": digest(
            subject.final_projection_api.producer_sources()),
        "final_projection_contract_sha256": digest(
            subject.final_projection_api.contract()),
        "private_salt_commitment": subject.PRIVATE_SALT_COMMITMENT,
        "private_salt_domain_sha256": sha256(
            subject.PRIVATE_SALT_DOMAIN).hexdigest(),
        "timeout_closeout_sha256": _fp("timeout-closeout-file"),
        "timeout_closeout_content_sha256": _fp("timeout-closeout-content"),
        "candidate_universe_sha256": _fp("candidate-universe-file"),
        "candidate_universe_content_sha256": _fp(
            "candidate-universe-content"),
        "candidate_universe_identity_sha256": _fp(
            "candidate-universe-identities"),
        "timing_calibration_sha256": _fp("timing-calibration-file"),
        "timing_calibration_content_sha256": _fp(
            "timing-calibration-content"),
    }
    key = digest(inputs)
    campaign = tmp_path / key
    campaign.mkdir()
    bindings = {
        "outer_result_content_sha256": _fp("outer-content"),
        "candidate_runtime_source_closure_sha256": _fp("runtime-sources"),
        "final_controller_source_closure_sha256": _fp("controller-sources"),
        "final_materializer_source_closure_sha256": digest(sources),
        "actor_feature_names_sha256": _fp("actor-features"),
        "final_projection_source_closure_sha256": inputs[
            "final_projection_source_closure_sha256"],
        "final_projection_contract_sha256": inputs[
            "final_projection_contract_sha256"],
        "private_salt_commitment": subject.PRIVATE_SALT_COMMITMENT,
        "private_salt_domain_sha256": inputs["private_salt_domain_sha256"],
        "timeout_closeout_content_sha256": inputs[
            "timeout_closeout_content_sha256"],
        "candidate_universe_content_sha256": inputs[
            "candidate_universe_content_sha256"],
        "candidate_universe_identity_sha256": inputs[
            "candidate_universe_identity_sha256"],
        "timing_calibration_content_sha256": inputs[
            "timing_calibration_content_sha256"],
    }
    value = {
        "version": subject.ANCHOR_VERSION, "status": subject.ANCHOR_STATUS,
        "attempt_key": key, "attempt_key_inputs": inputs,
        "bindings": bindings,
        "candidate_and_outer_authenticated_before_claim": True,
        "final_identity_or_rows_accessed_before_claim": False,
        "retry_allowed": False, "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    path = campaign / "attempt_anchor.json"
    path.write_text(canonical(value) + "\n")
    return path, value


def _scene(family: str, index: int) -> dict:
    return {
        "id": f"source-{family}-{index}", "split": "play_candidates",
        "seed": 100_000 + subject.FAMILY_IDS.index(family) * 1000 + index,
        "fingerprint": _fp(f"{family}:{index}"), "family_id": family,
        "batch_index": index % 3, "snapshot": {"frame": 0},
    }


def _population(extra=2):
    scenes = [_scene(family, index) for family in subject.FAMILY_IDS
              for index in range(subject.FAMILY_QUOTAS[family] + extra)]
    identities = [{
        "batch_index": scene["batch_index"],
        "family_offset": index,
        "family_id": scene["family_id"],
        "seed": scene["seed"],
        "fingerprint": scene["fingerprint"],
    } for index, scene in enumerate(scenes)]
    return scenes, identities, {s["fingerprint"]: s for s in scenes}


def _projection(scene: dict, offset: int) -> dict:
    ordered = [_fp(f"obs:{scene['fingerprint']}:{offset}:{i}") for i in range(3)]
    summary = {
        "local_scene_index": 0, "scene_index": offset,
        "fingerprint": scene["fingerprint"], "row_count": len(ordered),
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_observation_count": len(set(ordered)),
        "unique_observation_hashes_sha256": digest(sorted(set(ordered))),
    }
    return {
        "ordered_observation_hashes": ordered,
        "unique_observation_hashes": sorted(set(ordered)),
        "environment_steps": 7, "scenes": [summary],
    }


def test_claim_is_v13_projection_and_salt_bound(tmp_path):
    path, expected = _claim(tmp_path)
    assert subject._authenticate_claim(path) == (path, expected)
    changed = deepcopy(expected)
    changed["attempt_key_inputs"]["private_salt_commitment"] = _fp("other")
    changed["attempt_key"] = digest(changed["attempt_key_inputs"])
    changed["content_sha256"] = digest({k: v for k, v in changed.items()
                                         if k != "content_sha256"})
    other = tmp_path / changed["attempt_key"]
    other.mkdir()
    bad = other / path.name
    bad.write_text(canonical(changed) + "\n")
    with pytest.raises(ValueError, match="claim semantics"):
        subject._authenticate_claim(bad)


def test_invalid_claim_prevents_config_or_salt_access(tmp_path, monkeypatch):
    bad = tmp_path / "attempt_anchor.json"
    bad.write_text("{}\n")
    touched = []
    monkeypatch.setattr(subject, "_load_config", lambda: touched.append("config"))
    monkeypatch.setattr(subject, "_read_committed_salt",
                        lambda _: touched.append("salt"))
    with pytest.raises(ValueError):
        subject.materialize(claim=bad, output=tmp_path / "materializer_output.json")
    assert touched == []


def test_private_salt_is_exact_32_byte_mode_0600_and_domain_bound(
        tmp_path, monkeypatch):
    raw = b"x" * 32
    path = tmp_path / "salt.bin"
    path.write_bytes(raw)
    path.chmod(0o600)
    monkeypatch.setattr(subject, "PRIVATE_SALT_PATH", path)
    monkeypatch.setattr(subject, "PRIVATE_SALT_COMMITMENT",
                        sha256(subject.PRIVATE_SALT_DOMAIN + raw).hexdigest())
    assert subject._read_committed_salt(path) == raw
    path.chmod(0o644)
    with pytest.raises(ValueError, match="commitment"):
        subject._read_committed_salt(path)


def test_exact_full_trajectory_projection_drives_selection(monkeypatch):
    _, identities, mapping = _population(extra=3)
    calls = []

    def project(runtime, scenes, *, scene_offset, dense_critical):
        assert runtime is not None and dense_critical is False
        calls.append((scenes[0]["fingerprint"], scene_offset))
        return _projection(scenes[0], scene_offset)

    monkeypatch.setattr(subject.final_projection_api,
                        "project_observation_hashes", project)
    monkeypatch.setattr(subject.final_projection_api,
                        "validate_projection", lambda value: value)
    selected, stats = subject._select_and_replay(
        runtime=SimpleNamespace(), identities=identities,
        scene_by_fingerprint=mapping, salt=b"private-test",
        excluded_seeds=set(), excluded_fingerprints=set(),
        forbidden_observation_hashes=set())
    assert len(selected) == subject.FINAL_SCENE_COUNT
    assert stats["families"] == dict(sorted(subject.FAMILY_QUOTAS.items()))
    projection = stats["projection"]
    assert projection["scene_count"] == subject.FINAL_SCENE_COUNT
    assert projection["actions_read"] is False
    assert projection["probabilities_read"] is False
    assert projection["program_access"] is False
    assert projection["labels_read"] is False
    assert calls[0][1] == subject.FINAL_SCENE_OFFSET
    assert {offset for _, offset in calls} == set(range(
        subject.FINAL_SCENE_OFFSET,
        subject.FINAL_SCENE_OFFSET + subject.FINAL_SCENE_COUNT))


def test_full_projection_rejects_prior_overlap(monkeypatch):
    _, identities, mapping = _population(extra=4)
    ordered = subject._ordered_candidates(
        identities, family=subject.FAMILY_IDS[0], salt=b"salt")
    blocked_scene = ordered[0]["fingerprint"]
    blocked = _fp("blocked")

    def project(_runtime, scenes, *, scene_offset, dense_critical):
        value = _projection(scenes[0], scene_offset)
        if scenes[0]["fingerprint"] == blocked_scene:
            value["ordered_observation_hashes"] = [blocked]
            value["unique_observation_hashes"] = [blocked]
            value["scenes"][0].update({
                "row_count": 1,
                "ordered_observation_hashes_sha256": digest([blocked]),
                "unique_observation_count": 1,
                "unique_observation_hashes_sha256": digest([blocked]),
            })
        return value

    monkeypatch.setattr(subject.final_projection_api,
                        "project_observation_hashes", project)
    monkeypatch.setattr(subject.final_projection_api,
                        "validate_projection", lambda value: value)
    selected, stats = subject._select_and_replay(
        runtime=object(), identities=identities,
        scene_by_fingerprint=mapping, salt=b"salt",
        excluded_seeds=set(), excluded_fingerprints=set(),
        forbidden_observation_hashes={blocked})
    assert blocked_scene not in {s["fingerprint"] for s in selected}
    assert stats["rejected"]["prior_public_observation_overlap"] == 1


def _child(values, exclusions, semantic):
    keep = ~np.isin(np.asarray(values, dtype="U64"),
                    np.asarray(exclusions, dtype="U64"))
    retained = [v for v, chosen in zip(values, keep) if chosen]
    packed = np.ascontiguousarray(keep.astype(np.uint8))
    value = {
        "source_rows": len(values), "retained_rows": len(retained),
        "removed_rows": int(np.sum(~keep)),
        "source_unique_observations": len(set(values)),
        "retained_unique_observations": len(set(retained)),
        "fresh_outer_unique_observations": len(exclusions),
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": sha256(memoryview(packed).cast("B")).hexdigest(),
        "retained_observation_hashes_sha256": digest(retained),
        "private_development_members_read_before_mask_frozen": False,
        "retained_rows_semantic_sha256": _fp(semantic),
        "all_retained_rows_marked_development": True,
        "private_members_read_only_after_mask_frozen": True,
        "source_archive_reauthenticated_after_private_read": True,
    }
    value["content_sha256"] = digest(value)
    return value, keep


def test_v13_six_component_prior_union_preserves_promoted_development():
    components = {_fp(f"prior-{i}") for i in range(6)}
    promoted = sorted({_fp("v11"), _fp("v12-outer"), _fp("v12-final")})
    historical = sorted(components)
    prior = sorted(set(historical) | set(promoted))
    fresh = [_fp("fresh-v13")]
    base = [_fp("base"), historical[0], promoted[0], fresh[0]]
    base_scenes = [_fp(f"base-scene-{i}") for i in range(len(base))]
    promoted_rows = promoted + [promoted[0]]
    promoted_scenes = [_fp(f"promoted-scene-{i}")
                       for i in range(len(promoted_rows))]
    base_audit, base_keep = _child(base, sorted(set(prior) | set(fresh)), "base")
    promoted_audit, promoted_keep = _child(
        promoted_rows, sorted(set(historical) | set(fresh)), "promoted")
    retained_scenes = {
        scene for scene, keep in zip(base_scenes, base_keep) if keep
    } | {
        scene for scene, keep in zip(promoted_scenes, promoted_keep) if keep
    }
    audit = {
        "precedence": (
            "fresh-v13-validation > combined-promoted-development > "
            "older-development"),
        "base": base_audit, "combined_promoted": promoted_audit,
        "prior_unique_observations": len(prior),
        "historical_unique_observations": len(historical),
        "combined_promoted_unique_observations": len(promoted),
        "fresh_v13_unique_observations": len(fresh),
        "combined_promoted_projection_sha256": digest(promoted),
        "combined_source_rows": len(base) + len(promoted_rows),
        "combined_retained_rows": int(np.sum(base_keep) + np.sum(promoted_keep)),
        "combined_retained_scene_count": len(retained_scenes),
        "cross_component_observation_overlap": 0,
        "retained_fresh_outer_observation_overlap": 0,
        "retained_rows_semantic_sha256": _fp("combined"),
        "all_retained_rows_marked_development": True,
        "private_members_read_only_after_both_masks_frozen": True,
        "both_source_archives_reauthenticated_after_private_read": True,
    }
    audit["content_sha256"] = digest(audit)
    retained, _ = subject._retained_combined_development_hashes(
        base_ordered=base, base_scene_ordered=base_scenes,
        promoted_ordered=promoted_rows,
        promoted_scene_ordered=promoted_scenes,
        prior_unique_hashes=prior, promoted_unique_hashes=promoted,
        fresh_unique_hashes=fresh, validation_wins=audit)
    assert retained == {_fp("base"), *promoted}
    assert subject._v13_outer_audit_hashes(
        prior_unique_hashes=prior, promoted_unique_hashes=promoted,
        fresh_unique_hashes=fresh) == set(historical) | set(fresh)


def test_exposure_closure_requires_burned_v12_inputs(monkeypatch):
    seen = {}
    candidates = [{"seed": 1, "fingerprint": _fp("candidate")}]

    def replay(**kwargs):
        seen.update(kwargs)
        return {"candidates": candidates, "excluded_seeds": {2},
                "excluded_fingerprints": {_fp("old")},
                "row_fingerprints": {_fp("row")}}

    monkeypatch.setattr(subject.registry_api, "replay_exclusion_closure", replay)
    paths = {name: Path("/") / name for name in subject._CONFIG_PATH_FIELDS}
    bindings = {
        "failure_closeout_file_sha256": _fp("f"),
        "consumed_v9_attempt_closeout_file_sha256": _fp("v9"),
        "burned_v10_final_closeout_file_sha256": _fp("v10"),
        "consumed_v11_outer_closeout_file_sha256": _fp("v11"),
        "burned_v12_final_closeout_file_sha256": _fp("v12"),
    }
    current = [{"seed": 3, "fingerprint": _fp("current")}
               for _ in range(subject.registry_api.FRESH_OUTER_SCENE_COUNT)]
    # Distinct values are unnecessary to this narrow forwarding test.
    result = subject._exposure_closure(
        paths=paths, registry={"bindings": bindings,
                              "selected_outer_identities": current})
    assert seen["burned_v12_final_closeout_path"] == paths[
        "burned_v12_final_closeout"]
    assert seen["permanent_v12_final_closeout_registry"] == paths[
        "permanent_v12_final_closeout_registry"]
    assert 3 in result[1] and _fp("current") in result[2]


def _burned_identity(index: int) -> dict:
    return {
        "batch_index": index // 720,
        "family_offset": index % 120,
        "family_id": f"conflict_family_{index % 6 + 1:02d}",
        "seed": 50_000_000 + index,
        "fingerprint": _fp(f"burned-identity:{index}"),
    }


def _burned_companions() -> tuple[dict, dict]:
    ranking = [_burned_identity(index) for index in range(2160)]
    outer, eligible = ranking[:64], ranking[64:]
    evaluated = [{
        **identity,
        "evaluation_index": index,
        "projector_scene_index": 900_000 + index,
        "accepted": True,
        "accepted_scene_index": index,
        "row_count": 1,
        "ordered_observation_hashes_sha256": digest([_fp(f"old:{index}")]),
        "unique_observation_count": 1,
        "unique_observation_hashes_sha256": digest([_fp(f"old:{index}")]),
        "environment_steps": 1,
    } for index, identity in enumerate(eligible[:64])]
    universe = {
        "version": subject.timeout_closeout_api.VERSION
            + ".burned-candidate-universe.v1",
        "status": "all_v13_ranking_inputs_and_completed_prefix_retired",
        "ranking_input_identities": ranking,
        "ranking_input_count": len(ranking),
        "ranking_input_identities_sha256": digest(ranking),
        "historically_excluded_seeds": sorted(row["seed"] for row in outer),
        "historically_excluded_seed_count": len(outer),
        "historically_excluded_seeds_sha256": digest(sorted(
            row["seed"] for row in outer)),
        "historically_excluded_fingerprints": sorted(
            row["fingerprint"] for row in outer),
        "historically_excluded_fingerprint_count": len(outer),
        "historically_excluded_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in outer)),
        "projector_eligible_identities": eligible,
        "projector_eligible_identity_count": len(eligible),
        "projector_eligible_identities_sha256": digest(eligible),
        "v13_outer_identities": outer,
        "v13_outer_identity_count": len(outer),
        "v13_outer_identities_sha256": digest(outer),
        "completed_evaluated_prefix": evaluated,
        "completed_evaluated_prefix_count": len(evaluated),
        "completed_evaluated_prefix_sha256": digest(evaluated),
        "reconstructed_accepted_identities": eligible[:64],
        "reconstructed_accepted_identity_count": 64,
        "reconstructed_accepted_identities_sha256": digest(eligible[:64]),
        "all_ranking_input_identities_retired": True,
        "all_completed_prefix_identities_retired": True,
        "same_v13_candidate_universe_reuse_permitted": False,
        "scene_snapshots_included": False,
        "rng_state_or_salt_included": False,
        "formal_ready": False,
    }
    universe["content_sha256"] = digest(universe)
    ordered = [_fp(f"burned-observation:{index}") for index in range(64)]
    projection = {
        "version": subject.timeout_closeout_api.VERSION
            + ".burned-observation-projection.v1",
        "status": "completed_v13_evaluated_prefix_hashes_permanently_excluded",
        "selection_algorithm_version": "retired-v13",
        "selection_algorithm_source_closure_sha256": _fp("selection"),
        "projector_version": subject.final_projection_api.VERSION,
        "projector_contract": subject.final_projection_api.contract(),
        "projector_contract_sha256": digest(
            subject.final_projection_api.contract()),
        "projector_sources": {"projector.py": _fp("projector")},
        "projector_sources_sha256": digest(
            {"projector.py": _fp("projector")}),
        "completed_evaluated_prefix_count": len(evaluated),
        "accepted_scene_count": 64,
        "ordered_observation_hashes": ordered,
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_observation_hashes": sorted(set(ordered)),
        "unique_observation_hashes_sha256": digest(sorted(set(ordered))),
        "row_count": len(ordered),
        "unique_observation_count": len(set(ordered)),
        "environment_steps": 64,
        "frozen_accepted_projection": {
            "scene_count": 64,
            "ordered_observation_hashes_sha256": digest(ordered),
        },
        "timing": {"elapsed_seconds": 550.0, "parallel_workers": 1},
        "information_boundary": {
            "closeout_claim_preceded_config_identity_and_salt_access": True,
            "burned_salt_used_only_for_exact_ranking_reconstruction": True,
            "salt_or_rng_state_included": False,
            "raw_observations_included": False,
            "actor_actions_included": False,
            "actor_probabilities_included": False,
            "program_file_or_predictions_accessed": False,
            "labels_or_rows_accessed_or_generated": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "formal_ready": False,
    }
    projection["content_sha256"] = digest(projection)
    return universe, projection


def test_timeout_closeout_receipt_summary_is_not_used_as_full_exclusion(
        tmp_path, monkeypatch):
    universe, projection = _burned_companions()
    # The closeout reader's own suite validates the complete frozen companion
    # schemas. This test isolates the materializer integration boundary: the
    # receipt contains summaries, while selection must receive the full files.
    monkeypatch.setattr(subject.timeout_closeout_api, "_validate_universe",
                        lambda value: value)
    monkeypatch.setattr(subject.timeout_closeout_api, "_validate_projection",
                        lambda value, _universe: value)
    artifact = tmp_path / "artifact"
    permanent = tmp_path / "permanent"
    campaign = permanent / _fp("closeout-key")
    artifact.mkdir(); campaign.mkdir(parents=True)
    universe_path = artifact / subject.timeout_closeout_api.IDENTITY_NAME
    projection_path = artifact / subject.timeout_closeout_api.PROJECTION_NAME
    universe_path.write_text(canonical(universe) + "\n")
    projection_path.write_text(canonical(projection) + "\n")
    (campaign / universe_path.name).write_bytes(universe_path.read_bytes())
    (campaign / projection_path.name).write_bytes(projection_path.read_bytes())
    closeout = {
        "closeout_claim_key": campaign.name,
        "burned_candidate_universe": {
            "ranking_input_count": 2160,
            "ranking_input_identities_sha256": universe[
                "ranking_input_identities_sha256"],
        },
        "burned_observation_projection": {
            "unique_observation_hashes_sha256": projection[
                "unique_observation_hashes_sha256"],
        },
        "bindings": {
            "burned_candidate_universe_file_sha256": file_hash(universe_path),
            "burned_candidate_universe_content_sha256": universe[
                "content_sha256"],
            "burned_observation_hashes_file_sha256": file_hash(projection_path),
            "burned_observation_hashes_content_sha256": projection[
                "content_sha256"],
        },
    }
    receipt_path = artifact / subject.timeout_closeout_api.RECEIPT_NAME
    receipt_path.write_text("receipt-summary-only\n")
    full_universe, full_projection = (
        subject._authenticate_timeout_closeout_companions(
            closeout_path=receipt_path, closeout=closeout,
            permanent_registry=permanent))
    assert len(full_universe["ranking_input_identities"]) == 2160
    assert full_projection["unique_observation_hashes"] == sorted(
        set(projection["ordered_observation_hashes"]))


def test_selection_source_is_target_and_program_blind():
    source = inspect.getsource(subject._select_and_replay)
    assert "screen_scene" not in source
    assert "submitted_actions" not in source
    assert "policy_actions" not in source
    assert '["probabilities"]' not in source
    assert '["action_indices"]' not in source
    assert ".predict" not in source
    assert "final_projection_api.project_observation_hashes" in source
    assert "program" not in subject._CONFIG_PATH_FIELDS
