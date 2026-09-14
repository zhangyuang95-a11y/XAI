from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect

from backend.training import (
    warehouse_r41_diagnostic_final_candidate_universe_v14 as subject,
)
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def test_public_generator_is_balanced_deterministic_and_excludes_history(
    monkeypatch,
):
    monkeypatch.setattr(subject, "FAMILY_IDS", ("a", "b"))
    monkeypatch.setattr(subject, "SEED_STARTS", (100,))
    monkeypatch.setattr(subject, "BATCH_MARKERS", (14,))
    monkeypatch.setattr(subject, "PER_FAMILY_PER_BATCH", 2)
    monkeypatch.setattr(subject, "MAXIMUM_DRAWS_PER_BATCH", 20)

    def make_scene(scene_id, split, seed, *, batch_index, candidate_start):
        family = ("a", "b")[seed % 2]
        return {
            "id": scene_id, "split": split, "seed": seed,
            "fingerprint": _fp(f"scene-{seed}"), "family_id": family,
            "batch_index": batch_index, "snapshot": {"seed": seed},
        }

    monkeypatch.setattr(subject.scenes_api, "make_scene", make_scene)
    first = subject._generate_candidate_batches(
        excluded_seeds={100}, excluded_fingerprints={_fp("scene-101")})
    second = subject._generate_candidate_batches(
        excluded_seeds={100}, excluded_fingerprints={_fp("scene-101")})
    assert first == second
    batches, reports = first
    assert len(batches) == 1
    assert len(batches[0]) == 4
    assert {row["seed"] for row in batches[0]}.isdisjoint({100, 101})
    assert reports[0]["per_family"] == {"a": 2, "b": 2}


def test_closeout_companions_are_file_and_content_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(subject.closeout_api, "IDENTITY_NAME", "identity.json")
    monkeypatch.setattr(subject.closeout_api, "PROJECTION_NAME", "projection.json")
    ranked = []
    for index in range(2160):
        ranked.append({
            "batch_index": index // 720,
            "family_offset": index,
            "family_id": subject.FAMILY_IDS[index % len(subject.FAMILY_IDS)],
            "seed": index,
            "fingerprint": _fp(f"ranked-{index}"),
        })
    identity = {
        "ranking_input_identities": ranked,
        "ranking_input_identities_sha256": digest(ranked),
        "historically_excluded_seeds": [1, 3],
        "historically_excluded_seeds_sha256": digest([1, 3]),
        "historically_excluded_fingerprints": sorted([
            _fp("historical-a"), _fp("historical-b")]),
    }
    identity["historically_excluded_fingerprints_sha256"] = digest(
        identity["historically_excluded_fingerprints"])
    identity["content_sha256"] = digest(identity)
    projection = {"unique_observation_hashes": sorted([
        _fp("observation-a"), _fp("observation-b")])}
    projection["unique_observation_hashes_sha256"] = digest(
        projection["unique_observation_hashes"])
    projection["content_sha256"] = digest(projection)
    identity_path = tmp_path / "identity.json"
    projection_path = tmp_path / "projection.json"
    identity_path.write_text(canonical(identity) + "\n")
    projection_path.write_text(canonical(projection) + "\n")
    closeout_path = tmp_path / "closeout_receipt.json"
    closeout_path.write_text("{}\n")
    closeout = {"bindings": {
        "burned_candidate_universe_file_sha256": file_hash(identity_path),
        "burned_candidate_universe_content_sha256": identity["content_sha256"],
        "burned_observation_hashes_file_sha256": file_hash(projection_path),
        "burned_observation_hashes_content_sha256": projection[
            "content_sha256"],
    }}
    result = subject._closeout_exclusions(
        closeout, closeout_path=closeout_path)
    assert result["ranked_identities_sha256"] == digest(ranked)
    assert result["burned_unique_observation_count"] == 2
    changed = deepcopy(closeout)
    changed["bindings"]["burned_observation_hashes_file_sha256"] = "0" * 64
    try:
        subject._closeout_exclusions(changed, closeout_path=closeout_path)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered closeout companion accepted")


def test_information_boundary_excludes_policy_and_salt_inputs():
    value = subject.contract()
    assert value["program_access"] is False
    assert value["program_predictions_access"] is False
    assert value["action_labels_access"] is False
    assert value["actor_probabilities_access"] is False
    assert value["private_salt_access"] is False
    source = inspect.getsource(subject)
    assert "PRIVATE_SALT" not in source
    assert "actor_probs" not in source
    assert "action_indices" not in source


def test_candidate_identity_requires_frozen_five_fields():
    value = {
        "batch_index": 14, "family_offset": 0,
        "family_id": subject.FAMILY_IDS[0], "seed": 58_100_000,
        "fingerprint": _fp("new"),
    }
    assert subject._candidate_identity(value, "test") == value
    changed = deepcopy(value)
    changed["extra"] = True
    try:
        subject._candidate_identity(changed, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("expanded candidate identity accepted")


def test_saved_universe_accepts_lossless_json_coordinate_normalization(
    tmp_path, monkeypatch,
):
    saved = {
        "version": subject.VERSION,
        "status": subject.STATUS,
        "contract": subject.contract(),
        "candidate_scenes": [{"route": [[1, 2], [2, 2]]}],
        "formal_ready": False,
    }
    saved["content_sha256"] = digest(saved)
    path = tmp_path / "candidate_universe.json"
    path.write_text(canonical(saved) + "\n", encoding="utf-8")
    recreated = deepcopy(saved)
    recreated["candidate_scenes"][0]["route"] = [(1, 2), (2, 2)]
    monkeypatch.setattr(subject, "create_universe", lambda **_kwargs: recreated)

    result = subject.read_saved_universe(
        path,
        expected_universe_sha256=file_hash(path),
        timeout_closeout_path=tmp_path / "unused.json",
        expected_timeout_closeout_sha256="0" * 64,
        permanent_timeout_closeout_registry=tmp_path,
    )
    assert result == saved
