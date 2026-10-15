from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from backend.training import warehouse_r41_diagnostic_rcpd_v10_outer_split as subject
from backend.training.warehouse_native_common import digest


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _identity(index: int, family: str) -> dict:
    return {
        "batch_index": index,
        "family_offset": index,
        "family_id": family,
        "seed": 800_000 + index,
        "fingerprint": _fp(f"candidate-{index}"),
    }


def test_replacement_selection_excludes_all_consumed_v9_identities(
        tmp_path, monkeypatch):
    candidates = []
    consumed = []
    for family_index, family in enumerate(subject.FAMILY_IDS):
        first = _identity(family_index * 2, family)
        second = _identity(family_index * 2 + 1, family)
        candidates.extend((first, second))
        consumed.append(subject._public_identity(first))
    excluded_seeds = {row["seed"] for row in consumed}
    excluded_fingerprints = {row["fingerprint"] for row in consumed}
    monkeypatch.setattr(subject, "FIXED_CANDIDATE_SCENE_COUNT", 12)
    monkeypatch.setattr(subject, "EXPECTED_REMAINING_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "EXPECTED_REMAINING_FAMILY_COUNTS",
                        {family: 1 for family in subject.FAMILY_IDS})
    monkeypatch.setattr(subject, "FAMILY_QUOTAS",
                        {family: 1 for family in subject.FAMILY_IDS})
    monkeypatch.setattr(subject, "FRESH_OUTER_SCENE_COUNT", 6)
    remaining, selected, counts = subject._remaining_and_selected(
        candidates, excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    assert len(remaining) == len(selected) == 6
    assert counts == {family: 1 for family in subject.FAMILY_IDS}
    assert not ({row["seed"] for row in selected} & excluded_seeds)
    assert not ({row["fingerprint"] for row in selected}
                & excluded_fingerprints)


def test_fixed_contract_has_new_salt_and_exact_remaining_population():
    assert subject.SELECTION_SALT == (
        "warehouse-r41-v10-fresh-development-outer-20260914-v1")
    assert subject.EXPECTED_REMAINING_SCENE_COUNT == 1631
    assert subject.EXPECTED_REMAINING_FAMILY_COUNTS == {
        "conflict_family_01": 268,
        "conflict_family_02": 269,
        "conflict_family_03": 271,
        "conflict_family_04": 268,
        "conflict_family_05": 277,
        "conflict_family_06": 278,
    }
    contract = subject.contract()
    assert contract["consumed_v9_outer_permanently_excluded"] is True
    assert contract["selection_uses_consumed_v9_rows"] is False
    assert contract["selection_uses_prior_validation_wins_projection"] is False


def test_prior_projection_is_union_of_v8_and_consumed_v9_hashes():
    v8_hashes = sorted((_fp("shared"), _fp("v8-only")))
    v9_hashes = sorted((_fp("shared"), _fp("v9-only")))
    old_projection = {
        "outer_observation_hashes": v8_hashes,
        "content_sha256": _fp("v8-projection"),
    }
    old_closeout = {
        "content_sha256": _fp("v8-closeout"),
        "consumed_outer": {"observation_hash_projection": old_projection},
    }
    new_projection = {
        "outer_observation_hashes": v9_hashes,
        "source_projection_content_sha256": _fp("v9-projection"),
    }
    new_closeout = {
        "content_sha256": _fp("v9-closeout"),
        "consumed_outer": {"observation_hash_projection": new_projection},
    }
    result = subject._prior_validation_wins_projection(
        old_closeout=old_closeout, attempt_closeout=new_closeout)
    expected = sorted(set(v8_hashes) | set(v9_hashes))
    assert result["outer_observation_hashes"] == expected
    assert result["outer_observation_hashes_sha256"] == digest(expected)
    assert result["selection_used_this_projection"] is False
    assert result["raw_observations_included"] is False
    assert result["actions_included"] is False
    assert result["probabilities_included"] is False
    assert result["labels_included"] is False


def test_materialised_scene_uses_v10_identity(monkeypatch):
    identity = {
        "batch_index": 3,
        "family_id": subject.FAMILY_IDS[0],
        "seed": 7,
        "fingerprint": _fp("scene"),
    }

    class FakeEnv:
        def __init__(self):
            self.state = type("State", (), {"tasks": object()})()

    monkeypatch.setattr(subject, "R41DiagnosticConflictWarehouseEnv", FakeEnv)
    monkeypatch.setattr(subject.v9.retired_api, "_identity_only_reset",
                        lambda env, *, seed: None)
    monkeypatch.setattr(subject, "conflict_family_id",
                        lambda tasks: identity["family_id"])
    monkeypatch.setattr(subject.v9.scenes_api, "_candidate_start_state",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(subject, "diagnostic_scene_fingerprint",
                        lambda env: identity["fingerprint"])
    monkeypatch.setattr(
        subject.v9.scenes_api, "_scene_from_environment",
        lambda env, *, scene_id, split, seed, batch_index: {
            "id": scene_id,
            "split": split,
            "seed": seed,
            "batch_index": batch_index,
            "family_id": identity["family_id"],
            "fingerprint": identity["fingerprint"],
        })
    scene = subject._materialize_scene(identity, 4)
    assert scene["id"] == "diagnostic_v10_fresh_outer_0004"
    assert scene["split"] == "development_outer"


def test_source_does_not_open_consumed_v9_rows_or_protected_final():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "np.load" not in source
    assert "v9_collection_rows" not in source
    assert "fresh_final_holdout" not in source
