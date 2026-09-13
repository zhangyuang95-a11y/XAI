from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect

from backend.training import warehouse_r41_diagnostic_rcpd_v12_outer_split as subject
from backend.training.warehouse_native_common import digest


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def test_remaining_population_excludes_v11_quota_again():
    rows = []
    excluded_seeds, excluded_fingerprints = set(), set()
    for family_index, family in enumerate(subject.FAMILY_IDS):
        for offset in range(360):
            seed = family_index * 10_000 + offset
            fingerprint = _fp(f"{family}-{offset}")
            row = {"batch_index": 0, "family_id": family,
                   "family_offset": offset, "seed": seed,
                   "fingerprint": fingerprint}
            rows.append(row)
            if offset >= subject.EXPECTED_REMAINING_FAMILY_COUNTS[family]:
                excluded_seeds.add(seed)
                excluded_fingerprints.add(fingerprint)
    retained, selected, counts = subject._remaining_and_selected(
        rows, excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    assert len(retained) == 1503
    assert counts == subject.EXPECTED_REMAINING_FAMILY_COUNTS
    assert len(selected) == 64


def test_prior_projection_unions_v8_through_v11():
    def closeout(label: str, values: list[str]) -> dict:
        projection = {
            "outer_observation_hashes": values,
            "source_projection_content_sha256": _fp(label + "-projection"),
            "content_sha256": _fp(label + "-projection"),
        }
        return {"content_sha256": _fp(label),
                "consumed_outer": {"observation_hash_projection": projection}}

    a, b, c, d = (_fp("a"), _fp("b"), _fp("c"), _fp("d"))
    value = subject._prior_validation_wins_projection(
        old_closeout=closeout("v8", sorted([a, b])),
        attempt_closeout=closeout("v9", sorted([b, c])),
        final_closeout=closeout("v10", sorted([c, d])),
        consumed_v11_closeout=closeout("v11", sorted([a, d])))
    assert value["outer_observation_hashes"] == sorted([a, b, c, d])
    assert value["component_unique_counts"] == {
        "consumed_v8": 2, "consumed_v9": 2,
        "consumed_v10": 2, "consumed_v11": 2}
    assert value["source_v11_closeout_content_sha256"] == _fp("v11")
    assert value["selector_rule"] == (
        "remove fit rows matching any prior outer observation hash before "
        "reading actions, probabilities, or raw observations")
    assert value["content_sha256"] == digest({
        key: child for key, child in value.items()
        if key != "content_sha256"})


def test_hash_screen_rejects_ranked_v11_overlap(monkeypatch):
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
    monkeypatch.setattr(subject.v11.v9.manifest_binding, "build_runtime",
                        lambda **_: object())
    monkeypatch.setattr(subject, "_materialize_scene",
                        lambda row, index: {**deepcopy(row), "id": str(index)})

    def collect(_runtime, scenes, **_kwargs):
        value = forbidden if scenes[0]["seed"] == blocked_seed else _fp("clear")
        return [{"observation": value}], 1

    monkeypatch.setattr(subject.v11.projection_v10.rows_v7, "_collect", collect)
    monkeypatch.setattr(subject.v11.projection_v10.rows_v7.legacy, "_obs_hash",
                        lambda value: value)
    closure = {
        "consumed_v11_closeout": {"consumed_outer": {
            "observation_hash_projection": {
                "outer_observation_hashes": [forbidden]}}},
        "paths": {"actor": "actor", "protocol": "protocol"},
        "screening_manifest": "manifest",
    }
    selected, _scenes, audit = subject._hash_screened_selection(
        remaining=candidates, closure=closure)
    assert selected[0]["seed"] != blocked_seed
    assert audit["rejected_v11_observation_overlap_scene_count"] == 1
    assert audit["selected_v11_observation_overlap"] == 0


def test_information_boundary_has_no_final_or_private_salt_access():
    boundary = subject.RECOVERY_BOUNDARY
    assert boundary["consumed_v11_outer_closed_before_selection"] is True
    assert boundary["consumed_v11_identities_excluded"] is True
    assert boundary["consumed_v11_rows_opened"] is False
    assert boundary["protected_final_access"] is False
    assert boundary["private_salt_access"] is False
    source = inspect.getsource(subject)
    assert "fresh_final_holdout" not in source

