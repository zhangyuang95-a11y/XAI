from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

from backend.training import warehouse_r41_diagnostic_rcpd_v11_outer_split as subject
from backend.training.warehouse_native_common import digest


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def test_remaining_population_excludes_v10_quota_again():
    rows = []
    excluded_seeds, excluded_fingerprints = set(), set()
    remaining = subject.EXPECTED_REMAINING_FAMILY_COUNTS
    for family_index, family in enumerate(subject.FAMILY_IDS):
        for offset in range(360):
            seed = family_index * 10_000 + offset
            fingerprint = _fp(f"{family}-{offset}")
            row = {"batch_index": 0, "family_id": family,
                   "family_offset": offset, "seed": seed,
                   "fingerprint": fingerprint}
            rows.append(row)
            if offset >= remaining[family]:
                excluded_seeds.add(seed)
                excluded_fingerprints.add(fingerprint)
    retained, selected, counts = subject._remaining_and_selected(
        rows, excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    assert len(retained) == 1567
    assert counts == subject.EXPECTED_REMAINING_FAMILY_COUNTS
    assert len(selected) == 64


def test_prior_projection_unions_v8_v9_v10():
    def closeout(label: str, values: list[str]) -> dict:
        projection = {
            "outer_observation_hashes": values,
            "source_projection_content_sha256": _fp(label + "-projection"),
            "content_sha256": _fp(label + "-projection"),
        }
        return {"content_sha256": _fp(label),
                "consumed_outer": {"observation_hash_projection": projection}}
    a, b, c = _fp("a"), _fp("b"), _fp("c")
    value = subject._prior_validation_wins_projection(
        old_closeout=closeout("v8", sorted([a, b])),
        attempt_closeout=closeout("v9", sorted([b, c])),
        final_closeout=closeout("v10", [c]))
    assert value["outer_observation_hashes"] == sorted([a, b, c])
    assert value["component_unique_counts"] == {
        "consumed_v8": 2, "consumed_v9": 2, "consumed_v10": 1}
    assert value["content_sha256"] == digest({
        key: child for key, child in value.items() if key != "content_sha256"})


def test_hash_screen_rejects_ranked_overlap(monkeypatch):
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
    monkeypatch.setattr(subject.v9.manifest_binding, "build_runtime",
                        lambda **_: object())
    monkeypatch.setattr(subject, "_materialize_scene",
                        lambda row, index: {**deepcopy(row), "id": str(index)})

    def collect(_runtime, scenes, **_kwargs):
        value = forbidden if scenes[0]["seed"] == blocked_seed else _fp("clear")
        return [{"observation": value}], 1

    monkeypatch.setattr(subject.projection_v10.rows_v7, "_collect", collect)
    monkeypatch.setattr(subject.projection_v10.rows_v7.legacy, "_obs_hash",
                        lambda value: value)
    closure = {
        "burned_final_closeout": {"consumed_outer": {
            "observation_hash_projection": {
                "outer_observation_hashes": [forbidden]}}},
        "paths": {"actor": "actor", "protocol": "protocol"},
        "screening_manifest": "manifest",
    }
    selected, _scenes, audit = subject._hash_screened_selection(
        remaining=candidates, closure=closure)
    assert selected[0]["seed"] != blocked_seed
    assert audit["rejected_v10_observation_overlap_scene_count"] == 1
    assert audit["selected_v10_observation_overlap"] == 0
