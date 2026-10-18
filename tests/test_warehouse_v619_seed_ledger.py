import json

from scripts import audit_warehouse_v619_seed_ledger as ledger


def test_only_explicit_seed_values_and_ranges_are_extracted():
    seeds, intervals = ledger.extract({"mean_deliveries": 6153, "seed": 99,
        "stage_seeds": {"role": [100, 101]}, "seed_sha256": "102",
        "reserved_intervals": [{"start": 500004, "end_exclusive": 10500204}]})
    assert seeds == {99, 100, 101}
    assert intervals == {(500004, 10500204)}


def test_failed_claim_is_spent_and_new_declarations_excluded(tmp_path):
    root = tmp_path / "warehouse"
    root.mkdir()
    claim = root / "failed-sealed-claim.json"
    claim.write_text(json.dumps({"status": "failed", "seed": 100266001}))
    (root / "warehouse-v619-new-plan.json").write_text('{"seed":100266002}')
    assert ledger.classify(claim) == "claim_counts_as_spent_regardless_status"
    assert list(ledger.candidates(root)) == [claim]


def test_low_expanded_scenes_overlap_old_reservation_high_scenes_do_not():
    low, high = ledger.proposed_families(False), ledger.proposed_families(True)
    assert any(500004 <= seed < 10500204 for seed in low["fixed_fit:task_expanded"])
    assert not any(500004 <= seed < 10500204 for seed in high["fixed_fit:task_expanded"])
    assert max(high["fixed_verification:task_expanded"]) < 2**31
    assert len(high["fixed_verification:rng_contiguous_backup"]) == 216


def test_current_hash_rng_has_collision_with_legacy_reserved_interval():
    high = ledger.proposed_families(True)
    assert {seed for seed in high["fixed_verification:rng_proposal"] if 500004 <= seed < 10500204} == {3854737, 7269167, 8059535}
