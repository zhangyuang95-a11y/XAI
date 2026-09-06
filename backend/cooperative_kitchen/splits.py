"""Disjoint scenario identities for optimization, selection and final audits."""
import hashlib
import json

BASES = {"train": 10_000, "validation": 1_000_000, "extraction_fit": 2_000_000,
         "extraction_selection": 3_000_000, "extraction_test": 4_000_000,
         "final_test": 5_000_000, "calibration": 6_000_000, "questionnaire": 7_000_000}


def seeds(split, count=60, seed=0):
    if split not in BASES or not 0 < count < 100_000 or not 0 <= seed < 9:
        raise ValueError("Invalid kitchen split allocation")
    base = BASES[split] + seed * 100_000
    return range(base, base + count)


def scenario_fingerprint(public):
    """Exclude arbitrary IDs: renamed copies of a scenario are not independent."""
    content = {key: public[key] for key in ("map", "actors", "pot", "counters", "orders", "maxSteps", "targetOrders")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
