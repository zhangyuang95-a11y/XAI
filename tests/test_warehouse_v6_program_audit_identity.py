"""Program audit receipts must name the exact sealed episode sequence."""

from copy import deepcopy

from experiments.warehouse.acceptance import _program_audit


def test_audit_rejects_extra_or_reordered_episode_ids():
    seeds = (240100, 240101)
    role_hashes = {role: "a" * 64 for role in ("robot_1", "robot_2")}
    program_hashes = {role: "b" * 64 for role in role_hashes}
    metric = {
        "coverage": 0.5,
        "actionable_coverage": 0.4,
        "agreement": 0.98,
        "kl": 0.04,
        "illegal_count": 0,
        "rows": 1000,
        "actionable_rows": 800,
        "sample_count": 2000,
    }
    audits = {role: {
        "role": role,
        "actor_role_sha256": role_hashes[role],
        "program_sha256": program_hashes[role],
        "episode_ids": [str(seed) for seed in seeds],
        "episode_count": len(seeds),
        "sample_count": 2000,
        "max_depth": 5,
        "leaf_count": 16,
        "max_features_per_node": 3,
        "feedback_eligible": True,
        "explanation_eligible": True,
        "feedback": metric,
        "explanation": metric,
    } for role in role_hashes}
    kwargs = dict(role_hashes=role_hashes, program_hashes=program_hashes,
                  expected_seeds=seeds)
    assert _program_audit(audits, **kwargs)

    duplicated = deepcopy(audits)
    duplicated["robot_1"]["episode_ids"].append(str(seeds[-1]))
    assert not _program_audit(duplicated, **kwargs)

    reversed_ids = deepcopy(audits)
    reversed_ids["robot_2"]["episode_ids"].reverse()
    assert not _program_audit(reversed_ids, **kwargs)
