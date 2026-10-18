from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_warehouse_v619_cached_comparison as c


def toy_cache():
    cache = {}
    for split_index, split in enumerate(c.SPLITS):
        cache.update({
            f"{split}.obs": np.zeros((3, 2, 4), np.float32),
            f"{split}.mask": np.ones((3, 2, 5), np.float32),
            f"{split}.episode_ids": np.array([f"episode-{split_index}"] * 3),
            f"{split}.scenario_seeds": np.full(3, 5000 + split_index, np.int64),
            f"{split}.frames": np.arange(3),
            f"{split}.environment_episode_ids": np.full(3, split_index),
            f"{split}.sampled_actions": np.array([[0, 1], [2, 3], [4, 0]]),
            f"{split}.labels": np.full((3, 2), 99),  # Never an NN target.
        })
    return cache


def restore(cache):
    return c.restore_cache(cache, counts={split: 3 for split in c.SPLITS})


def test_all_frames_retained_with_role_actions_and_no_teacher_targets():
    cache = toy_cache()
    before = {key: c.array_digest(value) for key, value in cache.items()}
    parts, manifest = restore(cache)
    assert [row["sampled_action"] for row in parts["fit"]["robot_1"]] == ["UP", "LEFT", "WAIT"]
    assert [row["sampled_action"] for row in parts["fit"]["robot_2"]] == ["DOWN", "RIGHT", "UP"]
    assert all("label" not in row and "target" not in row for rows in parts.values() for group in rows.values() for row in group)
    assert all(item["physical_frames_per_role"] == 3 and item["excluded_frames"] == 0 for item in manifest.values())
    assert before == {key: c.array_digest(value) for key, value in cache.items()}


@pytest.mark.parametrize("mutation", ["omitted", "duplicate", "reordered", "cross_purpose", "illegal_sample", "fractional_mask"])
def test_invalid_full_frame_cache_is_rejected(mutation):
    cache = toy_cache()
    if mutation == "omitted":
        cache["validation.frames"][-1] = 3
    elif mutation == "duplicate":
        cache["validation.frames"][-1] = 1
    elif mutation == "reordered":
        cache["validation.frames"][:] = [0, 2, 1]
    elif mutation == "cross_purpose":
        cache["validation.scenario_seeds"][:] = cache["fit.scenario_seeds"]
    elif mutation == "illegal_sample":
        cache["validation.mask"][0, 0, 0] = 0
    else:
        cache["validation.mask"][0, 0, 1] = .5
    with pytest.raises(RuntimeError):
        restore(cache)


def test_topology_only_is_copied_without_old_leaf_outputs():
    program = SimpleNamespace(root_ref=0, split_weights=np.ones((1, 173)),
        split_bias=np.zeros(1), left_refs=np.array([-1]), right_refs=np.array([-2]),
        leaf_logits=np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]))
    structure = c.structure_only(program)
    assert set(structure) == {"root_ref", "split_weights", "split_bias", "left_refs", "right_refs", "leaf_count"}
    assert structure["split_weights"].dtype == np.float32
    assert structure["left_refs"].dtype == np.int64
    structure["split_weights"][0, 0] = 8
    assert program.split_weights[0, 0] == 1


def test_structure_sha_ignores_old_logits_but_detects_changed_split():
    from backend.training.warehouse_v6_program import program_structure_sha256
    structure = {"root_ref": 0, "split_weights": np.zeros((1, 173), np.float32),
        "split_bias": np.zeros(1), "left_refs": [-1], "right_refs": [-2], "leaf_count": 2}
    structure["split_weights"][0, 0] = 1
    before = program_structure_sha256(structure)
    assert program_structure_sha256({**structure, "initial_leaf_logits": [[1]*5]*2}) == before
    structure["split_bias"][0] = .5
    assert program_structure_sha256(structure) != before


def test_full_frame_and_complete_distribution_identity_detect_mutation():
    cache = toy_cache()
    _, old = restore(cache)
    cache["validation.obs"][2, 1, 3] = 1
    _, changed = restore(cache)
    assert old["validation"]["full_frame_sha256"] != changed["validation"]["full_frame_sha256"]
    with pytest.raises(RuntimeError, match="identity changed"):
        c.ensure_identity({"source": "a", "target": "p"}, {"source": "a", "target": "q"})


def test_p0_reproduction_and_global_illegal_scope_are_strict():
    metric = deepcopy(c.P0_EXPECTED["robot_1"])
    c.check_p0("robot_1", metric)
    metric["whole_tree_raw_illegal_count"] = 0
    with pytest.raises(RuntimeError, match="whole_tree_raw_illegal_count"):
        c.check_p0("robot_1", metric)


def test_solver_count_reports_failed_candidate_and_search_separately():
    report = {"constrained_leaf_fits": {"0": {"candidate_reports": [{"success": True}, {"success": False}]}},
        "fit_tree": {"actual_convex_solver_calls": 31, "rejected_solver_candidates": 2}}
    result = c.solver_counts(report)
    assert result["final_leaf_action_solves"] == 2
    assert result["final_leaf_unsuccessful_action_solves"] == 1
    assert result["topology_search"]["actual_convex_solver_calls"] == 31


def test_exclusive_receipts_cannot_overwrite_previous_results(tmp_path):
    path = tmp_path / "result.json"
    c.write_exclusive(path, {"publishable": False})
    with pytest.raises(FileExistsError):
        c.write_exclusive(path, {"publishable": True})
