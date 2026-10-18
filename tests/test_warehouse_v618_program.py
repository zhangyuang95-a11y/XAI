"""Independent extraction cannot obtain fidelity by modifying the NN."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training.warehouse_v6_actor import FEATURE_NAMES, FEATURE_DIM
from backend.training.warehouse_v6_program import (
    INDEPENDENT_SELECTIVE_METHOD, SELECTIVE_PROGRAM_VERSION,
    WarehouseV6Program, _metrics, audit_frozen_program,
    certified_program_actions, current_actor_fidelity, fit_program,
)
from backend.training.warehouse_v618_selective_tree import (
    fit_independent_structure, masked_probabilities, source_feature_cost,
)
from env.warehouse.navigation import ACTIONS
from tests.test_warehouse_v616_joint_actor import _encoded_rows


def _soft_actor():
    # A frozen complete distribution with considerable non-argmax mass. This
    # catches fitting only top1, using Actor leaf boxes, and fake coverage.
    target = np.array([.08, .07, .12, .13, .60], np.float32)
    role = SimpleNamespace(
        feature_dim=FEATURE_DIM, feature_names=FEATURE_NAMES,
        role_sha256="6"*64,
        weights={"protected_leaf_mask": np.ones(16, np.float32),
                 "tree.leaf_logits": np.full((16, 5), 40., np.float32)},
        logits_encoded=lambda x: np.broadcast_to(np.log(target), (len(x), 5)).copy(),
    )
    physical = _encoded_rows()[0, 141:146]
    masked_target = masked_probabilities(np.log(target)[None], physical[None])[0]
    return SimpleNamespace(roles={"robot_1": role},
                           role_sha256={"robot_1": role.role_sha256}), masked_target


def _parts():
    encoded = _encoded_rows()[0]
    obs, mask = encoded[:141].copy(), encoded[141:146].copy()
    groups = []
    for block in range(4):
        rows = []
        for episode in range(10):
            for frame in range(24):
                rows.append({"obs": obs.copy(), "mask": mask.copy(),
                             "episode": f"split:{block}:episode:{episode}",
                             "base_seed": 900000+100*block+episode,
                             "frame": frame,
                             "sampled_action": "WAIT" if frame < 15 else "RIGHT"})
        groups.append(rows)
    return groups


def test_independent_program_fits_full_soft_distribution_and_never_uses_actor_boxes():
    actor, target = _soft_actor()
    parts = _parts()
    before = {key: value.copy() for key, value in actor.roles["robot_1"].weights.items()}
    program = fit_program("robot_1", actor, *parts, method=INDEPENDENT_SELECTIVE_METHOD)
    assert program.report["fit_method"] == INDEPENDENT_SELECTIVE_METHOD
    assert program.report["fit_tree"]["topology_origin"] == "independent_fit_only_sparse_search"
    assert len(program.report["fit_tree"]["search_runs"]) == 3
    assert program.report["validation"]["development_feedback_passed"]
    metric = program.report["validation"]["feedback"]
    assert metric["coverage"] == 1.
    assert metric["actionable_coverage"] == 15/24
    assert metric["expected_actionable_coverage"] == pytest.approx(float(target[-1]), abs=.002)
    assert metric["agreement"] == 1.
    assert metric["kl"] < .001
    assert metric["physical_frame_count"] == len(parts[-1])
    np.testing.assert_allclose(list(program.lookup(parts[-1][0])["probabilities"].values()),
                               target, atol=.002)
    for key, value in before.items():
        np.testing.assert_array_equal(actor.roles["robot_1"].weights[key], value)
    serialized = program.to_dict()
    assert serialized["version"] == SELECTIVE_PROGRAM_VERSION
    restored = WarehouseV6Program.from_dict(serialized,
        actor_role_sha256=actor.roles["robot_1"].role_sha256)
    assert restored.sha256() == program.sha256()


def test_validation_does_not_select_topology_or_eligible_leaves_and_keeps_repeated_frames():
    actor, _ = _soft_actor()
    parts = _parts()
    first = fit_program("robot_1", actor, *parts, method=INDEPENDENT_SELECTIVE_METHOD)
    changed = [{**row, "sampled_action": "RIGHT"} for row in parts[-1]]
    second = fit_program("robot_1", actor, *parts[:-1], changed,
                         method=INDEPENDENT_SELECTIVE_METHOD)
    assert first.report["fit_tree"] == second.report["fit_tree"]
    np.testing.assert_array_equal(first.leaf_logits, second.leaf_logits)
    assert first.feedback_leaves == second.feedback_leaves
    assert second.report["validation"]["feedback"]["coverage"] == 1.
    assert second.report["validation"]["feedback"]["actionable_coverage"] == 0.
    assert not second.report["validation"]["feedback_passed"]
    broken = [{**row} for row in parts[-1]]
    del broken[3]
    with pytest.raises(ValueError, match="omitted physical frames"):
        fit_program("robot_1", actor, *parts[:-1], broken,
                    method=INDEPENDENT_SELECTIVE_METHOD)


def test_episode_and_base_scene_isolation_is_required():
    actor, _ = _soft_actor()
    parts = _parts()
    overlap = [{**row, "episode": "new:"+row["episode"],
                "base_seed": parts[0][0]["base_seed"]} for row in parts[-1]]
    with pytest.raises(ValueError, match="scenario seed splits overlap"):
        fit_program("robot_1", actor, *parts[:-1], overlap,
                    method=INDEPENDENT_SELECTIVE_METHOD)


def test_guard_cost_is_searched_even_when_all_fit_movements_are_legal():
    n = 240
    encoded = np.zeros((n, FEATURE_DIM), np.float32)
    masks = np.ones((n, 5), np.float32)
    for action, name in enumerate(ACTIONS):
        encoded[:, FEATURE_NAMES.index(f"legal_{name.lower()}")] = masks[:, action]
    encoded[:, FEATURE_NAMES.index("self_row")] = np.tile(np.arange(24)/24, 10)
    preferred = np.where(encoded[:, FEATURE_NAMES.index("self_row")] < .5, 2, 3)
    logits = np.zeros((n, 5), np.float32)
    logits[np.arange(n), preferred] = 4.
    target = masked_probabilities(logits, masks)
    episodes = [f"fit:{index//24}" for index in range(n)]
    structure, report = fit_independent_structure(encoded, masks, target, episodes,
        feature_names=FEATURE_NAMES, actor_role_sha256="b"*64)
    program_report = {"fit_method": INDEPENDENT_SELECTIVE_METHOD,
                      "leaf_count": structure["leaf_count"], "max_depth": 5}
    provisional = WarehouseV6Program("robot_1", "b"*64, structure["root_ref"],
        structure["split_weights"], structure["split_bias"],
        structure["left_refs"], structure["right_refs"],
        structure["initial_leaf_logits"], frozenset(), frozenset(), program_report)
    _, leaf_count, depth = provisional._structure()
    assert leaf_count <= 16 and depth <= 5
    assert report["guard_capacity_included_in_search"]
    certificates = certified_program_actions(provisional)
    assignments = provisional.leaf_indices(encoded)
    certified = np.array([int(provisional.leaf_logits[leaf].argmax()) in certificates[leaf]
                          for leaf in assignments])
    assert certified.mean() >= .9
    assert np.any([not np.any(assignments == leaf) for leaf in range(leaf_count)])
    assert all(source_feature_cost(weight, FEATURE_NAMES) <= 3
               for weight in provisional.split_weights)


def test_refitted_action_requires_its_own_new_path_proof():
    # A RIGHT guard cannot certify a refitted LEFT output, even when LEFT
    # happened to be legal in every supplied fitting example.
    w = np.zeros((1, FEATURE_DIM), np.float32)
    w[0, FEATURE_NAMES.index("legal_right")] = 1.
    z = np.zeros((2, 5), np.float32)
    z[:, ACTIONS.index("WAIT")] = 1.
    z[1, ACTIONS.index("RIGHT")] = 4.
    report = {"fit_method": INDEPENDENT_SELECTIVE_METHOD,
              "max_depth": 1, "leaf_count": 2,
              "max_features_per_node": 1}
    original = WarehouseV6Program("robot_1", "c"*64, 0, w,
        np.array([-.5], np.float32), np.array([-1]), np.array([-2]), z,
        frozenset({1}), frozenset({1}), report)
    original = WarehouseV6Program.from_dict(original.to_dict(), actor_role_sha256="c"*64)
    changed = z.copy()
    changed[1, ACTIONS.index("LEFT")] = 8.
    with pytest.raises(RuntimeError, match="lacks path legality proof"):
        WarehouseV6Program.from_dict(replace(original, leaf_logits=changed).to_dict(),
                                     actor_role_sha256="c"*64)


def test_independent_current_fidelity_requires_distribution_not_actor_tree_identity():
    actor, _ = _soft_actor()
    parts = _parts()
    program = fit_program("robot_1", actor, *parts, method=INDEPENDENT_SELECTIVE_METHOD)
    receipt = current_actor_fidelity(program, actor, parts[-1])
    assert receipt["passed"]
    audit = [{**row, "episode": "audit:"+row["episode"],
              "base_seed": row["base_seed"]+10000} for row in parts[-1]]
    assert audit_frozen_program(program, actor, audit)["passed"]
    actor.roles["robot_1"].logits_encoded = lambda x: np.tile([8., 0., 0., 0., 0.], (len(x), 1))
    assert not current_actor_fidelity(program, actor, parts[-1])["passed"]


def test_offline_topology_adapter_recognizes_new_actor_tensor_schema():
    from backend.training.warehouse_v6_actor import (
        FrozenWarehouseV6Actor, WarehouseV6Actor,
    )
    from backend.training.warehouse_v614_certificate_tree import (
        FeatureSchema, frozen_actor_topology,
    )
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    role = actor.roles["robot_1"]
    assert "leaf_bound_logits" not in role.weights
    topology = frozen_actor_topology(role.weights,
        schema=FeatureSchema(tuple(role.feature_names), tuple(ACTIONS)),
        frozen_actor_sha256=role.role_sha256)
    assert not topology.core_mask.any()
    with pytest.raises(ValueError, match="role hash mismatch"):
        frozen_actor_topology(role.weights,
            schema=topology.schema, frozen_actor_sha256="e"*64)


def test_selective_program_roundtrips_and_audits_real_legacy_npz(tmp_path):
    """New extraction can diagnose an old frozen NN without promoting it."""
    from copy import deepcopy
    from hashlib import sha256
    import json
    import torch
    from backend.training.warehouse_v6_actor import (
        FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor, _role_arrays,
        role_sha256, V617_MODEL_VERSION, LEGACY_MODEL_VERSION,
        LEGACY_CORE_BOUND_MODE, _LEGACY_TENSOR_NAMES,
        MIN_RESIDUAL_BOUND, MAX_RESIDUAL_BOUND, CORE_LOW_GATE_BOUND,
        GEOMETRY_DIM,
    )
    from env.warehouse.public_action_features_v614 import ACTOR_FEATURE_SCHEMA_SHA256
    template_path = tmp_path / "template.npz"
    export_actor(WarehouseV6Actor(), template_path, pair_seed=1,
                 variant="raw", candidate_id="legacy-unit-template")
    with np.load(template_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
    old_model = WarehouseV6Actor(model_version=V617_MODEL_VERSION)
    with torch.no_grad():
        for role in old_model.actors.values():
            role.tree.leaf_logits.zero_()
            for parameter in role.action_residual.parameters():
                parameter.zero_()
            role.action_residual[0].weight[0, FEATURE_DIM+GEOMETRY_DIM+4] = 1.
            role.action_residual[2].weight[0, 0] = 1.
            role.action_residual[4].weight[0, 0] = 4.
    arrays, hashes = {}, {}
    for name, role in old_model.actors.items():
        weights = _role_arrays(role)
        weights["core_leaf_mask"] = weights.pop("protected_leaf_mask")
        weights["tree.split.weight"] = weights["tree.split.weight"][:, :171]
        for key in ("gate.0.weight", "action_residual.0.weight"):
            weights[key] = np.concatenate((weights[key][:, :171], weights[key][:, 173:]), axis=1)
        assert set(weights) == set(_LEGACY_TENSOR_NAMES)
        hashes[name] = role_sha256(weights, model_version=LEGACY_MODEL_VERSION)
        arrays.update({f"{name}.{key}": value for key, value in weights.items()})
    metadata.update(model_version=LEGACY_MODEL_VERSION, core_bound_mode=LEGACY_CORE_BOUND_MODE,
                    feature_dim=171, feature_schema_sha256=ACTOR_FEATURE_SCHEMA_SHA256,
                    role_sha256=hashes, minimum_residual_bound=MIN_RESIDUAL_BOUND,
                    maximum_residual_bound=MAX_RESIDUAL_BOUND, core_low_gate_bound=CORE_LOW_GATE_BOUND)
    path = tmp_path / "historical-v614.npz"
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
    frozen = FrozenWarehouseV6Actor.load(path, expected_sha256=sha256(path.read_bytes()).hexdigest(),
                                          expected_role_sha256=hashes)
    assert frozen.model_version == LEGACY_MODEL_VERSION
    assert frozen.roles["robot_1"].feature_dim == 171
    parts = _parts()
    for name in ("robot_1", "robot_2"):
        program = fit_program(name, frozen, *parts, method=INDEPENDENT_SELECTIVE_METHOD)
        assert program.feature_dim == 171
        assert program.feature_schema_sha256 == ACTOR_FEATURE_SCHEMA_SHA256
        program_path = tmp_path / f"{name}.program.json"
        artifact_hash = program.save(program_path)
        loaded = WarehouseV6Program.load(program_path, expected_sha256=artifact_hash,
                                        actor_role_sha256=hashes[name])
        assert loaded.sha256() == program.sha256()
        rows = [{**row, "base_seed": row["base_seed"]+10000,
                 "episode": "audit:"+row["episode"]} for row in parts[-1]]
        audit = audit_frozen_program(loaded, frozen, rows)
        assert audit["feature_schema_sha256"] == ACTOR_FEATURE_SCHEMA_SHA256
        assert audit["actor_role_sha256"] == hashes[name]
        assert audit["sample_count"] == len(rows)
        assert _metrics(loaded, frozen.roles[name], rows, set(loaded.feedback_leaves))["physical_frame_count"] == len(rows)
        tampered = deepcopy(loaded.to_dict())
        from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256
        tampered["feature_schema_sha256"] = FEATURE_SCHEMA_SHA256
        with pytest.raises(RuntimeError, match="schema|complexity"):
            WarehouseV6Program.from_dict(tampered, actor_role_sha256=hashes[name])
        with pytest.raises(RuntimeError, match="Actor hash mismatch"):
            WarehouseV6Program.load(program_path, expected_sha256=artifact_hash,
                                    actor_role_sha256="f"*64)
    # Supporting old frozen diagnostics never reopens their export path.
    with pytest.raises(ValueError, match="Only the current"):
        export_actor(old_model, tmp_path / "forbidden.npz", pair_seed=1,
                     variant="raw", candidate_id="not-promoted")
