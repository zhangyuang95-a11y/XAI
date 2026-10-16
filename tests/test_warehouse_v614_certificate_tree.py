"""Synthetic and static tests only: no Warehouse environment or reset."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
from backend.training.warehouse_v614_certificate_tree import (
    CertificateSearchConfig, FeatureSchema, MAX_DEPTH, MAX_LEAVES,
    MAX_NODE_FEATURES, SparseProgramTree, fit_certificate_tree,
    refit_frozen_topology_fit_only, _masked_probabilities,
)
from backend.training.warehouse_v6_tree_routing import (
    certified_feature_bounds, certified_mask_branches,
)


ACTOR_SHA = "a" * 64
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")


def schema_for(width: int) -> FeatureSchema:
    if width == 146:
        observation_count, derived_count = 141, 0
    elif width == 171:
        observation_count, derived_count = 141, 25
    else:
        raise AssertionError("test schema width")
    names = [f"public_{index}" for index in range(observation_count)]
    names[-3:] = ["same_target_collision_streak_capped",
                   "robots_shortest_distance",
                   "self_charger_return_margin"]
    names += [f"legal_{action.lower()}" for action in ACTIONS]
    names += [f"derived_public_{index}" for index in range(derived_count)]
    return FeatureSchema(tuple(names), ACTIONS)


def synthetic_fit(width: int):
    schema = schema_for(width)
    episode_ids = tuple(f"synthetic-fit:{index // 10}" for index in range(300))
    x = np.zeros((300, width), dtype=np.float32)
    x[:, 0] = np.where(np.arange(300) % 10 < 5, 1.0, -1.0)
    x[:, schema.feature_names.index("robots_shortest_distance")] = 0.5
    x[:, schema.feature_names.index("self_charger_return_margin")] = 0.5
    masks = np.zeros((300, len(ACTIONS)), dtype=np.float32)
    masks[:, 0] = (x[:, 0] > 0).astype(np.float32)
    masks[:, 4] = 1.0
    x[:, schema.mask_feature_indices] = masks
    if width == 171:
        # A derived feature deliberately duplicates the mask.  It still does
        # not become a structural legality certificate.
        x[:, -1] = masks[:, 0]
    target = np.zeros_like(masks)
    target[:, 0] = 0.90 * masks[:, 0]
    target[:, 4] = 1.0 - target[:, 0]
    return schema, x, masks, target, episode_ids


class CertificateTreeTests(unittest.TestCase):
    def test_oblique_public_condition_proves_physical_legality(self) -> None:
        names = list(schema_for(171).feature_names)
        names[0] = "self_column"
        schema = FeatureSchema(tuple(names), ACTIONS)
        weights = np.zeros((MAX_LEAVES - 1, schema.feature_dim), dtype=np.float32)
        weights[0, schema.mask_feature_indices[0]] = 1.0
        weights[0, names.index("self_column")] = 1.0
        bias = np.zeros(MAX_LEAVES - 1, dtype=np.float32)
        bias[0] = np.float32(-1.4)
        left = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
        right = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
        left[0], right[0] = -1, -2
        logits = np.zeros((MAX_LEAVES, len(ACTIONS)), dtype=np.float32)
        logits[1, 0] = 2.0
        core = np.zeros(MAX_LEAVES, dtype=np.bool_)
        core[1] = True
        candidate = SparseProgramTree(schema, weights, bias, left, right, 1, 2,
                                      logits, core, {}, frozenset())
        proof = certified_mask_branches(
            weights[0], float(bias[0]), schema.mask_feature_indices,
            certified_feature_bounds(schema.feature_names))
        self.assertIn(0, proof[1])
        for mask in (0.0, 1.0):
            for coordinate in np.linspace(0.0, 1.0, 101):
                row = np.zeros((1, schema.feature_dim), dtype=np.float32)
                row[0, schema.mask_feature_indices[0]] = mask
                row[0, names.index("self_column")] = coordinate
                if candidate.leaf_indices(row)[0] == 1:
                    self.assertEqual(mask, 1.0)
        out_of_domain = np.zeros((1, schema.feature_dim), dtype=np.float32)
        out_of_domain[0, names.index("self_column")] = 1.1
        with self.assertRaisesRegex(ValueError, "outside proven domain"):
            candidate.leaf_indices(out_of_domain)
        names[1] = "charger_return_margin_if_up"
        margin_schema = FeatureSchema(tuple(names), ACTIONS)
        lower, upper = certified_feature_bounds(margin_schema.feature_names)[1]
        self.assertEqual((lower, upper), (-2.0, 1.0))
        guarded_margin = np.zeros(margin_schema.feature_dim, dtype=np.float32)
        guarded_margin[margin_schema.mask_feature_indices[0]] = -0.7 - lower
        guarded_margin[1] = -1.0
        margin_proof = certified_mask_branches(
            guarded_margin, lower, margin_schema.mask_feature_indices,
            certified_feature_bounds(margin_schema.feature_names))
        self.assertIn(0, margin_proof[1])
        # An unconstrained public value cannot stand in for a physical mask.
        weights[0, names.index("self_column")] = 0.0
        weights[0, names.index("public_2")] = 1.0
        with self.assertRaisesRegex(ValueError, "legality proof"):
            SparseProgramTree(schema, weights, bias, left, right, 1, 2,
                              logits, core, {}, frozenset())

    def test_dynamic_mask_indices_and_complete_distribution(self) -> None:
        for width in (146, 171):
            with self.subTest(width=width):
                schema, x, masks, target, episodes = synthetic_fit(width)
                self.assertEqual(schema.mask_feature_indices,
                                 tuple(range(141, 146)))
                fitted = fit_certificate_tree(
                    x, masks, target, episodes, schema=schema,
                    frozen_actor_sha256=ACTOR_SHA,
                    config=CertificateSearchConfig(
                        beam_width=1, max_search_leaves=3,
                        max_axis_features=3,
                        max_exact_candidates_per_round=8,
                        max_exact_candidates_per_leaf=2))
                self.assertLessEqual(fitted.report["maximum_depth"], MAX_DEPTH)
                self.assertLessEqual(fitted.leaf_count, MAX_LEAVES)
                self.assertLessEqual(
                    fitted.report["maximum_features_per_node"], MAX_NODE_FEATURES)
                self.assertGreaterEqual(fitted.report["fit_core_mass"], 0.35)
                self.assertTrue(fitted.report["topology_uses_fit_only"])
                self.assertFalse(fitted.report["calibration_eligibility_validation_used"])
                self.assertTrue(np.any(fitted.core_mask[:fitted.leaf_count]))
                leaves = fitted.leaf_indices(x)
                legal_up = np.flatnonzero(masks[:, 0] > 0)
                up_leaves = np.unique(leaves[legal_up])
                self.assertEqual(len(up_leaves), 1)
                up_leaf = int(up_leaves[0])
                self.assertTrue(fitted.core_mask[up_leaf])
                self.assertEqual(int(np.argmax(fitted.leaf_logits[up_leaf])), 0)
                q = _masked_probabilities(np.broadcast_to(
                    fitted.leaf_logits[up_leaf], target[legal_up].shape),
                    masks[legal_up])
                self.assertAlmostEqual(float(np.mean(q[:, 0])), 0.90, delta=0.03)
                gate_weights = fitted.split_weights[:fitted.node_count]
                self.assertTrue(any(
                    np.flatnonzero(weight).tolist() == [schema.mask_feature_indices[0]]
                    for weight in gate_weights))

                # The final frozen Actor may reuse exactly this topology, but
                # independent fit rows must refit program leaves afresh.
                with self.assertRaisesRegex(ValueError, "topology identity"):
                    refit_frozen_topology_fit_only(
                        fitted, x, masks, target, episodes,
                        frozen_actor_sha256=ACTOR_SHA,
                        expected_topology_sha256=fitted.topology_sha256())
                frozen_topology = SparseProgramTree(
                    schema, fitted.split_weights, fitted.split_bias,
                    fitted.left_refs, fitted.right_refs,
                    fitted.node_count, fitted.leaf_count,
                    fitted.leaf_logits,
                    np.zeros(MAX_LEAVES, dtype=np.bool_),
                    {"source": "final_frozen_full_actor_tree_topology",
                     "frozen_actor_sha256": ACTOR_SHA}, frozenset())
                refitted = refit_frozen_topology_fit_only(
                    frozen_topology, x, masks, target, episodes,
                    frozen_actor_sha256=ACTOR_SHA,
                    expected_topology_sha256=frozen_topology.topology_sha256())
                self.assertEqual(refitted.topology_sha256(),
                                 frozen_topology.topology_sha256())
                self.assertGreaterEqual(refitted.report["fit_core_mass"], 0.35)
                with self.assertRaisesRegex(ValueError, "topology identity"):
                    refit_frozen_topology_fit_only(
                        frozen_topology, x, masks, target, episodes,
                        frozen_actor_sha256=ACTOR_SHA,
                        expected_topology_sha256="0" * 64)

    def test_derived_public_feature_cannot_certify_move(self) -> None:
        schema = schema_for(171)
        weights = np.zeros((MAX_LEAVES - 1, schema.feature_dim), dtype=np.float32)
        weights[0, -1] = 1.0
        bias = np.zeros(MAX_LEAVES - 1, dtype=np.float32)
        bias[0] = -0.5
        left = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
        right = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
        left[0], right[0] = -1, -2
        logits = np.zeros((MAX_LEAVES, len(ACTIONS)), dtype=np.float32)
        logits[1, 0] = 2.0
        core = np.zeros(MAX_LEAVES, dtype=np.bool_)
        core[1] = True
        with self.assertRaisesRegex(ValueError, "legality proof"):
            SparseProgramTree(schema, weights, bias, left, right, 1, 2,
                              logits, core, {}, frozenset())
        weights[0, -1] = 0.0
        weights[0, schema.mask_feature_indices[0]] = 1.0
        SparseProgramTree(schema, weights, bias, left, right, 1, 2,
                          logits, core, {}, frozenset())

    def test_sparse_low_return_stratum_is_routed_to_residual(self) -> None:
        schema, x, masks, target, episodes = synthetic_fit(171)
        return_index = schema.feature_names.index("self_charger_return_margin")
        rare = np.flatnonzero(masks[:, 0] > 0)[:5]
        x[rare, return_index] = 0.02
        fitted = fit_certificate_tree(
            x, masks, target, episodes, schema=schema,
            frozen_actor_sha256=ACTOR_SHA,
            config=CertificateSearchConfig(
                beam_width=2, max_search_leaves=4,
                max_axis_features=3,
                max_exact_candidates_per_round=12,
                max_exact_candidates_per_leaf=5))
        leaves = fitted.leaf_indices(x)
        self.assertTrue(np.all(~fitted.core_mask[leaves[rare]]))
        self.assertGreaterEqual(fitted.report["fit_core_mass"], 0.35)
        self.assertTrue(any(
            np.flatnonzero(weight).tolist() == [return_index]
            for weight in fitted.split_weights[:fitted.node_count]))

    def test_rejects_wrong_mask_binding_and_target_identity(self) -> None:
        schema, x, masks, target, episodes = synthetic_fit(171)
        x_bad = x.copy()
        x_bad[:, schema.mask_feature_indices[0]] = 1.0 - masks[:, 0]
        with self.assertRaisesRegex(ValueError, "full masked fit target"):
            fit_certificate_tree(x_bad, masks, target, episodes,
                                 schema=schema, frozen_actor_sha256=ACTOR_SHA)
        with self.assertRaisesRegex(ValueError, "full frozen NN"):
            fit_certificate_tree(x, masks, target, episodes, schema=schema,
                                 frozen_actor_sha256=ACTOR_SHA,
                                 target_provenance="public_teacher_one_hot")

    def test_four_purpose_adapter_is_static_and_execution_is_guarded(self) -> None:
        core_path = (ROOT / "backend/training/warehouse_v614_certificate_tree.py")
        adapter_path = (ROOT / "backend/training/warehouse_v614_masked_refit.py")
        runner_path = (ROOT / "backend/training/warehouse_v6_runner.py")
        for path in (core_path, adapter_path):
            ast.parse(path.read_text())
        core_text = core_path.read_text()
        adapter_text = adapter_path.read_text()
        self.assertNotIn(".reset(", core_text)
        self.assertNotIn(".reset(", adapter_text)
        self.assertIn('required = ("fit", "calibration", "eligibility", "validation")',
                      adapter_text)
        self.assertIn("_scenario_set(groups[name])", adapter_text)
        self.assertIn("refit_frozen_topology_fit_only(", adapter_text)
        self.assertIn("expected_topology_sha256=expected_topology_sha256", adapter_text)
        runner = ast.parse(runner_path.read_text())
        self.assertIn("warehouse_v614_masked_refit", runner_path.read_text())
        guard = next(node for node in runner.body if isinstance(node, ast.Assign)
                     and any(isinstance(name, ast.Name) and
                             name.id == "FORMAL_TRAINING_DISABLED"
                             for name in node.targets))
        self.assertIs(guard.value.value, True)
        for name in ("_pretrain_pair_production", "pretrain_pair", "train_pair"):
            function = next(node for node in runner.body
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertIsInstance(function.body[1], ast.Expr)
            self.assertEqual(function.body[1].value.func.id,
                             "_require_locked_execution")

    def test_warmup_kl_does_not_reverse_teacher_gradient(self) -> None:
        import torch

        runner = ast.parse((ROOT / "backend/training/warehouse_v6_runner.py").read_text())
        weight = next(node.value.value for node in runner.body
                      if isinstance(node, ast.Assign) and
                      any(isinstance(name, ast.Name) and
                          name.id == "BC_REFIT_WARMUP_KL_WEIGHT"
                          for name in node.targets))
        self.assertEqual(weight, 0.0)
        def wrong_action_gradient(coefficient: float) -> float:
            logits = torch.tensor([[-2.0, 2.0]], requires_grad=True)
            target = torch.tensor([[0.9, 0.1]])
            teacher = torch.tensor([1])
            ce = torch.nn.functional.cross_entropy(logits, teacher)
            kl = torch.sum(target * (torch.log(target) -
                                     torch.log_softmax(logits, dim=-1)))
            (ce + coefficient * kl).backward()
            return float(logits.grad[0, 0])
        self.assertLess(wrong_action_gradient(1.0), 0.0)
        self.assertGreater(wrong_action_gradient(weight), 0.0)

    def test_review_guard_and_three_distinct_data_purposes(self) -> None:
        from backend.training import warehouse_v6_runner as runner

        self.assertTrue(runner.FORMAL_TRAINING_DISABLED)
        with self.assertRaisesRegex(RuntimeError, "review-only"):
            runner._environment(training_reward=False)
        with self.assertRaisesRegex(RuntimeError, "review-only"):
            runner.train_pair({})
        self.assertEqual(sum(count for _, count in
                             runner.BC_TOPOLOGY_STAGE_COUNTS.values()), 50)
        self.assertEqual(sum(count for _, count in
                             runner.BC_FINAL_CORE_STAGE_COUNTS.values()), 124)
        stages = [stage for stage, _ in (
            *runner.BC_TOPOLOGY_STAGE_COUNTS.values(),
            *runner.BC_FINAL_CORE_STAGE_COUNTS.values())]
        self.assertEqual(len(stages), len(set(stages)))
        # Dummy non-registered seed: verifies only the allocator's structure.
        blocks = [runner._bc_stage_seeds(1, stage, count)
                  for stage, count in (
                      *runner.BC_TOPOLOGY_STAGE_COUNTS.values(),
                      *runner.BC_FINAL_CORE_STAGE_COUNTS.values())]
        flattened = [seed for block in blocks for seed in block]
        self.assertEqual(len(flattened), 174)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(json.loads(json.dumps(runner.v614_executable_recipe())), json.loads((
            ROOT / "configs/warehouse_v614_exploratory_recipe.json").read_text()))

    def test_study_physics_binding_rejects_other_valid_configs(self) -> None:
        from dataclasses import replace
        from env.warehouse.domain import participant_study_config
        from env.warehouse.public_action_features_v614 import (
            ACTION_PHYSICS_CONFIG_SHA256, TREE_FEATURE_DIM, TREE_MASK_START,
            validate_study_action_physics,
        )

        config = participant_study_config()
        self.assertEqual(TREE_FEATURE_DIM, 171)
        self.assertEqual(TREE_MASK_START, 141)
        self.assertEqual(validate_study_action_physics(config),
                         ACTION_PHYSICS_CONFIG_SHA256)
        for invalid in (replace(config, move_battery_cost=3.0),
                        replace(config, charge_per_wait=7.0)):
            with self.assertRaisesRegex(ValueError, "action physics"):
                validate_study_action_physics(invalid)

    def test_reset_journal_records_exact_successful_order_without_environment(self) -> None:
        from env.warehouse.reset_audit_v614 import (
            capture_stage1_resets, record_successful_reset,
            stage1_reset_phase,
        )

        with tempfile.TemporaryDirectory(prefix="v614-audit-static-") as directory:
            journal = Path(directory) / "resets.jsonl"
            with capture_stage1_resets(journal) as rows:
                with stage1_reset_phase("bc"):
                    record_successful_reset(101, 1, "study-layout", 2.0, 10.0)
                with stage1_reset_phase("quality"):
                    record_successful_reset(102, 1, "study-layout", 2.0, 10.0)
            written = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(rows, written)
            self.assertEqual([(row["phase"], row["seed"]) for row in written],
                             [("bc", 101), ("quality", 102)])


if __name__ == "__main__":
    unittest.main()
