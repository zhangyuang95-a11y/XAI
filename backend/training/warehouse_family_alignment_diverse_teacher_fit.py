"""Bounded TRAIN-only RCPD search over the diverse development collection.

Every RCPD candidate sees only the ``fit`` pool.  Candidate programs and fit
audits are durably saved before this module computes the first ``dev`` metric.
The separate development pool may select a program, but never changes a tree.
No Actor, PyTorch checkpoint, environment, PPO update, or new label is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from core.program import ExecutableProgram
from core.rcpd_tree import _tree_to_program

from . import warehouse_family_branch_teacher_fit as original
from . import warehouse_family_intervention_fit as pure
from .warehouse_native_common import ROOT, digest, file_hash, jsonable


VERSION = "warehouse-family-alignment-diverse-train-only-rcpd.v1"
COLLECTOR_MODULE = (
    "backend.training.warehouse_family_alignment_diverse_collection"
)
# (maximum depth, maximum leaves, changed-pair fit multiplier,
#  hard-action structure weight).  This tuple is the complete search space.
CANDIDATES = (
    (12, 256, 1.0, 0.0),
    (14, 512, 2.0, 0.5),
    (16, 1024, 4.0, 0.5),
    (18, 2048, 4.0, 1.0),
    (20, 4096, 4.0, 1.0),
)
FIT_SCENES = 192
DEV_SCENES = 64
BACKEND_VERSION = "fixed-capacity-rcpd-decision-tree-regressor.v1"
REPLAY_CHUNK_ROWS = 16_384
REPLAY_ATOL = 1e-6


def contract():
    return {
        "version": VERSION,
        "collector_module": COLLECTOR_MODULE,
        "development_pools": {"fit": FIT_SCENES, "dev": DEV_SCENES},
        "candidates": [list(candidate) for candidate in CANDIDATES],
        "candidate_fields": [
            "max_depth",
            "max_leaf_nodes",
            "counterfactual_changed_pair_weight",
            "action_structure_weight",
        ],
        "backend": BACKEND_VERSION,
        "fixed_capacity_rcpd_candidates": len(CANDIDATES),
        "maximum_rcpd_calls": 0,
        "maximum_sklearn_fits": len(CANDIDATES),
        "min_samples_leaf": 8,
        "regularization_lambda": 0.0,
        "importance_weight_scale": 8.0,
        "counterfactual_loss_weight": 0.2,
        "fit_presentation": (
            "one unique saved fit row weighted by "
            "(1+8*NN_top_two_margin)*physical_view_multiplicity; every "
            "NN-changing physical-pair endpoint is then multiplied by the "
            "candidate changed-pair weight"
        ),
        "targets": (
            "five saved NN probabilities plus, when configured, sqrt(action "
            "structure weight) times their argmax one-hot; exported leaves "
            "retain and normalize only the five soft outputs"
        ),
        "fit_scope": (
            "all development.fit rows only; development.dev is never supplied "
            "to DecisionTreeRegressor.fit"
        ),
        "dev_access": (
            "all candidate programs and fit audits are saved and hashed "
            "before the fitter first inspects development.dev"
        ),
        "gates_source": original.VERSION,
        "gates_modified": False,
        "minimum_fidelity": 0.90,
        "minimum_non_wait_fidelity": 0.85,
        "minimum_critical_fidelity": 0.85,
        "minimum_direction_fidelity": 0.85,
        "minimum_critical_scenarios": 10,
        "maximum_mean_kl": 0.35,
        "selection": (
            "lowest original normalized complexity loss among candidates "
            "passing every original ordinary/all-row/counterfactual/by-role gate"
        ),
        "failure_selection": (
            "diagnostic ranking only; no selected_index or program.json"
        ),
        "required_external_hashes": [
            "plan_file_sha256",
            "manifest_file_sha256",
            "fit_data_sha256",
            "dev_data_sha256",
        ],
        "labels": "saved frozen-NN five-action soft probabilities only",
        "new_NN_queries": 0,
        "Actor_loads": 0,
        "PT_loads": 0,
        "environment_steps": 0,
        "PPO_updates": 0,
        "optimizer_updates": 0,
        "feedback_admission_granted": False,
        "independent_acceptance_executed": False,
        "explanation_qualified": False,
        "release_ready": False,
    }


def config(candidate):
    candidate = tuple(candidate)
    if candidate not in CANDIDATES:
        raise ValueError("Candidate is outside the predeclared diverse search")
    depth, leaves, pair_weight, structure = candidate
    return pure.RCPDConfig(
        max_depth=depth,
        max_leaf_nodes=leaves,
        max_predicates=None,
        min_samples_leaf=8,
        complexity_penalty=0.001,
        random_seed=260910,
        regularization_lambda=0.0,
        action_structure_weight=structure,
        counterfactual_changed_pair_weight=pair_weight,
        importance_weight_scale=8.0,
        counterfactual_loss_weight=0.2,
    )


def _load_collector():
    """Delay the parallel collector dependency until data access is requested."""
    return importlib.import_module(COLLECTOR_MODULE)


def _read_development_header(collector, source, *, allow_test_fixture):
    """Validate collection metadata without opening any episode data file."""
    plan_path = source / "plan.json"
    plan = collector._load(plan_path)
    if plan.get("phase") != "development":
        raise ValueError("Only a development collection may be fitted")
    scenario_path = plan.get("input_files", {}).get("scenarios", {}).get("path")
    if not isinstance(scenario_path, str):
        raise ValueError("Frozen scenario input binding is missing")
    scenarios = collector._load(scenario_path)
    contexts = collector._validate_plan(
        plan,
        "development",
        scenarios,
        allow_test_fixture=allow_test_fixture,
    )
    manifest = collector._load(source / "manifest.json")
    if (
        manifest.get("version") != collector.VERSION
        or manifest.get("phase") != "development"
        or manifest.get("status") != "completed"
        or manifest.get("pending") is not None
        or manifest.get("plan_sha256") != digest(plan)
        or manifest.get("plan_file_sha256") != file_hash(plan_path)
        or manifest.get("actor_bindings") != plan.get("actor_bindings")
        or manifest.get("qualification_evaluated") is not False
        or manifest.get("independent_acceptance_executed") is not False
        or len(manifest.get("completed", [])) != len(contexts)
    ):
        raise ValueError("Only the complete acknowledged development collection may be read")
    return plan, manifest


def _read_completed_pool(collector, source, plan, manifest, pool):
    """Open and verify only the episode files assigned to one named pool."""
    if pool not in ("fit", "dev") or pool not in plan.get("phase_pools", ()):
        raise ValueError("Requested pool is outside the development plan")
    selected = [
        index
        for index, context in enumerate(plan["contexts"])
        if context.get("pool") == pool
    ]
    subset_plan = {
        **plan,
        "phase_pools": [pool],
        "contexts": [plan["contexts"][index] for index in selected],
    }
    subset_manifest = {
        **manifest,
        "completed": [manifest["completed"][index] for index in selected],
    }
    pools = collector._merge_completed(source, subset_plan, subset_manifest)
    if not isinstance(pools, dict) or set(pools) != {pool}:
        raise ValueError("Pool-specific collector merge returned an unexpected scope")
    return pools[pool]


def _exact_sha(value, name):
    return original._sha(value, name)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _source_hashes(collector):
    hashes = dict(original.execution_sources())
    paths = [Path(__file__).resolve()]
    collector_file = getattr(collector, "__file__", None)
    if collector_file is not None:
        paths.append(Path(collector_file).resolve())
    for path in paths:
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        hashes[name] = file_hash(path)
    return dict(sorted(hashes.items()))


def _bound_collection_files(plan):
    """Return exact paths/hashes declared by the completed collector plan."""
    result = []
    input_files = plan.get("input_files")
    source_files = plan.get("source_files")
    if not isinstance(input_files, dict) or not isinstance(source_files, dict):
        raise ValueError("Collector plan must bind every input and source file")
    for name, item in input_files.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid collector input binding: " + str(name))
        expected = _exact_sha(item.get("sha256"), "collector input " + str(name))
        result.append((Path(item["path"]).expanduser().resolve(), expected))
    root = ROOT.resolve()
    for relative, raw_expected in source_files.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("Invalid collector source path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Collector source binding escapes the repository")
        expected = _exact_sha(raw_expected, "collector source " + relative)
        result.append((path, expected))
    return tuple(result)


def _verify_files(bindings):
    for path, expected in bindings:
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Bound collector input/source changed: " + str(path))


def _verify_collection_contract(
    collector,
    pools,
    plan,
    manifest,
    *,
    plan_file_sha256,
    allow_test_fixture,
):
    if not isinstance(pools, dict) or set(pools) != {"fit", "dev"}:
        raise ValueError("Development reader must expose exactly fit/dev pools")
    collector_version = getattr(collector, "VERSION", None)
    data_version = getattr(collector, "DATA_VERSION", None)
    if (
        not isinstance(plan, dict)
        or not isinstance(manifest, dict)
        or plan.get("version") != collector_version
        or plan.get("producer") != collector_version
        or plan.get("data_version") != data_version
        or plan.get("phase") != "development"
        or plan.get("phase_pools") != ["fit", "dev"]
        or manifest.get("phase") != "development"
    ):
        raise ValueError("Only the versioned diverse development collection is supported")
    if (
        manifest.get("status") != "completed"
        or manifest.get("pending") is not None
        or manifest.get("plan_sha256") != digest(plan)
        or manifest.get("plan_file_sha256") != plan_file_sha256
        or manifest.get("actor_bindings") != plan.get("actor_bindings")
    ):
        raise ValueError("Collection completion or plan binding differs")
    if plan.get("test_fixture") is not allow_test_fixture:
        raise ValueError("Fixture status must be explicit and match the fit request")
    names = plan.get("feature_names")
    if (
        not isinstance(names, list)
        or len(names) != 197
        or len(set(names)) != 197
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("Full ordered observed197 feature names required")
    bindings = plan.get("actor_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Frozen Actor bindings required")
    for key in (
        "actor_sha256",
        "actor_parameters_sha256",
        "protocol_sha256",
        "source_sha256",
        "runtime_signature",
    ):
        _exact_sha(bindings.get(key), key)
    declared = plan.get("pool_scene_counts")
    if not isinstance(declared, dict) or any(
        type(declared.get(name)) is not int for name in ("fit", "dev")
    ):
        raise ValueError("Collector must declare exact fit/dev scene counts")
    return tuple(names), bindings, _bound_collection_files(plan)


def _overlap(fit_data, dev_data):
    return {
        "scene_fingerprints": len(
            set(fit_data["scene_fingerprints"])
            & set(dev_data["scene_fingerprints"])
        ),
        "episode_ids": len(
            set(fit_data["episode_ids"]) & set(dev_data["episode_ids"])
        ),
        "exact_observations": len(
            {row.tobytes() for row in fit_data["observations"]}
            & {row.tobytes() for row in dev_data["observations"]}
        ),
    }


def _compressed_fit_rows(data):
    """Compress ``pure.samples`` duplicate views into exact per-row weights.

    ``pure.samples`` presents every saved row once and adds one view of each
    endpoint for every physically effective pair.  A weighted unique row has
    the same squared-error contribution while avoiding a much larger design
    matrix.  The candidate pair multiplier is applied separately below.
    """
    observations = data["observations"]
    probabilities = data["probabilities"].astype(np.float64, copy=False)
    ordered = np.sort(probabilities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    importance = 1.0 + 8.0 * np.maximum(0.0, margins)
    multiplicity = np.ones(len(observations), dtype=np.int64)
    changed_endpoints = np.zeros(len(observations), dtype=bool)
    physical_pairs = 0
    changed_pairs = 0
    for pair in data["pairs"]:
        if not pair["physical_effect"]:
            continue
        physical_pairs += 1
        left = pair["baseline_index"]
        right = pair["changed_index"]
        multiplicity[left] += 1
        multiplicity[right] += 1
        if pair["nn_changed"]:
            changed_pairs += 1
            changed_endpoints[left] = True
            changed_endpoints[right] = True
    compressed_weights = importance * multiplicity.astype(np.float64)
    audit = {
        "unique_rows": int(len(observations)),
        "equivalent_pure_sample_views": int(multiplicity.sum()),
        "physical_pairs": physical_pairs,
        "changed_physical_pairs": changed_pairs,
        "changed_endpoint_rows": int(changed_endpoints.sum()),
        "physical_view_multiplicity_min": int(multiplicity.min()),
        "physical_view_multiplicity_max": int(multiplicity.max()),
        "base_importance_weight_min": float(importance.min()),
        "base_importance_weight_max": float(importance.max()),
        "compressed_weight_sum": float(compressed_weights.sum()),
        "row_weight_formula": (
            "(1+8*NN_top_two_margin)*physical_view_multiplicity"
        ),
    }
    return (
        observations,
        probabilities,
        compressed_weights,
        changed_endpoints,
        audit,
    )


def _candidate_weights(compressed_weights, changed_endpoints, pair_weight):
    pair_weight = max(1.0, float(pair_weight))
    result = np.asarray(compressed_weights, dtype=np.float64).copy()
    result[np.asarray(changed_endpoints, dtype=bool)] *= pair_weight
    return result


def _structure_targets(probabilities, action_structure_weight):
    """Match core RCPD soft plus optional hard-action split targets."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weight = max(0.0, float(action_structure_weight))
    if weight <= 0.0:
        return probabilities
    hard = np.zeros_like(probabilities)
    hard[np.arange(len(probabilities)), np.argmax(probabilities, axis=1)] = np.sqrt(
        weight
    )
    return np.concatenate((probabilities, hard), axis=1)


def _normalized_soft_tree_output(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = np.maximum(values[:, : len(pure.ACTIONS)], 0.0)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.full_like(values, 1.0 / len(pure.ACTIONS)),
        where=totals > 0.0,
    )


def _training_replay(estimator, program, observations, feature_names):
    """Prove the exported program replays sklearn's soft tree outputs."""
    maximum_error = 0.0
    argmax_mismatches = 0
    chunks = 0
    for start in range(0, len(observations), REPLAY_CHUNK_ROWS):
        stop = min(len(observations), start + REPLAY_CHUNK_ROWS)
        expected = _normalized_soft_tree_output(
            estimator.predict(observations[start:stop])
        )
        replayed = pure.prediction.predict(
            program, observations[start:stop], feature_names
        ).astype(np.float64)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(expected - replayed))) if len(expected) else 0.0,
        )
        argmax_mismatches += int(
            np.count_nonzero(np.argmax(expected, axis=1) != np.argmax(replayed, axis=1))
        )
        chunks += 1
    passed = maximum_error <= REPLAY_ATOL and argmax_mismatches == 0
    result = {
        "rows": int(len(observations)),
        "chunks": chunks,
        "absolute_tolerance": REPLAY_ATOL,
        "maximum_probability_error": maximum_error,
        "argmax_mismatches": argmax_mismatches,
        "passed": passed,
    }
    if not passed:
        raise ValueError("core._tree_to_program replay differs from sklearn tree")
    return result


def _fit_fixed_capacity_candidate(
    observations,
    probabilities,
    compressed_weights,
    changed_endpoints,
    feature_names,
    candidate,
    metadata,
):
    """Perform exactly one fixed-depth sklearn fit for one RCPD candidate."""
    cfg = config(candidate)
    fit_weights = _candidate_weights(
        compressed_weights,
        changed_endpoints,
        cfg.counterfactual_changed_pair_weight,
    )
    targets = _structure_targets(probabilities, cfg.action_structure_weight)
    estimator = DecisionTreeRegressor(
        max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=max(
            1, min(cfg.min_samples_leaf, len(observations) // 2 or 1)
        ),
        random_state=cfg.random_seed,
    )
    estimator.fit(observations, targets, sample_weight=fit_weights)
    root = _tree_to_program(estimator, tuple(feature_names), len(pure.ACTIONS))
    program = ExecutableProgram(
        action_names=tuple(pure.ACTIONS),
        feature_names=tuple(feature_names),
        root=root,
        metadata={
            **deepcopy(metadata),
            "distillation_method": "regularity_constrained_policy_distillation",
            "distillation_backend": BACKEND_VERSION,
            "candidate_depth": cfg.max_depth,
            "candidate_max_leaf_nodes": cfg.max_leaf_nodes,
            "training_samples": int(len(observations)),
            "counterfactual_changed_pair_weight": float(
                cfg.counterfactual_changed_pair_weight
            ),
            "counterfactual_changed_training_samples": int(
                np.count_nonzero(changed_endpoints)
            ),
            "action_structure_weight": float(cfg.action_structure_weight),
            "fit_row_weight_sum": float(fit_weights.sum()),
            "sklearn_fit_count": 1,
        },
    )
    replay = _training_replay(estimator, program, observations, feature_names)
    details = {
        "sklearn_fit_count": 1,
        "target_columns": int(targets.shape[1]),
        "soft_target_columns": len(pure.ACTIONS),
        "hard_structure_target_columns": (
            len(pure.ACTIONS) if cfg.action_structure_weight > 0.0 else 0
        ),
        "fit_weight_sum": float(fit_weights.sum()),
        "fit_weight_min": float(fit_weights.min()),
        "fit_weight_max": float(fit_weights.max()),
        "training_replay": replay,
    }
    return program, details


def _failed_gate_checks(gate):
    failed = []
    for scope in ("ordinary", "all_rows", "counterfactual"):
        for name, passed in gate[scope]["checks"].items():
            if not passed:
                failed.append(scope + "/" + name)
    for role, role_gate in gate["by_role"].items():
        for scope in ("fidelity", "direction"):
            for name, passed in role_gate[scope]["checks"].items():
                if not passed:
                    failed.append("by_role/" + role + "/" + scope + "/" + name)
    return failed


def _selection(records):
    passing = [record for record in records if record["gates"]["teacher_target_met"]]
    if passing:
        selected, reason = original.choose(passing)
        return selected, reason, None, None
    diagnostic, diagnostic_reason = original.choose(records)
    return None, "diagnostic_only_no_candidate_passed", diagnostic, diagnostic_reason


def inspect_source(source):
    """Return hashes required by a later fit without fitting or evaluating."""
    source = Path(source).expanduser().resolve()
    collector = _load_collector()
    plan_path, manifest_path = source / "plan.json", source / "manifest.json"
    plan_file_sha256 = file_hash(plan_path)
    manifest_file_sha256 = file_hash(manifest_path)
    declared_plan = _read_json(plan_path)
    allow_fixture = declared_plan.get("test_fixture") is True
    pools, plan, manifest = collector.read_data(
        source,
        phase="development",
        allow_test_fixture=allow_fixture,
    )
    names, _, bindings = _verify_collection_contract(
        collector,
        pools,
        plan,
        manifest,
        plan_file_sha256=plan_file_sha256,
        allow_test_fixture=allow_fixture,
    )
    _verify_files(bindings)
    summaries = {
        "fit": original.validate_data(pools["fit"], pool="train"),
        "dev": original.validate_data(pools["dev"], pool="selection"),
    }
    return {
        "source": str(source),
        "plan_file_sha256": plan_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "fit_data_sha256": summaries["fit"]["data_sha256"],
        "dev_data_sha256": summaries["dev"]["data_sha256"],
        "feature_count": len(names),
        "scenes": {name: value["scenes"] for name, value in summaries.items()},
        "test_fixture": allow_fixture,
    }


def fit_collected(
    source,
    *,
    expected_plan_file_sha256,
    expected_manifest_file_sha256,
    expected_fit_data_sha256,
    expected_dev_data_sha256,
    output,
    allow_test_fixture=False,
):
    """Fit five frozen candidates, then evaluate the saved bytes on dev once."""
    if type(allow_test_fixture) is not bool:
        raise ValueError("Fixture scope must be an explicit bool")
    expected_plan_file_sha256 = _exact_sha(
        expected_plan_file_sha256, "expected plan file"
    )
    expected_manifest_file_sha256 = _exact_sha(
        expected_manifest_file_sha256, "expected manifest file"
    )
    expected_fit_data_sha256 = _exact_sha(
        expected_fit_data_sha256, "expected fit data"
    )
    expected_dev_data_sha256 = _exact_sha(
        expected_dev_data_sha256, "expected dev data"
    )
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if (
        output.exists()
        or output == source
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Use a separate fresh fit output; attempts are never overwritten")
    plan_path, manifest_path = source / "plan.json", source / "manifest.json"
    if (
        not plan_path.is_file()
        or file_hash(plan_path) != expected_plan_file_sha256
        or not manifest_path.is_file()
        or file_hash(manifest_path) != expected_manifest_file_sha256
    ):
        raise ValueError("Exact completed collection plan/manifest SHA256 differs")

    collector = _load_collector()
    plan, manifest = _read_development_header(
        collector,
        source,
        allow_test_fixture=allow_test_fixture,
    )
    declared_pools = dict.fromkeys(plan.get("phase_pools", ()))
    feature_names, actor_bindings, collection_files = _verify_collection_contract(
        collector,
        declared_pools,
        plan,
        manifest,
        plan_file_sha256=expected_plan_file_sha256,
        allow_test_fixture=allow_test_fixture,
    )
    _verify_files(collection_files)
    fit_data = _read_completed_pool(
        collector, source, plan, manifest, "fit"
    )
    fit_input = original.validate_data(fit_data, pool="train")
    if fit_input["data_sha256"] != expected_fit_data_sha256:
        raise ValueError("Exact fit observations, labels, or provenance SHA256 differs")
    if plan["pool_scene_counts"]["fit"] != fit_input["scenes"]:
        raise ValueError("Declared and reconstructed fit scene counts differ")
    if not allow_test_fixture and plan["pool_scene_counts"] != {
        "fit": FIT_SCENES,
        "dev": DEV_SCENES,
    }:
        raise ValueError("Production development pools require exactly 192/64 scenes")

    source_hashes = _source_hashes(collector)
    (
        fit_observations,
        fit_probabilities,
        compressed_weights,
        changed_endpoints,
        presentation_audit,
    ) = _compressed_fit_rows(fit_data)
    output.mkdir(parents=True, exist_ok=False)
    pure.put(
        output / "request.json",
        {
            "version": VERSION,
            "contract": contract(),
            "source": str(source),
            "plan_file_sha256": expected_plan_file_sha256,
            "manifest_file_sha256": expected_manifest_file_sha256,
            "semantic_plan_sha256": digest(plan),
            "semantic_manifest_sha256": digest(manifest),
            "fit_data_sha256": expected_fit_data_sha256,
            "dev_data_sha256": expected_dev_data_sha256,
            "actor_bindings": deepcopy(actor_bindings),
            "feature_names": list(feature_names),
            "inputs": {"fit": fit_input, "dev": "deferred_until_fit_phase_complete"},
            "fit_presentation": presentation_audit,
            "dev_content_inspected": False,
            "dev_metrics_evaluated": False,
            "source_hashes": source_hashes,
            "test_fixture": allow_test_fixture,
            "automatic_retry": False,
        },
    )

    completed = []
    dev_metrics_started = False
    started = time.monotonic()
    try:
        for index, candidate in enumerate(CANDIDATES):
            cfg = config(candidate)
            pure.put(
                output / f"fit_{index:02d}.request.json",
                {
                    "index": index,
                    "candidate": list(candidate),
                    "config": asdict(cfg),
                    "backend": BACKEND_VERSION,
                    "maximum_sklearn_fits": 1,
                    "external_validation_rows": 0,
                    "fit_unique_rows": int(len(fit_observations)),
                    "fit_equivalent_pure_sample_views": presentation_audit[
                        "equivalent_pure_sample_views"
                    ],
                    "dev_metrics_evaluated": False,
                    "retry_allowed": False,
                },
            )
            before = time.monotonic()
            program_metadata = {
                "native_feedback_version": VERSION,
                "native_source_actor_sha256": actor_bindings["actor_sha256"],
                "source_actor_bindings": deepcopy(actor_bindings),
                "fit_data_sha256": expected_fit_data_sha256,
                "source_plan_file_sha256": expected_plan_file_sha256,
                "source_manifest_file_sha256": expected_manifest_file_sha256,
                "prediction_semantics": pure.prediction.VERSION,
                "runtime_controller": "native_neural_actor_only",
                "role_scope": list(original.ROLES),
                "fit_pool_only": True,
                "development_metrics_evaluated": False,
                "feedback_eligible": False,
                "explanation_eligible": False,
                "explanation_qualified": False,
                "release_ready": False,
                "test_fixture": allow_test_fixture,
                "fit_presentation": deepcopy(presentation_audit),
            }
            with pure._isolated_host_rng():
                fitted_program, fit_details = _fit_fixed_capacity_candidate(
                    fit_observations,
                    fit_probabilities,
                    compressed_weights,
                    changed_endpoints,
                    feature_names,
                    candidate,
                    program_metadata,
                )
            metadata = deepcopy(fitted_program.metadata)
            metadata["metrics"] = {
                **deepcopy(metadata.get("metrics", {})),
                "feedback_eligible": False,
                "explanation_eligible": False,
                "feedback_weight": 0.0,
                "feedback_ineligibility_reasons": [
                    "offline_development_fit_only"
                ],
                "explanation_ineligibility_reasons": [
                    "independent_acceptance_not_executed"
                ],
            }
            program = pure.canonical_program(
                fitted_program, feature_names, metadata
            )
            program_path = output / f"program_{index:02d}.json"
            audit_path = output / f"fit_{index:02d}.audit.json"
            pure.put(program_path, program.to_dict())
            audit = {
                "index": index,
                "candidate": list(candidate),
                "config": asdict(cfg),
                "backend": BACKEND_VERSION,
                "sklearn_fit_count": 1,
                "fit_presentation": presentation_audit,
                **fit_details,
                "elapsed_seconds": time.monotonic() - before,
                "external_validation_rows": 0,
                "dev_metrics_evaluated": False,
                "program_sha256": file_hash(program_path),
            }
            pure.put(audit_path, audit)
            completed.append(
                {
                    "index": index,
                    "candidate": list(candidate),
                    "program_sha256": audit["program_sha256"],
                    "fit_audit_sha256": file_hash(audit_path),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "diverse_rcpd_fit_complete",
                        "index": index,
                        "depth": program.root.depth(),
                        "leaves": program.root.leaf_count(),
                    }
                ),
                flush=True,
            )

        if len(completed) != len(CANDIDATES):
            raise RuntimeError("Every predeclared candidate must finish before dev")
        pure.put(
            output / "fit_phase_complete.json",
            {
                "version": VERSION,
                "backend": BACKEND_VERSION,
                "completed_candidates": completed,
                "completed_fixed_capacity_rcpd_candidates": len(completed),
                "completed_sklearn_fits": len(completed),
                "dev_metrics_evaluated": False,
                "dev_content_inspected": False,
                "fit_data_sha256": expected_fit_data_sha256,
            },
        )

        # Reload every frozen candidate before reading any dev-pool content.
        programs = []
        for entry in completed:
            path = output / f"program_{entry['index']:02d}.json"
            if file_hash(path) != entry["program_sha256"]:
                raise ValueError("Saved candidate changed before dev evaluation")
            program = ExecutableProgram.from_dict(_read_json(path))
            programs.append(pure.canonical_program(program, feature_names))

        dev_data = _read_completed_pool(
            collector, source, plan, manifest, "dev"
        )
        dev_input = original.validate_data(dev_data, pool="selection")
        if dev_input["data_sha256"] != expected_dev_data_sha256:
            raise ValueError("Exact dev observations, labels, or provenance SHA256 differs")
        if plan["pool_scene_counts"]["dev"] != dev_input["scenes"]:
            raise ValueError("Declared and reconstructed dev scene counts differ")
        overlap = _overlap(fit_data, dev_data)
        if any(overlap.values()):
            raise ValueError(
                "Fit/dev overlap rejected without deleting rows: " + str(overlap)
            )
        inputs = {"fit": fit_input, "dev": dev_input}
        dev_metrics_started = True
        dev_base = original._subset(
            dev_data,
            [index for index, kind in enumerate(dev_data["kind"]) if kind == "base"],
        )
        records = []
        for index, (candidate, program) in enumerate(zip(CANDIDATES, programs)):
            selection_metrics = original.metrics(
                program, dev_data, feature_names
            )
            ordinary_metrics = original.metrics(program, dev_base, feature_names)
            gate = original.gates(selection_metrics, ordinary_metrics)
            record = {
                "index": index,
                "candidate": list(candidate),
                "depth_cap": candidate[0],
                "leaf_cap": candidate[1],
                "counterfactual_changed_pair_weight": candidate[2],
                "action_structure_weight": candidate[3],
                "config": asdict(config(candidate)),
                "selection_metrics": selection_metrics,
                "ordinary_metrics": ordinary_metrics,
                "fit_pool_metrics": original.metrics(
                    program, fit_data, feature_names
                ),
                "gates": gate,
                "failed_gate_checks": _failed_gate_checks(gate),
                "complexity": pure.program_complexity(
                    program,
                    max_depth=max(value[0] for value in CANDIDATES),
                    max_leaf_count=max(value[1] for value in CANDIDATES),
                    max_predicate_count=max(value[1] for value in CANDIDATES) - 1,
                ).to_dict(),
                "program_sha256": completed[index]["program_sha256"],
            }
            records.append(record)
            pure.put(output / f"metrics_{index:02d}.json", record)

        selected, selection_reason, diagnostic, diagnostic_reason = _selection(records)
        selected_index = None if selected is None else selected["index"]
        if selected is not None:
            selected_path = output / f"program_{selected_index:02d}.json"
            pure.put(output / "program.json", _read_json(selected_path))

        _verify_files(collection_files)
        if (
            file_hash(plan_path) != expected_plan_file_sha256
            or file_hash(manifest_path) != expected_manifest_file_sha256
            or _source_hashes(collector) != source_hashes
            or original._data_digest(fit_data) != expected_fit_data_sha256
            or original._data_digest(dev_data) != expected_dev_data_sha256
        ):
            raise ValueError("Bound fitting source or input changed during fitting")
        report = jsonable(
            {
                "version": VERSION,
                "status": "completed",
                "contract": contract(),
                "source": str(source),
                "plan_file_sha256": expected_plan_file_sha256,
                "manifest_file_sha256": expected_manifest_file_sha256,
                "fit_data_sha256": expected_fit_data_sha256,
                "dev_data_sha256": expected_dev_data_sha256,
                "source_hashes": source_hashes,
                "actor_bindings": deepcopy(actor_bindings),
                "inputs": inputs,
                "overlap": overlap,
                "fit_phase_complete_sha256": file_hash(
                    output / "fit_phase_complete.json"
                ),
                "candidates": records,
                "selected_index": selected_index,
                "selection_reason": selection_reason,
                "diagnostic_best_index": (
                    None if diagnostic is None else diagnostic["index"]
                ),
                "diagnostic_reason": diagnostic_reason,
                "development_target_met": selected is not None,
                "backend": BACKEND_VERSION,
                "actual_rcpd_calls": 0,
                "actual_fixed_capacity_rcpd_candidates": len(completed),
                "actual_sklearn_fits": len(completed),
                "maximum_sklearn_fits": len(CANDIDATES),
                "external_core_validation_rows": 0,
                "new_NN_queries": 0,
                "Actor_loads": 0,
                "PT_loads": 0,
                "environment_steps": 0,
                "PPO_updates": 0,
                "optimizer_updates": 0,
                "independent_acceptance_executed": False,
                "feedback_admission_granted": False,
                "explanation_qualified": False,
                "release_ready": False,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        pure.put(output / "report.json", report)
        pure.put(
            output / "completion.json",
            {
                "version": VERSION,
                "status": "completed",
                "selected_index": selected_index,
                "diagnostic_best_index": report["diagnostic_best_index"],
                "development_target_met": report["development_target_met"],
                "actual_rcpd_calls": report["actual_rcpd_calls"],
                "actual_fixed_capacity_rcpd_candidates": report[
                    "actual_fixed_capacity_rcpd_candidates"
                ],
                "actual_sklearn_fits": report["actual_sklearn_fits"],
                "report_sha256": file_hash(output / "report.json"),
                "release_ready": False,
            },
        )
        return report
    except BaseException as error:
        pure.put(
            output / "failure.json",
            {
                "version": VERSION,
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_fixed_capacity_rcpd_candidates": len(completed),
                "completed_sklearn_fits": len(completed),
                "dev_metrics_started": dev_metrics_started,
                "pending_fit_not_retried": True,
                "partial_fit_count_may_be_unknown": True,
                "release_ready": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inspect-source", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--plan-file-sha256")
    parser.add_argument("--manifest-file-sha256")
    parser.add_argument("--fit-data-sha256")
    parser.add_argument("--dev-data-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.execute and args.inspect_source:
        parser.error("Choose either --execute or --inspect-source")
    if args.inspect_source:
        if not args.source:
            parser.error("--inspect-source requires --source")
        print(json.dumps(inspect_source(args.source), sort_keys=True, indent=2))
        return
    if not args.execute:
        print(json.dumps(contract(), sort_keys=True, indent=2))
        return
    required = (
        args.source,
        args.plan_file_sha256,
        args.manifest_file_sha256,
        args.fit_data_sha256,
        args.dev_data_sha256,
        args.output,
    )
    if not all(required):
        parser.error("Execution requires source, output, and all four exact SHA256 values")
    result = fit_collected(
        args.source,
        expected_plan_file_sha256=args.plan_file_sha256,
        expected_manifest_file_sha256=args.manifest_file_sha256,
        expected_fit_data_sha256=args.fit_data_sha256,
        expected_dev_data_sha256=args.dev_data_sha256,
        output=args.output,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "selected_index",
                    "diagnostic_best_index",
                    "development_target_met",
                    "actual_fixed_capacity_rcpd_candidates",
                    "actual_sklearn_fits",
                    "release_ready",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
