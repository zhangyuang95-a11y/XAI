"""Independent saved-data acceptance for a frozen diverse alignment program.

This module routes one already serialized program over one completely collected
``accept`` pool.  It performs no Actor load or neural query, constructs no
environment, imports no PyTorch checkpoint, and fits nothing.  The unchanged
dual-role teacher metrics and gates determine explanation qualification.  A
passing run writes a new qualified program; it never edits the input program.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import time

from core.program import ExecutableProgram

from . import warehouse_family_alignment_diverse_teacher_fit as fitter
from . import warehouse_family_branch_teacher_fit as original
from . import warehouse_family_intervention_fit as pure
from .warehouse_native_common import ROOT, digest, file_hash, jsonable


VERSION = "warehouse-family-alignment-diverse-independent-acceptance.v1"
COLLECTOR_MODULE = "backend.training.warehouse_family_alignment_diverse_collection"
ACCEPT_SCENES = 64


def contract():
    return {
        "version": VERSION,
        "collector_module": COLLECTOR_MODULE,
        "program_producer": fitter.VERSION,
        "acceptance_pool": {"accept": ACCEPT_SCENES},
        "metrics_source": original.VERSION,
        "gates_source": original.VERSION,
        "gates_modified": False,
        "minimum_overall_fidelity": 0.90,
        "minimum_non_wait_fidelity": 0.85,
        "minimum_critical_fidelity": 0.85,
        "minimum_direction_fidelity": 0.85,
        "minimum_critical_scenarios": 10,
        "maximum_mean_kl": 0.35,
        "scope": (
            "all saved accept rows, base-only ordinary rows, both real roles, "
            "all three critical groups, and effective counterfactual direction"
        ),
        "required_external_hashes": [
            "program_file_sha256",
            "plan_file_sha256",
            "manifest_file_sha256",
            "accept_data_sha256",
        ],
        "qualification_output": "new program copy only when every unchanged gate passes",
        "web_integration_required_for_release": True,
        "Actor_loads": 0,
        "new_NN_queries": 0,
        "environment_constructions": 0,
        "environment_steps": 0,
        "PT_loads": 0,
        "RCPD_calls": 0,
        "tree_fits": 0,
        "optimizer_updates": 0,
        "PPO_updates": 0,
        "release_ready": False,
    }


def _exact_sha(value, name):
    return original._sha(value, name)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_collector():
    return importlib.import_module(COLLECTOR_MODULE)


def _source_hashes(collector):
    result = dict(original.execution_sources())
    for module in (collector, fitter):
        path = Path(module.__file__).resolve()
        result[str(path.relative_to(ROOT))] = file_hash(path)
    path = Path(__file__).resolve()
    result[str(path.relative_to(ROOT))] = file_hash(path)
    return dict(sorted(result.items()))


def execution_sources():
    """Current source closure recorded by a qualified program."""
    return _source_hashes(_load_collector())


def _collection_file_bindings(plan):
    inputs, sources = plan.get("input_files"), plan.get("source_files")
    if not isinstance(inputs, dict) or not isinstance(sources, dict):
        raise ValueError("Acceptance collection must bind every input and source")
    result = []
    for name, item in inputs.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Invalid collector input binding: " + str(name))
        result.append((Path(item["path"]).expanduser().resolve(),
                       _exact_sha(item["sha256"], "collector input " + str(name))))
    root = ROOT.resolve()
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("Invalid collector source path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Collector source path escapes the repository")
        result.append((path, _exact_sha(expected, "collector source " + relative)))
    return tuple(result)


def _verify_files(bindings):
    for path, expected in bindings:
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Externally bound input/source changed: " + str(path))


def _feature_names(plan):
    names = plan.get("feature_names")
    if (not isinstance(names, list) or len(names) != 197 or len(set(names)) != 197
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("Full ordered observed197 feature schema required")
    return tuple(names)


def _actor_bindings(plan):
    bindings = plan.get("actor_bindings")
    required = (
        "actor_sha256", "actor_parameters_sha256", "protocol_sha256",
        "source_sha256", "runtime_signature",
    )
    if not isinstance(bindings, dict) or set(bindings) != set(required):
        raise ValueError("Exact frozen Actor binding fields required")
    for key in required:
        _exact_sha(bindings[key], key)
    return bindings


def _validate_collection(collector, pools, plan, manifest, *, phase,
                         plan_file_sha256, allow_test_fixture):
    if phase != "acceptance" or set(pools) != {"accept"}:
        raise ValueError("Independent reader must expose exactly the acceptance pool")
    if (plan.get("version") != collector.VERSION
            or plan.get("producer") != collector.VERSION
            or plan.get("data_version") != collector.DATA_VERSION
            or plan.get("phase") != "acceptance"
            or plan.get("phase_pools") != ["accept"]
            or plan.get("pool_ranges") != {"accept": [448, 512]}
            or plan.get("pool_scene_counts") != {"accept": ACCEPT_SCENES}
            or plan.get("context_count") != ACCEPT_SCENES * 4
            or plan.get("test_fixture") is not allow_test_fixture
            or plan.get("sampling_uses_tree_predictions") is not False
            or plan.get("labels_from_unmodified_neural_actor") is not True
            or plan.get("both_actual_roles_physically_intervened") is not True
            or any(plan.get(key) != 0 for key in ("PPO_steps", "optimizer_updates", "tree_fits"))):
        raise ValueError("Only the fixed, pure-NN diverse acceptance collection is supported")
    if (manifest.get("version") != collector.VERSION
            or manifest.get("phase") != "acceptance"
            or manifest.get("status") != "completed"
            or manifest.get("pending") is not None
            or manifest.get("plan_sha256") != digest(plan)
            or manifest.get("plan_file_sha256") != plan_file_sha256
            or manifest.get("actor_bindings") != plan.get("actor_bindings")
            or manifest.get("independent_acceptance_executed") is not True
            or manifest.get("qualification_evaluated") is not False):
        raise ValueError("Acceptance completion or immutable plan binding differs")
    names = _feature_names(plan)
    actor_bindings = _actor_bindings(plan)
    if not allow_test_fixture:
        if (actor_bindings["actor_sha256"] != collector.PRODUCTION_ACTOR_SHA256
                or plan.get("actor_training_clock") != collector.PRODUCTION_ACTOR_TRAINING_CLOCK):
            raise ValueError("Production acceptance is locked to the 3.95M Actor")
    return names, actor_bindings, _collection_file_bindings(plan)


def _validate_program(raw, feature_names, actor_bindings, *, allow_test_fixture):
    program = ExecutableProgram.from_dict(raw)
    if program.to_dict() != raw:
        raise ValueError("Frozen input must be one canonical executable program")
    if (tuple(program.feature_names) != feature_names
            or tuple(program.action_names) != tuple(pure.ACTIONS)):
        raise ValueError("Program feature/action order differs from the acceptance Actor")
    metadata = program.metadata
    for name in ("fit_data_sha256", "source_plan_file_sha256", "source_manifest_file_sha256"):
        _exact_sha(metadata.get(name), "program " + name)
    metrics = metadata.get("metrics")
    if (metadata.get("native_feedback_version") != fitter.VERSION
            or metadata.get("native_source_actor_sha256") != actor_bindings["actor_sha256"]
            or metadata.get("source_actor_bindings") != actor_bindings
            or metadata.get("prediction_semantics") != pure.prediction.VERSION
            or metadata.get("runtime_controller") != "native_neural_actor_only"
            or metadata.get("role_scope") != list(original.ROLES)
            or metadata.get("fit_pool_only") is not True
            or metadata.get("test_fixture") is not allow_test_fixture
            or metadata.get("explanation_qualified") is not False
            or metadata.get("release_ready") is not False
            or metadata.get("explanation_eligible") is not False
            or not isinstance(metrics, dict)
            or metrics.get("explanation_eligible") is not False):
        raise ValueError("Frozen development program provenance or pre-acceptance status differs")
    return program


def _failed_gate_checks(gate):
    failed = []
    for scope in ("ordinary", "all_rows", "counterfactual"):
        failed.extend(scope + "/" + name for name, passed in gate[scope]["checks"].items()
                      if not passed)
    for role, role_gates in gate["by_role"].items():
        for scope in ("fidelity", "direction"):
            failed.extend("by_role/" + role + "/" + scope + "/" + name
                          for name, passed in role_gates[scope]["checks"].items()
                          if not passed)
    return failed


def _scorecard(all_metrics, ordinary_metrics):
    """Expose the exact action, role, group, and direction denominators/rates."""
    groups = tuple(original.GROUPS)
    roles = tuple(original.ROLES)
    return {
        "all_rows": {
            "overall": deepcopy(all_metrics["overall"]),
            "base": deepcopy(all_metrics["base"]),
            "counterfactual": deepcopy(all_metrics["counterfactual"]),
            "non_wait": deepcopy(all_metrics["non_wait"]),
            "mean_kl": all_metrics["mean_kl"],
            "by_action": deepcopy(all_metrics["by_action"]),
        },
        "base_ordinary": {
            "overall": deepcopy(ordinary_metrics["overall"]),
            "non_wait": deepcopy(ordinary_metrics["non_wait"]),
            "mean_kl": ordinary_metrics["mean_kl"],
            "by_action": deepcopy(ordinary_metrics["by_action"]),
        },
        "by_role": {
            role: {
                "all_rows": {
                    "overall": deepcopy(all_metrics["by_role"][role]["overall"]),
                    "non_wait": deepcopy(all_metrics["by_role"][role]["non_wait"]),
                    "mean_kl": all_metrics["by_role"][role]["mean_kl"],
                    "by_action": deepcopy(all_metrics["by_role"][role]["by_action"]),
                },
                "base_ordinary": {
                    "overall": deepcopy(ordinary_metrics["by_role"][role]["overall"]),
                    "non_wait": deepcopy(ordinary_metrics["by_role"][role]["non_wait"]),
                    "mean_kl": ordinary_metrics["by_role"][role]["mean_kl"],
                    "by_action": deepcopy(ordinary_metrics["by_role"][role]["by_action"]),
                },
                "critical_groups": {
                    group: deepcopy(all_metrics["by_role"][role]["critical"][group])
                    for group in groups
                },
                "counterfactual_direction": {
                    name: deepcopy(all_metrics["by_role"][role]["direction"][name])
                    for name in ("all", *groups)
                },
            }
            for role in roles
        },
        "critical_groups": {
            group: {
                "all_rows": deepcopy(all_metrics["critical"][group]),
                "base_ordinary": deepcopy(ordinary_metrics["critical"][group]),
            }
            for group in groups
        },
        "counterfactual_direction": {
            **{name: deepcopy(all_metrics["direction"][name]) for name in ("all", *groups)},
            "all_pairs": all_metrics["direction"]["all_pairs"],
            "valid_pairs": all_metrics["direction"]["valid_pairs"],
        },
    }


def inspect_source(source, *, allow_test_fixture=False):
    """Return external hashes and accept data digest without evaluating a program."""
    if type(allow_test_fixture) is not bool:
        raise ValueError("Fixture scope must be an explicit bool")
    source = Path(source).expanduser().resolve()
    collector = _load_collector()
    plan_path, manifest_path = source / "plan.json", source / "manifest.json"
    plan_sha, manifest_sha = file_hash(plan_path), file_hash(manifest_path)
    pools, plan, manifest = collector.read_data(
        source, phase="acceptance", allow_test_fixture=allow_test_fixture,
    )
    _, _, bindings = _validate_collection(
        collector, pools, plan, manifest, phase="acceptance",
        plan_file_sha256=plan_sha, allow_test_fixture=allow_test_fixture,
    )
    _verify_files(bindings)
    summary = original.validate_data(pools["accept"], pool="selection")
    return {
        "source": str(source),
        "plan_file_sha256": plan_sha,
        "manifest_file_sha256": manifest_sha,
        "accept_data_sha256": summary["data_sha256"],
        "accept_scenes": summary["scenes"],
        "accept_rows": summary["rows"],
        "test_fixture": allow_test_fixture,
    }


def evaluate_collected(program, source, *, expected_program_file_sha256,
                       expected_plan_file_sha256, expected_manifest_file_sha256,
                       expected_accept_data_sha256, output,
                       allow_test_fixture=False):
    """Evaluate saved acceptance rows and optionally emit a qualified copy."""
    if type(allow_test_fixture) is not bool:
        raise ValueError("Fixture scope must be an explicit bool")
    expected_program_file_sha256 = _exact_sha(expected_program_file_sha256, "program file")
    expected_plan_file_sha256 = _exact_sha(expected_plan_file_sha256, "plan file")
    expected_manifest_file_sha256 = _exact_sha(expected_manifest_file_sha256, "manifest file")
    expected_accept_data_sha256 = _exact_sha(expected_accept_data_sha256, "accept data")
    program_path = Path(program).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    protected = (program_path, source)
    if (output.exists() or any(output == path or output in path.parents or path in output.parents
                               for path in protected)):
        raise ValueError("Use a fresh acceptance output separate from every frozen input")
    plan_path, manifest_path = source / "plan.json", source / "manifest.json"
    if (not program_path.is_file() or file_hash(program_path) != expected_program_file_sha256
            or not plan_path.is_file() or file_hash(plan_path) != expected_plan_file_sha256
            or not manifest_path.is_file() or file_hash(manifest_path) != expected_manifest_file_sha256):
        raise ValueError("External program or acceptance collection SHA256 differs")

    collector = _load_collector()
    pools, plan, manifest = collector.read_data(
        source, phase="acceptance", allow_test_fixture=allow_test_fixture,
    )
    feature_names, actor_bindings, collection_files = _validate_collection(
        collector, pools, plan, manifest, phase="acceptance",
        plan_file_sha256=expected_plan_file_sha256,
        allow_test_fixture=allow_test_fixture,
    )
    _verify_files(collection_files)
    accept = pools["accept"]
    input_summary = original.validate_data(accept, pool="selection")
    if input_summary["data_sha256"] != expected_accept_data_sha256:
        raise ValueError("External acceptance observation/label/provenance SHA256 differs")
    if (input_summary["scenes"] != plan["pool_scene_counts"]["accept"]
            or (not allow_test_fixture and input_summary["scenes"] != ACCEPT_SCENES)):
        raise ValueError("Reconstructed acceptance scene count differs")
    raw_program = _read_json(program_path)
    frozen_program = _validate_program(
        raw_program, feature_names, actor_bindings,
        allow_test_fixture=allow_test_fixture,
    )
    sources = _source_hashes(collector)
    program_semantic_sha256 = digest(raw_program)
    accept_before = original._data_digest(accept)
    output.mkdir(parents=True, exist_ok=False)
    pure.put(output / "request.json", {
        "version": VERSION,
        "contract": contract(),
        "program": str(program_path),
        "source": str(source),
        "program_file_sha256": expected_program_file_sha256,
        "program_semantic_sha256": program_semantic_sha256,
        "plan_file_sha256": expected_plan_file_sha256,
        "manifest_file_sha256": expected_manifest_file_sha256,
        "semantic_plan_sha256": digest(plan),
        "semantic_manifest_sha256": digest(manifest),
        "accept_data_sha256": expected_accept_data_sha256,
        "actor_bindings": deepcopy(actor_bindings),
        "feature_names": list(feature_names),
        "input": input_summary,
        "source_hashes": sources,
        "test_fixture": allow_test_fixture,
        "automatic_retry": False,
    })

    started = time.monotonic()
    try:
        base_indices = [index for index, kind in enumerate(accept["kind"]) if kind == "base"]
        base = original._subset(accept, base_indices)
        all_metrics = original.metrics(frozen_program, accept, feature_names)
        ordinary_metrics = original.metrics(frozen_program, base, feature_names)
        gates = original.gates(all_metrics, ordinary_metrics)
        passed = gates["teacher_target_met"]
        failed = _failed_gate_checks(gates)
        scorecard = _scorecard(all_metrics, ordinary_metrics)
        evidence = jsonable({
            "version": VERSION,
            "program_file_sha256": expected_program_file_sha256,
            "program_semantic_sha256": program_semantic_sha256,
            "plan_file_sha256": expected_plan_file_sha256,
            "manifest_file_sha256": expected_manifest_file_sha256,
            "accept_data_sha256": expected_accept_data_sha256,
            "actor_bindings": deepcopy(actor_bindings),
            "feature_names_sha256": digest(list(feature_names)),
            "input": input_summary,
            "all_rows_metrics": all_metrics,
            "base_ordinary_metrics": ordinary_metrics,
            "scorecard": scorecard,
            "gates": gates,
            "failed_gate_checks": failed,
            "all_thresholds_passed": passed,
            "metrics_source": original.VERSION,
            "gates_modified": False,
            "acceptance_source_sha256": digest(sources),
            "independent_acceptance_executed": True,
            "Actor_loads": 0,
            "new_NN_queries": 0,
            "environment_steps": 0,
            "PT_loads": 0,
            "RCPD_calls": 0,
            "tree_fits": 0,
            "optimizer_updates": 0,
        })
        evidence_path = output / "acceptance_evidence.json"
        pure.put(evidence_path, evidence)
        evidence_sha256 = file_hash(evidence_path)

        qualified_path = None
        qualified_sha256 = None
        if passed:
            metadata = deepcopy(frozen_program.metadata)
            metadata.update({
                "explanation_eligible": True,
                "explanation_qualified": True,
                "independent_acceptance_executed": True,
                "independent_acceptance_version": VERSION,
                "independent_acceptance_source_sha256": digest(sources),
                "acceptance_evidence_file_sha256": evidence_sha256,
                "acceptance_data_sha256": expected_accept_data_sha256,
                "acceptance_plan_file_sha256": expected_plan_file_sha256,
                "acceptance_manifest_file_sha256": expected_manifest_file_sha256,
                "source_program_file_sha256": expected_program_file_sha256,
                "release_ready": False,
                "web_integration_completed": False,
            })
            metadata["metrics"] = {
                **deepcopy(metadata.get("metrics", {})),
                "explanation_eligible": True,
                "explanation_qualified": True,
                "independent_acceptance_executed": True,
                "acceptance_all_thresholds_passed": True,
                "acceptance_data_sha256": expected_accept_data_sha256,
                "explanation_ineligibility_reasons": [],
                "release_ready": False,
            }
            qualified = pure.canonical_program(frozen_program, feature_names, metadata)
            qualified_path = output / "qualified_program.json"
            pure.put(qualified_path, qualified.to_dict())
            qualified_sha256 = file_hash(qualified_path)

        _verify_files(collection_files)
        if (file_hash(program_path) != expected_program_file_sha256
                or digest(_read_json(program_path)) != program_semantic_sha256
                or file_hash(plan_path) != expected_plan_file_sha256
                or file_hash(manifest_path) != expected_manifest_file_sha256
                or original._data_digest(accept) != accept_before
                or accept_before != expected_accept_data_sha256
                or _source_hashes(collector) != sources):
            raise ValueError("Frozen program, acceptance data, or evaluation source changed")

        report = jsonable({
            "version": VERSION,
            "status": "completed",
            "accepted": passed,
            "all_thresholds_passed": passed,
            "failed_gate_checks": failed,
            "program": str(program_path),
            "source": str(source),
            "program_file_sha256": expected_program_file_sha256,
            "program_semantic_sha256": program_semantic_sha256,
            "plan_file_sha256": expected_plan_file_sha256,
            "manifest_file_sha256": expected_manifest_file_sha256,
            "accept_data_sha256": expected_accept_data_sha256,
            "actor_bindings": deepcopy(actor_bindings),
            "input": input_summary,
            "all_rows_metrics": all_metrics,
            "base_ordinary_metrics": ordinary_metrics,
            "scorecard": scorecard,
            "gates": gates,
            "acceptance_evidence_sha256": evidence_sha256,
            "qualified_program": None if qualified_path is None else str(qualified_path),
            "qualified_program_sha256": qualified_sha256,
            "source_hashes": sources,
            "acceptance_source_sha256": digest(sources),
            "independent_acceptance_executed": True,
            "explanation_qualified": passed,
            "release_ready": False,
            "web_integration_completed": False,
            "Actor_loads": 0,
            "new_NN_queries": 0,
            "environment_constructions": 0,
            "environment_steps": 0,
            "PT_loads": 0,
            "RCPD_calls": 0,
            "tree_fits": 0,
            "optimizer_updates": 0,
            "PPO_updates": 0,
            "test_fixture": allow_test_fixture,
            "elapsed_seconds": time.monotonic() - started,
        })
        pure.put(output / "report.json", report)
        pure.put(output / "completion.json", {
            "version": VERSION,
            "status": "completed",
            "accepted": passed,
            "all_thresholds_passed": passed,
            "report_sha256": file_hash(output / "report.json"),
            "acceptance_evidence_sha256": evidence_sha256,
            "qualified_program_sha256": qualified_sha256,
            "explanation_qualified": passed,
            "release_ready": False,
        })
        return report
    except BaseException as error:
        pure.put(output / "failure.json", {
            "version": VERSION,
            "error_type": type(error).__name__,
            "message": str(error),
            "automatic_retry": False,
            "release_ready": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inspect-source", action="store_true")
    parser.add_argument("--program")
    parser.add_argument("--program-file-sha256")
    parser.add_argument("--source")
    parser.add_argument("--plan-file-sha256")
    parser.add_argument("--manifest-file-sha256")
    parser.add_argument("--accept-data-sha256")
    parser.add_argument("--output")
    parser.add_argument("--allow-test-fixture", action="store_true")
    args = parser.parse_args()
    if args.execute and args.inspect_source:
        parser.error("Choose either --execute or --inspect-source")
    if args.inspect_source:
        if not args.source:
            parser.error("--inspect-source requires --source")
        print(json.dumps(inspect_source(
            args.source, allow_test_fixture=args.allow_test_fixture,
        ), sort_keys=True, indent=2))
        return
    if not args.execute:
        print(json.dumps(contract(), sort_keys=True, indent=2))
        return
    required = (
        args.program, args.program_file_sha256, args.source,
        args.plan_file_sha256, args.manifest_file_sha256,
        args.accept_data_sha256, args.output,
    )
    if not all(required):
        parser.error("Execution requires program, source, output, and all four exact SHA256 values")
    result = evaluate_collected(
        args.program, args.source,
        expected_program_file_sha256=args.program_file_sha256,
        expected_plan_file_sha256=args.plan_file_sha256,
        expected_manifest_file_sha256=args.manifest_file_sha256,
        expected_accept_data_sha256=args.accept_data_sha256,
        output=args.output,
        allow_test_fixture=args.allow_test_fixture,
    )
    print(json.dumps({
        key: result[key] for key in (
            "status", "accepted", "all_thresholds_passed",
            "explanation_qualified", "release_ready",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
