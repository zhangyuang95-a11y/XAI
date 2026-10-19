"""Read-only selection of one completed common BC; no execution capability.

The registered integer quality score is the only ranking input. Production
verification reuses the native BC receipt checker and additionally restores
all checkpoint tensors into a temporary Actor without an optimizer or reset.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path

from experiments.warehouse import v621_bc_protocol as protocol


SOURCE02_SHA256 = "35e9fe54e8de193ed79d3553b540e481a5738f1b2f47369d3a9ba5ef47a25c3a"
REGISTRATION_SHA256 = "1b1e4acac7fa246d3b353e7e7e8398123da5e823fc11dedc8dde069543a9cf25"
SELECTOR_ID = "quality_only_integer_numerator_then_smaller_initialization.v1"
MODEL_CODE_FILES = ("backend/training/warehouse_v621_actor.py",
    "backend/training/warehouse_v621_actor_training.py", "backend/training/warehouse_v621_program.py")
LOCK_SCHEMA = "warehouse-v621-selected-common-bc-lock.v1"
ROLES = ("robot_1", "robot_2")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file(root, path, *, expected=None):
    p = Path(path)
    p = p if p.is_absolute() else root/p
    p = p.absolute()
    if p.resolve(strict=True) != p or not p.is_relative_to(root) or not p.is_file() or p.is_symlink():
        raise RuntimeError("BC selection requires a contained regular file without symlinks")
    if expected is not None and p != expected:
        raise RuntimeError("BC evidence path differs from its fixed directory")
    return p


def _pointer(root, path, *, expected=None):
    p = _file(root, path, expected=expected)
    return {"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()}


def _read_pointer(root, ref, *, expected=None):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or not _sha(ref["sha256"]):
        raise RuntimeError("BC evidence requires an exact path/SHA pointer")
    path = _file(root, ref["path"], expected=expected)
    raw = path.read_bytes()
    if str(path) != ref["path"] or sha256(raw).hexdigest() != ref["sha256"]:
        raise RuntimeError("BC evidence SHA changed")
    return json.loads(raw)


def _verify_source_closure(root):
    manifest = json.loads(_file(root,
        "experiments/warehouse/evidence/warehouse-v621-bc-source-manifest-02.json").read_bytes())
    if (not isinstance(manifest, dict) or len(manifest) != 194
            or protocol.digest(manifest) != SOURCE02_SHA256 or protocol.source_files(root) != manifest):
        raise RuntimeError("BC source02 closure changed")
    return manifest


def _checked_result(spec, initialization, recipe):
    from scripts.run_warehouse_v621_bc_development import checked_result
    return checked_result(spec, initialization, recipe)


def _verify_prerequisites(root, spec):
    from backend.training.warehouse_v621_bc_capability import (
        CLEARANCE_SCHEMA, _checked_prerequisites, _strict_equal, reset_contract)
    bound = _checked_prerequisites(root, spec)
    clearance = _read_pointer(root, spec["seed_clearance"])
    expected = {"schema": CLEARANCE_SCHEMA, "candidate_id": protocol.diagnostics.CANDIDATE_ID,
        "batch_id": protocol.BATCH, "approved_for_bc_development": True,
        "source_closure_sha256": SOURCE02_SHA256,
        "model_implementation_sha256": spec["model_implementation_sha256"],
        "baseline_actor_sha256": protocol.diagnostics.ACTOR_SHA256,
        "recipe_sha256": spec["recipe"]["sha256"], "seed_plan_sha256": spec["seed_plan"]["sha256"],
        "authorized_reset_contract_sha256": protocol.digest(reset_contract()),
        "initializations": list(protocol.INITIALIZATIONS), "unresolved": [], "remaining_conflicts": [],
        "fresh_precheck_authorized": False, "PPO_joint_steps": 0, "pilot_opened": False, "publishable": False}
    if any(not _strict_equal(clearance.get(k), v) for k, v in expected.items()):
        raise RuntimeError("BC prerequisite seed clearance differs from the original finite scope")
    bound[spec["seed_clearance"]["path"]] = spec["seed_clearance"]["sha256"]
    return bound


def _implementation_sources():
    paths = (Path(__file__).resolve(), Path(__file__).resolve().parents[2]/"scripts/lock_warehouse_v621_common_bc.py")
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}


def quality_score(result, collection, *, cases):
    """Exact score from all 75 old natural cases; no model/program ranking."""
    if collection.get("cases") != cases or [r.get("case") for r in collection.get("outcomes", ())] != cases:
        raise RuntimeError("quality requires the complete ordered old development cohort")
    selected = [row for row in collection["outcomes"] if row["case"]["original_split"] == "validation"]
    if len(selected) != 75 or any(r["case"]["sampling_mode"] != "sampled" for r in selected):
        raise RuntimeError("quality requires exactly 75 natural validation episodes")
    for row in selected:
        if (any(type(row.get(k)) is not int or row[k] < 0 for k in ("deliveries", "robot_collisions"))
                or type(row.get("shutdown")) is not bool):
            raise RuntimeError("quality counts must be nonnegative integers and shutdown a bool")
    deliveries = sum(r["deliveries"] for r in selected)
    collisions = sum(r["robot_collisions"] for r in selected)
    shutdowns = sum(r["shutdown"] for r in selected)
    expected = {"mean_deliveries": deliveries/75, "mean_collisions": collisions/75,
        "shutdown_episode_rate": shutdowns/75}
    quality = result.get("quality")
    if (not isinstance(quality, dict) or set(quality) != set(expected)
            or any(type(quality[k]) not in (int, float) or not math.isfinite(quality[k])
                   or quality[k] != value for k, value in expected.items())):
        raise RuntimeError("quality summary differs from actual integer episode outcomes")
    passed = deliveries/75 >= 5.5 and collisions/75 <= 6. and shutdowns/75 <= .05
    if result.get("quality_passed") is not passed:
        raise RuntimeError("task quality gate does not recompute")
    return {"score_integer_numerator": 4*deliveries-collisions-20*shutdowns,
        "common_denominator": 300, "episode_count": 75,
        "total_deliveries": deliveries, "total_robot_collisions": collisions,
        "shutdown_episodes": shutdowns, "quality": expected}


def _cohort(results, recipe):
    if not isinstance(results, dict) or set(results) != set(protocol.INITIALIZATIONS):
        raise RuntimeError("all three complete BC initializations are required")
    for initialization, result in results.items():
        if (result.get("initialization") != initialization or result.get("recipe") != recipe
                or result.get("status") != "completed" or type(result.get("PPO_joint_steps")) is not int
                or result["PPO_joint_steps"] != 0 or result.get("publishable") is not False
                or any(type(result.get(k)) is not bool for k in
                    ("passed", "quality_passed", "program_passed", "gradient_probe_passed"))
                or result["passed"] is not all(result[k] for k in
                    ("quality_passed", "program_passed", "gradient_probe_passed"))):
            raise RuntimeError("BC cohort is incomplete, mixed-recipe or has inconsistent gates")


def select_quality_only(main_results, main_collections, *, backup_results=None,
                        backup_collections=None):
    """Pure decision over verified receipts; this function grants no authority."""
    cases = protocol.diagnostics.source_cases()
    _cohort(main_results, "shared_b035")
    if not isinstance(main_collections, dict) or set(main_collections) != set(protocol.INITIALIZATIONS):
        raise RuntimeError("every main candidate requires its complete quality collection")
    main_scores = {i: quality_score(main_results[i], main_collections[i], cases=cases)
        for i in protocol.INITIALIZATIONS}
    if all(r["passed"] for r in main_results.values()):
        recipe, results, collections = "shared_b035", main_results, main_collections
    else:
        if not any(r["quality_passed"] is False for r in main_results.values()):
            raise RuntimeError("program/gradient failure cannot unlock the backup recipe")
        _cohort(backup_results, "shared_b050")
        if not all(r["passed"] for r in backup_results.values()):
            raise RuntimeError("all three backup candidates must pass before selection")
        recipe, results, collections = "shared_b050", backup_results, backup_collections
    if not isinstance(collections, dict) or set(collections) != set(protocol.INITIALIZATIONS):
        raise RuntimeError("every candidate requires its complete quality collection")
    scores = main_scores if recipe == "shared_b035" else {
        i: quality_score(results[i], collections[i], cases=cases) for i in protocol.INITIALIZATIONS}
    winner = min(scores, key=lambda i: (-scores[i]["score_integer_numerator"], i))
    return {"selector_id": SELECTOR_ID, "candidate_recipe": recipe, "selected_initialization": winner,
        "scores": {str(i): scores[i] for i in protocol.INITIALIZATIONS},
        "ranking_uses_only_integer_quality_and_initialization": True}


def _load_cohort(root, spec, specification, recipe, bound):
    results, collections, refs = {}, {}, {}
    directory = Path(spec["output_directory"])
    for initialization in protocol.INITIALIZATIONS:
        folder = directory/f"init-{initialization}"/recipe
        if any((folder/name).exists() for name in ("capability.failed.json", "failure.json", "failed.json")):
            raise RuntimeError("a failed BC attempt cannot be selected")
        result, ref = _checked_result(spec, initialization, recipe)
        actual = _read_pointer(root, ref, expected=folder/"result.json")
        if actual != result or result.get("specification") != specification:
            raise RuntimeError("native BC result is bound to a different specification")
        complete_ref = _pointer(root, folder/"completed.json")
        complete = _read_pointer(root, complete_ref)
        cap_ref = _pointer(root, folder/"capability.completed.json")
        cap = _read_pointer(root, cap_ref)
        if (complete.get("result") != ref or complete.get("capability_completed") != cap_ref
                or complete.get("source_closure_sha256") != SOURCE02_SHA256
                or cap.get("schema") != "warehouse-v621-bc-capability-completed.v1"
                or cap.get("initialization") != initialization or cap.get("recipe") != recipe
                or cap.get("preauthorized_resets") != 1005):
            raise RuntimeError("BC completion pointer chain differs")
        for pointer in (ref, complete_ref, cap_ref, specification):
            bound[pointer["path"]] = pointer["sha256"]
        for key, name in (("success_journal", "native-successful-resets.jsonl"),
                          ("preauthorization_journal", "reset-preauthorization.jsonl")):
            pointer = result[key]
            if _pointer(root, pointer["path"], expected=folder/name) != pointer:
                raise RuntimeError("BC reset journal changed")
            bound[pointer["path"]] = pointer["sha256"]
        collection = _read_pointer(root, result["development_collection"],
            expected=folder/"development.collection.json")
        bound[result["development_collection"]["path"]] = result["development_collection"]["sha256"]
        for role in ROLES:
            pointer = result["roles"][role]["program"]
            if _pointer(root, pointer["path"], expected=folder/(role+".program.json")) != pointer:
                raise RuntimeError("BC program pointer changed")
            bound[pointer["path"]] = pointer["sha256"]
        actor = result["actor"]
        if _pointer(root, actor["path"], expected=folder/"final-bc.actor.npz") != {
                "path": actor["path"], "sha256": actor["sha256"]}:
            raise RuntimeError("BC Actor pointer changed")
        bound[actor["path"]] = actor["sha256"]
        results[initialization], collections[initialization], refs[str(initialization)] = result, collection, ref
    return results, collections, refs


def _checkpoint(root, result, spec, manifest, bound):
    import torch
    from backend.training import warehouse_v621_actor as runtime
    from backend.training import warehouse_v621_actor_training as training
    from backend.training.warehouse_v621_bc_pipeline import array_sha
    folder = Path(spec["output_directory"])/f"init-{result['initialization']}"/result["recipe"]
    training_report = _read_pointer(root, result["training_report"], expected=folder/"bc.training.json")
    if (training_report.get("both_roles_changed") is not True
            or type(training_report.get("external_program_feedback")) is not int
            or training_report["external_program_feedback"] != 0
            or type(training_report.get("external_program_actor_updates")) is not int
            or training_report["external_program_actor_updates"] != 0):
        raise RuntimeError("BC training report lacks both-role updates or has external program training")
    ref = training_report["checkpoint"]
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise RuntimeError("BC checkpoint requires an exact path/SHA pointer")
    path = _file(root, ref["path"], expected=folder/"final-bc.checkpoint.pt")
    raw = path.read_bytes()
    if ref["path"] != str(path) or not _sha(ref.get("sha256")) or sha256(raw).hexdigest() != ref["sha256"]:
        raise RuntimeError("BC checkpoint SHA changed")
    for module, name in ((runtime, MODEL_CODE_FILES[0]), (training, MODEL_CODE_FILES[1])):
        if sha256(Path(module.__file__).read_bytes()).hexdigest() != manifest[name]:
            raise RuntimeError("loaded Actor implementation differs from the pinned BC source")
    saved = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if (not isinstance(saved, dict) or saved.get("schema") != "warehouse-v621-common-bc-checkpoint.v1"
            or saved.get("model_version") != runtime.MODEL_VERSION
            or saved.get("recipe") != result["recipe"] or saved.get("initialization") != result["initialization"]
            or saved.get("source_closure_sha256") != SOURCE02_SHA256
            or saved.get("seed_plan_sha256") != spec["seed_plan"]["sha256"]
            or saved.get("raw_and_regularized_must_clone_exactly") is not True
            or type(saved.get("PPO_joint_steps")) is not int or saved["PPO_joint_steps"] != 0
            or saved.get("actor_artifact") != result["actor"]):
        raise RuntimeError("BC checkpoint source/model/recipe/start identity mismatch")
    bound_value = .35 if result["recipe"] == "shared_b035" else .50
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("BC restoration requires the original float32 constructor dtype")
    # Construction only: every random tensor is overwritten below. CPU RNG
    # state is restored on both success and failure; no optimizer is created.
    before_rng = torch.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            # Seed only the CPU generator used by CPU module constructors;
            # torch.manual_seed would also touch CUDA/MPS global generators.
            torch.default_generator.manual_seed(result["initialization"])
            model = training.WarehouseV621Actor(core_residual_bound=bound_value)
            expected, state = model.state_dict(), saved.get("model_state")
            if any(value.device.type != "cpu" for value in expected.values()):
                raise RuntimeError("BC restoration shell must contain CPU tensors only")
            if not isinstance(state, dict) or set(state) != set(expected):
                raise RuntimeError("BC checkpoint must cover the entire dual Actor and both critics")
            for name, value in state.items():
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.layout != torch.strided or value.shape != expected[name].shape
                        or value.dtype != expected[name].dtype or not torch.isfinite(value).all().item()):
                    raise RuntimeError("BC checkpoint tensor shape/dtype/finiteness mismatch: "+name)
            model.load_state_dict(state, strict=True)
            model.eval()
            if any(not torch.equal(value, model.state_dict()[name]) for name, value in state.items()):
                raise RuntimeError("BC restoration did not exactly cover every saved tensor")
            restored = runtime.FrozenWarehouseV621Actor.from_model(model)
            disk = runtime.FrozenWarehouseV621Actor.load(result["actor"]["path"],
                expected_sha256=result["actor"]["sha256"], expected_role_sha256=result["actor"]["role_sha256"])
            if (restored.role_sha256 != disk.role_sha256
                    or any(role.core_residual_bound != bound_value for role in disk.roles.values())
                    or any(r["topology_frozen_at_step"] != -1 for r in model.constraint_report().values())):
                raise RuntimeError("BC checkpoint differs from the exported full dual Actor or its recipe")
            for role in ROLES:
                actual = array_sha(**{k: v.detach().numpy() for k, v in model.actors[role].state_dict().items()})
                parameters = array_sha(**{k: v.detach().numpy() for k, v in model.actors[role].named_parameters()
                    if v.requires_grad})
                start = training_report.get("post_install_bc_start_parameter_sha256", {}).get(role)
                gradient = training_report.get("role_maximum_bc_gradient_norm", {}).get(role)
                if (training_report.get("trained_state_sha256", {}).get(role) != actual
                        or training_report.get("trained_parameter_sha256", {}).get(role) != parameters
                        or not _sha(start) or start == parameters or type(gradient) not in (int, float)
                        or not math.isfinite(gradient) or gradient <= 0.):
                    raise RuntimeError("BC training report does not match final role tensors or real update evidence")
            state_hash = array_sha(**{name: value.detach().numpy() for name, value in state.items()})
    finally:
        if not torch.equal(before_rng, torch.get_rng_state()):
            raise RuntimeError("checkpoint reconstruction changed the external CPU Torch RNG")
    for pointer in (result["training_report"], ref):
        bound[pointer["path"]] = pointer["sha256"]
    return {"training_report": result["training_report"], "checkpoint": ref,
        "actor": result["actor"], "full_model_state_sha256": state_hash,
        "restored_role_sha256": restored.role_sha256,
        "restoration": {"weights_only": True, "map_location": "cpu", "strict_entire_state": True,
            "constructor_rng_seed": result["initialization"],
            "constructor_rng_backend": "torch.default_generator(CPU)",
            "constructor_device": "cpu", "constructor_dtype": "float32",
            "constructor_rng_purpose": "temporary_empty_shell_all_random_tensors_overwritten_from_consumed_BC_checkpoint",
            "external_cpu_torch_rng_restored": True, "optimizer_created": False,
            "environment_resets": 0, "actor_updates": 0}}


def build_model_lock(*, root, specification_path, registration_path):
    """Return a checked lock payload; never collect, train, fit, or authorize."""
    root = Path(root).resolve(strict=True)
    implementation = _implementation_sources()
    manifest = _verify_source_closure(root)
    registration = _pointer(root, registration_path)
    if registration["sha256"] != REGISTRATION_SHA256:
        raise RuntimeError("pre-result selector registration changed")
    registered = _read_pointer(root, registration)
    if registered.get("selector_id") != SELECTOR_ID or registered.get("execution_authority") is not False:
        raise RuntimeError("selector registration is not the immutable non-executing rule")
    spec_ref = _pointer(root, specification_path)
    spec = _read_pointer(root, spec_ref)
    directory = root/"output/warehouse_v621"/protocol.BATCH
    from backend.training.warehouse_v621_bc_capability import SCHEMA
    if (spec.get("schema") != SCHEMA or spec.get("candidate_id") != protocol.diagnostics.CANDIDATE_ID
            or spec.get("batch_id") != protocol.BATCH or spec.get("source_closure_sha256") != SOURCE02_SHA256
            or spec.get("source_files_sha256") != manifest
            or spec.get("model_implementation_sha256") != protocol.digest({p: manifest[p] for p in MODEL_CODE_FILES})
            or spec.get("output_directory") != str(directory) or spec.get("fresh_precheck_authorized") is not False
            or type(spec.get("PPO_joint_steps")) is not int or spec["PPO_joint_steps"] != 0
            or spec.get("publishable") is not False):
        raise RuntimeError("BC specification source/model/fixed output or authority differs")
    for key, filename, expected in (("recipe", "warehouse_v621_bc_recipe.json", protocol.recipe()),
            ("seed_plan", "warehouse_v621_bc_seed_plan.json", protocol.seed_plan())):
        if _read_pointer(root, spec[key], expected=root/"configs"/filename) != expected:
            raise RuntimeError("BC specification registered recipe or seed declaration differs")
    bound = _verify_prerequisites(root, spec)
    bound.update({p["path"]: p["sha256"] for p in (registration, spec_ref, spec["recipe"], spec["seed_plan"])})
    main, main_collections, main_refs = _load_cohort(root, spec, spec_ref, "shared_b035", bound)
    _cohort(main, "shared_b035")
    control, _, control_refs = _load_cohort(root, spec, spec_ref, "additive_control", bound)
    _cohort(control, "additive_control")
    if all(row["passed"] for row in main.values()):
        candidate, candidate_refs = main, main_refs
        decision = select_quality_only(main, main_collections)
        summary_spec_ref, summary_name = spec_ref, "main.summary.json"
        required_refs = {"additive_control": control_refs, "shared_b035": main_refs}
    else:
        if not any(row["quality_passed"] is False for row in main.values()):
            raise RuntimeError("program/gradient failure cannot unlock the backup recipe")
        backup_spec_ref = _pointer(root, directory/"backup.spec.json")
        backup_spec = _read_pointer(root, backup_spec_ref)
        if backup_spec != {**spec, "main_results": main_refs}:
            raise RuntimeError("backup specification does not bind the same three completed main runs")
        backup, backup_collections, backup_refs = _load_cohort(root, backup_spec, backup_spec_ref, "shared_b050", bound)
        decision = select_quality_only(main, main_collections, backup_results=backup,
            backup_collections=backup_collections)
        candidate, candidate_refs = backup, backup_refs
        summary_spec_ref, summary_name = backup_spec_ref, "backup.summary.json"
        required_refs = {"additive_control": control_refs, "shared_b035": main_refs, "shared_b050": backup_refs}
    summary_ref = _pointer(root, directory/summary_name)
    summary = _read_pointer(root, summary_ref)
    if (summary.get("schema") != "warehouse-v621-bc-development-summary.v1"
            or summary.get("specification") != summary_spec_ref or summary.get("results") != required_refs
            or summary.get("candidate_id") != protocol.diagnostics.CANDIDATE_ID
            or summary.get("source_closure_sha256") != SOURCE02_SHA256
            or summary.get("candidate_recipe") != decision["candidate_recipe"]
            or summary.get("all_three_candidate_initializations_passed") is not True
            or summary.get("main_quality_failure") is not any(not row["quality_passed"] for row in main.values())
            or summary.get("next_stage") != "lock_common_bc_and_implement_fresh_precheck"
            or summary.get("fresh_precheck_authorized") is not False
            or type(summary.get("PPO_joint_steps")) is not int or summary["PPO_joint_steps"] != 0
            or summary.get("publishable") is not False or summary.get("pilot_opened") is not False):
        raise RuntimeError("complete native BC summary does not bind the selected recipe/cohort")
    bound[summary_ref["path"]] = summary_ref["sha256"]
    models = {str(i): _checkpoint(root, candidate[i], spec, manifest, bound) for i in protocol.INITIALIZATIONS}
    selected = models[str(decision["selected_initialization"])]
    for name, expected in bound.items():
        if _pointer(root, name)["sha256"] != expected:
            raise RuntimeError("BC bound evidence changed while selecting the model")
    if _verify_source_closure(root) != manifest or _implementation_sources() != implementation:
        raise RuntimeError("BC source changed while selecting the model")
    return {"schema": LOCK_SCHEMA, "locked": True, "candidate_id": protocol.diagnostics.CANDIDATE_ID,
        "bc_source_closure_sha256": SOURCE02_SHA256, "bc_specification": spec_ref,
        "selector_registration": registration, "native_bc_summary": summary_ref,
        **decision, "candidate_models": models, "candidate_results": candidate_refs,
        "selected_model": selected, "program_method": "shared_actor_topology_core_box_distribution_v1",
        "programs_require_new_fresh_fit": True,
        "implementation_sources": implementation,
        "bound_evidence_sha256": dict(sorted(bound.items())),
        "scope": "read_only_common_bc_selection_not_execution_authorization",
        "subsequent_validation_requires_new_source_seed_and_scope_capability": True,
        "environment_resets": 0, "actor_updates": 0, "fit_calls": 0, "PPO_joint_steps": 0,
        "fresh_precheck_authorized": False, "sealed_audit_authorized": False,
        "pilot_opened": False, "publishable": False}


def write_model_lock(*, root, specification_path, registration_path, output_path):
    """Exclusively create a read-only JSON lock inside its isolated directory."""
    root = Path(root).resolve(strict=True)
    output = Path(output_path)
    output = output if output.is_absolute() else root/output
    output = output.absolute()
    expected_parent = root/"output/warehouse_v621/common-bc-locks"
    if (output.parent != expected_parent or output.suffix != ".json" or output.exists()
            or output.is_symlink() or output.parent.resolve() != output.parent):
        raise RuntimeError("model lock output must be a new JSON in common-bc-locks without symlinks")
    payload = build_model_lock(root=root, specification_path=specification_path, registration_path=registration_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise RuntimeError("model lock output directory became a symlink")
    with output.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    output.chmod(0o444)
    return _pointer(root, output)
