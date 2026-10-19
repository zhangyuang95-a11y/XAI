"""Explicit replacement of an engineering-interrupted spent-data batch.

The original protocol and its source closure are unchanged.  This protocol
permits a new complete capacity comparison, never a resume or a combination of
old and new results.  It grants no collection, training or sealed evaluation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.warehouse.v622_protocol import *  # Re-export scientific API unchanged.
from experiments.warehouse import v622_protocol as original

ROOT = Path(__file__).resolve().parents[2]
BATCH_ID = "capacity-02"
FALLBACK_BATCH_ID = "fallback-02"
RECIPE_PATH = "configs/warehouse_v622_recovery02_recipe.json"
ORIGINAL_RECIPE_PATH = original.RECIPE_PATH
ORIGINAL_BATCH_PATH = "output/warehouse_v622/capacity-01"
INTERRUPTION_PATH = ORIGINAL_BATCH_PATH + "/interrupted.json"
INTERRUPTION_SHA256 = "155ae082c265d64bc947cb229c0600a15cd5b71670f3daa58c58548bbce0a1a0"
ORIGINAL_PLAN_SHA256 = "b85f8e09befa918e4c9a02229d7d037b8112792092b8576263f77263475bfdc2"
ORIGINAL_MANIFEST_SHA256 = "d525f6caa89bce874374c88039a7d9a92d9c85207c2ae425d3e5a9e1f6970b99"
ORIGINAL_SOURCE_SHA256 = "fb969ab32556b2a5d3163ed8a4e77874b0176ca733870004507f46e0b90545b6"
RECOVERY_SOURCE_FILES = (
    "experiments/warehouse/v622_recovery_protocol.py",
    "scripts/run_warehouse_v622_capacity_recovery.py",
    "scripts/audit_warehouse_v622_capacity_recovery.py",
    "scripts/execute_warehouse_v622_recovery_batch.py",
    RECIPE_PATH,
)


def output_directory(stage, root=None):
    if stage not in ("primary", "fallback"):
        raise ValueError("unknown recovery capacity stage")
    return (Path(ROOT if root is None else root).resolve() / OUTPUT_ROOT /
            (BATCH_ID if stage == "primary" else FALLBACK_BATCH_ID))


def _read(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("recovery evidence must be a regular, non-symlink file")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise RuntimeError("recovery evidence SHA mismatch: " + str(path))
    return raw


def _bound(root, ref, relative):
    path = (Path(root) / relative).resolve()
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or ref["path"] != str(path)):
        raise RuntimeError("recovery evidence path differs from original batch")
    return _read(path, ref["sha256"])


def _zero_flags(value, names, expected):
    if any(type(value.get(k)) is not type(expected) or value[k] != expected for k in names):
        raise RuntimeError("recovery evidence contains an unexpected execution claim")


def checked_interruption(root):
    """Verify the complete, fixed 9-complete/3-interrupted disposition.

    Old logs and result bytes are evidence of work already spent; their scores
    are never imported into the replacement batch's selection.  Historical
    process inspection is verified as recorded, not rerun while batch02 works.
    """
    root = Path(root).resolve()
    raw = _read(root / INTERRUPTION_PATH, INTERRUPTION_SHA256)
    receipt = json.loads(raw)
    _zero_flags(receipt, ("environment_resets", "actor_updates", "PPO_joint_steps"), 0)
    _zero_flags(receipt, ("complete_batch", "heldout_used", "sealed_audit_used",
                         "permits_training", "pilot_opened", "scientific_method_failure_claimed",
                         "scientific_timeout_claimed"), False)
    if (receipt.get("schema") != "warehouse-v622-engineering-interruption.v1"
            or receipt.get("candidate_id") != CANDIDATE_ID or receipt.get("batch_id") != "capacity-01"
            or type(receipt.get("completed_jobs")) is not int or receipt["completed_jobs"] != 9
            or type(receipt.get("interrupted_jobs")) is not int or receipt["interrupted_jobs"] != 3
            or receipt.get("selected_arm") is not None
            or receipt.get("source_closure_sha256") != ORIGINAL_SOURCE_SHA256
            or receipt.get("recovery_policy") != "new full 12-job batch with new source identity; no cross-batch selection or stitching; old compute reported separately"
            or receipt.get("process_inspection", {}).get("checkpoint_resume_available") is not False
            or type(receipt.get("process_inspection", {}).get("matching_scientific_processes")) is not int
            or receipt["process_inspection"]["matching_scientific_processes"] != 0):
        raise RuntimeError("original batch is not the registered engineering interruption")
    plan_raw = _bound(root, receipt.get("plan"), ORIGINAL_BATCH_PATH + "/plan.json")
    manifest_raw = _bound(root, receipt.get("source_manifest"), ORIGINAL_BATCH_PATH + "/source-manifest.json")
    if (sha256(plan_raw).hexdigest() != ORIGINAL_PLAN_SHA256
            or sha256(manifest_raw).hexdigest() != ORIGINAL_MANIFEST_SHA256):
        raise RuntimeError("original plan or source manifest binding changed")
    plan, manifest = json.loads(plan_raw), json.loads(manifest_raw)
    expected_jobs = original.jobs("primary")
    if (plan.get("schema") != "warehouse-v622-capacity-plan.v1" or plan.get("stage") != "primary"
            or plan.get("candidate_id") != CANDIDATE_ID or plan.get("jobs") != expected_jobs
            or plan.get("scientific_retry_permitted") is not False
            or plan.get("identity", {}).get("source_closure_sha256") != ORIGINAL_SOURCE_SHA256
            or manifest.get("source_closure_sha256") != ORIGINAL_SOURCE_SHA256
            or manifest.get("files_sha256") != plan["identity"].get("source_files_sha256")
            or original.digest(manifest["files_sha256"]) != ORIGINAL_SOURCE_SHA256):
        raise RuntimeError("original scientific plan or source closure changed")
    records = receipt.get("jobs")
    if not isinstance(records, list) or len(records) != 12 or [r.get("job") for r in records] != expected_jobs:
        raise RuntimeError("interruption disposition must include the original twelve ordered jobs")
    complete = interrupted = 0
    for row in records:
        job = row["job"]
        relative = ORIGINAL_BATCH_PATH + "/jobs/" + job["job_id"]
        started = json.loads(_bound(root, row.get("started"), relative + "/started.json"))
        _bound(root, row.get("log"), ORIGINAL_BATCH_PATH + "/logs/" + job["job_id"] + ".log")
        if started.get("job") != job or started.get("plan_sha256") != ORIGINAL_PLAN_SHA256:
            raise RuntimeError("original job start binding changed")
        if (root / relative / "failed.json").exists():
            raise RuntimeError("failed job cannot be represented as an engineering interruption")
        if row.get("status") == "completed_before_interruption":
            result_raw = _bound(root, row.get("result"), relative + "/result.json")
            result = json.loads(result_raw)
            completion = json.loads(_bound(root, row.get("completion"), relative + "/completed.json"))
            if (result.get("job") != job or result.get("plan_sha256") != ORIGINAL_PLAN_SHA256
                    or result.get("status") not in ("completed", "scientific_budget_exhausted")
                    or completion.get("plan_sha256") != ORIGINAL_PLAN_SHA256
                    or completion.get("result_sha256") != sha256(result_raw).hexdigest()):
                raise RuntimeError("completed old job is not bound to its own result")
            complete += 1
        elif row.get("status") == "interrupted_without_search_checkpoint":
            folder = root / relative
            if (job["arm"] != "A32" or row.get("result") is not None or row.get("completion") is not None
                    or (folder / "result.json").exists() or (folder / "completed.json").exists()
                    or any("checkpoint" in p.name.lower() for p in folder.rglob("*"))
                    or row.get("unlogged_in_flight_work") != "unknown; logged counters are lower bounds only"):
                raise RuntimeError("interrupted job has completion or checkpoint evidence")
            counts = row.get("last_logged_progress", {})
            if (set(counts) != {"convex_solver_calls", "global_rounds", "leaf_api_calls"}
                    or any(type(n) is not int or n < 0 for n in counts.values())):
                raise RuntimeError("interrupted work counters must be explicit lower bounds")
            interrupted += 1
        else:
            raise RuntimeError("unknown old job disposition")
    if (complete, interrupted) != (9, 3):
        raise RuntimeError("recovery requires exactly nine complete and three interrupted jobs")
    if (root / ORIGINAL_BATCH_PATH / "summary.json").exists() or (root / ORIGINAL_BATCH_PATH / "independent-audit.json").exists():
        raise RuntimeError("an incomplete old batch must not acquire a complete summary or audit")
    return receipt


def executable_recovery_recipe(root):
    receipt = checked_interruption(root)
    old_recipe_sha = original.validate_recipe(root)
    return {
        "schema": "warehouse-v622-capacity-recovery-recipe.v1",
        "candidate_id": CANDIDATE_ID, "batch_id": BATCH_ID, "fallback_batch_id": FALLBACK_BATCH_ID,
        "original_protocol_recipe": {"path": ORIGINAL_RECIPE_PATH, "sha256": old_recipe_sha},
        "original_source_closure_sha256": ORIGINAL_SOURCE_SHA256,
        "interruption": {"path": INTERRUPTION_PATH, "sha256": INTERRUPTION_SHA256},
        "original_plan": {"path": ORIGINAL_BATCH_PATH + "/plan.json", "sha256": ORIGINAL_PLAN_SHA256},
        "original_source_manifest": {"path": ORIGINAL_BATCH_PATH + "/source-manifest.json", "sha256": ORIGINAL_MANIFEST_SHA256},
        "old_disposition": {"completed_jobs": 9, "interrupted_jobs": 3,
            "interrupted_job_ids": [r["job"]["job_id"] for r in receipt["jobs"] if r["status"] == "interrupted_without_search_checkpoint"],
            "unlogged_in_flight_work": "unknown; logged counters are lower bounds only"},
        "replacement": {"full_primary_jobs": original.jobs("primary"), "conditional_fallback_jobs": original.jobs("fallback"),
            "reuse_old_completed_jobs": False, "resume_partial_search": False,
            "cross_batch_selection": False, "cross_batch_stitching": False,
            "report_old_compute_separately": True, "automatic_scientific_retry": False,
            "fallback_requires_complete_independently_audited_new_primary_failure": True},
        "scientific_arguments_unchanged": {"folds_sha256": digest(original.grouped_folds()),
            "actor_sha256": ACTOR_SHA256, "actor_role_sha256": ROLE_SHA256,
            "gates": GATES, "limits": LIMITS, "budgets": round_budget_contract()},
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "fresh_or_sealed_evaluation_authorized": False, "pilot_opened": False,
    }


def write_recipe(root):
    """Exclusively write metadata; no Actor/trace load, fit, or execution."""
    root = Path(root).resolve()
    value = executable_recovery_recipe(root)
    path = root / RECIPE_PATH
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return path


def validate_recipe(root):
    root = Path(root).resolve()
    raw = (root / RECIPE_PATH).read_bytes()
    expected = executable_recovery_recipe(root)
    if json.loads(raw) != expected:
        raise RuntimeError("immutable recovery recipe changed")
    return digest({"original_recipe_sha256": expected["original_protocol_recipe"]["sha256"],
                   "recovery_recipe_sha256": sha256(raw).hexdigest()})


def source_closure_files(root):
    root = Path(root).resolve()
    files = original.source_closure_files(root)
    if original.digest(files) != ORIGINAL_SOURCE_SHA256:
        raise RuntimeError("original frozen scientific source changed")
    files.update({p: sha256((root / p).read_bytes()).hexdigest() for p in RECOVERY_SOURCE_FILES})
    return dict(sorted(files.items()))


def source_closure_sha256(root):
    return digest(source_closure_files(root))
