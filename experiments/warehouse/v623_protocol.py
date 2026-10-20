"""Frozen v6.23 A32 refinement contract on the already spent v6.20 cohort."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.warehouse import v622_recovery_protocol as old

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "warehouse-v623-a32-refine-01"
OUTPUT = ROOT / "output/warehouse_v623" / CANDIDATE
RECIPE = "configs/warehouse_v623_recipe.json"
PRIOR_SUMMARY_SHA256 = "dbb6702e32e75c17d26822eac641f5fc0e158cdad5379050493b0ccfffadecff"
PRIOR_AUDIT_SHA256 = "d04573135d22ad01e559ee4643bcd3a2607b074a9f6b0b19952c62b1604cd884"
NEW_FILES = (RECIPE, "experiments/warehouse/v623_protocol.py",
    "backend/training/warehouse_v623_search.py",
    "scripts/run_warehouse_v623_refinement.py")
SPLITS = old.SPLITS
ROLES = old.ROLES


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def source_files(root=ROOT):
    root = Path(root).resolve()
    files = old.source_closure_files(root)
    files.update({p: sha256((root / p).read_bytes()).hexdigest() for p in NEW_FILES})
    return dict(sorted(files.items()))


def source_sha(root=ROOT):
    return digest(source_files(root))


def recipe(root=ROOT):
    raw = (Path(root) / RECIPE).read_bytes()
    value = json.loads(raw)
    if (value.get("schema") != "warehouse-v623-a32-refinement-recipe.v1"
            or value.get("candidate_id") != CANDIDATE
            or value.get("stage_A", {}).get("order") != ["S1", "S2_if_S1_fails"]
            or value.get("future_gates", {}).get("pilot_open") is not False):
        raise RuntimeError("unregistered v6.23 recipe")
    return sha256(raw).hexdigest()


def jobs():
    return [{"job_id": f"fold-{fold}-{role}", "fold": fold, "role": role}
            for fold in range(3) for role in ROLES]


def prior(job):
    return ROOT / "output/warehouse_v622/capacity-02/jobs" / (
        job["job_id"] + "-A32")


def checked_prior(job):
    summary_path = ROOT / "output/warehouse_v622/capacity-02/summary.json"
    audit_path = ROOT / "output/warehouse_v622/capacity-02/independent-audit.json"
    if (sha256(summary_path.read_bytes()).hexdigest() != PRIOR_SUMMARY_SHA256
            or sha256(audit_path.read_bytes()).hexdigest() != PRIOR_AUDIT_SHA256):
        raise RuntimeError("audited historical A32 batch changed")
    summary = json.loads(summary_path.read_bytes())
    audit = json.loads(audit_path.read_bytes())
    if audit.get("passed") is not True or summary.get("selected_arm") is not None:
        raise RuntimeError("historical A32 was not the complete audited failed batch")
    folder = prior(job)
    result_path = folder / "result.json"
    result = json.loads(result_path.read_bytes())
    completion = json.loads((folder / "completed.json").read_bytes())
    if (result.get("status") != "completed" or result.get("job", {}).get("arm") != "A32"
            or result.get("job", {}).get("fold") != job["fold"]
            or result.get("job", {}).get("role") != job["role"]
            or completion.get("result_sha256") != sha256(result_path.read_bytes()).hexdigest()
            or result.get("actor_weights_unchanged") is not True):
        raise RuntimeError("historical A32 result binding invalid")
    if summary.get("jobs", {}).get(job["job_id"] + "-A32") != {
            "path": str(result_path.resolve()),
            "sha256": sha256(result_path.read_bytes()).hexdigest()}:
        raise RuntimeError("historical A32 receipt absent from pinned batch")
    refs = {}
    for name in ("structure", "search_report", "program"):
        ref = result[name]
        path = folder / ("search-report.json" if name == "search_report" else name + ".json")
        if ref["path"] != str(path.resolve()) or sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise RuntimeError("historical A32 asset binding invalid: " + name)
        refs[name] = ref
    return {"result": {"path": str(result_path.resolve()),
                       "sha256": sha256(result_path.read_bytes()).hexdigest()}, **refs}


def folder(stage):
    if stage not in ("S1", "S2"):
        raise ValueError("unknown stage")
    return OUTPUT / stage
