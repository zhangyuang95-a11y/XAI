"""Fail-closed kitchen release gates and deployment-safe version identity."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/cooperative_kitchen/v3-id-pilot"
REQUIRED_GATES = ("training", "extraction", "qa", "calibration", "questionnaire", "protocol", "remote_load", "recovery")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_hash(config):
    """Bind provider/model/prompt semantics without recording a credential."""
    return hashlib.sha256(json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()


def runtime_hash():
    directories = (ROOT / "env/cooperative_kitchen", ROOT / "backend/cooperative_kitchen",
                   ROOT / "ui/cooperative_kitchen_web", ROOT / "core")
    paths = [p for directory in directories for p in directory.rglob("*") if p.suffix in {".py", ".js", ".html", ".css", ".svg"}]
    paths += [ROOT / "ui/cooperative_kitchen_server.py", ROOT / "ui/cooperative_kitchen_store.py"]
    paths += [ROOT / path for path in ("backend/__init__.py", "env/__init__.py", "ui/__init__.py",
        "backend/adapters/__init__.py", "backend/adapters/base.py", "requirements-kitchen.txt")]
    return hashlib.sha256(b"".join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in sorted(paths) if p.is_file())).hexdigest()


def _read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def load_release(output=DEFAULT_OUTPUT):
    output = Path(output)
    manifest = _read(output / "manifest.json", {})
    missing = []
    def artifact(key):
        entry = manifest.get("artifacts", {}).get(key)
        if not entry:
            missing.append(key)
            return None
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or file_hash(path) != entry.get("sha256"):
            missing.append(key + "_hash")
            return None
        return path
    actor = artifact("actor")
    program = artifact("program")
    scenarios_file = artifact("scenarios")
    questions_file = artifact("questionnaire")
    reports = {}
    for name in REQUIRED_GATES:
        path = artifact(name + "_report")
        report = _read(path, {}) if path else {}
        reports[name] = report
        if report.get("passed") is not True:
            missing.append(name + "_gate")
        if name in {"training", "extraction", "qa", "questionnaire", "remote_load"} and actor and report.get("actor_sha256") != file_hash(actor):
            missing.append(name + "_actor_binding")
        if name in {"qa", "remote_load"} and report.get("mode") != "real_remote":
            missing.append(name + "_real_api_required")
    code_hash = runtime_hash()
    if manifest.get("runtime_sha256") != code_hash:
        missing.append("runtime_version")
    for name in ("protocol", "recovery", "remote_load", "qa"):
        if reports[name].get("runtime_sha256") != code_hash:
            missing.append(name + "_runtime_binding")
    for name, path, key in (("extraction", program, "program_sha256"), ("qa", program, "program_sha256"),
                            ("calibration", scenarios_file, "scenarios_sha256"), ("questionnaire", questions_file, "questionnaire_sha256")):
        if path and reports[name].get(key) != file_hash(path):
            missing.append(name + "_artifact_binding")
    from .llm import KitchenLLMClient
    llm_client = KitchenLLMClient()
    llm_config = llm_client.config
    required_key_env = llm_client.required_key_env
    # A rolling provider alias may be used for an explicitly labelled internal
    # pilot, but it cannot satisfy the frozen-model requirement of a formal run.
    if llm_config.get("model_version_pinned") is not True:
        missing.append("qa_model_snapshot_unpinned")
    if manifest.get("qa_configuration") != llm_config:
        missing.append("qa_configuration_version")
    for name in ("qa", "remote_load"):
        if reports[name].get("qa_configuration") != llm_config:
            missing.append(name + "_configuration_binding")
    if manifest.get("status") == "formal_ready" and not manifest.get("human_pilot_review", {}).get("frozen_after_pilot"):
        missing.append("human_pilot_review")
    for name in ("DATABASE_URL", "KITCHEN_ADMIN_KEY"):
        if not os.environ.get(name): missing.append(name)
    if not llm_client.configured: missing.append(required_key_env)
    if os.environ.get("DATABASE_URL") and not os.environ["DATABASE_URL"].startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        missing.append("postgresql_required")
    if not actor:
        candidates = sorted(output.glob("seed_*/actor_*.npz"))
        actor = candidates[-1] if candidates else None
    scenarios = _read(scenarios_file, {}) if scenarios_file else {}
    question_bank = _read(questions_file, []) if questions_file else []
    survey_scales = question_bank.get("scales", []) if isinstance(question_bank, dict) else []
    scale_range = question_bank.get("scale_range", [1, 7]) if isinstance(question_bank, dict) else [1, 7]
    scale_anchors = question_bank.get("scale_anchors", {}) if isinstance(question_bank, dict) else {}
    if isinstance(question_bank, dict): question_bank = question_bank.get("items", [])
    pair_map = scenarios.get("pairs", {"X": ["base_empty", "detour_inprogress", "asymmetric_congestion"],
                                       "Y": ["mirror_empty", "asymmetric_inprogress", "detour_congestion"]})
    if isinstance(pair_map, list):
        pair_map = {label: [pair["scenarios"][index] for pair in pair_map] for index, label in enumerate(("X", "Y"))}
    return {"study_ready": not missing and manifest.get("status") in {"pilot_ready", "formal_ready"},
            "status": manifest.get("status", "candidate"), "policy_kind": "neural" if actor else "program",
            "missing_configuration": sorted(set(missing)), "actor_path": str(actor) if actor else None,
            "program_path": str(program) if program else None, "question_bank": question_bank,
            "scenarios": pair_map, "consent": manifest.get("consent", {}),
            "qa_configuration": llm_config, "qa_required_key_env": required_key_env, "qa_configured": llm_client.configured,
            "survey_scales": survey_scales, "scale_range": scale_range, "scale_anchors": scale_anchors,
            "versions": {"ui": "cooperative_kitchen_web_v3_id_pilot", "runtime_sha256": code_hash,
                         "actor_sha256": file_hash(actor) if actor else "program-baseline",
                         "program_sha256": file_hash(program) if program else None,
                         "protocol": "cooperative_kitchen_user_id_pilot_v3", "qa_configuration_sha256": configuration_hash(llm_config),
                         "manifest": file_hash(output / "manifest.json") if manifest else None},
            "manifest": manifest, "reports": reports}
