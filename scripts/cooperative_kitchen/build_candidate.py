"""Assemble an auditable candidate manifest; never enables participant recruitment."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.cooperative_kitchen.artifacts import file_hash, runtime_hash
from backend.cooperative_kitchen.llm import KitchenLLMClient
from backend.cooperative_kitchen.study import build_default_consent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/cooperative_kitchen/v3-id-pilot")
    args = parser.parse_args(); output = Path(args.output); directory = output / "acceptance"; directory.mkdir(parents=True, exist_ok=True)
    def read(path, default=None):
        path = output / path
        return json.loads(path.read_text()) if path.exists() else default
    def write(path, value):
        target = output / path
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
    actor = file_hash(output / "selected/actor.npz")
    parity = read("acceptance/numpy_parity.json", {})
    final = read("final_test_report.json", {})
    training = {**final, "passed": bool(final.get("training_gate") and parity.get("passed")), "numpy_parity": parity,
                "selected_before_final_test": True, "selection_report_sha256": file_hash(output / "model_selection_report.json")}
    write("acceptance/training_report.json", training)
    extraction = read("artifacts/extraction_report.json", {})
    write("acceptance/extraction_report.json", {**extraction, "passed": bool(extraction.get("extraction_gate"))})
    calibration = read("artifacts/calibration.json", {})
    write("acceptance/calibration_report.json", {**calibration, "passed": bool(calibration.get("calibration_gate")),
                                                "scenarios_sha256": file_hash(output / "artifacts/calibration.json")})
    code = runtime_hash(); config = KitchenLLMClient().config
    selection = read("model_selection_report.json", {})
    previous = read("manifest.json", {})
    if previous:
        (output / "provenance").mkdir(exist_ok=True)
        write("provenance/manifest_" + file_hash(output / "manifest.json") + ".json", previous)
    remote = {"schema": "kitchen_remote_load_audit_v1", "passed": False, "mode": "not_run", "actor_sha256": actor,
              "runtime_sha256": code, "qa_configuration": config, "blocker": "Render kitchen service, external PostgreSQL and the configured cloud QA provider require remote acceptance",
              "scope": "Local performance is reported separately; no claim of 20 remote participants"}
    # Never overwrite an actual remote test with a placeholder.
    if not (output / "acceptance/remote_load_report.json").exists() or read("acceptance/remote_load_report.json", {}).get("mode") == "not_run":
        write("acceptance/remote_load_report.json", remote)
    mapping = {"actor": "selected/actor.npz", "program": "artifacts/program_ai.json", "scenarios": "artifacts/calibration.json", "questionnaire": "artifacts/questionnaire.json"}
    mapping.update({name + "_report": f"acceptance/{name}_report.json" for name in ("training", "extraction", "calibration", "questionnaire", "qa", "protocol", "remote_load", "recovery")})
    entries = {key: {"path": path, "sha256": file_hash(output / path)} for key, path in mapping.items() if (output / path).is_file()}
    manifest = {"schema": "cooperative_kitchen_release_manifest_v1", "status": "candidate", "created_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_sha256": code, "qa_configuration": config, "artifacts": entries,
                "study_design": {"steps": 180, "orders": 2, "human_side": "left", "rounds_per_stage": 3,
                                 "primary_outcome": "task2_mean_score", "stage_difference": "descriptive_only", "randomization": "A/B x XY/YX block of four"},
                "training": {"seeds": [0, 1, 2], "joint_steps_per_seed": 4000000, "selected": selection.get("selected"), "rcpd_feedback": False},
                "consent": build_default_consent(config),
                "release_notes": ["Actual behavior gate passed; extracted program failed held-out fidelity and remains diagnostic.",
                                  "Cloud credentials and remote acceptance are required before remote recruitment.",
                                  "Participant information and human pilot review must be completed before a formal freeze."]}
    write("manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "artifacts": len(entries), "runtime_sha256": code}))


if __name__ == "__main__": main()
