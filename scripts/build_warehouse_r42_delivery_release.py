"""Build the compact r4.5 internal-pilot Render release."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r4_question_bank as question_api
from backend.warehouse_r45_runtime import R45WarehouseEnv, R45WarehouseRuntime
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.r41_diagnostic_conflict import diagnostic_scene_fingerprint
from env.warehouse_native.r44_charger import SNAPSHOT_KEY, VERSION as CHARGER_VERSION
from env.warehouse_native.partners import partner_action
from ui import warehouse_alignment_r42_release as release_api
from ui import warehouse_alignment_r42_tutorial as tutorial_api


VERSION = "warehouse-r45-internal-pilot-release-builder.v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, compact=False):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":") if compact else None,
                   indent=None if compact else 2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime(actor_path, protocol_path, manifest_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = deepcopy(manifest)
    claimed = content.pop("content_sha256", None)
    return R45WarehouseRuntime(
        actor_path,
        training_protocol_path=protocol_path,
        manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor_path),
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(manifest),
    )


def migrate_manifest(payload):
    """Bind every frozen scene to 3% movement and resumable rule state."""
    result = deepcopy(payload)
    migrated = {}
    for rows in result.get("splits", {}).values():
        for index, entry in enumerate(rows):
            identity = (str(entry.get("id")), str(entry.get("fingerprint")))
            if identity in migrated:
                rows[index] = deepcopy(migrated[identity])
                continue
            scene = deepcopy(entry)
            snapshot = deepcopy(scene["snapshot"])
            snapshot["configuration"]["move_battery_cost"] = 3.0
            snapshot.pop("r43_shared_charger", None)
            snapshot[SNAPSHOT_KEY] = {
                "version": CHARGER_VERSION,
                "streaks": {"robot_1": 0, "robot_2": 0},
                "penalized": {"robot_1": False, "robot_2": False},
                "last_event": None,
            }
            env = R45WarehouseEnv(
                collaborative_study_config(move_battery_cost=3.0))
            env.restore(snapshot)
            scene["snapshot"] = env.snapshot()
            scene["fingerprint"] = diagnostic_scene_fingerprint(env)
            migrated[identity] = deepcopy(scene)
            rows[index] = scene
    result.pop("content_sha256", None)
    result["content_sha256"] = digest(result)
    return result


def _agent_position(snapshot):
    return next(tuple(agent["position"]) for agent in snapshot["state"]["agents"]
                if agent["agent_id"] == "robot_2")


def build_bank(runtime, scenarios):
    scenes = scenarios["splits"]["play"][1:]
    candidates = []
    for scene_index, scene in enumerate(scenes):
        env = runtime.environment(scene)
        rng = random.Random(26091500 + scene_index)
        target_frames = {5 + scene_index, 13 + scene_index}
        for _ in range(max(target_frames)):
            player = partner_action(env, "robot_1", "skilled", rng)
            runtime.step(env, player)
            if env.done:
                break
            if env.state.frame not in target_frames:
                continue
            snapshot = deepcopy(env.snapshot())
            actions, _ = runtime.decision(env)
            counterfactual = runtime.counterfactual(snapshot, ["WAIT"], steps=3)
            if counterfactual["steps_executed"] != 3:
                continue
            finish = _agent_position(counterfactual["transitions"][-1]["after"])
            candidates.append({
                "scene": scene, "frame": int(env.state.frame),
                "preview": question_api._preview(env),
                "action": actions["robot_2"],
                "finish": finish,
                "start": tuple(env.state.by_id("robot_2").position),
            })
    if len(candidates) < 8:
        raise ValueError("insufficient r4.2 question candidates")
    next_rows = candidates[:4]
    wait_rows = candidates[4:8]
    items = []
    for index, row in enumerate(next_rows, 1):
        items.append({
            "id": f"r42_next_{index}", "kind": "next_action",
            "scenario_id": row["scene"]["id"], "frame": row["frame"],
            "preview": row["preview"], "answer": row["action"],
            "options": question_api._next_options(),
            "prompt": deepcopy(question_api.NEXT_PROMPT),
        })
    for index, row in enumerate(wait_rows, 1):
        items.append({
            "id": f"r42_wait_{index}", "kind": "wait_three",
            "scenario_id": row["scene"]["id"], "frame": row["frame"],
            "preview": row["preview"],
            "answer": f"{row['finish'][0]},{row['finish'][1]}",
            "options": question_api._passable_options(
                runtime.environment(row["scene"]), row["finish"], row["start"],
                260915 + index,
            ),
            "prompt": deepcopy(question_api.WAIT_PROMPT),
        })
    payload = {
        "version": release_api.R42CompactQuestionBank.VERSION,
        "actor_sha256": runtime.actor_sha256,
        "runtime_signature": runtime.signature,
        "items": items,
        "selection_split": "frozen_play_scenes",
        "next_action_items": 4, "wait_three_items": 4,
        "program_controls_answers": False,
    }
    payload["content_sha256"] = release_api._digest(payload)
    release_api.R42CompactQuestionBank(payload, runtime=runtime)
    return payload


def build(args):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    actor = Path(args.actor).expanduser().resolve()
    training_report = Path(args.training_report).expanduser().resolve()
    behavior_report = Path(args.behavior_report).expanduser().resolve()
    program = Path(args.program).expanduser().resolve()
    old_package = Path(args.source_release).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="warehouse-r42-builder-") as temporary_name:
        temporary = Path(temporary_name).resolve()
        with zipfile.ZipFile(old_package) as archive:
            protocol_raw = (
                Path(args.training_protocol).expanduser().resolve().read_bytes()
                if args.training_protocol else
                archive.read("artifacts/training_protocol.json")
            )
            runtime_manifest_raw = (
                Path(args.runtime_manifest).expanduser().resolve().read_bytes()
                if args.runtime_manifest else
                archive.read("artifacts/runtime_manifest.json")
            )
        protocol = temporary / "training_protocol.json"
        runtime_manifest = temporary / "runtime_manifest.json"
        protocol.write_bytes(protocol_raw)
        migrated_manifest = migrate_manifest(json.loads(runtime_manifest_raw))
        write_json(runtime_manifest, migrated_manifest, compact=True)
        runtime = _runtime(actor, protocol, runtime_manifest)
        scenarios = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        bank = build_bank(runtime, scenarios)
        tutorial = tutorial_api.build_tutorial(scenarios)
        tutorial_receipt = tutorial_api.validate_tutorial(
            tutorial, scenarios["splits"]["play"][0]
        )
        bank_path = output / "question_bank.json"
        tutorial_path = output / "tutorial.json"
        protocol_path = output / "training_protocol.json"
        runtime_manifest_path = output / "runtime_manifest.json"
        write_json(bank_path, bank, compact=True)
        write_json(tutorial_path, tutorial, compact=True)
        shutil.copyfile(protocol, protocol_path)
        shutil.copyfile(runtime_manifest, runtime_manifest_path)
        package = output / "warehouse_r45_internal_pilot_release.zip"
        receipt = release_api.build_standalone_package(
            output_path=package,
            actor=actor, protocol=protocol_path,
            runtime_manifest=runtime_manifest_path,
            program=program, question_bank=bank_path, tutorial=tutorial_path,
            training_report=training_report, behavior_report=behavior_report,
        )
        encoded = output / "warehouse_r45_internal_pilot_release.b64"
        receipt["base64_sha256"] = release_api.write_base64(package, encoded)
        receipt.update({
            "version": VERSION,
            "program_sha256": file_hash(program),
            "question_bank_sha256": file_hash(bank_path),
            "tutorial_sha256": file_hash(tutorial_path),
            "tutorial_signature": tutorial_receipt["tutorial_signature"],
            "runtime_signature": runtime.signature,
            "source_release_sha256": file_hash(old_package),
        })
        write_json(output / "release_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--behavior-report", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--training-protocol")
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
