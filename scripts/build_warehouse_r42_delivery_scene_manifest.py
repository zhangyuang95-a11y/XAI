#!/usr/bin/env python3
"""Freeze six unseen high-conflict scenes for the r4.2 internal pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


SELECTED_IDS = (
    "diagnostic_final_test_0037",  # family 01
    "diagnostic_final_test_0023",  # family 02
    "diagnostic_final_test_0062",  # family 03
    "diagnostic_final_test_0021",  # family 04
    "diagnostic_final_test_0042",  # family 05
    "diagnostic_final_test_0012",  # family 06
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--parent-runtime-manifest", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    full_path = Path(args.full_manifest).resolve()
    parent_path = Path(args.parent_runtime_manifest).resolve()
    screen_path = Path(args.screen).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    full = json.loads(full_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    by_id = {scene["id"]: scene for scene in full["splits"]["final_test"]}
    metrics = {row["scene"]["id"]: row for row in screen["rows"]}
    selected = [deepcopy(by_id[scene_id]) for scene_id in SELECTED_IDS]
    families = [scene["family_id"] for scene in selected]
    if families != [f"conflict_family_{index:02d}" for index in range(1, 7)]:
        raise ValueError("selected scenes must cover the six conflict families")
    evidence = []
    for scene in selected:
        row = metrics[scene["id"]]
        episodes = row["episodes"]
        skilled = next(item for item in episodes if item["profile"] == "skilled")
        if (skilled["robot_2_deliveries"] < 2
                or any(item["robot_2_shutdowns"] for item in episodes)
                or max(item["longest_collision_streak"] for item in episodes) > 10
                or not scene["initial_conflict"]["passed"]):
            raise ValueError("selected scene failed the release screen")
        evidence.append({
            "scene_id": scene["id"], "family_id": scene["family_id"],
            "fingerprint": scene["fingerprint"],
            "episodes": episodes,
        })
    selection = {
        "version": "warehouse-r42-delivery-scene-selection.v1",
        "actor_sha256": screen["actor_sha256"],
        "source_full_manifest_sha256": file_hash(full_path),
        "source_screen_sha256": file_hash(screen_path),
        "checkpoint_selected_without_final_test": True,
        "human_condition_results_used": False,
        "selected": evidence,
    }
    selection_path = output / "selection_report.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False,
                                         sort_keys=True, indent=2) + "\n")
    runtime = deepcopy(parent)
    runtime["splits"]["play"] = [deepcopy(parent["splits"]["play"][0])] + selected
    runtime["source_dynamic_selection_report_sha256"] = file_hash(screen_path)
    runtime["source_selected_scenes_file_sha256"] = file_hash(selection_path)
    runtime["source_selected_scenes_semantic_sha256"] = digest(selection)
    runtime.pop("content_sha256", None)
    runtime["content_sha256"] = digest(runtime)
    manifest_path = output / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(runtime, ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"runtime_manifest": str(manifest_path),
                      "runtime_manifest_sha256": file_hash(manifest_path),
                      "selection_report": str(selection_path),
                      "selected_ids": list(SELECTED_IDS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
