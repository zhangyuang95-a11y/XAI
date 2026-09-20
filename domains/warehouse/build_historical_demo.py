"""Rebuild the unmodified historical Render tutorial in an isolated checkout.

Run `python -m domains.warehouse.build_historical_demo`. Git is a build-time
requirement only; the deployed application reads the hash-pinned public asset.
This replays historical source, not a recording of a verified deployment.
"""
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "de16551d3d6b99c7ab426dfed1db2871159e6b5c"
ASSET = "configs/study_v3_warehouse_legacy_demo.json"


def replay():
    archive = subprocess.check_output(["git", "archive", COMMIT, "env", "backend", "core", "ui",
        "output/deployment/warehouse_mappo_v68_6x7_actor.npz"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="warehouse-demo-history-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(directory, filter="data")
        code = '''from dataclasses import asdict
import json
from ui.development_preview_server import build_development_tutorial
from env.warehouse.layouts import STUDY_MAP_LAYOUT
frames=build_development_tutorial()
print(json.dumps({"frames":[{"state":asdict(f.state),"actions":dict(f.actions),"events":list(f.events)} for f in frames],
"layout":{"width":STUDY_MAP_LAYOUT.cols,"height":STUDY_MAP_LAYOUT.rows,"walls":[[c,r] for r,c in sorted(STUDY_MAP_LAYOUT.blocked_positions)],"charger":list(reversed(STUDY_MAP_LAYOUT.charger_position)),"map_id":STUDY_MAP_LAYOUT.layout_id}},ensure_ascii=False))'''
        return json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=directory))


def project(raw):
    layout = raw["layout"]
    frames, captions, seen = [], [], set()
    charge_count = 0
    for source in raw["frames"]:
        s = source["state"]
        events = []
        for event in source["events"]:
            kind = event["event"]
            who = "The blue robot" if event.get("agent_id", event.get("carrier_agent_id")) == "robot_1" else "The orange robot"
            zhwho = "蓝色机器人" if who == "The blue robot" else "橙色机器人"
            if kind == "charging":
                charge_count += 1
                typ, en, zh = "charge", f"{who} waits at the charger and gains {event['energy_gained']:g} battery.", f"{zhwho}在充电位置等待，恢复{event['energy_gained']:g}点电量。"
            elif kind == "claimed":
                typ, en, zh = "pickup", f"{who} picks up a parcel at its A point.", f"{zhwho}到达A点并取走包裹。"
            elif kind == "delivered":
                typ, en, zh = "delivery", f"{who} delivers its parcel to the matching B point. A new shared job appears.", f"{zhwho}把包裹送达对应B点，系统补充一个共享订单。"
            elif kind == "coordination_yield":
                typ, en, zh = "coordination", "One robot makes room while its partner uses the shared route.", "一个机器人让出空间，与同伴配合使用共享通道。"
            else:
                continue
            events.append({"type": typ, "en": en, "zh": zh})
            if typ not in seen:
                captions.append({"index": s["frame"], "en": en, "zh": zh})
                seen.add(typ)
        terminal = bool(s["terminated"] or s["truncated"])
        if terminal:
            events.append({"type": "complete", "en": "The demonstration has finished.", "zh": "演示已结束。"})
        actors = {}
        for name, agent in zip(("human", "ai"), s["agents"]):
            actors[name] = {"x": agent["position"][1], "y": agent["position"][0], "battery": agent["battery"],
                "carrying": agent["carrying_task_id"], "heading": agent["heading"].lower(), "active": agent["active"]}
        orders = [{"id": t["task_id"], "owner": "shared", "pickup": list(reversed(t["pickup_position"])),
            "dropoff": list(reversed(t["delivery_position"])), "status": t["status"],
            "carrier": {"robot_1": "human", "robot_2": "ai"}.get(t["carrier_agent_id"])} for t in s["tasks"]]
        frames.append({"domain": "warehouse", "version": "warehouse-historical-demo.de16551", "task": 0,
            "turn": s["frame"], "max_turns": 120, "terminal": terminal, "events": events,
            **actors, "width": layout["width"], "height": layout["height"], "walls": layout["walls"],
            "orders": orders, "chargers": [layout["charger"]], "map_id": layout["map_id"],
            "stations": [{"id": "charger", "x": layout["charger"][0], "y": layout["charger"][1], "kind": "charger", "label_en": "Shared charger", "label_zh": "共享充电位置"}],
            "demo_actions": {"human" if k == "robot_1" else "ai": v.lower() for k, v in source["actions"].items()},
            "score": {"task_score": s["user_score"], "raw_score": s["user_score"], "score_max": None,
                "score_scale": "raw", "breakdown": s["score_breakdown"],
                "metrics": {"deliveries": s["total_deliveries"], "collision_events": s["robot_collision_events"],
                    "shutdown_events": s["shutdown_count"], "charge_events": charge_count,
                    "elapsed_turns": s["frame"], "human_detour_units": s["human_route_regret_units"],
                    "terminal_reason": s["terminal_reason"]}}})
    captions.insert(0, {"index": 0, "en": "Both robots move automatically in this demonstration. In the tasks, you control the blue robot and the orange robot is your AI teammate. Move parcels from A to matching B points.", "zh": "演示中两个机器人会自动行动。正式任务中，你控制蓝色机器人，橙色机器人是AI队友。把包裹从A点送到对应B点。"})
    captions.append({"index": 120, "en": "The 120-step demonstration is complete. It does not count toward your task scores.", "zh": "120步演示已完成，不计入你的任务成绩。"})
    trajectory = [{"turn": f["turn"], "human": f["human"], "ai": f["ai"], "orders": f["orders"],
                   "actions": f["demo_actions"], "score": f["score"]["raw_score"]} for f in frames]
    return {"frames": frames, "captions": sorted(captions, key=lambda c: c["index"]), "provenance": {
        "source_commit": COMMIT, "source_commit_time": "2026-09-02T12:54:11+08:00",
        "source_function": "ui/development_preview_server.py:build_development_tutorial",
        "source_render_command": "python -m ui.development_preview_server --host 0.0.0.0 --port $PORT",
        "seed": 40786, "initial_ai_battery": 35, "frame_count": len(frames),
        "original_score_preserved": True, "not_task_score": True,
        "historical_deployment_timestamp_verified": False, "kind": "historical_source_replay",
        "trajectory_sha256": sha256(json.dumps(trajectory, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}}


def build():
    return project(replay())


if __name__ == "__main__":
    output = build()
    content = (json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    (ROOT / ASSET).write_bytes(content)
    config_path = ROOT / "configs/study_v3_warehouse.json"
    config = json.loads(config_path.read_text())
    config["historical_demo"] = {"path": ASSET, "sha256": sha256(content).hexdigest(), "source_commit": COMMIT}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"asset": ASSET, "bytes": len(content), "sha256": sha256(content).hexdigest(), "provenance": output["provenance"]}))
