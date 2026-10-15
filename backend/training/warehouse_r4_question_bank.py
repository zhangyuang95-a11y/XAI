"""Build and independently replay the frozen r4 questionnaire.

The builder uses only the frozen NumPy Actor and the public warehouse state.
It selects four next-action questions with different answers and four
three-step waiting questions with different displacements.  The resulting
JSON is the exact compact payload consumed by the online release loader; it
does not grant formal-study eligibility.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from backend.training import warehouse_family_question_pool
from backend.training.warehouse_r4_production_admission import local_source_hashes
from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action


VERSION = "warehouse-r4-frozen-question-bank.v1"
PROJECTION_VERSION = "warehouse-alignment-portable-question-projection.v1"
VIEW_VERSION = "warehouse-alignment-frozen-metadata-bank-view.v1"
PORTABLE_BANK_VERSION = "warehouse-alignment-portable-question-bank.v1"
REGISTERED_POOL_MANIFEST_SHA256 = "d9182abc5cc3c2c06b7990537bcbfd71c2320d2a772cbe23657e2d3c5a58020a"
REGISTERED_POOL_SHA256 = "82fb56ba6f05489ca5440a031d05963591b38492f94a583c17d5f86f12d04d5f"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON_BYTES = 64 * 1024 * 1024
_PAYLOAD_FIELDS = frozenset((
    "version", "test_fixture", "source_bank_signature", "anchors",
    "bank_sha256", "replay_receipt_sha256", "pool_manifest_sha256",
    "actor_sha256", "runtime_signature", "protocol_sha256", "runtime_family",
    "sources", "checks", "items", "private_items_sha256",
    "public_items_sha256", "summary_sha256",
))
_ITEM_FIELDS = frozenset((
    "id", "kind", "scenario_id", "frame", "snapshot", "snapshot_sha256",
    "preview", "answer", "diversity_key", "evidence", "options", "prompt",
))
NEXT_PROMPT = {
    "zh": "根据当前画面，AI 2 下一步会选择哪个动作？",
    "en": "From the current frame, which action will AI 2 choose next?",
}
WAIT_PROMPT = {
    "zh": "假设玩家连续等待三步，AI 2 在第三步结束时位于哪一格？字母标记仅表示选项位置。",
    "en": "If the player waits for three consecutive steps, where will AI 2 be after the third step? Letter markers identify the answer options only.",
}
ACTION_LABELS = {
    "UP": {"zh": "上", "en": "Up"},
    "DOWN": {"zh": "下", "en": "Down"},
    "LEFT": {"zh": "左", "en": "Left"},
    "RIGHT": {"zh": "右", "en": "Right"},
    "WAIT": {"zh": "等待", "en": "Wait"},
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_name() -> str:
    return str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2]))


def _source_binding() -> dict[str, str]:
    return local_source_hashes((Path(__file__),))


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError("Questionnaire input is missing, linked, noncanonical, or oversized")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON field in questionnaire input")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in questionnaire input")))
    if not isinstance(value, dict):
        raise ValueError("Questionnaire input must be a JSON object")
    return value


def _write_private(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() and not path.is_symlink():
        raise FileExistsError(path)
    if (path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Questionnaire output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _point(value) -> list[int]:
    return [int(value[0]), int(value[1])]


def _public_map(env) -> dict[str, Any]:
    layout = env.layout
    return {
        "layout_id": str(layout.layout_id),
        "rows": int(layout.rows),
        "cols": int(layout.cols),
        "shelves": [_point(value) for value in layout.blocked_positions],
        "charger_position": _point(layout.charger_position),
        "robot_exit_positions": [_point(value) for value in layout.robot_exit_positions],
        "waiting_zone": [_point(value) for value in layout.passable_positions
                         if int(value[0]) == int(layout.rows) - 1
                         and tuple(value) != tuple(layout.charger_position)],
        "robot_start_positions": [_point(value) for value in layout.robot_start_positions],
        "shared_delivery_tasks": True,
    }


def _public_state(env) -> dict[str, Any]:
    state = env.state
    tasks = sorted(state.tasks, key=lambda item: str(item.task_id))
    slots = {str(task.task_id): index for index, task in enumerate(tasks, 1)}
    agents = []
    for agent in state.agents:
        carrying = agent.carrying_task_id
        agents.append({
            "id": str(agent.agent_id),
            "position": _point(agent.position),
            "battery": float(agent.battery),
            "carrying_task_id": carrying,
            "carrying_label": (f"A{slots[str(carrying)]}"
                               if carrying is not None and str(carrying) in slots else None),
            "deliveries_completed": int(agent.deliveries_completed),
            "active": bool(agent.active),
            "heading": str(agent.heading),
            "last_action": str(agent.last_action),
            "last_executed_action": str(agent.last_executed_action),
            "selected": agent.agent_id == "robot_2",
        })
    return {
        "episode_id": int(state.episode_id),
        "frame": int(state.frame),
        "total_deliveries": int(state.total_deliveries),
        "active_count": sum(agent["active"] for agent in agents),
        "collision_count": int(state.collision_count),
        "shutdown_count": int(state.shutdown_count),
        "terminated": bool(state.terminated),
        "truncated": bool(state.truncated),
        "terminal_reason": state.terminal_reason,
        "selected_agent": "robot_2",
        "agents": agents,
        "tasks": [{
            "task_id": str(task.task_id),
            "pickup_position": _point(task.pickup_position),
            "delivery_position": _point(task.delivery_position),
            "status": str(task.status),
            "carrier_agent_id": task.carrier_agent_id,
            "created_frame": int(task.created_frame),
            "claimed_frame": (int(task.claimed_frame)
                              if task.claimed_frame is not None else None),
        } for task in tasks],
        "user_score": float(state.user_score),
        "score_breakdown": {str(key): float(value)
                            for key, value in state.score_breakdown.items()},
        "robot_collision_events": int(state.robot_collision_events),
        "invalid_move_count": int(state.invalid_move_count),
        "events": None,
    }


def _preview(env) -> dict[str, Any]:
    return {
        "map": _public_map(env),
        "state": _public_state(env),
        "public_feedback": deepcopy(env.public_history()),
    }


def _next_options() -> list[dict[str, Any]]:
    return [{"value": action, "label": deepcopy(ACTION_LABELS[action])}
            for action in ACTIONS]


def _passable_options(env, answer: Sequence[int], start: Sequence[int], seed: int):
    answer = tuple(map(int, answer))
    candidates = [answer]
    origin = tuple(map(int, start))
    if origin not in candidates and env.layout.is_passable(origin):
        candidates.append(origin)
    remaining = sorted((tuple(map(int, point)) for point in env.layout.passable_positions),
                       key=lambda point: (abs(point[0] - answer[0]) + abs(point[1] - answer[1]), point))
    for point in remaining:
        if point not in candidates:
            candidates.append(point)
        if len(candidates) == 4:
            break
    if len(candidates) != 4:
        raise ValueError("Four distinct wait-three option cells are required")
    random.Random(seed).shuffle(candidates)
    result = []
    for marker, point in zip("ABCD", candidates):
        result.append({
            "value": f"{point[0]},{point[1]}",
            "marker": marker,
            "position": list(point),
            "label": {
                "zh": f"{marker} · 第 {point[0] + 1} 行，第 {point[1] + 1} 列",
                "en": f"{marker} · Row {point[0] + 1}, column {point[1] + 1}",
            },
        })
    return result


def public_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": item["id"], "type": "choice", "prediction_kind": item["kind"],
        "required": True, "prompt": deepcopy(item["prompt"]),
        "options": [{"value": option["value"], "label": deepcopy(option["label"])}
                    for option in item["options"]],
        "preview": deepcopy(item["preview"]), "source_frame": item["frame"],
        "source_scenario": item["scenario_id"],
    } for item in items]


def _summary(bank_sha256: str) -> dict[str, Any]:
    return {
        "status": "candidate_ready", "formal_ready": False,
        "release_ready": False, "model_capability_evaluated": False,
        "available": True, "item_count": 8, "test_fixture": False,
        "preview_history_available_to_both_conditions": True,
        "version": VIEW_VERSION, "eligible": False,
        "participant_enabled": False,
        "independent_replay_previously_verified": True,
        "physics_replay_on_load": False,
        "runtime_family": "alignment_feedback197",
        "bank_sha256": bank_sha256,
        "replay_receipt_sha256": None,
    }


def _candidate(runtime, env, scene_id: str) -> dict[str, Any] | None:
    # Frame zero has no confirmed public transition and is deliberately kept
    # out of both questionnaire kinds.
    if env.done or int(env.state.frame) < 1:
        return None
    snapshot = deepcopy(env.snapshot())
    actions, decision = runtime.decision(env)
    start = tuple(env.state.by_id("robot_2").position)
    counterfactual = runtime.counterfactual(snapshot, ["WAIT"], steps=3)
    if counterfactual["steps_executed"] != 3:
        return None
    last = counterfactual["transitions"][-1]["after"]["state"]
    finish_agent = next(agent for agent in last["agents"] if agent["agent_id"] == "robot_2")
    finish = tuple(finish_agent["position"])
    probability = decision["probabilities"]["robot_2"]
    ordered = sorted(probability, reverse=True)
    return {
        "scenario_id": scene_id, "frame": int(env.state.frame),
        "snapshot": snapshot, "snapshot_sha256": digest(snapshot),
        "preview": _preview(env), "decision": decision,
        "action": actions["robot_2"], "confidence_margin": ordered[0] - ordered[1],
        "counterfactual": counterfactual, "start": start, "finish": finish,
        "displacement": (finish[0] - start[0], finish[1] - start[1]),
    }


def _collect(runtime, scenes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for scene_index, scene in enumerate(scenes):
        env = runtime.environment(scene)
        rng = random.Random(260_910_800 + scene_index)
        for _ in range(min(80, int(env.config.horizon))):
            candidate = _candidate(runtime, env, str(scene["id"]))
            if candidate is not None:
                candidates.append(candidate)
            if env.done:
                break
            before = digest(env.snapshot())
            action = partner_action(env, "robot_1", "assertive", rng)
            if digest(env.snapshot()) != before:
                raise RuntimeError("Question partner changed the source frame")
            transition = runtime.step(env, action)
            if transition["submitted_actions"]["robot_2"] != transition["policy_actions"]["robot_2"]:
                raise RuntimeError("Question trajectory contains an Actor override")
    return candidates


def _select(candidates: Sequence[Mapping[str, Any]]):
    used_scenes: set[str] = set()
    selected_next = []
    by_action: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_action.setdefault(str(row["action"]), []).append(row)
    ranked_actions = sorted(by_action, key=lambda action: (-len(by_action[action]), ACTIONS.index(action)))
    for action in ranked_actions:
        choices = sorted(by_action[action], key=lambda row: (-row["confidence_margin"], row["frame"], row["scenario_id"]))
        chosen = next((row for row in choices if row["scenario_id"] not in used_scenes), None)
        if chosen is not None:
            selected_next.append(chosen); used_scenes.add(chosen["scenario_id"])
        if len(selected_next) == 4:
            break
    if len(selected_next) != 4:
        raise RuntimeError("Frozen Actor did not provide four distinct next-action outcomes")
    selected_wait = []
    by_displacement: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_displacement.setdefault(tuple(row["displacement"]), []).append(row)
    ranked_displacements = sorted(by_displacement,
        key=lambda value: (-len(by_displacement[value]), value))
    for displacement in ranked_displacements:
        choices = sorted(by_displacement[displacement],
                         key=lambda row: (row["frame"], row["scenario_id"]))
        chosen = next((row for row in choices if row["scenario_id"] not in used_scenes), None)
        if chosen is not None:
            selected_wait.append(chosen); used_scenes.add(chosen["scenario_id"])
        if len(selected_wait) == 4:
            break
    if len(selected_wait) != 4:
        raise RuntimeError("Frozen Actor did not provide four distinct wait-three outcomes")
    return selected_next, selected_wait


def _items(runtime, selected_next, selected_wait):
    items = []
    for index, row in enumerate(selected_next, 1):
        items.append({
            "id": f"prediction_next_action_{index}", "kind": "next_action",
            "scenario_id": row["scenario_id"], "frame": row["frame"],
            "snapshot": deepcopy(row["snapshot"]), "snapshot_sha256": row["snapshot_sha256"],
            "preview": deepcopy(row["preview"]), "answer": row["action"],
            "diversity_key": row["action"], "evidence": deepcopy(row["decision"]),
            "options": _next_options(), "prompt": deepcopy(NEXT_PROMPT),
        })
    for index, row in enumerate(selected_wait, 1):
        finish = list(row["finish"])
        preview = deepcopy(row["preview"])
        options = _passable_options(runtime.from_snapshot(row["snapshot"]), finish,
                                    row["start"], 260_910_900 + index)
        preview["question_markers"] = [
            {"marker": option["marker"], "position": option["position"]}
            for option in options
        ]
        items.append({
            "id": f"prediction_wait_three_{index}", "kind": "wait_three",
            "scenario_id": row["scenario_id"], "frame": row["frame"],
            "snapshot": deepcopy(row["snapshot"]), "snapshot_sha256": row["snapshot_sha256"],
            "preview": preview, "answer": f"{finish[0]},{finish[1]}",
            "diversity_key": f"{row['displacement'][0]},{row['displacement'][1]}",
            "evidence": {
                "assumed_player_actions": ["WAIT", "WAIT", "WAIT"],
                "transitions": deepcopy(row["counterfactual"]["transitions"]),
                "transitions_sha256": digest(row["counterfactual"]["transitions"]),
            },
            "options": options, "prompt": deepcopy(WAIT_PROMPT),
        })
    return items


def _independent_replay(runtime, items):
    checks = []
    for item in items:
        env = runtime.from_snapshot(item["snapshot"])
        if item["kind"] == "next_action":
            actions, decision = runtime.decision(env)
            answer = actions["robot_2"]
            evidence_match = digest(decision) == digest(item["evidence"])
        else:
            result = runtime.counterfactual(item["snapshot"], ["WAIT"], steps=3)
            last = result["transitions"][-1]["after"]["state"]
            agent = next(value for value in last["agents"] if value["agent_id"] == "robot_2")
            answer = f"{agent['position'][0]},{agent['position'][1]}"
            evidence_match = digest(result["transitions"]) == item["evidence"]["transitions_sha256"]
        checks.append({"id": item["id"], "answer_match": answer == item["answer"],
                       "evidence_match": evidence_match,
                       "source_unchanged": digest(env.snapshot()) == item["snapshot_sha256"]})
    if not all(all(value for key, value in check.items() if key != "id") for check in checks):
        raise RuntimeError("Independent questionnaire replay failed")
    return checks


def validate_payload(runtime: OnlineAlignmentRuntime,
                     payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a frozen questionnaire from its private snapshots.

    The portable loader checks archive framing and private/public projections,
    but the questionnaire's own hashes are not an authenticity boundary.  This
    validator therefore replays all eight questions with the loaded frozen
    Actor and rebuilds every derived hash before admission or packaging.
    """
    if (type(payload) is not dict or set(payload) != _PAYLOAD_FIELDS
            or payload.get("version") != PROJECTION_VERSION
            or payload.get("test_fixture") is not False
            or payload.get("runtime_family") != "alignment_feedback197"):
        raise ValueError("Exact non-fixture r4 question payload required")
    for name in ("source_bank_signature", "bank_sha256", "replay_receipt_sha256",
                 "pool_manifest_sha256", "actor_sha256", "runtime_signature",
                 "protocol_sha256", "private_items_sha256", "public_items_sha256",
                 "summary_sha256"):
        if type(payload.get(name)) is not str or _HEX.fullmatch(payload[name]) is None:
            raise ValueError("Invalid r4 questionnaire SHA-256: " + name)
    if (payload["actor_sha256"] != runtime.actor_sha256
            or payload["protocol_sha256"] != runtime.protocol_sha256
            or payload["runtime_signature"] != runtime.signature
            or payload["pool_manifest_sha256"] != REGISTERED_POOL_MANIFEST_SHA256):
        raise ValueError("R4 questionnaire belongs to another frozen runtime")
    sources = _source_binding()
    if payload.get("sources") != sources:
        raise ValueError("R4 questionnaire producer source changed")
    items = payload.get("items")
    expected_ids = {
        **{f"prediction_next_action_{index}": "next_action" for index in range(1, 5)},
        **{f"prediction_wait_three_{index}": "wait_three" for index in range(1, 5)},
    }
    if (type(items) is not list or len(items) != 8
            or any(type(item) is not dict or set(item) != _ITEM_FIELDS for item in items)
            or {item.get("id") for item in items} != set(expected_ids)
            or any(item.get("kind") != expected_ids.get(item.get("id")) for item in items)
            or len({item.get("scenario_id") for item in items}) != 8):
        raise ValueError("R4 questionnaire requires eight distinct complete items")
    for item in items:
        frame = item.get("frame")
        if (type(item.get("scenario_id")) is not str or not item["scenario_id"]
                or type(frame) is not int or frame < 1
                or type(item.get("snapshot")) is not dict
                or digest(item["snapshot"]) != item.get("snapshot_sha256")
                or item["snapshot"].get("state", {}).get("frame") != frame):
            raise ValueError("R4 questionnaire snapshot identity differs")
        env = runtime.from_snapshot(item["snapshot"])
        base_preview = _preview(env)
        if item["kind"] == "next_action":
            if (item.get("prompt") != NEXT_PROMPT or item.get("options") != _next_options()
                    or item.get("diversity_key") != item.get("answer")
                    or item.get("preview") != base_preview):
                raise ValueError("R4 next-action question projection differs")
        else:
            result = runtime.counterfactual(item["snapshot"], ["WAIT"], steps=3)
            if result.get("steps_executed") != 3:
                raise ValueError("R4 wait-three replay ended early")
            last = result["transitions"][-1]["after"]["state"]
            finish_agent = next(agent for agent in last["agents"]
                                if agent["agent_id"] == "robot_2")
            finish = tuple(map(int, finish_agent["position"]))
            start_agent = next(agent for agent in item["snapshot"]["state"]["agents"]
                               if agent["agent_id"] == "robot_2")
            start = tuple(map(int, start_agent["position"]))
            index = int(item["id"].rsplit("_", 1)[-1])
            options = _passable_options(env, finish, start, 260_910_900 + index)
            preview = deepcopy(base_preview)
            preview["question_markers"] = [
                {"marker": option["marker"], "position": option["position"]}
                for option in options
            ]
            evidence = item.get("evidence")
            if (item.get("prompt") != WAIT_PROMPT or item.get("options") != options
                    or item.get("answer") != f"{finish[0]},{finish[1]}"
                    or item.get("diversity_key") != f"{finish[0] - start[0]},{finish[1] - start[1]}"
                    or item.get("preview") != preview
                    or type(evidence) is not dict
                    or set(evidence) != {"assumed_player_actions", "transitions",
                                         "transitions_sha256"}
                    or evidence.get("assumed_player_actions") != ["WAIT", "WAIT", "WAIT"]
                    or digest(evidence.get("transitions")) != digest(result["transitions"])
                    or evidence.get("transitions_sha256") != digest(result["transitions"])):
                raise ValueError("R4 wait-three question projection differs")
    replay_checks = _independent_replay(runtime, items)
    category_checks = {
        "next_action": {"count": 4, "independent_scenarios": 4,
                        "distinct_outcomes": len({item["answer"] for item in items
                                                  if item["kind"] == "next_action"})},
        "wait_three": {"count": 4, "independent_scenarios": 4,
                       "distinct_outcomes": len({item["diversity_key"] for item in items
                                                 if item["kind"] == "wait_three"})},
    }
    if any(value["distinct_outcomes"] != 4 for value in category_checks.values()):
        raise ValueError("R4 questionnaire outcome diversity differs")
    expected_checks = {"passed": True, "categories": category_checks,
                       "independent_replay": replay_checks, "formal_ready": False}
    anchors = {"actor_sha256": runtime.actor_sha256,
               "protocol_sha256": runtime.protocol_sha256,
               "pool_manifest_sha256": payload["pool_manifest_sha256"],
               "items_sha256": digest(items)}
    bank_sha = digest({"version": VERSION, "items": items, "checks": category_checks})
    replay_sha = digest(replay_checks)
    source_signature = digest({"version": VIEW_VERSION, "anchors": anchors,
                               "bank_sha256": bank_sha, "sources": sources})
    summary = _summary(bank_sha)
    summary["replay_receipt_sha256"] = replay_sha
    expected = {
        "anchors": anchors,
        "bank_sha256": bank_sha,
        "replay_receipt_sha256": replay_sha,
        "source_bank_signature": source_signature,
        "checks": expected_checks,
        "private_items_sha256": digest(items),
        "public_items_sha256": digest(public_items(items)),
        "summary_sha256": digest(summary),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("R4 questionnaire derived evidence or signature differs")
    return {"checks": deepcopy(expected_checks), "items": deepcopy(items),
            "source_bank_signature": source_signature,
            "pool_manifest_sha256": payload["pool_manifest_sha256"]}


def read_registered_pool(pool_path: str | Path,
                         manifest_path: str | Path) -> tuple[dict[str, Any], str]:
    """Authenticate the complete registered question-pool publication."""
    pool_path = Path(pool_path).expanduser().absolute()
    manifest_path = Path(manifest_path).expanduser().absolute()
    if (pool_path.name != "pool.json" or manifest_path.name != "manifest.json"
            or pool_path.parent != manifest_path.parent
            or pool_path.resolve() != pool_path or manifest_path.resolve() != manifest_path
            or pool_path.is_symlink() or manifest_path.is_symlink()):
        raise ValueError("Use pool.json and manifest.json from one registered question-pool directory")
    manifest_sha256 = file_hash(manifest_path)
    if manifest_sha256 != REGISTERED_POOL_MANIFEST_SHA256:
        raise ValueError("Question pool is not the frozen registered publication")
    registered = warehouse_family_question_pool.read_registered(
        pool_path.parent, expected_manifest_sha256=manifest_sha256,
        allow_test_fixture=False,
    )
    if (registered["pool_sha256"] != REGISTERED_POOL_SHA256
            or registered["pool_sha256"] != file_hash(pool_path)):
        raise ValueError("Registered question pool bytes differ")
    return registered, manifest_sha256


def build(runtime: OnlineAlignmentRuntime, pool: Mapping[str, Any], *, output: str | Path,
          pool_manifest_sha256: str) -> dict[str, Any]:
    output = Path(output).expanduser().absolute()
    if (output.exists() or output.is_symlink() or output.resolve() != output
            or output.parent.is_symlink()
            or (output.parent.exists() and output.parent.resolve() != output.parent.absolute())):
        raise FileExistsError(output)
    scenes = pool.get("scenes")
    if (pool.get("test_fixture") is not False or not isinstance(scenes, list)
            or len(scenes) < 8 or len({scene.get("id") for scene in scenes}) != len(scenes)):
        raise ValueError("A non-fixture independent question pool is required")
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    candidates = _collect(runtime, scenes)
    selected_next, selected_wait = _select(candidates)
    items = _items(runtime, selected_next, selected_wait)
    replay_checks = _independent_replay(runtime, items)
    category_checks = {
        "next_action": {"count": 4, "independent_scenarios": 4,
                        "distinct_outcomes": len({item["answer"] for item in items[:4]})},
        "wait_three": {"count": 4, "independent_scenarios": 4,
                       "distinct_outcomes": len({item["diversity_key"] for item in items[4:]})},
    }
    if any(value["distinct_outcomes"] != 4 for value in category_checks.values()):
        raise RuntimeError("Questionnaire diversity check failed")
    sources = _source_binding()
    anchors = {
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "pool_manifest_sha256": pool_manifest_sha256,
        "items_sha256": digest(items),
    }
    bank_sha = digest({"version": VERSION, "items": items, "checks": category_checks})
    replay_receipt_sha = digest(replay_checks)
    source_signature = digest({"version": VIEW_VERSION, "anchors": anchors,
                               "bank_sha256": bank_sha, "sources": sources})
    summary = _summary(bank_sha)
    summary["replay_receipt_sha256"] = replay_receipt_sha
    payload = {
        "version": PROJECTION_VERSION, "test_fixture": False,
        "source_bank_signature": source_signature, "anchors": anchors,
        "bank_sha256": bank_sha, "replay_receipt_sha256": replay_receipt_sha,
        "pool_manifest_sha256": pool_manifest_sha256,
        "actor_sha256": runtime.actor_sha256,
        "runtime_signature": runtime.signature,
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_family": "alignment_feedback197", "sources": sources,
        "checks": {"passed": True, "categories": category_checks,
                   "independent_replay": replay_checks, "formal_ready": False},
        "items": items, "private_items_sha256": digest(items),
        "public_items_sha256": digest(public_items(items)),
        "summary_sha256": digest(summary),
    }
    report = {
        "version": VERSION, "status": "candidate_ready",
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature,
        "pool_manifest_sha256": pool_manifest_sha256,
        "candidate_frames": len(candidates), "selected_scenarios": 8,
        "checks": payload["checks"], "source_bank_signature": source_signature,
        "payload_sha256": digest(payload), "formal_ready": False,
    }
    validate_payload(runtime, payload)
    _write_private(output / "question_bank.json", payload)
    _write_private(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    registered, manifest_sha256 = read_registered_pool(args.pool, args.pool_manifest)
    protocol = _read_json(args.protocol)
    runtime = OnlineAlignmentRuntime(
        args.actor, protocol=protocol,
        expected_actor_sha256=file_hash(args.actor),
        expected_protocol_sha256=digest(protocol), allow_test_fixture=False,
    )
    report = build(runtime, registered, output=args.output,
                   pool_manifest_sha256=manifest_sha256)
    print(canonical(report))


if __name__ == "__main__":
    main()


__all__ = ["VERSION", "PROJECTION_VERSION", "build", "validate_payload",
           "read_registered_pool", "public_items", "main"]
