"""Hash-bound r4.2 internal-pilot envelope over the admitted r4.1 artifacts."""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Mapping
import zipfile

from backend.warehouse_r42_program import R42DecisionProgram
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from ui import warehouse_alignment_r41_diagnostic_release_v9 as parent_release
from ui import warehouse_alignment_r42_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r42-internal-pilot-envelope.v1"
STANDALONE_VERSION = "warehouse-r42-delivery-first-package.v1"
CONTEXT_VERSION = "warehouse-r42-internal-pilot-release.v1"
PUBLIC_RELEASE_VERSION = "r4.2-internal-pilot"
MANIFEST_NAME = "manifest.json"
PARENT_NAME = "artifacts/r41_parent_release.zip"
STANDALONE_ARTIFACTS = {
    "actor": "artifacts/actor.npz",
    "protocol": "artifacts/training_protocol.json",
    "runtime_manifest": "artifacts/runtime_manifest.json",
    "program": "artifacts/program.json",
    "question_bank": "artifacts/question_bank.json",
    "tutorial": "artifacts/tutorial.json",
    "training_report": "artifacts/training_report.json",
    "behavior_report": "artifacts/behavior_report.json",
}
MAX_PACKAGE_BYTES = 750_000
MAX_BASE64_BYTES = 1_000_000
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _shortest_path_distance(env: Any, start: tuple[int, int],
                            goal: tuple[int, int]) -> int:
    """Runtime-only BFS that avoids importing the offline RCPD package.

    Importing ``env.warehouse.navigation`` executes ``env.warehouse``'s package
    initializer, which eventually imports the scikit-learn based offline RCPD
    fitter.  The participant service intentionally ships only NumPy.  The
    restored environment already exposes the authoritative layout, so compute
    the same four-neighbour distance directly from it.
    """

    if start == goal:
        return 0
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    queue = deque(((start, 0),))
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        for row_delta, column_delta in moves:
            candidate = (position[0] + row_delta,
                         position[1] + column_delta)
            if candidate in visited or not env.layout.is_passable(candidate):
                continue
            if candidate == goal:
                return distance + 1
            visited.add(candidate)
            queue.append((candidate, distance + 1))
    return int(env.layout.rows * env.layout.cols)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (Path(__file__), ROOT / "ui/warehouse_alignment_r42_server.py",
             ROOT / "ui/warehouse_alignment_r42_tutorial.py",
             ROOT / "backend/warehouse_r42_program.py",
             ROOT / "ui/warehouse_family_feedback_research/index.html",
             ROOT / "ui/warehouse_family_feedback_research/app.js",
             ROOT / "ui/warehouse_family_feedback_research/styles.css")
    return {str(path.relative_to(ROOT)): _file_hash(path) for path in paths}


class R42ReadableExplainer:
    """Add fixed intent routing while retaining the admitted evidence renderer."""

    _CANONICAL = {
        "action_reason": {
            "zh": ("机器人2刚才为什么这样行动？", "executed"),
            "en": ("Why did Robot 2 choose that action?", "executed"),
        },
        "wait_reason": {
            "zh": ("机器人2刚才为什么等待？", "executed"),
            "en": ("Why did Robot 2 wait?", "executed"),
        },
        "collision_reason": {
            "zh": ("我们刚才为什么发生碰撞？", "executed"),
            "en": ("Why did we just collide?", "executed"),
        },
        "human_influence": {
            "zh": ("我的上一步动作影响了机器人2吗？", "executed"),
            "en": ("Did my last action affect Robot 2?", "executed"),
        },
        "task_direction": {
            "zh": ("机器人2当前在朝哪个任务前进？", "next"),
            "en": ("Which task is Robot 2 moving toward?", "next"),
        },
        "charging_need": {
            "zh": ("机器人2现在需要充电吗？", "next"),
            "en": ("Does Robot 2 need to charge now?", "next"),
        },
    }

    def __init__(self, parent):
        self.parent = parent
        self.signature = _digest({"version": "warehouse-r42-readable-explainer.v1",
                                  "parent": str(parent.signature)})
        self.program_sha256 = parent.program_sha256
        self.artifact_binding = deepcopy(parent.artifact_binding)

    def _assert_current(self, runtime):
        return self.parent._assert_current(runtime)

    @staticmethod
    def _charging_answer(question, record, runtime):
        language = question.get("language", "zh")
        snapshot = record.get("after") if isinstance(record, Mapping) else None
        if not isinstance(snapshot, Mapping):
            text = ("无法可靠判断：所选帧缺少可核验的电量状态。" if language == "zh"
                    else "Cannot determine reliably: the selected frame lacks a verified battery state.")
            return {"answer": text, "evidence_detail": ""}
        env = runtime.from_snapshot(deepcopy(snapshot))
        agent = env.state.by_id("robot_2")
        charger = env.layout.charger_position
        direct = _shortest_path_distance(env, agent.position, charger)
        route = None
        carrying = agent.carrying_task_id
        if carrying:
            task = next((item for item in env.state.tasks
                         if item.task_id == carrying), None)
            if task is not None:
                route = (_shortest_path_distance(
                    env, agent.position, task.delivery_position)
                         + _shortest_path_distance(
                             env, task.delivery_position, charger))
        else:
            candidates = []
            for task in env.state.tasks:
                if task.status != "available":
                    continue
                candidates.append(
                    _shortest_path_distance(
                        env, agent.position, task.pickup_position)
                    + _shortest_path_distance(
                        env, task.pickup_position, task.delivery_position)
                    + _shortest_path_distance(
                        env, task.delivery_position, charger))
            route = min(candidates) if candidates else None
        planned_moves = route if route is not None else direct
        move_cost = float(getattr(env.config, "move_battery_cost", 2.0))
        required = move_cost * (planned_moves + 2)
        needs = (not agent.active) or agent.battery <= max(20.0, required)
        if language == "zh":
            answer = ((f"现在需要充电。机器人2当前电量为 {agent.battery:.0f}%，按完成最近一项可执行任务后返回充电格估计需要约 {required:.0f}% 电量。")
                      if needs else
                      (f"暂时不需要充电。机器人2当前电量为 {agent.battery:.0f}%，按完成最近一项可执行任务后返回充电格估计需要约 {required:.0f}% 电量。"))
            detail = (f"核验状态：电量 {agent.battery:.0f}%；直接到充电格的通路距离 {direct} 步；"
                      f"任务及返程估计 {planned_moves} 步，并保留两步余量。")
        else:
            answer = ((f"Charging is needed now. Robot 2 has {agent.battery:.0f}% battery; completing the nearest feasible task and returning to the charger is estimated to require about {required:.0f}%.")
                      if needs else
                      (f"Charging is not needed yet. Robot 2 has {agent.battery:.0f}% battery; completing the nearest feasible task and returning to the charger is estimated to require about {required:.0f}%."))
            detail = (f"Verified state: {agent.battery:.0f}% battery; direct charger distance {direct}; "
                      f"task plus return estimate {planned_moves} moves, with a two-move reserve.")
        return {"answer": answer, "evidence_detail": detail}

    def answer_study(self, question, record, runtime, *, access_context):
        intent = question.get("intent_id")
        text = str(question.get("question", "")).casefold()
        if intent == "charging_need" or any(token in text for token in
                ("充电", "电量", "charge", "battery")):
            # The parent access validator remains authoritative.
            from backend.warehouse_r41_diagnostic_online_explanation_v9 import explanation_access
            if not explanation_access(access_context)["allowed"]:
                raise ValueError("explanation_access_denied")
            return self._charging_answer(question, record, runtime)
        delegated = dict(question)
        language = "en" if question.get("language") == "en" else "zh"
        if intent in self._CANONICAL:
            delegated["question"], delegated["focus"] = \
                self._CANONICAL[intent][language]
        return self.parent.answer_study(delegated, record, runtime,
                                        access_context=access_context)


class R42CompactQuestionBank:
    """Small frozen eight-item bank bound to the replacement Actor."""

    VERSION = "warehouse-r42-compact-question-bank.v1"

    def __init__(self, payload, *, runtime):
        value = deepcopy(dict(payload))
        claimed = value.pop("content_sha256", None)
        items = value.get("items")
        if (claimed != _digest(value) or value.get("version") != self.VERSION
                or value.get("actor_sha256") != runtime.actor_sha256
                or value.get("runtime_signature") != runtime.signature
                or not isinstance(items, list) or len(items) != 8
                or len({item.get("id") for item in items}) != 8
                or sum(item.get("kind") == "next_action" for item in items) != 4
                or sum(item.get("kind") == "wait_three" for item in items) != 4):
            raise ValueError("r4.2 compact question bank differs")
        self._items = deepcopy(items)
        self.actor_sha256 = runtime.actor_sha256
        self.runtime_signature = runtime.signature
        self.signature = str(claimed)
        self.test_fixture = False

    def public_items(self):
        return [{
            "id": item["id"], "type": "choice",
            "prediction_kind": item["kind"], "required": True,
            "prompt": deepcopy(item["prompt"]),
            "options": [{"value": option["value"],
                         "label": deepcopy(option["label"])}
                        for option in item["options"]],
            "preview": deepcopy(item["preview"]),
            "source_frame": int(item["frame"]),
            "source_scenario": item["scenario_id"],
        } for item in self._items]

    def grade(self, answers):
        if not isinstance(answers, dict):
            raise ValueError("Prediction answers must be an object")
        result = {"bank_signature": self.signature, "candidate_only": True,
                  "formal_ready": False}
        for kind in ("next_action", "wait_three"):
            items = [item for item in self._items if item["kind"] == kind]
            if any(answers.get(item["id"]) not in {
                    option["value"] for option in item["options"]}
                   for item in items):
                raise ValueError("Incomplete or invalid prediction answers")
            correct = sum(answers[item["id"]] == item["answer"] for item in items)
            result[kind] = {"correct": correct, "total": len(items),
                            "accuracy": correct / len(items)}
        return result


class R42DirectExplainer:
    """Readable evidence renderer bound to the replacement Actor and program."""

    _LABELS = {
        "zh": {"UP": "向上", "DOWN": "向下", "LEFT": "向左",
               "RIGHT": "向右", "WAIT": "等待"},
        "en": {"UP": "move up", "DOWN": "move down", "LEFT": "move left",
               "RIGHT": "move right", "WAIT": "wait"},
    }
    _DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
               "RIGHT": (0, 1), "WAIT": (0, 0)}

    def __init__(self, program):
        self.program = program
        self.program_sha256 = program.signature
        self.signature = _digest({"version": "warehouse-r42-direct-explainer.v1",
                                  "program": program.signature,
                                  "actor": program.actor_sha256})
        self.artifact_binding = {
            "actor_sha256": program.actor_sha256,
            "program_content_sha256": program.signature,
        }

    def _assert_current(self, runtime):
        runtime.verify_binding()
        if runtime.actor_sha256 != self.program.actor_sha256:
            raise ValueError("r4.2 explanation Actor differs")
        return self.signature

    @staticmethod
    def _allowed(access_context):
        from backend.warehouse_r41_diagnostic_online_explanation_v9 import explanation_access
        if not explanation_access(access_context)["allowed"]:
            raise ValueError("explanation_access_denied")

    @staticmethod
    def _goal(env, action):
        agent = env.state.by_id("robot_2")
        if agent.carrying_task_id:
            task = env.state.task_by_id(agent.carrying_task_id)
            return task, task.delivery_position, "delivery"
        available = [task for task in env.state.tasks if task.status == "available"]
        if not available:
            return None, env.layout.charger_position, "charger"
        delta = R42DirectExplainer._DELTAS[action]
        candidate = (agent.position[0] + delta[0], agent.position[1] + delta[1])
        if not env.layout.is_passable(candidate):
            candidate = agent.position
        task = min(available, key=lambda item: (
            _shortest_path_distance(env, candidate, item.pickup_position),
            str(item.task_id),
        ))
        return task, task.pickup_position, "pickup"

    def answer_study(self, question, record, runtime, *, access_context):
        self._allowed(access_context)
        self._assert_current(runtime)
        language = "en" if question.get("language") == "en" else "zh"
        intent = question.get("intent_id")
        text = str(question.get("question", "")).casefold()
        if intent is None:
            if any(token in text for token in ("充电", "电量", "charge", "battery")):
                intent = "charging_need"
            elif any(token in text for token in ("碰撞", "撞", "collid", "crash")):
                intent = "collision_reason"
            elif any(token in text for token in ("等待", "没动", "wait", "stationary")):
                intent = "wait_reason"
            elif any(token in text for token in ("影响", "affect", "influence")):
                intent = "human_influence"
            elif any(token in text for token in ("任务", "目标", "方向", "task", "target")):
                intent = "task_direction"
            else:
                intent = "action_reason"
        focus = question.get("focus", "executed")
        snapshot = (record.get("before") if focus == "executed"
                    else record.get("after"))
        if not isinstance(snapshot, Mapping):
            snapshot = record.get("after")
        if not isinstance(snapshot, Mapping):
            answer = ("所选帧没有足够的可核验状态。" if language == "zh"
                      else "The selected frame has insufficient verified state.")
            return {"answer": answer, "evidence_detail": ""}
        env = runtime.from_snapshot(deepcopy(snapshot))
        if intent == "charging_need":
            return R42ReadableExplainer._charging_answer(question, record, runtime)
        if focus == "executed":
            action = str(record.get("submitted_actions", {}).get("robot_2", "WAIT"))
        else:
            action = runtime.decision(env)[0]["robot_2"]
        program_action = self.program.action(env.observations()["robot_2"])
        agreement = program_action == action
        label = self._LABELS[language][action]
        agent = env.state.by_id("robot_2")
        events = record.get("events", []) if isinstance(record, Mapping) else []
        collision = any(event.get("event") == "collision" for event in events
                        if isinstance(event, Mapping))
        collision_kind = str(record.get("info", {}).get("collision_kind", "none"))
        if intent == "collision_reason":
            if collision:
                answer = ((f"机器人2选择了{label}，但两台机器人发生了{collision_kind}冲突，所以环境让它们停在原位。")
                          if language == "zh" else
                          (f"Robot 2 chose to {label}, but a {collision_kind} conflict made the environment keep both robots in place."))
            else:
                answer = ("所选这一步没有发生机器人碰撞。" if language == "zh"
                          else "No robot collision occurred on the selected step.")
        elif intent == "human_influence":
            if collision:
                answer = ("两台机器人从同一行动前状态同时决策；你的动作没有改变机器人2的选择，但共同造成了这次碰撞。"
                          if language == "zh" else
                          "Both robots decided from the same prior state; your action did not change Robot 2's choice, but the joint actions caused this collision.")
            else:
                answer = ("两台机器人从同一行动前状态同时决策，因此你的这一步不会被机器人2提前读取。"
                          if language == "zh" else
                          "Both robots decided from the same prior state, so Robot 2 could not read your current action in advance.")
        else:
            task, goal, mode = self._goal(env, action)
            before_distance = _shortest_path_distance(env, agent.position, goal)
            delta = self._DELTAS[action]
            candidate = (agent.position[0] + delta[0], agent.position[1] + delta[1])
            after_distance = (_shortest_path_distance(env, candidate, goal)
                              if env.layout.is_passable(candidate) else before_distance)
            task_label = str(task.task_id).replace("task_", "") if task else ""
            supported = agreement and action != "WAIT" and after_distance < before_distance
            if intent == "wait_reason":
                if action != "WAIT":
                    answer = ((f"机器人2在所选这一步没有等待，而是选择了{label}。")
                              if language == "zh" else
                              (f"Robot 2 did not wait on this step; it chose to {label}."))
                elif agent.position == env.layout.charger_position and agent.battery < 100:
                    answer = ((f"机器人2刚才等待是为了在充电格补充电量；当时电量为 {agent.battery:.0f}%。")
                              if language == "zh" else
                              (f"Robot 2 waited on the charger to restore energy; its battery was {agent.battery:.0f}%."))
                else:
                    answer = ("机器人2刚才选择了等待，但现有证据无法可靠说明更具体的原因。"
                              if language == "zh" else
                              "Robot 2 chose to wait, but the available evidence cannot reliably establish a more specific reason.")
            elif intent == "task_direction":
                if supported:
                    destination = "交付点" if mode == "delivery" else "取货点"
                    answer = ((f"机器人2当前在朝任务{task_label}的{destination}前进；这一步会把距离从 {before_distance} 格缩短到 {after_distance} 格。")
                              if language == "zh" else
                              (f"Robot 2 is moving toward task {task_label}'s {mode} point; this step reduces the distance from {before_distance} to {after_distance}."))
                else:
                    answer = ("现有证据无法可靠判断机器人2正在推进哪个具体任务。"
                              if language == "zh" else
                              "The available evidence cannot reliably identify a specific task Robot 2 is advancing.")
            elif supported:
                destination = "交付点" if mode == "delivery" else "取货点"
                answer = ((f"机器人2刚才{label}，是在靠近任务{task_label}的{destination}；距离从 {before_distance} 格缩短到 {after_distance} 格。")
                          if language == "zh" else
                          (f"Robot 2 chose to {label} while approaching task {task_label}'s {mode} point; the distance fell from {before_distance} to {after_distance}."))
            else:
                answer = ((f"机器人2刚才选择了{label}，但现有证据无法可靠说明更具体的原因。")
                          if language == "zh" else
                          (f"Robot 2 chose to {label}, but the available evidence cannot reliably establish a more specific reason."))
        detail = ((f"所选帧 {snapshot['state']['frame']}；实际策略动作：{action}；策略程序近似：{program_action}；两者{'一致' if agreement else '不一致'}。")
                  if language == "zh" else
                  (f"Selected frame {snapshot['state']['frame']}; policy action: {action}; program approximation: {program_action}; {'agreement' if agreement else 'disagreement'}."))
        return {"answer": answer, "evidence_detail": detail}


@dataclass
class R42ReleaseContext:
    runtime: object
    scenarios: dict
    explainer: object
    question_bank: object
    tutorial: dict
    tutorial_signature: str
    evidence: dict
    source_binding: dict
    manifest_sha256: str
    signature: str
    release: dict
    provenance: dict
    root: Path
    package_sha256: str
    _parent: object
    closed: bool = False

    def close(self):
        if not self.closed:
            self.closed = True
            closer = getattr(self._parent, "close", None)
            if callable(closer):
                closer()
            else:
                cleanup = getattr(self._parent, "cleanup", None)
                if callable(cleanup):
                    cleanup()


def _standalone_manifest(artifacts: Mapping[str, Path]) -> dict[str, Any]:
    raw = {key: path.read_bytes() for key, path in artifacts.items()}
    actor_sha = sha256(raw["actor"]).hexdigest()
    program = json.loads(raw["program"])
    bank = json.loads(raw["question_bank"])
    behavior = json.loads(raw["behavior_report"])
    training = json.loads(raw["training_report"])
    if (program.get("actor_sha256") != actor_sha
            or bank.get("actor_sha256") != actor_sha
            or behavior.get("actor_sha256") != actor_sha
            or behavior.get("passed") is not True
            or training.get("action_authority", {}).get("runtime_overrides") != 0):
        raise ValueError("r4.2 standalone behavior or Actor binding did not pass")
    value = {
        "version": STANDALONE_VERSION,
        "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": "internal_pilot",
        "formal_ready": False,
        "formal_sample_eligible": False,
        "data_persistent": False,
        "runtime_action_override": False,
        "behavior_performance_gate_passed": True,
        "behavior_performance_gate_waived": False,
        "actor_sha256": actor_sha,
        "program_content_sha256": program.get("content_sha256"),
        "question_bank_content_sha256": bank.get("content_sha256"),
        "artifacts": {
            key: {"path": STANDALONE_ARTIFACTS[key],
                  "sha256": sha256(value_raw).hexdigest(),
                  "size": len(value_raw)}
            for key, value_raw in raw.items()
        },
        "protocol": {
            "task_order": ["X1", "X2", "X3", "Y1", "Y2", "Y3"],
            "horizon": 120, "continuous_animation_ms": 380,
            "manual_preview_groups": ["A", "B"],
            "automatic_allocation_retained": True,
            "condition_a_explanation_stage": "task1",
            "condition_b_explanations": False,
            "task2_explanations": False,
            "latest_answer_only": True,
            "parallel_active_tasks": 2,
            "delivery_cap": None,
        },
        "sources": source_hashes(),
    }
    value["content_sha256"] = _digest(value)
    return value


def build_standalone_package(*, output_path: str | Path, **artifact_paths):
    if set(artifact_paths) != set(STANDALONE_ARTIFACTS):
        raise ValueError("all r4.2 standalone artifacts are required")
    paths = {key: Path(value).expanduser().resolve()
             for key, value in artifact_paths.items()}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("r4.2 standalone artifact is missing or linked")
    manifest = _standalone_manifest(paths)
    stream = io.BytesIO()
    members = [(MANIFEST_NAME, _canonical(manifest))] + [
        (STANDALONE_ARTIFACTS[key], paths[key].read_bytes())
        for key in sorted(paths)
    ]
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_BZIP2,
                         compresslevel=9) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_BZIP2
            archive.writestr(info, raw, compress_type=zipfile.ZIP_BZIP2,
                             compresslevel=9)
    raw = stream.getvalue()
    if len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("r4.2 standalone release exceeds package limit")
    destination = Path(output_path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    os.chmod(destination, 0o600)
    return {"package_sha256": sha256(raw).hexdigest(),
            "manifest_sha256": sha256(_canonical(manifest)).hexdigest(),
            "manifest_content_sha256": manifest["content_sha256"],
            "actor_sha256": manifest["actor_sha256"],
            "package_size": len(raw)}


def _manifest(parent_raw: bytes, parent_manifest_sha256: str,
              actor_sha256: str, program_sha256: str,
              scene_fingerprints: list[str]) -> dict[str, Any]:
    value = {
        "version": VERSION,
        "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": "internal_pilot",
        "formal_ready": False,
        "formal_sample_eligible": False,
        "data_persistent": False,
        "runtime_action_override": False,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "parent": {
            "package_sha256": sha256(parent_raw).hexdigest(),
            "manifest_sha256": parent_manifest_sha256,
            "actor_sha256": actor_sha256,
            "program_sha256": program_sha256,
        },
        "scene_fingerprints": scene_fingerprints,
        "protocol": {
            "task_order": ["X1", "X2", "X3", "Y1", "Y2", "Y3"],
            "condition_a_explanation_stage": "task1",
            "condition_b_explanations": False,
            "task2_explanations": False,
            "horizon": 120,
            "continuous_animation_ms": 380,
            "latest_answer_only": True,
            "round_preview_required": True,
            "parallel_active_tasks": 2,
            "delivery_cap": None,
        },
        "analysis": {
            "primary": "task2_mean_score_ab_difference",
            "secondary": [
                "task2_deliveries", "task2_efficiency",
                "task2_collision_cancellation", "task2_no_progress_steps",
                "task2_robot2_delivery_contribution",
                "task1_to_task2_scene_standardized_score_change",
            ],
            "ordinary_percentage_change_forbidden": True,
            "condition_hidden_from_participant": True,
        },
        "sources": source_hashes(),
    }
    value["content_sha256"] = _digest(value)
    return value


def build_package(parent_path: str | Path, output_path: str | Path,
                  *, parent_manifest_sha256: str) -> dict[str, str]:
    parent_path, output_path = Path(parent_path), Path(output_path)
    parent_raw = parent_path.read_bytes()
    parent_manifest = parent_release.inspect_online_release(
        expected_package_sha256=sha256(parent_raw).hexdigest(),
        expected_manifest_sha256=parent_manifest_sha256,
        package_path=parent_path)
    identities = parent_manifest["identities"]
    scenes = [row["fingerprint"] for row in parent_manifest["play_scenes"][1:7]]
    manifest = _manifest(parent_raw, parent_manifest_sha256,
                         identities["actor_sha256"], identities["program_sha256"],
                         scenes)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in ((MANIFEST_NAME, _canonical(manifest)),
                          (PARENT_NAME, parent_raw)):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, raw)
    raw = stream.getvalue()
    if len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("r4.2 release exceeds package limit")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    os.chmod(output_path, 0o600)
    return {"package_sha256": sha256(raw).hexdigest(),
            "manifest_sha256": sha256(_canonical(manifest)).hexdigest(),
            "manifest_content_sha256": manifest["content_sha256"]}


def write_base64(package_path: str | Path, output_path: str | Path) -> str:
    encoded = base64.b64encode(Path(package_path).read_bytes()) + b"\n"
    if len(encoded) > MAX_BASE64_BYTES:
        raise ValueError("r4.2 Base64 exceeds Render Secret File limit")
    path = Path(output_path); path.write_bytes(encoded); os.chmod(path, 0o600)
    return sha256(encoded).hexdigest()


def _read_outer(*, package_path=None, base64_path=None):
    if (package_path is None) == (base64_path is None):
        raise ValueError("choose exactly one r4.2 package source")
    if base64_path is not None:
        encoded = b"".join(Path(base64_path).read_bytes().split())
        raw = base64.b64decode(encoded, validate=True)
    else:
        raw = Path(package_path).read_bytes()
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("invalid r4.2 package size")
    return raw


def _open_outer(raw: bytes, expected_package_sha256: str,
                expected_manifest_sha256: str):
    if sha256(raw).hexdigest() != expected_package_sha256:
        raise ValueError("r4.2 package hash differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.comment or MANIFEST_NAME not in archive.namelist():
            raise ValueError("r4.2 archive members differ")
        manifest_raw = archive.read(MANIFEST_NAME)
        if sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
            raise ValueError("r4.2 manifest hash differs")
        manifest = json.loads(manifest_raw)
        standalone = manifest.get("version") == STANDALONE_VERSION
        expected_members = ({MANIFEST_NAME, *STANDALONE_ARTIFACTS.values()}
                            if standalone else {MANIFEST_NAME, PARENT_NAME})
        if set(archive.namelist()) != expected_members:
            raise ValueError("r4.2 archive members differ")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (path.is_absolute() or ".." in path.parts or info.is_dir()
                    or info.create_system != 3 or stat.S_IFMT(mode) != stat.S_IFREG
                    or stat.S_IMODE(mode) != 0o600
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != (zipfile.ZIP_BZIP2 if standalone
                                               else zipfile.ZIP_STORED)):
                raise ValueError("unsafe r4.2 archive member")
        payload = ({key: archive.read(path)
                    for key, path in STANDALONE_ARTIFACTS.items()}
                   if standalone else archive.read(PARENT_NAME))
    content = dict(manifest); claimed = content.pop("content_sha256", None)
    if (manifest.get("version") not in {VERSION, STANDALONE_VERSION}
            or claimed != _digest(content)
            or manifest.get("sources") != source_hashes()
            or manifest.get("release_version") != PUBLIC_RELEASE_VERSION
            or manifest.get("runtime_action_override") is not False):
        raise ValueError("r4.2 manifest differs")
    if standalone:
        if (manifest.get("behavior_performance_gate_passed") is not True
                or manifest.get("behavior_performance_gate_waived") is not False
                or set(manifest.get("artifacts", {})) != set(STANDALONE_ARTIFACTS)):
            raise ValueError("r4.2 standalone classification differs")
        for key, artifact_raw in payload.items():
            identity = manifest["artifacts"].get(key, {})
            if (identity.get("path") != STANDALONE_ARTIFACTS[key]
                    or identity.get("size") != len(artifact_raw)
                    or identity.get("sha256") != sha256(artifact_raw).hexdigest()):
                raise ValueError("r4.2 standalone artifact differs")
    elif sha256(payload).hexdigest() != manifest.get("parent", {}).get("package_sha256"):
        raise ValueError("r4.2 parent release differs")
    return manifest, payload


def inspect_online_release(*, expected_package_sha256: str,
                           expected_manifest_sha256: str,
                           package_path=None, base64_path=None):
    raw = _read_outer(package_path=package_path, base64_path=base64_path)
    return _open_outer(raw, expected_package_sha256,
                       expected_manifest_sha256)[0]


def _load_standalone(manifest, artifacts, *, expected_package_sha256,
                     expected_manifest_sha256):
    temporary = tempfile.TemporaryDirectory(prefix="warehouse-r42-release-")
    root = Path(temporary.name).resolve()
    try:
        paths = {}
        for key, raw in artifacts.items():
            path = root / Path(STANDALONE_ARTIFACTS[key]).name
            path.write_bytes(raw)
            os.chmod(path, 0o600)
            paths[key] = path
        protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        scenarios = json.loads(paths["runtime_manifest"].read_text(encoding="utf-8"))
        scenario_content = deepcopy(scenarios)
        scenario_claimed = scenario_content.pop("content_sha256", None)
        runtime = R41DiagnosticOnlineAlignmentRuntime(
            paths["actor"],
            training_protocol_path=paths["protocol"],
            manifest_path=paths["runtime_manifest"],
            expected_actor_sha256=manifest["actor_sha256"],
            expected_training_protocol_file_sha256=_file_hash(paths["protocol"]),
            expected_training_protocol_content_sha256=_digest(protocol),
            expected_manifest_file_sha256=_file_hash(paths["runtime_manifest"]),
            expected_manifest_content_sha256=scenario_claimed,
            expected_manifest_semantic_sha256=_digest(scenarios),
        )
        program = R42DecisionProgram(json.loads(
            paths["program"].read_text(encoding="utf-8")
        ))
        explainer = R42DirectExplainer(program)
        question_bank = R42CompactQuestionBank(json.loads(
            paths["question_bank"].read_text(encoding="utf-8")
        ), runtime=runtime)
        tutorial = json.loads(paths["tutorial"].read_text(encoding="utf-8"))
        tutorial_replay = tutorial_api.validate_tutorial(
            tutorial, scenarios["splits"]["play"][0]
        )
        behavior = json.loads(paths["behavior_report"].read_text(encoding="utf-8"))
        training = json.loads(paths["training_report"].read_text(encoding="utf-8"))
        if (runtime.actor_sha256 != manifest["actor_sha256"]
                or program.actor_sha256 != runtime.actor_sha256
                or behavior.get("passed") is not True
                or behavior.get("actor_sha256") != runtime.actor_sha256
                or training.get("action_authority", {}).get("runtime_overrides") != 0):
            raise ValueError("r4.2 standalone runtime evidence differs")
        release = {
            "release_version": PUBLIC_RELEASE_VERSION,
            "pilot_class": "internal_pilot",
            "namespace": "internal_pilot",
            "model_ready": True, "explanation_ready": True,
            "study_ready": True, "participant_enabled": True,
            "formal_ready": False, "formal_sample_eligible": False,
            "human_explanation_effect_validated": False,
            "data_persistent": False, "online_portable": True,
            "runtime_action_override": False,
            "behavior_performance_gate_passed": True,
            "behavior_performance_gate_waived": False,
            "test_fixture": False,
        }
        provenance = {
            "version": CONTEXT_VERSION,
            "release_version": PUBLIC_RELEASE_VERSION,
            "pilot_class": "internal_pilot",
            "package_sha256": expected_package_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "actor_sha256": runtime.actor_sha256,
            "program_content_sha256": program.signature,
            "tutorial_signature": tutorial_replay["tutorial_signature"],
            "formal_ready": False, "formal_sample_eligible": False,
            "data_persistent": False,
        }
        signature = _digest({
            "version": CONTEXT_VERSION,
            "package_sha256": expected_package_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "runtime_signature": runtime.signature,
            "explainer_signature": explainer.signature,
            "question_bank_signature": question_bank.signature,
            "tutorial_signature": tutorial_replay["tutorial_signature"],
        })
        return R42ReleaseContext(
            runtime, deepcopy(scenarios), explainer, question_bank,
            tutorial, tutorial_replay["tutorial_signature"],
            {"manifest": deepcopy(manifest), "behavior": behavior,
             "training": training, "program_audit": deepcopy(program.audit),
             "tutorial": tutorial_replay},
            source_hashes(), expected_manifest_sha256, signature,
            release, provenance, root, expected_package_sha256, temporary,
        )
    except BaseException:
        temporary.cleanup()
        raise


def load_online_release(*, expected_package_sha256: str,
                        expected_manifest_sha256: str,
                        package_path=None, base64_path=None):
    raw = _read_outer(package_path=package_path, base64_path=base64_path)
    manifest, parent_raw = _open_outer(raw, expected_package_sha256,
                                       expected_manifest_sha256)
    if manifest.get("version") == STANDALONE_VERSION:
        return _load_standalone(
            manifest, parent_raw,
            expected_package_sha256=expected_package_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    temporary = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        temporary.write(parent_raw); temporary.flush(); os.fsync(temporary.fileno())
        temporary.close()
        parent = parent_release.load_online_release(
            expected_package_sha256=manifest["parent"]["package_sha256"],
            expected_manifest_sha256=manifest["parent"]["manifest_sha256"],
            package_path=temporary.name)
    finally:
        try: os.unlink(temporary.name)
        except FileNotFoundError: pass
    tutorial = tutorial_api.build_tutorial(parent.scenarios)
    tutorial_replay = tutorial_api.validate_tutorial(
        tutorial, parent.scenarios["splits"]["play"][0])
    explainer = R42ReadableExplainer(parent.explainer)
    release = {**deepcopy(parent.release),
        "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": "internal_pilot", "data_persistent": False,
        "formal_ready": False, "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
    }
    provenance = {**deepcopy(parent.provenance),
        "version": CONTEXT_VERSION, "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": "internal_pilot", "package_sha256": expected_package_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "parent_package_sha256": manifest["parent"]["package_sha256"],
        "tutorial_signature": tutorial_replay["tutorial_signature"],
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False,
    }
    signature = _digest({"version": CONTEXT_VERSION,
        "package_sha256": expected_package_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "runtime_signature": parent.runtime.signature,
        "explainer_signature": explainer.signature,
        "tutorial_signature": tutorial_replay["tutorial_signature"]})
    return R42ReleaseContext(
        parent.runtime, deepcopy(parent.scenarios), explainer,
        parent.question_bank, tutorial, tutorial_replay["tutorial_signature"],
        {"parent": deepcopy(parent.evidence), "tutorial": tutorial_replay,
         "manifest": deepcopy(manifest)}, source_hashes(),
        expected_manifest_sha256, signature, release, provenance,
        parent.root, expected_package_sha256, parent)


__all__ = ["VERSION", "STANDALONE_VERSION", "CONTEXT_VERSION",
           "PUBLIC_RELEASE_VERSION", "build_package",
           "build_standalone_package", "write_base64",
           "inspect_online_release", "load_online_release"]
