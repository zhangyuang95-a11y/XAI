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

from ui import warehouse_alignment_r41_diagnostic_release_v9 as parent_release
from ui import warehouse_alignment_r42_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r42-internal-pilot-envelope.v1"
CONTEXT_VERSION = "warehouse-r42-internal-pilot-release.v1"
PUBLIC_RELEASE_VERSION = "r4.2-internal-pilot"
MANIFEST_NAME = "manifest.json"
PARENT_NAME = "artifacts/r41_parent_release.zip"
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
             ROOT / "ui/warehouse_alignment_r42_tutorial.py")
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
            self._parent.close()


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
        if archive.comment or set(archive.namelist()) != {MANIFEST_NAME, PARENT_NAME}:
            raise ValueError("r4.2 archive members differ")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (path.is_absolute() or ".." in path.parts or info.is_dir()
                    or info.create_system != 3 or stat.S_IFMT(mode) != stat.S_IFREG
                    or stat.S_IMODE(mode) != 0o600
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_STORED):
                raise ValueError("unsafe r4.2 archive member")
        manifest_raw = archive.read(MANIFEST_NAME)
        parent_raw = archive.read(PARENT_NAME)
    if sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("r4.2 manifest hash differs")
    manifest = json.loads(manifest_raw)
    content = dict(manifest); claimed = content.pop("content_sha256", None)
    if (manifest.get("version") != VERSION or claimed != _digest(content)
            or manifest.get("sources") != source_hashes()
            or manifest.get("release_version") != PUBLIC_RELEASE_VERSION
            or manifest.get("runtime_action_override") is not False
            or sha256(parent_raw).hexdigest()
                != manifest.get("parent", {}).get("package_sha256")):
        raise ValueError("r4.2 manifest differs")
    return manifest, parent_raw


def inspect_online_release(*, expected_package_sha256: str,
                           expected_manifest_sha256: str,
                           package_path=None, base64_path=None):
    raw = _read_outer(package_path=package_path, base64_path=base64_path)
    return _open_outer(raw, expected_package_sha256,
                       expected_manifest_sha256)[0]


def load_online_release(*, expected_package_sha256: str,
                        expected_manifest_sha256: str,
                        package_path=None, base64_path=None):
    raw = _read_outer(package_path=package_path, base64_path=base64_path)
    manifest, parent_raw = _open_outer(raw, expected_package_sha256,
                                       expected_manifest_sha256)
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


__all__ = ["VERSION", "CONTEXT_VERSION", "PUBLIC_RELEASE_VERSION",
           "build_package", "write_base64", "inspect_online_release",
           "load_online_release"]
