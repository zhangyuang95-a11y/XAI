"""Private observed197 prediction candidates; no CLI or automatic pool sampling.

All physical work, including loader replays, is reserved through caller-owned
callbacks. A successful content check never authorizes a model or public release.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import uuid

from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.warehouse_public_history_runtime import PublicHistoryRuntime
from env.warehouse_native.policy import ACTIONS
from env.warehouse_native.scenarios import scenario_fingerprint
from ui.warehouse_native_bank import KINDS, ACTION_LABELS, _positions, _select, _checks
from ui.warehouse_view import serialize_warehouse_state, warehouse_map_payload

VERSION = "warehouse-observed197-prediction-bank.v1"
REQUIRED_EXCLUSIONS = frozenset(("train", "validation", "extraction", "final_test", "play"))
_FIELDS = frozenset(("version", "status", "formal_ready", "release_ready", "test_fixture",
    "actor_sha256", "protocol_sha256", "runtime_signature", "sources", "sources_sha256",
    "excluded_pools_sha256", "pool_namespace", "pool_scenes", "trajectory_steps", "minimum_frame",
    "trajectories", "items", "checks", "generation_audit", "counterfactual_filter"))
FILTER = "exactly_three_nonterminal_steps_without_new_random_replacement_tasks"


def bank_sources(runtime):
    sources = deepcopy(runtime.sources)
    for name in ("ui/warehouse_public_history_bank.py", "ui/warehouse_native_bank.py", "ui/warehouse_view.py"):
        sources[name] = file_hash(ROOT / name)
    return sources


class _Operations:
    def __init__(self, before, after, phase):
        if not callable(before) or not callable(after):
            raise ValueError("Explicit reservation and durable completion callbacks are required")
        self.before, self.after, self.phase = before, after, phase
        self.namespace = uuid.uuid4().hex
        self.completed = self.steps = self.reserved = 0

    def call(self, function, *, kind, scene, frame, maximum_steps):
        context = {"operation_id": f"bank_{self.namespace}_{self.completed:06d}",
            "phase": self.phase, "kind": kind, "scenario_id": scene, "frame": frame,
            "maximum_steps": maximum_steps, "training_steps": 0, "participant_derived": False}
        permission = self.before(deepcopy(context))
        if not isinstance(permission, dict) or permission.get("execution_permitted") is not True:
            raise ValueError("Question-bank operation has no sampling permission")
        self.reserved += maximum_steps
        # Any exception leaves the caller's reservation unconfirmed. No refund,
        # retry, hidden probe, or optimistic completion is performed here.
        result = function()
        actual = (result["steps_executed"] if kind == "wait_three" else 0 if maximum_steps == 0 else 1)
        if type(actual) is not int or not 0 <= actual <= maximum_steps:
            raise ValueError("Question-bank execution exceeded its reservation")
        self.after({**deepcopy(context), "actual_steps": actual, "result_sha256": digest(result)})
        self.steps += actual
        self.completed += 1
        return result

    def report(self):
        return {"completed_operations": self.completed, "reserved_upper_bound": self.reserved,
                "actual_environment_steps": self.steps, "automatic_resume": False, "reservations_refunded": False}


def _pools(runtime, scenes, excluded):
    if type(runtime) is not PublicHistoryRuntime:
        raise ValueError("An actual observed197 runtime is required")
    runtime.verify_binding()
    if (not isinstance(excluded, dict) or not REQUIRED_EXCLUSIONS <= set(excluded)
            or any(not isinstance(rows, list) for rows in excluded.values())):
        raise ValueError("Explicit train/validation/extraction/final_test/play exclusion pools are required")
    fingerprints = set()
    for rows in excluded.values():
        for entry in rows:
            if any(key.startswith("public_feedback") for key in entry.get("snapshot", {})):
                env = runtime.from_snapshot(entry["snapshot"])
                if scenario_fingerprint(env) != entry.get("fingerprint"):
                    raise ValueError("Excluded public-history snapshot fingerprint differs")
            else:
                env = runtime.environment(entry)
            fingerprints.add(scenario_fingerprint(env))
    if not isinstance(scenes, list) or not 4 <= len(scenes) <= 100:
        raise ValueError("Question bank needs 4 to 100 independent supplied scenarios")
    found, ids = set(), set()
    for scene in scenes:
        if type(scene.get("id")) is not str or not scene["id"] or scene["id"] in ids:
            raise ValueError("Duplicate or invalid independent scenario identity")
        env = runtime.environment(scene)
        actual = scenario_fingerprint(env)
        if env.state.frame != 0 or env.done or actual in fingerprints or actual in found:
            raise ValueError("Question-pool initial state overlaps an excluded pool, itself, or a noninitial frame")
        ids.add(scene["id"]); found.add(actual)
    return {scene["id"]: scene for scene in scenes}


def _preview(runtime, snapshot):
    env = runtime.from_snapshot(snapshot)
    state = serialize_warehouse_state(env.get_state(), selected_agent="robot_2", reveal_policy=False)
    for key in ("human_route_regret_units", "policy_hidden"):
        state.pop(key, None)
    for agent in state["agents"]:
        agent.pop("proposed_action", None); agent.pop("reward", None)
    return {"state": state, "map": warehouse_map_payload(env.layout),
            "public_feedback": env.public_history()}


def _candidate(runtime, env, scene_id, kind, operations):
    snapshot = env.snapshot(); before = digest(snapshot)
    item = {"kind": kind, "scenario_id": scene_id, "frame": env.state.frame, "snapshot": snapshot,
            "snapshot_sha256": before, "preview": _preview(runtime, snapshot)}
    if kind == "next_action":
        actions, evidence = operations.call(lambda: runtime.decision(env), kind=kind,
            scene=scene_id, frame=env.state.frame, maximum_steps=0)
        item.update(answer=actions["robot_2"], diversity_key=actions["robot_2"], evidence=evidence,
            prompt={"zh": "在下图状态，AI 2 下一步会选择哪个动作（碰撞处理前）？",
                    "en": "In this state, which action will AI 2 select next, before collision resolution?"},
            options=[{"value": action, "label": deepcopy(ACTION_LABELS[action])} for action in ACTIONS])
    elif kind == "wait_three":
        branch = operations.call(lambda: runtime.counterfactual(snapshot, ["WAIT"] * 3, steps=3),
            kind=kind, scene=scene_id, frame=env.state.frame, maximum_steps=3)
        rows = branch["transitions"]
        if (len(rows) != 3 or any(row["done"] for row in rows)
                or any(row["after"]["state"]["next_task_index"] != row["before"]["state"]["next_task_index"] for row in rows)):
            return None
        end = rows[-1]["after"]["state"]["agents"][1]["position"]
        start = list(env.state.agents[1].position)
        alternatives = [position for position in _positions(env, start) if list(position) != list(end)]
        if len(alternatives) < 3:
            return None
        rng = random.Random(int(before[:16], 16)); rng.shuffle(alternatives)
        choices = [tuple(end), *alternatives[:3]]; rng.shuffle(choices)
        options = [{"value": f"{r},{c}", "label": {"zh": f"{chr(65+i)} · 第 {r+1} 行，第 {c+1} 列",
            "en": f"{chr(65+i)} · Row {r+1}, column {c+1}"}, "position": [r, c], "marker": chr(65+i)}
            for i, (r, c) in enumerate(choices)]
        item.update(answer=f"{end[0]},{end[1]}", diversity_key=f"{end[0]-start[0]},{end[1]-start[1]}",
            evidence={"transitions": rows, "transitions_sha256": digest(rows), "assumed_player_actions": ["WAIT"] * 3},
            options=options, prompt={"zh": "假设玩家连续等待三步，AI 2 在第三步结束时位于哪一格？字母标记仅表示选项位置。",
            "en": "If the player waits for three consecutive steps, where will AI 2 be after the third step? Letter markers identify the answer options only."})
        item["preview"]["question_markers"] = [{"position": option["position"], "label": option["marker"]} for option in options]
    else:
        raise ValueError("Unknown prediction kind")
    if digest(env.snapshot()) != before:
        raise ValueError("Question construction changed the source frame or RNG")
    return item


def generate_bank(runtime, pool_scenes, excluded_pools, *, trajectory_steps,
                  before_operation, after_operation, minimum_frame=1):
    """Generate bounded candidate bytes in memory; caller decides private persistence."""
    if (type(trajectory_steps) is not int or not 1 <= trajectory_steps <= runtime.config.horizon
            or type(minimum_frame) is not int or not 0 <= minimum_frame < trajectory_steps):
        raise ValueError("Invalid bounded source trajectory/frame interval")
    scenes = deepcopy(pool_scenes); exclusions = deepcopy(excluded_pools)
    _pools(runtime, scenes, exclusions)
    sources = bank_sources(runtime)
    operations = _Operations(before_operation, after_operation, "generation")
    candidates, trajectories = [], {}
    for scene in scenes:
        env = runtime.environment(scene); rows = []
        for _ in range(trajectory_steps):
            if env.done: break
            if env.state.frame >= minimum_frame:
                for kind in KINDS:
                    item = _candidate(runtime, env, scene["id"], kind, operations)
                    if item is not None: candidates.append(item)
            actions, _ = operations.call(lambda: runtime.decision(env), kind="selfplay_decision",
                scene=scene["id"], frame=env.state.frame, maximum_steps=0)
            rows.append(operations.call(lambda: runtime.step(env, actions["robot_1"]), kind="trajectory",
                scene=scene["id"], frame=env.state.frame, maximum_steps=1))
        trajectories[scene["id"]] = rows
    runtime.verify_binding()
    if bank_sources(runtime) != sources: raise ValueError("Question-bank execution source changed")
    items = _select(candidates)
    return {"version": VERSION, "status": "candidate", "formal_ready": False, "release_ready": False,
        "test_fixture": runtime.test_fixture, "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256, "runtime_signature": runtime.signature,
        "sources": sources, "sources_sha256": digest(sources), "excluded_pools_sha256": digest(exclusions),
        "pool_namespace": "independent_question_bank_development_observed197",
        "pool_scenes": scenes, "trajectory_steps": trajectory_steps, "minimum_frame": minimum_frame,
        "trajectories": trajectories, "items": items, "checks": _checks(items),
        "generation_audit": operations.report(), "counterfactual_filter": FILTER}


class PublicHistoryQuestionBank:
    """Strict private content loader. Source trajectories and answers are replayed."""
    def __init__(self, path, runtime, excluded_pools, *, expected_bank_sha256,
                 before_operation, after_operation, allow_test_fixture=False):
        path = Path(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream: raw = stream.read()
        if type(expected_bank_sha256) is not str or sha256(raw).hexdigest() != expected_bank_sha256:
            raise ValueError("Question-bank external byte anchor differs")
        b = json.loads(raw)
        sources = bank_sources(runtime)
        if (set(b) != _FIELDS or b["version"] != VERSION or b["status"] != "candidate"
                or b["formal_ready"] is not False or b["release_ready"] is not False
                or type(allow_test_fixture) is not bool or b["test_fixture"] is not runtime.test_fixture
                or (runtime.test_fixture and not allow_test_fixture)
                or b["actor_sha256"] != runtime.actor_sha256 or b["protocol_sha256"] != runtime.protocol_sha256
                or b["runtime_signature"] != runtime.signature or b["sources"] != sources
                or b["sources_sha256"] != digest(sources) or b["excluded_pools_sha256"] != digest(excluded_pools)
                or b["pool_namespace"] != "independent_question_bank_development_observed197"
                or b["counterfactual_filter"] != FILTER):
            raise ValueError("Question-bank source, fixture, schema or Actor identity differs")
        scenes = _pools(runtime, b["pool_scenes"], excluded_pools)
        limit, minimum = b["trajectory_steps"], b["minimum_frame"]
        if (type(limit) is not int or not 1 <= limit <= runtime.config.horizon or type(minimum) is not int
                or not 0 <= minimum < limit or set(b["trajectories"]) != set(scenes)):
            raise ValueError("Invalid source trajectory coverage")
        if (not isinstance(b["items"], list) or len(b["items"]) > 8
                or len({item["id"] for item in b["items"]}) != len(b["items"])):
            raise ValueError("Invalid bank item identities/count")
        for item in b["items"]:
            if (item["kind"] not in KINDS or item["scenario_id"] not in scenes
                    or item["id"] not in {f"prediction_{item['kind']}_{i}" for i in range(1, 5)}
                    or type(item["frame"]) is not int or not minimum <= item["frame"] < limit):
                raise ValueError("Unknown or out-of-range question source")
        operations = _Operations(before_operation, after_operation, "load_reverification")
        snapshots, candidates = {}, []
        for scene_id, scene in scenes.items():
            env = runtime.environment(scene); rows = b["trajectories"][scene_id]
            if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
                raise ValueError("Missing source trajectory")
            for row in rows:
                if env.done: raise ValueError("Source trajectory continues past its terminal")
                snapshots[(scene_id, env.state.frame)] = env.snapshot()
                if env.state.frame >= minimum:
                    for kind in KINDS:
                        candidate = _candidate(runtime, env, scene_id, kind, operations)
                        if candidate is not None: candidates.append(candidate)
                actions, _ = operations.call(lambda: runtime.decision(env), kind="selfplay_decision",
                    scene=scene_id, frame=env.state.frame, maximum_steps=0)
                actual = operations.call(lambda: runtime.step(env, actions["robot_1"]), kind="trajectory",
                    scene=scene_id, frame=env.state.frame, maximum_steps=1)
                if digest(actual) != digest(row):
                    raise ValueError("Recorded source is not the same-Actor legal full-history trajectory")
            if len(rows) != limit and not env.done:
                raise ValueError("Source trajectory was silently truncated")
        expected_items = _select(candidates)
        if digest(expected_items) != digest(b["items"]):
            raise ValueError("Question selection, answer, options, public history or evidence differ")
        for item in b["items"]:
            snapshot = snapshots.get((item["scenario_id"], item["frame"]))
            if (snapshot is None or digest(snapshot) != item["snapshot_sha256"]
                    or digest(item["snapshot"]) != item["snapshot_sha256"]):
                raise ValueError("Question snapshot differs from its legal trajectory")
            runtime.from_snapshot(item["snapshot"])
        checks = _checks(b["items"])
        if b["checks"] != checks: raise ValueError("Saved content-check claims differ")
        if b["generation_audit"] != operations.report():
            raise ValueError("Saved generation work differs from complete deterministic reexecution")
        runtime.verify_binding()
        if bank_sources(runtime) != sources: raise ValueError("Question-bank source changed while replaying")
        self._bank, self._items = deepcopy(b), deepcopy(b["items"])
        self.checks, self.content_eligible, self.eligible = checks, checks["passed"], checks["passed"]
        self.test_fixture, self.actor_sha256, self.runtime_signature = runtime.test_fixture, runtime.actor_sha256, runtime.signature
        self.excluded_pools_sha256 = b["excluded_pools_sha256"]
        self.signature = digest({"version": VERSION, "bank_sha256": expected_bank_sha256,
            "sources_sha256": digest(sources), "runtime_signature": runtime.signature})
        self.audit = operations.report()

    @property
    def bank(self): return deepcopy(self._bank)

    @property
    def items(self): return deepcopy(self._items)

    def public_items(self):
        if not self.content_eligible: return []
        return [{"id": item["id"], "type": "choice", "prediction_kind": item["kind"], "required": True,
            "prompt": deepcopy(item["prompt"]),
            "options": [{"value": option["value"], "label": deepcopy(option["label"])} for option in item["options"]],
            "preview": deepcopy(item["preview"]), "source_frame": item["frame"], "source_scenario": item["scenario_id"]}
            for item in self._items]

    def summary(self):
        return {"status": "candidate_ready" if self.content_eligible else "candidate_failed", "formal_ready": False,
            "release_ready": False, "model_capability_evaluated": False, "available": self.content_eligible,
            "item_count": len(self.public_items()), "test_fixture": self.test_fixture,
            "preview_history_available_to_both_conditions": True}

    def grade(self, answers):
        if not self.content_eligible: raise ValueError("Question bank content checks have not passed")
        result = {"bank_signature": self.signature, "candidate_only": True, "formal_ready": False}
        for kind in KINDS:
            items = [item for item in self._items if item["kind"] == kind]
            if any(answers.get(item["id"]) not in {option["value"] for option in item["options"]} for item in items):
                raise ValueError("Incomplete or invalid prediction answers")
            correct = sum(answers[item["id"]] == item["answer"] for item in items)
            result[kind] = {"correct": correct, "total": len(items), "accuracy": correct / len(items)}
        return result
