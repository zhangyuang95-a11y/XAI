"""Versioned private question bank for the two actual observed197 runtime families.

The source/answer replay algorithm follows the frozen public-history bank. Its
pure candidate, preview, selection and reservation operations are reused with
actual registered runtimes; no metadata relabelling or class substitution occurs.
Content checks do not grant model, explanation or participant qualification.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path

from backend import warehouse_runtime_family as registry
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from env.warehouse_native.scenarios import scenario_fingerprint
from ui import warehouse_public_history_bank as original
from ui.warehouse_public_history_bank import (
    _candidate, _preview, _Operations, _select, _checks, KINDS,
    REQUIRED_EXCLUSIONS, FILTER,
)

VERSION = "warehouse-runtime-family-prediction-bank.v1"
POOL_NAMESPACE = "independent_question_bank_development_runtime_family"
_FIELDS = original._FIELDS | {"runtime_family", "runtime_version", "registry_sources_sha256"}


def bank_sources(runtime):
    registry.family(runtime)
    sources = original.bank_sources(runtime)
    sources.update(registry.execution_sources())
    sources[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return sources


def _pools(runtime, scenes, excluded, *, allow_test_fixture):
    registry.verify(runtime, allow_test_fixture=allow_test_fixture)
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



def generate_bank(runtime, pool_scenes, excluded_pools, *, trajectory_steps,
                  before_operation, after_operation, minimum_frame=1, allow_test_fixture=False):
    """Generate bounded candidate bytes in memory; caller decides private persistence."""
    if (type(trajectory_steps) is not int or not 1 <= trajectory_steps <= runtime.config.horizon
            or type(minimum_frame) is not int or not 0 <= minimum_frame < trajectory_steps):
        raise ValueError("Invalid bounded source trajectory/frame interval")
    scenes = deepcopy(pool_scenes); exclusions = deepcopy(excluded_pools)
    identity = registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    _pools(runtime, scenes, exclusions, allow_test_fixture=allow_test_fixture)
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
    if registry.verify(runtime, allow_test_fixture=allow_test_fixture) != identity:
        raise ValueError("Runtime family changed during question generation")
    if bank_sources(runtime) != sources: raise ValueError("Question-bank execution source changed")
    items = _select(candidates)
    return {"version": VERSION, "status": "candidate", "formal_ready": False, "release_ready": False,
        "test_fixture": runtime.test_fixture, "runtime_family": identity["family"],
        "runtime_version": identity["runtime_version"], "registry_sources_sha256": identity["registry_sources_sha256"],
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256, "runtime_signature": runtime.signature,
        "sources": sources, "sources_sha256": digest(sources), "excluded_pools_sha256": digest(exclusions),
        "pool_namespace": POOL_NAMESPACE,
        "pool_scenes": scenes, "trajectory_steps": trajectory_steps, "minimum_frame": minimum_frame,
        "trajectories": trajectories, "items": items, "checks": _checks(items),
        "generation_audit": operations.report(), "counterfactual_filter": FILTER}


class FamilyQuestionBank(original.PublicHistoryQuestionBank):
    """Independent family admission and replay; reuse unchanged public rendering/grade methods."""
    def __init__(self, path, runtime, excluded_pools, *, expected_bank_sha256,
                 before_operation, after_operation, allow_test_fixture=False):
        identity = registry.verify(runtime, allow_test_fixture=allow_test_fixture)
        path = Path(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream: raw = stream.read()
        if type(expected_bank_sha256) is not str or sha256(raw).hexdigest() != expected_bank_sha256:
            raise ValueError("Question-bank external byte anchor differs")
        b = json.loads(raw)
        sources = bank_sources(runtime)
        if (set(b) != _FIELDS or b["version"] != VERSION or b["status"] != "candidate"
                or b["runtime_family"] != identity["family"] or b["runtime_version"] != identity["runtime_version"]
                or b["registry_sources_sha256"] != identity["registry_sources_sha256"]
                or b["formal_ready"] is not False or b["release_ready"] is not False
                or type(allow_test_fixture) is not bool or b["test_fixture"] is not runtime.test_fixture
                or (runtime.test_fixture and not allow_test_fixture)
                or b["actor_sha256"] != runtime.actor_sha256 or b["protocol_sha256"] != runtime.protocol_sha256
                or b["runtime_signature"] != runtime.signature or b["sources"] != sources
                or b["sources_sha256"] != digest(sources) or b["excluded_pools_sha256"] != digest(excluded_pools)
                or b["pool_namespace"] != POOL_NAMESPACE
                or b["counterfactual_filter"] != FILTER):
            raise ValueError("Question-bank source, fixture, schema or Actor identity differs")
        scenes = _pools(runtime, b["pool_scenes"], excluded_pools, allow_test_fixture=allow_test_fixture)
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
        if registry.verify(runtime, allow_test_fixture=allow_test_fixture) != identity:
            raise ValueError("Runtime family changed during question verification")
        if bank_sources(runtime) != sources: raise ValueError("Question-bank source changed while replaying")
        self._bank, self._items = deepcopy(b), deepcopy(b["items"])
        self.checks, self.content_eligible, self.eligible = checks, checks["passed"], checks["passed"]
        self.test_fixture, self.actor_sha256, self.runtime_signature = runtime.test_fixture, runtime.actor_sha256, runtime.signature
        self.excluded_pools_sha256 = b["excluded_pools_sha256"]
        self.signature = digest({"version": VERSION, "bank_sha256": expected_bank_sha256,
            "sources_sha256": digest(sources), "runtime_signature": runtime.signature})
        self.audit = operations.report()

