"""Private, replayable audits of the real renderer, without release eligibility.

This CLI neither trains a network nor opens a web explanation interface. A
plain private context lets the production rendering body be audited before
its first acceptance report exists. It cannot be used as a NativeExplainer
and never fabricates acceptance files or changes eligibility properties.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from backend.training.warehouse_native_common import atomic_json, digest, file_hash
from core.program import ExecutableProgram
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.explanation import EXPLANATION_VERSION, NativeExplainer
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS, NumPyNativeActor
from env.warehouse_native.runtime import NativeRuntime, runtime_signature
from env.warehouse_native.scenarios import reset_scenario, scenario_fingerprint

VERSION = "warehouse-native-offline-answer-audit.v1"
COLLECTION = {"version": "warehouse-native-answer-rollouts.v1",
    "split": "explanation_test", "first_scenarios": 12,
    "profiles_in_order": ["skilled", "assertive", "noisy"],
    "partner_rng": "numpy.SeedSequence([98131, scene_index, 1])",
    "neural_execution": "deterministic", "rollout": "initial_state_to_true_terminal"}


def _weights_digest(actor):
    result = sha256()
    for key, value in sorted(actor.weights.items()):
        result.update(digest([key, str(value.dtype), list(value.shape)]).encode())
        result.update(value.tobytes())
    return result.hexdigest()


class _OfflineRenderingContext:
    """Evidence integrity only; deliberately has no eligibility interface."""
    def __init__(self, program_path, runtime, *, allow_test_fixture=False):
        from ui.warehouse_native_answer_verification import answer_verification_sources
        if type(runtime) is not NativeRuntime or type(runtime.actor) is not NumPyNativeActor:
            raise ValueError("offline_audit_requires_original_native_runtime")
        fresh = NumPyNativeActor(runtime.actor.path)
        if (fresh.sha256 != runtime.actor.sha256 or fresh.metadata != runtime.actor.metadata
                or _weights_digest(fresh) != _weights_digest(runtime.actor)):
            raise ValueError("offline_audit_actor_changed_before_context")
        if runtime.actor.metadata.get("test_fixture") is True and not allow_test_fixture:
            raise ValueError("synthetic Actor cannot enter a real answer audit")
        if not allow_test_fixture and (type(runtime.actor.metadata.get("joint_steps")) is not int
                                      or runtime.actor.metadata["joint_steps"] <= 0):
            raise ValueError("offline_audit_requires_recorded_training_steps")
        self.program_path = Path(program_path).resolve()
        payload = json.loads(self.program_path.read_text())
        source = payload["program"] if payload.get("version") == "warehouse_native_rcpd_feedback_v1" else payload
        self.program = ExecutableProgram.from_dict(source)
        self.actor_sha256 = runtime.actor.sha256
        env = NativeWarehouseEnv()
        if (self.program.metadata.get("native_source_actor_sha256") != self.actor_sha256
                or tuple(self.program.action_names) != ACTIONS
                or len(self.program.feature_names) != len(env.feature_names)
                or set(self.program.feature_names) != set(env.feature_names)
                or self.program.metadata.get("action_legality_features")
                or self.program.metadata.get("action_constraint_reason_features")):
            raise ValueError("offline_audit_program_actor_or_feature_contract")
        stack = [(self.program.root, 0)]
        leaves = 0
        while stack:
            node, depth = stack.pop()
            if depth > 8:
                raise ValueError("offline_audit_tree_exceeds_depth_limit")
            if node.is_leaf:
                p = np.asarray(node.probabilities)
                leaves += 1
                if (leaves > 64 or p.shape != (5,) or not np.isfinite(p).all()
                        or np.any(p < 0) or not np.isclose(p.sum(), 1)):
                    raise ValueError("offline_audit_invalid_tree_leaf")
            else:
                if (node.feature not in env.feature_names or node.threshold is None
                        or not np.isfinite(node.threshold) or node.left is None or node.right is None):
                    raise ValueError("offline_audit_invalid_tree_predicate")
                stack += [(node.left, depth + 1), (node.right, depth + 1)]
        self._runtime = runtime
        self._actor_path = runtime.actor.path.resolve()
        self._actor_memory = _weights_digest(runtime.actor)
        self._actor_metadata = digest(runtime.actor.metadata)
        self._program_file = file_hash(self.program_path)
        self._program_memory = digest(self.program.to_dict())
        self._runtime_signature = runtime.signature
        self._sources = answer_verification_sources()
        self._assert_current(runtime)

    def _assert_current(self, runtime):
        from ui.warehouse_native_answer_verification import answer_verification_sources
        if (runtime is not self._runtime or type(runtime) is not NativeRuntime
                or type(runtime.actor) is not NumPyNativeActor
                or runtime.actor.path.resolve() != self._actor_path
                or runtime.actor.sha256 != self.actor_sha256
                or runtime.actor.artifact_sha256 != self.actor_sha256
                or file_hash(self._actor_path) != self.actor_sha256
                or digest(runtime.actor.metadata) != self._actor_metadata
                or _weights_digest(runtime.actor) != self._actor_memory):
            raise ValueError("offline_audit_actor_changed")
        if (runtime.signature != self._runtime_signature
                or runtime_signature(runtime.actor) != self._runtime_signature
                or answer_verification_sources() != self._sources):
            raise ValueError("offline_audit_source_or_runtime_changed")
        if (file_hash(self.program_path) != self._program_file
                or digest(self.program.to_dict()) != self._program_memory):
            raise ValueError("offline_audit_program_changed")


def offline_context(program_path, runtime, *, allow_test_fixture=False):
    """Python-only context; fixture opt-in is deliberately absent from the CLI."""
    return _OfflineRenderingContext(program_path, runtime, allow_test_fixture=allow_test_fixture)


class AuditStepCounter:
    """Observe actual environment calls without replacing any execution code.

    sys.setprofile is thread-local. This CLI runs in its own process; the
    original web server, its environment, and its profiler are unaffected.
    Counts include failed attempts if they really enter env.step, rather than
    guessing from the number of expected answers or prediction horizon.
    """
    def __init__(self):
        self.counts = {}
        self._code = NativeWarehouseEnv.step.__code__
        self._stage = None

    @contextmanager
    def stage(self, name):
        if self._stage is not None or sys.getprofile() is not None:
            raise ValueError("offline_audit_cannot_replace_an_existing_profiler")
        self._stage = name
        self.counts.setdefault(name, 0)
        def observe(frame, event, arg):
            if event == "call" and frame.f_code is self._code:
                self.counts[name] += 1
        sys.setprofile(observe)
        try:
            yield
        finally:
            sys.setprofile(None)
            self._stage = None


def collect_cases(runtime, context, scenarios, *, counter=None):
    """Run the full prespecified matrix; never skip an unsuccessful anchor."""
    from ui.warehouse_native_answer_verification import (
        CASE_SPEC_VERSION, SCENARIO_COUNT, VERSION as ANSWER_VERSION,
        answer_verification_sources, case_specs, request_for_spec, trajectory_spec)
    counter = counter or AuditStepCounter()
    specs = case_specs()
    scenes = scenarios["splits"]["explanation_test"][:SCENARIO_COUNT]
    if (len(scenes) != SCENARIO_COUNT or len({s["id"] for s in scenes}) != SCENARIO_COUNT
            or len({s["fingerprint"] for s in scenes}) != SCENARIO_COUNT
            or any(not s["id"].startswith("explanation_test_") for s in scenes)):
        raise ValueError("offline_audit_requires_fixed_twelve_scene_pool")
    header = {"version": ANSWER_VERSION, "renderer_version": EXPLANATION_VERSION,
        "actor_sha256": runtime.actor.sha256, "program_sha256": file_hash(context.program_path),
        "runtime_signature": runtime.signature, "scenario_manifest_sha256": digest(scenarios),
        "case_spec_version": CASE_SPEC_VERSION, "case_spec_sha256": digest(specs),
        "trajectory_spec_sha256": digest(trajectory_spec()),
        "sources": answer_verification_sources(), "collection": COLLECTION,
        "test_fixture": runtime.actor.metadata.get("test_fixture") is True,
        "passed": False, "eligible": False, "formal_ready": False,
        "neural_updates": 0, "scope": "Offline renderer evidence audit; no capability or tree-fidelity qualification."}
    cases, rollouts = [], []
    for index, scene in enumerate(scenes):
        env = NativeWarehouseEnv()
        reset_scenario(env, scene)
        if env.state.frame != 0 or scenario_fingerprint(env) != scene["fingerprint"]:
            raise ValueError("offline_audit_requires_true_initial_state")
        frames = {0: {"after": env.snapshot()}}
        actions = []
        profile = COLLECTION["profiles_in_order"][index % 3]
        rng = np.random.default_rng(np.random.SeedSequence([98131, index, 1]))
        with counter.stage("collect_rollouts"):
            while not env.done:
                player = partner_action(env, "robot_1", profile, rng)
                record = runtime.step(env, player)
                actions.append(player)
                frames[env.state.frame] = record
        terminal_hash = digest(env.snapshot())
        rollouts.append({"scenario_id": scene["id"], "initial_fingerprint": scene["fingerprint"],
            "profile": profile, "player_actions": actions,
            "terminal_snapshot_sha256": terminal_hash, "joint_steps": len(actions)})
        with counter.stage("render_answers"):
            for spec in specs:
                target = {"initial": 0, "first": 1, "third": 3, "terminal": len(actions)}[spec["anchor"]]
                request = request_for_spec(spec, target)
                row = {"case_id": scene["id"] + "/" + spec["id"], "scenario_id": scene["id"],
                    "initial_fingerprint": scene["fingerprint"], "player_actions": actions[:target],
                    "question": spec["question"], "language": spec["language"],
                    "intent": spec["expected"]["intent"], "expected": deepcopy(spec["expected"]),
                    "request": request, "passed": False, "frame_binding_passed": False,
                    "evidence_verified": False, "answer": ""}
                if target not in frames:
                    row.update(missing_case=True, error="required_anchor_missing", record_sha256=None)
                else:
                    record = frames[target]
                    row["record_sha256"] = digest(record)
                    try:
                        row["answer"] = NativeExplainer.answer(context, request, deepcopy(record), runtime)
                    except ValueError as error:
                        row["error"] = str(error)
                    if digest(record) != row["record_sha256"] or digest(env.snapshot()) != terminal_hash:
                        raise ValueError("offline_rendering_changed_source_record_or_rollout")
                row["answer_sha256"] = sha256(row["answer"].encode()).hexdigest()
                cases.append(row)
        context._assert_current(runtime)
    return {**header, "cases": cases}, rollouts


def run(actor_path, program_path, scenarios_path, output):
    """Create a new private report directory; never amend an acceptance file."""
    from ui.warehouse_native_answer_verification import (
        answer_verification_sources, case_specs, verify_answer_report)
    from ui.warehouse_native_release import verify_scenarios
    actor_path, program_path, scenarios_path, output = (
        Path(p).resolve() for p in (actor_path, program_path, scenarios_path, output))
    if output.exists():
        raise ValueError("Refusing to overwrite an existing answer audit directory")
    runtime = NativeRuntime(actor_path)
    context = offline_context(program_path, runtime)
    scenarios = json.loads(scenarios_path.read_text())
    verify_scenarios(scenarios)
    if runtime.actor.metadata.get("scenario_manifest_sha256") != digest(scenarios):
        raise ValueError("offline_audit_actor_scenario_mismatch")
    scene_sha = file_hash(scenarios_path)
    sources = answer_verification_sources()
    output.mkdir(parents=True)
    atomic_json(output / "case_spec.json", {"collection": COLLECTION, "cases": case_specs(),
        "sources": sources, "frozen_before_collection": True})
    atomic_json(output / "progress.json", {"version": VERSION, "status": "collecting",
        "neural_updates": 0, "eligible": False})
    counter = AuditStepCounter()
    try:
        raw, rollouts = collect_cases(runtime, context, scenarios, counter=counter)
        atomic_json(output / "rollouts.json", rollouts)
        atomic_json(output / "answers_before_verification.json", raw)
        with counter.stage("independent_verification"):
            verified = verify_answer_report(raw, program_path, runtime, scenarios)
        outcome = {c["case_id"]: c for c in verified["cases"]}
        for case in raw["cases"]:
            actual = outcome[case["case_id"]]
            for key in ("passed", "frame_binding_passed", "evidence_verified"):
                case[key] = actual.get(key) is True
        raw["passed"] = verified["passed"]
        context._assert_current(runtime)
        if sources != answer_verification_sources() or scene_sha != file_hash(scenarios_path):
            raise ValueError("offline_audit_frozen_input_changed")
        atomic_json(output / "answers.json", raw)
        atomic_json(output / "verification.json", verified)
        summary = {"version": VERSION, "status": "completed", "content_audit_passed": verified["passed"],
            "eligible": False, "formal_ready": False, "neural_updates": 0,
            "audit_environment_steps": counter.counts, "total_audit_environment_steps": sum(counter.counts.values()),
            "cases": len(raw["cases"]), "passed_cases": sum(c["passed"] for c in verified["cases"]),
            "actor_sha256": runtime.actor.sha256, "program_sha256": file_hash(program_path),
            "scenario_manifest_sha256": digest(scenarios), "runtime_signature": runtime.signature,
            "sources": sources, "answers_sha256": file_hash(output / "answers.json"),
            "verification_sha256": file_hash(output / "verification.json"), "final_test_read": False,
            "scope": "Exact renderer facts on frozen bilingual cases; capability, tree fidelity and human explanation effects remain separate."}
        atomic_json(output / "progress.json", summary)
        return summary
    except BaseException as error:
        atomic_json(output / "progress.json", {"version": VERSION, "status": "failed",
            "reason": str(error), "eligible": False, "neural_updates": 0,
            "audit_environment_steps": counter.counts,
            "total_audit_environment_steps": sum(counter.counts.values())})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.actor, args.program, args.scenarios, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, ensure_ascii=False))
    return 0 if result["content_audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
