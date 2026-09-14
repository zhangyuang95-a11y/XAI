"""Readable, frame-bound explanations for the r4.1 diagnostic v9 program.

The participant answer is deliberately separated from the audit evidence.  A
participant sees one or two plain-language sentences.  Neural distributions,
the public-tree trace, artifact identities, and isolated-branch details remain
in ``evidence_detail``.  This component renders evidence only; a release
admission module must separately authorize participant use.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from backend.warehouse_alignment_online_runtime import ACTIONS as _RUNTIME_ACTIONS
from backend.warehouse_alignment_online_runtime import digest
from backend.warehouse_alignment_online_explanation import (
    LABELS,
    _agent,
    _best_progress,
    _distance,
    _objective_candidates,
    _objective_label,
    _position_after_action,
    _response,
    _task_slot,
    clarify_answer,
    parse_question,
    verify_historical_transition,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    GROUPS,
    R41DiagnosticPublicTreeProgramV9,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r41-diagnostic-online-readable-answers.v9"
RCPD_VERSION = "warehouse-r41-diagnostic-rcpd-v9-fit.v1"
EXACT_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
ARTIFACT_BINDING_VERSION = "warehouse-r41-diagnostic-explanation-artifact-binding.v9"
ACCESS_VERSION = "warehouse-r41-diagnostic-explanation-access.v9"
ACCESS_SURFACES = ("live", "history", "round_review")

QUICK_QUESTIONS = {
    "zh": (
        "机器人2刚才为什么这样行动？",
        "机器人2刚才为什么等待？",
        "我们刚才为什么发生碰撞？",
        "我的上一步动作影响了机器人2吗？",
        "机器人2当前在朝哪个任务前进？",
        "机器人2现在需要充电吗？",
    ),
    "en": (
        "Why did Robot 2 choose that action?",
        "Why did Robot 2 wait?",
        "Why did we just collide?",
        "Did my last action affect Robot 2?",
        "Which task is Robot 2 moving toward?",
        "Does Robot 2 need to charge now?",
    ),
}

_FRAME_REFERENCE = re.compile(
    r"第\s*(\d+)\s*(?:帧|步)|\bframe\s+(\d+)\b|\bat step\s+(\d+)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[。！？]|(?<!\d)[.!?](?!\d)")


def explanation_sources() -> dict[str, str]:
    """Return the complete source closure used by the v9 renderer."""

    paths = (
        Path(__file__),
        ROOT / "backend/warehouse_alignment_online_explanation.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "backend/warehouse_r41_diagnostic_boosted_tree.py",
        ROOT / "backend/warehouse_r41_diagnostic_public_features_v9.py",
        ROOT / "backend/warehouse_r41_diagnostic_public_tree_program_v9.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
    )
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("r4.1 v9 explanation source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(label + " must be an exact lowercase SHA-256")
    return value


def make_artifact_binding(
    *, actor_sha256: str, program_sha256: str, runtime_signature: str,
    runtime_manifest_sha256: str, candidate_lock_sha256: str,
    public_feature_contract_sha256: str,
) -> dict[str, Any]:
    """Build the hash-only interface used by a later admitted artifact set."""

    value: dict[str, Any] = {
        "version": ARTIFACT_BINDING_VERSION,
        "actor_sha256": _sha(actor_sha256, "actor"),
        "program_sha256": _sha(program_sha256, "program"),
        "runtime_signature": _sha(runtime_signature, "runtime signature"),
        "runtime_manifest_sha256": _sha(
            runtime_manifest_sha256, "runtime manifest"),
        "candidate_lock_sha256": _sha(candidate_lock_sha256, "candidate lock"),
        "public_feature_contract_sha256": _sha(
            public_feature_contract_sha256, "public feature contract"),
        "renderer_sources_sha256": digest(explanation_sources()),
        "runtime_action_override": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def validate_artifact_binding(
    value: Mapping[str, Any], *, actor_sha256: str, program_sha256: str,
    runtime_signature: str, runtime_manifest_sha256: str,
    public_feature_contract_sha256: str,
) -> dict[str, Any]:
    """Authenticate identities without opening evaluation rows or predictions."""

    expected_fields = {
        "version", "actor_sha256", "program_sha256", "runtime_signature",
        "runtime_manifest_sha256", "candidate_lock_sha256",
        "public_feature_contract_sha256", "renderer_sources_sha256",
        "runtime_action_override", "formal_ready", "content_sha256",
    }
    if (not isinstance(value, Mapping) or set(value) != expected_fields
            or value.get("version") != ARTIFACT_BINDING_VERSION
            or value.get("actor_sha256") != _sha(actor_sha256, "actor")
            or value.get("program_sha256") != _sha(program_sha256, "program")
            or value.get("runtime_signature")
                != _sha(runtime_signature, "runtime signature")
            or value.get("runtime_manifest_sha256")
                != _sha(runtime_manifest_sha256, "runtime manifest")
            or value.get("public_feature_contract_sha256")
                != _sha(public_feature_contract_sha256,
                        "public feature contract")
            or re.fullmatch(r"[0-9a-f]{64}", str(
                value.get("candidate_lock_sha256", ""))) is None
            or value.get("renderer_sources_sha256")
                != digest(explanation_sources())
            or value.get("runtime_action_override") is not False
            or value.get("formal_ready") is not False
            or value.get("content_sha256") != digest({
                key: child for key, child in value.items()
                if key != "content_sha256"
            })):
        raise ValueError("v9_explanation_artifact_binding_mismatch")
    return deepcopy(dict(value))


def explanation_access(context: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the A/Task-1-only rule to live, history, and round review."""

    fields = {
        "condition", "stage", "surface", "request_kind", "active_run_id",
        "bound_run_id", "selected_frame", "available_frames", "round_closed",
    }
    if not isinstance(context, Mapping) or set(context) != fields:
        raise ValueError("explanation_access_context_malformed")
    condition = context.get("condition")
    stage = context.get("stage")
    surface = context.get("surface")
    request_kind = context.get("request_kind")
    active_run_id = context.get("active_run_id")
    bound_run_id = context.get("bound_run_id")
    selected_frame = context.get("selected_frame")
    frames = context.get("available_frames")
    round_closed = context.get("round_closed")
    if (condition not in ("A", "B") or stage not in ("task1", "task2")
            or surface not in ACCESS_SURFACES
            or request_kind not in ("new_question", "read_answer")
            or type(active_run_id) is not str or not active_run_id
            or type(bound_run_id) is not str or not bound_run_id
            or type(selected_frame) is not int or selected_frame < 0
            or not isinstance(frames, (list, tuple)) or not frames
            or any(type(frame) is not int or frame < 0 for frame in frames)
            or len(set(frames)) != len(frames)
            or type(round_closed) is not bool):
        raise ValueError("explanation_access_context_malformed")

    reason = "allowed"
    if active_run_id != bound_run_id:
        reason = "cross_run_binding_denied"
    elif selected_frame not in frames:
        reason = "frame_not_available"
    elif stage == "task2":
        reason = "task2_explanations_closed"
    elif condition != "A":
        reason = "condition_b_replay_only"
    elif surface == "live" and (round_closed or selected_frame != max(frames)):
        reason = "live_requires_latest_open_frame"
    elif surface == "round_review" and not round_closed:
        reason = "round_review_requires_closed_round"
    elif surface == "history" and round_closed:
        reason = "closed_round_uses_round_review"
    return {
        "version": ACCESS_VERSION,
        "allowed": reason == "allowed",
        "reason": reason,
        "bound_run_id": bound_run_id,
        "selected_frame": selected_frame,
        "surface": surface,
        "request_kind": request_kind,
    }


def require_explanation_access(context: Mapping[str, Any]) -> dict[str, Any]:
    decision = explanation_access(context)
    if not decision["allowed"]:
        raise PermissionError(decision["reason"])
    return decision


def _participant_response(answer: str, evidence_detail: str) -> dict[str, str]:
    """Enforce the short participant/audit separation at the last boundary."""

    result = _response(answer, evidence_detail)
    if len(_SENTENCE_END.findall(result["answer"])) > 2:
        raise ValueError("Participant answer must contain at most two sentences")
    return result


def _probability_line(values: np.ndarray, language: str) -> str:
    pairs = [
        f"{LABELS[language][action]} {100.0 * float(values[index]):.2f}%"
        for index, action in enumerate(EXACT_ACTIONS)
    ]
    return "、".join(pairs) if language == "zh" else ", ".join(pairs)


def _validated_program_trace(trace: Any) -> dict[str, Any]:
    """Validate the structured v8 trace before rendering any part of it."""

    if not isinstance(trace, Mapping):
        raise ValueError("v9_program_trace_malformed")
    prediction = trace.get("prediction")
    probabilities = np.asarray(trace.get("final_probabilities", ()), dtype=float)
    routes = trace.get("routes")
    triggered = trace.get("triggered_routes")
    base = trace.get("base")
    if (prediction not in EXACT_ACTIONS or probabilities.shape != (5,)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=2e-12)
            or not isinstance(routes, list) or len(routes) != len(GROUPS)
            or not isinstance(triggered, list)
            or any(group not in GROUPS for group in triggered)
            or not isinstance(base, Mapping)
            or not isinstance(base.get("program_trace"), Mapping)):
        raise ValueError("v9_program_trace_malformed")
    if [item.get("group") for item in routes if isinstance(item, Mapping)] != list(GROUPS):
        raise ValueError("v9_program_trace_malformed")
    for item in routes:
        active = item.get("triggered")
        program_trace = item.get("program_trace")
        if type(active) is not bool:
            raise ValueError("v9_program_trace_malformed")
        if active != (item["group"] in triggered):
            raise ValueError("v9_program_trace_malformed")
        if active != isinstance(program_trace, Mapping):
            raise ValueError("v9_program_trace_malformed")
    return dict(trace)


def _selected_path_conditions(trace: Mapping[str, Any], action: str,
                              language: str) -> tuple[str, ...]:
    """Summarize a few exact public predicates from the structured trace.

    These strings are audit-only.  They are never used to decide an action or
    presented as a neural internal cause.
    """

    class_index = EXACT_ACTIONS.index(action)
    component_traces: list[tuple[str, Mapping[str, Any]]] = [
        ("base", trace["base"]["program_trace"]),
    ]
    for route in trace["routes"]:
        if route["triggered"]:
            component_traces.append((route["group"], route["program_trace"]))

    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for component, program_trace in component_traces:
        trees = program_trace.get("trees")
        if not isinstance(trees, list):
            raise ValueError("v9_program_trace_malformed")
        for tree in trees:
            if (not isinstance(tree, Mapping)
                    or type(tree.get("class_index")) is not int
                    or not isinstance(tree.get("path"), list)):
                raise ValueError("v9_program_trace_malformed")
            if tree["class_index"] == class_index and tree["path"]:
                candidates.append((abs(float(tree.get("contribution", 0.0))),
                                   component, tree))
    candidates.sort(key=lambda item: (-item[0], item[1], int(item[2]["tree_index"])))
    conditions: list[str] = []
    seen: set[tuple[str, str, float, bool]] = set()
    for _, component, tree in candidates:
        for step in tree["path"]:
            if not isinstance(step, Mapping):
                raise ValueError("v9_program_trace_malformed")
            feature = step.get("feature_name")
            observed = step.get("observed")
            threshold = step.get("threshold")
            went_left = step.get("went_left")
            if (type(feature) is not str or type(went_left) is not bool
                    or type(observed) not in (int, float)
                    or type(threshold) not in (int, float)
                    or not np.isfinite(float(observed))
                    or not np.isfinite(float(threshold))):
                raise ValueError("v9_program_trace_malformed")
            key = (feature, "<=", float(threshold), went_left)
            if key in seen:
                continue
            seen.add(key)
            relation = "≤" if went_left else ">"
            if language == "zh":
                rendered = (f"{component}: {feature} {relation} {float(threshold):.6g}"
                            f"（实际 {float(observed):.6g}）")
            else:
                rendered = (f"{component}: {feature} {relation} {float(threshold):.6g} "
                            f"(observed {float(observed):.6g})")
            conditions.append(rendered)
            if len(conditions) == 3:
                return tuple(conditions)
    return tuple(conditions)


def _technical_detail(*, language: str, frame: int, focus: str,
                      actor_action: str | None = None,
                      actor_probabilities: np.ndarray | None = None,
                      program_action: str | None = None,
                      program_trace: Mapping[str, Any] | None = None,
                      program_matches: bool | None = None,
                      actor_sha256: str | None = None,
                      replay_verified: bool = False,
                      counterfactual: bool = False,
                      extra: tuple[str, ...] = ()) -> str:
    zh = language == "zh"
    lines = [
        (f"绑定帧：{frame}（{'已执行动作' if focus == 'executed' else '下一次决策'}）"
         if zh else
         f"Bound frame: {frame} ({'executed action' if focus == 'executed' else 'next decision'})")
    ]
    if actor_action in EXACT_ACTIONS:
        lines.append((f"策略提交动作：{LABELS[language][actor_action]}" if zh else
                      f"Policy-submitted action: {LABELS[language][actor_action]}"))
    if actor_probabilities is not None:
        lines.append(("动作概率：" if zh else "Action probabilities: ")
                     + _probability_line(actor_probabilities, language))
    if program_action in EXACT_ACTIONS:
        lines.append((
            f"策略近似程序：{LABELS[language][program_action]}（与提交动作{'一致' if program_matches else '不一致'}）"
            if zh else
            f"Policy approximation: {LABELS[language][program_action]} "
            f"({'agrees' if program_matches else 'disagrees'} with the submitted action)"
        ))
    if program_trace is not None and program_action in EXACT_ACTIONS:
        trace = _validated_program_trace(program_trace)
        program_probabilities = np.asarray(trace["final_probabilities"], dtype=float)
        lines.append(("近似程序分布：" if zh else "Approximation distribution: ")
                     + _probability_line(program_probabilities, language))
        if program_matches:
            triggered = trace["triggered_routes"]
            lines.append(("触发的公开状态路由：" if zh else
                          "Triggered public-state routes: ")
                         + (("、" if zh else ", ").join(triggered)
                            if triggered else ("无" if zh else "none")))
            conditions = _selected_path_conditions(trace, program_action, language)
            if conditions:
                lines.append(("近似程序路径：" if zh else
                              "Approximation paths: ")
                             + ("；" if zh else "; ").join(conditions))
        else:
            lines.append(("近似程序与实际动作不一致，因此未显示或使用其路径。" if zh else
                          "The approximation disagreed with the actual action, so its paths were omitted and not used."))
    if replay_verified:
        lines.append(("历史动作、策略提交和物理结果已通过隔离重放核验。" if zh else
                      "Historical action, policy submission, and physical outcome passed isolated replay verification."))
    if counterfactual:
        lines.append(("反事实使用同一冻结策略的隔离副本，未改变真实回合。" if zh else
                      "The counterfactual used an isolated copy of the same frozen policy and did not change the live run."))
    lines.extend(str(item) for item in extra if item)
    if actor_sha256:
        lines.append((f"Actor SHA-256：{actor_sha256}" if zh else
                      f"Actor SHA-256: {actor_sha256}"))
    return "\n".join(lines)


def _task_event(events: Any, objective: Mapping[str, Any]) -> bool:
    expected = "delivery" if objective.get("kind") == "delivery" else "pickup"
    task = objective.get("task")
    task_id = getattr(task, "task_id", None)
    return any(
        isinstance(item, Mapping)
        and item.get("agent_id") == "robot_2"
        and item.get("event") == expected
        and item.get("task_id") == task_id
        for item in (events if isinstance(events, list) else ())
    )


def _physical_progress_counterfactual(
    *, runtime: Any, bound: Mapping[str, Any], current: Mapping[str, Any],
    objective: Mapping[str, Any], chosen_action: str, comparison_action: str,
    player_action: str, use_before: bool,
) -> dict[str, Any]:
    """Compare two robot actions through isolated public warehouse physics.

    This never routes an action into the live runtime.  It restores two fresh
    environments from the bound snapshot and advances each exactly once.  For
    an executed explanation, support additionally requires that the chosen
    branch reproduces the already confirmed robot position.
    """

    if (chosen_action not in EXACT_ACTIONS or comparison_action not in EXACT_ACTIONS
            or player_action not in EXACT_ACTIONS):
        raise ValueError("physical_counterfactual_action_registry_differs")
    source_hash = digest(bound)
    chosen_env = runtime.from_snapshot(deepcopy(bound))
    comparison_env = runtime.from_snapshot(deepcopy(bound))
    _, _, _, _, chosen_info = chosen_env.step({
        "robot_1": player_action, "robot_2": chosen_action,
    })
    _, _, _, _, comparison_info = comparison_env.step({
        "robot_1": player_action, "robot_2": comparison_action,
    })
    if digest(bound) != source_hash:
        raise ValueError("physical_counterfactual_mutated_source")

    target = tuple(objective["target"])
    before_env = runtime.from_snapshot(deepcopy(bound))
    before_distance = _distance(before_env, _agent(before_env).position, target)
    chosen_distance = _distance(chosen_env, _agent(chosen_env).position, target)
    comparison_distance = _distance(
        comparison_env, _agent(comparison_env).position, target)
    chosen_event = _task_event(chosen_info.get("events"), objective)
    comparison_event = _task_event(comparison_info.get("events"), objective)
    chosen_progress = chosen_event or chosen_distance < before_distance
    physically_better = ((chosen_event and not comparison_event)
                         or chosen_distance < comparison_distance)
    reproduced = True
    if use_before:
        confirmed_env = runtime.from_snapshot(deepcopy(current))
        reproduced = (tuple(_agent(confirmed_env).position)
                      == tuple(_agent(chosen_env).position))
    return {
        "supported": bool(use_before and reproduced and chosen_progress
                          and physically_better),
        "actual_public_progress": bool(use_before and reproduced and chosen_progress),
        "chosen_branch_reproduced": bool(reproduced),
        "comparison_action": comparison_action,
        "assumed_player_action": player_action,
        "before_distance": before_distance,
        "chosen_distance": chosen_distance,
        "comparison_distance": comparison_distance,
        "chosen_event": chosen_event,
        "comparison_event": comparison_event,
        "source_snapshot_sha256": source_hash,
    }


def _physical_detail(value: Mapping[str, Any], language: str) -> str:
    if language == "zh":
        return (
            f"隔离物理对照：玩家假设{LABELS[language][value['assumed_player_action']]}，"
            f"机器人2实际动作分支的目标距离 {value['before_distance']}→{value['chosen_distance']}，"
            f"改为{LABELS[language][value['comparison_action']]}时为 {value['comparison_distance']}；"
            f"实际进展={'是' if value['actual_public_progress'] else '否'}，"
            f"对照支持={'是' if value['supported'] else '否'}。"
        )
    return (
        f"Isolated physics comparison: assuming the player {LABELS[language][value['assumed_player_action']]}, "
        f"the actual Robot 2 branch changes target distance {value['before_distance']}→{value['chosen_distance']}; "
        f"using {LABELS[language][value['comparison_action']]} instead gives {value['comparison_distance']}. "
        f"Actual progress={'yes' if value['actual_public_progress'] else 'no'}; "
        f"comparison support={'yes' if value['supported'] else 'no'}."
    )


def _goal_purpose_supported(program_matches: bool,
                            physical: Mapping[str, Any] | None) -> bool:
    """Require every independent support leg for a task-purpose statement."""

    return bool(
        program_matches is True
        and isinstance(physical, Mapping)
        and physical.get("actual_public_progress") is True
        and physical.get("chosen_branch_reproduced") is True
        and physical.get("supported") is True
    )


class R41DiagnosticOnlineAlignmentExplainerV9:
    """Evidence renderer bound to the exact runtime and explicit v9 program."""

    def __init__(self, program_path: str | Path, *, expected_program_sha256: str,
                 runtime: Any, artifact_binding: Mapping[str, Any] | None = None,
                 allow_test_fixture: bool = False):
        from backend.warehouse_r41_diagnostic_online_runtime import (
            R41DiagnosticOnlineAlignmentRuntime as SourceRuntime,
        )
        from backend.warehouse_r41_diagnostic_online_runtime_portable_v1 import (
            R41DiagnosticOnlineAlignmentRuntime as PortableRuntime,
        )

        if tuple(_RUNTIME_ACTIONS) != EXACT_ACTIONS:
            raise ValueError("Runtime action registry differs from the exact five-action contract")
        if (type(runtime) not in (SourceRuntime, PortableRuntime)
                or type(allow_test_fixture) is not bool
                or runtime.test_fixture is not allow_test_fixture):
            raise ValueError("Explicit matching r4.1 diagnostic runtime scope is required")
        runtime.verify_binding()
        self.program_path = Path(program_path).expanduser().resolve()
        if self.program_path.is_symlink() or not self.program_path.is_file():
            raise ValueError("Program must be a canonical regular file")
        raw = self.program_path.read_bytes()
        if (not isinstance(expected_program_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_program_sha256) is None
                or sha256(raw).hexdigest() != expected_program_sha256):
            raise ValueError("Program differs from its external hash")
        try:
            payload = json.loads(raw)
            self.program = R41DiagnosticPublicTreeProgramV9.from_dict(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("RCPD v9 public-tree program differs") from error

        metadata = self.program.metadata
        expected_features = tuple(runtime.actor.metadata["feature_names"])
        expected_metadata = {
            "version": RCPD_VERSION,
            "public_feature_contract_sha256": digest(
                self.program.relations.contract()),
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
            "formal_ready": False,
        }
        if (tuple(self.program.base_feature_names) != expected_features
                or tuple(self.program.action_names) != EXACT_ACTIONS
                or tuple(self.program.base_program.classes) != tuple(range(5))
                or any(tuple(item["program"]["classes"]) != tuple(range(5))
                       for item in self.program.to_dict()["specialists"])
                or any(metadata.get(key) != value
                       for key, value in expected_metadata.items())
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    metadata.get("binding_sha256", ""))) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    metadata.get("fit_config_sha256", ""))) is None
                or metadata.get("action_legality_features") not in (None, {}, [])
                or metadata.get("action_constraint_reason_features") not in (None, {}, [])):
            raise ValueError("Program source, v9 public schema, or action authority differs")

        self.actor_sha256 = runtime.actor_sha256
        self.runtime_signature = runtime.signature
        self.program_sha256 = expected_program_sha256
        self.program_content_sha256 = digest(self.program.to_dict())
        self.sources = explanation_sources()
        self.test_fixture = allow_test_fixture
        if artifact_binding is None:
            if not allow_test_fixture:
                raise ValueError("Explicit v9 explanation artifact binding required")
            self.artifact_binding = None
        else:
            self.artifact_binding = validate_artifact_binding(
                artifact_binding,
                actor_sha256=self.actor_sha256,
                program_sha256=self.program_sha256,
                runtime_signature=self.runtime_signature,
                runtime_manifest_sha256=runtime.manifest_file_sha256,
                public_feature_contract_sha256=digest(
                    self.program.relations.contract()),
            )
        self.eligible = False
        self.participant_enabled = False
        self.study_ready = False
        self.explanation_qualified = True
        self.release_ready = False
        self.fixture_enabled = allow_test_fixture
        self.signature = digest({
            "version": VERSION,
            "scope": "r4.1-diagnostic",
            "runtime": self.runtime_signature,
            "program": self.program_sha256,
            "artifact_binding": (None if self.artifact_binding is None
                                 else self.artifact_binding["content_sha256"]),
            "sources": self.sources,
        })
        self.contract_report = {
            "version": VERSION,
            "signature": self.signature,
            "test_fixture": allow_test_fixture,
            "eligibility_evaluated": True,
            "explanation_eligible": True,
            "explanation_qualified": True,
            "study_ready": False,
            "participant_enabled": False,
            "release_ready": False,
            "scope": "diagnostic_component_requires_separate_admission",
            "action_registry": list(EXACT_ACTIONS),
        }

    def _assert_current(self, runtime: Any) -> None:
        from backend.warehouse_r41_diagnostic_online_runtime import (
            R41DiagnosticOnlineAlignmentRuntime as SourceRuntime,
        )
        from backend.warehouse_r41_diagnostic_online_runtime_portable_v1 import (
            R41DiagnosticOnlineAlignmentRuntime as PortableRuntime,
        )

        if (type(runtime) not in (SourceRuntime, PortableRuntime)
                or runtime.verify_binding() != self.runtime_signature
                or runtime.actor_sha256 != self.actor_sha256
                or runtime.test_fixture is not self.test_fixture
                or tuple(runtime.actor.metadata["feature_names"])
                    != tuple(self.program.base_feature_names)
                or tuple(self.program.action_names) != EXACT_ACTIONS
                or explanation_sources() != self.sources):
            raise ValueError("diagnostic_explanation_runtime_version_mismatch")
        if (sha256(self.program_path.read_bytes()).hexdigest() != self.program_sha256
                or digest(self.program.to_dict()) != self.program_content_sha256):
            raise ValueError("diagnostic_explanation_program_changed")
        if self.artifact_binding is not None:
            validate_artifact_binding(
                self.artifact_binding,
                actor_sha256=self.actor_sha256,
                program_sha256=self.program_sha256,
                runtime_signature=self.runtime_signature,
                runtime_manifest_sha256=runtime.manifest_file_sha256,
                public_feature_contract_sha256=digest(
                    self.program.relations.contract()),
            )
        if (self.eligible or self.participant_enabled or self.study_ready
                or not self.explanation_qualified or self.release_ready):
            raise ValueError("This component cannot grant participant qualification")

    def answer_study(
        self, request: Mapping[str, Any], frame_record: Mapping[str, Any],
        runtime: Any, *, access_context: Mapping[str, Any],
    ) -> dict[str, str]:
        """Render only after the server supplies an authorized study binding."""

        access = require_explanation_access(access_context)
        if request.get("frame") != access["selected_frame"]:
            raise ValueError("explanation_frame_mismatch")
        return self.answer(request, frame_record, runtime)

    def answer(self, request: Mapping[str, Any], frame_record: Mapping[str, Any],
               runtime: Any) -> dict[str, str]:
        """Render one frame-bound answer without mutating either input."""

        request_copy = deepcopy(request)
        record_copy = deepcopy(frame_record)
        request_hash = digest(request_copy)
        record_hash = digest(record_copy)
        result = self._answer(request_copy, record_copy, runtime)
        if digest(request_copy) != request_hash or digest(record_copy) != record_hash:
            raise ValueError("explanation_mutated_bound_input")
        return result

    def _answer(self, request: Mapping[str, Any], frame_record: Mapping[str, Any],
                runtime: Any) -> dict[str, str]:
        self._assert_current(runtime)
        language = "en" if request.get("language") == "en" else "zh"
        parsed = parse_question(request.get("question"), request.get("focus"))
        if frame_record.get("runtime_signature") != runtime.signature:
            raise ValueError("explanation_runtime_version_mismatch")
        runtime.from_snapshot(frame_record["after"])
        current = frame_record["after"]
        selected_frame = current["state"]["frame"]
        if ("frame" in request
                and (type(request["frame"]) is not int
                     or request["frame"] != selected_frame)):
            raise ValueError("explanation_frame_mismatch")
        mentions = _FRAME_REFERENCE.findall(str(request.get("question", "")))
        if any(int(next(value for value in match if value)) != selected_frame
               for match in mentions):
            parsed = {"intent": "clarify", "reason": "select_requested_frame"}
        if parsed["intent"] == "clarify":
            return _participant_response(
                clarify_answer(parsed["reason"], selected_frame, language),
                _technical_detail(
                    language=language, frame=selected_frame,
                    focus=request.get("focus", "executed"),
                    actor_sha256=self.actor_sha256,
                    extra=(("问题未能绑定到单一、受支持的证据查询。"
                            if language == "zh" else
                            "The question could not be bound to one supported evidence query."),),
                ),
            )

        focus = parsed.get("focus", "executed")
        use_before = focus == "executed"
        if use_before and "before" not in frame_record:
            return _participant_response(
                clarify_answer("no_executed_action_at_initial_frame",
                               selected_frame, language),
                _technical_detail(language=language, frame=selected_frame,
                                  focus=focus, actor_sha256=self.actor_sha256),
            )
        replay_verified = False
        if "before" in frame_record:
            if frame_record["before"]["state"]["frame"] + 1 != selected_frame:
                raise ValueError("historical_frame_sequence_mismatch")
            frame_record = verify_historical_transition(
                frame_record, runtime, self.actor_sha256)
            current = frame_record["after"]
            replay_verified = True
        bound = frame_record["before"] if use_before else current

        if parsed["intent"] == "counterfactual":
            branch = runtime.counterfactual(
                deepcopy(bound), list(parsed["player_actions"]),
                steps=parsed["steps"])
            assumed = branch["assumed_player_actions"]
            transitions = branch["transitions"]
            teammate_actions = [item["submitted_actions"]["robot_2"]
                                for item in transitions]
            if any(action not in EXACT_ACTIONS for action in teammate_actions):
                raise ValueError("counterfactual_action_registry_differs")
            deliveries = sum(
                item["after"]["state"]["total_deliveries"]
                - item["before"]["state"]["total_deliveries"]
                for item in transitions)
            collisions = sum(
                item["after"]["state"]["robot_collision_events"]
                - item["before"]["state"]["robot_collision_events"]
                for item in transitions)
            if not teammate_actions:
                answer = ("该帧的回合已经结束，不能继续推进。" if language == "zh" else
                          "The round has already ended at this frame.")
            elif language == "zh":
                choices = "、".join(LABELS[language][a] for a in teammate_actions)
                answer = (f"如果你按设定动作推进 {len(teammate_actions)} 步，机器人2会依次选择{choices}。"
                          f"团队新增配送 {deliveries} 件、发生 {collisions} 次碰撞，真实回合保持不变。")
            else:
                choices = ", ".join(LABELS[language][a] for a in teammate_actions)
                answer = (f"If you take the specified actions for {len(teammate_actions)} steps, "
                          f"Robot 2 chooses {choices}. The team adds {deliveries} deliveries and "
                          f"has {collisions} collisions; the live round remains unchanged.")
            rows = []
            for item in transitions:
                after = item["after"]["state"]
                submitted = item["submitted_actions"]["robot_2"]
                executed = item["executed_actions"]["robot_2"]
                rows.append((
                    f"第 {after['frame']} 步：你{LABELS[language][item['participant_action']]}；"
                    f"策略提交{LABELS[language][submitted]}，实际执行{LABELS[language][executed]}。"
                    if language == "zh" else
                    f"Step {after['frame']}: you {LABELS[language][item['participant_action']]}; "
                    f"policy submitted {LABELS[language][submitted]}, physically executed {LABELS[language][executed]}."
                ))
            return _participant_response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256,
                replay_verified=replay_verified, counterfactual=True,
                extra=(("玩家动作假设：" if language == "zh" else
                        "Player-action assumptions: ")
                       + ("、" if language == "zh" else ", ").join(
                           LABELS[language][action] for action in assumed), *rows),
            ))

        env = runtime.from_snapshot(deepcopy(bound))
        if parsed["intent"] == "rules":
            answer = (
                f"成功移动一格消耗 {env.config.move_battery_cost:g}% 电量，在充电格停留一步最多恢复 {env.config.charge_per_wait:g}%。"
                "撞墙或机器人冲突会取消移动，但仍消耗一步。"
                if language == "zh" else
                f"A successful one-cell move costs {env.config.move_battery_cost:g}% battery, and staying on the charger restores up to {env.config.charge_per_wait:g}% per turn. "
                "A wall or robot conflict cancels movement but still consumes a turn."
            )
            return _participant_response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256,
                extra=(("依据：公开环境配置与物理规则。" if language == "zh" else
                        "Basis: public environment configuration and physics."),),
            ))
        if parsed["intent"] == "failure":
            reason = current["state"].get("terminal_reason")
            labels = {
                "battery_shutdown": ("至少一台机器人在充电格外耗尽电量",
                                     "at least one robot exhausted its battery away from the charger"),
                "horizon": ("达到本局步数上限", "the round reached its turn limit"),
            }
            if reason not in labels:
                answer = ("这一帧没有记录回合失败。" if language == "zh" else
                          "No round failure is recorded at this frame.")
            else:
                answer = (f"本局在第 {selected_frame} 帧结束，因为{labels[reason][0]}。"
                          if language == "zh" else
                          f"The round ended at frame {selected_frame} because {labels[reason][1]}.")
            return _participant_response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256,
                replay_verified=replay_verified,
                extra=((f"终局记录：{reason or 'none'}" if language == "zh" else
                        f"Terminal record: {reason or 'none'}"),),
            ))
        if not use_before and env.done:
            answer = (f"第 {selected_frame} 帧回合已结束，没有下一次决策。"
                      if language == "zh" else
                      f"The round ended at frame {selected_frame}; there is no next decision.")
            return _participant_response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256))

        obs = np.asarray(env.observations()["robot_2"], dtype=np.float32)
        _, decision = runtime.decision(env)
        actor_probabilities = np.asarray(
            decision["probabilities"]["robot_2"], dtype=float)
        actor_action = decision["policy_actions"]["robot_2"]
        if (actor_action not in EXACT_ACTIONS or actor_probabilities.shape != (5,)
                or not np.isfinite(actor_probabilities).all()):
            raise ValueError("runtime_action_registry_differs")
        features = dict(zip(env.feature_names, map(float, obs)))
        program_action, program_trace = self.program.predict_with_trace(features)
        program_trace = _validated_program_trace(program_trace)
        if program_action != program_trace["prediction"] or program_action not in EXACT_ACTIONS:
            raise ValueError("v9_program_trace_prediction_differs")
        program_matches = program_action == actor_action

        if use_before:
            recorded = frame_record.get("decision", {})
            recorded_probabilities = np.asarray(
                recorded.get("probabilities", {}).get("robot_2", ()), dtype=float)
            if (recorded.get("actor_sha256") != self.actor_sha256
                    or recorded.get("frame") != bound["state"]["frame"]
                    or recorded.get("policy_actions", {}).get("robot_2") != actor_action
                    or recorded.get("observation_hashes", {}).get("robot_2")
                        != sha256(obs.tobytes()).hexdigest()
                    or recorded_probabilities.shape != (5,)
                    or not np.isfinite(recorded_probabilities).all()
                    or not np.allclose(recorded_probabilities, actor_probabilities,
                                       atol=1e-6, rtol=1e-5)
                    or frame_record.get("submitted_actions", {}).get("robot_2")
                        != actor_action):
                raise ValueError("historical_neural_decision_cannot_be_verified")

        def evidence(*extra: str) -> str:
            return _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_action=actor_action,
                actor_probabilities=actor_probabilities,
                program_action=program_action,
                program_trace=program_trace,
                program_matches=program_matches,
                actor_sha256=self.actor_sha256,
                replay_verified=replay_verified,
                extra=tuple(extra),
            )

        teammate = _agent(env)
        after_env = runtime.from_snapshot(deepcopy(current))
        after_teammate = _agent(after_env)
        history = after_env.public_history() if use_before else env.public_history()
        collision_kind = history.get("collision_kind") if history.get("valid") else "none"

        if parsed["intent"] == "collision":
            if not use_before:
                answer = ("是否碰撞还取决于你下一步的动作，目前不能提前确定。"
                          if language == "zh" else
                          "A collision also depends on your next action, so it cannot be determined yet.")
            elif collision_kind == "same_target":
                player_action = frame_record["submitted_actions"]["robot_1"]
                answer = (f"机器人2选择了{LABELS[language][actor_action]}，你选择了{LABELS[language][player_action]}；双方要进入同一格，所以环境取消了两人的移动。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]} and you chose {LABELS[language][player_action]}. Both moves targeted the same cell, so the environment canceled them.")
            elif collision_kind == "swap":
                player_action = frame_record["submitted_actions"]["robot_1"]
                answer = (f"机器人2选择了{LABELS[language][actor_action]}，你选择了{LABELS[language][player_action]}；双方会互换位置，所以环境取消了两人的移动。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]} and you chose {LABELS[language][player_action]}. The moves would swap positions, so the environment canceled them.")
            elif collision_kind == "occupied_stationary":
                canceled = history.get("move_canceled", {})
                player_action = frame_record["submitted_actions"]["robot_1"]
                if canceled.get("robot_2") is True and canceled.get("robot_1") is False:
                    answer = (f"机器人2选择了{LABELS[language][actor_action]}，但目标格仍被你占用；环境取消了它的移动，所以它实际留在原地。"
                              if language == "zh" else
                              f"Robot 2 chose {LABELS[language][actor_action]}, but you remained in its target cell. The environment canceled its move, so it physically stayed in place.")
                elif canceled.get("robot_1") is True and canceled.get("robot_2") is False:
                    answer = (f"机器人2选择了{LABELS[language][actor_action]}并留在原格；你选择了{LABELS[language][player_action]}进入该格，所以环境取消了你的移动。"
                              if language == "zh" else
                              f"Robot 2 chose {LABELS[language][actor_action]} and remained in its cell. You chose {LABELS[language][player_action]} into that cell, so the environment canceled your move.")
                else:
                    answer = (f"机器人2选择了{LABELS[language][actor_action]}，你选择了{LABELS[language][player_action]}；其中一方要进入另一方没有离开的格子，所以环境取消了冲突移动。"
                              if language == "zh" else
                              f"Robot 2 chose {LABELS[language][actor_action]} and you chose {LABELS[language][player_action]}. One move entered a cell the other robot did not leave, so the environment canceled the conflict.")
            else:
                answer = ("所选步骤没有记录机器人碰撞。" if language == "zh" else
                          "No robot collision is recorded for the selected step.")
            return _participant_response(answer, evidence(
                f"物理冲突类型：{collision_kind or 'none'}" if language == "zh" else
                f"Physical conflict type: {collision_kind or 'none'}"))

        if parsed["intent"] == "influence":
            if use_before:
                answer = ("你的本步动作没有影响机器人2对同一步的选择：双方先依据行动前状态分别选择，再同时执行。"
                          if language == "zh" else
                          "Your action did not affect Robot 2's choice on the same step: both chose separately from the pre-action state and then acted together.")
            else:
                answer = ("你上一项已确认的动作属于机器人2当前可见的历史，但现有证据无法单独确认它是否改变了这次选择。"
                          if language == "zh" else
                          "Your preceding confirmed action is visible in Robot 2's current history, but the evidence cannot isolate whether it changed this choice.")
            return _participant_response(answer, evidence(
                "因果边界：玩家本步命令不进入同一步策略输入。" if language == "zh" else
                "Causal boundary: the player's current command is absent from the same-step policy input."))

        if parsed["intent"] == "energy":
            charger = tuple(env.layout.charger_position)
            distance = _distance(env, teammate.position, charger)
            required = distance * env.config.move_battery_cost
            if tuple(teammate.position) == charger:
                recoverable = min(env.config.charge_per_wait,
                                  max(0.0, 100.0 - teammate.battery))
                answer = (f"机器人2当前电量为 {teammate.battery:g}%，并在充电格上；停留一步可恢复 {recoverable:g}%。"
                          if language == "zh" else
                          f"Robot 2 has {teammate.battery:g}% battery and is on the charger; staying for one turn restores {recoverable:g}%.")
            elif teammate.battery < required:
                answer = (f"机器人2当前电量为 {teammate.battery:g}%，到充电格还有 {distance} 格；仅抵达约需 {required:g}%，当前电量不足。"
                          if language == "zh" else
                          f"Robot 2 has {teammate.battery:g}% battery and is {distance} cells from the charger; reaching it needs about {required:g}%, so its current battery is insufficient.")
            else:
                answer = (f"机器人2当前电量为 {teammate.battery:g}%，到充电格还有 {distance} 格；抵达约需 {required:g}%电量。"
                          if language == "zh" else
                          f"Robot 2 has {teammate.battery:g}% battery and is {distance} cells from the charger; reaching it needs about {required:g}% battery.")
            return _participant_response(answer, evidence(
                "这里只核验公开电量与抵达成本，不推断长期计划。" if language == "zh" else
                "This checks public battery and travel cost only; it does not infer a long-term plan."))

        if parsed["intent"] == "goal":
            objectives = _objective_candidates(env)
            if teammate.carrying_task_id and objectives:
                target = _objective_label(objectives[0], language)
                answer = (f"机器人2正携带任务{objectives[0]['slot']}的货物，当前需要送到{target}。"
                          if language == "zh" else
                          f"Robot 2 is carrying task {objectives[0]['slot']}; its current task is delivery to {target}.")
            else:
                projected = _position_after_action(env, actor_action)
                progress = _best_progress(env, projected)
                if progress:
                    target = _objective_label(progress, language)
                    answer = (f"机器人2目前空载，下一步会选择{LABELS[language][actor_action]}；若移动未被阻挡，到{target}的距离会从 {progress['before']} 格变为 {progress['after']} 格。"
                              if language == "zh" else
                              f"Robot 2 is empty-handed and will choose {LABELS[language][actor_action]} next. If the move is not blocked, its distance to {target} changes from {progress['before']} to {progress['after']} cells.")
                else:
                    labels = [_objective_label(item, language) for item in objectives]
                    available = (("、".join(labels) or "无") if language == "zh"
                                 else (", ".join(labels) or "none"))
                    answer = (f"机器人2目前空载，可领取目标有{available}；现有证据无法确认它已选择其中哪一个。"
                              if language == "zh" else
                              f"Robot 2 is empty-handed; available pickup targets are {available}. The evidence does not show that it has selected one of them.")
            return _participant_response(answer, evidence(
                "任务方向只依据公开持货状态和本步可核验的几何变化。" if language == "zh" else
                "Task direction uses only public carrying state and verifiable one-step geometry."))

        alternative = parsed.get("alternative_action")
        mentioned = parsed.get("mentioned_action")
        executed_action = (frame_record.get("executed_actions", {}).get("robot_2")
                           if use_before else None)
        if alternative == actor_action or (alternative is None and mentioned
                                           and mentioned != actor_action):
            if use_before and executed_action == "WAIT" and actor_action != "WAIT":
                reason = ("机器人冲突" if collision_kind != "none" else "墙或边界")
                reason_en = ("a robot conflict" if collision_kind != "none"
                             else "a wall or boundary")
                answer = (f"机器人2选择了{LABELS[language][actor_action]}，但{reason}阻止了移动，所以实际留在原地。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]}, but {reason_en} blocked the move, so it physically stayed in place.")
            else:
                answer = (f"机器人2在所选步骤选择了{LABELS[language][actor_action]}，请确认问题中的动作。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]} on the selected step; check the action named in the question.")
            return _participant_response(answer, evidence())

        destination = (tuple(after_teammate.position) if use_before else
                       _position_after_action(env, actor_action))
        progress = _best_progress(env, destination)

        if parsed["intent"] == "alternative":
            if alternative is None:
                answer = ("请指定一个替代动作，例如“为什么没有向上”。" if language == "zh" else
                          "Name one alternative action, such as “Why not move up?”")
                return _participant_response(answer, evidence())
            physical = None
            purpose_supported = False
            if progress and use_before:
                physical = _physical_progress_counterfactual(
                    runtime=runtime, bound=bound, current=current,
                    objective=progress, chosen_action=actor_action,
                    comparison_action=alternative,
                    player_action=frame_record["submitted_actions"]["robot_1"],
                    use_before=True,
                )
                purpose_supported = _goal_purpose_supported(
                    program_matches, physical)
            if purpose_supported:
                target = _objective_label(progress, language)
                answer = (f"机器人2选择{LABELS[language][actor_action]}而没有选择{LABELS[language][alternative]}，是为了靠近{target}；实际距离从 {progress['before']} 格缩短到 {progress['after']} 格。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]} rather than {LABELS[language][alternative]} to approach {target}. The confirmed distance fell from {progress['before']} to {progress['after']} cells.")
            elif progress:
                alternative_position = _position_after_action(env, alternative)
                alternative_distance = _distance(env, alternative_position,
                                                 progress["target"])
                target = _objective_label(progress, language)
                answer = (f"机器人2选择了{LABELS[language][actor_action]}，没有选择{LABELS[language][alternative]}；这一步到{target}的距离变为 {progress['after']} 格，替代方向预计为 {alternative_distance} 格，但现有证据无法可靠说明更具体的原因。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]} rather than {LABELS[language][alternative]}. The selected move made its distance to {target} {progress['after']} cells and the alternative would make it {alternative_distance}, but the evidence cannot reliably identify a more specific reason.")
            else:
                answer = (f"机器人2选择了{LABELS[language][actor_action]}，没有选择{LABELS[language][alternative]}；现有证据无法可靠说明更具体的原因。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][actor_action]} rather than {LABELS[language][alternative]}; the available evidence cannot reliably identify a more specific reason.")
            extra = (_physical_detail(physical, language),) if physical else ()
            return _participant_response(answer, evidence(*extra))

        if use_before and executed_action == "WAIT" and actor_action != "WAIT":
            reason = ("双方发生机器人冲突" if collision_kind != "none" else
                      "目标格是墙或边界")
            reason_en = ("the robots conflicted" if collision_kind != "none" else
                         "the target cell was a wall or boundary")
            answer = (f"机器人2选择了{LABELS[language][actor_action]}，但{reason}，所以实际留在原地。"
                      if language == "zh" else
                      f"Robot 2 chose {LABELS[language][actor_action]}, but {reason_en}, so it physically stayed in place.")
            return _participant_response(answer, evidence(
                "选择动作与环境执行结果已分别核验。" if language == "zh" else
                "The chosen action and physical outcome were verified separately."))

        before_battery = float(teammate.battery)
        after_battery = float(after_teammate.battery)
        if actor_action == "WAIT":
            if use_before and after_battery > before_battery:
                answer = (f"机器人2刚才在充电格等待，电量从 {before_battery:g}% 恢复到 {after_battery:g}%。"
                          if language == "zh" else
                          f"Robot 2 waited on the charger; its battery rose from {before_battery:g}% to {after_battery:g}%.")
            else:
                answer = ("机器人2选择了等待；现有证据无法可靠说明更具体的原因。"
                          if language == "zh" else
                          "Robot 2 chose to wait; the available evidence cannot reliably identify a more specific reason.")
            return _participant_response(answer, evidence())

        event = next((item for item in frame_record.get("events", ())
                      if item.get("agent_id") == "robot_2"
                      and item.get("event") in ("pickup", "delivery")), None) \
            if use_before else None
        if event:
            slot = _task_slot(env, event.get("task_id"))
            if event["event"] == "pickup":
                answer = (f"机器人2刚才{LABELS[language][actor_action]}到达任务{slot}的A点，并领取了货物。"
                          if language == "zh" else
                          f"Robot 2 moved {LABELS[language][actor_action]} to task {slot}'s A point and collected the item.")
            else:
                answer = (f"机器人2刚才{LABELS[language][actor_action]}到达任务{slot}的B点，并完成了交付。"
                          if language == "zh" else
                          f"Robot 2 moved {LABELS[language][actor_action]} to task {slot}'s B point and completed the delivery.")
            return _participant_response(answer, evidence())

        if progress:
            target = _objective_label(progress, language)
            physical = None
            purpose_supported = False
            if use_before:
                physical = _physical_progress_counterfactual(
                    runtime=runtime, bound=bound, current=current,
                    objective=progress, chosen_action=actor_action,
                    comparison_action="WAIT",
                    player_action=frame_record["submitted_actions"]["robot_1"],
                    use_before=True,
                )
                purpose_supported = _goal_purpose_supported(
                    program_matches, physical)
            if language == "zh":
                if purpose_supported:
                    answer = (f"机器人2刚才{LABELS[language][actor_action]}，是为了靠近{target}；"
                              f"这一步把距离从 {progress['before']} 格缩短到 {progress['after']} 格。")
                elif use_before:
                    answer = (f"机器人2刚才选择了{LABELS[language][actor_action]}，这一步把它到{target}的距离从 {progress['before']} 格缩短到 {progress['after']} 格；现有证据无法可靠说明更具体的原因。")
                else:
                    answer = (f"机器人2下一步会选择{LABELS[language][actor_action]}；若移动未被阻挡，到{target}的距离会从 {progress['before']} 格变为 {progress['after']} 格。")
            else:
                if purpose_supported:
                    answer = (f"Robot 2 moved {LABELS[language][actor_action]} to approach {target}. "
                              f"The confirmed step reduced the distance from {progress['before']} to {progress['after']} cells.")
                elif use_before:
                    answer = (f"Robot 2 chose {LABELS[language][actor_action]}, and the step reduced its distance to {target} from {progress['before']} to {progress['after']} cells. "
                              "The available evidence cannot reliably identify a more specific reason.")
                else:
                    answer = (f"Robot 2 will choose {LABELS[language][actor_action]} next. If the move is not blocked, its distance to {target} changes from {progress['before']} to {progress['after']} cells.")
            extra = (_physical_detail(physical, language),) if physical else ()
            return _participant_response(answer, evidence(*extra))

        answer = (f"机器人2选择了{LABELS[language][actor_action]}；这一步没有缩短它到当前取货点或交付点的路线，现有证据无法可靠说明更具体的原因。"
                  if language == "zh" else
                  f"Robot 2 chose {LABELS[language][actor_action]}; the move did not shorten its route to a current pickup or delivery point, so the evidence cannot reliably identify a more specific reason.")
        return _participant_response(answer, evidence())


DiagnosticOnlineExplainer = R41DiagnosticOnlineAlignmentExplainerV9

__all__ = [
    "VERSION", "RCPD_VERSION", "ARTIFACT_BINDING_VERSION", "ACCESS_VERSION",
    "EXACT_ACTIONS", "QUICK_QUESTIONS", "ACCESS_SURFACES",
    "R41DiagnosticOnlineAlignmentExplainerV9", "DiagnosticOnlineExplainer",
    "explanation_sources", "make_artifact_binding",
    "validate_artifact_binding", "explanation_access",
    "require_explanation_access",
]
