"""Paired trace-ablation records and randomized human-study statistics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from backend.simulation.query_engine import (
    EXPLANATION_MODE_NO_TRACE,
    EXPLANATION_MODE_RCPD_TRACE,
)


CONDITIONS = (EXPLANATION_MODE_NO_TRACE, EXPLANATION_MODE_RCPD_TRACE)


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    episode_id: str
    frame: int
    question: str
    language: str
    seed: int
    checkpoint_sha256: str
    program_sha256: str


def load_manifest(
    path: str | Path,
    *,
    checkpoint_sha256: str,
    program_sha256: str,
    available_frames: Mapping[str, set[int]],
) -> tuple[AblationCase, ...]:
    """Load and strictly validate a preregistered held-out case manifest."""

    cases: list[AblationCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Manifest line {line_number} is not valid JSON."
            ) from exc
        if not isinstance(item, Mapping):
            raise ValueError(f"Manifest line {line_number} must be an object.")
        case_id = str(item.get("case_id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Duplicate or empty case_id on line {line_number}.")
        if item.get("split") != "heldout":
            raise ValueError(f"Case {case_id!r} is not marked split=heldout.")
        episode_id = str(item.get("episode_id", "")).strip()
        frame = int(item.get("frame", -1))
        if frame not in available_frames.get(episode_id, set()):
            raise ValueError(
                f"Case {case_id!r} references missing frame "
                f"{episode_id!r}/{frame}."
            )
        expected_checkpoint = str(item.get("checkpoint_sha256", ""))
        expected_program = str(item.get("program_sha256", ""))
        if expected_checkpoint != checkpoint_sha256:
            raise ValueError(f"Case {case_id!r} checkpoint hash mismatch.")
        if expected_program != program_sha256:
            raise ValueError(f"Case {case_id!r} program hash mismatch.")
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Case {case_id!r} has an empty question.")
        seen.add(case_id)
        cases.append(
            AblationCase(
                case_id=case_id,
                episode_id=episode_id,
                frame=frame,
                question=question,
                language=str(item.get("language", "auto")),
                seed=int(item.get("seed", 2026)),
                checkpoint_sha256=expected_checkpoint,
                program_sha256=expected_program,
            )
        )
    if not cases:
        raise ValueError("The ablation manifest contains no cases.")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def canonical_evidence_hash(payload: Mapping[str, Any]) -> str:
    """Hash evidence after removing the randomized program treatment."""

    import hashlib

    common = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    common["program_trace"] = []
    common["disagreement"] = {}
    encoded = json.dumps(
        common,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def answer_metrics(answer: Mapping[str, Any]) -> dict[str, float | bool | None]:
    """Compute symmetric diagnostics from one serialized QueryAnswer."""

    verdicts = tuple(answer.get("raw_verdicts", ()))
    unsupported = sum(
        str(item.get("status", "")) != "SUPPORTED"
        for item in verdicts
        if isinstance(item, Mapping)
    )
    unsupported_rate = unsupported / len(verdicts) if verdicts else None
    action_verdicts = tuple(
        item
        for item in verdicts
        if isinstance(item, Mapping)
        and isinstance(item.get("claim"), Mapping)
        and str(item["claim"].get("claim_type", "")).lower() == "action"
    )
    action_correct = (
        bool(action_verdicts)
        and all(str(item.get("status")) == "SUPPORTED" for item in action_verdicts)
    )
    grounding = answer.get("generation_grounding", {})
    grounding = grounding if isinstance(grounding, Mapping) else {}
    semantic_plan = grounding.get("semantic_plan", {})
    semantic_plan = semantic_plan if isinstance(semantic_plan, Mapping) else {}
    mandatory = {
        str(value)
        for value in semantic_plan.get("mandatory_requirement_keys", ())
    }
    reason_keys = {
        str(item.get("requirement_key", ""))
        for field in ("reason_facts", "objective_facts", "program_facts")
        for item in semantic_plan.get(field, ())
        if isinstance(item, Mapping) and item.get("requirement_key")
    }
    required = mandatory & reason_keys
    covered = {str(value) for value in grounding.get("covered_requirement_keys", ())}
    coverage = len(required & covered) / len(required) if required else None
    raw_text = str(answer.get("raw_explanation", ""))
    display_text = str(answer.get("explanation", ""))
    audit = answer.get("trace_audit", {})
    audit = audit if isinstance(audit, Mapping) else {}
    query_plan = answer.get("query_plan", {})
    query_plan = query_plan if isinstance(query_plan, Mapping) else {}
    counterfactual_claims = tuple(
        item
        for item in verdicts
        if isinstance(item, Mapping)
        and isinstance(item.get("claim"), Mapping)
        and str(item["claim"].get("claim_type", "")).lower()
        in {"counterfactual", "comparison"}
    )
    counterfactual_consistent = (
        bool(counterfactual_claims)
        and all(
            str(item.get("status")) == "SUPPORTED"
            for item in counterfactual_claims
        )
        if query_plan.get("requires_scene_edit")
        else None
    )
    ir_payload = grounding.get("explanation_ir", {})
    ir_payload = ir_payload if isinstance(ir_payload, Mapping) else {}
    document = answer.get("explanation_document", {})
    document = document if isinstance(document, Mapping) else {}
    diagnostics = answer.get("generation_diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    units = tuple(
        item
        for item in ir_payload.get("units", ())
        if isinstance(item, Mapping)
    )
    used_units = {
        str(value)
        for value in document.get("used_unit_ids", ())
    }
    rationale_units = {
        str(item.get("unit_id", ""))
        for item in units
        if str(item.get("layer", "")) == "proposal_rationale"
    }
    rationale_coverage = (
        len(rationale_units & used_units) / len(rationale_units)
        if rationale_units
        else 0.0
    )
    coordination_expected = any(
        str(item.get("predicate", "")) == "coordination_resolution"
        for item in units
    )
    return {
        "unsupported_claim_rate": unsupported_rate,
        "action_statement_correct": action_correct,
        "required_reason_coverage": coverage,
        "counterfactual_consistent": counterfactual_consistent,
        "proposal_rationale_coverage": rationale_coverage,
        "arbitration_correct": (
            _predicate_supported(verdicts, "coordination_resolution")
            if coordination_expected
            else None
        ),
        "final_action_correct": _predicate_supported(
            verdicts,
            "final_action",
        ),
        "fallback_used": str(audit.get("fallback_status", ""))
        not in {
            "all_claims_supported",
            "all_units_valid",
            "deterministic_conversational",
        },
        "trace_eligible": bool(audit.get("eligible", False)),
        "trace_exposed": bool(audit.get("exposed", False)),
        "raw_character_count": float(len(raw_text)),
        "display_character_count": float(len(display_text)),
        "display_section_count": float(
            len(document.get("sentences", document.get("sections", ())))
        ),
        "latency_ms": _as_float(answer.get("latency_ms")),
        "model_call_count": _as_float(
            diagnostics.get("model_call_count")
        ),
        "realization_model_calls": _as_float(
            diagnostics.get("realization_model_calls")
        ),
        "question_cache_hit": float(
            bool(diagnostics.get("question_cache_hit", False))
        ),
        "ir_cache_hit": float(
            bool(diagnostics.get("ir_cache_hit", False))
        ),
        "document_cache_hit": float(
            bool(diagnostics.get("document_cache_hit", False))
        ),
    }


def paired_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_rounds: int = 10_000,
    permutation_rounds: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Summarize all cases and the preregistered trace-eligible subset."""

    prepared: list[dict[str, Any]] = []
    for record in records:
        conditions = record.get("conditions", {})
        if not isinstance(conditions, Mapping) or any(
            name not in conditions for name in CONDITIONS
        ):
            raise ValueError("Every result must contain both conditions.")
        prepared.append(
            {
                "case_id": str(record.get("case_id", "")),
                **{
                    name: answer_metrics(conditions[name])
                    for name in CONDITIONS
                },
            }
        )
    eligible = [
        item
        for item in prepared
        if item[EXPLANATION_MODE_RCPD_TRACE]["trace_eligible"]
    ]
    return {
        "case_count": len(prepared),
        "trace_eligible_count": len(eligible),
        "trace_coverage": len(eligible) / len(prepared) if prepared else 0.0,
        "all_cases": _metric_summary(
            prepared,
            bootstrap_rounds=bootstrap_rounds,
            permutation_rounds=permutation_rounds,
            seed=seed,
        ),
        "trace_eligible_cases": _metric_summary(
            eligible,
            bootstrap_rounds=bootstrap_rounds,
            permutation_rounds=permutation_rounds,
            seed=seed + 1,
        ),
    }


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_rounds: int,
    permutation_rounds: int,
    seed: int,
) -> dict[str, Any]:
    keys = (
        "unsupported_claim_rate",
        "action_statement_correct",
        "required_reason_coverage",
        "counterfactual_consistent",
        "proposal_rationale_coverage",
        "arbitration_correct",
        "final_action_correct",
        "fallback_used",
        "raw_character_count",
        "display_character_count",
        "display_section_count",
        "latency_ms",
        "model_call_count",
        "question_cache_hit",
        "ir_cache_hit",
        "document_cache_hit",
    )
    result: dict[str, Any] = {}
    for offset, key in enumerate(keys):
        pairs = [
            (
                _as_float(row[EXPLANATION_MODE_NO_TRACE].get(key)),
                _as_float(row[EXPLANATION_MODE_RCPD_TRACE].get(key)),
            )
            for row in rows
        ]
        pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
        if not pairs:
            result[key] = {"paired_case_count": 0}
            continue
        no_trace = np.asarray([item[0] for item in pairs], dtype=np.float64)
        trace = np.asarray([item[1] for item in pairs], dtype=np.float64)
        differences = trace - no_trace
        rng = np.random.default_rng(seed + offset)
        result[key] = {
            "paired_case_count": len(pairs),
            "no_trace_mean": float(no_trace.mean()),
            "rcpd_trace_mean": float(trace.mean()),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": _bootstrap_ci(
                differences,
                rounds=bootstrap_rounds,
                rng=rng,
            ),
            "paired_permutation_p": _paired_permutation_p(
                differences,
                rounds=permutation_rounds,
                rng=rng,
            ),
        }
        if key == "latency_ms":
            result[key].update(
                {
                    "no_trace_p50": float(np.percentile(no_trace, 50)),
                    "no_trace_p95": float(np.percentile(no_trace, 95)),
                    "rcpd_trace_p50": float(np.percentile(trace, 50)),
                    "rcpd_trace_p95": float(np.percentile(trace, 95)),
                }
            )
    return result


def _predicate_supported(
    verdicts: Sequence[Any],
    predicate: str,
) -> bool:
    matching = [
        item
        for item in verdicts
        if isinstance(item, Mapping)
        and isinstance(item.get("claim"), Mapping)
        and str(item["claim"].get("predicate", "")) == predicate
    ]
    return bool(matching) and all(
        str(item.get("status", "")) == "SUPPORTED"
        for item in matching
    )


def analyze_study_log(
    path: str | Path,
    *,
    study_phase: str,
    bootstrap_rounds: int = 10_000,
    permutation_rounds: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compute the preregistered participant-level ITT analysis."""

    parsed_events: list[Mapping[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            parsed_events.append(event)
    return _analyze_collaborative_study(
        parsed_events,
        study_phase=study_phase,
        bootstrap_rounds=bootstrap_rounds,
        permutation_rounds=permutation_rounds,
        seed=seed,
    )

def _analyze_collaborative_study(
    events: Sequence[Mapping[str, Any]],
    *,
    study_phase: str,
    bootstrap_rounds: int,
    permutation_rounds: int,
    seed: int,
) -> dict[str, Any]:
    participants: dict[str, dict[str, Any]] = {}
    for event in events:
        participant = str(event.get("participant_id", "")).strip()
        if not participant:
            continue
        row = participants.setdefault(
            participant,
            {"rounds": {}, "explanation_count": 0},
        )
        kind = str(event.get("event", ""))
        if kind == "study_started":
            assignment = event.get("assignment", {})
            if not isinstance(assignment, Mapping):
                continue
            row.update(
                {
                    "study_phase": str(assignment.get("study_phase", "pilot")),
                    "condition": str(assignment.get("condition", "")),
                    "block_index": int(assignment.get("block_index", -1)),
                    "form_id": int(assignment.get("form_id", -1)),
                }
            )
        elif kind == "round_completed":
            round_name = str(event.get("round_name", ""))
            if round_name in {"task1", "task2"}:
                row["rounds"][round_name] = dict(event)
        elif kind == "explanation_presented":
            row["explanation_count"] = int(row.get("explanation_count", 0)) + 1
            row["explanation_seconds"] = float(
                row.get("explanation_seconds", 0.0)
            ) + float(event.get("response_seconds", 0.0))
        elif kind == "explanation_exploration_completed":
            row["explanation_duration_seconds"] = float(
                event.get("duration_seconds", 0.0)
            )
        elif kind == "study_completed":
            row["completed"] = True

    enrolled = [
        row
        for row in participants.values()
        if row.get("study_phase") == study_phase
    ]
    completed: list[dict[str, Any]] = []
    for row in enrolled:
        rounds = row.get("rounds", {})
        if not row.get("completed") or not all(
            name in rounds for name in ("task1", "task2")
        ):
            continue
        task1 = rounds["task1"]
        task2 = rounds["task2"]
        row["delta"] = float(task2["score"]) - float(task1["score"])
        completed.append(row)
    control = [row for row in completed if row.get("condition") == "control"]
    explanation = [
        row for row in completed if row.get("condition") == "explanation"
    ]
    enrolled_control = [row for row in enrolled if row.get("condition") == "control"]
    enrolled_explanation = [
        row for row in enrolled if row.get("condition") == "explanation"
    ]
    rng = np.random.default_rng(seed)
    effect: float | None = None
    ci: tuple[float, float] | None = None
    p_value: float | None = None
    if control and explanation:
        effect = float(
            mean(float(row["delta"]) for row in explanation)
            - mean(float(row["delta"]) for row in control)
        )
        samples: list[float] = []
        left = np.asarray([float(row["delta"]) for row in control])
        right = np.asarray([float(row["delta"]) for row in explanation])
        for _ in range(int(bootstrap_rounds)):
            samples.append(
                float(
                    np.mean(rng.choice(right, size=len(right), replace=True))
                    - np.mean(rng.choice(left, size=len(left), replace=True))
                )
            )
        ci = (
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )
        observed = abs(effect)
        extreme = 0
        for _ in range(int(permutation_rounds)):
            permuted: list[dict[str, Any]] = []
            for block in sorted({int(row["block_index"]) for row in completed}):
                block_rows = [
                    row for row in completed if int(row["block_index"]) == block
                ]
                labels = [str(row["condition"]) for row in block_rows]
                rng.shuffle(labels)
                permuted.extend(
                    {**row, "condition": label}
                    for row, label in zip(block_rows, labels)
                )
            p_control = [
                float(row["delta"])
                for row in permuted
                if row["condition"] == "control"
            ]
            p_explanation = [
                float(row["delta"])
                for row in permuted
                if row["condition"] == "explanation"
            ]
            if p_control and p_explanation and abs(
                mean(p_explanation) - mean(p_control)
            ) >= observed:
                extreme += 1
        p_value = (extreme + 1) / (int(permutation_rounds) + 1)

    def metric(
        condition_rows: Sequence[Mapping[str, Any]],
        round_name: str,
        name: str,
    ) -> float | None:
        values = [
            float(row["rounds"][round_name][name])
            for row in condition_rows
            if row["rounds"][round_name].get(name) is not None
        ]
        return mean(values) if values else None

    secondary_names = (
        "deliveries",
        "robot_collisions",
        "shutdowns",
        "human_route_regret_units",
        "mean_" + "deliv" + "ery_latency",
    )

    def secondary_delta(
        condition_rows: Sequence[Mapping[str, Any]],
        name: str,
    ) -> float | None:
        values = [
            float(row["rounds"]["task2"][name])
            - float(row["rounds"]["task1"][name])
            for row in condition_rows
            if row["rounds"]["task1"].get(name) is not None
            and row["rounds"]["task2"].get(name) is not None
        ]
        return mean(values) if values else None

    task1_control_scores = [
        float(row["rounds"]["task1"]["score"]) for row in control
    ]
    task1_explanation_scores = [
        float(row["rounds"]["task1"]["score"]) for row in explanation
    ]
    all_round_scores = [
        float(row["rounds"][round_name]["score"])
        for row in completed
        for round_name in ("task1", "task2")
    ]
    shutdown_rounds = [
        float(row["rounds"][round_name].get("shutdowns", 0.0)) > 0
        for row in completed
        for round_name in ("task1", "task2")
    ]

    return {
        "study_design": "collaborative_task1_task2",
        "study_phase": study_phase,
        "enrolled": len(enrolled),
        "completed": len(completed),
        "attrition": len(enrolled) - len(completed),
        "condition_counts": {
            "control": len(control),
            "explanation": len(explanation),
        },
        "enrolled_by_condition": {
            "control": len(enrolled_control),
            "explanation": len(enrolled_explanation),
        },
        "attrition_by_condition": {
            "control": len(enrolled_control) - len(control),
            "explanation": len(enrolled_explanation) - len(explanation),
        },
        "mean_score_delta": {
            "control": mean(float(row["delta"]) for row in control) if control else None,
            "explanation": mean(float(row["delta"]) for row in explanation) if explanation else None,
        },
        "itt_effect": effect,
        "bootstrap_95_ci": ci,
        "block_permutation_p": p_value,
        "secondary_task2": {
            condition: {
                name: metric(rows, "task2", name)
                for name in secondary_names
            }
            for condition, rows in (
                ("control", control),
                ("explanation", explanation),
            )
        },
        "secondary_by_round": {
            condition: {
                round_name: {
                    name: metric(rows, round_name, name)
                    for name in secondary_names
                }
                for round_name in ("task1", "task2")
            }
            for condition, rows in (
                ("control", control),
                ("explanation", explanation),
            )
        },
        "secondary_mean_change": {
            condition: {
                name: secondary_delta(rows, name)
                for name in secondary_names
            }
            for condition, rows in (
                ("control", control),
                ("explanation", explanation),
            )
        },
        "explanation_uptake": (
            mean(int(row.get("explanation_count", 0)) > 0 for row in explanation)
            if explanation
            else None
        ),
        "mean_explanation_count": (
            mean(int(row.get("explanation_count", 0)) for row in explanation)
            if explanation
            else None
        ),
        "mean_explanation_duration_seconds": (
            mean(
                float(row.get("explanation_duration_seconds", 0.0))
                for row in explanation
            )
            if explanation
            else None
        ),
        "pilot_checks": {
            "task1_mean_score": {
                "control": (
                    mean(task1_control_scores) if task1_control_scores else None
                ),
                "explanation": (
                    mean(task1_explanation_scores)
                    if task1_explanation_scores
                    else None
                ),
            },
            "task1_baseline_score_difference": (
                mean(task1_explanation_scores) - mean(task1_control_scores)
                if task1_control_scores and task1_explanation_scores
                else None
            ),
            "observed_score_min": min(all_round_scores) if all_round_scores else None,
            "observed_score_max": max(all_round_scores) if all_round_scores else None,
            "shutdown_round_rate": (
                mean(shutdown_rounds) if shutdown_rounds else None
            ),
            "explanation_use_rate": (
                mean(
                    int(row.get("explanation_count", 0)) > 0
                    for row in explanation
                )
                if explanation
                else None
            ),
        },
        "pilot_target": 24 if study_phase == "pilot" else None,
        "pilot_target_reached": len(enrolled) == 24 if study_phase == "pilot" else None,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _bootstrap_ci(
    values: np.ndarray,
    *,
    rounds: int,
    rng: np.random.Generator,
) -> list[float]:
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    draws = rng.choice(values, size=(max(1, rounds), len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def _paired_permutation_p(
    differences: np.ndarray,
    *,
    rounds: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))
    signs = rng.choice((-1.0, 1.0), size=(max(1, rounds), len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))
