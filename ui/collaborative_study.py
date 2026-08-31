"""Two-round human/AI collaborative delivery study state machine."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import random
from typing import Any, Callable, Literal, Mapping, Sequence

from env.warehouse.seed_calibration import ParallelSeedPair
from env.warehouse.contracts import STUDY_LOG_VERSION


StudyCondition = Literal["control", "explanation"]
StudyStage = Literal[
    "idle",
    "instructions",
    "task1",
    "task1_complete",
    "task2",
    "survey",
    "completed",
    "abandoned",
]


@dataclass(frozen=True)
class CollaborativeStudyConfig:
    horizon: int = 120
    seed: int = 51000
    require_instructions: bool = True
    require_survey: bool = True
    event_sink: Callable[[Mapping[str, object]], None] | None = None

    def __post_init__(self) -> None:
        if self.horizon != 120:
            raise ValueError("The preregistered collaborative rounds use 120 steps.")


@dataclass(frozen=True)
class CollaborativeStudyAssignment:
    participant_id: str
    enrollment_index: int
    block_index: int
    condition: StudyCondition
    controlled_agent: str
    target_agent: str
    form_id: int
    demo_seed: int
    task1_seed: int
    task2_seed: int
    study_phase: str
    randomization_seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RoundSummary:
    round_name: str
    seed: int
    score: float
    steps: int
    deliveries: int
    robot_collisions: int
    shutdowns: int
    human_route_regret_units: float
    mean_delivery_latency: float | None
    terminal_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CollaborativeConditionAllocator:
    """Balanced 2-condition x 4-form allocation in blocks of eight."""

    block_size = 8
    pilot_size = 24
    demo_seed = 40221
    base_form_seed = 51000

    def __init__(
        self,
        *,
        randomization_seed: int = 51000,
        study_phase: str = "pilot",
        parallel_seed_pairs: Sequence[ParallelSeedPair] | None = None,
        demo_seed: int = 40221,
    ) -> None:
        self.randomization_seed = int(randomization_seed)
        if study_phase not in {"pilot", "confirmatory"}:
            raise ValueError("study_phase must be pilot or confirmatory")
        self.study_phase = study_phase
        self.demo_seed = int(demo_seed)
        self.parallel_seed_pairs = tuple(parallel_seed_pairs or ())
        if self.parallel_seed_pairs and (
            len(self.parallel_seed_pairs) != 4
            or {item.form_id for item in self.parallel_seed_pairs} != set(range(4))
        ):
            raise ValueError("Four calibrated parallel seed forms are required.")

    def _cells_for_block(self, block_index: int) -> list[tuple[str, int]]:
        cells = [
            (condition, form_id)
            for condition in ("control", "explanation")
            for form_id in range(4)
        ]
        random.Random(self.randomization_seed + block_index).shuffle(cells)
        return cells

    def _assignment_for_index(
        self,
        participant_id: str,
        enrollment_index: int,
    ) -> CollaborativeStudyAssignment:
        block_index, offset = divmod(int(enrollment_index), self.block_size)
        condition, form_id = self._cells_for_block(block_index)[offset]
        calibrated = (
            self.parallel_seed_pairs[form_id]
            if self.parallel_seed_pairs
            else None
        )
        base = self.base_form_seed + form_id * 1_000
        return CollaborativeStudyAssignment(
            participant_id=str(participant_id),
            enrollment_index=int(enrollment_index),
            block_index=block_index,
            condition=condition,  # type: ignore[arg-type]
            controlled_agent="robot_1",
            target_agent="robot_2",
            form_id=form_id,
            demo_seed=self.demo_seed,
            task1_seed=(calibrated.task1_seed if calibrated else base),
            task2_seed=(calibrated.task2_seed if calibrated else base + 500),
            study_phase=self.study_phase,
            randomization_seed=self.randomization_seed,
        )

class CollaborativeDeliveryStudy:
    """State and audit log for demo -> task1 -> task2 -> survey.

    Treatment explanations are collected live during Task 1.  They are not a
    separate stage and therefore cannot alter the preregistered round order.
    """

    allowed_conditions = ("control", "explanation")

    def __init__(
        self,
        config: CollaborativeStudyConfig | None = None,
    ) -> None:
        self.config = config or CollaborativeStudyConfig()
        self.stage: StudyStage = "idle"
        self.participant_id = ""
        self.assignment: CollaborativeStudyAssignment | None = None
        self.condition: StudyCondition = "explanation"
        self.language = "en"
        self.round_summaries: dict[str, RoundSummary] = {}
        self.explanation_count = 0
        self.survey: dict[str, object] | None = None

    @property
    def target_agent(self) -> str:
        return self.assignment.target_agent if self.assignment else "robot_2"

    @property
    def group_code(self) -> str | None:
        if self.assignment is None:
            return None
        return "A" if self.condition == "explanation" else "B"

    @property
    def controlled_agent(self) -> str:
        return self.assignment.controlled_agent if self.assignment else "robot_1"

    def start(
        self,
        assignment: CollaborativeStudyAssignment,
        *,
        language: str,
    ) -> None:
        if self.stage not in {"idle", "completed", "abandoned"}:
            raise RuntimeError("The current study run is still active.")
        if assignment.condition not in self.allowed_conditions:
            raise ValueError("Unknown collaborative study condition.")
        if assignment.controlled_agent != "robot_1" or assignment.target_agent != "robot_2":
            raise ValueError("The study fixes robot 1 as human and robot 2 as AI.")
        self.assignment = assignment
        self.participant_id = assignment.participant_id
        self.condition = assignment.condition
        self.language = language
        self.round_summaries = {}
        self.explanation_count = 0
        self.survey = None
        self.stage = "instructions" if self.config.require_instructions else "task1"
        self._write_event(
            {
                "event": "study_started",
                "study_design": "human_ai_task1_live_explanations_task2_transfer",
                "assignment": assignment.to_dict(),
                "group_code": self.group_code,
                "explanation_available": self.condition == "explanation",
                "horizon": self.config.horizon,
                "score_formula": {
                    "delivery": 100,
                    "robot_collision": -200,
                    "shutdown": -50,
                    "step": -1,
                    "human_route_regret_unit": -2,
                },
            }
        )

    def begin_task1(self) -> None:
        if self.stage != "instructions":
            raise RuntimeError("The demonstration is not awaiting completion.")
        self.stage = "task1"
        self._write_event(
            {"event": "round_started", "round": "task1", "seed": self.assignment.task1_seed}
        )

    def record_step(self, payload: Mapping[str, object]) -> None:
        if self.stage not in {"task1", "task2"}:
            raise RuntimeError("No collaborative round is active.")
        self._write_event({"event": "collaborative_step", "round": self.stage, **dict(payload)})

    def finish_round(self, summary: RoundSummary) -> None:
        if self.stage != summary.round_name or self.stage not in {"task1", "task2"}:
            raise RuntimeError("Round summary does not match the active stage.")
        self.round_summaries[summary.round_name] = summary
        self._write_event({"event": "round_completed", **summary.to_dict()})
        if summary.round_name == "task1":
            self.stage = "task1_complete"
            self._write_event(
                {
                    "event": "task1_completion_presented",
                    "task1_score": float(summary.score),
                    "next_stage": "task2",
                    "live_explanation_count": self.explanation_count,
                }
            )
        else:
            self.stage = "survey" if self.config.require_survey else "completed"
            if self.stage == "completed":
                self._complete()

    def record_explanation(
        self,
        *,
        question: str,
        report: Mapping[str, object],
        response_seconds: float,
    ) -> None:
        if self.stage != "task1" or self.condition != "explanation":
            raise RuntimeError("Live explanations are available only to Group A during Task 1.")
        self.explanation_count += 1
        self._write_event(
            {
                "event": "live_explanation_presented",
                "exposure_index": self.explanation_count,
                "question": question,
                "round": "task1",
                "current_frame": report.get("current_frame"),
                "question_type": report.get("question_focus"),
                "target_agent": report.get("target_agent"),
                "question_seed": report.get("question_seed"),
                "question_sequence": report.get("question_sequence"),
                "selected_timeline_frame": report.get("selected_timeline_frame"),
                "decision_evidence_frame": report.get("decision_evidence_frame"),
                "trajectory_kind": report.get("trajectory_kind"),
                "trajectory_seed": report.get("trajectory_seed"),
                "agent_control": report.get("agent_control", {}),
                "anchor_frame": report.get("anchor_frame"),
                "context_frames": report.get("context_frames", ()),
                "answer_en": report.get("answer_en"),
                "answer_zh": report.get("answer_zh"),
                "structured_evidence": report.get("structured_evidence", {}),
                "fact_validation": report.get("fact_validation", {}),
                "recent_collision": report.get("recent_collision", False),
                "response_seconds": max(0.0, float(response_seconds)),
                "post_question_action": None,
            }
        )

    def begin_task2(self) -> None:
        """Acknowledge the control transition page and start the second round."""

        if self.stage != "task1_complete":
            raise RuntimeError("Task 2 can be started only after Task 1 is complete.")
        assert self.assignment is not None
        self._write_event(
            {
                "event": "task1_completion_acknowledged",
                "task1_score": float(self.round_summaries["task1"].score),
                "task2_seed": int(self.assignment.task2_seed),
            }
        )
        self.stage = "task2"
        self._write_event(
            {"event": "round_started", "round": "task2", "seed": self.assignment.task2_seed}
        )

    def submit_survey(self, payload: Mapping[str, object]) -> None:
        if self.stage != "survey":
            raise RuntimeError("The study is not waiting for the survey.")
        normalized: dict[str, object] = {}
        for name in ("coordination_understanding", "ai_predictability", "interface_clarity"):
            value = int(payload.get(name, 0))
            if value not in {1, 2, 3, 4, 5}:
                raise ValueError(f"Survey item '{name}' must be from 1 to 5.")
            normalized[name] = value
        comment = str(payload.get("comment", "")).strip()
        if len(comment) > 1000:
            raise ValueError("Survey comment must not exceed 1000 characters.")
        normalized["comment"] = comment
        self.survey = normalized
        self._write_event({"event": "survey_submitted", "survey": normalized})
        self.stage = "completed"
        self._complete()

    def _complete(self) -> None:
        task1 = self.round_summaries.get("task1")
        task2 = self.round_summaries.get("task2")
        delta = task2.score - task1.score if task1 and task2 else None
        self._write_event(
            {
                "event": "study_completed",
                "rounds": {
                    name: summary.to_dict()
                    for name, summary in self.round_summaries.items()
                },
                "score_delta": delta,
            }
        )

    @property
    def score_delta(self) -> float | None:
        task1 = self.round_summaries.get("task1")
        task2 = self.round_summaries.get("task2")
        return task2.score - task1.score if task1 and task2 else None

    def set_language(self, language: str) -> None:
        if language not in {"en", "zh-CN"}:
            raise ValueError("Language must be 'en' or 'zh-CN'.")
        self.language = language

    def abandon(self, reason: str) -> None:
        if self.stage in {"idle", "completed", "abandoned"}:
            return
        self.stage = "abandoned"
        self._write_event({"event": "study_abandoned", "reason": str(reason)})

    def checkpoint(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "participant_id": self.participant_id,
            "assignment": self.assignment,
            "condition": self.condition,
            "language": self.language,
            "round_summaries": dict(self.round_summaries),
            "explanation_count": self.explanation_count,
            "survey": dict(self.survey) if self.survey is not None else None,
        }

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        for name, value in checkpoint.items():
            setattr(self, name, value)

    def _write_event(self, payload: Mapping[str, object]) -> None:
        record = {
            "schema_version": STUDY_LOG_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "participant_id": self.participant_id,
            "condition": self.condition,
            "language": self.language,
            **dict(payload),
        }
        if self.config.event_sink is None:
            raise RuntimeError("Collaborative study events require a transactional sink.")
        self.config.event_sink(record)
