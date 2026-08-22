"""Server-facing application state for the Warehouse XAI web interface.

This module deliberately contains no HTTP or browser code.  It adapts the
existing policy, simulator, explanation engine, and human-study state machine
to JSON-compatible commands so the same backend can be hosted by any web
server later.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch

from backend.adapters.warehouse import WarehouseAdapter
from backend.artifacts import file_sha256
from backend.artifact_contracts import (
    validate_posthoc_rcpd_metadata,
    validate_reference_trajectory_manifest,
)
from backend.nlp.explanation_generator import (
    ExecutionGroundedExplanationGenerator,
)
from backend.nlp.semantic_query_planner import (
    SemanticTransformerQueryPlanner as TransformerQueryPlanner,
)
from backend.nlp.tokenizer import HuggingFaceStructuredTransformer
from .collaborative_study import (
    CollaborativeStudyAssignment as StudyAssignment,
    CollaborativeConditionAllocator as StudyConditionAllocator,
)
from .study_store import StudyConflict, StudyStore, normalize_participant_id
from .tutorial import (
    build_verified_tutorial,
    reference_event_frames,
    validate_tutorial_seed_isolated,
)
from backend.simulation.query_engine import WarehouseQueryEngine
from core.program import ExecutableProgram
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.seed_calibration import load_parallel_seed_library

from .web_session import WarehouseWebSession


class WarehouseWebApplication:
    """Heavy shared models plus bounded per-browser session state."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        transformer_model: str,
        program_path: str | Path | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
        seed: int = 2026,
        study_steps: int = 120,
        study_db: str | Path | None = None,
        study_namespace: str | None = None,
        study_randomization_seed: int = 41000,
        study_phase: str = "pilot",
        test_condition_selector: bool = False,
        maximum_sessions: int = 16,
        parallel_seed_library: str | Path | None = None,
        reference_trajectory: str | Path | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Neural policy checkpoint not found: {checkpoint_path}"
            )
        self.policy = MAPPOPolicy.load(checkpoint_path, device=device)
        self.policy_artifact_hash = file_sha256(checkpoint_path)
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.seed = int(seed)
        self.study_steps = int(study_steps)
        if self.study_steps != 120:
            raise ValueError("The collaborative study requires exactly 120 steps per round.")
        self.test_condition_selector = bool(test_condition_selector)
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        formal_eligible = bool(
            checkpoint_payload.get("training_metadata", {})
            .get("program_regularization", {})
            .get("explanation_eligible", False)
        )
        if not formal_eligible and not self.test_condition_selector:
            raise ValueError(
                "The v5 checkpoint is not explanation-eligible and cannot be used "
                "for a formal participant study."
            )
        self.study_namespace = study_namespace or (
            "development" if self.test_condition_selector else study_phase
        )
        seed_library_path = (
            Path(parallel_seed_library)
            if parallel_seed_library is not None
            else checkpoint_path.with_name("parallel_seed_pairs.json")
        )
        if not seed_library_path.exists():
            raise FileNotFoundError(
                "Calibrated parallel-seed library not found: "
                f"{seed_library_path}. Generate it with the collaborative policy."
            )
        self.parallel_seed_pairs = load_parallel_seed_library(seed_library_path)
        reference_path = (
            Path(reference_trajectory)
            if reference_trajectory is not None
            else checkpoint_path.with_name("reference_trajectory.json")
        )
        if not reference_path.is_file():
            raise FileNotFoundError(
                "Frozen v5 reference trajectory manifest not found: "
                f"{reference_path}. Calibrate it with the accepted v5 policy."
            )
        reference_manifest = json.loads(reference_path.read_text(encoding="utf-8"))
        validate_reference_trajectory_manifest(
            reference_manifest,
            model_version=self.policy.model_version,
            environment_version=WarehouseMultiAgentEnv.environment_name,
            map_layout_id=self.policy.environment_config.map_layout_id,
        )
        reference_seed = int(reference_manifest["seed"])
        validate_tutorial_seed_isolated(
            reference_seed,
            task1_seeds=tuple(item.task1_seed for item in self.parallel_seed_pairs),
            task2_seeds=tuple(item.task2_seed for item in self.parallel_seed_pairs),
        )
        study_db_path = (
            Path(study_db)
            if study_db is not None
            else checkpoint_path.with_name("collaborative_study.sqlite3")
        )
        self.study_store = StudyStore(
            study_db_path,
            namespace=self.study_namespace,
            abandon_on_start=True,
        )
        self.study_allocator = StudyConditionAllocator(
            randomization_seed=study_randomization_seed,
            study_phase=study_phase,
            parallel_seed_pairs=self.parallel_seed_pairs,
            demo_seed=reference_seed,
        )
        self._study_prewarm_lock = threading.RLock()
        self._study_prewarm_keys: set[tuple[int, ...]] = set()
        # Build the next participant's deterministic seed bank while the much
        # larger language model is loading.  By the time the browser can open,
        # study enrollment normally needs only cache lookup and selection.
        self._schedule_study_prewarm()
        self.backend = HuggingFaceStructuredTransformer(
            transformer_model,
            device=device,
            local_files_only=local_files_only,
            json_repair_attempts=0,
            max_input_tokens=2048,
        )
        self.backend.warmup()
        program_file = Path(program_path) if program_path else None
        if program_file is not None and not program_file.exists():
            raise FileNotFoundError(
                f"Collaborative RCPD program not found: {program_file}"
            )
        self.program = (
            ExecutableProgram.load_json(program_file)
            if program_file is not None and program_file.exists()
            else None
        )
        if self.program is not None:
            validate_posthoc_rcpd_metadata(self.program.metadata)
        self.program_artifact_hash = (
            file_sha256(program_file)
            if self.program is not None and program_file is not None
            else None
        )
        # The tutorial is generated by the same deterministic neural policy,
        # before the service accepts traffic. Missing semantic events abort
        # startup instead of exposing an incomplete lesson.
        self.tutorial = build_verified_tutorial(self.policy, seed=reference_seed)
        if len(self.tutorial.frames) + 1 != int(reference_manifest["frame_count"]):
            raise ValueError("Reference trajectory manifest frame count does not match policy replay.")
        if reference_event_frames(self.tutorial) != reference_manifest.get("event_frames"):
            raise ValueError("Reference trajectory events do not match the frozen manifest.")
        self.maximum_sessions = max(1, int(maximum_sessions))
        self._sessions: OrderedDict[str, tuple[float, WarehouseWebSession]] = (
            OrderedDict()
        )
        # This lock protects only the small session registry.  Session state has
        # its own lock and Transformer generations use the backend inference
        # queue, so a long explanation does not block unrelated browser state.
        self.lock = threading.RLock()

    def _schedule_study_prewarm(self) -> None:
        # Collaborative rounds are initialized in milliseconds and do not need
        # additional prewarming threads.
        return

    def _assignment_for_participant(
        self,
        participant_id: str,
        *,
        condition_override: str = "auto",
    ) -> tuple[StudyAssignment, str]:
        display, participant_key = normalize_participant_id(participant_id)
        if condition_override not in {"auto", "control", "explanation"}:
            raise ValueError("Unknown test condition override.")
        if condition_override != "auto" and not self.test_condition_selector:
            raise ValueError("Condition overrides require development test mode.")
        existing = self.study_store.participant_assignment(participant_key)
        if existing is not None and not self.test_condition_selector:
            values = dict(existing)
            values["participant_id"] = display
            assignment = StudyAssignment(**values)
            self._validate_assignment_seed_isolation(assignment)
            return assignment, participant_key
        assignments = []
        raw_assignments = (
            self.study_store.run_assignments()
            if self.test_condition_selector
            else self.study_store.assignments()
        )
        for raw in raw_assignments:
            try:
                assignments.append(StudyAssignment(**raw))
            except (TypeError, ValueError):
                continue
        phase_assignments = tuple(
            item for item in assignments if item.study_phase == self.study_allocator.study_phase
        )
        assignment = self.study_allocator._assignment_for_index(
            display,
            len(phase_assignments),
        )
        if condition_override != "auto":
            assignment = replace(assignment, condition=condition_override)
        self._validate_assignment_seed_isolation(assignment)
        return assignment, participant_key

    def _validate_assignment_seed_isolation(
        self,
        assignment: StudyAssignment,
    ) -> None:
        if len({assignment.demo_seed, assignment.task1_seed, assignment.task2_seed}) != 3:
            raise RuntimeError("Demo, task 1, and task 2 seeds must be isolated.")

    @staticmethod
    def _persisted_session(session: WarehouseWebSession) -> dict[str, Any]:
        return {
            "stage": session.human_study.stage,
            "locale": session.locale,
            "tutorial_index": session.tutorial_index,
            "tutorial_max_index": session.tutorial_max_index,
            "tutorial_complete": session.tutorial_complete,
            "survey": session.human_study.survey,
        }

    def _start_command(
        self,
        *,
        resolved_id: str,
        owner_id: str,
        session: WarehouseWebSession,
        envelope: Mapping[str, Any],
        force_restart: bool,
    ) -> dict[str, Any]:
        cached = self.study_store.cached_operation(str(envelope["operation_id"]))
        if cached is not None:
            if session.run_id == cached.get("run_id"):
                return cached
            raise StudyConflict(
                "The cached operation belongs to a run that is no longer active in this page.",
                code="cached_run_not_active",
                current={"stage": session.human_study.stage},
            )
        payload = envelope["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("Command payload must be an object.")
        allowed_payload = {
            "participant_id", "locale", "viewport_width", "condition_override",
        }
        unexpected = set(payload) - allowed_payload
        if unexpected:
            raise ValueError("Unknown start fields: " + ", ".join(sorted(unexpected)))
        try:
            viewport_width = int(payload.get("viewport_width", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("viewport_width must be an integer.") from exc
        if viewport_width < 1024:
            raise ValueError(
                "This study requires a desktop or laptop viewport at least 1024 pixels wide."
            )
        locale = str(payload.get("locale", "en"))
        if locale not in {"en", "zh-CN"}:
            raise ValueError("Language must be 'en' or 'zh-CN'.")
        requested_participant = (
            str(payload.get("participant_id", "")).strip()
            or session.human_study.participant_id
        )
        condition_override = str(payload.get("condition_override", "auto")).strip().lower()
        assignment, participant_key = self._assignment_for_participant(
            requested_participant,
            condition_override=condition_override,
        )
        checkpoint = session.checkpoint()
        new_run_id = uuid4().hex
        session.reset_study_machine()
        session.run_id = new_run_id
        session.owner_page_id = owner_id
        session.locale = locale

        def mutate(attempt: int):
            result = session.start_study(assignment=assignment, language=locale)
            result["run_id"] = new_run_id
            result["attempt"] = attempt
            return result, session.drain_events(), self._persisted_session(session)

        try:
            result = self.study_store.create_run(
                operation_id=str(envelope["operation_id"]),
                run_id=new_run_id,
                browser_session_id=owner_id,
                participant_id=assignment.participant_id,
                participant_key=participant_key,
                assignment=assignment.to_dict(),
                locale=locale,
                force_restart=force_restart,
                mutate=mutate,
            )
        except BaseException:
            session.restore_checkpoint(checkpoint)
            raise
        session.state_version = int(result["state_version"])
        return result

    def _study_command(
        self,
        *,
        resolved_id: str,
        owner_id: str,
        session: WarehouseWebSession,
        envelope: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = {
            "operation_id", "run_id", "expected_stage",
            "expected_state_version", "command", "payload",
        }
        missing = required - set(envelope)
        unexpected = set(envelope) - required
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unexpected:
                details.append("unknown " + ", ".join(sorted(unexpected)))
            raise ValueError("Invalid command envelope: " + "; ".join(details))
        operation_id = str(envelope["operation_id"]).strip()
        if not operation_id or len(operation_id) > 128:
            raise ValueError("operation_id must contain 1-128 characters.")
        command = str(envelope["command"])
        if not isinstance(envelope["payload"], Mapping):
            raise ValueError("Command payload must be an object.")
        if command == "start":
            if envelope["run_id"] not in {None, ""}:
                raise ValueError("start must not include an existing run_id.")
            if str(envelope["expected_stage"]) != "idle" or int(envelope["expected_state_version"]) != 0:
                raise ValueError("start requires idle stage and state version 0.")
            return self._start_command(
                resolved_id=resolved_id,
                owner_id=owner_id,
                session=session,
                envelope=envelope,
                force_restart=False,
            )
        if command == "restart":
            return self._start_command(
                resolved_id=resolved_id,
                owner_id=owner_id,
                session=session,
                envelope={
                    **envelope,
                    "payload": {
                        "participant_id": envelope["payload"].get(
                            "participant_id", session.human_study.participant_id
                        ),
                        "locale": envelope["payload"].get("locale", session.locale),
                        "viewport_width": envelope["payload"].get("viewport_width", 1024),
                        "condition_override": envelope["payload"].get(
                            "condition_override", "auto"
                        ),
                    },
                },
                force_restart=True,
            )
        cached = self.study_store.cached_operation(operation_id)
        if cached is not None:
            if str(cached.get("run_id", "")) != str(envelope["run_id"]):
                raise StudyConflict(
                    "operation_id was already used by a different run.",
                    code="operation_id_reused",
                    current=session.view(),
                )
            return cached
        if command not in session.allowed_commands():
            raise StudyConflict(
                f"Command '{command}' is not allowed during {session.human_study.stage}.",
                code="command_not_allowed",
                current=session.view(),
            )
        payload = dict(envelope["payload"])
        checkpoint = session.checkpoint()

        explanation_was_accepted = False

        def mutate():
            if command == "set_language":
                result = session.set_language(str(payload.get("locale", "")))
            elif command == "tutorial_advance":
                result = {"view": session.tutorial_advance()}
            elif command == "tutorial_restart":
                result = {"view": session.tutorial_restart()}
            elif command == "tutorial_select":
                result = {"view": session.select_frame(int(payload["index"]))}
            elif command == "begin_task1":
                result = session.begin_task1()
            elif command == "human_action":
                result = session.submit_human_action(
                    str(payload.get("action", ""))
                )
            elif command == "ask_explanation":
                result = session.explain_study(
                    str(payload.get("question", "")),
                    target_agent=str(payload.get("target_agent", "robot_2")),
                    accepted_before_deadline=explanation_was_accepted,
                    selected_frame=(
                        int(payload["selected_frame"])
                        if payload.get("selected_frame") is not None
                        else None
                    ),
                    trajectory_hash=(
                        str(payload["trajectory_hash"])
                        if payload.get("trajectory_hash")
                        else None
                    ),
                    question_kind=(
                        str(payload["question_kind"])
                        if payload.get("question_kind")
                        else None
                    ),
                )
            elif command == "finish_explanation":
                result = session.finish_explanation()
            elif command == "begin_task2":
                result = session.begin_task2()
            elif command == "submit_survey":
                result = session.submit_survey(payload)
            elif command == "timeline_select":
                result = {"view": session.select_frame(int(payload["index"]))}
            elif command == "timeline_back":
                result = {"view": session.back()}
            elif command == "timeline_forward":
                result = {"view": session.forward()}
            else:
                raise KeyError(f"Unknown study command: {command}")
            return result, session.drain_events(), self._persisted_session(session)

        if command == "ask_explanation":
            explanation_was_accepted = bool(
                session.human_study.stage == "explanation"
                and not session.human_study.explanation_time_expired
            )
            cached = self.study_store.reserve_long_operation(
                operation_id=operation_id,
                run_id=str(envelope["run_id"]),
                browser_session_id=owner_id,
                expected_stage=str(envelope["expected_stage"]),
                expected_version=int(envelope["expected_state_version"]),
                command=command,
            )
            if cached is not None:
                return cached
            try:
                mutation = mutate()
                result = self.study_store.complete_long_operation(
                    operation_id=operation_id,
                    run_id=str(envelope["run_id"]),
                    browser_session_id=owner_id,
                    expected_stage=str(envelope["expected_stage"]),
                    expected_version=int(envelope["expected_state_version"]),
                    command=command,
                    mutation=mutation,
                )
            except StudyConflict:
                session.restore_checkpoint(checkpoint)
                self.study_store.cancel_long_operation(operation_id)
                raise
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                session.restore_checkpoint(checkpoint)
                self.study_store.cancel_long_operation(operation_id)
                raise StudyConflict(
                    str(exc),
                    code="invalid_command",
                    current=session.view(),
                ) from exc
            except BaseException:
                session.restore_checkpoint(checkpoint)
                self.study_store.cancel_long_operation(operation_id)
                raise
            session.state_version = int(result["state_version"])
            return result

        try:
            result = self.study_store.execute(
                operation_id=operation_id,
                run_id=str(envelope["run_id"]),
                browser_session_id=owner_id,
                expected_stage=str(envelope["expected_stage"]),
                expected_version=int(envelope["expected_state_version"]),
                command=command,
                mutate=mutate,
            )
        except StudyConflict:
            session.restore_checkpoint(checkpoint)
            raise
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            session.restore_checkpoint(checkpoint)
            raise StudyConflict(
                str(exc),
                code="invalid_command",
                current=session.view(),
            ) from exc
        except BaseException:
            session.restore_checkpoint(checkpoint)
            raise
        session.state_version = int(result["state_version"])
        return result

    def _engine(
        self,
        adapter: WarehouseAdapter,
        policy: MAPPOPolicy,
    ) -> WarehouseQueryEngine:
        return WarehouseQueryEngine(
            adapter=adapter,
            policy=policy,
            planner=TransformerQueryPlanner(
                self.backend,
                verify_response_language=True,
            ),
            explanation_generator=ExecutionGroundedExplanationGenerator(
                self.backend,
                semantics=adapter,
            ),
            program=self.program,
            policy_artifact_hash=self.policy_artifact_hash,
            program_artifact_hash=self.program_artifact_hash,
        )

    def session(self, session_id: str | None = None) -> tuple[str, WarehouseWebSession]:
        with self.lock:
            if session_id and session_id in self._sessions:
                _, value = self._sessions.pop(session_id)
                self._sessions[session_id] = (time.time(), value)
                return session_id, value
            new_id = uuid4().hex
            value = WarehouseWebSession(
                policy=self.policy,
                engine_factory=self._engine,
                seed=self.seed,
                tutorial=self.tutorial,
                test_condition_selector=self.test_condition_selector,
            )
            self._sessions[new_id] = (time.time(), value)
            while len(self._sessions) > self.maximum_sessions:
                removable = next(
                    (
                        key
                        for key, (_, candidate) in self._sessions.items()
                        if candidate.human_study.stage in {"idle", "completed", "abandoned"}
                        and key != new_id
                    ),
                    None,
                )
                if removable is None:
                    break
                self._sessions.pop(removable)
            return new_id, value

    def dispatch(
        self,
        session_id: str | None,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        data = dict(payload or {})
        page_id = str(data.pop("__page_id", "") or "anonymous")
        resolved_id, session = self.session(session_id)
        owner_id = f"{resolved_id}:{page_id}"
        # Test doubles and external session implementations may be immutable;
        # production WarehouseWebSession instances own an isolated state lock.
        session_lock = getattr(session, "lock", nullcontext())
        with session_lock:
            if operation == "study_command":
                result = self._study_command(
                    resolved_id=resolved_id,
                    owner_id=owner_id,
                    session=session,
                    envelope=data,
                )
            elif operation == "view":
                result = session.view()
                if (
                    session.run_id
                    and session.human_study.stage not in {"idle", "completed", "abandoned"}
                    and session.owner_page_id not in {None, owner_id}
                ):
                    result = {**result, "study": {**result["study"]}}
                    result["study"].update(
                        {
                            "stage": "abandoned",
                            "interrupted": True,
                            "interruption_code": "active_in_another_page",
                            "allowed_commands": ["restart"],
                        }
                    )
            elif operation == "reference_trajectory":
                result = session.reference_trajectory_payload()
            elif operation == "timeline_events":
                operation_id = str(data.get("operation_id", "")).strip()
                run_id = str(data.get("run_id", "")).strip()
                if not session.run_id or run_id != session.run_id:
                    raise ValueError("Timeline events do not match the active run.")
                reference = session.reference_trajectory_payload()
                if not session.reference_trajectory_hash_is_compatible(
                    str(data.get("trajectory_hash", ""))
                ):
                    raise ValueError("Timeline events reference a stale trajectory.")
                raw_events = data.get("events", ())
                if not isinstance(raw_events, list):
                    raise ValueError("events must be a list.")
                cleaned: list[dict[str, Any]] = []
                for raw in raw_events:
                    if not isinstance(raw, Mapping):
                        raise ValueError("Each timeline event must be an object.")
                    index = int(raw.get("timeline_index", -1))
                    if index < 1 or index > session.timeline.max_index:
                        raise ValueError("Timeline event index is outside the reference trajectory.")
                    cleaned.append(
                        {
                            "timeline_index": index,
                            "environment_frame": int(
                                session.timeline.frames[index].snapshot.frame
                            ),
                            "dwell_ms": max(0, int(raw.get("dwell_ms", 0))),
                            "trajectory_kind": session.trajectory_kind,
                            "trajectory_seed": session.trajectory_seed,
                            "trajectory_hash": reference["trajectory_hash"],
                            "agent_control": dict(session.trajectory_agent_control),
                            "immutable": True,
                        }
                    )
                result = self.study_store.record_timeline_events(
                    operation_id=operation_id,
                    run_id=run_id,
                    events=cleaned,
                )
            else:
                raise KeyError(f"Unknown API operation: {operation}")
            return resolved_id, result
