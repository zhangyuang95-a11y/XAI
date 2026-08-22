"""Generate and analyze the preregistered RCPD trace ablation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

from backend.adapters.warehouse import WarehouseAdapter
from backend.artifacts import CollaborativeArtifactPaths, file_sha256
from backend.nlp.explanation_generator import ExecutionGroundedExplanationGenerator
from backend.nlp.semantic_query_planner import (
    SemanticTransformerQueryPlanner as TransformerQueryPlanner,
)
from backend.nlp.tokenizer import HuggingFaceStructuredTransformer
from backend.simulation.query_engine import (
    EXPLANATION_MODE_NO_TRACE,
    EXPLANATION_MODE_RCPD_TRACE,
    WarehouseQueryEngine,
)
from backend.simulation.trajectory_store import TrajectoryStore
from core.program import ExecutableProgram
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.contracts import ARTIFACT_NAMESPACE
from evaluation.trace_ablation import (
    CONDITIONS,
    analyze_study_log,
    answer_metrics,
    canonical_evidence_hash,
    load_manifest,
    paired_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = CollaborativeArtifactPaths.under(
    PROJECT_ROOT,
    ARTIFACT_NAMESPACE,
)
DEFAULT_ARTIFACT_ROOT = DEFAULT_ARTIFACTS.root


@dataclass(frozen=True)
class RuntimeArtifacts:
    checkpoint: Path
    program: Path
    checkpoint_sha256: str
    program_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired offline and human-study analysis for the RCPD trace ablation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Run both explanation conditions on every preregistered held-out case.",
    )
    generate.add_argument(
        "--trajectory",
        default=str(DEFAULT_ARTIFACTS.training_trajectory),
    )
    generate.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest with preregistered held-out cases and artifact hashes.",
    )
    generate.add_argument(
        "--checkpoint",
        default=str(DEFAULT_ARTIFACTS.model),
    )
    generate.add_argument(
        "--program",
        default=str(DEFAULT_ARTIFACTS.rcpd_program),
    )
    generate.add_argument("--transformer-model", required=True)
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--local-files-only", action="store_true")
    generate.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACTS.trace_ablation_results),
    )

    summarize = subparsers.add_parser(
        "summarize",
        help="Compute paired automatic secondary outcomes.",
    )
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", required=True)
    _add_resampling_arguments(summarize)

    study = subparsers.add_parser(
        "analyze-study",
        help="Compute the preregistered participant-level ITT analysis.",
    )
    study.add_argument("--input", required=True)
    study.add_argument("--output", required=True)
    study.add_argument(
        "--study-phase",
        choices=("pilot", "confirmatory"),
        required=True,
        help="Pilot and confirmatory participants are never pooled.",
    )
    _add_resampling_arguments(study)
    return parser


def _add_resampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    parser.add_argument("--permutation-rounds", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=2026)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        generate_records(args)
    elif args.command == "summarize":
        records = _read_jsonl(Path(args.input))
        result = paired_summary(
            records,
            bootstrap_rounds=args.bootstrap_rounds,
            permutation_rounds=args.permutation_rounds,
            seed=args.analysis_seed,
        )
        _write_json(Path(args.output), result)
    elif args.command == "analyze-study":
        result = analyze_study_log(
            args.input,
            study_phase=args.study_phase,
            bootstrap_rounds=args.bootstrap_rounds,
            permutation_rounds=args.permutation_rounds,
            seed=args.analysis_seed,
        )
        _write_json(Path(args.output), result)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


def generate_records(args: argparse.Namespace) -> None:
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.is_file():
        raise SystemExit(f"Trajectory not found: {trajectory_path}")
    artifacts = _resolve_artifacts(Path(args.checkpoint), Path(args.program))
    store = TrajectoryStore.load(trajectory_path)
    frames = {
        episode_id: {frame.frame for frame in store.frames(episode_id)}
        for episode_id in store.episode_ids()
    }
    cases = load_manifest(
        args.manifest,
        checkpoint_sha256=artifacts.checkpoint_sha256,
        program_sha256=artifacts.program_sha256,
        available_frames=frames,
    )
    engine = _build_engine(args, artifacts)
    output = Path(args.output)
    existing = _resume_records(output, artifacts)

    for case in cases:
        if case.case_id in existing:
            continue
        snapshot = store.decision_snapshot(case.episode_id, case.frame)
        # Parsing is deliberately outside the condition loop: one case has
        # exactly one semantic QueryPlan in both arms.
        plan = engine.planner.parse(
            case.question,
            selected_frame=case.frame,
            environment_schema={
                "observations": dict(engine.adapter.observation_schema()),
                "actions": list(engine.adapter.action_schema()),
                "entities": dict(engine.adapter.entity_schema()),
                **dict(
                    engine.adapter.question_vocabulary()
                    if hasattr(engine.adapter, "question_vocabulary")
                    else {}
                ),
                "focus_entity": (
                    engine.adapter.default_target_entity(snapshot)
                    if hasattr(engine.adapter, "default_target_entity")
                    else None
                ),
            },
            cache_context=(
                engine.question_cache_context(snapshot)
                if hasattr(engine, "question_cache_context")
                else {}
            ),
        )
        if getattr(plan, "clarification_required", False):
            raise RuntimeError(
                f"Case {case.case_id!r} requires clarification: "
                f"{plan.clarification_reason}"
            )
        parse_diagnostics = dict(
            getattr(engine.planner, "last_diagnostics", {})
        )
        order = _condition_order(case.case_id)
        conditions: dict[str, Any] = {}
        common_hash: str | None = None
        for condition in order:
            engine.adapter.restore(snapshot, engine.policy)
            started = time.perf_counter()
            answer = engine.execute_plan(
                plan,
                snapshot,
                language=(
                    plan.response_language
                    if case.language.lower() in {"auto", "und"}
                    else case.language
                ),
                seed=case.seed,
                snapshot_resolver=lambda frame_id, episode=case.episode_id: (
                    store.decision_snapshot(episode, frame_id)
                ),
                explanation_mode=condition,
            )
            latency_ms = (time.perf_counter() - started) * 1_000.0
            payload = answer.to_dict()
            evidence_hash = canonical_evidence_hash(payload["evidence"])
            if common_hash is None:
                common_hash = evidence_hash
            elif evidence_hash != common_hash:
                raise RuntimeError(
                    f"Case {case.case_id!r} produced different common evidence "
                    "between randomized conditions."
                )
            conditions[condition] = {
                **payload,
                "latency_ms": latency_ms,
                "metrics": answer_metrics(payload),
            }

        semantic_diff = _semantic_plan_diff(
            conditions[EXPLANATION_MODE_NO_TRACE]["generation_grounding"].get(
                "semantic_plan", {}
            ),
            conditions[EXPLANATION_MODE_RCPD_TRACE]["generation_grounding"].get(
                "semantic_plan", {}
            ),
        )
        eligible = bool(
            conditions[EXPLANATION_MODE_RCPD_TRACE]["trace_audit"].get(
                "eligible", False
            )
        )
        if conditions[EXPLANATION_MODE_NO_TRACE]["trace_audit"].get(
            "exposed", False
        ):
            raise RuntimeError(
                f"No-trace case {case.case_id!r} exposed a program path."
            )
        if eligible and not conditions[EXPLANATION_MODE_RCPD_TRACE][
            "trace_audit"
        ].get("exposed", False):
            raise RuntimeError(
                f"Eligible case {case.case_id!r} failed to expose its RCPD path."
            )
        _validate_semantic_diff(case.case_id, semantic_diff, eligible=eligible)
        existing[case.case_id] = {
            "case_id": case.case_id,
            "episode_id": case.episode_id,
            "frame": case.frame,
            "question": case.question,
            "language": case.language,
            "seed": case.seed,
            "split": "heldout",
            "checkpoint_sha256": artifacts.checkpoint_sha256,
            "program_sha256": artifacts.program_sha256,
            "query_plan": plan.to_dict(),
            "question_ir": (
                engine.planner.last_question_ir.to_dict()
                if getattr(engine.planner, "last_question_ir", None) is not None
                else None
            ),
            "parse_diagnostics": parse_diagnostics,
            "common_evidence_sha256": common_hash,
            "condition_order": list(order),
            "semantic_plan_diff": list(semantic_diff),
            "conditions": conditions,
        }
        _write_jsonl_sorted(output, existing)


def _resolve_artifacts(checkpoint: Path, program: Path) -> RuntimeArtifacts:
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not program.is_file():
        raise SystemExit(f"RCPD program not found: {program}")
    return RuntimeArtifacts(
        checkpoint=checkpoint,
        program=program,
        checkpoint_sha256=file_sha256(checkpoint),
        program_sha256=file_sha256(program),
    )


def _build_engine(
    args: argparse.Namespace,
    artifacts: RuntimeArtifacts,
) -> WarehouseQueryEngine:
    policy = MAPPOPolicy.load(artifacts.checkpoint, device=args.device)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=2026)
    adapter = WarehouseAdapter(environment)
    program = ExecutableProgram.load_json(artifacts.program)
    backend = HuggingFaceStructuredTransformer(
        args.transformer_model,
        device=args.device,
        local_files_only=args.local_files_only,
        json_repair_attempts=0,
        max_input_tokens=2048,
    )
    backend.warmup()
    return WarehouseQueryEngine(
        adapter=adapter,
        policy=policy,
        planner=TransformerQueryPlanner(backend, verify_response_language=True),
        explanation_generator=ExecutionGroundedExplanationGenerator(
            backend,
            semantics=adapter,
        ),
        program=program,
        policy_artifact_hash=artifacts.checkpoint_sha256,
        program_artifact_hash=artifacts.program_sha256,
    )


def _condition_order(case_id: str) -> tuple[str, str]:
    import hashlib

    parity = hashlib.sha256(case_id.encode("utf-8")).digest()[0] & 1
    return CONDITIONS if parity == 0 else tuple(reversed(CONDITIONS))


def _semantic_plan_diff(left: Any, right: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right), key=str)
        paths: list[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_semantic_plan_diff(left[key], right[key], path))
        return tuple(paths)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if list(left) == list(right):
            return ()
        # Derived requirements and program records are atomic treatment fields;
        # indexing their internals makes acceptance errors less readable.
        return (prefix,)
    return () if left == right else (prefix,)


_TRACE_DERIVED_PLAN_FIELDS = frozenset(
    {
        "program_facts",
        "explanation_method",
        "available_evidence_ids",
        "mandatory_evidence_ids",
        "mandatory_requirement_keys",
    }
)


def _validate_semantic_diff(
    case_id: str,
    diff: Sequence[str],
    *,
    eligible: bool,
) -> None:
    unexpected = [path for path in diff if path not in _TRACE_DERIVED_PLAN_FIELDS]
    if unexpected:
        raise RuntimeError(
            f"Case {case_id!r} changed non-trace semantic-plan fields: {unexpected}"
        )
    if not eligible and diff:
        raise RuntimeError(
            f"Trace-ineligible case {case_id!r} changed its semantic plan: {list(diff)}"
        )


def _resume_records(
    path: Path,
    artifacts: RuntimeArtifacts,
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path) if path.exists() else []
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in result:
            raise ValueError("Existing output contains duplicate or empty case_id.")
        if row.get("checkpoint_sha256") != artifacts.checkpoint_sha256:
            raise ValueError("Existing output checkpoint hash does not match this run.")
        if row.get("program_sha256") != artifacts.program_sha256:
            raise ValueError("Existing output program hash does not match this run.")
        result[case_id] = row
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object.")
        rows.append(value)
    return rows


def _write_jsonl_sorted(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(rows[case_id], ensure_ascii=False, sort_keys=True, default=str)
        + "\n"
        for case_id in sorted(rows)
    )
    _atomic_write(path, content)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


if __name__ == "__main__":
    main()
