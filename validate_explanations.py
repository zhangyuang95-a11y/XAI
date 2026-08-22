"""Validate typed ExplanationIR grounding on a recorded frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from backend.adapters.warehouse import WarehouseAdapter
from backend.artifacts import CollaborativeArtifactPaths, file_sha256
from core.program import ExecutableProgram
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.contracts import ARTIFACT_NAMESPACE
from backend.nlp.explanation_generator import ExecutionGroundedExplanationGenerator
from backend.nlp.semantic_query_planner import (
    SemanticTransformerQueryPlanner as TransformerQueryPlanner,
)
from backend.nlp.schemas import ClaimVerdictStatus
from backend.nlp.tokenizer import (
    CallableTransformerBackend,
    HuggingFaceStructuredTransformer,
)
from backend.simulation.query_engine import WarehouseQueryEngine
from backend.simulation.trajectory_store import TrajectoryStore


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = CollaborativeArtifactPaths.under(
    PROJECT_ROOT,
    ARTIFACT_NAMESPACE,
)
DEFAULT_ARTIFACT_ROOT = DEFAULT_ARTIFACTS.root
DEFAULT_TRAJECTORY = DEFAULT_ARTIFACTS.training_trajectory
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS.explanation_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate ExplanationIR V2 grounding on a Warehouse frame."
    )
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--program", default=None)
    parser.add_argument("--episode", default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument(
        "--question",
        default="为什么机器人2在这一帧采取了记录中的动作？",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "Response language tag (for example zh-CN, en, ja, or auto). "
            "The default lets the Transformer follow the user's language."
        ),
    )
    parser.add_argument("--transformer-model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def _smoke_backend(language: str = "zh") -> CallableTransformerBackend:
    """Explicit test double; production code never selects it implicitly."""

    del language

    def json_generator(_prompt: str, schema_name: str) -> Mapping[str, Any]:
        if schema_name == "QuestionIRV2":
            return {
                "intent": "explanatory",
                "target_entity": "robot_2",
                "referenced_entities": ["robot_2"],
                "entity_roles": [
                    {
                        "entity_id": "robot_2",
                        "roles": ["prediction_target"],
                        "source_span": "机器人2",
                    },
                ],
                "primitive_interventions": [],
                "relational_constraints": [],
                "preserved_variables": [],
                "target_variables": ["robot_2.observed_action"],
                "causal_variables": [],
                "desired_outcomes": {},
                "assumed_outcomes": {},
                "horizon": 1,
                "rollout_count": 1,
                "ambiguities": [],
                "unsupported_components": [],
            }
        if schema_name == "ExplanationDocumentV2":
            # Exercise the deterministic typed renderer in smoke mode.
            return {"sections": []}
        raise ValueError(f"Unsupported smoke schema: {schema_name}")

    return CallableTransformerBackend(json_generator)


def main() -> None:
    args = build_parser().parse_args()
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        raise SystemExit(f"Trajectory not found: {trajectory_path}")
    store = TrajectoryStore.load(trajectory_path)
    episode_id = args.episode or (store.episode_ids()[0] if store.episode_ids() else None)
    if episode_id is None:
        raise SystemExit("The trajectory contains no episodes.")
    frames = store.frames(episode_id)
    if not frames:
        raise SystemExit(f"Trajectory episode {episode_id!r} contains no frames.")
    selected = (
        store.frame(episode_id, args.frame)
        if args.frame is not None
        else frames[0]
    )

    checkpoint_value = args.checkpoint or store.metadata.get("policy_checkpoint")
    if not checkpoint_value:
        raise SystemExit(
            "--checkpoint is required when trajectory metadata has no policy checkpoint."
        )
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.exists():
        raise SystemExit(f"Policy checkpoint not found: {checkpoint}")
    policy = MAPPOPolicy.load(checkpoint, device=args.device)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=args.frame or 2026)
    adapter = WarehouseAdapter(environment)

    program_path = (
        Path(args.program)
        if args.program
        else checkpoint.with_name("rcpd_program.json")
    )
    program = ExecutableProgram.load_json(program_path) if program_path.exists() else None
    if args.smoke_test:
        backend = _smoke_backend(args.language)
        backend_kind = "injected_test_transformer_stub"
    else:
        if not args.transformer_model:
            raise SystemExit(
                "--transformer-model is required outside --smoke-test; "
                "there is no keyword or regex fallback."
            )
        backend = HuggingFaceStructuredTransformer(
            args.transformer_model,
            device=args.device,
            local_files_only=args.local_files_only,
            max_input_tokens=2048,
            json_repair_attempts=0,
        )
        backend.warmup()
        backend_kind = f"huggingface:{args.transformer_model}"

    engine = WarehouseQueryEngine(
        adapter=adapter,
        policy=policy,
        # Language identification is a separate Transformer task. Keeping it
        # independent prevents the execution-plan prompt from making an
        # English question return a Chinese answer (or vice versa).
        planner=TransformerQueryPlanner(
            backend,
            verify_response_language=True,
        ),
        explanation_generator=ExecutionGroundedExplanationGenerator(backend),
        program=program,
        policy_artifact_hash=file_sha256(checkpoint),
        program_artifact_hash=(
            file_sha256(program_path) if program is not None else None
        ),
    )
    answer = engine.answer(
        args.question,
        store.decision_snapshot(
            episode_id,
            selected.frame,
        ),
        selected_frame=selected.frame,
        language=args.language,
        snapshot_resolver=lambda frame_id: store.decision_snapshot(
            episode_id,
            frame_id,
        ),
    )
    validation_failures = [
        f"{verdict.status.value}: {verdict.claim.text}"
        for verdict in answer.verdicts
        if verdict.status != ClaimVerdictStatus.SUPPORTED
    ]
    validation_failures.extend(answer.posthoc_warnings)
    if not answer.claims:
        validation_failures.append(
            "The final explanation produced no typed IR claims."
        )
    report = {
        "status": "passed" if not validation_failures else "failed",
        "validation_failures": validation_failures,
        "backend": backend_kind,
        "generator_received_evaluation_labels": False,
        "online_claim_source": "typed_explanation_ir",
        "episode": episode_id,
        "frame": selected.frame,
        "checkpoint": str(checkpoint.resolve()),
        "program": str(program_path.resolve()) if program else None,
        "answer": answer.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"Validation saved to: {output}")
    if validation_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
