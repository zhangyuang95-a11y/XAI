"""
validate_explanations.py -- Evaluate explanation-definition validity in the RL maze environment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .agent import HeuristicAgent, RLAgent
from .environment import MazeEnvironment
from .evidence_recorder import EvidenceRecorder, EvidenceRecord
from .explanation_engine import (
    ExplanationEngine,
    SYMBOLIC_SUPPORT_VALIDATION_KEY,
)
from .question_parser import ParsedQuestion, QuestionIntent, QuestionParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate explanation definitions inside the RL Pac-Man environment."
    )
    parser.add_argument("--agent", choices=["auto", "heuristic", "rl"], default="auto")
    parser.add_argument("--model-path", default="models/dqn_pacman.pt")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--num-monsters", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/explanation_validation.json")
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument(
        "--no-symbolic-policy",
        action="store_true",
        help="Disable symbolic policy loading and validate evidence-only explanations.",
    )
    parser.add_argument(
        "--require-symbolic-policy",
        action="store_true",
        help="Fail if an RL agent has no compatible symbolic policy artifact.",
    )
    return parser


def symbolic_artifact_for_model(model_path: Path) -> Path:
    if model_path.name == "dqn_pacman.pt":
        return Path("models/dqn_pacman_symbolic.joblib")
    return model_path.with_name(f"{model_path.stem}_symbolic.joblib")


def create_agent(agent_kind: str, model_path: Path):
    if agent_kind == "heuristic":
        return HeuristicAgent(danger_radius=3, danger_penalty=80.0)
    if agent_kind == "rl":
        if not model_path.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {model_path}")
        return RLAgent(model_path=model_path)
    if model_path.exists():
        return RLAgent(model_path=model_path)
    return HeuristicAgent(danger_radius=3, danger_penalty=80.0)


def load_symbolic_policy_for_agent(agent, model_path: Path, *, disabled: bool, required: bool):
    if disabled:
        print("[xai  ] symbolic policy disabled; validating evidence-only explanations")
        return None

    if not isinstance(agent, RLAgent):
        return None

    symbolic_path = symbolic_artifact_for_model(model_path)
    if not symbolic_path.exists():
        message = f"Symbolic policy artifact not found: {symbolic_path}. Validating evidence-only explanations."
        if required:
            raise FileNotFoundError(message)
        print(f"[xai  ] {message}")
        return None

    try:
        from .symbolic_policy_adapter import load_symbolic_policy

        symbolic_policy = load_symbolic_policy(symbolic_path)
        symbolic_policy.validate_compatibility(agent.metadata)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        message = f"Symbolic policy unavailable ({exc}). Validating evidence-only explanations."
        if required:
            raise RuntimeError(message) from exc
        print(f"[xai  ] {message}")
        return None
    return symbolic_policy


def build_questions(record: EvidenceRecord) -> list[str]:
    questions = [
        "Why did you choose this action?",
        "Is it safe here?",
        "What is happening now?",
        "What is your overall policy?",
    ]

    if record.nearest_monster_id >= 0:
        questions.append(f"Did monster #{record.nearest_monster_id} influence this decision?")

    if record.dots_remaining > 0:
        questions.append("Why collect that dot?")
        questions.append("When does the exit open?")
    else:
        questions.append("How far is the exit?")

    alternatives = [action for action in record.available_actions if action != record.chosen_action]
    if alternatives:
        action_names = {
            "UP": "up",
            "DOWN": "down",
            "LEFT": "left",
            "RIGHT": "right",
            "STAY": "stay",
        }
        alt = alternatives[0]
        questions.append(f"Why not go {action_names.get(alt, alt.lower())}?")

    return questions


def summarize_validation(validation: dict[str, bool]) -> dict[str, int]:
    return {key: int(bool(value)) for key, value in validation.items()}


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path)
    agent = create_agent(args.agent, model_path)
    symbolic_policy = load_symbolic_policy_for_agent(
        agent,
        model_path,
        disabled=args.no_symbolic_policy,
        required=args.require_symbolic_policy,
    )

    grid_size = args.grid_size
    num_monsters = args.num_monsters
    if isinstance(agent, RLAgent):
        trained_grid = agent.metadata.get("grid_size")
        trained_monsters = agent.metadata.get("num_monsters")
        if trained_grid:
            grid_size = int(trained_grid)
        if trained_monsters:
            num_monsters = int(trained_monsters)

    env = MazeEnvironment(
        grid_size=grid_size,
        num_monsters=num_monsters,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    recorder = EvidenceRecorder(max_history=max(64, args.max_steps))
    parser = QuestionParser(semantic=True)
    engine = ExplanationEngine(symbolic_policy=symbolic_policy)

    totals = Counter()
    by_intent: dict[str, Counter] = defaultdict(Counter)
    failure_examples: list[dict] = []
    episode_summaries: list[dict] = []

    for episode in range(1, args.episodes + 1):
        env.reset(seed=args.seed + episode)
        recorder.clear()
        done = False
        state = env.get_state()
        episode_reward = 0.0
        step_samples = 0
        episode_checks = 0
        episode_explain_true = 0

        while not done:
            current_state = state
            action = agent.choose_action(current_state)
            _, reward, done, info = env.step_rl(action)
            state = info["state"]
            record = recorder.record(current_state, agent, action)
            episode_reward += reward

            should_sample = (
                record.step == 1
                or done
                or record.step % max(1, args.sample_every) == 0
            )
            if not should_sample:
                continue

            step_samples += 1
            for question_text in build_questions(record):
                parsed = parser.parse(question_text)
                result = engine.generate_explanation(record, parsed)
                validation = result["validation"]
                intent_key = parsed.intent.value
                explain_true = bool(validation.get("Explain_u(Q, t, x)", False))

                totals["questions"] += 1
                totals["explain_true"] += int(explain_true)
                totals["symbolic_match"] += int(bool(result.get("symbolic_match")))
                totals["symbolic_support"] += int(bool(validation.get(SYMBOLIC_SUPPORT_VALIDATION_KEY, False)))
                totals["fallback_used"] += int(bool(result.get("symbolic_rule", {}).get("fallback_used", False)))
                episode_checks += 1
                episode_explain_true += int(explain_true)

                for key, value in summarize_validation(validation).items():
                    totals[key] += value
                    by_intent[intent_key][key] += value
                by_intent[intent_key]["questions"] += 1
                by_intent[intent_key]["explain_true"] += int(explain_true)

                if not explain_true and len(failure_examples) < args.max_failures:
                    failure_examples.append(
                        {
                            "episode": episode,
                            "step": record.step,
                            "question": question_text,
                            "intent": intent_key,
                            "chosen_action": record.chosen_action,
                            "validation": validation,
                            "evidence_used": result["evidence_used"]["factors"],
                            "explanation_text": result["explanation_text"]["text"],
                        }
                    )

        episode_summaries.append(
            {
                "episode": episode,
                "reward": round(episode_reward, 4),
                "steps": state["step_count"],
                "won": state["game_state"].value == "won",
                "samples": step_samples,
                "questions": episode_checks,
                "explain_true_rate": round(
                    episode_explain_true / max(1, episode_checks),
                    4,
                ),
            }
        )

    by_intent_summary = {}
    for intent, counter in sorted(by_intent.items()):
        question_count = counter["questions"]
        by_intent_summary[intent] = {
            "questions": question_count,
            "explain_true": counter["explain_true"],
            "explain_true_rate": round(counter["explain_true"] / max(1, question_count), 4),
            "basis_rate": round(counter["Basis_{u,t}(E, Q)"] / max(1, question_count), 4),
            "minimal_rate": round(counter["Minimal(E)"] / max(1, question_count), 4),
            "readable_rate": round(counter["Readable_u(x)"] / max(1, question_count), 4),
        }

    total_questions = totals["questions"]
    summary = {
        "agent": type(agent).__name__,
        "parser_backend": parser.backend,
        "episodes": args.episodes,
        "grid_size": grid_size,
        "num_monsters": num_monsters,
        "max_steps": args.max_steps,
        "sample_every": args.sample_every,
        "questions": total_questions,
        "explain_true": totals["explain_true"],
        "explain_true_rate": round(totals["explain_true"] / max(1, total_questions), 4),
        "symbolic_match_rate": round(totals["symbolic_match"] / max(1, total_questions), 4),
        "symbolic_support_rate": round(totals["symbolic_support"] / max(1, total_questions), 4),
        "fallback_rate": round(totals["fallback_used"] / max(1, total_questions), 4),
        "basis_rate": round(totals["Basis_{u,t}(E, Q)"] / max(1, total_questions), 4),
        "minimal_rate": round(totals["Minimal(E)"] / max(1, total_questions), 4),
        "render_match_rate": round(totals["x = R_u(E, Q)"] / max(1, total_questions), 4),
        "readable_rate": round(totals["Readable_u(x)"] / max(1, total_questions), 4),
    }

    if symbolic_policy is not None:
        summary["distillation_metrics"] = symbolic_policy.metrics

    payload = {
        "summary": summary,
        "by_intent": by_intent_summary,
        "episodes": episode_summaries,
        "failures": failure_examples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[validate] agent={summary['agent']} backend={summary['parser_backend']}")
    print(
        f"[validate] explain_true={summary['explain_true']}/{summary['questions']} "
        f"({summary['explain_true_rate']:.2%})"
    )
    print(
        f"[validate] basis={summary['basis_rate']:.2%} "
        f"minimal={summary['minimal_rate']:.2%} "
        f"readable={summary['readable_rate']:.2%}"
    )
    print(
        f"[validate] symbolic_match={summary['symbolic_match_rate']:.2%} "
        f"symbolic_support={summary['symbolic_support_rate']:.2%} "
        f"fallback={summary['fallback_rate']:.2%}"
    )
    print(f"[validate] report -> {output_path}")


if __name__ == "__main__":
    main()
