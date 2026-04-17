"""
run.py -- Entry point for the Pac-Man XAI demo.

Examples:
    py -3 run.py
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
import tkinter as tk

from .agent import RLAgent
from .environment import MazeEnvironment
from .evidence_recorder import EvidenceRecorder
from .explanation_engine import ExplanationEngine
from .question_parser import QuestionParser
from .ui import MazeGameUI


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FIXED_MODEL_PATH = Path("models/dqn_pacman.pt")
FIXED_SYMBOLIC_PATH = Path("models/dqn_pacman_symbolic.joblib")
FIXED_GRID_SIZE = 11
FIXED_NUM_MONSTERS = 2
FIXED_REWARD_PRESET = "stable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Pac-Man XAI demo.",
        epilog="Fixed configuration: 11x11 maze, 2 monsters, manual question input, RL model only.",
    )
    parser.add_argument(
        "--symbolic-path",
        default=str(FIXED_SYMBOLIC_PATH),
        help="Optional symbolic policy artifact. If missing, the UI uses evidence-only explanations.",
    )
    parser.add_argument(
        "--no-symbolic-policy",
        action="store_true",
        help="Disable the symbolic policy lens and use evidence-only explanations.",
    )
    parser.add_argument(
        "--require-symbolic-policy",
        action="store_true",
        help="Fail fast if the symbolic policy artifact cannot be loaded.",
    )
    return parser


def build_fixed_runtime(cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=FIXED_MODEL_PATH,
        symbolic_path=Path(cli_args.symbolic_path),
        grid_size=FIXED_GRID_SIZE,
        num_monsters=FIXED_NUM_MONSTERS,
        seed=None,
        no_symbolic_policy=bool(cli_args.no_symbolic_policy),
        require_symbolic_policy=bool(cli_args.require_symbolic_policy),
    )


def create_agent(model_path: Path) -> RLAgent:
    if not model_path.exists():
        raise FileNotFoundError(
            f"RL checkpoint not found: {model_path}. Train the fixed model first with `py -3 train_rl.py`."
        )
    return RLAgent(model_path=model_path)


def load_optional_symbolic_policy(agent: RLAgent, symbolic_path: Path, *, disabled: bool, required: bool):
    if disabled:
        print("[xai  ] symbolic policy disabled; using evidence-only explanations")
        return None

    if not symbolic_path.exists():
        message = f"Symbolic policy artifact not found: {symbolic_path}. Using evidence-only explanations."
        if required:
            raise FileNotFoundError(message)
        print(f"[xai  ] {message}")
        return None

    try:
        from .symbolic_policy_adapter import load_symbolic_policy

        symbolic_policy = load_symbolic_policy(symbolic_path)
        symbolic_policy.validate_compatibility(agent.metadata)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        message = f"Symbolic policy unavailable ({exc}). Using evidence-only explanations."
        if required:
            raise RuntimeError(message) from exc
        print(f"[xai  ] {message}")
        return None
    return symbolic_policy


def main() -> None:
    cli_args = build_parser().parse_args()
    args = build_fixed_runtime(cli_args)
    model_path = Path(args.model_path)
    symbolic_path = Path(args.symbolic_path)
    print("[cfg  ] fixed demo configuration -> 11x11 maze, 2 monsters, manual question input")
    agent = create_agent(model_path)
    symbolic_policy = load_optional_symbolic_policy(
        agent,
        symbolic_path,
        disabled=args.no_symbolic_policy,
        required=args.require_symbolic_policy,
    )

    grid_size = args.grid_size
    num_monsters = args.num_monsters
    max_steps = None
    reward_preset = FIXED_REWARD_PRESET
    trained_grid = agent.metadata.get("grid_size")
    trained_monsters = agent.metadata.get("num_monsters")
    trained_max_steps = agent.metadata.get("max_steps")
    trained_reward_preset = agent.metadata.get("reward_preset")
    if trained_grid:
        grid_size = int(trained_grid)
    if trained_monsters:
        num_monsters = int(trained_monsters)
    if trained_max_steps:
        max_steps = int(trained_max_steps)
    if trained_reward_preset:
        reward_preset = str(trained_reward_preset)

    root = tk.Tk()
    root.resizable(True, True)

    env = MazeEnvironment(
        grid_size=grid_size,
        num_monsters=num_monsters,
        seed=args.seed,
        max_steps=max_steps,
        reward_preset=reward_preset,
    )
    recorder = EvidenceRecorder(max_history=40)
    parser = QuestionParser(semantic=True)
    engine = ExplanationEngine(symbolic_policy=symbolic_policy)

    MazeGameUI(root, env, agent, recorder, parser, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
