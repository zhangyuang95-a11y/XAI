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

from agent import RLAgent
from environment import MazeEnvironment
from evidence_recorder import EvidenceRecorder
from explanation_engine import ExplanationEngine
from question_parser import QuestionParser
from ui import MazeGameUI


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FIXED_MODEL_PATH = Path("models/dqn_pacman.pt")
FIXED_GRID_SIZE = 11
FIXED_NUM_MONSTERS = 2
FIXED_REWARD_PRESET = "stable"


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Run the fixed Pac-Man XAI demo.",
        epilog="Fixed configuration: 11x11 maze, 2 monsters, manual question input, RL model only.",
    )


def build_fixed_runtime() -> argparse.Namespace:
    return argparse.Namespace(
        model_path=FIXED_MODEL_PATH,
        grid_size=FIXED_GRID_SIZE,
        num_monsters=FIXED_NUM_MONSTERS,
        seed=None,
    )


def create_agent(model_path: Path) -> RLAgent:
    if not model_path.exists():
        raise FileNotFoundError(
            f"RL checkpoint not found: {model_path}. Train the fixed model first with `py -3 train_rl.py`."
        )
    return RLAgent(model_path=model_path)


def main() -> None:
    build_parser().parse_args()
    args = build_fixed_runtime()
    model_path = Path(args.model_path)
    print("[cfg  ] fixed demo configuration -> 11x11 maze, 2 monsters, manual question input")
    agent = create_agent(model_path)

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
    engine = ExplanationEngine()

    MazeGameUI(root, env, agent, recorder, parser, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
