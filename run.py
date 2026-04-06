"""
run.py -- Entry point for the Pac-Man XAI demo.

Examples:
    py -3 run.py
    py -3 run.py --agent rl --model-path models/dqn_pacman.pt
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
import tkinter as tk

from agent import HeuristicAgent, RLAgent
from environment import MazeEnvironment
from evidence_recorder import EvidenceRecorder
from explanation_engine import ExplanationEngine
from question_parser import QuestionParser
from ui import MazeGameUI


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Pac-Man XAI demo.")
    parser.add_argument("--agent", choices=["auto", "heuristic", "rl"], default="auto")
    parser.add_argument("--model-path", default="models/dqn_pacman.pt")
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--num-monsters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    return parser


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


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path)
    agent = create_agent(args.agent, model_path)

    grid_size = args.grid_size
    num_monsters = args.num_monsters
    if isinstance(agent, RLAgent):
        trained_grid = agent.metadata.get("grid_size")
        trained_monsters = agent.metadata.get("num_monsters")
        if trained_grid:
            grid_size = int(trained_grid)
        if trained_monsters:
            num_monsters = int(trained_monsters)

    root = tk.Tk()
    root.resizable(True, True)

    env = MazeEnvironment(
        grid_size=grid_size,
        num_monsters=num_monsters,
        seed=args.seed,
    )
    recorder = EvidenceRecorder(max_history=40)
    parser = QuestionParser(semantic=True)
    engine = ExplanationEngine()

    MazeGameUI(root, env, agent, recorder, parser, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
