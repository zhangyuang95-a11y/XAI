"""
run.py -- Entry point for the Pac-Man style XAI demo.

Usage:
    py -3 run.py
"""

from __future__ import annotations

import io
import sys
import tkinter as tk

from agent import HeuristicAgent
from environment import MazeEnvironment
from evidence_recorder import EvidenceRecorder
from explanation_engine import ExplanationEngine
from question_parser import QuestionParser
from ui import MazeGameUI


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    root = tk.Tk()
    root.resizable(True, True)

    env = MazeEnvironment(grid_size=21, num_monsters=8)
    agent = HeuristicAgent(danger_radius=3, danger_penalty=80.0)
    recorder = EvidenceRecorder(max_history=40)
    parser = QuestionParser(semantic=True)
    engine = ExplanationEngine()

    MazeGameUI(root, env, agent, recorder, parser, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
