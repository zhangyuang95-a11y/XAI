#!/usr/bin/env python3
"""Start or safely resume Cooperative Pong PPO/RCPD training."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domains.pong.training.runner import run

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-joint-steps", type=int)
    parser.add_argument("--max-runtime-minutes", type=float)
    args = parser.parse_args()
    run(args.config, args.output, device=args.device, resume=args.resume,
        max_joint_steps=args.max_joint_steps, max_runtime_minutes=args.max_runtime_minutes)
    return 0
if __name__ == "__main__": raise SystemExit(main())
