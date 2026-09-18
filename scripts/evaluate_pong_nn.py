#!/usr/bin/env python3
"""Read-only fixed-seed Cooperative Pong evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from domains.pong.training.evaluation import ALL_CONDITIONS, evaluate_trainer
from domains.pong.training.runner import PongPPOTrainer, _device, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "final_test"), default="validation")
    parser.add_argument("--partner", choices=("all", *ALL_CONDITIONS), default="all")
    parser.add_argument("--controller-mode", choices=("pure_nn", "hybrid", "coordinated", "rule_only", "all"), default="pure_nn")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(args.run)
    trainer = PongPPOTrainer(load_config(root / "config.yaml"), device=_device(args.device))
    checkpoint = root / args.checkpoint
    trainer.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    report = evaluate_trainer(
        trainer, split=args.split, partner=args.partner, controller_mode=args.controller_mode,
        progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
    )
    report["checkpoint"] = checkpoint.name
    report["joint_steps"] = trainer.joint_steps
    report_path = root / "evaluations" / f"{args.split}_{args.controller_mode}_{args.partner}_{trainer.joint_steps:07d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
