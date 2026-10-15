"""Prepare and run isolated structured Warehouse v5.4 development variants.

The ``bc`` mode saves the common offline initialization. ``raw`` and
``regularized`` each resume from that identical cache. After both 6099 pilot
checkpoints are complete, ``finalize-pilot`` writes the separate structured
acceptance and applies the existing development preflight. No mode here opens
the independent final program audit or held-out episode seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from backend.training import warehouse_dual_structured_v54 as trainer
from backend.training.warehouse_v54_gate import development_preflight


BC_EPISODES = 200
BC_EPOCHS = 30


def _work_dir(output_dir: Path) -> Path:
    return output_dir / "_warehouse_dual_structured_v54_work"


def _bc_path(output_dir: Path, pair_seed: int) -> Path:
    return _work_dir(output_dir) / f"pair_{pair_seed}_bc_structured_v54.pt"


def _load_bc(path: Path, pair_seed: int) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("pair_seed") != pair_seed
            or saved.get("bc_episodes") != BC_EPISODES
            or saved.get("bc_epochs") != BC_EPOCHS
            or saved.get("actor_model_version") != trainer.STRUCTURED_MODEL_VERSION
            or saved.get("feature_schema_sha256") != trainer.STRUCTURED_FEATURE_SCHEMA_SHA256
            or saved.get("architecture_version") != trainer.BC_ARCHITECTURE_VERSION):
        raise RuntimeError("structured Warehouse variant BC cache mismatch")
    model = trainer.DualNN()
    model.load_state_dict(saved["state_dict"])
    report = saved.get("report") or {}
    if report.get("pretrained_role_sha256") != trainer._weight_hashes(model):
        raise RuntimeError("structured Warehouse BC Actor hashes disagree with cache")
    return saved


def _prepare_bc(output_dir: Path, pair_seed: int) -> dict:
    path = _bc_path(output_dir, pair_seed)
    if path.is_file():
        return _load_bc(path, pair_seed)
    torch.manual_seed(pair_seed)
    model = trainer.DualNN()
    report = trainer.pretrain_pair(
        model, seed=pair_seed, episodes=BC_EPISODES, epochs=BC_EPOCHS)
    payload = {"pair_seed": pair_seed, "bc_episodes": BC_EPISODES,
               "bc_epochs": BC_EPOCHS,
               "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
               "feature_schema_sha256": trainer.STRUCTURED_FEATURE_SCHEMA_SHA256,
               "architecture_version": trainer.BC_ARCHITECTURE_VERSION,
               "state_dict": model.state_dict(), "report": report}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return _load_bc(path, pair_seed)


def _pilot_checkpoints_complete(output_dir: Path, pair_seed: int,
                                joint_steps: int) -> None:
    for variant in ("raw", "regularized"):
        path = _work_dir(output_dir) / f"pair_{pair_seed}_{variant}_matched.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("pair_seed") != pair_seed or saved.get("variant") != variant
                or saved.get("joint_steps") != joint_steps
                or saved.get("actor_model_version") != trainer.STRUCTURED_MODEL_VERSION
                or saved.get("feature_schema_sha256") != trainer.STRUCTURED_FEATURE_SCHEMA_SHA256
                or saved.get("bc_architecture_version") != trainer.BC_ARCHITECTURE_VERSION
                or saved.get("program_feedback_version") != trainer.PROGRAM_FEEDBACK_VERSION
                or saved.get("complete") is not True):
            raise RuntimeError(f"structured Warehouse {variant} pilot checkpoint incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("bc", "raw", "regularized", "finalize-pilot"),
                        required=True)
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-steps", type=int, default=trainer.FULL_JOINT_STEPS)
    parser.add_argument("--development-pilot", action="store_true")
    args = parser.parse_args()
    if args.development_pilot:
        if args.pair_seed != trainer.DEVELOPMENT_PILOT_SEED:
            raise ValueError("structured development pilot must use seed 6099")
    elif args.pair_seed not in trainer.DEFAULT_SEEDS:
        raise ValueError("structured formal training requires one of the three fixed pair seeds")
    if args.variant == "finalize-pilot" and not args.development_pilot:
        raise ValueError("finalize-pilot requires the development pilot seed")
    if args.joint_steps <= 0:
        raise ValueError("joint steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.pair_seed)
    if args.variant == "bc":
        saved = _prepare_bc(args.output_dir, args.pair_seed)
        result = {"stage": "bc", "pair_seed": args.pair_seed,
                  "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
                  "role_sha256": saved["report"]["pretrained_role_sha256"],
                  "path": str(_bc_path(args.output_dir, args.pair_seed))}
    elif args.variant == "finalize-pilot":
        if args.joint_steps != trainer.FULL_JOINT_STEPS:
            raise ValueError("pilot acceptance requires the full 200k joint steps")
        _load_bc(_bc_path(args.output_dir, args.pair_seed), args.pair_seed)
        _pilot_checkpoints_complete(args.output_dir, args.pair_seed, args.joint_steps)
        acceptance = trainer.run_experiment(
            pair_seeds=(args.pair_seed,), joint_steps=args.joint_steps,
            bc_episodes=BC_EPISODES, bc_epochs=BC_EPOCHS,
            output_dir=args.output_dir, development_pilot=True)
        result = {"stage": "pilot-acceptance", "pair_seed": args.pair_seed,
                  "path": str(args.output_dir / "warehouse_dual_structured_v54_acceptance.json"),
                  "development_preflight": development_preflight(acceptance)}
    else:
        saved = _load_bc(_bc_path(args.output_dir, args.pair_seed), args.pair_seed)
        _, programs, report = trainer.train_variant(
            saved["state_dict"], pair_seed=args.pair_seed,
            variant=args.variant, joint_steps=args.joint_steps,
            development_seeds=(trainer.PILOT_DEVELOPMENT_SEEDS if args.development_pilot
                               else trainer.DEVELOPMENT_SEEDS),
            checkpoint_path=_work_dir(args.output_dir) /
                            f"pair_{args.pair_seed}_{args.variant}_matched.pt")
        result = {"stage": args.variant, "pair_seed": args.pair_seed,
                  "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
                  "feature_schema_sha256": trainer.STRUCTURED_FEATURE_SCHEMA_SHA256,
                  "selected_development": report["selected_development"],
                  "selected_feedback_rows": report["selected_feedback_rows"],
                  "programs": {role: program.sha256() for role, program in
                               (programs or {}).items() if program is not None}}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
