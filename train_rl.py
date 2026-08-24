"""Run the recommended two-robot MAPPO + RCPD training configuration."""

from __future__ import annotations

from pathlib import Path
import sys

from backend.artifacts import CollaborativeArtifactPaths
from env.warehouse.contracts import ARTIFACT_NAMESPACE


def main() -> None:
    if len(sys.argv) == 1:
        import torch

        artifacts = CollaborativeArtifactPaths.under(
            Path(__file__).resolve().parent,
            ARTIFACT_NAMESPACE,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sys.argv.extend(
            [
                "--agents", "2",
                "--max-agents", "2",
                "--episodes", "2800",
                "--horizon", "120",
                "--hidden-dim", "512",
                "--intent-dim", "64",
                "--actor-lr", "0.00002",
                "--actor-lr-final", "0.000005",
                "--critic-lr", "0.001",
                "--critic-lr-final", "0.0001",
                "--entropy-coef-start", "0.005",
                "--entropy-coef-final", "0.001",
                "--energy-curriculum-probability", "0.40",
                "--energy-curriculum-min-battery", "15",
                "--energy-curriculum-max-battery", "35",
                "--energy-curriculum-fade-start", "0.60",
                "--energy-curriculum-fade-end", "0.90",
                "--coordination-curriculum-probability", "0.30",
                "--coordination-curriculum-fade-start", "0.70",
                "--coordination-curriculum-fade-end", "0.95",
                "--behavior-cloning-samples", "65536",
                "--behavior-cloning-epochs", "30",
                "--behavior-cloning-batch-size", "512",
                "--skill-retention-samples", "32768",
                "--skill-retention-weight", "1.0",
                "--skill-retention-fade-end", "0.50",
                "--learner-state-relabel-every", "0",
                "--learner-state-relabel-warmup-rounds", "0",
                "--learner-state-relabel-samples", "0",
                "--learner-state-relabel-replay-capacity", "65536",
                "--learner-state-relabel-epochs", "50",
                "--learner-state-relabel-lr", "0.0003",
                "--learner-state-detour-samples", "16",
                "--learner-state-detour-search-episodes", "80",
                "--learner-state-collision-samples", "32",
                "--learner-state-collision-search-episodes", "128",
                "--learner-state-charger-cycle-samples", "64",
                "--learner-state-task-starvation-samples", "64",
                "--learner-state-commitment-search-episodes", "64",
                "--learner-state-commitment-curriculum-samples", "4096",
                "--learner-state-non-wait-margin", "0",
                "--learner-state-non-wait-weight", "0",
                "--learner-state-escape-wait-margin", "0",
                "--learner-state-escape-wait-weight", "0",
                "--learner-state-correction-margin", "0",
                "--learner-state-correction-weight", "0",
                "--learner-state-wait-margin", "0",
                "--learner-state-wait-weight", "0",
                "--periodic-eval-every", "200",
                "--periodic-eval-episodes", "20",
                "--eval-episodes", "200",
                "--seed", "4",
                "--device", device,
                "--use-rcpd",
                "--rcpd-max-depth", "6",
                "--rcpd-max-leaf-nodes", "24",
                "--rcpd-replay-records", "4096",
                "--checkpoint-every", "200",
                "--log-every", "25",
                "--output", str(artifacts.model),
                "--checkpoint-output", str(artifacts.training_checkpoint),
                "--metrics-output", str(artifacts.metrics),
                "--plot-output", str(artifacts.training_plot),
                "--summary-output", str(artifacts.training_summary),
                "--rcpd-program-output", str(artifacts.rcpd_program),
                "--rcpd-python-output", str(artifacts.rcpd_python),
                "--trajectory-output", str(artifacts.training_trajectory),
                "--parallel-seed-output", str(artifacts.parallel_seed_pairs),
                "--reference-trajectory-output", str(artifacts.reference_trajectory),
            ]
        )
        print(
            "[正式训练] 从零训练双机器人共享配送策略；"
            f"设备={device}，episodes=2800，输出={artifacts.root}。"
        )

    from backend.training.warehouse import main as train

    train()


if __name__ == "__main__":
    main()
