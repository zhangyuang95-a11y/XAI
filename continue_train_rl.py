"""Continue the same-version scratch candidate after formal diagnosis.

Run this file directly from PyCharm (the gutter triangle is available beside
``main``).  The v21 checkpoint began from random initialization and is treated
as immutable input.  Continued artifacts use a separate directory, so the
2,800-episode candidate and its formal report remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys

from backend.artifacts import CollaborativeArtifactPaths
from env.warehouse.contracts import ARTIFACT_NAMESPACE


def main() -> None:
    if len(sys.argv) == 1:
        import torch

        project_root = Path(__file__).resolve().parent
        source_checkpoint = (
            project_root
            / "output"
            / "collaborative"
            / ARTIFACT_NAMESPACE
            / "training_checkpoint.pt"
        )
        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                "The immutable v21 scratch-training checkpoint is missing: "
                f"{source_checkpoint}"
            )
        artifacts = CollaborativeArtifactPaths.under(
            project_root,
            f"{ARTIFACT_NAMESPACE}_continued_3800",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sys.argv.extend(
            [
                "--agents", "2",
                "--max-agents", "2",
                "--episodes", "3800",
                "--horizon", "120",
                "--resume", str(source_checkpoint),
                "--resume-behavior-cloning-epochs", "8",
                "--actor-lr", "0.00002",
                "--actor-lr-final", "0.000005",
                "--critic-lr", "0.0003",
                "--critic-lr-final", "0.00005",
                "--entropy-coef-start", "0.002",
                "--entropy-coef-final", "0.0005",
                "--energy-curriculum-probability", "0.20",
                "--energy-curriculum-min-battery", "15",
                "--energy-curriculum-max-battery", "35",
                "--energy-curriculum-fade-start", "0.80",
                "--energy-curriculum-fade-end", "0.98",
                "--coordination-curriculum-probability", "0.45",
                "--coordination-curriculum-fade-start", "0.80",
                "--coordination-curriculum-fade-end", "0.98",
                "--behavior-cloning-samples", "32768",
                "--behavior-cloning-batch-size", "512",
                "--behavior-cloning-lr", "0.00015",
                "--skill-retention-samples", "16384",
                "--skill-retention-weight", "1.0",
                "--learner-state-relabel-every", "100",
                "--learner-state-relabel-samples", "8192",
                "--learner-state-relabel-replay-capacity", "32768",
                "--learner-state-relabel-epochs", "12",
                "--learner-state-relabel-lr", "0.00012",
                "--learner-state-detour-samples", "64",
                "--learner-state-detour-search-episodes", "256",
                "--learner-state-collision-samples", "64",
                "--learner-state-collision-search-episodes", "256",
                "--learner-state-non-wait-margin", "1.0",
                "--learner-state-non-wait-weight", "0.5",
                "--learner-state-escape-wait-margin", "2.0",
                "--learner-state-escape-wait-weight", "1.0",
                "--learner-state-correction-margin", "3.0",
                "--learner-state-correction-weight", "3.0",
                "--learner-state-wait-margin", "1.0",
                "--learner-state-wait-weight", "1.0",
                "--periodic-eval-every", "200",
                "--periodic-eval-episodes", "50",
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
                "--reference-trajectory-output",
                str(artifacts.reference_trajectory),
            ]
        )
        print(
            "[候选续训] v21 的 2,800 回合 checkpoint -> 独立 continued 目录；"
            f"设备={device}，目标 episodes=3800，输出={artifacts.root}。"
        )

    from backend.training.warehouse import main as train

    train()


if __name__ == "__main__":
    main()
