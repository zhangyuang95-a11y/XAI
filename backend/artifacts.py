"""Stable identifiers for model and program artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CollaborativeArtifactPaths:
    """Canonical filenames for one immutable experiment artifact namespace."""

    root: Path
    model: Path
    training_checkpoint: Path
    metrics: Path
    training_plot: Path
    training_summary: Path
    rcpd_program: Path
    rcpd_python: Path
    training_trajectory: Path
    parallel_seed_pairs: Path
    reference_trajectory: Path
    formal_evaluation: Path
    study_database: Path
    explanation_validation: Path
    trace_ablation_results: Path

    @classmethod
    def under(
        cls,
        project_root: str | Path,
        namespace: str,
    ) -> "CollaborativeArtifactPaths":
        root = (
            Path(project_root).resolve()
            / "output"
            / "collaborative"
            / str(namespace)
        )
        return cls(
            root=root,
            model=root / "warehouse_mappo.pt",
            training_checkpoint=root / "training_checkpoint.pt",
            metrics=root / "training_metrics.csv",
            training_plot=root / "training_progress.png",
            training_summary=root / "training_summary.json",
            rcpd_program=root / "rcpd_program.json",
            rcpd_python=root / "rcpd_program.py",
            training_trajectory=root / "training_trajectory.pkl.gz",
            parallel_seed_pairs=root / "parallel_seed_pairs.json",
            reference_trajectory=root / "reference_trajectory.json",
            formal_evaluation=root / "formal_evaluation.json",
            study_database=root / "collaborative_study.sqlite3",
            explanation_validation=root / "explanation_validation.json",
            trace_ablation_results=root / "trace_ablation_results.jsonl",
        )


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one existing artifact."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
