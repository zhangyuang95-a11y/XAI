"""Launch the development UI for the two-robot collaborative experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from backend.artifacts import CollaborativeArtifactPaths, file_sha256
from env.warehouse.contracts import ARTIFACT_NAMESPACE, FORMAL_ACCEPTANCE_CHECKS


def _accepted_ui_artifacts(
    bundle: CollaborativeArtifactPaths,
) -> tuple[Path, Path] | None:
    """Return one bundle only after every formal acceptance gate passed."""

    required = (
        bundle.model,
        bundle.rcpd_program,
        bundle.training_summary,
        bundle.formal_evaluation,
        bundle.parallel_seed_pairs,
        bundle.reference_trajectory,
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        training = json.loads(bundle.training_summary.read_text(encoding="utf-8"))
        evaluation = json.loads(bundle.formal_evaluation.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    checks = evaluation.get("acceptance_checks")
    regularization = training.get("program_regularization")
    if (
        evaluation.get("formal_candidate") is not True
        or not isinstance(checks, dict)
        or not FORMAL_ACCEPTANCE_CHECKS.issubset(checks)
        or int(evaluation.get("episodes_per_condition", 0)) < 200
        or not all(value is True for value in checks.values())
        or not isinstance(regularization, dict)
        or regularization.get("explanation_eligible") is not True
        or training.get("model_version") != evaluation.get("model_version")
    ):
        return None
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str):
        return None
    try:
        if Path(checkpoint).resolve() != bundle.model.resolve():
            return None
    except OSError:
        return None
    hashes = evaluation.get("artifact_hashes")
    expected_hash_paths = {
        "model": bundle.model,
        "program": bundle.rcpd_program,
        "training_summary": bundle.training_summary,
        "parallel_seed_library": bundle.parallel_seed_pairs,
        "reference_trajectory": bundle.reference_trajectory,
    }
    if not isinstance(hashes, dict):
        return None
    try:
        if any(
            not isinstance(hashes.get(name), str)
            or hashes[name] != file_sha256(path)
            for name, path in expected_hash_paths.items()
        ):
            return None
    except OSError:
        return None
    return bundle.model, bundle.rcpd_program


def _default_ui_artifacts() -> tuple[Path, Path] | None:
    project_root = Path(__file__).resolve().parent
    bundle = CollaborativeArtifactPaths.under(project_root, ARTIFACT_NAMESPACE)
    return _accepted_ui_artifacts(bundle)


def _has_option(name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in sys.argv[1:])


def _option_value(name: str, fallback: str) -> str:
    """Return the effective CLI value without parsing or mutating argv."""

    arguments = sys.argv[1:]
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return fallback


def main() -> None:
    formal_study = "--formal-study" in sys.argv
    if formal_study:
        sys.argv.remove("--formal-study")
        while "--test-condition-selector" in sys.argv:
            sys.argv.remove("--test-condition-selector")
    elif not _has_option("--test-condition-selector"):
        # The commonly clicked PyCharm entry is intentionally safe for
        # repeated interface testing.  Formal collection has its own clearly
        # named entry point and never enables manual condition selection.
        sys.argv.append("--test-condition-selector")
    if not _has_option("--transformer-model"):
        selected = _default_ui_artifacts()
        if selected is None:
            raise SystemExit(
                "尚无可用于界面的双机器人模型与 RCPD 程序。\n"
                "只有通过全部正式评估门槛的模型才会被默认加载。\n"
                "开发调试可显式传入 checkpoint/program，正式运行前请完成训练与评估。"
            )
        checkpoint, program = selected
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        defaults = [
            ("--checkpoint", str(checkpoint)),
            ("--program", str(program)),
            ("--transformer-model", "Qwen/Qwen2.5-3B-Instruct"),
            ("--device", device),
            ("--seed", "2026"),
        ]
        for option, value in defaults:
            if not _has_option(option):
                sys.argv.extend([option, value])
        print(
            "[Warehouse UI] "
            f"mode={'formal' if formal_study else 'development'} "
            f"checkpoint={checkpoint} program={program} "
            f"device={_option_value('--device', device)}"
        )

    from ui.web_server import main as run_web_ui

    run_web_ui()


if __name__ == "__main__":
    main()
