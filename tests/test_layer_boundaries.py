from __future__ import annotations

import ast
from pathlib import Path

from core.rcpd import implementation_audit


ROOT = Path(__file__).resolve().parents[1]


def test_root_layout_contains_only_clear_project_groups() -> None:
    directories = {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir()
        and path.name
        not in {
            ".git",
            ".idea",
            ".playwright-cli",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
        }
    }
    assert directories == {
        "backend",
        "core",
        "docs",
        "env",
        "evaluation",
        "output",
        "tests",
        "ui",
    }
    assert (ROOT / "output" / ".gitignore").is_file()


def test_root_entrypoints_expose_pycharm_run_gutters() -> None:
    for filename in (
        "train_rl.py",
        "run.py",
        "validate_explanations.py",
        "evaluate_trace_ablation.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":\n    main()' in source


def test_rcpd_and_regularizer_are_environment_independent_core_modules() -> None:
    source_files = sorted(
        path.name for path in (ROOT / "core").glob("*.py")
    )
    assert source_files == [
        "__init__.py",
        "policy_contracts.py",
        "policy_program_regularizer.py",
        "program.py",
        "rcpd.py",
        "rcpd_config.py",
        "rcpd_tree.py",
    ]
    audit = implementation_audit()
    assert audit["environment_independent"] is True
    assert audit["forbidden_environment_imports"] == []


def test_generic_layers_do_not_import_environment_packages() -> None:
    generic_roots = (
        ROOT / "core",
        ROOT / "evaluation",
        ROOT / "backend" / "nlp",
        ROOT / "backend" / "simulation",
    )
    violations: list[str] = []
    for root in generic_roots:
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    module = ",".join(alias.name for alias in node.names)
                if any(
                    name.strip().startswith("env")
                    for name in module.split(",")
                ):
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert violations == []


def test_generic_evaluator_contains_no_warehouse_predicate_vocabulary() -> None:
    forbidden = {
        "warehouse",
        "charger",
        "battery",
        "carrying_item",
        "movement_intent",
        "goal_distance",
        "pickup",
        "delivery",
    }
    sources = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (ROOT / "evaluation").glob("*.py")
    )
    assert sorted(token for token in forbidden if token in sources) == []


def test_environment_has_no_backend_ui_evaluation_dependencies() -> None:
    violations: list[str] = []
    for path in (ROOT / "env" / "warehouse").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    # A relative module such as ``.evaluation_diagnostics``
                    # remains inside env.warehouse; it is not a dependency
                    # on the top-level evaluation package.
                    continue
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(
                    alias.name for alias in node.names
                )
            if any(
                name.strip().startswith(("backend", "ui", "evaluation"))
                for name in module.split(",")
            ):
                violations.append(
                    f"{path.relative_to(ROOT)} imports {module}"
                )
    assert violations == []


def test_ui_contains_no_training_or_explanation_algorithm_implementations() -> None:
    source = (ROOT / "ui" / "web_runtime.py").read_text(
        encoding="utf-8"
    )
    forbidden_symbols = (
        "class WarehouseStudyController",
        "def _lime_explanation",
        "def _viper_explanation",
        "MAPPOTrainer",
        "CounterfactualEngine(",
        "ObjectiveValidityEvaluator(",
    )
    assert not any(symbol in source for symbol in forbidden_symbols)


def test_generation_path_does_not_contain_evaluation_verdict_labels() -> None:
    source = (
        ROOT / "backend" / "nlp" / "explanation_generator.py"
    ).read_text(encoding="utf-8")
    assert "ClaimVerdictStatus" not in source
    assert "SUPPORTED" not in source
    assert "CONTRADICTED" not in source
    assert "UNVERIFIABLE" not in source
