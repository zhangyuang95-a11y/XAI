from __future__ import annotations

import ast
from pathlib import Path
import sys

from backend.artifacts import CollaborativeArtifactPaths
from backend.adapters.base import ActionDistribution as AdapterActionDistribution
from core.policy_contracts import ActionDistribution
from env.warehouse import environment
from env.warehouse.contracts import ARTIFACT_NAMESPACE, CURRENT_VERSIONS
from env.warehouse.domain import WarehouseConfig
from env.warehouse.navigation import shortest_path_distance


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = ("backend", "core", "env", "evaluation", "ui")


def _production_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for root_name in PRODUCTION_ROOTS:
        for path in (ROOT / root_name).rglob("*.py"):
            if path.name == "__init__.py":
                continue
            modules[".".join(path.relative_to(ROOT).with_suffix("").parts)] = path
    return modules


def _dependency_graph() -> dict[str, set[str]]:
    modules = _production_modules()
    graph = {name: set() for name in modules}
    for source, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = source.split(".")[:-node.level]
                    imported.append(".".join(package + ([node.module] if node.module else [])))
                elif node.module:
                    imported.append(node.module)
            for requested in imported:
                for target in modules:
                    if requested == target or requested.startswith(f"{target}."):
                        graph[source].add(target)
                        break
    return graph


def test_production_dependency_graph_is_acyclic() -> None:
    graph = _dependency_graph()
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, path: tuple[str, ...]) -> None:
        if module in visiting:
            cycle_start = path.index(module)
            cycle = (*path[cycle_start:], module)
            raise AssertionError("dependency cycle: " + " -> ".join(cycle))
        if module in visited:
            return
        visiting.add(module)
        for dependency in sorted(graph[module]):
            visit(dependency, (*path, module))
        visiting.remove(module)
        visited.add(module)

    for module in sorted(graph):
        visit(module, ())


def test_environment_compatibility_exports_are_identity_preserving() -> None:
    assert environment.WarehouseConfig is WarehouseConfig
    assert environment.shortest_path_distance is shortest_path_distance
    assert AdapterActionDistribution is ActionDistribution


def test_online_policy_consumers_do_not_import_training_module() -> None:
    consumers = (
        ROOT / "ui" / "web_session.py",
        ROOT / "ui" / "web_application.py",
        ROOT / "ui" / "tutorial.py",
        ROOT / "env" / "warehouse" / "seed_calibration.py",
    )
    for path in consumers:
        source = path.read_text(encoding="utf-8")
        assert "env.warehouse.mappo" not in source
        assert "from .mappo" not in source


def test_refactored_modules_do_not_regress_into_unbounded_god_files() -> None:
    oversized = {
        str(path.relative_to(ROOT)): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in _production_modules().values()
        if len(path.read_text(encoding="utf-8").splitlines()) > 2000
    }
    assert oversized == {}


def test_compatibility_facades_remain_thin() -> None:
    web_facade = ROOT / "ui" / "web_runtime.py"
    assert len(web_facade.read_text(encoding="utf-8").splitlines()) < 60
    assert not (ROOT / "env" / "warehouse" / "train_rl.py").exists()


def test_launcher_reports_the_explicit_device_value(monkeypatch) -> None:
    from run import _option_value

    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--device", "cpu", "--port=8765"],
    )
    assert _option_value("--device", "cuda") == "cpu"
    assert _option_value("--port", "8000") == "8765"


def test_persisted_version_literals_have_one_authoritative_source() -> None:
    contract_source = (ROOT / "env" / "warehouse" / "contracts.py").read_text(
        encoding="utf-8"
    )
    other_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _production_modules().values()
        if path.name != "contracts.py"
    )
    for value in CURRENT_VERSIONS.__dict__.values():
        assert str(value) in contract_source
        assert str(value) not in other_sources


def test_default_artifact_bundle_has_no_floating_current_alias() -> None:
    artifacts = CollaborativeArtifactPaths.under(ROOT, ARTIFACT_NAMESPACE)
    assert artifacts.root.name == "simultaneous_compact6_v60_live_human_ai"
    assert "current" not in artifacts.root.parts
    assert artifacts.model.name == "warehouse_mappo.pt"
    assert artifacts.reference_trajectory.name == "reference_trajectory.json"
