from __future__ import annotations

import ast
from functools import cache
from pathlib import Path
import sys

from backend.artifacts import CollaborativeArtifactPaths
from backend.adapters.base import ActionDistribution as AdapterActionDistribution
from core.policy_contracts import ActionDistribution
from env.warehouse import environment
from env.warehouse.contracts import ARTIFACT_NAMESPACE, CURRENT_VERSIONS
from env.warehouse.domain import WarehouseConfig
from env.warehouse.navigation import shortest_path_distance
from study_v3.domain_contracts import LAYERS


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "backend", "core", "domains", "env", "evaluation", "experiments",
    "scripts", "study_v3", "ui",
)
# Current HTTP, training/admission, and release commands. The registry imports
# participant adapters by name, so its declared domain layers are roots too.
# R4/R41 scripts and interfaces are retained for historical evidence, but are
# not entrypoints for the v6 release.
ACTIVE_ENTRYPOINTS = (
    "ui.domain_hub_server",  # render.yaml
    "study_v3.server",  # README.md local launch
    "experiments.warehouse.admission",
    "experiments.warehouse.acceptance",
    "backend.training.warehouse_v6_runner",
    "scripts.package_warehouse_v6_release",
    "scripts.start_warehouse_v6_candidate_release",
    *(module for layers in LAYERS.values() for module in vars(layers).values()),
)
ACTIVE_LINE_LIMIT = 2000
# Existing shared v6 physics module exceeds the general limit. Freeze this
# specific debt at its current size; all other active modules, including the
# v6 runner, retain the 2,000-line ceiling.
ACTIVE_LINE_DEBT = {"env.warehouse.environment": 2171}


def _repository_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for root_name in SOURCE_ROOTS:
        for path in (ROOT / root_name).rglob("*.py"):
            if path.name == "__init__.py":
                continue
            modules[".".join(path.relative_to(ROOT).with_suffix("").parts)] = path
    return modules


def _local_module(requested: str, modules: dict[str, Path]) -> str | None:
    parts = requested.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in modules:
            return candidate
    return None


def _imports(
    source: str, node: ast.AST, modules: dict[str, Path]
) -> set[str]:
    if isinstance(node, ast.Import):
        requests = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        package = source.split(".")[:-node.level] if node.level else []
        base = ".".join((*package, *([node.module] if node.module else [])))
        # `from package import module` and `from . import module` name the
        # imported module in an alias rather than in node.module.
        requests = [f"{base}.{alias.name}" for alias in node.names]
        requests.append(base)
    else:
        return set()
    return {
        target for requested in requests
        if (target := _local_module(requested, modules)) is not None
    }


def _module_dependencies(
    source: str, tree: ast.AST, modules: dict[str, Path]
) -> tuple[set[str], set[str]]:
    all_imports: set[str] = set()
    import_time_imports: set[str] = set()

    def visit(node: ast.AST, deferred: bool = False) -> None:
        dependencies = _imports(source, node, modules)
        all_imports.update(dependencies)
        if not deferred:
            import_time_imports.update(dependencies)
        child_deferred = deferred or isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        )
        for child in ast.iter_child_nodes(node):
            visit(child, child_deferred)

    visit(tree)
    return all_imports, import_time_imports


@cache
def _active_graph() -> tuple[dict[str, Path], dict[str, set[str]]]:
    modules = _repository_modules()
    missing = set(ACTIVE_ENTRYPOINTS) - modules.keys()
    assert not missing, f"missing active entrypoints: {sorted(missing)}"
    active: dict[str, Path] = {}
    import_time_graph: dict[str, set[str]] = {}
    pending = list(ACTIVE_ENTRYPOINTS)
    while pending:
        source = pending.pop()
        if source in active:
            continue
        path = modules[source]
        active[source] = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        all_imports, import_time_imports = _module_dependencies(source, tree, modules)
        import_time_graph[source] = import_time_imports
        pending.extend(all_imports - active.keys())
    return active, import_time_graph


def _active_modules() -> dict[str, Path]:
    return _active_graph()[0]


def test_production_dependency_graph_is_acyclic() -> None:
    # A function-local import is deferred until that function runs. Such
    # backreferences do not create Python import-initialization cycles.
    graph = _active_graph()[1]
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


def test_active_scope_includes_v6_and_excludes_archived_candidates() -> None:
    active = _active_modules()
    assert "backend.training.warehouse_v6_runner" in active
    assert "env.warehouse.environment" in active
    assert "domains.warehouse.neural_turnbased" in active
    assert "backend.training.warehouse_r41_diagnostic_rcpd_v8" not in active
    assert "ui.warehouse_alignment_r42_server" not in active


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
        str(path.relative_to(ROOT)): line_count
        for module, path in _active_modules().items()
        if (line_count := len(path.read_text(encoding="utf-8").splitlines()))
        > ACTIVE_LINE_DEBT.get(module, ACTIVE_LINE_LIMIT)
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
        for path in _active_modules().values()
        if path.name != "contracts.py"
    )
    for value in CURRENT_VERSIONS.__dict__.values():
        assert str(value) in contract_source
        assert str(value) not in other_sources


def test_default_artifact_bundle_has_no_floating_current_alias() -> None:
    artifacts = CollaborativeArtifactPaths.under(ROOT, ARTIFACT_NAMESPACE)
    assert artifacts.root.name == ARTIFACT_NAMESPACE
    assert "current" not in artifacts.root.parts
    assert artifacts.model.name == "warehouse_mappo.pt"
    assert artifacts.reference_trajectory.name == "reference_trajectory.json"
