"""Small, release-independent source-closure hasher for diagnostic evidence.

Evidence producers must not depend on the mutable release admission module just
to discover their Python import closure.  Keeping this utility separate means a
concurrent packaging edit cannot invalidate a long-running scientific audit.
"""
from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


def _module_files(parts: Sequence[str]) -> list[Path]:
    if not parts or any(not part or part in (".", "..") for part in parts):
        return []
    base = ROOT.joinpath(*parts)
    result = []
    source = base.with_suffix(".py")
    package = base / "__init__.py"
    if source.is_file():
        result.append(source)
    if package.is_file():
        result.append(package)
    return result


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def local_source_hashes(seed_paths: Sequence[str | Path], *,
                        exclude_paths: Sequence[str | Path] = ()) -> dict[str, str]:
    """Hash the recursive local-Python import closure plus explicit assets."""
    excluded = {Path(path).resolve() for path in exclude_paths}
    queue = []
    for value in seed_paths:
        supplied = Path(value)
        if supplied.is_symlink():
            raise ValueError("Diagnostic source closure cannot contain a symlink")
        queue.append(supplied.resolve())
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen or path in excluded:
            continue
        if (path.is_symlink() or not path.is_file()
                or (path != ROOT and ROOT not in path.parents)):
            raise ValueError("Diagnostic source closure contains a missing or external path")
        seen.add(path)
        if path.suffix != ".py":
            continue
        relative = path.relative_to(ROOT)
        module_parts = list(relative.with_suffix("").parts)
        package_parts = module_parts[:-1]
        if module_parts[-1] == "__init__":
            package_parts = module_parts[:-1]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (SyntaxError, UnicodeError) as error:
            raise ValueError(
                "Cannot parse diagnostic source closure file: " + str(relative)
            ) from error
        imported: list[list[str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = len(package_parts) - (node.level - 1)
                    if keep < 0:
                        continue
                    base = package_parts[:keep]
                else:
                    base = []
                if node.module:
                    base = [*base, *node.module.split(".")]
                if base:
                    imported.append(base)
                for alias in node.names:
                    if alias.name != "*":
                        imported.append([*base, *alias.name.split(".")])
        for parts in imported:
            for imported_path in _module_files(parts):
                if imported_path.is_symlink():
                    raise ValueError("Diagnostic source closure cannot contain a symlink")
                if imported_path not in seen and imported_path not in excluded:
                    queue.append(imported_path)
            for index in range(1, len(parts)):
                initializer = ROOT.joinpath(*parts[:index], "__init__.py")
                if initializer.is_symlink():
                    raise ValueError("Diagnostic source closure cannot contain a symlink")
                if (initializer.is_file() and initializer not in seen
                        and initializer not in excluded):
                    queue.append(initializer)
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(seen)}


__all__ = ["ROOT", "file_hash", "local_source_hashes"]
