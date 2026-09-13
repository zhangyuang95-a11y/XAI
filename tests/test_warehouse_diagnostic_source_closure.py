from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

import pytest

from backend.training import warehouse_diagnostic_source_closure as subject


def _replace(path: Path, content: str) -> None:
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_text(content, encoding="utf-8")
    os.replace(replacement, path)


def _source_tree(tmp_path: Path) -> tuple[Path, str, str]:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "dep_a.py").write_text("VALUE = 'a'\n", encoding="utf-8")
    (package / "dep_b.py").write_text("VALUE = 'b'\n", encoding="utf-8")
    source_a = "import pkg.dep_a\n"
    source_b = "import pkg.dep_b\n"
    seed = tmp_path / "seed.py"
    seed.write_text(source_a, encoding="utf-8")
    return seed, source_a, source_b


def test_import_graph_and_digest_use_the_same_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, source_a, source_b = _source_tree(tmp_path)
    monkeypatch.setattr(subject, "ROOT", tmp_path)

    # This recreates the old split-read attack: an AST read observes A, the
    # pathname changes to B before a later hash read, then is restored to A.
    # The hardened implementation calls neither split operation.
    original_read_text = Path.read_text

    def separated_ast_read(path: Path, *args, **kwargs) -> str:
        value = original_read_text(path, *args, **kwargs)
        if path == seed:
            _replace(seed, source_b)
        return value

    def separated_hash(path: str | Path) -> str:
        candidate = Path(path)
        raw = candidate.read_bytes()
        if candidate == seed:
            _replace(seed, source_a)
        return sha256(raw).hexdigest()

    monkeypatch.setattr(Path, "read_text", separated_ast_read)
    monkeypatch.setattr(subject, "file_hash", separated_hash)

    result = subject.local_source_hashes((seed,))

    assert result["seed.py"] == sha256(source_a.encode("utf-8")).hexdigest()
    assert "pkg/dep_a.py" in result
    assert "pkg/dep_b.py" not in result
    assert seed.read_text(encoding="utf-8") == source_a


def test_atomic_path_replacement_cannot_split_one_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, source_a, source_b = _source_tree(tmp_path)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    seed_inode = seed.stat().st_ino
    original_os_read = subject.os.read
    attacked = False

    def replace_around_descriptor_read(descriptor: int, count: int) -> bytes:
        nonlocal attacked
        if not attacked and os.fstat(descriptor).st_ino == seed_inode:
            attacked = True
            preserved_a = seed.with_name("seed-a.py")
            replacement_b = seed.with_name("seed-b.py")
            preserved_a.write_text(source_a, encoding="utf-8")
            replacement_b.write_text(source_b, encoding="utf-8")
            os.replace(replacement_b, seed)
            raw = original_os_read(descriptor, count)
            os.replace(preserved_a, seed)
            return raw
        return original_os_read(descriptor, count)

    monkeypatch.setattr(subject.os, "read", replace_around_descriptor_read)

    with pytest.raises(RuntimeError, match="changed while it was read"):
        subject.local_source_hashes((seed,))

    assert attacked is True
    assert seed.read_text(encoding="utf-8") == source_a


def test_source_changed_before_boundary_recheck_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, _source_a, source_b = _source_tree(tmp_path)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    original_read = subject._read_regular_bytes
    calls = 0

    def mutate_after_first_read(value, *, maximum):
        nonlocal calls
        result = original_read(value, maximum=maximum)
        if Path(value) == seed:
            calls += 1
            if calls == 1:
                _replace(seed, source_b)
        return result

    monkeypatch.setattr(subject, "_read_regular_bytes", mutate_after_first_read)

    with pytest.raises(RuntimeError, match="changed during discovery"):
        subject.local_source_hashes((seed,))


def test_dependency_hidden_during_first_discovery_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, _source_a, _source_b = _source_tree(tmp_path)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    original_module_files = subject._module_files
    hidden_once = False

    def temporarily_hidden(parts, *, root):
        nonlocal hidden_once
        if list(parts) == ["pkg", "dep_a"] and not hidden_once:
            hidden_once = True
            return []
        return original_module_files(parts, root=root)

    monkeypatch.setattr(subject, "_module_files", temporarily_hidden)

    with pytest.raises(RuntimeError, match="changed during discovery"):
        subject.local_source_hashes((seed,))

    assert hidden_once is True
    assert (tmp_path / "pkg" / "dep_a.py").is_file()
