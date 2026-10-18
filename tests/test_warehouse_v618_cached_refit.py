"""A posthoc cache refit cannot overwrite scientific source evidence."""
import pytest

from scripts import refit_warehouse_v618_cached_programs as command


def test_refit_refuses_original_tree_and_preserves_original_bytes(tmp_path, monkeypatch):
    source = tmp_path / "original"
    source.mkdir()
    evidence = source / "old.program.json"
    evidence.write_bytes(b"original evidence")
    monkeypatch.setattr(command, "NativeBackend", lambda: pytest.fail("Actor must not load"))
    for output in (source, source / "replacement"):
        with pytest.raises(RuntimeError, match="isolated output"):
            command.main(["--input-dir", str(source), "--output-dir", str(output)])
    assert evidence.read_bytes() == b"original evidence"


def test_refit_refuses_existing_result_directory_before_loading_an_actor(tmp_path, monkeypatch):
    source, output = tmp_path / "original", tmp_path / "existing"
    source.mkdir()
    output.mkdir()
    artifact = output / "program.json"
    artifact.write_bytes(b"immutable prior program")
    monkeypatch.setattr(command, "NativeBackend", lambda: pytest.fail("Actor must not load"))
    with pytest.raises(RuntimeError, match="must be empty"):
        command.main(["--input-dir", str(source), "--output-dir", str(output)])
    assert artifact.read_bytes() == b"immutable prior program"
