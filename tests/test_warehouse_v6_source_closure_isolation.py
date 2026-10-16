"""The declared source closure must import without the working checkout."""

from __future__ import annotations

import shutil
import subprocess
import sys

from backend.training.warehouse_v6_runner import _SOURCE_FILES, _SOURCE_ROOT


def test_declared_source_closure_imports_from_isolated_copy(tmp_path):
    for relative in set(_SOURCE_FILES):
        source = _SOURCE_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    script = """
import sys
sys.path.insert(0, sys.argv[1])
from backend.training.warehouse_v6_runner import source_closure_sha256
import domains.warehouse.neural_turnbased
import experiments.warehouse.acceptance
from study_v3.release_catalog import warehouse_runtime_source_closure_sha256
assert source_closure_sha256() == warehouse_runtime_source_closure_sha256()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
