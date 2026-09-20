"""Canvas-command regression for carried pickup labels; no browser claims."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


def test_warehouse_cargo_badges_match_displayed_orders_and_clear_battery():
    root = Path(__file__).resolve().parents[1]
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    node = os.environ.get("STUDY_TEST_NODE") or shutil.which("node") or (str(bundled) if bundled.exists() else None)
    if not node:
        pytest.skip("Node.js is required for the actual canvas-renderer regression")
    result = subprocess.run([node, str(root / "tests/warehouse_board_badges.cjs")], cwd=root,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "battery separation passed" in result.stdout
