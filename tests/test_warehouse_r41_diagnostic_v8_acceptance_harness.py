"""Self-tests for the v8 HTTP/browser acceptance tools.

Every service used here is the synthetic fixture in this test tree.  The
tests never discover or read a final/holdout release directory.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

from scripts import check_warehouse_r41_diagnostic_v8_http as http_check
from scripts import serve_warehouse_r41_diagnostic_v8_https as https_server


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SHA = "8" * 64
MANIFEST_SHA = "9" * 64
ACTOR_SHA = "a" * 64


def _port():
    value = socket.socket()
    value.bind(("127.0.0.1", 0))
    port = value.getsockname()[1]
    value.close()
    return port


def _fixture(tmp_path, database, certificate, key, port):
    process = subprocess.Popen(
        [sys.executable, "-m",
         "tests.warehouse_r41_diagnostic_v8_https_fixture",
         "--database", str(database), "--certificate", str(certificate),
         "--private-key", str(key), "--port", str(port)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    line = process.stdout.readline().strip()
    if '"ready"' not in line:
        error = process.stderr.read()
        process.kill(); process.wait()
        raise AssertionError("synthetic HTTPS fixture did not start: " + line + error)
    return process


def _stop(process):
    process.terminate()
    process.wait(timeout=10)


def _http_args(base, database, certificate, output, phase):
    return [
        "--execute", "--phase", phase, "--base", base,
        "--database", str(database), "--ca-certificate", str(certificate),
        "--expected-package-sha256", PACKAGE_SHA,
        "--expected-manifest-sha256", MANIFEST_SHA,
        "--expected-actor-sha256", ACTOR_SHA,
        "--output", str(output), "--allow-synthetic-fixture",
    ]


def _node_runtime():
    candidates = [shutil.which("node")]
    dependency_root = (Path.home() / ".cache" / "codex-runtimes" /
                       "codex-primary-runtime" / "dependencies" / "node")
    candidates.append(str(dependency_root / "bin" / "node"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            modules = dependency_root / "node_modules"
            environment = dict(os.environ)
            if modules.is_dir():
                old = environment.get("NODE_PATH")
                environment["NODE_PATH"] = (str(modules) if not old
                                              else str(modules) + os.pathsep + old)
            probe = subprocess.run(
                [candidate, "-e", "require('playwright')"], cwd=ROOT,
                env=environment, capture_output=True)
            if probe.returncode == 0:
                return candidate, environment
    return None, None


def test_http_checker_has_no_runtime_or_final_artifact_dependency():
    source = (ROOT / "scripts/check_warehouse_r41_diagnostic_v8_http.py").read_text()
    tree = ast.parse(source)
    imports = {node.names[0].name for node in ast.walk(tree)
               if isinstance(node, ast.Import)}
    imports.update(node.module for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(name.startswith(("backend", "env", "ui")) for name in imports)
    assert "final/" not in source and "holdout/" not in source
    assert http_check.CONTEXT_VERSION.endswith(".v8")
    browser = (ROOT / "scripts/check_warehouse_r41_diagnostic_v8_browser.cjs").read_text()
    fixture = (ROOT / "tests/warehouse_r41_diagnostic_v8_https_fixture.py").read_text()
    assert "final/" not in browser and "holdout/" not in browser
    assert "final/" not in fixture and "holdout/" not in fixture
    assert "authenticated_tls_probe" in browser


def test_generated_certificate_is_private_and_loopback_https_is_enforced(tmp_path):
    certificate, key = https_server.generate_certificate(tmp_path / "tls")
    assert certificate.is_file() and key.is_file()
    assert key.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="loopback HTTPS"):
        https_server.https_server(
            object(), host="0.0.0.0", port=443,
            public_origin="https://example.com", certificate=certificate,
            private_key=key)


def test_synthetic_real_https_http_flow_survives_process_restart(tmp_path):
    certificate, key = https_server.generate_certificate(tmp_path / "tls")
    database = tmp_path / "synthetic_diagnostic_v8_http_qa.sqlite3"
    output = tmp_path / "http_report"
    port = _port(); base = f"https://127.0.0.1:{port}"
    process = _fixture(tmp_path, database, certificate, key, port)
    try:
        assert http_check.main(_http_args(
            base, database, certificate, output, "before")) == 0
    finally:
        _stop(process)
    process = _fixture(tmp_path, database, certificate, key, port)
    try:
        assert http_check.main(_http_args(
            base, database, certificate, output, "after")) == 0
    finally:
        _stop(process)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed_synthetic_harness_validation"
    assert report["release_acceptance_eligible"] is False
    assert report["exact_same_service_restart_restored"] is True
    assert report["lost_response_retry_idempotent"] is True
    assert {tuple(cell) for cell in report["four_cell_allocation"]} == {
        ("A", "XY"), ("A", "YX"), ("B", "XY"), ("B", "YX")}


def test_synthetic_playwright_flow_covers_matrix_and_intermediate_motion(tmp_path):
    node, environment = _node_runtime()
    if node is None:
        pytest.skip("Playwright Node runtime is unavailable")
    if not (Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").is_file()
            or (Path.home() / "Library/Caches/ms-playwright").is_dir()):
        pytest.skip("No isolated Chromium/Chrome executable")
    certificate, key = https_server.generate_certificate(tmp_path / "tls")
    database = tmp_path / "synthetic_diagnostic_v8_browser_qa.sqlite3"
    output = tmp_path / "browser_report"
    port = _port(); base = f"https://127.0.0.1:{port}"
    process = _fixture(tmp_path, database, certificate, key, port)
    try:
        completed = subprocess.run([
            node, "scripts/check_warehouse_r41_diagnostic_v8_browser.cjs",
            "--execute", "--base", base, "--database", str(database),
            "--ca-certificate", str(certificate),
            "--expected-package-sha256", PACKAGE_SHA,
            "--expected-manifest-sha256", MANIFEST_SHA,
            "--expected-actor-sha256", ACTOR_SHA,
            "--output", str(output), "--allow-synthetic-fixture",
        ], cwd=ROOT, env=environment, capture_output=True, text=True,
           timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    finally:
        _stop(process)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed_synthetic_browser_harness_validation"
    assert report["release_acceptance_eligible"] is False
    assert report["frontend_timing"]["declared_motion_duration_ms"] == 380
    assert report["authenticated_tls_probe"]["protocol"] in {"TLSv1.2", "TLSv1.3"}
    assert {tuple((row["condition"], row["task_order"]))
            for row in report["participants"]} == {
                ("A", "XY"), ("A", "YX"), ("B", "XY"), ("B", "YX")}
    assert {row["kind"] for row in report["motion_reports"]
            if row["moved"]} >= {"tutorial_advance", "action"}
    assert all(row["interior_samples"] >= 2
               for row in report["motion_reports"] if row["moved"])
    assert len(report["screenshots"]) >= 16
