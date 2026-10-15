from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import run_warehouse_r41_diagnostic_final_once_v9 as subject


_PATH_ARGUMENTS = (
    "candidate-lock", "actor", "protocol", "runtime-manifest",
    "designation", "failed-outer-closeout", "promoted-v11-closeout",
    "promoted-v11-permanent-registry", "fresh-outer-registry",
    "fresh-outer-registry-report", "prior-outer-hash-projection",
    "outer-hash-projection",
    "outer-hash-projection-receipt", "development-rows",
    "promoted-v11-rows", "program",
    "selector-report", "outer-result", "outer-permanent-registry",
    "permanent-final-registry", "output", "final-materializer-config",
)


def _arguments(tmp_path: Path) -> list[str]:
    values: list[str] = []
    for name in _PATH_ARGUMENTS:
        values.extend(("--" + name, str(tmp_path / name)))
    values.extend(("--expected-candidate-lock-sha256", "a" * 64))
    values.extend(("--expected-promoted-v11-closeout-sha256", "c" * 64))
    values.extend(("--expected-outer-result-sha256", "b" * 64))
    return values


def test_help_lists_every_required_input_without_a_sensitive_default(capsys):
    with pytest.raises(SystemExit) as stopped:
        subject.main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for name in (*_PATH_ARGUMENTS, "expected-candidate-lock-sha256",
                 "expected-promoted-v11-closeout-sha256",
                 "expected-outer-result-sha256"):
        assert "--" + name in help_text
    assert "/Users/" not in help_text
    assert ".config/policylens" not in help_text
    assert "--final-materializer-source" not in help_text


def test_cli_rejects_caller_selected_materializer(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        subject.main([
            *_arguments(tmp_path),
            "--final-materializer-source", str(tmp_path / "copied.py"),
        ])
    assert stopped.value.code == 2
    assert "unrecognized arguments: --final-materializer-source" in (
        capsys.readouterr().err)


def test_main_passes_every_argument_and_scopes_materializer_config(
        tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_final_once(**kwargs):
        captured.update(kwargs)
        captured["materializer_config"] = os.environ.get(
            subject.MATERIALIZER_CONFIG_ENV)
        return {"status": subject.final_once.STATUS_PASSED,
                "attempt_key": "c" * 64}

    monkeypatch.setattr(subject.final_once, "run_final_once", fake_run_final_once)
    monkeypatch.setenv(subject.MATERIALIZER_CONFIG_ENV, "previous-config")
    assert subject.main(_arguments(tmp_path)) == 0
    expected_names = {
        "candidate_lock_path", "expected_candidate_lock_sha256", "actor_path",
        "protocol_path", "runtime_manifest_path", "designation_path",
        "failed_outer_closeout_path", "promoted_v11_closeout_path",
        "expected_promoted_v11_closeout_sha256",
        "permanent_v11_outer_registry", "fresh_outer_registry_path",
        "fresh_outer_registry_report_path", "prior_outer_hash_projection_path",
        "outer_hash_projection_path",
        "outer_hash_projection_receipt_path", "development_rows_path",
        "promoted_v11_rows_path",
        "program_path", "selector_report_path", "outer_result_path",
        "expected_outer_result_sha256", "outer_permanent_registry",
        "permanent_final_registry", "output",
    }
    assert set(captured) == expected_names | {"materializer_config"}
    assert captured["expected_candidate_lock_sha256"] == "a" * 64
    assert captured["expected_promoted_v11_closeout_sha256"] == "c" * 64
    assert captured["expected_outer_result_sha256"] == "b" * 64
    assert captured["candidate_lock_path"] == tmp_path / "candidate-lock"
    assert captured["materializer_config"] == str(
        (tmp_path / "final-materializer-config").absolute())
    assert os.environ[subject.MATERIALIZER_CONFIG_ENV] == "previous-config"
    output = capsys.readouterr().out
    assert '"status":"completed_passed"' in output
    assert '"attempt_key":"' + "c" * 64 + '"' in output
