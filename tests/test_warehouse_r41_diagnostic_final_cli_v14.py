from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_final_materializer_v14 as materializer
from scripts import build_warehouse_r41_diagnostic_final_materializer_config_v14 as config_cli
from scripts import run_warehouse_r41_diagnostic_final_once_v14 as run_cli


def _paths(tmp_path: Path) -> dict[str, Path]:
    values = {name: tmp_path / name for name in config_cli.PATH_FIELDS}
    values["private_salt"] = materializer.PRIVATE_SALT_PATH
    return values


def test_config_v4_has_exact_38_paths_without_accessing_inputs(
        tmp_path, monkeypatch):
    values = _paths(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("configured inputs must not be accessed")

    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    config = config_cli.create_config(values)
    assert config["version"] == materializer.CONFIG_VERSION
    assert set(config["paths"]) == materializer._CONFIG_PATH_FIELDS
    assert len(config["paths"]) == 38
    assert config["paths"]["private_salt"] == str(
        materializer.PRIVATE_SALT_PATH)


def test_config_v4_is_created_exclusively(tmp_path):
    value = config_cli.create_config(_paths(tmp_path))
    output = tmp_path / "materializer_config.json"
    assert config_cli.write_exclusive(output, value) == output
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        config_cli.write_exclusive(output, value)


def _run_args(tmp_path: Path) -> list[str]:
    path_options = (
        "candidate-lock", "actor", "protocol", "runtime-manifest",
        "designation", "failed-outer-closeout", "promotion-closeout",
        "permanent-promotion-closeout-registry", "fresh-outer-registry",
        "fresh-outer-registry-report", "prior-outer-hash-projection",
        "outer-hash-projection", "outer-hash-projection-receipt",
        "development-rows", "combined-promoted-rows", "program",
        "selector-report", "outer-result", "outer-permanent-registry",
        "timeout-closeout", "permanent-timeout-closeout-registry",
        "candidate-universe", "timing-calibration",
        "permanent-final-registry", "output", "final-materializer-config",
    )
    sha_options = (
        "expected-candidate-lock-sha256",
        "expected-promotion-closeout-sha256",
        "expected-outer-result-sha256",
        "expected-timeout-closeout-sha256",
        "expected-candidate-universe-sha256",
        "expected-timing-calibration-sha256",
    )
    result: list[str] = []
    for name in path_options:
        result.extend(("--" + name, str(tmp_path / name)))
    for index, name in enumerate(sha_options):
        result.extend(("--" + name, f"{index + 1:064x}"))
    return result


def test_v14_cli_forwards_every_new_public_binding_and_config_opaquely(
        tmp_path, monkeypatch):
    argv = _run_args(tmp_path)
    seen = {}
    expected_parameters = set(inspect.signature(
        run_cli.final_once.run_final_once).parameters)

    def run_final_once(**kwargs):
        seen.update(kwargs)
        assert Path(materializer.os.environ[materializer.CONFIG_ENV]) == (
            tmp_path / "final-materializer-config")
        assert not (tmp_path / "final-materializer-config").exists()
        return {"status": run_cli.final_once.STATUS_PASSED,
                "attempt_key": "a" * 64}

    monkeypatch.delenv(materializer.CONFIG_ENV, raising=False)
    monkeypatch.setattr(run_cli.final_once, "run_final_once", run_final_once)
    assert run_cli.main(argv) == 0
    assert materializer.CONFIG_ENV not in materializer.os.environ
    assert set(seen) == expected_parameters
    assert seen["timeout_closeout_path"] == tmp_path / "timeout-closeout"
    assert seen["candidate_universe_path"] == tmp_path / "candidate-universe"
    assert seen["timing_calibration_path"] == tmp_path / "timing-calibration"


def test_cli_parser_requires_new_recovery_evidence():
    options = {option for action in run_cli.parser()._actions
               for option in action.option_strings}
    assert {
        "--timeout-closeout", "--expected-timeout-closeout-sha256",
        "--permanent-timeout-closeout-registry", "--candidate-universe",
        "--expected-candidate-universe-sha256", "--timing-calibration",
        "--expected-timing-calibration-sha256", "--final-materializer-config",
    } <= options
    assert not any("salt" in option for option in options)
