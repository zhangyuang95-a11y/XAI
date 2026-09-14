from __future__ import annotations

from pathlib import Path
import inspect
import re

import pytest

from backend.training import warehouse_r41_diagnostic_admission_v10 as admission
from backend.training import warehouse_r41_diagnostic_final_once_v13 as final_v13
from backend.training import warehouse_r41_diagnostic_release_receipt_v10 as receipt
from backend.training import warehouse_r41_diagnostic_study_materials_v10 as study
from backend.training.warehouse_native_common import file_hash
from scripts import build_warehouse_r41_diagnostic_admission_v10 as admission_cli
from scripts import build_warehouse_r41_diagnostic_compact_program_v10 as compact_cli
from scripts import build_warehouse_r41_diagnostic_study_materials_v10 as study_cli
from scripts import preflight_warehouse_r41_diagnostic_render_v10 as preflight
from ui import warehouse_alignment_online_server as server
from ui import warehouse_alignment_r41_diagnostic_release_v9 as release


def _write(path: Path, raw: bytes = b"x") -> Path:
    path.write_bytes(raw)
    return path


def test_v10_is_append_only_adapter_for_v13_evidence_and_v9_runtime():
    assert admission.VERSION == "warehouse-r41-diagnostic-admission.v10"
    assert admission.outer_api.VERSION.startswith(
        "warehouse-r41-diagnostic-rcpd-v13-")
    assert admission.final_api.VERSION == "warehouse-r41-diagnostic-final-once.v13"
    assert study.outer_api is admission.outer_api
    assert release.VERSION == "warehouse-r41-diagnostic-online-release.v9"
    assert release.PUBLIC_RELEASE_VERSION == "r4.1-diagnostic"
    assert release.PILOT_CLASS == "internal_diagnostic"
    assert release.ANIMATION_DURATION_MS == 380
    assert receipt.release is release
    assert preflight.RELEASE_MODULE == server.R41_DIAGNOSTIC_RELEASE_MODULE_V9
    assert release.assemble_from_v10_admitted_components is not None


def test_legacy_v9_assembler_stays_callable_and_outside_frozen_final_closure():
    legacy = inspect.signature(release.assemble_from_admitted_components).parameters
    adapter = inspect.signature(
        release.assemble_from_v10_admitted_components).parameters
    assert "promoted_v11_permanent_registry" in legacy
    assert "permanent_promotion_closeout_registry" not in legacy
    assert "permanent_promotion_closeout_registry" in adapter
    assert "promoted_v11_permanent_registry" not in adapter
    _path, materializer_sources = final_v13._official_materializer_binding()
    assert "ui/warehouse_alignment_r41_diagnostic_release_v9.py" not in (
        materializer_sources)


def test_v10_admission_binds_v13_closeout_and_projection_parity():
    required = {
        "promotion_closeout", "promotion_identity_registry",
        "promotion_observation_projection", "combined_promoted_projection",
        "promoted_burned_final_rows", "combined_promoted_rows",
        "development_rows", "outer_collection_report", "outer_rows",
        "final_projection_parity",
    }
    assert required <= set(admission.ARTIFACT_NAMES)
    assert {"projection_parity_file_sha256",
            "projection_parity_content_sha256"} <= study._FINAL_BINDINGS


def test_promotion_authentication_requires_exact_sibling_bundle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    api = admission.promoted_closeout_api
    names = {
        "promotion_closeout": api.RECEIPT_NAME,
        "promotion_identity_registry": api.IDENTITY_NAME,
        "promotion_observation_projection": api.PROMOTED_PROJECTION_NAME,
        "combined_promoted_projection": api.COMBINED_PROJECTION_NAME,
        "promoted_burned_final_rows": api.PROMOTED_ROWS_NAME,
        "combined_promoted_rows": api.COMBINED_ROWS_NAME,
    }
    files = {key: _write(tmp_path / name, key.encode())
             for key, name in names.items()}
    hashes = {key: file_hash(path) for key, path in files.items()}
    lock = {
        "combined_promoted_rows_sha256": hashes["combined_promoted_rows"],
        "promotion_closeout_sha256": hashes["promotion_closeout"],
    }
    saved = {
        "combined_promoted_development": {
            "rows_sha256": hashes["combined_promoted_rows"],
            "rows_semantic_sha256": "a" * 64,
        }
    }
    calls = []
    monkeypatch.setattr(
        api, "read_saved_closeout_public",
        lambda path, **kwargs: calls.append((Path(path), kwargs)) or saved)
    value, semantic = admission._authenticate_promotion(
        files, hashes, lock, permanent_registry=tmp_path)
    assert value is saved
    assert semantic == "a" * 64
    assert calls == [(files["promotion_closeout"], {
        "expected_closeout_sha256": hashes["promotion_closeout"],
        "permanent_closeout_registry": tmp_path,
    })]

    files["promotion_identity_registry"] = _write(
        tmp_path / "wrong-name.json", b"identity")
    with pytest.raises(ValueError, match="sibling layout"):
        admission._authenticate_promotion(
            files, {**hashes,
                    "promotion_identity_registry": file_hash(
                        files["promotion_identity_registry"])},
            lock, permanent_registry=tmp_path)


def test_v10_cli_and_preflight_use_v13_promotion_boundary():
    admission_options = {
        option for action in admission_cli.parser()._actions
        for option in action.option_strings
    }
    study_options = {
        option for action in study_cli.parser()._actions
        for option in action.option_strings
    }
    for name in admission.ARTIFACT_NAMES:
        assert "--" + name.replace("_", "-") in admission_options
    assert "--permanent-promotion-closeout-registry" in admission_options
    assert "--combined-promoted-rows" in study_options
    assert "--expected-combined-promoted-rows-sha256" in study_options
    assert "--promotion-closeout" in study_options
    assert "--expected-promotion-closeout-sha256" in study_options
    assert "--permanent-promotion-closeout-registry" in study_options
    compact_options = {
        option for action in compact_cli.parser()._actions
        for option in action.option_strings
    }
    for name in ("program", "development-rows", "combined-promoted-rows",
                 "outer-rows", "final-rows"):
        assert "--" + name in compact_options
        assert "--expected-" + name + "-sha256" in compact_options
    assert preflight.RELEASE_MODULE.endswith("_v9")
    assert "--release-module " + preflight.RELEASE_MODULE in preflight.START_COMMAND


def test_release_source_closure_contains_v10_boundary():
    sources = release.release_sources()
    assert "ui/warehouse_alignment_r41_diagnostic_release_v9.py" in sources
    assert "ui/warehouse_alignment_online_server.py" in sources
    assert "backend/warehouse_r41_diagnostic_online_explanation_v9.py" in sources
    assert not any("holdout_salt.bin" in name for name in sources)


def test_v10_receipt_requires_exact_v10_admission_parent():
    admission_value = {
        "content_sha256": "b" * 64,
        "bindings": {"actor_sha256": "c" * 64},
        "gates": {"final_passed": True},
    }
    admission_sha = "a" * 64
    exact = {
        "version": admission.VERSION,
        "status": admission.STATUS,
        "admission_sha256": admission_sha,
        "admission_content_sha256": admission_value["content_sha256"],
        "bindings_sha256": receipt.digest(admission_value["bindings"]),
        "gates_sha256": receipt.digest(admission_value["gates"]),
    }
    receipt._require_exact_admission_parent(
        {"parent": exact}, admission_value, admission_sha)

    for field, replacement in (
        ("version", "warehouse-r41-diagnostic-admission.v9"),
        ("status", "admitted_internal_diagnostic_v9"),
        ("gates_sha256", "d" * 64),
    ):
        forged = {**exact, field: replacement}
        with pytest.raises(ValueError, match="exact v10 admission"):
            receipt._require_exact_admission_parent(
                {"parent": forged}, admission_value, admission_sha)


def test_v10_preflight_rejects_render_start_without_explicit_release_module(
        tmp_path: Path):
    package_sha, manifest_sha = "a" * 64, "b" * 64
    text = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(
        encoding="utf-8")
    text = re.sub(
        r"(?m)^    startCommand:.*$",
        "    startCommand: " + preflight.START_COMMAND,
        text,
    )
    text = re.sub(
        r"(?m)(^      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256\n"
        r"        value: )[0-9a-f]{64}$", r"\g<1>" + package_sha, text)
    text = re.sub(
        r"(?m)(^      - key: WAREHOUSE_RELEASE_MANIFEST_SHA256\n"
        r"        value: )[0-9a-f]{64}$", r"\g<1>" + manifest_sha, text)
    candidate = tmp_path / "render.v10.yaml"
    candidate.write_text(text, encoding="utf-8")
    assert preflight.check_render_yaml(
        candidate, package_sha256=package_sha,
        manifest_sha256=manifest_sha)["release_module"] == preflight.RELEASE_MODULE

    candidate.write_text(text.replace(
        " --release-module " + preflight.RELEASE_MODULE, ""), encoding="utf-8")
    with pytest.raises(ValueError, match="single free service"):
        preflight.check_render_yaml(
            candidate, package_sha256=package_sha,
            manifest_sha256=manifest_sha)
