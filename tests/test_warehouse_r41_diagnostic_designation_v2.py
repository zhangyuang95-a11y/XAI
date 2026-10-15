from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import pytest

from backend.training import warehouse_r41_diagnostic_designation as designation_v1
from backend.training import warehouse_r41_diagnostic_designation_v2 as subject
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as binding
from backend.training.warehouse_native_common import file_hash


ROOT = Path(__file__).resolve().parents[1]
DESIGNATION_V1 = (
    ROOT / "output/warehouse_native/r41_diagnostic_release_20260912_v4/"
    "diagnostic_actor_designation.json"
)
DESIGNATION_V2 = (
    ROOT / "output/warehouse_native/r41_diagnostic_designation_v2_sourceclosure2_20260913/"
    "diagnostic_actor_designation.json"
)
COMPONENTS = {
    "actor": ROOT / (
        "output/warehouse_native/r41_active_2m_20260911/boundaries/"
        "step_2000000/actor.npz"
    ),
    "protocol": ROOT / "output/warehouse_native/r41_active_2m_20260911/protocol.json",
    "training_ledger": ROOT / (
        "output/warehouse_native/r41_failure_closeout_20260911_194133-ledger/"
        "training_ledger.json"
    ),
    "dual_evaluation": ROOT / (
        "output/warehouse_native/r41_active_2m_20260911/boundaries/"
        "step_2000000/evaluation/paired_dual_report.json"
    ),
    "failure_closeout": ROOT / (
        "output/warehouse_native/r41_failure_closeout_20260911_194133/"
        "failure_closeout.json"
    ),
}


def _require_real_chain() -> None:
    missing = [path for path in [DESIGNATION_V1, DESIGNATION_V2, *COMPONENTS.values()]
               if not path.is_file()]
    if missing:
        pytest.skip("frozen designation evidence is unavailable")


def test_real_v2_reauthenticates_same_actor_without_release_cycle() -> None:
    _require_real_chain()
    saved = subject.read_saved_designation(
        DESIGNATION_V2,
        expected_sha256=file_hash(DESIGNATION_V2),
        components=COMPONENTS,
    )
    predecessor = subject.read_superseded_designation(
        DESIGNATION_V1, components=COMPONENTS)

    assert file_hash(DESIGNATION_V2) == (
        "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
    )
    assert saved["supersedes_designation_sha256"] == file_hash(DESIGNATION_V1)
    assert saved["self_path"] == DESIGNATION_V2.relative_to(ROOT).as_posix()
    assert saved["bindings"]["actor_sha256"] == predecessor["bindings"]["actor_sha256"]
    assert saved["bindings"]["actor_parameters_sha256"] == (
        predecessor["bindings"]["actor_parameters_sha256"]
    )
    assert saved["bindings"]["actor_sha256"] == subject.EXPECTED_ACTOR_SHA256
    assert saved["bindings"]["actor_parameters_sha256"] == (
        subject.EXPECTED_ACTOR_PARAMETERS_SHA256
    )
    assert saved["sources"] == subject.source_closure()
    assert not any(
        token in path for path in saved["sources"]
        for token in ("admission", "release", "preflight")
    )
    assert binding.read_bound_designation(DESIGNATION_V2) == saved


def test_v1_live_reader_is_retired_but_exact_predecessor_bytes_remain_valid() -> None:
    _require_real_chain()
    with pytest.raises(ValueError, match="sources differs from live evidence"):
        designation_v1.read_saved_designation(
            DESIGNATION_V1,
            expected_sha256=subject.SUPERSEDES_DESIGNATION_SHA256,
            components=COMPONENTS,
        )
    assert subject.read_superseded_designation(
        DESIGNATION_V1, components=COMPONENTS)["bindings"]["actor_sha256"] == (
            subject.EXPECTED_ACTOR_SHA256
        )


@pytest.mark.parametrize("changed", subject.ARTIFACT_NAMES)
def test_any_fixed_component_tamper_is_rejected(changed: str) -> None:
    _require_real_chain()
    with tempfile.TemporaryDirectory(prefix=".designation-v2-test-", dir=ROOT) as directory:
        copied = dict(COMPONENTS)
        source = COMPONENTS[changed]
        target = Path(directory) / (changed + source.suffix)
        shutil.copyfile(source, target)
        with target.open("ab") as stream:
            stream.write(b"\n")
        copied[changed] = target
        with pytest.raises(ValueError, match="input identity differs"):
            subject.validate_components(copied)


def test_saved_reader_rejects_wrong_self_path_even_with_valid_json() -> None:
    _require_real_chain()
    value = json.loads(DESIGNATION_V2.read_text(encoding="utf-8"))
    value["self_path"] = "output/warehouse_native/wrong/designation.json"
    with tempfile.TemporaryDirectory(prefix=".designation-v2-test-", dir=ROOT) as directory:
        copy = Path(directory) / "designation.json"
        copy.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="Exact cycle-free"):
            subject.read_saved_designation(
                copy,
                expected_sha256=file_hash(copy),
                components=COMPONENTS,
            )
