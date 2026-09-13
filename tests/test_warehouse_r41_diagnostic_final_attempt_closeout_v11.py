from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_v11 as subject


def test_contract_records_pre_secret_burn_and_outer_exclusion():
    value = subject.contract()
    assert value["final_attempt_consumed"] is True
    assert value["failure_proven_before_private_salt_read"] is True
    assert value["protected_final_identity_revealed"] is False
    assert value["v10_outer_identity_reuse_permitted"] is False
    assert value["v10_outer_observation_hash_reuse_permitted"] is False
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "open(config_paths[\"private_salt\"]" not in source
    assert "stat(config_paths[\"private_salt\"]" not in source


def test_historical_materializer_calls_public_auth_before_salt():
    raw = subject._historical_materializer_bytes()
    order = subject._call_order(raw)
    assert order.index("_authenticate_public_inputs") < order.index(
        "_read_committed_salt")
    assert sha256(raw).hexdigest() == subject.HISTORICAL_MATERIALIZER_SHA256


def test_hash_projection_reads_only_named_hash_member(tmp_path):
    path = tmp_path / "rows.npz"
    hashes = np.asarray(["a" * 64, "b" * 64], dtype="S64")
    np.savez_compressed(path, observation_hashes=hashes,
                        forbidden=np.asarray([[1.0]], dtype=np.float32))
    assert subject._project_hashes(path) == ["a" * 64, "b" * 64]
