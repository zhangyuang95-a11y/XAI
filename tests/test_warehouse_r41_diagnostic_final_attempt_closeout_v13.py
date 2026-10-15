from __future__ import annotations

from hashlib import sha256
import inspect
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_v13 as subject
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training.warehouse_native_common import digest


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _arrays(tag: int, count: int = 2) -> dict[str, np.ndarray]:
    """Small schema-correct arrays for deterministic bundle unit tests."""
    result: dict[str, np.ndarray] = {}
    for name in rows_api._FIELDS:
        if name == "observations":
            result[name] = np.full((count, 3), tag, dtype=np.float32)
        elif name == "probabilities":
            result[name] = np.tile(
                np.asarray([[0.1, 0.2, 0.3, 0.15, 0.25]], dtype=np.float32),
                (count, 1),
            )
        elif name in {
            "observation_hashes", "physical_hashes", "source_state_hashes",
            "scene_fingerprints", "episode_ids", "anchor_ids", "kinds",
        }:
            text = {
                "observation_hashes": chr(96 + tag) * 64,
                "physical_hashes": "",
                "source_state_hashes": HEX_C,
                "scene_fingerprints": HEX_B,
                "episode_ids": f"e{tag}",
                "anchor_ids": f"a{tag}",
                "kinds": "ordinary",
            }[name]
            result[name] = np.asarray([text.encode("ascii")] * count)
        elif name in {"split_validation", "submitted_equal", "trajectory_done"}:
            result[name] = np.ones(count, dtype=np.bool_)
        elif name in {"weights"}:
            result[name] = np.ones(count, dtype=np.float32)
        elif name in {"frames", "episode_ids"}:
            result[name] = np.arange(count, dtype=np.int32)
        else:
            result[name] = np.full(count, tag, dtype=np.uint8)
    return result


def test_public_identity_registry_strips_private_scene_material() -> None:
    scenes = [
        {
            "id": f"private-final-{index}",
            "batch_index": index,
            "family_id": "conflict_family_01",
            "seed": 100 + index,
            "fingerprint": ("a" if index == 0 else "b") * 64,
            "snapshot": {"rng_state": "must-not-escape"},
            "rng_state": "must-not-escape",
            "hidden_rank": index,
        }
        for index in range(2)
    ]
    value = subject._public_identity_registry(scenes, expected_count=2)
    assert value["identities"] == [
        {
            "batch_index": 0, "family_id": "conflict_family_01",
            "seed": 100, "fingerprint": HEX_A,
        },
        {
            "batch_index": 1, "family_id": "conflict_family_01",
            "seed": 101, "fingerprint": HEX_B,
        },
    ]
    encoded = repr(value["identities"])
    assert "snapshot" not in encoded
    assert "rng_state" not in encoded
    assert "hidden_rank" not in encoded
    assert value["identities_sha256"] == digest(value["identities"])


def test_three_source_bundle_is_deterministic_and_marks_development() -> None:
    v11, v12, burned = _arrays(1), _arrays(2), _arrays(3)
    combined = subject._combine_rows(v11, v12, burned)
    assert set(combined) == set(rows_api._FIELDS)
    assert all(len(value) == 6 for value in combined.values())
    assert np.array_equal(combined["action_indices"], [1, 1, 2, 2, 3, 3])
    assert np.all(combined["split_validation"])
    assert np.array_equal(combined["observations"][:, 0], [1, 1, 2, 2, 3, 3])

    again = subject._combine_rows(v11, v12, burned)
    assert subject._arrays_digest(combined) == subject._arrays_digest(again)


def test_bundle_rejects_schema_dtype_and_non_development_rows() -> None:
    left, middle, right = _arrays(1), _arrays(2), _arrays(3)
    del left["weights"]
    with pytest.raises(ValueError, match="schema"):
        subject._combine_rows(left, middle, right)

    left = _arrays(1)
    middle["action_indices"] = middle["action_indices"].astype(np.int16)
    with pytest.raises(ValueError, match="dtype"):
        subject._combine_rows(left, middle, right)

    middle = _arrays(2)
    right["split_validation"][0] = False
    with pytest.raises(ValueError, match="development"):
        subject._combine_rows(left, middle, right)


def test_projection_contains_only_one_way_hashes() -> None:
    arrays = _arrays(1)
    arrays["observation_hashes"] = np.asarray(
        [HEX_B.encode("ascii"), HEX_A.encode("ascii"), HEX_B.encode("ascii")]
    )
    # Keep every companion field row-aligned for this helper's contract.
    for name, value in list(arrays.items()):
        if name != "observation_hashes":
            arrays[name] = np.concatenate((value, value[:1]), axis=0)
    projection = subject._observation_projection(
        arrays, version="test-projection.v1", source="burned-final-recollection"
    )
    assert projection["ordered_observation_hashes"] == [HEX_B, HEX_A, HEX_B]
    assert projection["outer_observation_hashes"] == [HEX_A, HEX_B]
    assert projection["outer_observation_hashes_sha256"] == digest([HEX_A, HEX_B])
    serialized = repr(projection)
    for forbidden in ("observations", "actions", "probabilities", "labels"):
        assert forbidden not in serialized


def test_source_contract_cannot_select_or_reuse_protected_final() -> None:
    source = inspect.getsource(subject)
    signature = inspect.signature(subject.create_closeout)
    assert "private_salt" not in signature.parameters
    assert "materializer_config" not in signature.parameters
    assert "run_final_once" not in source
    assert "_run_materializer" not in source
    contract = subject.contract()
    assert contract["same_final_attempt_retry_permitted"] is False
    assert contract["protected_salt_reused"] is False
    assert contract["post_closeout_deterministic_development_recollection"] is True
    assert contract["combined_promoted_source_order"] == [
        "consumed_v11_outer", "consumed_v12_outer", "burned_v12_final",
    ]


def test_write_exclusive_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    subject._write_exclusive(target, b"first")
    with pytest.raises(FileExistsError):
        subject._write_exclusive(target, b"second")
    assert target.read_bytes() == b"first"


def test_arrays_digest_binds_dtype_shape_and_bytes() -> None:
    arrays = _arrays(1)
    original = subject._arrays_digest(arrays)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["action_indices"][0] += 1
    assert subject._arrays_digest(changed) != original
    summary = {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(value.shape),
            "sha256": sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    assert original == digest(summary)
