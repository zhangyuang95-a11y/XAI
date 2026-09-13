from __future__ import annotations

from hashlib import sha256
import inspect
import json
import lzma
import struct

import numpy as np
import pytest

from backend.training.warehouse_native_common import canonical, digest
from backend import warehouse_r41_diagnostic_compact_public_tree_v9 as subject
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    VERSION as TREE_VERSION,
)
from backend.warehouse_r41_diagnostic_compact_public_tree_v9 import (
    MAGIC,
    MAX_BASE64_BYTES,
    MAX_DECOMPRESSED_BYTES,
    MAX_SECRET_FILE_BYTES,
    SECRET_FILE_RESERVE_BYTES,
    CompactProgramError,
    base64_encoded_size,
    decode_program,
    encode_program,
    program_json_bytes,
)
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    AGGREGATION,
    GROUPS,
    VERSION as PROGRAM_VERSION,
)


def _features():
    required = sorted(R41DiagnosticPublicRelationsV9._required_names())
    return tuple((*required, *(f"public.compact_test.{index}"
                  for index in range(197 - len(required)))))


def _component(feature_names, left_values, right_values, name):
    trees = []
    for output, (left, right) in enumerate(zip(left_values, right_values)):
        trees.append({
            "iteration": 0,
            "output_index": output,
            "nodes": [
                {"kind": "split", "feature_index": 0, "threshold": 0.25,
                 "missing_go_to_left": True, "left": 1, "right": 2},
                {"kind": "leaf", "value": left},
                {"kind": "leaf", "value": right},
            ],
        })
    return {
        "version": TREE_VERSION,
        "kind": MODEL_KIND,
        "feature_names": list(feature_names),
        "classes": list(range(5)),
        "action_names": ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"],
        "output_kind": "multiclass_logits",
        "baseline": [0.0] * 5,
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1,
        "trees": trees,
        "metadata": {"component": name, "public_only": True},
    }


def _source_bytes():
    names = _features()
    relations = R41DiagnosticPublicRelationsV9(names)
    # The two leading left leaves are closer than one int16 interval once the
    # +/-100 right leaves set the component range.  Unpatched quantization
    # changes the left-branch argmax from DOWN to UP.
    base = _component(
        relations.feature_names,
        [0.0004, 0.0005, -10.0, -11.0, -12.0],
        [-100.0, 100.0, -20.0, -21.0, -22.0],
        "base",
    )
    zero = _component(relations.feature_names, [0.0] * 5, [0.0] * 5, "zero")
    payload = {
        "version": PROGRAM_VERSION,
        "relations": relations.contract(),
        "action_names": ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"],
        "aggregation": AGGREGATION,
        "base": base,
        "specialists": [{
            "group": group,
            "route": {"feature_name": "derived.critical." + group,
                      "operator": ">", "threshold": 0.5},
            "mix_weight": 0.0,
            "program": zero,
        } for group in GROUPS],
        "metadata": {"runtime_controller": "native_neural_actor_only"},
    }
    return (canonical(payload) + "\n").encode(), names


def _source_bindings(raw):
    payload = json.loads(raw)
    return {
        "expected_source_program_sha256": sha256(raw).hexdigest(),
        "expected_actor_feature_names_sha256": digest(
            payload["relations"]["base_feature_names"]),
        "expected_public_feature_contract_sha256": digest(payload["relations"]),
    }


def _roundtrip():
    raw, names = _source_bytes()
    observations = np.zeros((3, 197), dtype=np.float32)
    observations[1, 0] = 1.0
    observations[2, 0] = -1.0
    compact, report = encode_program(raw, {"development": observations})
    program, header = decode_program(
        compact,
        expected_compact_sha256=sha256(compact).hexdigest(),
        **_source_bindings(raw),
    )
    return raw, names, observations, compact, report, program, header


def test_compact_tree_roundtrip_preserves_audited_actions_topology_and_trace():
    raw, names, observations, compact, report, program, header = _roundtrip()
    from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
        R41DiagnosticPublicTreeProgramV9,
    )
    source = R41DiagnosticPublicTreeProgramV9.from_json(raw.decode())
    assert np.array_equal(source.predict_batch(observations),
                          program.predict_batch(observations))
    assert report["audit"]["exact_action_parity"] is True
    assert report["audit"]["patch_count"] > 0
    assert report["audit"]["row_count"] == 3
    assert 0 <= report["audit"]["maximum_absolute_probability_error"] <= 1
    assert header["source"]["content_sha256"] == digest(source.to_dict())
    assert header["source"]["actor_feature_names_sha256"] == digest(
        list(source.base_feature_names))
    assert header["source"]["public_feature_contract_sha256"] == digest(
        source.relations.contract())
    assert report["actor_feature_names_sha256"] == header["source"][
        "actor_feature_names_sha256"]
    assert report["public_feature_contract_sha256"] == header["source"][
        "public_feature_contract_sha256"]
    original = source.to_dict()["base"]["trees"][0]["nodes"][0]
    restored = program.to_dict()["base"]["trees"][0]["nodes"][0]
    assert restored == original  # exact threshold and topology
    observation = {name: 0.0 for name in names}
    trace = program.trace(observation)
    assert trace["prediction"] == "DOWN"
    assert trace["base"]["program_trace"]["trees"][0]["path"]
    assert len(compact) < len(raw)
    assert program_json_bytes(program).endswith(b"\n")
    assert len(program.feature_names) == 668
    assert report["runtime_action_override"] is False
    assert report["tree_controls_runtime"] is False
    assert report["neural_actor_modified"] is False
    assert report["transport"]["base64_bytes"] == base64_encoded_size(len(compact))
    assert report["transport"]["base64_bytes"] <= MAX_BASE64_BYTES
    assert report["transport"]["reserved_secret_file_bytes"] \
        >= SECRET_FILE_RESERVE_BYTES
    threshold_record = next(
        row for row in header["arrays"] if row["name"] == "c0.threshold_indices")
    assert threshold_record["dtype"] == "<u4"


def test_compact_tree_reader_rejects_hash_drift_trailing_stream_and_bomb():
    raw, _, _, compact, _, _, _ = _roundtrip()
    with pytest.raises(CompactProgramError, match="artifact hash"):
        decode_program(
            compact,
            expected_compact_sha256="0" * 64,
            **_source_bindings(raw),
        )
    trailing = compact + lzma.compress(b"extra")
    with pytest.raises(CompactProgramError, match="decompressed size"):
        decode_program(
            trailing,
            expected_compact_sha256=sha256(trailing).hexdigest(),
            **_source_bindings(raw),
        )
    bomb = lzma.compress(b"x" * (MAX_DECOMPRESSED_BYTES + 1))
    with pytest.raises(CompactProgramError, match="decompressed size"):
        decode_program(
            bomb,
            expected_compact_sha256=sha256(bomb).hexdigest(),
            **_source_bindings(raw),
        )


def test_compact_tree_reader_rejects_authenticated_array_tamper():
    raw, _, _, compact, _, _, _ = _roundtrip()
    plain = bytearray(lzma.decompress(compact))
    header_size = struct.unpack("<I", plain[len(MAGIC):len(MAGIC) + 4])[0]
    array_start = len(MAGIC) + 4 + header_size
    plain[array_start] ^= 1
    tampered = lzma.compress(bytes(plain), preset=9 | lzma.PRESET_EXTREME)
    with pytest.raises(CompactProgramError, match="array hash"):
        decode_program(
            tampered,
            expected_compact_sha256=sha256(tampered).hexdigest(),
            **_source_bindings(raw),
        )


def test_compact_tree_reader_rejects_authenticated_public_feature_binding_tamper():
    raw, _, _, compact, _, _, _ = _roundtrip()
    plain = bytearray(lzma.decompress(compact))
    header_size = struct.unpack("<I", plain[len(MAGIC):len(MAGIC) + 4])[0]
    header_start = len(MAGIC) + 4
    header_end = header_start + header_size
    header = json.loads(plain[header_start:header_end])
    header["source"]["public_feature_contract_sha256"] = "0" * 64
    encoded = canonical(header).encode("utf-8")
    assert len(encoded) == header_size
    plain[header_start:header_end] = encoded
    tampered = lzma.compress(bytes(plain), preset=9 | lzma.PRESET_EXTREME)
    with pytest.raises(CompactProgramError, match="feature-contract binding"):
        decode_program(
            tampered,
            expected_compact_sha256=sha256(tampered).hexdigest(),
            **_source_bindings(raw),
        )


def test_v9_transport_contract_reserves_secret_file_capacity_and_is_explanation_only():
    contract = subject.transport_contract()
    assert contract["raw_public_feature_count"] == 197
    assert contract["derived_public_feature_count"] == 668
    assert contract["threshold_dictionary_index_dtype"] == "uint32"
    assert contract["explicit_preorder_topology_preserved"] is True
    assert contract["trace_supported_after_decode"] is True
    assert contract["audited_action_parity_required"] is True
    assert contract["runtime_action_override"] is False
    assert contract["tree_controls_runtime"] is False
    assert contract["neural_actor_modified"] is False
    assert MAX_BASE64_BYTES + SECRET_FILE_RESERVE_BYTES == MAX_SECRET_FILE_BYTES
    assert base64_encoded_size(subject.MAX_COMPRESSED_BYTES) <= MAX_BASE64_BYTES


def test_v9_codec_rejects_a_different_public_tree_version():
    raw, _ = _source_bytes()
    payload = json.loads(raw)
    payload["version"] = "warehouse-r41-diagnostic-public-tree-program.v8"
    wrong = (canonical(payload) + "\n").encode()
    with pytest.raises(CompactProgramError, match="Source public-tree program"):
        encode_program(wrong, {"development": np.zeros((1, 197), dtype=np.float32)})


def test_v9_codec_has_no_actor_or_runtime_action_controller_dependency():
    source = inspect.getsource(subject)
    assert "NumPyNativeActor" not in source
    assert "from env.warehouse_native.policy" not in source
    assert "tree_controls_runtime\": False" in source
