from __future__ import annotations

from hashlib import sha256
import lzma
import struct

import numpy as np
import pytest

from backend.training.warehouse_native_common import canonical, digest
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    VERSION as TREE_VERSION,
)
from backend.warehouse_r41_diagnostic_compact_public_tree_v8 import (
    MAGIC,
    MAX_DECOMPRESSED_BYTES,
    CompactProgramError,
    decode_program,
    encode_program,
    program_json_bytes,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    AGGREGATION,
    GROUPS,
    VERSION as PROGRAM_VERSION,
)


def _features():
    required = sorted(R41DiagnosticPublicRelationsV8._required_names())
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
    relations = R41DiagnosticPublicRelationsV8(names)
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


def _roundtrip():
    raw, names = _source_bytes()
    observations = np.zeros((3, 197), dtype=np.float32)
    observations[1, 0] = 1.0
    observations[2, 0] = -1.0
    compact, report = encode_program(raw, {"development": observations})
    program, header = decode_program(
        compact,
        expected_compact_sha256=sha256(compact).hexdigest(),
        expected_source_program_sha256=sha256(raw).hexdigest(),
    )
    return raw, names, observations, compact, report, program, header


def test_compact_tree_roundtrip_preserves_audited_actions_topology_and_trace():
    raw, names, observations, compact, report, program, header = _roundtrip()
    from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
        R41DiagnosticPublicTreeProgramV8,
    )
    source = R41DiagnosticPublicTreeProgramV8.from_json(raw.decode())
    assert np.array_equal(source.predict_batch(observations),
                          program.predict_batch(observations))
    assert report["audit"]["exact_action_parity"] is True
    assert report["audit"]["patch_count"] > 0
    assert report["audit"]["row_count"] == 3
    assert 0 <= report["audit"]["maximum_absolute_probability_error"] <= 1
    assert header["source"]["content_sha256"] == digest(source.to_dict())
    original = source.to_dict()["base"]["trees"][0]["nodes"][0]
    restored = program.to_dict()["base"]["trees"][0]["nodes"][0]
    assert restored == original  # exact threshold and topology
    observation = {name: 0.0 for name in names}
    trace = program.trace(observation)
    assert trace["prediction"] == "DOWN"
    assert trace["base"]["program_trace"]["trees"][0]["path"]
    assert len(compact) < len(raw)
    assert program_json_bytes(program).endswith(b"\n")


def test_compact_tree_reader_rejects_hash_drift_trailing_stream_and_bomb():
    raw, _, _, compact, _, _, _ = _roundtrip()
    with pytest.raises(CompactProgramError, match="artifact hash"):
        decode_program(
            compact,
            expected_compact_sha256="0" * 64,
            expected_source_program_sha256=sha256(raw).hexdigest(),
        )
    trailing = compact + lzma.compress(b"extra")
    with pytest.raises(CompactProgramError, match="decompressed size"):
        decode_program(
            trailing,
            expected_compact_sha256=sha256(trailing).hexdigest(),
            expected_source_program_sha256=sha256(raw).hexdigest(),
        )
    bomb = lzma.compress(b"x" * (MAX_DECOMPRESSED_BYTES + 1))
    with pytest.raises(CompactProgramError, match="decompressed size"):
        decode_program(
            bomb,
            expected_compact_sha256=sha256(bomb).hexdigest(),
            expected_source_program_sha256=sha256(raw).hexdigest(),
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
            expected_source_program_sha256=sha256(raw).hexdigest(),
        )
