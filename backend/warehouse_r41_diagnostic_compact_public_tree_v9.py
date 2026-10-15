"""Compact, explicit transport for the r4.1 v9 public-tree program.

The audited JSON program is intentionally verbose: every boosted-tree node is
an object with repeated field names.  That representation is useful for the
offline audit, but it does not fit in Render's one-megabyte Secret File after
Base64 encoding.  This module provides a deterministic transport format that
keeps the complete public split structure and exact float64 thresholds.

Leaf contributions use per-component affine int16 codes.  Any leaf reached
by an audited observation whose action would otherwise change is stored as an
exact float64 patch.  Encoding is accepted only when every supplied audited
public observation has exactly the same action before and after compaction.
The resulting program is still an explicit, traceable tree and never controls
or modifies the neural policy.  The v9 format widens the lossless threshold
dictionary index to uint32 for the larger 671-feature program family.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import lzma
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import canonical, digest
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    GROUPS,
    R41DiagnosticPublicTreeProgramV9,
)


VERSION = "warehouse-r41-diagnostic-compact-public-tree.v2"
MAGIC = b"PLR41CT2"
COMPRESSION = "XZ_LZMA2_PRESET_9_EXTREME"
QUANTIZATION = "per_component_affine_int16_with_exact_audited_leaf_patches"
TOPOLOGY = "preorder_full_binary_tree"
THRESHOLDS = "lossless_ieee754_binary64_dictionary"
MAX_SECRET_FILE_BYTES = 1_000_000
SECRET_FILE_RESERVE_BYTES = 40_000
MAX_BASE64_BYTES = MAX_SECRET_FILE_BYTES - SECRET_FILE_RESERVE_BYTES
MAX_COMPRESSED_BYTES = (MAX_BASE64_BYTES // 4) * 3
MAX_DECOMPRESSED_BYTES = 8_000_000
MAX_HEADER_BYTES = 512_000
MAX_ARRAY_BYTES = 7_000_000
MAX_AUDIT_ROWS = 5_000_000
PREDICTION_BATCH = 65_536
MAX_PATCH_ROUNDS = 8
COMPONENT_COUNT = 1 + len(GROUPS)

_HEADER_FIELDS = frozenset((
    "version", "source", "encoding", "program", "components", "arrays",
    "audit",
))
_SOURCE_FIELDS = frozenset((
    "file_sha256", "content_sha256", "actor_feature_names_sha256",
    "public_feature_contract_sha256",
))
_ENCODING_FIELDS = frozenset((
    "compression", "quantization", "topology", "thresholds",
))
_PROGRAM_FIELDS = frozenset((
    "version", "relations", "action_names", "aggregation", "metadata",
    "specialists",
))
_COMPONENT_FIELDS = frozenset((
    "version", "kind", "classes", "action_names", "output_kind", "baseline",
    "leaf_value_semantics", "n_iterations", "metadata", "tree_count",
    "node_count", "split_count", "leaf_count", "leaf_offset", "leaf_scale",
))
_ARRAY_FIELDS = frozenset(("name", "dtype", "shape", "offset", "size", "sha256"))
_AUDIT_FIELDS = frozenset(("sets", "row_count", "source_actions_sha256",
                           "compact_actions_sha256", "exact_action_parity",
                           "maximum_absolute_probability_error",
                           "mean_absolute_probability_error",
                           "patch_count", "patch_rounds"))
_AUDIT_SET_FIELDS = frozenset(("name", "rows", "observation_sha256"))


class CompactProgramError(ValueError):
    """Raised when compact program bytes or parity evidence differ."""


def transport_contract() -> dict[str, Any]:
    """Return the fixed v9 transport and deployment-capacity boundary."""
    return {
        "version": VERSION,
        "source_program": "warehouse-r41-diagnostic-public-tree-program.v9",
        "raw_public_feature_count": 197,
        "derived_public_feature_count": 671,
        "explicit_preorder_topology_preserved": True,
        "thresholds_preserved_as_ieee754_binary64": True,
        "threshold_dictionary_index_dtype": "uint32",
        "audited_action_parity_required": True,
        "trace_supported_after_decode": True,
        "maximum_compact_bytes": MAX_COMPRESSED_BYTES,
        "maximum_base64_bytes": MAX_BASE64_BYTES,
        "secret_file_limit_bytes": MAX_SECRET_FILE_BYTES,
        "reserved_secret_file_bytes": SECRET_FILE_RESERVE_BYTES,
        "runtime_action_override": False,
        "tree_controls_runtime": False,
        "neural_actor_modified": False,
    }


def base64_encoded_size(byte_count: int) -> int:
    if type(byte_count) is not int or byte_count < 0:
        raise CompactProgramError("Compact byte count differs")
    return 4 * ((byte_count + 2) // 3)


def _sha(value: Any, label: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise CompactProgramError(label + " must be an exact lowercase SHA-256")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise CompactProgramError(label + " must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CompactProgramError(label + " must be finite")
    return result


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = canonical({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_sha256": sha256(array.tobytes(order="C")).hexdigest(),
    })
    return sha256(descriptor.encode("utf-8")).hexdigest()


def _observation_sets(
    values: Mapping[str, Any], *, expected_features: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not isinstance(values, Mapping) or not values:
        raise CompactProgramError("At least one audited public-observation set is required")
    normalized: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    total = 0
    for name in sorted(values):
        if type(name) is not str or not name or len(name) > 128:
            raise CompactProgramError("Audited observation-set name differs")
        try:
            array = np.asarray(values[name], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise CompactProgramError("Audited public observations differ") from error
        if (array.ndim != 2 or array.shape[1] != expected_features
                or not len(array) or not np.isfinite(array).all()):
            raise CompactProgramError("Audited public-observation shape differs")
        array = np.ascontiguousarray(array)
        total += len(array)
        if total > MAX_AUDIT_ROWS:
            raise CompactProgramError("Audited public-observation scope exceeds its limit")
        normalized[name] = array
        records.append({
            "name": name,
            "rows": int(len(array)),
            "observation_sha256": _array_hash(array),
        })
    return normalized, records


def _components(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    specialists = payload.get("specialists")
    if not isinstance(specialists, list) or len(specialists) != len(GROUPS):
        raise CompactProgramError("Public-tree specialist registry differs")
    return [payload["base"], *[item["program"] for item in specialists]]


def _topology_end(nodes: Sequence[Mapping[str, Any]], index: int) -> int:
    node = nodes[index]
    if node["kind"] == "leaf":
        return index + 1
    if node["left"] != index + 1:
        raise CompactProgramError("Boosted-tree nodes are not canonical preorder")
    right = _topology_end(nodes, index + 1)
    if node["right"] != right:
        raise CompactProgramError("Boosted-tree right-child topology differs")
    return _topology_end(nodes, right)


def _component_header(component: Mapping[str, Any], *, leaf_offset: float,
                      leaf_scale: float) -> dict[str, Any]:
    trees = component["trees"]
    node_count = sum(len(tree["nodes"]) for tree in trees)
    leaf_count = sum(
        node["kind"] == "leaf" for tree in trees for node in tree["nodes"])
    return {
        key: deepcopy(component[key])
        for key in (
            "version", "kind", "classes", "action_names", "output_kind",
            "baseline", "leaf_value_semantics", "n_iterations", "metadata",
        )
    } | {
        "tree_count": len(trees),
        "node_count": node_count,
        "split_count": node_count - leaf_count,
        "leaf_count": leaf_count,
        "leaf_offset": leaf_offset,
        "leaf_scale": leaf_scale,
    }


def _program_header(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "relations": deepcopy(payload["relations"]),
        "action_names": deepcopy(payload["action_names"]),
        "aggregation": deepcopy(payload["aggregation"]),
        "metadata": deepcopy(payload["metadata"]),
        "specialists": [{
            "group": item["group"],
            "route": deepcopy(item["route"]),
            "combination": item["combination"],
            "mix_weight": item["mix_weight"],
        } for item in payload["specialists"]],
    }


def _threshold_dictionary(components: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[bytes, int]]:
    bits = {
        np.float64(node["threshold"]).tobytes()
        for component in components for tree in component["trees"]
        for node in tree["nodes"] if node["kind"] == "split"
    }
    ordered = sorted(bits, key=lambda value: int.from_bytes(value, "little"))
    if not ordered:
        array = np.empty(0, dtype="<u8")
    else:
        array = np.frombuffer(b"".join(ordered), dtype="<u8").copy()
    return array, {value: index for index, value in enumerate(ordered)}


def _base_arrays(payload: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    components = _components(payload)
    threshold_bits, threshold_index = _threshold_dictionary(components)
    if len(threshold_bits) > np.iinfo(np.uint32).max:
        raise CompactProgramError("Threshold dictionary exceeds uint32 capacity")
    arrays: dict[str, np.ndarray] = {"threshold_bits": threshold_bits}
    headers: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        if len(component["trees"]) > np.iinfo(np.uint16).max:
            raise CompactProgramError("Boosted-tree count exceeds compact limits")
        original_leaves = np.asarray([
            _finite(node["value"], "leaf value")
            for tree in component["trees"] for node in tree["nodes"]
            if node["kind"] == "leaf"
        ], dtype=np.float64)
        if not len(original_leaves):
            raise CompactProgramError("Boosted-tree component has no leaves")
        leaf_offset = float(original_leaves.min())
        leaf_scale = float(
            (original_leaves.max() - leaf_offset) / np.iinfo(np.uint16).max)
        headers.append(_component_header(
            component, leaf_offset=leaf_offset, leaf_scale=leaf_scale))
        counts: list[int] = []
        leaf_mask: list[bool] = []
        features: list[int] = []
        thresholds: list[int] = []
        missing_left: list[bool] = []
        leaves: list[float] = []
        for tree in component["trees"]:
            nodes = tree["nodes"]
            if (len(nodes) > np.iinfo(np.uint16).max
                    or _topology_end(nodes, 0) != len(nodes)):
                raise CompactProgramError("Boosted-tree topology exceeds compact limits")
            counts.append(len(nodes))
            for node in nodes:
                is_leaf = node["kind"] == "leaf"
                leaf_mask.append(is_leaf)
                if is_leaf:
                    value = _finite(node["value"], "leaf value")
                    code = (0 if leaf_scale == 0.0 else int(np.rint(
                        (value - leaf_offset) / leaf_scale)))
                    if not 0 <= code <= np.iinfo(np.uint16).max:
                        raise CompactProgramError("Leaf int16 code is out of range")
                    leaves.append(code - 32768)
                else:
                    if not 0 <= int(node["feature_index"]) <= np.iinfo(np.uint16).max:
                        raise CompactProgramError(
                            "Feature index exceeds compact uint16 capacity")
                    features.append(int(node["feature_index"]))
                    thresholds.append(threshold_index[
                        np.float64(node["threshold"]).tobytes()])
                    missing_left.append(bool(node["missing_go_to_left"]))
        prefix = f"c{component_index}."
        arrays[prefix + "node_counts"] = np.asarray(counts, dtype="<u2")
        arrays[prefix + "leaf_mask"] = np.packbits(
            np.asarray(leaf_mask, dtype=np.uint8), bitorder="little")
        arrays[prefix + "feature_indices"] = np.asarray(features, dtype="<u2")
        arrays[prefix + "threshold_indices"] = np.asarray(thresholds, dtype="<u4")
        arrays[prefix + "missing_left"] = np.packbits(
            np.asarray(missing_left, dtype=np.uint8), bitorder="little")
        arrays[prefix + "leaf_codes"] = np.asarray(leaves, dtype="<i2")
    arrays.update({
        "patch_components": np.empty(0, dtype=np.uint8),
        "patch_trees": np.empty(0, dtype="<u2"),
        "patch_nodes": np.empty(0, dtype="<u2"),
        "patch_values": np.empty(0, dtype="<f8"),
    })
    return arrays, headers


def _array_records(arrays: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], bytes]:
    records: list[dict[str, Any]] = []
    blocks: list[bytes] = []
    offset = 0
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        raw = array.tobytes(order="C")
        records.append({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": offset,
            "size": len(raw),
            "sha256": sha256(raw).hexdigest(),
        })
        blocks.append(raw)
        offset += len(raw)
    if offset > MAX_ARRAY_BYTES:
        raise CompactProgramError("Compact program arrays exceed their limit")
    return records, b"".join(blocks)


def _decode_payload(header: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    compact_components = header["components"]
    program_header = header["program"]
    threshold_bits = arrays["threshold_bits"]
    threshold_values = threshold_bits.view("<f8")
    components: list[dict[str, Any]] = []
    patch_map = {
        (int(component), int(tree), int(node)): float(value)
        for component, tree, node, value in zip(
            arrays["patch_components"], arrays["patch_trees"],
            arrays["patch_nodes"], arrays["patch_values"], strict=True)
    }
    if len(patch_map) != len(arrays["patch_values"]):
        raise CompactProgramError("Compact leaf patches contain duplicate locations")
    relations = program_header["relations"]
    feature_names = [
        *relations["base_feature_names"],
        *relations["derived_feature_names"],
    ]
    for component_index, component_header in enumerate(compact_components):
        prefix = f"c{component_index}."
        counts = arrays[prefix + "node_counts"]
        leaf_mask = np.unpackbits(
            arrays[prefix + "leaf_mask"], bitorder="little")[
                :component_header["node_count"]].astype(bool)
        features = arrays[prefix + "feature_indices"]
        thresholds = arrays[prefix + "threshold_indices"]
        missing = np.unpackbits(
            arrays[prefix + "missing_left"], bitorder="little")[
                :component_header["split_count"]].astype(bool)
        leaf_codes = arrays[prefix + "leaf_codes"]
        leaf_offset_value = _finite(
            component_header["leaf_offset"], "leaf offset")
        leaf_scale_value = _finite(
            component_header["leaf_scale"], "leaf scale")
        if leaf_scale_value < 0.0:
            raise CompactProgramError("Compact leaf scale is negative")
        trees = []
        node_offset = split_offset = leaf_offset = 0
        output_count = 1 if len(component_header["classes"]) == 2 else len(
            component_header["classes"])
        for tree_index, raw_count in enumerate(counts):
            count = int(raw_count)
            local_leaf = leaf_mask[node_offset:node_offset + count]
            nodes: list[dict[str, Any]] = []
            local_split = split_offset
            local_value = leaf_offset
            for node_index, is_leaf in enumerate(local_leaf):
                patch_key = (component_index, tree_index, node_index)
                if is_leaf:
                    code = int(leaf_codes[local_value]) + 32768
                    value = patch_map.get(
                        patch_key,
                        leaf_offset_value + code * leaf_scale_value,
                    )
                    nodes.append({"kind": "leaf", "value": value})
                    local_value += 1
                else:
                    threshold_index = int(thresholds[local_split])
                    if threshold_index >= len(threshold_values):
                        raise CompactProgramError("Compact threshold index is out of range")
                    nodes.append({
                        "kind": "split",
                        "feature_index": int(features[local_split]),
                        "threshold": float(threshold_values[threshold_index]),
                        "missing_go_to_left": bool(missing[local_split]),
                        "left": node_index + 1,
                        "right": -1,
                    })
                    local_split += 1
            # Recover right children from the canonical preorder leaf mask.
            def close(index: int) -> int:
                if nodes[index]["kind"] == "leaf":
                    return index + 1
                right = close(index + 1)
                nodes[index]["right"] = right
                return close(right)
            if not nodes or close(0) != len(nodes):
                raise CompactProgramError("Compact preorder topology is malformed")
            trees.append({
                "iteration": tree_index // output_count,
                "output_index": tree_index % output_count,
                "nodes": nodes,
            })
            node_offset += count
            split_offset = local_split
            leaf_offset = local_value
        if (node_offset != component_header["node_count"]
                or split_offset != component_header["split_count"]
                or leaf_offset != component_header["leaf_count"]):
            raise CompactProgramError("Compact component dimensions differ")
        component = {
            key: deepcopy(component_header[key])
            for key in (
                "version", "kind", "classes", "action_names", "output_kind",
                "baseline", "leaf_value_semantics", "n_iterations", "metadata",
            )
        }
        component["feature_names"] = deepcopy(feature_names)
        component["trees"] = trees
        components.append(component)
    payload = {
        key: deepcopy(program_header[key])
        for key in ("version", "relations", "action_names", "aggregation", "metadata")
    }
    payload["base"] = components[0]
    payload["specialists"] = [{
        **deepcopy(program_header["specialists"][index]),
        "program": components[index + 1],
    } for index in range(len(GROUPS))]
    return payload


def _predict_outputs(
    program: R41DiagnosticPublicTreeProgramV9,
    observations: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    actions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, values in observations.items():
        chunks: list[np.ndarray] = []
        for start in range(0, len(values), PREDICTION_BATCH):
            chunks.append(program.predict_proba_batch(
                values[start:start + PREDICTION_BATCH]))
        probability = np.concatenate(chunks).astype(np.float64, copy=False)
        probabilities[name] = probability
        actions[name] = np.argmax(probability, axis=1).astype(np.uint8)
    return actions, probabilities


def _actions_hash(actions: Mapping[str, np.ndarray]) -> str:
    return digest([{
        "name": name,
        "rows": len(actions[name]),
        "actions_sha256": _array_hash(actions[name]),
    } for name in sorted(actions)])


def _leaf_node(tree: Mapping[str, Any], observation: np.ndarray) -> int:
    nodes = tree["nodes"]
    index = 0
    while nodes[index]["kind"] == "split":
        node = nodes[index]
        index = node["left"] if (
            observation[node["feature_index"]] <= node["threshold"]
        ) else node["right"]
    return index


def _patch_mismatches(
    source: R41DiagnosticPublicTreeProgramV9,
    compact: R41DiagnosticPublicTreeProgramV9,
    observation_sets: Mapping[str, np.ndarray],
    source_actions: Mapping[str, np.ndarray],
    compact_actions: Mapping[str, np.ndarray],
    patches: dict[tuple[int, int, int], float],
) -> int:
    components = [source.base_program, *[
        source._specialist_programs[group] for group in GROUPS]]
    added = 0
    for name in sorted(observation_sets):
        mismatch = np.flatnonzero(compact_actions[name] != source_actions[name])
        if not len(mismatch):
            continue
        expanded = source.relations.transform_batch(observation_sets[name][mismatch])
        for mismatch_index, row in zip(mismatch, expanded, strict=True):
            # Restore every contribution reached by this audited mismatch.
            # The public program averages probabilities from independently
            # normalized components, so restoring only two competing classes
            # is not sufficient in general.  Exact reached leaves make this
            # complete observation's component distributions identical while
            # retaining int16 values everywhere else.
            active = [True, *[
                bool(source._route_mask(row[None, :], item["route"])[0])
                for item in source._specialists
            ]]
            for component_index, component in enumerate(components):
                if not active[component_index]:
                    continue
                for tree_index, tree in enumerate(component._trees):
                    node_index = _leaf_node(tree, row)
                    key = (component_index, tree_index, node_index)
                    if key in patches:
                        continue
                    exact = float(tree["nodes"][node_index]["value"])
                    patches[key] = exact
                    added += 1
    return added


def _install_patches(arrays: dict[str, np.ndarray],
                     patches: Mapping[tuple[int, int, int], float]) -> None:
    ordered = sorted(patches.items())
    arrays["patch_components"] = np.asarray(
        [key[0] for key, _ in ordered], dtype=np.uint8)
    arrays["patch_trees"] = np.asarray(
        [key[1] for key, _ in ordered], dtype="<u2")
    arrays["patch_nodes"] = np.asarray(
        [key[2] for key, _ in ordered], dtype="<u2")
    arrays["patch_values"] = np.asarray(
        [value for _, value in ordered], dtype="<f8")


def _build_header(*, source_raw: bytes, source_payload: Mapping[str, Any],
                  components: Sequence[Mapping[str, Any]],
                  arrays: Mapping[str, np.ndarray],
                  audit: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    records, blocks = _array_records(arrays)
    header = {
        "version": VERSION,
        "source": {
            "file_sha256": sha256(source_raw).hexdigest(),
            "content_sha256": digest(source_payload),
            "actor_feature_names_sha256": digest(
                source_payload["relations"]["base_feature_names"]),
            "public_feature_contract_sha256": digest(
                source_payload["relations"]),
        },
        "encoding": {
            "compression": COMPRESSION,
            "quantization": QUANTIZATION,
            "topology": TOPOLOGY,
            "thresholds": THRESHOLDS,
        },
        "program": _program_header(source_payload),
        "components": deepcopy(list(components)),
        "arrays": records,
        "audit": deepcopy(dict(audit)),
    }
    encoded = canonical(header).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise CompactProgramError("Compact program header exceeds its limit")
    return header, MAGIC + struct.pack("<I", len(encoded)) + encoded + blocks


def encode_program(
    source_raw: bytes, audited_observations: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Encode one canonical v9 JSON program and prove audited action parity."""

    if type(source_raw) is not bytes or not source_raw:
        raise CompactProgramError("Source public-tree bytes are required")
    try:
        source_payload = json.loads(source_raw)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CompactProgramError("Source public-tree JSON differs") from error
    if (canonical(source_payload) + "\n").encode("utf-8") != source_raw:
        raise CompactProgramError("Source public-tree JSON is not canonical")
    try:
        source = R41DiagnosticPublicTreeProgramV9.from_dict(source_payload)
    except ValueError as error:
        raise CompactProgramError("Source public-tree program differs") from error
    observations, set_records = _observation_sets(
        audited_observations, expected_features=len(source.base_feature_names))
    arrays, component_headers = _base_arrays(source_payload)
    source_actions, source_probabilities = _predict_outputs(source, observations)
    patches: dict[tuple[int, int, int], float] = {}
    rounds = 0
    while True:
        _install_patches(arrays, patches)
        provisional_audit = {
            "sets": set_records,
            "row_count": sum(row["rows"] for row in set_records),
            "source_actions_sha256": _actions_hash(source_actions),
            "compact_actions_sha256": "0" * 64,
            "exact_action_parity": False,
            "maximum_absolute_probability_error": 0.0,
            "mean_absolute_probability_error": 0.0,
            "patch_count": len(patches),
            "patch_rounds": rounds,
        }
        provisional_header, _ = _build_header(
            source_raw=source_raw, source_payload=source_payload,
            components=component_headers, arrays=arrays, audit=provisional_audit)
        compact = R41DiagnosticPublicTreeProgramV9.from_dict(
            _decode_payload(provisional_header, arrays))
        compact_actions, compact_probabilities = _predict_outputs(
            compact, observations)
        if all(np.array_equal(source_actions[name], compact_actions[name])
               for name in observations):
            break
        if rounds >= MAX_PATCH_ROUNDS:
            raise CompactProgramError("Compact leaf patching did not converge")
        added = _patch_mismatches(
            source, compact, observations, source_actions, compact_actions,
            patches)
        if not added:
            raise CompactProgramError("Compact action mismatch could not be patched")
        rounds += 1
    absolute_errors = [
        np.abs(source_probabilities[name] - compact_probabilities[name])
        for name in sorted(observations)
    ]
    error_count = sum(error.size for error in absolute_errors)
    audit = {
        "sets": set_records,
        "row_count": sum(row["rows"] for row in set_records),
        "source_actions_sha256": _actions_hash(source_actions),
        "compact_actions_sha256": _actions_hash(compact_actions),
        "exact_action_parity": True,
        "maximum_absolute_probability_error": max(
            float(error.max(initial=0.0)) for error in absolute_errors),
        "mean_absolute_probability_error": (
            sum(float(error.sum()) for error in absolute_errors) / error_count),
        "patch_count": len(patches),
        "patch_rounds": rounds,
    }
    header, plain = _build_header(
        source_raw=source_raw, source_payload=source_payload,
        components=component_headers, arrays=arrays, audit=audit)
    compressed = lzma.compress(
        plain, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    if not compressed or len(compressed) > MAX_COMPRESSED_BYTES:
        raise CompactProgramError("Compact public-tree artifact exceeds its limit")
    encoded_size = base64_encoded_size(len(compressed))
    if encoded_size > MAX_BASE64_BYTES:
        raise CompactProgramError("Compact public-tree Base64 transport exceeds its limit")
    # Decode the final bytes through the strict reader before returning them.
    decoded, checked = decode_program(
        compressed,
        expected_compact_sha256=sha256(compressed).hexdigest(),
        expected_source_program_sha256=sha256(source_raw).hexdigest(),
        expected_actor_feature_names_sha256=header["source"][
            "actor_feature_names_sha256"],
        expected_public_feature_contract_sha256=header["source"][
            "public_feature_contract_sha256"],
    )
    final_actions, final_probabilities = _predict_outputs(decoded, observations)
    if (_actions_hash(final_actions) != audit["source_actions_sha256"]
            or any(not np.array_equal(
                final_probabilities[name], compact_probabilities[name])
                   for name in observations)
            or checked["audit"] != audit):
        raise CompactProgramError("Compact public-tree final parity differs")
    report = {
        "version": VERSION,
        "status": "encoded_with_exact_audited_action_parity",
        "compact_file_sha256": sha256(compressed).hexdigest(),
        "compact_size": len(compressed),
        "source_program_file_sha256": header["source"]["file_sha256"],
        "source_program_content_sha256": header["source"]["content_sha256"],
        "actor_feature_names_sha256": header["source"][
            "actor_feature_names_sha256"],
        "public_feature_contract_sha256": header["source"][
            "public_feature_contract_sha256"],
        "decoded_program_content_sha256": digest(decoded.to_dict()),
        "audit": deepcopy(audit),
        "transport": {
            "contract_sha256": digest(transport_contract()),
            "compact_bytes": len(compressed),
            "base64_bytes": encoded_size,
            "maximum_compact_bytes": MAX_COMPRESSED_BYTES,
            "maximum_base64_bytes": MAX_BASE64_BYTES,
            "secret_file_limit_bytes": MAX_SECRET_FILE_BYTES,
            "reserved_secret_file_bytes": (
                MAX_SECRET_FILE_BYTES - encoded_size),
            "configured_minimum_reserve_bytes": SECRET_FILE_RESERVE_BYTES,
            "fits_secret_file_capacity": True,
        },
        "runtime_action_override": False,
        "tree_controls_runtime": False,
        "neural_actor_modified": False,
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    return compressed, report


def _decompress(raw: bytes) -> bytes:
    if type(raw) is not bytes or not raw or len(raw) > MAX_COMPRESSED_BYTES:
        raise CompactProgramError("Compact public-tree artifact size differs")
    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 << 20)
    try:
        plain = decompressor.decompress(raw, max_length=MAX_DECOMPRESSED_BYTES + 1)
    except lzma.LZMAError as error:
        raise CompactProgramError("Compact public-tree compression differs") from error
    if (len(plain) > MAX_DECOMPRESSED_BYTES or not decompressor.eof
            or decompressor.unused_data):
        raise CompactProgramError("Compact public-tree decompressed size differs")
    return plain


def _parse(raw: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    plain = _decompress(raw)
    if len(plain) < len(MAGIC) + 4 or plain[:len(MAGIC)] != MAGIC:
        raise CompactProgramError("Compact public-tree magic differs")
    header_size = struct.unpack("<I", plain[len(MAGIC):len(MAGIC) + 4])[0]
    header_start = len(MAGIC) + 4
    block_start = header_start + header_size
    if not 0 < header_size <= MAX_HEADER_BYTES or block_start > len(plain):
        raise CompactProgramError("Compact public-tree header size differs")
    try:
        header = json.loads(plain[header_start:block_start])
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise CompactProgramError("Compact public-tree header differs") from error
    if canonical(header).encode("utf-8") != plain[header_start:block_start]:
        raise CompactProgramError("Compact public-tree header is not canonical")
    if (not isinstance(header, Mapping) or set(header) != _HEADER_FIELDS
            or header.get("version") != VERSION
            or not isinstance(header.get("source"), Mapping)
            or set(header["source"]) != _SOURCE_FIELDS
            or not isinstance(header.get("encoding"), Mapping)
            or set(header["encoding"]) != _ENCODING_FIELDS
            or header["encoding"] != {
                "compression": COMPRESSION, "quantization": QUANTIZATION,
                "topology": TOPOLOGY, "thresholds": THRESHOLDS,
            }
            or not isinstance(header.get("program"), Mapping)
            or set(header["program"]) != _PROGRAM_FIELDS
            or not isinstance(header.get("components"), list)
            or len(header["components"]) != COMPONENT_COUNT
            or any(not isinstance(value, Mapping)
                   or set(value) != _COMPONENT_FIELDS
                   for value in header["components"])):
        raise CompactProgramError("Compact public-tree header schema differs")
    _sha(header["source"].get("file_sha256"), "source program")
    _sha(header["source"].get("content_sha256"), "source program content")
    _sha(header["source"].get("actor_feature_names_sha256"),
         "Actor feature names")
    _sha(header["source"].get("public_feature_contract_sha256"),
         "public feature contract")
    audit = header.get("audit")
    if (not isinstance(audit, Mapping) or set(audit) != _AUDIT_FIELDS
            or audit.get("exact_action_parity") is not True
            or type(audit.get("row_count")) is not int
            or not 0 < audit["row_count"] <= MAX_AUDIT_ROWS
            or type(audit.get("patch_count")) is not int
            or audit["patch_count"] < 0
            or type(audit.get("patch_rounds")) is not int
            or not 0 <= audit["patch_rounds"] <= MAX_PATCH_ROUNDS
            or not 0.0 <= _finite(
                audit.get("maximum_absolute_probability_error"),
                "maximum absolute probability error") <= 1.0
            or not 0.0 <= _finite(
                audit.get("mean_absolute_probability_error"),
                "mean absolute probability error") <= 1.0
            or not isinstance(audit.get("sets"), list)
            or not audit["sets"]
            or any(not isinstance(value, Mapping)
                   or set(value) != _AUDIT_SET_FIELDS
                   for value in audit["sets"])
            or sum(value.get("rows", -1) for value in audit["sets"])
                != audit["row_count"]):
        raise CompactProgramError("Compact public-tree audit schema differs")
    _sha(audit.get("source_actions_sha256"), "source actions")
    if _sha(audit.get("compact_actions_sha256"), "compact actions") != audit[
            "source_actions_sha256"]:
        raise CompactProgramError("Compact public-tree audited actions differ")
    for value in audit["sets"]:
        if (type(value.get("name")) is not str or not value["name"]
                or type(value.get("rows")) is not int or value["rows"] <= 0):
            raise CompactProgramError("Compact public-tree audit set differs")
        _sha(value.get("observation_sha256"), "audited observations")
    names = [value["name"] for value in audit["sets"]]
    if names != sorted(names) or len(names) != len(set(names)):
        raise CompactProgramError("Compact public-tree audit-set order differs")
    records = header.get("arrays")
    if not isinstance(records, list) or not records:
        raise CompactProgramError("Compact public-tree array registry differs")
    arrays: dict[str, np.ndarray] = {}
    expected_offset = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _ARRAY_FIELDS:
            raise CompactProgramError("Compact public-tree array record differs")
        name = record.get("name")
        dtype_name = record.get("dtype")
        shape = record.get("shape")
        offset = record.get("offset")
        size = record.get("size")
        if (type(name) is not str or not name or name in arrays
                or type(dtype_name) is not str
                or not isinstance(shape, list) or len(shape) != 1
                or type(shape[0]) is not int or shape[0] < 0
                or type(offset) is not int or offset != expected_offset
                or type(size) is not int or size < 0):
            raise CompactProgramError("Compact public-tree array dimensions differ")
        try:
            dtype = np.dtype(dtype_name)
        except (TypeError, ValueError) as error:
            raise CompactProgramError("Compact public-tree dtype differs") from error
        if (dtype.hasobject or dtype.subdtype is not None
                or dtype.kind not in "buif" or dtype.itemsize > 8
                or size != shape[0] * dtype.itemsize):
            raise CompactProgramError("Compact public-tree dtype is unsafe")
        start, end = block_start + offset, block_start + offset + size
        if end > len(plain):
            raise CompactProgramError("Compact public-tree array exceeds payload")
        block = plain[start:end]
        if sha256(block).hexdigest() != _sha(record.get("sha256"), "array"):
            raise CompactProgramError("Compact public-tree array hash differs")
        arrays[name] = np.frombuffer(block, dtype=dtype).copy()
        expected_offset += size
    if block_start + expected_offset != len(plain):
        raise CompactProgramError("Compact public-tree trailing bytes differ")
    required = {"threshold_bits", "patch_components", "patch_trees",
                "patch_nodes", "patch_values"}
    for component_index in range(COMPONENT_COUNT):
        prefix = f"c{component_index}."
        required.update({prefix + suffix for suffix in (
            "node_counts", "leaf_mask", "feature_indices",
            "threshold_indices", "missing_left", "leaf_codes",
        )})
    if set(arrays) != required:
        raise CompactProgramError("Compact public-tree array set differs")
    exact_dtypes = {
        "threshold_bits": np.dtype("<u8"),
        "patch_components": np.dtype("u1"),
        "patch_trees": np.dtype("<u2"),
        "patch_nodes": np.dtype("<u2"),
        "patch_values": np.dtype("<f8"),
    }
    for component_index in range(COMPONENT_COUNT):
        prefix = f"c{component_index}."
        exact_dtypes.update({
            prefix + "node_counts": np.dtype("<u2"),
            prefix + "leaf_mask": np.dtype("u1"),
            prefix + "feature_indices": np.dtype("<u2"),
            prefix + "threshold_indices": np.dtype("<u4"),
            prefix + "missing_left": np.dtype("u1"),
            prefix + "leaf_codes": np.dtype("<i2"),
        })
    if any(arrays[name].dtype != dtype for name, dtype in exact_dtypes.items()):
        raise CompactProgramError("Compact public-tree array dtype registry differs")
    patch_count = audit["patch_count"]
    if any(len(arrays[name]) != patch_count for name in (
            "patch_components", "patch_trees", "patch_nodes", "patch_values")):
        raise CompactProgramError("Compact public-tree patch dimensions differ")
    if not np.isfinite(arrays["threshold_bits"].view("<f8")).all():
        raise CompactProgramError("Compact public-tree thresholds are not finite")
    if not np.isfinite(arrays["patch_values"]).all():
        raise CompactProgramError("Compact public-tree patches are not finite")
    for component_index, component in enumerate(header["components"]):
        integer_fields = (
            "tree_count", "node_count", "split_count", "leaf_count",
        )
        if (any(type(component.get(name)) is not int
                or component[name] < 0 for name in integer_fields)
                or component["tree_count"] <= 0
                or component["node_count"] <= 0
                or component["leaf_count"] <= 0
                or component["node_count"]
                    != component["split_count"] + component["leaf_count"]
                or _finite(component.get("leaf_scale"), "leaf scale") < 0.0):
            raise CompactProgramError("Compact component header dimensions differ")
        _finite(component.get("leaf_offset"), "leaf offset")
        prefix = f"c{component_index}."
        node_counts = arrays[prefix + "node_counts"]
        if (len(node_counts) != component["tree_count"]
                or np.any(node_counts == 0)
                or int(node_counts.astype(np.uint64).sum())
                    != component["node_count"]
                or len(arrays[prefix + "leaf_mask"])
                    != (component["node_count"] + 7) // 8
                or len(arrays[prefix + "feature_indices"])
                    != component["split_count"]
                or len(arrays[prefix + "threshold_indices"])
                    != component["split_count"]
                or len(arrays[prefix + "missing_left"])
                    != (component["split_count"] + 7) // 8
                or len(arrays[prefix + "leaf_codes"])
                    != component["leaf_count"]):
            raise CompactProgramError("Compact component array dimensions differ")
        if (len(arrays["threshold_bits"]) == 0
                and component["split_count"] != 0):
            raise CompactProgramError("Compact threshold dictionary is empty")
        if (len(arrays[prefix + "threshold_indices"])
                and int(arrays[prefix + "threshold_indices"].max())
                    >= len(arrays["threshold_bits"])):
            raise CompactProgramError("Compact threshold index is out of range")
        # Padding is required to be zero so one tree has one canonical byte form.
        for name, bit_count in (("leaf_mask", component["node_count"]),
                                ("missing_left", component["split_count"])):
            packed = arrays[prefix + name]
            if bit_count % 8 and len(packed):
                allowed = (1 << (bit_count % 8)) - 1
                if int(packed[-1]) & ~allowed:
                    raise CompactProgramError("Compact bit-array padding differs")
    patch_keys: set[tuple[int, int, int]] = set()
    for component, tree, node in zip(
            arrays["patch_components"], arrays["patch_trees"],
            arrays["patch_nodes"], strict=True):
        key = (int(component), int(tree), int(node))
        if key in patch_keys or key[0] >= COMPONENT_COUNT:
            raise CompactProgramError("Compact leaf patch location differs")
        component_header = header["components"][key[0]]
        if key[1] >= component_header["tree_count"]:
            raise CompactProgramError("Compact leaf patch tree differs")
        counts = arrays[f"c{key[0]}.node_counts"]
        if key[2] >= int(counts[key[1]]):
            raise CompactProgramError("Compact leaf patch node differs")
        offset = int(counts[:key[1]].astype(np.uint64).sum()) + key[2]
        leaf_mask = arrays[f"c{key[0]}.leaf_mask"]
        if not ((int(leaf_mask[offset // 8]) >> (offset % 8)) & 1):
            raise CompactProgramError("Compact patch must target a leaf")
        patch_keys.add(key)
    return dict(header), arrays


def decode_program(
    raw: bytes, *, expected_compact_sha256: str,
    expected_source_program_sha256: str,
    expected_actor_feature_names_sha256: str,
    expected_public_feature_contract_sha256: str,
) -> tuple[R41DiagnosticPublicTreeProgramV9, dict[str, Any]]:
    """Authenticate and decode one compact tree artifact."""

    if sha256(raw).hexdigest() != _sha(
            expected_compact_sha256, "compact public-tree artifact"):
        raise CompactProgramError("Compact public-tree artifact hash differs")
    header, arrays = _parse(raw)
    if header["source"]["file_sha256"] != _sha(
            expected_source_program_sha256, "source public-tree program"):
        raise CompactProgramError("Compact source public-tree binding differs")
    if header["source"]["actor_feature_names_sha256"] != _sha(
            expected_actor_feature_names_sha256, "Actor feature names"):
        raise CompactProgramError("Compact Actor feature-name binding differs")
    if header["source"]["public_feature_contract_sha256"] != _sha(
            expected_public_feature_contract_sha256, "public feature contract"):
        raise CompactProgramError("Compact public feature-contract binding differs")
    try:
        payload = _decode_payload(header, arrays)
        program = R41DiagnosticPublicTreeProgramV9.from_dict(payload)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        if isinstance(error, CompactProgramError):
            raise
        raise CompactProgramError("Compact public-tree payload differs") from error
    if (digest(list(program.base_feature_names))
            != header["source"]["actor_feature_names_sha256"]
            or digest(program.relations.contract())
            != header["source"]["public_feature_contract_sha256"]):
        raise CompactProgramError("Compact public feature binding differs")
    return program, header


def program_json_bytes(program: R41DiagnosticPublicTreeProgramV9) -> bytes:
    return (canonical(program.to_dict()) + "\n").encode("utf-8")


__all__ = [
    "VERSION", "MAGIC", "COMPRESSION", "QUANTIZATION", "TOPOLOGY",
    "THRESHOLDS", "MAX_COMPRESSED_BYTES", "MAX_BASE64_BYTES",
    "MAX_SECRET_FILE_BYTES", "SECRET_FILE_RESERVE_BYTES",
    "CompactProgramError", "transport_contract", "base64_encoded_size",
    "encode_program", "decode_program", "program_json_bytes",
]
