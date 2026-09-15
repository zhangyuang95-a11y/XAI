"""Compress an admitted r4.5 neural Actor without adding an action controller."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np

from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r45-packed-per-output-channel.v2"


def _pack_int12(value: np.ndarray) -> np.ndarray:
    flat = value.astype(np.int16, copy=False).reshape(-1)
    if flat.size % 2:
        flat = np.append(flat, np.int16(0))
    unsigned = (flat.astype(np.int32) & 0xFFF).astype(np.uint16).reshape(-1, 2)
    packed = np.empty((len(unsigned), 3), dtype=np.uint8)
    packed[:, 0] = (unsigned[:, 0] & 0xFF).astype(np.uint8)
    packed[:, 1] = (((unsigned[:, 0] >> 8) & 0x0F)
                    | ((unsigned[:, 1] & 0x0F) << 4)).astype(np.uint8)
    packed[:, 2] = ((unsigned[:, 1] >> 4) & 0xFF).astype(np.uint8)
    return packed.reshape(-1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, choices=(8, 12), default=12)
    args = parser.parse_args(argv)
    source = Path(args.actor).resolve()
    parent = NumPyNativeActor(source)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(parent.metadata)
    storage = ("int8_per_output_channel_v1" if args.bits == 8
               else "int12_per_output_channel_v1")
    metadata.update({
        "weight_storage": storage,
        "weight_storage_version": VERSION,
        "uncompressed_actor_sha256": parent.sha256,
        "runtime_action_override": False,
    })
    arrays: dict[str, np.ndarray] = {}
    scales: dict[str, list[float]] = {}
    shapes: dict[str, list[int]] = {}
    maximum_integer = 127 if args.bits == 8 else 2047
    for name, weight in parent.weights.items():
        value = np.asarray(weight, dtype=np.float32)
        if name.endswith(".weight"):
            maximum = np.max(np.abs(value), axis=1)
            scale = np.maximum(maximum / maximum_integer, np.float32(1e-12)).astype(np.float32)
            quantized = np.clip(np.rint(value / scale[:, None]),
                                -maximum_integer, maximum_integer)
            arrays[name] = (quantized.astype(np.int8) if args.bits == 8
                            else _pack_int12(quantized.astype(np.int16)))
            scales[name] = [float(item) for item in scale]
            shapes[name] = list(value.shape)
        else:
            arrays[name] = value.astype(np.float16)
    metadata["weight_scales"] = scales
    metadata["weight_shapes"] = shapes
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle,
                            metadata_json=json.dumps(metadata, sort_keys=True),
                            **arrays)
    temporary.replace(destination)
    quantized = NumPyNativeActor(destination)
    report = {
        "version": VERSION,
        "source": str(source),
        "source_sha256": parent.sha256,
        "actor": str(destination),
        "actor_sha256": sha256(destination.read_bytes()).hexdigest(),
        "hidden": quantized.hidden,
        "runtime_action_override": False,
        "weight_storage": metadata["weight_storage"],
    }
    report_path = destination.with_name("quantization_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
