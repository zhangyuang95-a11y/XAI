#!/usr/bin/env python3
"""Build the one append-only v14 closeout of the timed-out v13 final."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_final_timeout_closeout_v14 as closeout
from backend.training.warehouse_native_common import file_hash


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--final-anchor", type=Path, required=True)
    result.add_argument("--final-completion", type=Path, required=True)
    result.add_argument("--permanent-final-registry", type=Path, required=True)
    result.add_argument("--failed-public-output", type=Path, required=True)
    result.add_argument("--materializer-config", type=Path, required=True)
    result.add_argument("--permanent-closeout-registry", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    value = closeout.build(
        final_anchor_path=args.final_anchor,
        final_completion_path=args.final_completion,
        permanent_final_registry=args.permanent_final_registry,
        failed_public_output=args.failed_public_output,
        materializer_config_path=args.materializer_config,
        permanent_closeout_registry=args.permanent_closeout_registry,
        output=args.output)
    receipt = args.output / closeout.RECEIPT_NAME
    print(json.dumps({
        "version": closeout.VERSION,
        "status": value["status"],
        "closeout_key": value["closeout_key"],
        "receipt": str(receipt),
        "receipt_sha256": file_hash(receipt),
        "ranking_input_count": value[
            "burned_candidate_universe"]["ranking_input_count"],
        "evaluated_scene_count": value[
            "burned_candidate_universe"]["completed_evaluated_prefix_count"],
        "accepted_scene_count": value[
            "burned_candidate_universe"]["accepted_scene_count"],
        "row_count": value["burned_observation_projection"]["row_count"],
        "environment_steps": value[
            "burned_observation_projection"]["environment_steps"],
        "elapsed_seconds": value[
            "burned_observation_projection"]["elapsed_seconds"],
        "formal_ready": False,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

