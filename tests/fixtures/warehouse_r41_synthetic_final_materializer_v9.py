"""Synthetic-only subprocess fixture for the v9 final materializer protocol."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # This is a test-only value. Its read proves that source execution happens
    # in the post-claim subprocess without touching any real holdout secret.
    salt = os.environ["WAREHOUSE_R41_SYNTHETIC_FINAL_SALT"]
    anchor = json.loads(Path(args.claim).read_text(encoding="utf-8"))
    sources = dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))
    scenes = [{
        "id": f"synthetic-final-{index}", "seed": 800_000 + index,
        "family_id": f"synthetic-family-{index % 6}",
        "fingerprint": sha256(
            f"{salt}:synthetic-final:{index}".encode("utf-8")).hexdigest(),
    } for index in range(64)]
    value = {
        "version": "warehouse-r41-diagnostic-final-material.v9",
        "status": "materialized_after_irrevocable_final_claim",
        "claim": {
            "attempt_key": anchor["attempt_key"],
            "attempt_anchor_content_sha256": anchor["content_sha256"],
        },
        "scenes": scenes,
        "selection": {
            "whole_scene_selection": True, "scene_count": 64,
            "program_access": False, "program_predictions_access": False,
            "actor_outputs_access": False, "action_labels_access": False,
            "salt_access_after_permanent_claim": True,
            "candidate_adaptation": False, "runtime_action_override": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
