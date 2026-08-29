"""Register a rejected formal seed family before evaluating a new candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from backend.training.seed_ledger import (
    evaluation_seed_span,
    reserve_evaluation_seed_span,
)


def _formal_ranges(payload: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    seed_ranges = payload.get("seed_ranges")
    if not isinstance(seed_ranges, Mapping):
        raise ValueError("Rejected report has no seed_ranges mapping.")
    ranges: list[tuple[int, int]] = []
    for value in seed_ranges.values():
        items = value.values() if isinstance(value, Mapping) else (value,)
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Every formal seed range must be [start, count].")
            start, count = int(item[0]), int(item[1])
            if count <= 0:
                raise ValueError("Formal seed range counts must be positive.")
            ranges.append((start, start + count))
    if not ranges:
        raise ValueError("Rejected report declares no formal seed ranges.")
    return tuple(ranges)


def register_rejected_evaluation(
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    report_path: Path,
    report_sha256: str,
) -> dict[str, Any]:
    """Return a summary that conservatively reserves the rejected seed span."""

    if bool(report.get("formal_candidate", False)):
        raise ValueError("A passing formal report cannot be registered as rejected.")
    checks = report.get("acceptance_checks")
    if not isinstance(checks, Mapping) or all(bool(value) for value in checks.values()):
        raise ValueError("Rejected report must contain at least one failed check.")
    exact_ranges = _formal_ranges(report)
    seed_ranges = report["seed_ranges"]
    start, end_exclusive = evaluation_seed_span(seed_ranges)
    seed = int(report.get("seed_ranges", {}).get("ai_ai", [start])[0])
    name = f"rejected_formal_evaluation_{seed}"

    updated = deepcopy(dict(summary))
    ledger = updated.get("seed_ledger", {})
    if not isinstance(ledger, Mapping):
        raise ValueError("Training summary seed_ledger must be a mapping.")
    updated["seed_ledger"] = reserve_evaluation_seed_span(
        ledger,
        seed_ranges,
        name=name,
    )

    failures = sorted(key for key, value in checks.items() if not bool(value))
    prior_intervals = ledger.get("reserved_intervals", ())
    overlaps = sorted(
        str(item.get("name", "unnamed"))
        for item in prior_intervals
        if isinstance(item, Mapping)
        and item.get("name") != name
        and any(
            int(item["start"]) < formal_end
            and formal_start < int(item["end_exclusive"])
            for formal_start, formal_end in exact_ranges
        )
    )
    record = {
        "name": name,
        "path": str(report_path.resolve()),
        "sha256": report_sha256,
        "model_sha256": report.get("artifact_hashes", {}).get("model"),
        "start": start,
        "end_exclusive": end_exclusive,
        "failed_checks": failures,
        "detected_prior_seed_overlaps": overlaps,
        "methodologically_independent": not overlaps,
        "used_for_weight_updates": False,
        "used_for_evaluation_metric_diagnosis": True,
    }
    rejected = updated.setdefault("rejected_formal_evaluations", [])
    if not isinstance(rejected, list):
        raise ValueError("rejected_formal_evaluations must be a list.")
    previous = next(
        (
            item
            for item in rejected
            if isinstance(item, Mapping) and item.get("name") == name
        ),
        None,
    )
    if previous is None:
        rejected.append(record)
    elif dict(previous) != record:
        raise ValueError("Rejected formal record already exists with other evidence.")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_summary", type=Path)
    parser.add_argument("rejected_report", type=Path)
    args = parser.parse_args()
    report_bytes = args.rejected_report.read_bytes()
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    report = json.loads(report_bytes.decode("utf-8"))
    updated = register_rejected_evaluation(
        summary,
        report,
        report_path=args.rejected_report,
        report_sha256=sha256(report_bytes).hexdigest(),
    )
    temporary = args.training_summary.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.training_summary)
    print(
        json.dumps(
            {
                "training_summary": str(args.training_summary.resolve()),
                "rejected_report": str(args.rejected_report.resolve()),
                "seed_ledger": updated["seed_ledger"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
