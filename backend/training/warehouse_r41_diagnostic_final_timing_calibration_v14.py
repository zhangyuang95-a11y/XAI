"""Freeze a public timing bound for the v14 final materializer.

The only measurement is the completed hash-only v13 reconstruction after its
permanent timeout closeout.  At that point all ranked inputs are retired and
the receipt contains no actions, probabilities, labels, program predictions,
raw observations, or salt.  The v14 private salt and candidate universe are
therefore absent from this pre-claim calibration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from backend.training import (
    warehouse_r41_diagnostic_final_timeout_closeout_public_v14 as closeout_api,
)
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-timing-calibration.v14"
STATUS = "passed_public_retired_development_timing_calibration"
LEGACY_TIMEOUT_SECONDS = 600
HEADROOM_NUMERATOR = 11
HEADROOM_DENOMINATOR = 10
TIMEOUT_MULTIPLIER = 3
MIN_TIMEOUT_SECONDS = 1_800
MAX_TIMEOUT_SECONDS = 14_400
RANKING_INPUT_COUNT = 2_160
FINAL_SCENE_COUNT = 64
MAX_JSON_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "measurement_source": (
            "permanently-retired v13 candidate prefix exact hash-only replay"),
        "ranking_input_count": RANKING_INPUT_COUNT,
        "accepted_scene_count": FINAL_SCENE_COUNT,
        "legacy_timeout_seconds": LEGACY_TIMEOUT_SECONDS,
        "upper_bound_headroom_ratio": [
            HEADROOM_NUMERATOR, HEADROOM_DENOMINATOR],
        "timeout_multiplier": TIMEOUT_MULTIPLIER,
        "minimum_timeout_seconds": MIN_TIMEOUT_SECONDS,
        "maximum_timeout_seconds": MAX_TIMEOUT_SECONDS,
        "calibration_precedes_v14_final_claim": True,
        "v14_candidate_universe_access": False,
        "v14_private_salt_path_stat_or_read": False,
        "v14_final_identity_or_rows_access": False,
        "actions_probabilities_labels_or_program_access": False,
        "formal_ready": False,
    }


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(label + " must be a lowercase SHA-256 digest")
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({name: child for name, child in value.items()
                                   if name != "content_sha256"}))


def _regular(value: str | Path, label: str, *, expected_sha256: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat(follow_symlinks=False).st_size <= MAX_JSON_BYTES
            or file_hash(path) != _sha(expected_sha256, label)):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _timing(closeout: Mapping[str, Any]) -> dict[str, Any]:
    failure = closeout.get("v13_timeout_failure")
    burned = closeout.get("burned_candidate_universe")
    projection = closeout.get("burned_observation_projection")
    disposition = closeout.get("disposition")
    boundary = closeout.get("information_boundary")
    if (not isinstance(failure, Mapping) or not isinstance(burned, Mapping)
            or not isinstance(projection, Mapping)
            or not isinstance(disposition, Mapping)
            or not isinstance(boundary, Mapping)
            or failure.get("controller_materializer_timeout_seconds")
                != LEGACY_TIMEOUT_SECONDS
            or failure.get("public_output_published") is not False
            or failure.get("materializer_output_published") is not False
            or failure.get("retry_allowed") is not False
            or burned.get("ranking_input_count") != RANKING_INPUT_COUNT
            or burned.get("accepted_scene_count") != FINAL_SCENE_COUNT
            or burned.get("all_ranking_inputs_retired") is not True
            or projection.get("accepted_scene_count") != FINAL_SCENE_COUNT
            or disposition.get("same_v13_final_attempt_retry_permitted") is not False
            or disposition.get("v13_candidate_and_program_reusable_without_refit")
                is not True
            or disposition.get("v13_passed_outer_reusable_without_rerun") is not True
            or boundary.get("salt_value_published") is not False
            or boundary.get("actions_probabilities_labels_or_program_accessed")
                is not False
            or boundary.get("labeled_rows_generated") is not False):
        raise ValueError("V13 timeout closeout cannot calibrate v14")
    elapsed = projection.get("elapsed_seconds")
    if (type(elapsed) not in (int, float) or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed)) or float(elapsed) <= 0.0):
        raise ValueError("Completed v13 public replay timing differs")
    conservative_base = max(float(elapsed), float(LEGACY_TIMEOUT_SECONDS))
    upper = int((
        Decimal(str(conservative_base)) * Decimal(HEADROOM_NUMERATOR)
        / Decimal(HEADROOM_DENOMINATOR)
    ).to_integral_value(rounding=ROUND_CEILING))
    required = max(MIN_TIMEOUT_SECONDS, TIMEOUT_MULTIPLIER * upper)
    if required > MAX_TIMEOUT_SECONDS:
        raise ValueError("Measured public timing exceeds the frozen v14 cap")
    return {
        "observed_completed_replay_seconds": float(elapsed),
        "conservative_base_seconds": conservative_base,
        "measured_upper_bound_seconds": upper,
        "required_timeout_seconds": required,
        "legacy_timeout_seconds": LEGACY_TIMEOUT_SECONDS,
        "headroom_numerator": HEADROOM_NUMERATOR,
        "headroom_denominator": HEADROOM_DENOMINATOR,
        "timeout_multiplier": TIMEOUT_MULTIPLIER,
        "minimum_timeout_seconds": MIN_TIMEOUT_SECONDS,
        "maximum_timeout_seconds": MAX_TIMEOUT_SECONDS,
        "completed_evaluated_prefix_count": burned[
            "completed_evaluated_prefix_count"],
        "environment_steps": projection["environment_steps"],
        "row_count": projection["row_count"],
        "unique_observation_count": projection["unique_observation_count"],
    }


def create_calibration(
    *, timeout_closeout_path: str | Path,
    expected_timeout_closeout_sha256: str,
    permanent_timeout_closeout_registry: str | Path,
) -> dict[str, Any]:
    closeout_path = _regular(
        timeout_closeout_path, "v13 timeout closeout",
        expected_sha256=expected_timeout_closeout_sha256)
    closeout = closeout_api.read_saved_closeout_public(
        closeout_path,
        expected_closeout_sha256=expected_timeout_closeout_sha256,
        permanent_closeout_registry=permanent_timeout_closeout_registry)
    sources = producer_sources()
    value: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "timeout_closeout_file_sha256": file_hash(closeout_path),
            "timeout_closeout_content_sha256": closeout["content_sha256"],
            "burned_ranked_identities_sha256": closeout[
                "burned_candidate_universe"]["ranking_input_identities_sha256"],
            "burned_completed_prefix_sha256": closeout[
                "burned_candidate_universe"]["completed_evaluated_prefix_sha256"],
            "burned_observation_hashes_sha256": closeout[
                "burned_observation_projection"][
                    "unique_observation_hashes_sha256"],
        },
        "timing": _timing(closeout),
        "information_boundary": {
            "source_is_permanently_retired_before_calibration": True,
            "v14_candidate_universe_access": False,
            "v14_private_salt_path_stat_or_read": False,
            "v14_final_identity_or_rows_access": False,
            "actions_probabilities_labels_or_program_access": False,
            "participant_data_access": False,
            "formal_ready": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    value = create_calibration(**kwargs)
    destination = Path(output).expanduser().absolute()
    if (not destination.parent.is_dir() or destination.parent.is_symlink()
            or destination.exists() or destination.is_symlink()):
        raise FileExistsError("V14 timing output must be a new canonical path")
    _write_exclusive(destination, value)
    return deepcopy(value)


def read_saved_calibration(
    path: str | Path, *, expected_calibration_sha256: str,
    timeout_closeout_path: str | Path,
    expected_timeout_closeout_sha256: str,
    permanent_timeout_closeout_registry: str | Path,
) -> dict[str, Any]:
    calibration_path = _regular(
        path, "v14 timing calibration",
        expected_sha256=expected_calibration_sha256)
    raw = calibration_path.read_bytes()
    try:
        saved = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("V14 timing calibration must be JSON") from error
    if (not isinstance(saved, Mapping)
            or raw != (canonical(saved) + "\n").encode("utf-8")
            or saved.get("version") != VERSION or saved.get("status") != STATUS
            or saved.get("contract") != contract()
            or not _content_valid(saved) or saved.get("formal_ready") is not False):
        raise ValueError("Saved v14 timing calibration differs")
    recreated = create_calibration(
        timeout_closeout_path=timeout_closeout_path,
        expected_timeout_closeout_sha256=expected_timeout_closeout_sha256,
        permanent_timeout_closeout_registry=permanent_timeout_closeout_registry)
    if dict(saved) != recreated or file_hash(calibration_path) \
            != expected_calibration_sha256:
        raise ValueError("Saved v14 timing calibration differs from public replay")
    return deepcopy(dict(saved))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-closeout", required=True)
    parser.add_argument("--expected-timeout-closeout-sha256", required=True)
    parser.add_argument("--permanent-timeout-closeout-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build(
        output=args.output, timeout_closeout_path=args.timeout_closeout,
        expected_timeout_closeout_sha256=args.expected_timeout_closeout_sha256,
        permanent_timeout_closeout_registry=(
            args.permanent_timeout_closeout_registry))
    print(canonical({
        "status": value["status"],
        "content_sha256": value["content_sha256"],
        "required_timeout_seconds": value["timing"][
            "required_timeout_seconds"],
    }))


if __name__ == "__main__":
    main()


__all__ = [
    "VERSION", "STATUS", "LEGACY_TIMEOUT_SECONDS", "TIMEOUT_MULTIPLIER",
    "MIN_TIMEOUT_SECONDS", "MAX_TIMEOUT_SECONDS", "contract",
    "producer_sources", "create_calibration", "build",
    "read_saved_calibration", "main",
]
