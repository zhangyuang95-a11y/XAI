"""Reproduce the burned-v12 final isolation failure from promoted evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_public_v13 as closeout_api
from backend.training import warehouse_r41_diagnostic_final_once_v9 as final_api
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as old_projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training import warehouse_r41_diagnostic_rcpd_v12_fit_selector as selector_api
from backend.training import warehouse_r41_diagnostic_rcpd_v12_outer_once as outer_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-isolation-diagnosis.v13"
STATUS = "burned_v12_final_technical_isolation_failure_reproduced"
REPORT_NAME = "diagnosis.json"
EXPECTED_FINAL_UNIQUE = 31_645
EXPECTED_DEVELOPMENT_OVERLAP = 3
EXPECTED_OUTER_OVERLAP = 23
EXPECTED_UNION_OVERLAP = 26
MAX_JSON_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "inputs_already_permanently_closed_and_development_exposed": True,
        "protected_materializer_invoked": False,
        "protected_salt_access": False,
        "new_outer_or_final_attempt": False,
        "diagnosis_uses_only_one_way_hashes_and_public_identities": True,
        "program_metrics_or_actor_outputs_used_for_cause_selection": False,
        "formal_ready": False,
    }


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _regular(value: str | Path, label: str, *, expected_sha256: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > MAX_JSON_BYTES
            or file_hash(path) != expected_sha256):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _content_valid(value: Mapping[str, Any]) -> bool:
    return (type(value.get("content_sha256")) is str
            and _HEX.fullmatch(value["content_sha256"]) is not None
            and value["content_sha256"] == digest({
                key: child for key, child in value.items()
                if key != "content_sha256"}))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result
    value = json.loads(path.read_text("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError("Non-finite JSON in " + label)))
    if not isinstance(value, dict) or not _content_valid(value):
        raise ValueError(label + " semantics differ")
    return value


def _code_contract() -> dict[str, Any]:
    old = inspect.getsource(old_projection_api._exact_final_workload_observations)
    collector = inspect.getsource(rows_api._collect)
    facts = {
        "selection_projection_partner_rng": (
            "17000 + partner_index * 1000 + scene_index"),
        "audit_collector_partner_rng": (
            "41_900_000 + scene_index * 101 + partner_index"),
        "selection_projection_critical_anchor_period": 10,
        "audit_collector_critical_anchor_period": 5,
        "selection_projection_all_three_partners": True,
        "audit_collector_all_three_partners": True,
        "projection_contracts_match": False,
    }
    required = (
        "17000 + partner_index * 1000 + scene_index" in old
        and "env.state.frame % 10 == 0" in old
        and "41_900_000 + scene_index * 101 + partner_index" in collector
        and "env.state.frame % 5 == 0" in collector
    )
    if not required:
        raise ValueError("Historical projection/collector code contract changed")
    return facts


def create_diagnosis(
    *, closeout_path: str | Path, expected_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
    base_development_rows_path: str | Path,
    expected_base_development_rows_sha256: str,
    v11_rows_path: str | Path,
    selector_report_path: str | Path, expected_selector_report_sha256: str,
    prior_outer_projection_path: str | Path,
    expected_prior_outer_projection_sha256: str,
    expected_prior_outer_projection_content_sha256: str,
) -> dict[str, Any]:
    closeout = closeout_api.read_saved_closeout_public(
        closeout_path, expected_closeout_sha256=expected_closeout_sha256,
        permanent_closeout_registry=permanent_closeout_registry)
    base_path = Path(base_development_rows_path).expanduser().absolute()
    v11_path = Path(v11_rows_path).expanduser().absolute()
    base = outer_api._safe_row_projection(
        base_path, expected_sha256=expected_base_development_rows_sha256,
        fields=frozenset(("observation_hashes", "scene_fingerprints")),
        label="closed v12 base development")
    v11 = outer_api._safe_row_projection(
        v11_path,
        expected_sha256=closeout["consumed_v11_outer"]["rows_sha256"],
        fields=frozenset(("observation_hashes", "scene_fingerprints")),
        label="closed v12 promoted v11 development")
    selector_path = _regular(
        selector_report_path, "closed v12 selector report",
        expected_sha256=expected_selector_report_sha256)
    selector = _read_json(selector_path, "closed v12 selector report")
    prior_path = _regular(
        prior_outer_projection_path, "closed v12 prior projection",
        expected_sha256=expected_prior_outer_projection_sha256)
    prior = selector_api.read_prior_outer_hash_projection(
        prior_path, expected_sha256=expected_prior_outer_projection_sha256,
        expected_content_sha256=expected_prior_outer_projection_content_sha256)

    decode = outer_api._decode
    promoted_hashes = closeout["consumed_v11_outer"][
        "observation_hash_projection"]["outer_observation_hashes"]
    fresh_hashes = closeout["consumed_v12_outer"][
        "observation_hash_projection"]["outer_observation_hashes"]
    development, development_scenes = (
        final_api._retained_combined_development_hashes(
            base_ordered=decode(base["observation_hashes"], "base hashes"),
            base_scene_ordered=decode(
                base["scene_fingerprints"], "base scenes"),
            promoted_ordered=decode(v11["observation_hashes"], "v11 hashes"),
            promoted_scene_ordered=decode(
                v11["scene_fingerprints"], "v11 scenes"),
            prior_unique_hashes=prior["outer_observation_hashes"],
            promoted_unique_hashes=promoted_hashes,
            fresh_unique_hashes=fresh_hashes,
            validation_wins=selector["development"]["validation_wins"]))
    outer = final_api._v12_outer_audit_hashes(
        prior_unique_hashes=prior["outer_observation_hashes"],
        promoted_unique_hashes=promoted_hashes,
        fresh_unique_hashes=fresh_hashes)
    final_hashes = set(closeout["burned_final"][
        "observation_hash_projection"]["outer_observation_hashes"])
    final_scenes = {
        row["fingerprint"] for row in closeout["burned_final"]["identities"]}
    development_overlap = sorted(final_hashes & development)
    outer_overlap = sorted(final_hashes & outer)
    union_overlap = sorted(final_hashes & (development | outer))
    fresh_overlap = sorted(final_hashes & set(fresh_hashes))
    historical_outer = set(prior["outer_observation_hashes"]) - set(promoted_hashes)
    historical_overlap = sorted(final_hashes & historical_outer)
    expected = (
        len(final_hashes) == EXPECTED_FINAL_UNIQUE
        and len(development_overlap) == EXPECTED_DEVELOPMENT_OVERLAP
        and len(outer_overlap) == EXPECTED_OUTER_OVERLAP
        and len(union_overlap) == EXPECTED_UNION_OVERLAP
        and len(fresh_overlap) == 0
        and historical_overlap == outer_overlap
        and not (development & outer)
        and not (final_scenes & development_scenes)
    )
    if not expected:
        raise ValueError("Burned-v12 final isolation failure no longer reproduces")
    sources = producer_sources()
    report: dict[str, Any] = {
        "version": VERSION, "status": STATUS, "contract": contract(),
        "bindings": {
            "closeout_sha256": expected_closeout_sha256,
            "closeout_content_sha256": closeout["content_sha256"],
            "base_development_rows_sha256": file_hash(base_path),
            "v11_rows_sha256": file_hash(v11_path),
            "selector_report_sha256": file_hash(selector_path),
            "prior_outer_projection_sha256": file_hash(prior_path),
            "prior_outer_projection_content_sha256": prior["content_sha256"],
            "burned_final_rows_semantic_sha256": closeout["burned_final"][
                "rows_semantic_sha256"],
            "burned_final_identities_sha256": closeout["burned_final"][
                "identities_sha256"],
        },
        "reproduction": {
            "final_unique_observations": len(final_hashes),
            "retained_development_unique_observations": len(development),
            "outer_unique_observations": len(outer),
            "development_outer_overlap": len(development & outer),
            "final_development_overlap": len(development_overlap),
            "final_development_overlap_sha256": digest(development_overlap),
            "final_outer_overlap": len(outer_overlap),
            "final_outer_overlap_sha256": digest(outer_overlap),
            "final_fresh_v12_outer_overlap": len(fresh_overlap),
            "final_historical_outer_overlap": len(historical_overlap),
            "final_union_overlap": len(union_overlap),
            "final_union_overlap_sha256": digest(union_overlap),
            "final_prior_scene_overlap": len(final_scenes & development_scenes),
            "audit_rows_expected_exception": (
                "ValueError: Final scenes or public observations overlap prior splits"),
            "technical_failure_reproduced": True,
        },
        "root_cause": {
            **_code_contract(),
            "summary": (
                "The materializer screened a different partner RNG and a sparser "
                "intervention cadence than the final collector, so its accepted "
                "hash set was not the audit input set."),
            "explanation_nine_gates_passed": closeout[
                "failure_audit"]["nine_gate"]["passed"],
            "independent_physical_replay_passed": closeout[
                "failure_audit"]["physical_counterfactual_audit"]["passed"],
        },
        "required_fix": {
            "shared_observation_only_projection": True,
            "collector_partner_rng_exact": True,
            "three_partners": list(rows_api.PARTNERS),
            "scene_offset": final_api.FINAL_SCENE_OFFSET,
            "critical_anchor_period": 5,
            "all_nonterminal_player_action_intervention_endpoints": True,
            "program_or_target_access_during_selection": False,
            "parity_required_before_new_final_claim": True,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    return report


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def build(*, output: str | Path, permanent_registry: str | Path,
          **kwargs: Any) -> dict[str, Any]:
    report = create_diagnosis(**kwargs)
    key = digest({
        "version": VERSION,
        "closeout_sha256": report["bindings"]["closeout_sha256"],
        "producer_sources_sha256": report["producer_sources_sha256"],
    })
    report = deepcopy(report); report["diagnosis_key"] = key
    report["content_sha256"] = digest({
        name: value for name, value in report.items() if name != "content_sha256"})
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "diagnosis output parent")
    permanent = _directory(permanent_registry, "permanent diagnosis registry")
    campaign = permanent / key
    if destination.exists() or destination.is_symlink() or campaign.exists():
        raise FileExistsError("v13 isolation diagnosis already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / REPORT_NAME, report)
        os.mkdir(campaign, 0o700)
        _write_exclusive(campaign / REPORT_NAME, report)
        if (temporary / REPORT_NAME).read_bytes() != (campaign / REPORT_NAME).read_bytes():
            raise RuntimeError("Permanent diagnosis bytes differ")
        os.rename(temporary, destination); temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "closeout", "expected-closeout-sha256", "permanent-closeout-registry",
        "base-development-rows", "expected-base-development-rows-sha256",
        "v11-rows", "selector-report", "expected-selector-report-sha256",
        "prior-outer-projection", "expected-prior-outer-projection-sha256",
        "expected-prior-outer-projection-content-sha256",
        "permanent-registry", "output"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    result = build(
        output=args.pop("output"), permanent_registry=args.pop("permanent_registry"),
        **{name + "_path" if name in {
            "closeout", "base_development_rows", "v11_rows", "selector_report",
            "prior_outer_projection"} else name: value
           for name, value in args.items()})
    print(canonical({"status": result["status"],
                     "diagnosis_key": result["diagnosis_key"],
                     "overlap": result["reproduction"]["final_union_overlap"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "STATUS", "REPORT_NAME", "contract",
           "producer_sources", "create_diagnosis", "build", "main"]
