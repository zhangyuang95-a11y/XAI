"""Audit the source-closure-only supersession of the first v12 registry.

The first registry was rejected by the label-blind projection reader before
replay because its exclusion-digest map omitted an already bound historical
v10 observation digest.  The same deterministic identities were rebuilt after
adding that digest.  This receipt preserves both byte identities and proves
that the selected scenes did not change and that the rejection occurred before
new replay, labels, probabilities, a one-shot outer claim, protected-final
access, or private-salt access.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v12 as projection_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-v12-registry-supersession.v1"
STATUS = "superseded_before_fresh_projection_replay"
RECEIPT_NAME = "supersession_receipt.json"
MAX_JSON_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((
        Path(__file__).resolve(), Path(projection_api.__file__).resolve(),
    )).items()))


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: child for key, child in value.items()
                                   if key != "content_sha256"}))


def _strict_json(path: str | Path, *, expected_sha256: str,
                 label: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().absolute()
    if (not candidate.is_file() or candidate.is_symlink()
            or candidate.resolve() != candidate
            or candidate.stat().st_size <= 0
            or candidate.stat().st_size > MAX_JSON_BYTES
            or type(expected_sha256) is not str
            or _HEX.fullmatch(expected_sha256) is None
            or file_hash(candidate) != expected_sha256):
        raise ValueError("Exact " + label + " bytes required")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(candidate.read_text(encoding="utf-8"),
                           object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               ValueError("Non-finite JSON in " + label)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(value, dict) or not _content_valid(value):
        raise ValueError(label + " content digest differs")
    return candidate, value


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "supersession_reason": (
            "first registry omitted the consumed-v10 observation digest from "
            "its exclusion-digest map"),
        "selected_identity_change_permitted": False,
        "old_registry_bytes_preserved": True,
        "failure_before_fresh_projection_replay": True,
        "labels_or_probabilities_accessed": False,
        "outer_attempt_claimed": False,
        "protected_final_access": False,
        "private_salt_access": False,
        "formal_ready": False,
    }


def create_receipt(
    *, old_registry_path: str | Path, old_registry_sha256: str,
    old_report_path: str | Path, old_report_sha256: str,
    old_prior_projection_path: str | Path, old_prior_projection_sha256: str,
    new_registry_path: str | Path, new_registry_sha256: str,
    new_report_path: str | Path, new_report_sha256: str,
    new_prior_projection_path: str | Path, new_prior_projection_sha256: str,
    failed_projection_output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    old_registry_file, old_registry = _strict_json(
        old_registry_path, expected_sha256=old_registry_sha256,
        label="superseded v12 registry")
    old_report_file, old_report = _strict_json(
        old_report_path, expected_sha256=old_report_sha256,
        label="superseded v12 registry report")
    old_projection_file, old_projection = _strict_json(
        old_prior_projection_path, expected_sha256=old_prior_projection_sha256,
        label="superseded v12 prior projection")
    new_registry_file, new_registry = _strict_json(
        new_registry_path, expected_sha256=new_registry_sha256,
        label="replacement v12 registry")
    new_report_file, new_report = _strict_json(
        new_report_path, expected_sha256=new_report_sha256,
        label="replacement v12 registry report")
    new_projection_file, new_projection = _strict_json(
        new_prior_projection_path, expected_sha256=new_prior_projection_sha256,
        label="replacement v12 prior projection")
    old_ids = old_registry.get("selected_outer_identities")
    new_ids = new_registry.get("selected_outer_identities")
    old_digests = old_registry.get("exclusion_digests")
    new_digests = new_registry.get("exclusion_digests")
    missing = "consumed_v10_outer_observation_hashes_sha256"
    if (not isinstance(old_ids, list) or len(old_ids) != 64
            or new_ids != old_ids
            or old_report.get("selection", {}).get("selected_identity_sha256")
                != digest(old_ids)
            or new_report.get("selection", {}).get("selected_identity_sha256")
                != digest(new_ids)
            or old_report.get("registry_file_sha256") != old_registry_sha256
            or new_report.get("registry_file_sha256") != new_registry_sha256
            or old_report.get("registry_content_sha256")
                != old_registry["content_sha256"]
            or new_report.get("registry_content_sha256")
                != new_registry["content_sha256"]
            or not isinstance(old_digests, Mapping)
            or not isinstance(new_digests, Mapping)
            or missing in old_digests
            or new_digests.get(missing)
                != new_registry.get("bindings", {}).get(
                    "consumed_v10_outer_observation_hashes_sha256")
            or {key: value for key, value in new_digests.items()
                if key != missing} != dict(old_digests)
            or old_projection.get("outer_observation_hashes")
                != new_projection.get("outer_observation_hashes")
            or old_projection.get("outer_observation_hashes_sha256")
                != new_projection.get("outer_observation_hashes_sha256")):
        raise ValueError("V12 registry supersession is not source-closure-only")
    failed_output = Path(failed_projection_output).expanduser().absolute()
    if failed_output.exists() or failed_output.is_symlink():
        raise ValueError("Failed projection output must remain unpublished")
    build_source = inspect.getsource(projection_api.build)
    validation_call_index = build_source.index("_validate_registry(")
    replay_call_index = build_source.index("_replay_outer(")
    if validation_call_index > replay_call_index:
        raise RuntimeError("Projection validation no longer precedes replay")
    bindings = {
        "old_registry_sha256": file_hash(old_registry_file),
        "old_registry_content_sha256": old_registry["content_sha256"],
        "old_report_sha256": file_hash(old_report_file),
        "old_report_content_sha256": old_report["content_sha256"],
        "old_prior_projection_sha256": file_hash(old_projection_file),
        "old_prior_projection_content_sha256": old_projection["content_sha256"],
        "new_registry_sha256": file_hash(new_registry_file),
        "new_registry_content_sha256": new_registry["content_sha256"],
        "new_report_sha256": file_hash(new_report_file),
        "new_report_content_sha256": new_report["content_sha256"],
        "new_prior_projection_sha256": file_hash(new_projection_file),
        "new_prior_projection_content_sha256": new_projection["content_sha256"],
        "selected_identity_sha256": digest(old_ids),
        "contract_sha256": digest(contract()),
    }
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": bindings,
        "identity_audit": {
            "scene_count": len(old_ids),
            "old_selected_identity_sha256": digest(old_ids),
            "new_selected_identity_sha256": digest(new_ids),
            "selected_identities_exactly_equal": True,
        },
        "repair_audit": {
            "added_field": missing,
            "added_value": new_digests[missing],
            "all_other_exclusion_digests_exactly_equal": True,
            "prior_projection_observation_hashes_exactly_equal": True,
        },
        "failure_boundary": {
            "failure_stage": "registry_consumer_pre_replay_self_audit",
            "failed_projection_output": str(failed_output),
            "fresh_projection_replay_started": False,
            "fresh_projection_output_published": False,
            "labels_or_probabilities_accessed": False,
            "outer_attempt_claimed": False,
            "protected_final_access": False,
            "private_salt_access": False,
            "projection_consumer_file_sha256": file_hash(
                Path(projection_api.__file__).resolve()),
            "projection_consumer_build_source_sha256": sha256(
                build_source.encode("utf-8")).hexdigest(),
            "registry_validation_call_index": validation_call_index,
            "fresh_replay_call_index": replay_call_index,
            "registry_validation_precedes_fresh_replay": True,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    return receipt


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    value = create_receipt(**kwargs)
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent:
        raise ValueError("Supersession output parent is unsafe")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        target = temporary / RECEIPT_NAME
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((canonical(value) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(value)


def read_saved_receipt(
    path: str | Path, *, expected_receipt_sha256: str,
) -> dict[str, Any]:
    _receipt_file, value = _strict_json(
        path, expected_sha256=expected_receipt_sha256,
        label="v12 registry supersession receipt")
    identity = value.get("identity_audit")
    boundary = value.get("failure_boundary")
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or value.get("contract") != contract()
            or value.get("producer_sources") != producer_sources()
            or value.get("producer_sources_sha256")
                != digest(value["producer_sources"])
            or not isinstance(identity, Mapping)
            or identity.get("scene_count") != 64
            or identity.get("selected_identities_exactly_equal") is not True
            or identity.get("old_selected_identity_sha256")
                != identity.get("new_selected_identity_sha256")
            or not isinstance(boundary, Mapping)
            or boundary.get("registry_validation_precedes_fresh_replay") is not True
            or boundary.get("fresh_projection_replay_started") is not False
            or boundary.get("fresh_projection_output_published") is not False
            or boundary.get("labels_or_probabilities_accessed") is not False
            or boundary.get("outer_attempt_claimed") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("private_salt_access") is not False
            or value.get("formal_ready") is not False):
        raise ValueError("V12 registry supersession receipt differs")
    return deepcopy(value)


__all__ = ["VERSION", "STATUS", "RECEIPT_NAME", "contract",
           "producer_sources", "create_receipt", "build",
           "read_saved_receipt"]
