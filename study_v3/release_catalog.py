"""Immutable asset inventory for the three-domain v6 candidate.

Pong and Kitchen keep their original v5 acceptance receipts. Their byte-for-byte
copies live in the v6 release package so the active study never loads an actor
from a mutable development output directory. Warehouse remains pending until
its independent v6 audit and held-out evaluation are complete.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import ast
import json
from pathlib import Path
import re
from threading import RLock

from . import RELEASE_ID

UPSTREAM_RELEASE_ID = "policylens-three-domain-20261014.v5.0"
PONG_V5_CONFIG_SHA256 = "1ade6a67b51884113594884ee27215af70d4a85528c91f03839e4e6961af7d42"
ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "releases" / RELEASE_ID / "catalog.json"
_WAREHOUSE_RELEASE = re.compile(r"policylens-three-domain-20261016\.v6\.(0|[1-9][0-9]*)\Z")
_WAREHOUSE_EVIDENCE_CACHE: dict[str, tuple[tuple, dict]] = {}
_WAREHOUSE_EVIDENCE_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class WarehouseReleaseSpec:
    release_id: str
    catalog_path: Path
    policy_ids: dict[str, str]


def warehouse_release_spec(release_id: str = RELEASE_ID) -> WarehouseReleaseSpec:
    """Address one immutable candidate without activating it for enrollment."""
    match = _WAREHOUSE_RELEASE.fullmatch(release_id) if isinstance(release_id, str) else None
    if match is None:
        raise ReleaseAssetUnavailable("warehouse_release_id_invalid")
    suffix = "v6" if match.group(1) == "0" else f"v6.{match.group(1)}"
    return WarehouseReleaseSpec(
        release_id=release_id,
        catalog_path=(CATALOG_PATH if release_id == RELEASE_ID else
                      ROOT / "releases" / release_id / "catalog.json"),
        policy_ids={variant: f"warehouse-{variant}-{suffix}"
                    for variant in ("raw", "regularized")},
    )


class ReleaseAssetUnavailable(RuntimeError):
    """The packaged model or its immutable provenance is incomplete."""


def digest(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ReleaseAssetUnavailable("release_asset_path_missing")
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ReleaseAssetUnavailable("release_asset_path_outside_repository")
    try:
        return sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReleaseAssetUnavailable("release_asset_missing") from exc


def catalog_sha256() -> str:
    return digest(str(CATALOG_PATH.relative_to(ROOT)))


def warehouse_runtime_source_closure_sha256() -> str:
    """Recompute the trainer's frozen source closure without importing Torch.

    The authoritative explicit file list is parsed from the trainer source, so
    adding a physics, observation, or training dependency changes the release
    gate rather than silently omitting it here.
    """
    trainer = ROOT / "backend/training/warehouse_v6_runner.py"
    try:
        source = trainer.read_text()
        module = ast.parse(source)
        values = [node.value for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "_SOURCE_FILES"
                          for target in node.targets)]
        if (len(values) != 1 or not isinstance(values[0], ast.Tuple)
                or len(values[0].elts) < 3):
            raise ValueError("Warehouse source file list unavailable")
        # Parse only the two literal, reviewable globs used by the trainer.
        # Do not execute trainer code or accept a substring containing a glob:
        # either could turn a changed closure into an accepted release hash.
        dynamic_roots = ("warehouse", "warehouse_legacy_v3")
        for index, directory in enumerate(dynamic_roots):
            candidate = values[0].elts[index]
            template = ast.parse(
                '(*(path.relative_to(_SOURCE_ROOT).as_posix() '
                'for path in sorted((_SOURCE_ROOT / "env" / '
                f'"{directory}").glob("*.py"))),)',
                mode="eval",
            ).body.elts[0]
            if ast.dump(candidate, include_attributes=False) != ast.dump(
                    template, include_attributes=False):
                raise ValueError("Warehouse dynamic source file list changed")
        if any(isinstance(node, ast.Starred) for node in values[0].elts[2:]):
            raise ValueError("Warehouse dynamic source file list changed")
        static_paths = tuple(ast.literal_eval(node) for node in values[0].elts[2:])
        dynamic_groups = tuple(
            tuple(path.relative_to(ROOT).as_posix()
                  for path in sorted((ROOT / "env" / directory).glob("*.py")))
            for directory in dynamic_roots
        )
        if any(not group for group in dynamic_groups):
            raise ValueError("Warehouse dynamic source directory empty")
        dynamic_paths = tuple(path for group in dynamic_groups for path in group)
        paths = dynamic_paths + static_paths
        if len(paths) != len(set(paths)):
            raise ValueError("Warehouse source file list duplicated")
        if any(not isinstance(path, str) or not path or path.startswith("/")
               or ".." in Path(path).parts or not (path.endswith(".py")
               or path.endswith(".json")
               or path == "requirements-warehouse-v6-training.txt") for path in paths):
            raise ValueError("Warehouse source file list invalid")
        hashes = {}
        for path in sorted(set(paths)):
            target = (ROOT / path).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError("Warehouse source file outside repository")
            hashes[path] = sha256(target.read_bytes()).hexdigest()
        return sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        raise ReleaseAssetUnavailable("warehouse_source_closure_unavailable") from exc


def _catalog(release_id: str = RELEASE_ID, *,
             catalog_path: Path | None = None) -> dict:
    return _catalog_with_digest(release_id, catalog_path=catalog_path)[0]


def _catalog_with_digest(release_id: str = RELEASE_ID, *,
                         catalog_path: Path | None = None) -> tuple[dict, str]:
    spec = warehouse_release_spec(release_id)
    path = catalog_path if catalog_path is not None else spec.catalog_path
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ReleaseAssetUnavailable("release_catalog_unavailable") from exc
    if (not isinstance(value, dict)
            or value.get("format") != "policylens-release-catalog.v1"
            or value.get("release_id") != release_id
            or value.get("upstream_release_id") != UPSTREAM_RELEASE_ID):
        raise ReleaseAssetUnavailable("release_catalog_invalid")
    return value, sha256(raw).hexdigest()


def _asset(path: str, expected: str, release_id: str = RELEASE_ID) -> None:
    prefix = f"releases/{release_id}/assets/"
    if (not isinstance(path, str) or not path.startswith(prefix)
            or not isinstance(expected, str) or len(expected) != 64
            or path.split("/")[-2] != expected
            or digest(path) != expected):
        raise ReleaseAssetUnavailable("release_asset_hash_mismatch")


def _catalog_section(catalog: dict, name: str) -> dict:
    """Treat a malformed nested catalog section as an unavailable release."""
    section = catalog.get(name)
    if not isinstance(section, dict):
        raise ReleaseAssetUnavailable("release_catalog_invalid")
    return section


def runtime_config(domain: str, release_id: str = RELEASE_ID) -> Path:
    """Read a frozen physical scenario config only from its verified package."""
    item = _catalog_section(_catalog(release_id), "runtime_configs").get(domain)
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ReleaseAssetUnavailable("release_runtime_config_missing")
    if domain == "pong" and item["sha256"] != PONG_V5_CONFIG_SHA256:
        raise ReleaseAssetUnavailable("release_pong_config_identity_mismatch")
    _asset(item["path"], item["sha256"], release_id)
    return ROOT / item["path"]


def policy_artifacts(domain: str) -> dict:
    """Return only complete packaged Pong/Kitchen entries for the active v6 study."""
    if domain not in ("pong", "kitchen"):
        raise ReleaseAssetUnavailable("release_domain_unavailable")
    catalog = _catalog()
    runtime_config(domain)
    entries = _catalog_section(catalog, "policies").get(domain)
    if not isinstance(entries, dict) or set(entries) != {"raw", "regularized"}:
        raise ReleaseAssetUnavailable("release_policy_pair_incomplete")
    for variant, entry in entries.items():
        if (not isinstance(entry, dict)
                or entry.get("policy_id") != f"{domain}-{variant}-v5"
                or entry.get("upstream_release_id") != UPSTREAM_RELEASE_ID):
            raise ReleaseAssetUnavailable("release_policy_identity_mismatch")
        for path_key, hash_key in (("actor_path", "actor_sha256"),
                                   ("training_report_path", "training_report_sha256"),
                                   ("acceptance_path", "acceptance_sha256")):
            _asset(entry.get(path_key), entry.get(hash_key))
        paths = entry.get("program_paths")
        if (not isinstance(paths, list)
                or len(paths) != (2 if variant == "regularized" else 0)
                or any(not isinstance(path, str) for path in paths)):
            raise ReleaseAssetUnavailable("release_role_programs_incomplete")
        program_hashes = entry.get("program_sha256")
        expected_keys = (set(paths) if domain == "pong" else {"human", "ai"})
        if (not isinstance(program_hashes, dict)
                or set(program_hashes) != (expected_keys if variant == "regularized" else set())):
            raise ReleaseAssetUnavailable("release_role_programs_incomplete")
        for role, path in zip(("human", "ai"), paths):
            key = path if domain == "pong" else role
            _asset(path, program_hashes[key])
        roles = entry.get("role_sha256")
        if (not isinstance(roles, dict) or set(roles) != {"human", "ai"}
                or any(not isinstance(value, str) or len(value) != 64
                       for value in roles.values())):
            raise ReleaseAssetUnavailable("release_role_hashes_incomplete")
    return deepcopy(entries)


def warehouse_policy_artifacts(release_id: str = RELEASE_ID) -> dict:
    """Resolve the accepted v6 pair only from the content-addressed release.

    The Warehouse training directory is deliberately not a runtime fallback.
    A pending catalog or a single missing asset leaves the entire pair closed.
    """
    from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256, MODEL_VERSION
    from env.warehouse.public_action_features_v614 import ACTION_PHYSICS_CONFIG_SHA256

    spec = warehouse_release_spec(release_id)
    catalog = _catalog(release_id)
    warehouse = _catalog_section(catalog, "warehouse")
    if (warehouse.get("status") != "accepted"
            or warehouse.get("policy_ids") != spec.policy_ids):
        raise ReleaseAssetUnavailable("warehouse_release_pending")
    entries = _catalog_section(catalog, "policies").get("warehouse")
    if not isinstance(entries, dict) or set(entries) != {"raw", "regularized"}:
        raise ReleaseAssetUnavailable("warehouse_policy_pair_incomplete")
    for variant, entry in entries.items():
        if (not isinstance(entry, dict)
                or entry.get("policy_id") != spec.policy_ids[variant]
                or entry.get("model_version") != MODEL_VERSION
                or entry.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or entry.get("action_physics_config_sha256") !=
                ACTION_PHYSICS_CONFIG_SHA256):
            raise ReleaseAssetUnavailable("warehouse_policy_identity_mismatch")
        for path_key, hash_key in (("actor_path", "actor_sha256"),
                                   ("training_report_path", "training_report_sha256"),
                                   ("acceptance_path", "acceptance_sha256")):
            _asset(entry.get(path_key), entry.get(hash_key), release_id)
        roles = entry.get("role_sha256")
        if (not isinstance(roles, dict) or set(roles) != {"human", "ai"}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in roles.values())):
            raise ReleaseAssetUnavailable("warehouse_role_hashes_incomplete")
        paths = entry.get("program_paths")
        hashes = entry.get("program_sha256")
        if (not isinstance(paths, list) or not isinstance(hashes, dict)
                or len(paths) != (2 if variant == "regularized" else 0)
                or set(hashes) != ({"robot_1", "robot_2"}
                                  if variant == "regularized" else set())):
            raise ReleaseAssetUnavailable("warehouse_role_programs_incomplete")
        for role, path in zip(("robot_1", "robot_2"), paths):
            _asset(path, hashes.get(role), release_id)
    warehouse_evidence(release_id)
    return deepcopy(entries)


def warehouse_evidence(release_id: str = RELEASE_ID, *,
                       catalog_path: Path | None = None) -> dict:
    """Return a verified copy of the release evidence."""
    return _warehouse_evidence_snapshot(release_id, catalog_path=catalog_path)[0]


def warehouse_evidence_snapshot(release_id: str = RELEASE_ID) -> tuple[dict, tuple]:
    """Return evidence and its revision from one verified cache entry.

    Consumers must recheck the revision after using the evidence before they
    cache a semantic acceptance decision. Separate evidence and revision
    reads can otherwise bind an old valid proof to a newer catalog.
    """
    evidence, revision = _warehouse_evidence_snapshot(release_id)
    if revision is None:
        raise ReleaseAssetUnavailable("warehouse_evidence_revision_missing")
    return evidence, revision


def _warehouse_evidence_snapshot(release_id: str = RELEASE_ID, *,
                                 catalog_path: Path | None = None
                                 ) -> tuple[dict, tuple | None]:
    """Verify every original Warehouse v6 admission receipt in the package."""
    from experiments.warehouse.protocol import BUDGETS, DEVELOPMENT_PAIR_SEEDS

    catalog, catalog_digest = _catalog_with_digest(
        release_id, catalog_path=catalog_path)
    section = _catalog_section(catalog, "warehouse")
    if section.get("status") != "accepted":
        raise ReleaseAssetUnavailable("warehouse_release_pending")
    budget = section.get("locked_budget")
    if budget not in BUDGETS:
        raise ReleaseAssetUnavailable("warehouse_budget_lock_missing")
    evidence = section.get("evidence")
    if not isinstance(evidence, dict):
        raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
    expected = {
        "phase_receipts": {
            "budget.lock.json", "formal.selection.json",
            "formal.training.6101.json", "formal.training.6102.json",
            "formal.training.6103.json", "program.preaudit.json",
            "final_program_audit.json", "heldout.json"},
        "one_use_claims": {"final_program_audit", "heldout"},
        "pair_reports": {"6101", "6102", "6103"},
        "development_receipts": {str(value) for value in BUDGETS
                                 if value <= budget},
        "training_assets": {"6101", "6102", "6103"},
        "development_training_assets": {str(value) for value in BUDGETS
                                        if value <= budget},
        "development_pair_reports": {str(value) for value in BUDGETS
                                     if value <= budget},
        "program_traces": {"development", "preaudit", "final_audit"},
        "final_fit_traces": {"development", "formal"},
        "final_fit_replays": {"development", "formal"},
        "quality_replays": {"development", "formal", "heldout"},
    }
    if set(evidence) != set(expected):
        raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
    revision = None
    if catalog_path is None:
        revision = _warehouse_evidence_revision(release_id, evidence)
        if revision[0] != catalog_digest:
            raise ReleaseAssetUnavailable("warehouse_catalog_changed_during_validation")
        with _WAREHOUSE_EVIDENCE_CACHE_LOCK:
            cached = _WAREHOUSE_EVIDENCE_CACHE.get(release_id)
            if cached is not None and cached[0] == revision:
                return deepcopy(cached[1]), cached[0]
    for category, keys in expected.items():
        rows = evidence[category]
        if not isinstance(rows, dict) or set(rows) != keys:
            raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
        if category == "program_traces":
            if (not all(isinstance(rows[name], dict) for name in rows)
                    or set(rows["development"]) != set(map(str, (
                    value for value in BUDGETS if value <= budget)))
                    or any(set(rows[name]) != {"robot_1", "robot_2"}
                           for name in ("preaudit", "final_audit"))):
                raise ReleaseAssetUnavailable("warehouse_program_traces_incomplete")
            for name in ("preaudit", "final_audit"):
                for item in rows[name].values():
                    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                        raise ReleaseAssetUnavailable("warehouse_program_traces_incomplete")
                    _asset(item["path"], item["sha256"], release_id)
            for budget_key, by_seed in rows["development"].items():
                if (not isinstance(by_seed, dict)
                        or set(by_seed) != set(map(str, DEVELOPMENT_PAIR_SEEDS))):
                    raise ReleaseAssetUnavailable("warehouse_program_traces_incomplete")
                for by_role in by_seed.values():
                    if (not isinstance(by_role, dict)
                            or set(by_role) not in (set(), {"robot_1", "robot_2"})):
                        raise ReleaseAssetUnavailable("warehouse_program_traces_incomplete")
                    for item in by_role.values():
                        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                            raise ReleaseAssetUnavailable("warehouse_program_traces_incomplete")
                        _asset(item["path"], item["sha256"], release_id)
        elif category in ("final_fit_traces", "final_fit_replays"):
            if (set(rows["formal"]) != {"6101", "6102", "6103"}
                    or set(rows["development"]) != {
                        str(value) for value in BUDGETS if value <= budget}):
                raise ReleaseAssetUnavailable("warehouse_final_fit_inventory_incomplete")
            groups = [rows["formal"]]
            for budget_rows in rows["development"].values():
                if (not isinstance(budget_rows, dict)
                        or set(budget_rows) != set(map(str, DEVELOPMENT_PAIR_SEEDS))):
                    raise ReleaseAssetUnavailable("warehouse_final_fit_inventory_incomplete")
                groups.append(budget_rows)
            for group in groups:
                for item in group.values():
                    if category == "final_fit_traces":
                        if (not isinstance(item, dict)
                                or set(item) != {"fit", "calibration",
                                                 "eligibility", "validation"}):
                            raise ReleaseAssetUnavailable(
                                "warehouse_final_fit_inventory_incomplete")
                        assets = item.items()
                    else:
                        assets = (("final_fit_replay.receipt", item),)
                    for name, asset in assets:
                        if (not isinstance(asset, dict)
                                or set(asset) != {"path", "sha256"}
                                or Path(asset["path"]).name != f"{name}.jsonl"
                                and not (category == "final_fit_replays" and
                                         Path(asset["path"]).name ==
                                         "final_fit_replay.receipt.json")):
                            raise ReleaseAssetUnavailable(
                                "warehouse_final_fit_inventory_incomplete")
                        _asset(asset["path"], asset["sha256"], release_id)
        elif category == "quality_replays":
            if (set(rows["formal"]) != {"6101", "6102", "6103"}
                    or set(rows["development"]) != {
                        str(value) for value in BUDGETS if value <= budget}
                    or set(rows["heldout"]) != {"raw", "regularized"}):
                raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
            for seed, variants in rows["formal"].items():
                if not isinstance(variants, dict) or set(variants) != {"raw", "regularized"}:
                    raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                for variant, item in variants.items():
                    if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                            or Path(item["path"]).name !=
                            f"formal.{seed}.{variant}.quality-replay.json"):
                        raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                    _asset(item["path"], item["sha256"], release_id)
            for budget_key, pairs in rows["development"].items():
                if (not isinstance(pairs, dict)
                        or set(pairs) != set(map(str, DEVELOPMENT_PAIR_SEEDS))):
                    raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                for seed, variants in pairs.items():
                    if (not isinstance(variants, dict)
                            or set(variants) not in (set(), {"raw", "regularized"})):
                        raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                    for variant, item in variants.items():
                        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                                or Path(item["path"]).name !=
                                f"development.{budget_key}.{seed}.{variant}.quality-replay.json"):
                            raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                        _asset(item["path"], item["sha256"], release_id)
            for variant, tasks in rows["heldout"].items():
                if not isinstance(tasks, dict) or set(tasks) != {"1", "2", "3"}:
                    raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                for task, item in tasks.items():
                    if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                            or Path(item["path"]).name !=
                            f"heldout.{variant}.task{task}.quality-replay.json"):
                        raise ReleaseAssetUnavailable("warehouse_quality_replays_incomplete")
                    _asset(item["path"], item["sha256"], release_id)
        elif category in ("development_receipts", "development_training_assets",
                        "development_pair_reports"):
            for budget_key, budget_rows in rows.items():
                if (not isinstance(budget_rows, dict) or
                        set(budget_rows) != set(map(str, DEVELOPMENT_PAIR_SEEDS))):
                    raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
                for seed, item in budget_rows.items():
                    if category == "development_receipts":
                        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                                or not isinstance(item["path"], str)
                                or Path(item["path"]).name !=
                                f"development.{budget_key}.{seed}.json"):
                            raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
                        _asset(item["path"], item["sha256"], release_id)
                    elif category == "development_training_assets":
                        _warehouse_training_assets(
                            item, allow_missing_regularized_programs=
                            int(budget_key) < budget,
                            release_id=release_id)
                    else:
                        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                                or not isinstance(item["path"], str)
                                or Path(item["path"]).name != "pair.training.json"):
                            raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
                        _asset(item["path"], item["sha256"], release_id)
        elif category == "training_assets":
            for item in rows.values():
                _warehouse_training_assets(item, release_id=release_id)
        else:
            for item in rows.values():
                if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                    raise ReleaseAssetUnavailable("warehouse_evidence_incomplete")
                _asset(item["path"], item["sha256"], release_id)
    if catalog_path is None:
        # A file changing while the initial full SHA audit ran cannot seed a
        # valid cache. A later request must validate the new bytes afresh.
        if _warehouse_evidence_revision(release_id, evidence) != revision:
            raise ReleaseAssetUnavailable("warehouse_evidence_changed_during_validation")
        with _WAREHOUSE_EVIDENCE_CACHE_LOCK:
            _WAREHOUSE_EVIDENCE_CACHE[release_id] = (revision, deepcopy(evidence))
    return deepcopy(evidence), revision


def warehouse_evidence_revision(release_id: str = RELEASE_ID) -> tuple:
    """Return the verified catalog and file identity for the admission cache."""
    return warehouse_evidence_snapshot(release_id)[1]


def _warehouse_evidence_revision(release_id: str, evidence: dict) -> tuple:
    def assets(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                yield value["path"], value["sha256"]
            else:
                for item in value.values():
                    yield from assets(item)
        elif isinstance(value, list):
            for item in value:
                yield from assets(item)

    try:
        catalog_path = warehouse_release_spec(release_id).catalog_path
        catalog_hash = sha256(catalog_path.read_bytes()).hexdigest()
        file_rows = []
        for path, expected in sorted(set(assets(evidence))):
            target = (ROOT / path).resolve()
            if not target.is_relative_to(ROOT):
                raise ReleaseAssetUnavailable("warehouse_evidence_path_invalid")
            stat = target.stat()
            file_rows.append((path, expected, stat.st_dev, stat.st_ino,
                              stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return catalog_hash, tuple(file_rows)
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseAssetUnavailable("warehouse_evidence_stat_unavailable") from exc


def _warehouse_training_assets(
        item: object, *, allow_missing_regularized_programs: bool = False,
        release_id: str = RELEASE_ID) -> None:
    if not isinstance(item, dict) or set(item) != {"raw", "regularized"}:
        raise ReleaseAssetUnavailable("warehouse_training_assets_incomplete")
    for variant, row in item.items():
        expected_programs = ({"robot_1", "robot_2"}
                             if variant == "regularized" else set())
        if (not isinstance(row, dict)
                or set(row) != {"actor", "training_report", "programs"}
                or not isinstance(row["programs"], dict)
                or set(row["programs"]) not in (
                    (expected_programs, set()) if
                    variant == "regularized" and allow_missing_regularized_programs
                    else (expected_programs,))):
            raise ReleaseAssetUnavailable("warehouse_training_assets_incomplete")
        assets = [row["actor"], row["training_report"],
                  *row["programs"].values()]
        for asset in assets:
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
                raise ReleaseAssetUnavailable("warehouse_training_assets_incomplete")
            _asset(asset["path"], asset["sha256"], release_id)


def source_provenance() -> dict:
    """Describe original v5 sources without making runtime execution depend on them."""
    catalog = _catalog()
    index = catalog.get("source_asset_index")
    if not isinstance(index, dict) or not index:
        raise ReleaseAssetUnavailable("release_source_provenance_missing")
    for source_path, item in index.items():
        if not isinstance(source_path, str) or not isinstance(item, dict):
            raise ReleaseAssetUnavailable("release_source_provenance_invalid")
        _asset(item.get("bundle_path"), item.get("sha256"))
    return deepcopy(index)
