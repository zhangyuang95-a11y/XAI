#!/usr/bin/env python3
"""Package a fully accepted Warehouse pair without changing any evidence bytes.

The output catalog remains pending until all content-addressed assets and the
independent acceptance verifier pass against a staged catalog. Existing v5
Pong and Kitchen entries are copied through unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.warehouse import protocol as p
from study_v3 import release_catalog as catalog_module


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = catalog_module.RELEASE_ID
CATALOG = catalog_module.CATALOG_PATH
ASSETS = CATALOG.parent / "assets"


def _read(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source(path: str, expected: str, output_root: Path, basename: str | None = None) -> Path:
    if (not isinstance(path, str) or not isinstance(expected, str)
            or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("unbound Warehouse source asset")
    source = Path(path).resolve()
    if (not source.is_relative_to(output_root) or not source.is_file()
            or (basename is not None and source.name != basename)
            or _sha(source) != expected):
        raise ValueError(f"Warehouse source asset changed: {source}")
    return source


def _bundle(path: str, expected: str, output_root: Path,
            basename: str | None = None, *,
            assets_root: Path = ASSETS) -> dict[str, str]:
    source = _source(path, expected, output_root, basename)
    destination = assets_root / expected / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha(destination) != expected:
            raise ValueError(f"content-addressed asset changed: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(source.read_bytes())
            if _sha(temporary) != expected:
                raise ValueError("source asset changed during copy")
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination.relative_to(ROOT)), "sha256": expected}


def _training_assets(variants: dict, output_root: Path, *,
                     allow_missing_regularized_programs: bool = False,
                     assets_root: Path = ASSETS) -> dict:
    if set(variants) != set(p.VARIANTS):
        raise ValueError("Warehouse paired training assets incomplete")
    assets = {}
    for variant in p.VARIANTS:
        row = variants[variant]
        artifact = row["actor_artifact"]
        programs = row.get("program_artifacts") or {}
        allowed = ({"robot_1", "robot_2"} if variant == "regularized" else set())
        if (set(programs) != allowed and
                not (variant == "regularized" and
                     allow_missing_regularized_programs and
                     set(programs) < allowed)):
            raise ValueError("Warehouse role programs incomplete")
        assets[variant] = {
            "actor": _bundle(artifact["path"], artifact["sha256"], output_root,
                             assets_root=assets_root),
            "training_report": _bundle(row["training_report_path"],
                                       row["training_report_sha256"], output_root,
                                       assets_root=assets_root),
            "programs": {role: _bundle(programs[role]["path"],
                                       programs[role]["sha256"], output_root,
                                       assets_root=assets_root)
                         for role in programs},
        }
    return assets


def _audit_trace_assets(audits: dict, output_root: Path, *,
                        required: bool, assets_root: Path) -> dict:
    """Keep the exact pre-action rows behind each program-audit summary."""
    if not audits and not required:
        return {}
    if not isinstance(audits, dict) or set(audits) != set(p.ROLES):
        raise ValueError("Warehouse role audit traces incomplete")
    assets = {}
    for role in p.ROLES:
        row = audits[role]
        path = row.get("trace_path")
        digest = row.get("trace_sha256")
        if (not isinstance(path, str) or not path.endswith(".jsonl")
                or row.get("trace_row_count") != row.get("sample_count")):
            raise ValueError("Warehouse role audit trace identity incomplete")
        assets[role] = _bundle(path, digest, output_root,
                               assets_root=assets_root)
    return assets


def _final_fit_trace_assets(pair: dict, output_root: Path, *,
                            assets_root: Path) -> dict:
    """Bundle every post-selection fitting episode, including failed fits."""
    regularized = pair["variants"]["regularized"]
    reports = regularized.get("final_fresh_program_reports")
    if not isinstance(reports, dict) or set(reports) != set(p.ROLES):
        raise ValueError("Warehouse final program reports incomplete")
    schedule = p.final_program_fit_seed_schedule(
        pair["release_id"], pair["pair_seed"])
    traces = reports[p.ROLES[0]].get("final_fit_traces")
    if (not isinstance(traces, dict) or set(traces) != set(schedule)
            or any(reports[role].get("final_fit_traces") != traces
                   for role in p.ROLES)):
        raise ValueError("Warehouse final fit split inventory incomplete")
    assets = {}
    for split, seeds in schedule.items():
        claim = traces[split]
        if (not isinstance(claim, dict)
                or set(claim) != {"path", "sha256", "row_count", "seeds"}
                or claim["seeds"] != seeds
                or type(claim["row_count"]) is not int
                or claim["row_count"] < len(seeds)):
            raise ValueError("Warehouse final fit split identity incomplete")
        assets[split] = _bundle(claim["path"], claim["sha256"],
                                output_root, f"{split}.jsonl",
                                assets_root=assets_root)
    return assets


def _evidence(receipt: dict, output_root: Path, *,
              release_id: str = RELEASE_ID,
              assets_root: Path = ASSETS) -> dict:
    phases = receipt["phase_receipt_path"]
    phase_hashes = receipt["phase_receipt_sha256"]
    phase_names = {"budget.lock.json", "formal.selection.json",
                   *(f"formal.training.{seed}.json" for seed in p.FORMAL_PAIR_SEEDS),
                   "program.preaudit.json", "final_program_audit.json", "heldout.json"}
    if set(phases) != phase_names or set(phase_hashes) != phase_names:
        raise ValueError("Warehouse phase receipts incomplete")
    claims = receipt["one_use_claim_path"]
    claim_hashes = receipt["one_use_claim_sha256"]
    if set(claims) != {"final_program_audit", "heldout"} or set(claim_hashes) != set(claims):
        raise ValueError("Warehouse one-use claims incomplete")
    budget = receipt["locked_budget"]
    dev_paths = receipt["development_receipt_path"]
    dev_hashes = receipt["development_receipt_sha256"]
    expected_budgets = {str(value) for value in p.BUDGETS if value <= budget}
    if set(dev_paths) != expected_budgets or set(dev_hashes) != expected_budgets:
        raise ValueError("Warehouse development receipt budget history incomplete")
    development = {}
    development_pairs = {}
    development_traces = {}
    development_final_fit_traces = {}
    for budget_key in expected_budgets:
        if (set(dev_paths[budget_key]) != set(map(str, p.DEVELOPMENT_PAIR_SEEDS))
                or set(dev_hashes[budget_key]) != set(dev_paths[budget_key])):
            raise ValueError("Warehouse development pairs incomplete")
        development[budget_key] = {
            seed: _bundle(path, dev_hashes[budget_key][seed], output_root,
                          f"development.{budget_key}.{seed}.json",
                          assets_root=assets_root)
            for seed, path in dev_paths[budget_key].items()}
        development_pairs[budget_key] = {}
        development_traces[budget_key] = {}
        development_final_fit_traces[budget_key] = {}
        for seed, item in development[budget_key].items():
            dev_receipt = _read(ROOT / item["path"])
            pair_path = (output_root / receipt["candidate_id"] /
                         f"pair_{seed}_steps_{budget_key}" / "pair.training.json")
            development_pairs[budget_key][seed] = _bundle(
                str(pair_path), dev_receipt["pair_report_sha256"],
                output_root, "pair.training.json", assets_root=assets_root)
            development_traces[budget_key][seed] = _audit_trace_assets(
                dev_receipt.get("program_audit", {}), output_root,
                required=bool(dev_receipt.get("program_audit")),
                assets_root=assets_root)
            development_final_fit_traces[budget_key][seed] = _final_fit_trace_assets(
                _read(ROOT / development_pairs[budget_key][seed]["path"]),
                output_root, assets_root=assets_root)
    pairs = receipt["candidate_training"]
    if (len(pairs) != len(p.FORMAL_PAIR_SEEDS)
            or {row["pair_seed"] for row in pairs} != set(p.FORMAL_PAIR_SEEDS)):
        raise ValueError("Warehouse formal pair reports incomplete")
    pair_reports = {str(row["pair_seed"]): _bundle(
        row["pair_report_path"], row["pair_report_sha256"],
        output_root, "pair.training.json", assets_root=assets_root)
        for row in pairs}
    formal_final_fit_traces = {
        seed: _final_fit_trace_assets(_read(ROOT / item["path"]), output_root,
                                      assets_root=assets_root)
        for seed, item in pair_reports.items()}
    phase_receipts = {name: _bundle(phases[name], phase_hashes[name],
                                    output_root, name, assets_root=assets_root)
                      for name in sorted(phase_names)}
    audit_traces = {
        "development": development_traces,
        "preaudit": _audit_trace_assets(
            _read(ROOT / phase_receipts["program.preaudit.json"]["path"])["audit"],
            output_root, required=True, assets_root=assets_root),
        "final_audit": _audit_trace_assets(
            _read(ROOT / phase_receipts["final_program_audit.json"]["path"])["audit"],
            output_root, required=True, assets_root=assets_root),
    }
    return {
        "phase_receipts": phase_receipts,
        "one_use_claims": {phase: _bundle(claims[phase], claim_hashes[phase],
                                           ROOT / "output" /
                                           "warehouse_v6_sealed_claims" / release_id,
                                           f"{phase}.claim.json",
                                           assets_root=assets_root)
                           for phase in ("final_program_audit", "heldout")},
        "pair_reports": pair_reports,
        "development_receipts": development,
        "development_pair_reports": development_pairs,
        "program_traces": audit_traces,
        "final_fit_traces": {"formal": formal_final_fit_traces,
                             "development": development_final_fit_traces},
        "training_assets": {seed: _training_assets(
            _read(ROOT / item["path"])["variants"], output_root,
            allow_missing_regularized_programs=
            seed != str(receipt["selected_pair_seed"]),
            assets_root=assets_root)
            for seed, item in pair_reports.items()},
        "development_training_assets": {budget_key: {
            seed: _training_assets(_read(ROOT / item["path"])["training"],
                                   output_root,
                                   allow_missing_regularized_programs=
                                   int(budget_key) < budget,
                                   assets_root=assets_root)
            for seed, item in section.items()}
            for budget_key, section in development.items()},
    }


def _policies(receipt: dict, evidence: dict, output_root: Path,
              acceptance_asset: dict, *, assets_root: Path = ASSETS) -> dict:
    selected = str(receipt["selected_pair_seed"])
    pair = _read(ROOT / evidence["pair_reports"][selected]["path"])
    if (pair.get("candidate_id") != receipt.get("candidate_id")
            or pair.get("pair_seed") != receipt["selected_pair_seed"]
            or pair.get("joint_steps") != receipt["locked_budget"]):
        raise ValueError("Warehouse selected pair identity mismatch")
    entries = {}
    for variant in p.VARIANTS:
        trained = pair["variants"][variant]
        artifact = trained["actor_artifact"]
        if artifact.get("action_physics_config_sha256") != receipt.get(
                "action_physics_config_sha256"):
            raise ValueError("Warehouse selected Actor physics config differs from acceptance")
        actor = _bundle(artifact["path"], artifact["sha256"], output_root,
                        assets_root=assets_root)
        report = _bundle(trained["training_report_path"],
                         trained["training_report_sha256"], output_root,
                         assets_root=assets_root)
        if (receipt["artifact_sha256"][variant] != actor["sha256"]
                or receipt["training_report_sha256"][variant] != report["sha256"]
                or receipt["role_sha256"][variant] != artifact["role_sha256"]):
            raise ValueError("Warehouse selected Actor or training report differs from acceptance")
        programs = trained.get("program_artifacts") or {}
        if set(programs) != ({"robot_1", "robot_2"}
                             if variant == "regularized" else set()):
            raise ValueError("Warehouse role program pair incomplete")
        bundled_programs = {role: _bundle(programs[role]["path"],
                                          programs[role]["sha256"], output_root,
                                          assets_root=assets_root)
                            for role in programs}
        if variant == "regularized" and {
                role: item["sha256"] for role, item in bundled_programs.items()} != \
                receipt["program_sha256"]["regularized"]:
            raise ValueError("Warehouse program differs from acceptance")
        entries[variant] = {
            "policy_id": receipt["policy_id"][variant],
            "model_version": receipt["actor_model_version"],
            "feature_schema_sha256": receipt["feature_schema_sha256"],
            "action_physics_config_sha256": receipt[
                "action_physics_config_sha256"],
            "actor_path": actor["path"], "actor_sha256": actor["sha256"],
            "role_sha256": {"human": artifact["role_sha256"]["robot_1"],
                            "ai": artifact["role_sha256"]["robot_2"]},
            "program_paths": [bundled_programs[role]["path"] for role in p.ROLES
                              if role in bundled_programs],
            "program_sha256": {role: item["sha256"]
                               for role, item in bundled_programs.items()},
            "training_report_path": report["path"],
            "training_report_sha256": report["sha256"],
            "acceptance_path": acceptance_asset["path"],
            "acceptance_sha256": acceptance_asset["sha256"],
        }
    return entries


def _verify_training_package(receipt: dict, evidence: dict) -> None:
    """Open every formal and development Actor, report, and role program."""
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v6_program import WarehouseV6Program

    groups = (
        (evidence["pair_reports"], evidence["training_assets"],
         None, receipt["locked_budget"]),
        *((evidence["development_pair_reports"][budget],
           evidence["development_training_assets"][budget],
           evidence["development_receipts"][budget], int(budget))
          for budget in evidence["development_pair_reports"]),
    )
    for pairs, assets, development, budget in groups:
        if set(pairs) != set(assets):
            raise ValueError("Warehouse paired asset inventory mismatch")
        for seed, pair_asset in pairs.items():
            pair = _read(ROOT / pair_asset["path"])
            if (pair.get("candidate_id") != receipt["candidate_id"]
                    or pair.get("pair_seed") != int(seed)
                    or pair.get("joint_steps") != budget
                    or pair.get("action_physics_config_sha256") !=
                    p.REGISTERED_BC_SPEC["action_physics_config_sha256"]
                    or pair.get("program_fit_method") !=
                    receipt.get("program_fit_method")
                    or pair.get("program_fit_method") not in
                    p.PROGRAM_FIT_METHOD_CANDIDATES
                    or pair.get("topology_refresh_steps") !=
                    list(p.topology_refresh_steps(budget))
                    or pair.get("topology_frozen_at_step") !=
                    p.topology_frozen_at_step(budget)
                    or pair.get("bc_recipe_sha256") !=
                    p.REGISTERED_BC_SPEC_SHA256
                    or pair.get("risk_dual_config_sha256") !=
                    p.REGISTERED_RISK_DUAL_CONFIG_SHA256
                    or pair.get("bc_data_seed_schedule_sha256") !=
                    p._hash_json(p.bc_data_seed_schedule(int(seed)))
                    or not p.bc_report_valid(
                        pair.get("bc", {}), int(seed),
                        pair.get("source_closure_sha256", ""))):
                raise ValueError("Warehouse packaged pair identity mismatch")
            # The pair report embeds the full final-BC internal result, but
            # Stage 1 also wrote it independently before any PPO reset. Verify
            # those exact original bytes and the frozen BC Actor now; the
            # release pair report itself later preserves their identities.
            internal = pair["bc"]["final_bc_internal_precheck"]
            independent = pair["bc"]["final_bc_internal_precheck_receipt"]
            final_actor = internal["final_bc_actor"]
            if (not Path(independent["path"]).is_file()
                    or _sha(Path(independent["path"])) != independent["sha256"]
                    or _read(Path(independent["path"])) != internal
                    or not Path(final_actor["path"]).is_file()
                    or _sha(Path(final_actor["path"])) != final_actor["sha256"]):
                raise ValueError("Warehouse final BC independent evidence changed")
            if development is not None:
                dev = _read(ROOT / development[seed]["path"])
                if (dev.get("pair_report_sha256") != pair_asset["sha256"]
                        or dev.get("training") != pair["variants"]):
                    raise ValueError("Warehouse development training receipt mismatch")
            for variant in p.VARIANTS:
                trained = pair["variants"][variant]
                if (trained.get("program_fit_method") !=
                        pair["program_fit_method"]
                        or trained.get("topology_refresh_steps") !=
                        pair["topology_refresh_steps"]
                        or trained.get("topology_frozen_at_step") !=
                        pair["topology_frozen_at_step"]
                        or not p.topology_refresh_history_valid(
                            trained.get("topology_refresh_history"),
                            joint_steps=budget)
                        or not p.frozen_topology_valid(trained)
                        or trained.get("action_physics_config_sha256") !=
                        p.REGISTERED_BC_SPEC["action_physics_config_sha256"]
                        or trained.get("bc_recipe_sha256") !=
                        p.REGISTERED_BC_SPEC_SHA256
                        or trained.get("risk_dual_config_sha256") !=
                        p.REGISTERED_RISK_DUAL_CONFIG_SHA256):
                    raise ValueError("Warehouse packaged training method mismatch")
                packaged = assets[seed][variant]
                if (packaged["actor"]["sha256"] !=
                        trained["actor_artifact"]["sha256"]
                        or packaged["training_report"]["sha256"] !=
                        trained["training_report_sha256"]
                        or {role: item["sha256"] for role, item in
                            packaged["programs"].items()} != {
                                role: item["sha256"] for role, item in
                                trained.get("program_artifacts", {}).items()}):
                    raise ValueError("Warehouse packaged training asset hash mismatch")
                actor = FrozenWarehouseV6Actor.load(
                    ROOT / packaged["actor"]["path"],
                    expected_sha256=packaged["actor"]["sha256"],
                    expected_role_sha256=trained["selected_role_sha256"])
                for role, program in packaged["programs"].items():
                    loaded = WarehouseV6Program.load(
                        ROOT / program["path"], expected_sha256=program["sha256"],
                        actor_role_sha256=actor.role_sha256[role])
                    if loaded.role != role:
                        raise ValueError("Warehouse packaged program role mismatch")
                    if (loaded.report.get("fit_method") !=
                            pair["program_fit_method"]):
                        raise ValueError("Warehouse package rejects axis program")


def _offline_replay_final_fit(receipt: dict, evidence: dict,
                              output_root: Path, *,
                              release_id: str,
                              assets_root: Path) -> dict:
    """Re-fit every frozen role program from packaged, physically replayed rows.

    This runs only in the training/package environment. The study runtime checks
    the same rows and leaf metrics with NumPy, without importing a trainer.
    """
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v6_program import fit_program
    from experiments.warehouse.acceptance import (
        _final_fit_verified_rows, _verify_final_fit_pair)

    result = {"formal": {}, "development": {}}
    groups = [(
        "formal", None, evidence["pair_reports"],
        evidence["training_assets"], evidence["final_fit_traces"]["formal"])]
    groups.extend((
        "development", budget, evidence["development_pair_reports"][budget],
        evidence["development_training_assets"][budget],
        evidence["final_fit_traces"]["development"][budget])
        for budget in evidence["development_pair_reports"])
    for phase, budget, pairs, assets, traces in groups:
        if phase == "development":
            result[phase][budget] = {}
        for seed, item in pairs.items():
            pair = _read(ROOT / item["path"])
            samples = _final_fit_verified_rows(
                root=ROOT, release_id=release_id, index=traces[seed],
                pair=pair, assets=assets[seed])
            if not _verify_final_fit_pair(
                    root=ROOT, release_id=release_id, index=traces[seed],
                    pair=pair, assets=assets[seed],
                    verified_samples=samples):
                raise ValueError("Warehouse final fit leaf evidence differs")
            regularized = pair["variants"]["regularized"]
            actor = FrozenWarehouseV6Actor.load(
                ROOT / assets[seed]["regularized"]["actor"]["path"],
                expected_sha256=assets[seed]["regularized"]["actor"]["sha256"],
                expected_role_sha256=regularized["selected_role_sha256"])
            fitted = {}
            for role in p.ROLES:
                try:
                    program = fit_program(
                        role, actor, *(samples[role][name] for name in
                                       p.FINAL_PROGRAM_SPLIT_COUNTS),
                        method=pair["program_fit_method"])
                except (ValueError, RuntimeError) as exc:
                    if regularized["final_fresh_program_reports"][role].get(
                            "status") != "failed":
                        raise ValueError(
                            "Warehouse previously successful program no longer fits") from exc
                    fitted[role] = None
                else:
                    expected = regularized["program_artifacts"][role]["sha256"]
                    if (program.report.get("fit_method") !=
                            pair["program_fit_method"]
                            or program.sha256() != expected
                            or regularized["final_fresh_program_reports"][role].get(
                                "program_sha256") != expected):
                        raise ValueError("Warehouse final program refit changed bytes")
                    fitted[role] = expected
            replay = {
                "format": "warehouse-final-fit-replay.v1",
                "release_id": release_id,
                "candidate_id": receipt["candidate_id"],
                "pair_seed": int(seed),
                "joint_steps": pair["joint_steps"],
                "source_closure_sha256": receipt["source_closure_sha256"],
                "pair_report_sha256": item["sha256"],
                "actor_sha256": actor.artifact_sha256,
                "program_fit_method": pair["program_fit_method"],
                "trace_sha256": {name: traces[seed][name]["sha256"]
                                 for name in p.FINAL_PROGRAM_SPLIT_COUNTS},
                "refitted_program_sha256": fitted,
                "passed": True,
            }
            source = (output_root / receipt["candidate_id"] /
                      f"pair_{seed}_steps_{pair['joint_steps']}" /
                      "final_fit_replay.receipt.json")
            encoded = (json.dumps(replay, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode()
            if source.is_file() and source.read_bytes() != encoded:
                raise ValueError("Warehouse immutable final fit replay changed")
            if not source.is_file():
                source.write_bytes(encoded)
            bundled = _bundle(str(source), sha256(encoded).hexdigest(), output_root,
                              "final_fit_replay.receipt.json",
                              assets_root=assets_root)
            if phase == "formal":
                result[phase][seed] = bundled
            else:
                result[phase][budget][seed] = bundled
    return result


def _offline_replay_quality(receipt: dict, evidence: dict,
                            output_root: Path, *, assets_root: Path) -> dict:
    """Replay each saved quality action stream before admitting a release.

    The sealed heldout claim was created before its original evaluation. This
    is an integrity replay of that saved trajectory, not a new model test or
    another checkpoint selection. A content-addressed receipt lets the live
    server validate the replay without rerunning thousands of episodes.
    """
    from experiments.warehouse.quality_replay import replay_series

    result = {"development": {}, "formal": {}, "heldout": {}}

    def certify(*, name: str, rows: list, seeds: list[tuple[int, int]],
                actor_item: dict, role_hashes: dict) -> dict:
        replay = replay_series(
            actor_path=ROOT / actor_item["path"],
            actor_sha256=actor_item["sha256"], role_sha256=role_hashes,
            rows=rows, seeds=seeds)
        source = (output_root / receipt["candidate_id"] / "quality_replays" /
                  f"{name}.quality-replay.json")
        source.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(replay, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode()
        if source.is_file() and source.read_bytes() != encoded:
            raise ValueError("Warehouse immutable quality replay changed")
        if not source.is_file():
            temporary = source.with_name(f".{source.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(encoded)
                os.link(temporary, source)
            finally:
                temporary.unlink(missing_ok=True)
        return _bundle(str(source), sha256(encoded).hexdigest(), output_root,
                       source.name, assets_root=assets_root)

    stress_seeds = [(scenario_seed, action_seed)
                    for _, scenario_seed, action_seed in p.stress_schedule()]
    for budget, pairs in evidence["development_pair_reports"].items():
        result["development"][budget] = {}
        for seed in pairs:
            row = _read(ROOT / evidence["development_receipts"][budget][seed]["path"])
            pair = _read(ROOT / pairs[seed]["path"])
            assets = evidence["development_training_assets"][budget][seed]
            result["development"][budget][seed] = {}
            if not row.get("stress"):
                continue
            if set(row["stress"]) != set(p.VARIANTS):
                raise ValueError("Warehouse development stress variants incomplete")
            for variant in p.VARIANTS:
                result["development"][budget][seed][variant] = certify(
                    name=f"development.{budget}.{seed}.{variant}",
                    rows=row["stress"][variant]["episodes"],
                    seeds=stress_seeds, actor_item=assets[variant]["actor"],
                    role_hashes=pair["variants"][variant]["selected_role_sha256"])

    formal_seeds = [(seed, seed) for seed in p.FORMAL_SELECTION_SEEDS]
    for seed, pair_item in evidence["pair_reports"].items():
        row = _read(ROOT / evidence["phase_receipts"][
            f"formal.training.{seed}.json"]["path"])
        pair = _read(ROOT / pair_item["path"])
        assets = evidence["training_assets"][seed]
        result["formal"][seed] = {}
        for variant in p.VARIANTS:
            result["formal"][seed][variant] = certify(
                name=f"formal.{seed}.{variant}",
                rows=row["variants"][variant]["episodes"],
                seeds=formal_seeds, actor_item=assets[variant]["actor"],
                role_hashes=pair["variants"][variant]["selected_role_sha256"])

    held = _read(ROOT / evidence["phase_receipts"]["heldout.json"]["path"])
    selected = str(receipt["selected_pair_seed"])
    pair = _read(ROOT / evidence["pair_reports"][selected]["path"])
    assets = evidence["training_assets"][selected]
    schedule = p.heldout_schedule(receipt["release_id"])
    for variant in p.VARIANTS:
        result["heldout"][variant] = {}
        for task in p.TASKS:
            result["heldout"][variant][str(task)] = certify(
                name=f"heldout.{variant}.task{task}",
                rows=held["outcomes"][variant]["tasks"][str(task)]["episodes"],
                seeds=[(seed, seed) for seed in schedule[task]],
                actor_item=assets[variant]["actor"],
                role_hashes=pair["variants"][variant]["selected_role_sha256"])
    return result


def package(acceptance_path: str | Path,
            *, release_id: str | None = None) -> dict:
    source = Path(acceptance_path).resolve()
    receipt = _read(source)
    release_id = release_id or receipt.get("release")
    spec = catalog_module.warehouse_release_spec(release_id)
    catalog_path = spec.catalog_path
    assets_root = catalog_path.parent / "assets"
    if (source.name != "warehouse_v6_acceptance.json"
            or source.parent.name != "admission"
            or source.parent.parent.name != receipt.get("candidate_id")
            or receipt.get("release") != release_id
            or receipt.get("release_id") != release_id
            or receipt.get("policy_id") != spec.policy_ids
            or receipt.get("actor_model_version") !=
            p.REGISTERED_BC_SPEC["actor_model_version"]
            or receipt.get("passed") is not True
            or receipt.get("heldout_evaluated") is not True):
        raise ValueError("Warehouse v6 acceptance is unavailable or incomplete")
    budget = receipt.get("locked_budget")
    if (budget not in p.BUDGETS
            or receipt.get("topology_refresh_steps") !=
            list(p.topology_refresh_steps(budget))
            or receipt.get("topology_frozen_at_step") !=
            p.topology_frozen_at_step(budget)):
        raise ValueError("Warehouse topology training schedule is not accepted")
    if (receipt.get("source_closure_sha256") !=
            catalog_module.warehouse_runtime_source_closure_sha256()):
        raise ValueError("Warehouse source closure differs from accepted training")
    output_root = source.parent.parent.parent
    original_bytes = catalog_path.read_bytes()
    catalog = json.loads(original_bytes)
    if (catalog.get("release_id") != release_id
            or catalog.get("warehouse", {}).get("status") != "pending"
            or catalog.get("warehouse", {}).get("policy_ids") != spec.policy_ids):
        raise ValueError("Warehouse release catalog is no longer pending")
    acceptance_asset = _bundle(str(source), _sha(source), output_root,
                               "warehouse_v6_acceptance.json",
                               assets_root=assets_root)
    evidence = _evidence(receipt, output_root, release_id=release_id,
                         assets_root=assets_root)
    entries = _policies(receipt, evidence, output_root, acceptance_asset,
                        assets_root=assets_root)
    _verify_training_package(receipt, evidence)
    evidence["final_fit_replays"] = _offline_replay_final_fit(
        receipt, evidence, output_root, release_id=release_id,
        assets_root=assets_root)
    evidence["quality_replays"] = _offline_replay_quality(
        receipt, evidence, output_root, assets_root=assets_root)
    candidate = deepcopy(catalog)
    candidate["warehouse"] = {**candidate["warehouse"],
                              "status": "accepted",
                              "locked_budget": receipt["locked_budget"],
                              "candidate_id": receipt["candidate_id"],
                              "evidence": evidence}
    candidate["policies"]["warehouse"] = entries
    staged = catalog_path.with_name(f".catalog.{uuid4().hex}.json")
    try:
        staged.write_text(json.dumps(candidate, sort_keys=True, indent=2) + "\n")
        from experiments.warehouse.acceptance import validate_acceptance
        from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
        for variant, entry in entries.items():
            binding = {"domain": "warehouse", "variant": variant,
                       "policy_id": entry["policy_id"],
                       "actor_sha256": entry["actor_sha256"],
                       "human_actor_sha256": entry["role_sha256"]["human"],
                       "ai_actor_sha256": entry["role_sha256"]["ai"],
                       "actor_model_version": entry["model_version"],
                       "feature_schema_sha256": entry["feature_schema_sha256"],
                       "action_physics_config_sha256": entry[
                           "action_physics_config_sha256"],
                       "program_hashes": {path: sha
                                          for path, sha in zip(
                                              entry["program_paths"],
                                              (entry["program_sha256"].get(role)
                                               for role in p.ROLES))}}
            FrozenWarehouseV6Actor.load(
                ROOT / entry["actor_path"],
                expected_sha256=entry["actor_sha256"],
                expected_role_sha256={
                    "robot_1": binding["human_actor_sha256"],
                    "robot_2": binding["ai_actor_sha256"]})
            if not validate_acceptance(
                    receipt, release_id=release_id, entries=entries,
                    evidence=catalog_module.warehouse_evidence(
                        release_id, catalog_path=staged),
                    repository_root=ROOT):
                raise ValueError(f"Warehouse {variant} admission gate failed")
        if catalog_path.read_bytes() != original_bytes:
            raise RuntimeError("Warehouse release catalog changed during packaging")
        os.replace(staged, catalog_path)
    finally:
        staged.unlink(missing_ok=True)
    return {"release_id": release_id, "candidate_id": receipt["candidate_id"],
            "catalog_sha256": _sha(catalog_path), "policy_ids": receipt["policy_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True,
                        help="immutable Warehouse admission/warehouse_v6_acceptance.json")
    parser.add_argument("--release-id", default=None,
                        help="candidate release ID; defaults to the acceptance receipt")
    args = parser.parse_args()
    print(json.dumps(package(args.acceptance, release_id=args.release_id),
                     sort_keys=True))


if __name__ == "__main__":
    main()
