"""Final program fitting episodes are physically replayable and NN-bound."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor)
from backend.training.warehouse_v6_runner import (
    _write_final_fit_trace, collect_program_audit_rows)
from experiments.warehouse import protocol as p
from experiments.warehouse.acceptance import (
    _final_fit_index_claim, _final_fit_verified_rows, _same_recomputed)


RELEASE = p.INITIAL_RELEASE_ID
SCHEDULE = {name: [700_001 + index] for index, name in enumerate(
    p.FINAL_PROGRAM_SPLIT_COUNTS)}


def _bundle(root: Path, source: Path) -> dict:
    digest = sha256(source.read_bytes()).hexdigest()
    relative = f"releases/{RELEASE}/assets/{digest}/{source.name}"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"path": relative, "sha256": digest}


@pytest.fixture
def fit_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "final_program_fit_seed_schedule", lambda *_: SCHEDULE)
    torch.manual_seed(6391)
    source = tmp_path / "source"
    source.mkdir()
    artifact = export_actor(WarehouseV6Actor(), source / "actor.npz",
                            pair_seed=6101, variant="regularized",
                            candidate_id="final-fit-integrity")
    actor = FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    actor_asset = _bundle(tmp_path, Path(artifact["path"]))
    collected = collect_program_audit_rows(actor, [seed for block in
                                                  SCHEDULE.values() for seed in block])
    traces, index = {}, {}
    for split, seeds in SCHEDULE.items():
        selected = {role: [row for row in collected[role]
                           if row["base_seed"] in seeds]
                    for role in p.ROLES}
        traces[split] = _write_final_fit_trace(
            selected, split=split, seeds=seeds,
            trace_root=source / "traces", actor_sha256=artifact["sha256"])
        index[split] = _bundle(tmp_path, Path(traces[split]["path"]))
    reports = {role: {
        "status": "failed", "release_id": RELEASE,
        "actor_artifact_sha256": artifact["sha256"],
        "final_fit_traces": traces,
        "split_seeds": SCHEDULE,
        "split_seed_schedule_sha256": sha256(json.dumps(
            SCHEDULE, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    } for role in p.ROLES}
    trained = {
        "release_id": RELEASE, "actor_artifact": artifact,
        "selected_role_sha256": artifact["role_sha256"],
        "final_program_fit_seed_schedule": SCHEDULE,
        "final_program_fit_seed_schedule_sha256": reports[p.ROLES[0]][
            "split_seed_schedule_sha256"],
        "final_fresh_program_reports": reports,
        "training_episodes": [], "program_artifacts": {},
        "selected_program_reports": {},
    }
    return {
        "root": tmp_path, "release_id": RELEASE, "index": index,
        "pair": {"pair_seed": 6101, "release_id": RELEASE,
                 "variants": {"regularized": trained}},
        "assets": {"regularized": {"actor": actor_asset, "programs": {}}},
    }


def test_final_fit_trace_replays_all_roles_and_rejects_forged_action(fit_proof):
    samples = _final_fit_verified_rows(**fit_proof)
    assert set(samples) == set(p.ROLES)
    assert all(len(samples[role][split]) > 0 for role in p.ROLES
               for split in SCHEDULE)
    changed = deepcopy(fit_proof)
    item = changed["index"]["fit"]
    rows = [json.loads(line) for line in
            (fit_proof["root"] / item["path"]).read_text().splitlines()]
    first = rows[0]["roles"]["robot_2"]
    first["sampled_action"] = "wait" if first["sampled_action"] != "wait" else "north"
    forged = fit_proof["root"] / "source" / "forged" / "fit.jsonl"
    forged.parent.mkdir()
    forged.write_text("".join(json.dumps(row, sort_keys=True,
                                          separators=(",", ":")) + "\n"
                              for row in rows))
    changed["index"]["fit"] = _bundle(fit_proof["root"], forged)
    for report in changed["pair"]["variants"]["regularized"][
            "final_fresh_program_reports"].values():
        report["final_fit_traces"]["fit"]["sha256"] = changed["index"]["fit"]["sha256"]
        report["final_fit_traces"]["fit"]["path"] = str(forged)
    with pytest.raises(ValueError, match="NN action differs"):
        _final_fit_verified_rows(**changed)


def test_final_fit_replay_imports_without_torch(fit_proof):
    proof = fit_proof["root"] / "source" / "proof.json"
    proof.write_text(json.dumps({
        **fit_proof, "root": str(fit_proof["root"])}))
    script = """
import builtins, json, sys
from pathlib import Path
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('no Torch in runtime')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from experiments.warehouse import protocol as p
from experiments.warehouse.acceptance import _final_fit_verified_rows
p.final_program_fit_seed_schedule = lambda *_: {
    name: [700001 + i] for i, name in enumerate(p.FINAL_PROGRAM_SPLIT_COUNTS)}
proof = json.load(open(sys.argv[1]))
proof['root'] = Path(proof['root'])
assert _final_fit_verified_rows(**proof)
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(proof)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unselected_pair_trace_bytes_are_hashed_and_runtime_metrics_tolerate_roundoff(
        fit_proof):
    assert _final_fit_index_claim(**fit_proof)
    target = fit_proof["root"] / fit_proof["index"]["eligibility"]["path"]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        _final_fit_index_claim(**fit_proof)
    assert _same_recomputed({"kl": 0.1000005}, {"kl": 0.1})
    assert not _same_recomputed({"kl": 0.10001}, {"kl": 0.1})
