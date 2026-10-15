"""A saved Warehouse audit must reproduce NN draws and physical trajectories."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor)
from backend.training.warehouse_v6_program import WarehouseV6Program
from backend.training.warehouse_v6_runner import audit_programs
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_observation_v6 import FEATURE_DIM
from experiments.warehouse.acceptance import _verify_program_trace


RELEASE = "policylens-three-domain-20261016.v6.0"


def _bundle(root: Path, source: Path) -> dict:
    digest = sha256(source.read_bytes()).hexdigest()
    relative = f"releases/{RELEASE}/assets/{digest}/{source.name}"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"path": relative, "sha256": digest}


def _program(role: str, actor: FrozenWarehouseV6Actor) -> WarehouseV6Program:
    # An intentionally unqualified leaf keeps this test about trace integrity,
    # not the statistical coverage of a trained candidate.
    report = {
        "max_depth": 1, "leaf_count": 2, "max_features_per_node": 1,
        "split_episode_ids": {name: [name] for name in
                              ("fit", "calibration", "eligibility", "validation")},
        "split_scenario_seed_ids": {name: [name] for name in
                                    ("fit", "calibration", "eligibility", "validation")},
        "validation": {"feedback_online_passed": False,
                       "explanation_online_passed": False,
                       "feedback_passed": False,
                       "explanation_passed": False},
    }
    return WarehouseV6Program(
        role=role, actor_role_sha256=actor.role_sha256[role], root_ref=0,
        split_weights=np.zeros((1, FEATURE_DIM), dtype=np.float32),
        split_bias=np.zeros(1, dtype=np.float32),
        left_refs=np.asarray([-1], dtype=np.int64),
        right_refs=np.asarray([-2], dtype=np.int64),
        leaf_logits=np.zeros((2, len(ACTIONS)), dtype=np.float32),
        feedback_leaves=frozenset(), explanation_leaves=frozenset(),
        report=report)


def test_packaged_trace_recomputes_nn_draws_and_rejects_forged_rows(tmp_path):
    torch.manual_seed(6291)
    source = tmp_path / "source"
    source.mkdir()
    artifact = export_actor(WarehouseV6Actor(), source / "actor.npz",
                            pair_seed=999901, variant="regularized",
                            candidate_id="trace-integrity")
    actor = FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    programs = {role: _program(role, actor) for role in actor.roles}
    program_files = {}
    for role, program in programs.items():
        path = source / f"{role}.program.json"
        program.save(path)
        program_files[role] = _bundle(tmp_path, path)
    actor_file = _bundle(tmp_path, Path(artifact["path"]))
    seed = 999901  # unregistered; sealed and heldout blocks remain untouched
    audits = audit_programs(actor, programs, [seed],
                            trace_dir=source / "traces")
    index = {role: _bundle(tmp_path, Path(audits[role]["trace_path"]))
             for role in actor.roles}
    entry = {
        "actor_path": actor_file["path"],
        "actor_sha256": actor_file["sha256"],
        "role_sha256": {"human": actor.role_sha256["robot_1"],
                        "ai": actor.role_sha256["robot_2"]},
        "program_paths": [program_files[role]["path"] for role in actor.roles],
        "program_sha256": {role: program_files[role]["sha256"]
                           for role in actor.roles},
    }
    proof = dict(root=tmp_path, release_id=RELEASE, index=index,
                 audits=audits, entry=entry, expected_seeds=[seed])
    assert _verify_program_trace(**proof)

    # A rewritten receipt and content hash cannot turn a different sampled
    # action into an actual frozen-Actor RNG draw.
    changed_audits = deepcopy(audits)
    changed_index = deepcopy(index)
    role = "robot_1"
    rows = [json.loads(line) for line in
            (tmp_path / index[role]["path"]).read_text().splitlines()]
    first = rows[0]
    first["nn_sampled_action"] = next(action for action in ACTIONS
                                      if action != first["nn_sampled_action"])
    forged = source / "forged.jsonl"
    forged.write_text("".join(json.dumps(row, sort_keys=True,
                                          separators=(",", ":")) + "\n"
                              for row in rows))
    changed_index[role] = _bundle(tmp_path, forged)
    changed_audits[role]["trace_sha256"] = changed_index[role]["sha256"]
    changed_audits[role]["trace_path"] = str(forged)
    assert not _verify_program_trace(**{**proof, "index": changed_index,
                                        "audits": changed_audits})

    # Both action fields describe the same executed NN draw. A contradictory
    # secondary field must fail even when the primary draw is intact.
    secondary_rows = [json.loads(line) for line in
                      (tmp_path / index[role]["path"]).read_text().splitlines()]
    secondary_rows[0]["sampled_action"] = next(
        action for action in ACTIONS
        if action != secondary_rows[0]["nn_sampled_action"])
    contradictory = source / "contradictory.jsonl"
    contradictory.write_text("".join(json.dumps(row, sort_keys=True,
                                                 separators=(",", ":")) + "\n"
                                     for row in secondary_rows))
    changed_index[role] = _bundle(tmp_path, contradictory)
    changed_audits[role]["trace_sha256"] = changed_index[role]["sha256"]
    changed_audits[role]["trace_path"] = str(contradictory)
    assert not _verify_program_trace(**{**proof, "index": changed_index,
                                        "audits": changed_audits})

    # Dropping an inconvenient frame is also rejected by the full trajectory
    # replay even if all three trace identity fields are updated.
    skipped = source / "skipped.jsonl"
    skipped.write_text("".join(json.dumps(row, sort_keys=True,
                                           separators=(",", ":")) + "\n"
                               for row in rows[1:]))
    changed_index[role] = _bundle(tmp_path, skipped)
    changed_audits[role]["trace_sha256"] = changed_index[role]["sha256"]
    changed_audits[role]["trace_path"] = str(skipped)
    changed_audits[role]["trace_row_count"] -= 1
    changed_audits[role]["sample_count"] -= 1
    assert not _verify_program_trace(**{**proof, "index": changed_index,
                                        "audits": changed_audits})
