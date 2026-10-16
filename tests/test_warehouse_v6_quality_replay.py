"""Quality claims must reproduce the bound frozen NN's physical rollout."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from tests.warehouse_v614_test_scopes import isolated_evaluation_scope


def test_frozen_actor_quality_trace_replays_and_detects_tampering(tmp_path):
    from backend.training.warehouse_v6_actor import (
        FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor)
    from backend.training.warehouse_v6_runner import evaluate
    from experiments.warehouse.quality_replay import claimed_series, replay_series

    artifact = export_actor(WarehouseV6Actor(), tmp_path / "raw.actor.npz",
                            pair_seed=10077, variant="raw", candidate_id="replay-test")
    actor = FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    with isolated_evaluation_scope():
        rows = evaluate(actor, [1009001], action_seed_offset=73, trace=True)["episodes"]
    seeds = [(1009001, 1009074)]
    certified = replay_series(
        actor_path=artifact["path"], actor_sha256=artifact["sha256"],
        role_sha256=artifact["role_sha256"], rows=rows, seeds=seeds)
    assert certified == claimed_series(
        actor_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"],
        rows=rows, seeds=seeds)

    from experiments.warehouse.acceptance import _quality_replay_receipt

    release_id = "policylens-three-domain-20261016.v6.0"
    raw = json.dumps(certified, sort_keys=True, separators=(",", ":")).encode()
    digest = sha256(raw).hexdigest()
    relative = f"releases/{release_id}/assets/{digest}/quality-replay.json"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    asset = {"path": relative, "sha256": digest}
    assert _quality_replay_receipt(
        root=tmp_path, release_id=release_id, item=asset,
        actor_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"],
        rows=rows, seeds=seeds)

    changed = deepcopy(rows)
    changed[0]["deliveries"] += 1
    assert not _quality_replay_receipt(
        root=tmp_path, release_id=release_id, item=asset,
        actor_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"],
        rows=changed, seeds=seeds)
    target.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        _quality_replay_receipt(
            root=tmp_path, release_id=release_id, item=asset,
            actor_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"],
            rows=rows, seeds=seeds)

    changed = deepcopy(rows)
    changed[0]["deliveries"] += 1
    with pytest.raises(ValueError, match="outcome"):
        replay_series(actor_path=artifact["path"], actor_sha256=artifact["sha256"],
                      role_sha256=artifact["role_sha256"], rows=changed, seeds=seeds)

    changed = deepcopy(rows)
    changed[0]["quality_trace"]["chain_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="chain"):
        replay_series(actor_path=artifact["path"], actor_sha256=artifact["sha256"],
                      role_sha256=artifact["role_sha256"], rows=changed, seeds=seeds)

    changed = deepcopy(rows)
    changed[0]["quality_trace"]["actor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        replay_series(actor_path=artifact["path"], actor_sha256=artifact["sha256"],
                      role_sha256=artifact["role_sha256"], rows=changed, seeds=seeds)
