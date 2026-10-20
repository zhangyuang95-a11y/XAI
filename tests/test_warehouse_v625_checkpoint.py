"""Durability and failure behavior; no scientific fitting or trajectories."""
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training.warehouse_v625_checkpoint import (
    BudgetExhausted, CheckpointError, CheckpointSession, ContentStore,
    DeadlineExceeded,
)


IDENTITY = {"candidate": "synthetic-engineering", "source_sha256": "a"*64}


def test_fast_canonical_encoder_matches_legacy_bytes_for_supported_types_and_keys():
    from backend.training.warehouse_v625_checkpoint import canonical_bytes, plain
    cases = [None, True, False, 3, -2, .125, -0., "中文与转义\n\"", [], {},
        {"a": [1, (2, 3), np.float32(.1), np.int64(9), np.bool_(True)],
         "b": np.arange(6, dtype=np.float32).reshape(2, 3)},
        {2: "two", 10: "ten", -3: "negative"},
        {2.5: "two point five", .01: "small", -10.: "negative"},
        {True: "boolean", False: "other"}, {None: "null key"},
        {"None": "literal", None: "collision"},
        {"true": 3, "false": 2, "null": 1},
        {"outer": [{10: "ten", 2: "two"}, {False: 1, None: 2}]},
        {(1, 2): "tuple key", np.int64(7): "numpy key"},
        {float("inf"): "infinite key", float("nan"): "nan key"},
        {"message": 'not a key: {"20": true}, 1', "numpy": np.asarray([1, 2])}]
    for value in cases:
        expected = json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert canonical_bytes(value) == expected
    for value in ({"nonfinite": float("nan")}, np.asarray([float("inf")])):
        with pytest.raises(ValueError):
            canonical_bytes(value)


def session(tmp_path, **kwargs):
    return CheckpointSession(tmp_path/"checkpoint.json", IDENTITY, max_calls=32,
                             wall_seconds=60., **kwargs)


def test_call_is_durably_predebited_before_solver_and_failed_fit_is_not_refunded(tmp_path):
    state = session(tmp_path)
    token = state.consume({"support": [1, 2]})
    record = json.loads(state.ledger_path.read_text().splitlines()[-1])
    assert record["event"] == "call_started" and record["data"]["call"] == token == 1
    assert state.calls == 1 and state.pending
    with pytest.raises(CheckpointError, match="not settled"):
        state.consume()
    state.finish(token, success=False, detail={"solver_status": 9})
    assert state.calls == 1 and not state.pending
    state.commit({"best": "unchanged"}, status="paused")
    restored = session(tmp_path, resume=True)
    assert restored.calls == 1
    assert restored.consume() == 2


def test_pause_restores_state_clock_calls_and_requires_identical_identity(tmp_path):
    state = session(tmp_path)
    token = state.consume()
    state.finish(token)
    manifest = state.commit({"beam": ["a"], "cursor": 3}, status="paused")
    assert state.path.stat().st_size < 2048
    with pytest.raises(CheckpointError, match="identical"):
        CheckpointSession(state.path, {**IDENTITY, "source_sha256": "b"*64},
                          max_calls=32, wall_seconds=60., resume=True)
    restored = session(tmp_path, resume=True)
    assert restored.state == {"beam": ["a"], "cursor": 3}
    assert restored.elapsed_seconds >= manifest["elapsed_upper_bound_seconds"]
    assert restored.calls == 1
    assert restored.dirty_path.exists()


def test_repeated_resume_retains_cumulative_io_and_actual_time_bounds(tmp_path):
    clock = [0.]
    def tail(event, context):
        if event == "before_dirty_remove":
            clock[0] += .25
    state = session(tmp_path, clock=lambda: clock[0], fault_hook=tail)
    # Deterministic injected instrumentation counters isolate the accounting
    # assertion from filesystem speed. No operation cost is reset on resume.
    state.store.io_seconds = 3.
    state.ledger_io_seconds = 1.
    state.manifest_io_seconds = .5
    clock[0] = 10.
    manifest = state.commit({"cursor": 1}, status="paused")
    assert manifest["checkpoint_total_io_lower_bound_at_snapshot_seconds"] >= 4.5
    assert manifest["actual_elapsed_at_snapshot_seconds"] == 10.
    restored = session(tmp_path, clock=lambda: clock[0], resume=True)
    value = restored.summary()
    assert value["checkpoint_total_io_lower_bound_seconds"] >= 4.5
    assert value["checkpoint_total_io_upper_bound_seconds"] >= 5.5
    assert value["checkpoint_segment_io_seconds"] < value["checkpoint_total_io_lower_bound_seconds"]
    assert value["actual_elapsed_lower_bound_seconds"] == 10.
    assert value["actual_elapsed_upper_bound_seconds"] == 11.
    assert value["checkpoint_total_io_upper_bound_seconds"]-value["checkpoint_total_io_lower_bound_seconds"] == pytest.approx(1.)
    clock[0] += 10.
    restored.fault_hook = tail
    restored.commit({"cursor": 2}, status="paused")
    again = session(tmp_path, clock=lambda: clock[0], resume=True)
    value = again.summary()
    assert value["checkpoint_total_io_lower_bound_seconds"] >= 4.5
    assert value["checkpoint_total_io_upper_bound_seconds"]-value["checkpoint_total_io_lower_bound_seconds"] == pytest.approx(2.)
    assert value["actual_elapsed_lower_bound_seconds"] == 20.
    assert value["actual_elapsed_upper_bound_seconds"] == 22.


@pytest.mark.parametrize("status", ["running", "completed"])
def test_periodic_or_completed_checkpoint_cannot_resume(tmp_path, status):
    state = session(tmp_path)
    state.commit({"cursor": 1}, status=status)
    with pytest.raises(CheckpointError):
        session(tmp_path, resume=True)


def test_crash_or_unsettled_call_cannot_rewind_to_prior_snapshot(tmp_path):
    state = session(tmp_path)
    state.snapshot({"cursor": 1})
    state.consume()
    state.close()
    with pytest.raises(CheckpointError, match="dirty/crashed"):
        session(tmp_path, resume=True)
    # Even somebody removing dirty evidence cannot refund a logged call.
    state.dirty_path.unlink()
    with pytest.raises(CheckpointError):
        session(tmp_path, resume=True)


@pytest.mark.parametrize("fault", ["before_manifest_replace", "after_manifest_replace", "before_dirty_remove"])
def test_commit_fault_keeps_dirty_evidence_and_refuses_resume(tmp_path, fault):
    def inject(event, context):
        if event == fault:
            raise OSError("injected crash")
    state = session(tmp_path, fault_hook=inject)
    with pytest.raises(OSError, match="injected"):
        state.commit({"cursor": 1})
    assert state.dirty_path.exists()
    with pytest.raises(CheckpointError, match="dirty/crashed"):
        session(tmp_path, resume=True)


def test_corrupt_object_and_post_pause_ledger_tampering_refuse_restore(tmp_path):
    state = session(tmp_path)
    ref = state.store.put_array(np.arange(100, dtype=np.int64))
    state.commit({"object": ref})
    path = state.store.physical_path(ref)
    path.write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="hash or size"):
        session(tmp_path, resume=True)


def test_ledger_hash_chain_detects_modified_predebit(tmp_path):
    state = session(tmp_path)
    token = state.consume(); state.finish(token)
    state.commit({"cursor": 1})
    rows = state.ledger_path.read_text().splitlines()
    value = json.loads(rows[1]); value["data"]["call"] = 999
    rows[1] = json.dumps(value)
    state.ledger_path.write_text("\n".join(rows)+"\n")
    with pytest.raises(CheckpointError, match="corrupt"):
        session(tmp_path, resume=True)


def test_content_objects_persist_once_and_snapshots_only_reference_large_payload(tmp_path):
    state = session(tmp_path)
    payload = np.arange(4096*173, dtype=np.float32).reshape(4096, 173)
    ref = state.store.put_array(payload)
    for cursor in range(8):
        assert state.store.put_array(payload) == ref
        state.snapshot({"payload": ref, "cursor": cursor, "queue": list(range(256))})
    assert state.store.written_bytes < payload.nbytes+50000
    assert state.store.logical_bytes_added < payload.nbytes+20000
    state.commit({"payload": ref, "cursor": 8})
    restored = session(tmp_path, resume=True)
    np.testing.assert_array_equal(restored.store.get_array(restored.state["payload"]), payload)
    assert restored.path.stat().st_size < 2048
    assert not list(state.store.root.glob("*.pkl"))


def test_array_pickle_and_symlinks_are_not_accepted(tmp_path):
    store = ContentStore(tmp_path/"objects")
    with pytest.raises(ValueError, match="Python objects"):
        store.put_array(np.asarray([object()], dtype=object))
    ref = store.put_json({"value": 1})
    registry = store.flush()
    path = store.physical_path(ref)
    elsewhere = tmp_path/"elsewhere"; path.rename(elsewhere); path.symlink_to(elsewhere)
    with pytest.raises(CheckpointError, match="symlink"):
        ContentStore(store.root).load_registry(registry)


def test_incremental_pack_batches_many_objects_and_never_rewrites_old_packs(tmp_path):
    store = ContentStore(tmp_path/"objects")
    refs = [store.put_json({"counter": i, "payload": [i]*8}) for i in range(1000)]
    assert not list(store.root.iterdir())  # the active run is dirty, not safe
    first = store.flush()
    assert len(list(store.root.iterdir())) == 3  # data, index, registry
    originals = {p.name: p.read_bytes() for p in store.root.iterdir()}
    assert store.flush() == first
    next_ref = store.put_json({"cursor": 1000, "previous": refs[-1]})
    second = store.flush()
    assert len(list(store.root.iterdir())) == 6
    assert all((store.root/name).read_bytes() == raw for name, raw in originals.items())
    restored = ContentStore(store.root)
    restored.load_registry(second)
    assert restored.get_json(refs[0])["counter"] == 0
    assert restored.get_json(refs[-1])["counter"] == 999
    assert restored.get_json(next_ref)["previous"] == refs[-1]
    assert restored.verify_graph(next_ref) == 2


def test_pack_index_offset_or_logical_hash_corruption_is_rejected(tmp_path):
    from backend.training.warehouse_v625_checkpoint import canonical_bytes
    store = ContentStore(tmp_path/"objects")
    store.put_json({"valid": True})
    registry_ref = store.flush()
    registry = json.loads(store._physical_read(registry_ref))
    index = json.loads(store._physical_read(registry["packs"][0]))
    index["entries"][0][3] = 1  # valid content hashes, but invalid offset table
    invalid_index = store._publish(canonical_bytes(index), "json")
    invalid_registry = store._publish(canonical_bytes({
        "schema": "warehouse-v625-CAS-pack-registry.v1", "packs": [invalid_index]}), "json")
    with pytest.raises(CheckpointError, match="offsets"):
        ContentStore(store.root).load_registry(invalid_registry)


def test_budget_predebit_cannot_exceed_limit_and_deadline_covers_io(tmp_path):
    clock = [0.]
    state = CheckpointSession(tmp_path/"checkpoint.json", IDENTITY, max_calls=1,
                              wall_seconds=10., clock=lambda: clock[0])
    token = state.consume(); state.finish(token, success=False)
    with pytest.raises(BudgetExhausted):
        state.consume()
    def slow(event, context):
        if event == "before_manifest_replace":
            clock[0] += 2.
    state.fault_hook = slow
    with pytest.raises(DeadlineExceeded, match="small commit"):
        state.commit({"done": True})
    assert state.dirty_path.exists()


def test_hard_deadline_interruption_keeps_dirty_and_never_marks_success(tmp_path):
    import time
    state = CheckpointSession(tmp_path/"checkpoint.json", IDENTITY,
                              max_calls=1, wall_seconds=.02)
    with pytest.raises(DeadlineExceeded, match="hard registered"):
        with state.hard_deadline():
            time.sleep(.1)
    assert state.dirty_path.exists()
    with pytest.raises(CheckpointError, match="dirty/crashed"):
        CheckpointSession(state.path, IDENTITY, max_calls=1, wall_seconds=.02, resume=True)
