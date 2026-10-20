#!/usr/bin/env python3
"""Single-use native engineering benchmark, never admission or a search retry.

Preparation pins spent fit rows and a pure S1 topology without optimization.
Execution charges four complete 4096-call reservations before any solver, and
every actual API also receives a durable predebit. Each condition freshly fits
the initial tree and the same 15-candidate native S1 prefix. Numerical replay is
outside timing; it cannot inflate the native-work denominator.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.warehouse_v625_checkpoint import (
    BudgetExhausted, CheckpointError, CheckpointSession, DeadlineExceeded,
    canonical_bytes, digest, plain, _directory_fsync,
)

CANDIDATE = "warehouse-v625-sparse-logistic-01"
ORDER = ("baseline-P0", "checkpoint-P0", "checkpoint-P1", "baseline-P1")
PREFIX, PAUSE, CALL_LIMIT, WALL_LIMIT = 15, 7, 4096, 1200
ZERO = {"environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
    "pilot_opened": False, "publishable": False, "sealed_audit_used": False, "heldout_used": False}


def expected_recipe():
    return {"schema": "warehouse-v625-native-engineering-recipe.v1", "candidate_id": CANDIDATE,
        "attempt_id": "engineering-native-04", "fold": 1, "role": "robot_2",
        "fit_physical_frames": 14900, "condition_order": list(ORDER), "candidate_prefix": PREFIX,
        "pause_after_candidates": PAUSE, "proposal_recipe_index": 0, "proposal_mode": "S1",
        "maximum_api_calls_per_condition": CALL_LIMIT, "outer_reserved_api_calls": 4 * CALL_LIMIT,
        "wall_seconds_per_condition": WALL_LIMIT, "maximum_checkpoint_fraction": .10,
        "maximum_workers": 1, "BLAS_threads": 1, "shared_fit_cache": False,
        "baseline_per_call_durable_predebit": True, "full_generated_queue_preserved": True,
        "model_and_score_hash_equivalence": True, "distribution_hash_replay_outside_timing": True,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "fresh_data_used": False, "calibration_eligibility_validation_used": False,
        "scientific_result_reuse": False, "publishable": False, "pilot_opened": False}


def file_sha(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("missing/symlink native engineering binding: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def write_new(path, value, *, readonly=False):
    path = Path(path)
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        _directory_fsync(path.parent, lambda *args, **kwargs: None)
        if readonly:
            path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path.resolve()), "sha256": sha256(raw).hexdigest()}


@contextmanager
def real_deadline(started, seconds):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("native benchmark requires a POSIX main-thread hard deadline")
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] or old_timer[1]:
        raise RuntimeError("an existing deadline cannot be repurposed")
    def expired(signum, frame):
        raise DeadlineExceeded("fixed native engineering condition deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(1.e-6, seconds - (time.monotonic() - started)))
    try:
        yield
        if time.monotonic() - started > seconds:
            raise DeadlineExceeded("native condition completed after its fixed wall budget")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class _NoStateObjects:
    """Opaque memory-only references; no serialization, CAS, or state files."""
    def __init__(self):
        self.count = 0
    def put_json(self, value):
        self.count += 1
        return {"sha256": f"{self.count:064x}", "kind": "json", "size": 0}


class BaselineLedger:
    """True no-state-checkpoint baseline, with mandatory per-API fsync ledger."""
    def __init__(self, path, identity, *, started_at, max_calls=CALL_LIMIT, wall_seconds=WALL_LIMIT):
        self.path, self.identity = Path(path), plain(identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise RuntimeError("baseline ledger is single-use")
        self.started, self.max_calls, self.wall_seconds = started_at, max_calls, wall_seconds
        self.calls, self.sequence, self.pending, self.tail = 0, 0, {}, "0" * 64
        self.closed, self.ledger_io_seconds = False, 0.
        self.store, self.rows = _NoStateObjects(), []
        self._append("created", {"identity_sha256": digest(identity), "schema": "warehouse-v625-native-baseline-call-ledger.v1",
            "maximum_calls": max_calls, "wall_seconds": wall_seconds})
    def _append(self, event, data):
        begun = time.monotonic()
        body = {"sequence": self.sequence + 1, "previous_sha256": self.tail, "event": event,
            "data": plain(data), "elapsed_seconds": time.monotonic() - self.started}
        row = dict(body, record_sha256=digest(body))
        created = not self.path.exists()
        with self.path.open("ab") as handle:
            handle.write(canonical_bytes(row) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        if created:
            _directory_fsync(self.path.parent, lambda *args, **kwargs: None)
        self.rows.append(row)
        self.sequence, self.tail = body["sequence"], row["record_sha256"]
        self.ledger_io_seconds += time.monotonic() - begun
    def check(self, *, reserve_seconds=0.):
        if self.closed:
            raise CheckpointError("baseline ledger is closed")
        if time.monotonic() - self.started + reserve_seconds >= self.wall_seconds:
            raise DeadlineExceeded("fixed baseline wall budget exhausted")
    def consume(self, metadata=None):
        self.check()
        if self.pending:
            raise CheckpointError("previous baseline API has not settled")
        if self.calls >= self.max_calls:
            raise BudgetExhausted("fixed native API reservation exhausted")
        token = self.calls + 1
        self._append("call_started", {"call": token, "metadata": plain(metadata or {})})
        self.calls = token
        self.pending[token] = True
        self.check()
        return token
    def finish(self, token=None, *, success=True, detail=None):
        if token is None and len(self.pending) == 1:
            token = next(iter(self.pending))
        if token not in self.pending:
            raise CheckpointError("unknown/already settled baseline API")
        self._append("call_completed", {"call": token, "success": bool(success), "detail": plain(detail or {})})
        self.pending.pop(token)
    def events(self):
        begun = time.monotonic()
        try:
            return [json.loads(line) for line in self.path.read_bytes().splitlines()]
        finally:
            self.ledger_io_seconds += time.monotonic() - begun
    def close(self):
        self.closed = True


class DirectCheckpointOverhead:
    """Disjoint CP-only CPU/wall intervals plus I/O outside those intervals."""
    def __init__(self, sessions):
        self.sessions, self.intervals = sessions, []
        self.wall_seconds, self.covered_io_seconds = 0., 0.
    def io(self):
        return sum(s.store.io_seconds + s.ledger_io_seconds + s.manifest_io_seconds for s in self.sessions)
    @contextmanager
    def measure(self, label):
        before, before_io = time.monotonic(), self.io()
        try:
            yield
        finally:
            end = time.monotonic()
            elapsed, io = end - before, max(0., self.io() - before_io)
            self.wall_seconds += elapsed
            self.covered_io_seconds += io
            self.intervals.append({"label": label, "started_monotonic": before, "completed_monotonic": end,
                "wall_seconds": elapsed, "included_IO_seconds": io})
    def report(self):
        total = self.io()
        outside = max(0., total - self.covered_io_seconds)
        return {"wrapped_CP_only_seconds": self.wall_seconds,
            "IO_seconds_inside_wrappers": self.covered_io_seconds,
            "total_actual_CP_IO_seconds": total, "IO_seconds_outside_wrappers": outside,
            "direct_full_checkpoint_overhead_seconds": self.wall_seconds + outside,
            "intervals": self.intervals,
            "definition": "disjoint_queue_codec_commit_restore_wall_plus_all_actual_IO_outside_wrappers"}


def checked_reservation(pointer, directory, identity, condition_id):
    """Reject ordinary direct fits before a fixed outer reservation exists."""
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise RuntimeError("native fitting requires a bound outer reservation")
    path = Path(pointer["path"]).resolve(strict=True)
    if path != Path(directory).resolve().parent.parent / "attempt-start.json" or file_sha(path) != pointer["sha256"]:
        raise RuntimeError("native reservation path/hash mismatch")
    plan_path, plan, inputs = checked_plan(path.parent / "plan.json")
    attempt = json.loads(path.read_bytes())
    reservations = [{"condition_id": name, "charged_call_slots": CALL_LIMIT} for name in ORDER]
    if (attempt.get("schema") != "warehouse-v625-native-attempt.v1"
            or attempt.get("recipe") != expected_recipe() or attempt.get("reservations") != reservations
            or attempt.get("total_charged_call_slots") != 4 * CALL_LIMIT
            or attempt.get("unused_slots_refundable") is not False
            or attempt.get("plan_sha256") != identity.get("plan_sha256")
            or attempt.get("source_sha256") != identity.get("source_sha256") or condition_id not in ORDER):
        raise RuntimeError("native reserved recipe/source/budget changed")
    if (identity.get("condition_id") != condition_id or identity.get("plan_sha256") != file_sha(plan_path)
            or identity.get("source_sha256") != plan["source_sha256"]
            or identity.get("fit_inputs_sha256") != plan["fit_inputs"]["sha256"]
            or identity.get("initial_structure_sha256") != plan["initial_structure_sha256"]
            or identity.get("fixed_recipe") != expected_recipe()):
        raise RuntimeError("native direct call does not match its actual pinned plan")
    return attempt, inputs


def _queue(ctx, initial_state):
    from backend.training.warehouse_program_search import _BeamEntry
    from backend.training.warehouse_v619_tree import SEARCH_INITIALIZATIONS
    from backend.training.warehouse_v622_search import SearchConfig
    from backend.training.warehouse_v623_search import _proposal_trees
    parent = _BeamEntry(initial_state)
    return [(tree, kind, label, parent) for tree, kind, label, mass in _proposal_trees(
        ctx, initial_state, SEARCH_INITIALIZATIONS[0], "S1", SearchConfig(), ctx.indices)]


def queue_identity(queue):
    from backend.training.warehouse_v619_tree import _tree_key
    return digest([[ _tree_key(tree).hex(), kind, label, parent.state.key.hex()]
        for tree, kind, label, parent in queue])


def _proposal_binding(inputs):
    """Pure fit-feature/NN-distribution menu, with no fitted leaf outputs."""
    from types import SimpleNamespace
    from backend.training.warehouse_v619_tree import _State, _tree_key
    from backend.training.warehouse_v623_search import _tree
    from backend.training.warehouse_v625_search import _Context
    context = _Context(inputs["encoded"], inputs["masks"], inputs["targets"], inputs["episodes"],
        inputs["names"], inputs["indices"], "P0", SimpleNamespace(check=lambda **kwargs: None))
    tree = _tree(inputs["structure"])
    regions = list(context.regions(tree))
    state = _State(tree, [], regions, (0., 0., 0., -float(inputs["structure"]["leaf_count"])), _tree_key(tree))
    queue = _queue(context, state)
    if len(queue) < PREFIX:
        raise RuntimeError("native S1 initial queue has fewer than the registered prefix")
    return {"full_queue_count": len(queue), "full_queue_sha256": queue_identity(queue),
        "prefix_count": PREFIX, "prefix_sha256": queue_identity(queue[:PREFIX]),
        "proposal_uses_fitted_leaf_outputs": False}


def prepare(config_path, output):
    """Pin all scientific inputs/sources before creating any fit authority."""
    from experiments.warehouse import v625_protocol as v
    from scripts.run_warehouse_v622_capacity import arrays
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    from backend.training.warehouse_v622_program import normalize_structure, program_structure_sha256
    config_path, output = Path(config_path).resolve(strict=True), Path(output).resolve()
    if json.loads(config_path.read_bytes()) != expected_recipe():
        raise RuntimeError("native benchmark recipe differs from its fixed analytical registration")
    sources = v.source_files()
    relative_config = str(config_path.relative_to(ROOT))
    if str(Path(__file__).resolve().relative_to(ROOT)) not in sources or relative_config not in sources:
        raise RuntimeError("native recipe and implementation must be inside the pinned source closure")
    source = digest(sources)
    actor, actorpath = v.old.checked_actor(ROOT)
    episodes, manifest = v.old.checked_inputs(ROOT)
    baseline, initial = v.checked_baseline()
    initial_ref = initial["fold-1-robot_2"]["structure"]
    structure, _ = normalize_structure(v.bound(initial_ref))
    fold = v.old.grouped_folds()[1]
    fit_rows = [row for case in fold["splits"]["fit"] for row in episodes[case["base"]]["robot_2"]]
    x, masks, targets, episode_ids = arrays(actor, "robot_2", fit_rows)
    names = tuple(actor.roles["robot_2"].feature_names)
    indices = tuple(names.index("legal_" + a.lower()) for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))
    if len(x) != expected_recipe()["fit_physical_frames"]:
        raise RuntimeError("fixed spent fit case has a different physical frame count")
    inputs = {"encoded": x, "masks": masks, "targets": targets, "episodes": np.asarray(episode_ids, str),
        "names": names, "indices": indices, "structure": structure}
    proposal = _proposal_binding(inputs)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source" / source
    for relative, expected in sources.items():
        raw = (ROOT / relative).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise RuntimeError("source changed while pinning the native diagnostic")
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
        path.chmod(0o444)
    input_path = output / "fit-inputs.npz"
    with input_path.open("xb") as handle:
        np.savez_compressed(handle, encoded=np.asarray(x, np.float32), masks=np.asarray(masks, np.float32),
            targets=np.asarray(targets, np.float32), episodes=np.asarray(episode_ids, str), names=np.asarray(names, str))
        handle.flush(); os.fsync(handle.fileno())
    input_path.chmod(0o444)
    structure_ref = write_new(output / "initial-structure.json", structure, readonly=True)
    proposal_ref = write_new(output / "fixed-native-queue.json", proposal, readonly=True)
    plan = dict(ZERO, schema="warehouse-v625-native-engineering-plan.v1", candidate_id=CANDIDATE,
        recipe=expected_recipe(), config={"path": str(config_path), "sha256": file_sha(config_path)},
        source_files=sources, source_sha256=source, source_snapshot=str(snapshot.resolve()),
        actor={"path": str(Path(actorpath).resolve()), "sha256": v.old.ACTOR_SHA256},
        actor_role_sha256=dict(v.old.ROLE_SHA256), historical_inputs=manifest,
        runtime=runtime_identity(),
        baseline=baseline, initial_S1=initial_ref, initial_structure=structure_ref,
        initial_structure_sha256=program_structure_sha256(structure), proposal=proposal_ref,
        fit_inputs={"path": str(input_path), "sha256": file_sha(input_path)},
        fit_episode_ids_sha256=digest(list(map(str, episode_ids))),
        no_fits_during_preparation=True, scientific_result_reuse=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    if digest(v.source_files()) != source:
        raise RuntimeError("source changed during native engineering preparation")
    plan_ref = write_new(output / "plan.json", plan, readonly=True)
    write_new(output / "plan.claim.json", {"sha256": plan_ref["sha256"], "single_use": True}, readonly=True)
    return Path(plan_ref["path"])


def checked_plan(path):
    from experiments.warehouse import v625_protocol as v
    from backend.training.warehouse_v622_program import normalize_structure, program_structure_sha256
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    claim = json.loads(path.with_name("plan.claim.json").read_bytes())
    if (claim != {"sha256": file_sha(path), "single_use": True}
            or plan.get("schema") != "warehouse-v625-native-engineering-plan.v1"
            or plan.get("recipe") != expected_recipe() or plan.get("candidate_id") != CANDIDATE
            or plan.get("runtime") != runtime_identity()
            or plan.get("source_files") != v.source_files() or plan.get("source_sha256") != digest(v.source_files())):
        raise RuntimeError("native engineering plan/source/recipe changed")
    for key, expected in ZERO.items():
        if type(plan.get(key)) is not type(expected) or plan[key] != expected:
            raise RuntimeError("native engineering authority changed")
    for relative, expected in plan["source_files"].items():
        file = Path(plan["source_snapshot"]) / relative
        if file_sha(file) != expected or file.stat().st_mode & 0o222:
            raise RuntimeError("native read-only source snapshot changed")
    for field in ("config", "actor", "initial_S1", "initial_structure", "proposal", "fit_inputs"):
        pointer = plan[field]
        if file_sha(pointer["path"]) != pointer["sha256"]:
            raise RuntimeError("native bound asset changed: " + field)
    if json.loads(Path(plan["config"]["path"]).read_bytes()) != expected_recipe():
        raise RuntimeError("native fixed recipe changed")
    structure, _ = normalize_structure(json.loads(Path(plan["initial_structure"]["path"]).read_bytes()))
    if program_structure_sha256(structure) != plan["initial_structure_sha256"]:
        raise RuntimeError("native pure initial tree changed")
    with np.load(plan["fit_inputs"]["path"], allow_pickle=False) as archive:
        if set(archive.files) != {"encoded", "masks", "targets", "episodes", "names"}:
            raise RuntimeError("unknown pinned fit input schema")
        inputs = {key: archive[key].copy() for key in ("encoded", "masks", "targets")}
        inputs["episodes"], inputs["names"] = archive["episodes"].copy(), tuple(archive["names"].tolist())
    inputs["indices"] = tuple(inputs["names"].index("legal_" + a.lower()) for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))
    inputs["structure"] = structure
    if (len(inputs["encoded"]) != expected_recipe()["fit_physical_frames"]
            or digest(list(map(str, inputs["episodes"]))) != plan["fit_episode_ids_sha256"]):
        raise RuntimeError("pinned fit frame/episode identity changed")
    for value in inputs.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return path, plan, inputs


def _numerical_state(state, inputs):
    """Independent post-timing replay; no fit and no production asset output."""
    from backend.training.warehouse_v625_program import evaluate_leaf
    raw = np.empty((len(inputs["encoded"]), 5), np.float32)
    q = np.empty_like(raw)
    models = []
    for leaf in state.leaves:
        rows = leaf.region.rows
        raw[rows], q[rows] = evaluate_leaf(leaf.model, inputs["encoded"][rows], inputs["masks"][rows])
        models.append({key: leaf.model[key] for key in ("version", "arithmetic", "support", "support_names", "bias", "weights", "witness", "certificate")})
    return {"tree_sha256": state.key.hex(), "score": list(state.score), "models_sha256": digest(models),
        "raw_logits_sha256": sha256(np.ascontiguousarray(raw, dtype="<f4").tobytes()).hexdigest(),
        "probabilities_sha256": sha256(np.ascontiguousarray(q, dtype="<f4").tobytes()).hexdigest(),
        "raw_argmax_sha256": sha256(np.ascontiguousarray(raw.argmax(-1), dtype="<i8").tobytes()).hexdigest(),
        "masked_argmax_sha256": sha256(np.ascontiguousarray(q.argmax(-1), dtype="<i8").tobytes()).hexdigest()}


def run_condition(condition_id, inputs, proposal, directory, identity, *, reservation=None):
    """Only callers holding a prepared, single-use outer reservation may call."""
    from backend.training.warehouse_program_search import _BeamEntry, _improves
    from backend.training.warehouse_v619_tree import _State, _size, _tree_key
    from backend.training.warehouse_v619_leaf_fit import ConstrainedLeafFitError
    from backend.training.warehouse_v623_search import _tree
    from backend.training.warehouse_v625_search import (
        _Context, _Codec, _score, _bounded_transitions, STATE_CACHE_ENTRIES, _tuple_tree, _solver_calls,
    )
    if condition_id not in ORDER:
        raise ValueError("unregistered native benchmark condition")
    _, inputs = checked_reservation(reservation, directory, identity, condition_id)
    mode, checkpoint = condition_id.rsplit("-", 1)[1], condition_id.startswith("checkpoint-")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    trace_states, candidate_trace, summaries = [], [], []
    started = time.monotonic()
    session, current_session, report = None, None, None
    all_cp_sessions = []
    direct = DirectCheckpointOverhead(all_cp_sessions)
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    try:
        with real_deadline(started, WALL_LIMIT):
            write_new(directory / "phase-started.json", {"condition_id": condition_id,
                "pid": os.getpid(), "started_monotonic": started, "hard_deadline_monotonic": started + WALL_LIMIT,
                "wall_seconds": WALL_LIMIT})
            # Fresh arrays and cache ownership in every condition. No fitted
            # numerical values are accepted from any earlier condition.
            x, masks, target = (np.array(inputs[key], copy=True) for key in ("encoded", "masks", "targets"))
            episodes, names, indices = inputs["episodes"].copy(), tuple(inputs["names"]), inputs["indices"]
            random.seed(62504); np.random.seed(62504)
            checkpoint_path = directory / "checkpoint.json"
            if checkpoint:
                with direct.measure("initial_checkpoint_session"):
                    session = CheckpointSession(checkpoint_path, identity, max_calls=CALL_LIMIT,
                        wall_seconds=WALL_LIMIT, started_at=started)
                    all_cp_sessions.append(session)
            else:
                session = BaselineLedger(directory / "baseline-call-ledger.jsonl", identity,
                    max_calls=CALL_LIMIT, wall_seconds=WALL_LIMIT, started_at=started)
            current_session = session
            context = _Context(x, masks, target, episodes, names, indices, mode, session)
            codec = _Codec(context) if checkpoint else None
            states = OrderedDict()
            def evaluate(tree):
                key = _tree_key(tree)
                if key in states:
                    return states[key]
                if _size(tree)[0] > 32 or _size(tree)[1] > 6:
                    raise RuntimeError("native candidate exceeds the registered tree limits")
                regions = list(context.regions(tree))
                leaves = [context.fit(region) for region in regions if region.tree is None]
                state = _State(tree, leaves, regions, _score(leaves), key)
                states[key] = state
                if len(states) > STATE_CACHE_ENTRIES:
                    states.popitem(last=False)
                return state
            start_state = best = evaluate(_tree(inputs["structure"]))
            trace_states.append(start_state)
            context.phase = "proposal"
            queue = _queue(context, start_state)
            if len(queue) != proposal["full_queue_count"]:
                raise RuntimeError("fresh native S1 queue length differs from the pre-fit pinned menu")
            cursor, transitions = 0, []
            queue_ref = None
            if checkpoint:
                with direct.measure("full_queue_codec"):
                    queue_ref = codec.queue_ref(queue)
            def payload():
                return {"schema": "warehouse-v625-native-prefix-checkpoint.v1", "cursor": cursor,
                    "queue_ref": queue_ref, "start": codec.state_ref(start_state), "best": codec.state_ref(best),
                    "states": [codec.state_ref(state) for state in states.values()],
                    "transitions": [[codec.state_ref(entry.state), entry.nonimproving_steps] for entry in transitions],
                    "leaf_cache": list(context.cache_refs.items()), "cache_hits": context.cache_hits,
                    "cache_evictions": context.cache_evictions, "candidate_trace": candidate_trace,
                    "python_rng": plain(random.getstate()), "numpy_rng": plain(np.random.get_state())}
            while cursor < PREFIX:
                context.phase = "candidate"
                tree, kind, label, parent = queue[cursor]
                entry = {"cursor": cursor, "tree_sha256": _tree_key(tree).hex(), "kind": kind,
                    "label": label, "parent_sha256": parent.state.key.hex()}
                try:
                    fitted = evaluate(tree)
                except ConstrainedLeafFitError as error:
                    trace_states.append(None)
                    entry.update(status="constrained_fit_failed", error_type=type(error).__name__)
                else:
                    debt = 0 if _improves(fitted.score, parent.state.score) else parent.nonimproving_steps + 1
                    if debt <= 2:
                        transitions = _bounded_transitions(transitions, _BeamEntry(fitted, debt))
                    if _improves(fitted.score, best.score) or fitted.score == best.score and fitted.key < best.key:
                        best = fitted
                    trace_states.append(fitted)
                    entry.update(status="fitted", score=list(fitted.score))
                candidate_trace.append(entry)
                cursor += 1
                if checkpoint and cursor == PAUSE:
                    with direct.measure("safe_pause_payload_codec_and_commit"):
                        session.commit(payload(), status="paused")
                        summaries.append(session.summary())
                    # Only the stored queue/cache/state controls work after
                    # restore. Earlier objects survive solely for untimed audit.
                    with direct.measure("safe_resume_full_state_cache_queue_decode"):
                        restored = CheckpointSession(checkpoint_path, identity, max_calls=CALL_LIMIT,
                            wall_seconds=WALL_LIMIT, resume=True)
                        all_cp_sessions.append(restored)
                        current_session = restored
                        context = _Context(x, masks, target, episodes, names, indices, mode, restored)
                        codec = _Codec(context)
                        saved = restored.state
                        for key, ref in saved["leaf_cache"]:
                            context.cache_refs[key], context.cache[key] = ref, restored.store.get_json(ref)
                        context.cache_hits, context.cache_evictions = saved["cache_hits"], saved["cache_evictions"]
                        states = OrderedDict()
                        for ref in saved["states"]:
                            state = codec.state(ref)
                            states[state.key] = state
                        start_state, best = codec.state(saved["start"]), codec.state(saved["best"])
                        transitions = [_BeamEntry(codec.state(ref), debt) for ref, debt in saved["transitions"]]
                        queue_ref = saved["queue_ref"]
                        queue = codec.queue(queue_ref)
                        cursor, candidate_trace = saved["cursor"], saved["candidate_trace"]
                        random.setstate(_tuple_tree(saved["python_rng"]))
                        numpy = saved["numpy_rng"]
                        np.random.set_state((numpy[0], np.asarray(numpy[1], np.uint32), numpy[2], numpy[3], numpy[4]))
                        if len(queue) != proposal["full_queue_count"]:
                            raise RuntimeError("restored full native queue length changed")
                    session = restored
            if checkpoint:
                with direct.measure("final_payload_codec_and_commit"):
                    session.commit(payload(), status="completed")
                    summaries.append(session.summary())
            else:
                session._append("baseline_completed", {"candidate_count": PREFIX, "calls": session.calls})
                session.close()
            # Production search reads/parses the durable ledger before taking
            # its runner stop clock. Do the same in both timing conditions.
            events = session.events()
            if checkpoint:
                summaries[-1] = session.summary()
            wall = time.monotonic() - started
            if wall > WALL_LIMIT or cursor != PREFIX or session.pending:
                raise RuntimeError("native prefix did not complete its preset work within budget")
            # Capture only actual measured segment I/O. The conservative
            # one-second commit reservations are billing, not real I/O time.
            io = direct.io() if checkpoint else session.ledger_io_seconds
            report = {"condition_id": condition_id, "mode": mode, "checkpoint": checkpoint,
                "status": "completed", "initial_tree_fitted_fresh": True, "candidate_prefix_completed": cursor,
                "pause_cursor": PAUSE if checkpoint else None, "full_queue_count": len(queue),
                "full_queue_sha256": proposal["full_queue_sha256"], "prefix_sha256": proposal["prefix_sha256"],
                "wall_seconds": wall, "actual_measured_IO_seconds": io, "checkpoint_segments": summaries,
                "direct_checkpoint_overhead": direct.report() if checkpoint else None,
                "actual_leaf_api_calls": session.calls, "reserved_call_slots": CALL_LIMIT,
                "cache_hits": context.cache_hits, "cache_evictions": context.cache_evictions,
                "leaf_cache_entries": len(context.cache), "state_cache_entries": len(states),
                "baseline_state_checkpoint_objects_created": False if not checkpoint else None,
                "ledger_path": str(session.ledger_path if checkpoint else session.path),
                "candidate_trace_sha256": digest(candidate_trace), "candidate_trace": candidate_trace}
        # Never place equivalence/model replay in the timing denominator.
        write_new(directory / "phase-completed.json", {"condition_id": condition_id, "pid": os.getpid(),
            "native_wall_seconds": report["wall_seconds"], "completed_monotonic": time.monotonic()})
        verification_started = time.monotonic()
        actual_queue_sha, actual_prefix_sha = queue_identity(queue), queue_identity(queue[:PREFIX])
        if actual_queue_sha != proposal["full_queue_sha256"] or actual_prefix_sha != proposal["prefix_sha256"]:
            raise RuntimeError("fresh/restored native queue order differs from its pinned pre-fit menu")
        numerical = [None if state is None else _numerical_state(state, inputs) for state in trace_states]
        report["numerical_trace_sha256"] = digest(numerical)
        report["numerical_trace"] = numerical
        report["post_timing_equivalence_seconds"] = time.monotonic() - verification_started
        report["actual_solver_calls"] = _solver_calls(events)
        report["settled_api_calls"] = sum(e["event"] == "call_completed" for e in events)
        report["ledger_sha256"] = file_sha(report["ledger_path"])
        report["native_work_scope"] = "fresh_initial_fit_full_proposal_queue_15_candidate_fits_statistics_all_checkpoint_commit_and_restore"
        return report
    except BaseException as error:
        if current_session is not None:
            failure = {"condition_id": condition_id, "status": "failed", "exception": type(error).__name__,
                "message": str(error), "actual_leaf_api_calls": current_session.calls,
                "reserved_call_slots": CALL_LIMIT, "wall_seconds": time.monotonic() - started,
                "candidate_trace": candidate_trace, "candidate_prefix_completed": len(candidate_trace),
                "refund_authorized": False, "retry_authorized": False}
            write_new(directory / "failed.json", failure)
            current_session.close()
        raise
    finally:
        random.setstate(saved_python); np.random.set_state(saved_numpy)


def paired_comparison(results):
    if [r["condition_id"] for r in results] != list(ORDER):
        raise RuntimeError("all four fixed balanced-order conditions are required")
    by_name = {r["condition_id"]: r for r in results}
    pairs = {}
    for mode in ("P0", "P1"):
        baseline, checkpoint = by_name["baseline-" + mode], by_name["checkpoint-" + mode]
        equal_keys = ("candidate_prefix_completed", "full_queue_count", "full_queue_sha256", "prefix_sha256",
            "actual_leaf_api_calls", "actual_solver_calls", "settled_api_calls", "cache_hits", "cache_evictions",
            "leaf_cache_entries", "state_cache_entries", "candidate_trace_sha256", "numerical_trace_sha256")
        checks = {key: baseline[key] == checkpoint[key] for key in equal_keys}
        difference = max(0., checkpoint["wall_seconds"] - baseline["wall_seconds"])
        direct = checkpoint["direct_checkpoint_overhead"]["direct_full_checkpoint_overhead_seconds"]
        overhead = max(difference, checkpoint["actual_measured_IO_seconds"], direct)
        fraction = overhead / checkpoint["wall_seconds"]
        pairs[mode] = {"equivalence_passed": all(checks.values()), "equivalence_checks": checks,
            "baseline_wall_seconds": baseline["wall_seconds"], "checkpoint_wall_seconds": checkpoint["wall_seconds"],
            "paired_positive_wall_difference_seconds": difference,
            "actual_checkpoint_IO_seconds": checkpoint["actual_measured_IO_seconds"],
            "direct_full_checkpoint_overhead_seconds": direct,
            "overhead_seconds": overhead, "checkpoint_fraction": fraction, "maximum_checkpoint_fraction": .10,
            "passed": all(checks.values()) and fraction <= .10}
    return {"passed": all(pair["passed"] for pair in pairs.values()), "pairs": pairs,
        "overhead_definition": "max(positive_paired_wall_difference,direct_full_CP_codec_restore_and_IO)/CP_wall",
        "post_timing_distribution_verification_excluded_from_denominator": True}


def checked_receipt(path, *, expected_source_sha256=None):
    """Read-only strict native admission evidence check, with no fits/resets."""
    path = Path(path).resolve(strict=True)
    receipt = json.loads(path.read_bytes())
    if (receipt.get("schema") != "warehouse-v625-native-engineering-receipt.v1"
            or receipt.get("candidate_id") != CANDIDATE or receipt.get("recipe") != expected_recipe()
            or receipt.get("all_four_conditions_completed") is not True
            or receipt.get("total_reserved_call_slots") != 4 * CALL_LIMIT
            or expected_source_sha256 is not None and receipt.get("source_sha256") != expected_source_sha256):
        raise RuntimeError("not a complete registered native-04 engineering receipt")
    for key, expected in dict(ZERO, scientific_result_reuse=False, fresh_data_used=False,
                              production_program_generated=False, automatic_retry=False).items():
        if type(receipt.get(key)) is not type(expected) or receipt[key] != expected:
            raise RuntimeError("native receipt authority or data-use changed")
    plan_pointer = receipt.get("plan", {})
    if file_sha(plan_pointer.get("path", "")) != plan_pointer.get("sha256"):
        raise RuntimeError("native receipt plan pointer changed")
    plan_path, plan, _ = checked_plan(plan_pointer["path"])
    if (receipt.get("plan_sha256") != file_sha(plan_path) or receipt.get("source_sha256") != plan["source_sha256"]
            or receipt.get("source_files") != plan["source_files"]):
        raise RuntimeError("native receipt source/plan binding changed")
    attempt_pointer = receipt.get("attempt_start", {})
    if file_sha(attempt_pointer.get("path", "")) != attempt_pointer.get("sha256"):
        raise RuntimeError("native upfront reservation changed")
    attempt = json.loads(Path(attempt_pointer["path"]).read_bytes())
    if (Path(attempt_pointer["path"]).resolve() != plan_path.parent / "attempt-start.json"
            or attempt.get("recipe") != expected_recipe() or attempt.get("plan_sha256") != file_sha(plan_path)
            or attempt.get("total_charged_call_slots") != 4 * CALL_LIMIT
            or attempt.get("reservations") != [{"condition_id": name, "charged_call_slots": CALL_LIMIT} for name in ORDER]
            or attempt.get("unused_slots_refundable") is not False):
        raise RuntimeError("native reservation scope/budget mismatch")
    pointers = receipt.get("conditions")
    if not isinstance(pointers, list) or len(pointers) != 4:
        raise RuntimeError("all four native condition results are required")
    results = []
    for condition_id, pointer in zip(ORDER, pointers, strict=True):
        expected = plan_path.parent / "conditions" / condition_id / "result.json"
        if Path(pointer["path"]).resolve() != expected or file_sha(expected) != pointer["sha256"]:
            raise RuntimeError("native condition result identity changed")
        result = json.loads(expected.read_bytes())
        if (result.get("condition_id") != condition_id or result.get("status") != "completed"
                or result.get("candidate_prefix_completed") != PREFIX or result.get("reserved_call_slots") != CALL_LIMIT
                or type(result.get("actual_leaf_api_calls")) is not int or not 0 < result["actual_leaf_api_calls"] <= CALL_LIMIT
                or result.get("settled_api_calls") != result["actual_leaf_api_calls"]
                or not 0 < result.get("wall_seconds", 0) <= WALL_LIMIT
                or file_sha(result["ledger_path"]) != result["ledger_sha256"]):
            raise RuntimeError("native condition work/budget/ledger changed")
        tail, sequence, calls, pending = "0" * 64, 0, 0, set()
        for line in Path(result["ledger_path"]).read_bytes().splitlines():
            row = json.loads(line)
            observed = row.pop("record_sha256")
            if row["sequence"] != sequence + 1 or row["previous_sha256"] != tail or digest(row) != observed:
                raise RuntimeError("native durable API ledger chain changed")
            sequence, tail = row["sequence"], observed
            if row["event"] == "call_started":
                if pending or row["data"]["call"] != calls + 1:
                    raise RuntimeError("native solver lacks its preceding unique durable predebit")
                calls += 1; pending.add(calls)
            elif row["event"] == "call_completed":
                token = row["data"]["call"]
                if token not in pending:
                    raise RuntimeError("native solver settlement has no debit")
                pending.remove(token)
        if pending or calls != result["actual_leaf_api_calls"]:
            raise RuntimeError("native actual API calls differ from nonrefundable ledger")
        directory = expected.parent
        external = json.loads((directory / "external-controller.json").read_bytes())
        phase = json.loads((directory / "phase-started.json").read_bytes())
        completed = json.loads((directory / "phase-completed.json").read_bytes())
        if (external.get("all_owned_processes_waited") is not True or external.get("returncode") != 0
                or phase.get("pid") != external.get("pid") or completed.get("pid") != external.get("pid")
                or completed.get("native_wall_seconds") != result["wall_seconds"]
                or any(item.get("condition_id") != condition_id for item in (external, phase, completed))):
            raise RuntimeError("native external deadline/reaping evidence differs")
        results.append(result)
    comparison = paired_comparison(results)
    if (receipt.get("comparison") != comparison or receipt.get("passed") != comparison["passed"]
            or receipt.get("actual_leaf_api_calls") != sum(r["actual_leaf_api_calls"] for r in results)
            or receipt.get("actual_solver_calls") != sum(r["actual_solver_calls"] for r in results)):
        raise RuntimeError("native receipt comparison/counts differ from its four results")
    return receipt


def _worker_authorization(output, condition_id, expected_sha256):
    path = Path(output) / "authorizations" / (condition_id + ".json")
    if file_sha(path) != expected_sha256:
        raise RuntimeError("native worker authorization hash mismatch")
    value = json.loads(path.read_bytes())
    if (value.get("condition_id") != condition_id or value.get("order_index") != ORDER.index(condition_id)
            or value.get("schema") != "warehouse-v625-native-condition-authority.v1"):
        raise RuntimeError("native worker condition/ordinal is not authorized")
    previous = value.get("previous_results")
    if not isinstance(previous, list) or len(previous) != ORDER.index(condition_id):
        raise RuntimeError("native balanced condition order changed")
    for index, pointer in enumerate(previous):
        if file_sha(pointer["path"]) != pointer["sha256"]:
            raise RuntimeError("earlier native condition result changed")
        result = json.loads(Path(pointer["path"]).read_bytes())
        if result.get("condition_id") != ORDER[index] or result.get("status") != "completed":
            raise RuntimeError("earlier reserved native condition did not complete")
    return value


def condition_worker(plan_path, condition_id, authorization_sha256):
    from threadpoolctl import threadpool_info, threadpool_limits
    path, plan, inputs = checked_plan(plan_path)
    authority = _worker_authorization(path.parent, condition_id, authorization_sha256)
    reservation_path = path.parent / "attempt-start.json"
    reservation = {"path": str(reservation_path), "sha256": file_sha(reservation_path)}
    if (authority.get("plan_sha256") != file_sha(path) or authority.get("source_sha256") != plan["source_sha256"]
            or authority.get("reservation") != reservation):
        raise RuntimeError("native worker authority is not bound to this source/plan/reservation")
    identity = {"engineering_native_attempt": "04", "condition_id": condition_id, "plan_sha256": file_sha(path),
        "source_sha256": plan["source_sha256"], "fit_inputs_sha256": plan["fit_inputs"]["sha256"],
        "initial_structure_sha256": plan["initial_structure_sha256"], "fixed_recipe": expected_recipe()}
    proposal = json.loads(Path(plan["proposal"]["path"]).read_bytes())
    with threadpool_limits(limits=1):
        if any(item.get("num_threads", 1) != 1 for item in threadpool_info()):
            raise RuntimeError("native worker numerical libraries must be single-threaded")
        result = run_condition(condition_id, inputs, proposal, path.parent / "conditions" / condition_id,
            identity, reservation=reservation)
    write_new(path.parent / "conditions" / condition_id / "result.json", result)
    return result


def _stop_and_wait(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    else:
        process.wait()


def supervise_condition(plan_path, condition_id, authority_sha256):
    """Serial external process-group deadline, independent of child alarms."""
    path = Path(plan_path)
    directory = path.parent / "conditions" / condition_id
    logs = path.parent / "logs"
    logs.mkdir(exist_ok=True)
    environment = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    process, stop_reason = None, None
    launched = time.monotonic()
    with (logs / (condition_id + ".stdout.log")).open("xb") as stdout, (logs / (condition_id + ".stderr.log")).open("xb") as stderr:
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--condition", condition_id,
                "--plan", str(path), "--authorization-sha256", authority_sha256], cwd=ROOT,
                env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                now = time.monotonic()
                started_path, completed_path = directory / "phase-started.json", directory / "phase-completed.json"
                if started_path.exists():
                    phase = json.loads(started_path.read_bytes())
                    if phase.get("pid") != process.pid or phase.get("condition_id") != condition_id:
                        stop_reason = "native phase marker identity mismatch"
                    elif completed_path.exists():
                        completed = json.loads(completed_path.read_bytes())
                        if (completed.get("pid") != process.pid or completed.get("condition_id") != condition_id
                                or completed.get("native_wall_seconds", WALL_LIMIT + 1) > WALL_LIMIT):
                            stop_reason = "late/malformed native completion marker"
                        elif now - completed["completed_monotonic"] > 120:
                            stop_reason = "fixed post-timing verification envelope exhausted"
                    elif now >= phase["hard_deadline_monotonic"]:
                        stop_reason = "external native 1200-second hard deadline"
                elif now - launched > 120:
                    stop_reason = "fixed native pre-timing setup envelope exhausted"
                if stop_reason:
                    raise DeadlineExceeded(stop_reason)
                time.sleep(.25)
            returncode = process.wait()
            if returncode:
                raise RuntimeError("native condition subprocess failed: " + condition_id + " exit=" + str(returncode))
            result_path = directory / "result.json"
            result = json.loads(result_path.read_bytes())
            phase = json.loads((directory / "phase-started.json").read_bytes())
            completed = json.loads((directory / "phase-completed.json").read_bytes())
            if (result.get("status") != "completed" or result.get("wall_seconds", WALL_LIMIT + 1) > WALL_LIMIT
                    or phase.get("pid") != process.pid or completed.get("pid") != process.pid
                    or completed.get("native_wall_seconds") != result["wall_seconds"]
                    or phase.get("condition_id") != condition_id or completed.get("condition_id") != condition_id):
                raise RuntimeError("native subprocess did not produce an on-time completed condition")
            write_new(directory / "external-controller.json", {"condition_id": condition_id, "pid": process.pid,
                "returncode": returncode, "all_owned_processes_waited": True, "hard_deadline_seconds": WALL_LIMIT,
                "startup_and_postverification_envelopes_seconds": 120, "stop_reason": None})
            return result
        finally:
            if process is not None:
                _stop_and_wait(process)
                if stop_reason and directory.exists():
                    write_new(directory / "external-controller-stopped.json", {"condition_id": condition_id,
                        "pid": process.pid, "stop_reason": stop_reason, "all_owned_processes_waited": True,
                        "retry_authorized": False})


def execute(plan_path):
    """Execute only after root has pinned the recipe and implementation source."""
    from experiments.warehouse import v625_protocol as v
    path, plan, inputs = checked_plan(plan_path)
    output = path.parent
    reservation = write_new(output / "attempt-start.json", dict(ZERO, schema="warehouse-v625-native-attempt.v1",
        plan_sha256=file_sha(path), source_sha256=plan["source_sha256"], recipe=expected_recipe(),
        reservations=[{"condition_id": name, "charged_call_slots": CALL_LIMIT} for name in ORDER],
        total_charged_call_slots=4 * CALL_LIMIT, unused_slots_refundable=False,
        started_utc=datetime.now(timezone.utc).isoformat()))
    results, condition_refs = [], []
    try:
        for index, name in enumerate(ORDER):
            if digest(v.source_files()) != plan["source_sha256"]:
                raise RuntimeError("native source changed before a reserved condition")
            authority = write_new(output / "authorizations" / (name + ".json"), {
                "schema": "warehouse-v625-native-condition-authority.v1", "condition_id": name,
                "order_index": index, "previous_results": list(condition_refs), "plan_sha256": file_sha(path),
                "source_sha256": plan["source_sha256"], "reservation": reservation}, readonly=True)
            result = supervise_condition(path, name, authority["sha256"])
            result_path = output / "conditions" / name / "result.json"
            result_ref = {"path": str(result_path.resolve()), "sha256": file_sha(result_path)}
            results.append(result); condition_refs.append(result_ref)
        comparison = paired_comparison(results)
        if digest(v.source_files()) != plan["source_sha256"]:
            raise RuntimeError("native source changed during the benchmark")
        receipt = dict(ZERO, schema="warehouse-v625-native-engineering-receipt.v1", candidate_id=CANDIDATE,
            source_sha256=plan["source_sha256"], source_files=plan["source_files"], plan_sha256=file_sha(path),
            plan={"path": str(path), "sha256": file_sha(path)}, attempt_start=reservation,
            recipe=expected_recipe(), passed=comparison["passed"], status="passed" if comparison["passed"] else "failed_native_overhead_or_equivalence",
            comparison=comparison, conditions=condition_refs, total_reserved_call_slots=4 * CALL_LIMIT,
            actual_leaf_api_calls=sum(r["actual_leaf_api_calls"] for r in results),
            actual_solver_calls=sum(r["actual_solver_calls"] for r in results),
            all_four_conditions_completed=True, scientific_result_reuse=False,
            fresh_data_used=False, production_program_generated=False, automatic_retry=False,
            measurement_scope="new_native_fit_runner_protocol_not_a_reinterpretation_of_failed_routing_only_01_02_03",
            actual_scientific_jobs_still_require_checkpoint_fraction_at_most_ten_percent=True,
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        failure = dict(ZERO, schema="warehouse-v625-native-engineering-failure.v1", candidate_id=CANDIDATE,
            source_sha256=plan["source_sha256"], passed=False, exception=type(error).__name__, message=str(error),
            total_reserved_call_slots=4 * CALL_LIMIT, completed_conditions=condition_refs,
            actual_calls_in_completed_conditions=sum(r["actual_leaf_api_calls"] for r in results),
            no_refund_or_retry=True, fresh_data_used=False, scientific_result_reuse=False)
        write_new(output / "failed.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--condition", choices=ORDER)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args()
    if args.prepare:
        if args.config is None or args.output is None:
            parser.error("--prepare requires --config and --output")
        result = {"plan": str(prepare(args.config, args.output)), "fit_calls": 0}
    elif args.execute:
        if args.plan is None:
            parser.error("--execute requires --plan")
        receipt = execute(args.plan)
        result = {"passed": receipt["passed"], "receipt": str(args.plan.parent / "receipt.json"),
            "actual_leaf_api_calls": receipt["actual_leaf_api_calls"]}
    else:
        if args.plan is None or args.authorization_sha256 is None:
            parser.error("--condition requires a pinned --plan and --authorization-sha256")
        condition = condition_worker(args.plan, args.condition, args.authorization_sha256)
        result = {"condition_id": args.condition, "status": condition["status"], "actual_leaf_api_calls": condition["actual_leaf_api_calls"]}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
