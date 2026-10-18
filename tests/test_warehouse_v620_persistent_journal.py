from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import ast

import pytest


@pytest.fixture
def wrapper(monkeypatch):
    # Isolate the two extracted definitions. No production backend, policy,
    # Torch, or environment module is imported by this synthetic test.
    backend, training = ModuleType("backend"), ModuleType("backend.training")
    backend.__path__, training.__path__ = [], []
    auth = ModuleType("backend.training.warehouse_v6_evaluation_authorization")
    auth.validate_evaluation_authorization = lambda **kw: {
        "purpose": "unsealed_development_diagnostic", "expected_source_closure_sha256": "synthetic"}
    for name, value in (("backend", backend), ("backend.training", training),
            ("backend.training.warehouse_v6_evaluation_authorization", auth)):
        monkeypatch.setitem(sys.modules, name, value)
    state = SimpleNamespace(rows=None, handle=None, options=None)

    @contextmanager
    def capture(path):
        rows = []
        with Path(path).open("xb") as handle:
            state.rows, state.handle = rows, handle
            try:
                yield rows
            finally:
                handle.flush()

    @contextmanager
    def phase(_):
        yield

    def core(actor, seeds, **options):
        state.options = options
        for seed in seeds:
            state.rows.append({"seed": seed})
            state.handle.write((str(seed)+"\n").encode())
        if "outcome_sink" in options:
            options["outcome_sink"].append({"seed": seeds[0], "frame_count": 1})
        return {"synthetic_rows": True}

    namespace = {"__name__": "backend.training._synthetic_wrapper",
        "Path": Path, "TemporaryDirectory": TemporaryDirectory, "sha256": sha256,
        "_ACTIVE_LOCKED_EXECUTION": ContextVar("synthetic_marker", default=None),
        "_LOCKED_EVALUATION_MARKER": object(), "capture_stage1_resets": capture,
        "stage1_reset_phase": phase, "_assert_source_closure": lambda expected: None,
        "collect_program_audit_rows": core}
    source = (Path(__file__).resolve().parents[1]/"backend/training/warehouse_v6_runner.py").read_text()
    parsed = ast.parse(source)
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_authorized_frozen_operation", "collect_authorized_program_rows")]
    assert len(functions) == 2
    text = "from __future__ import annotations\n" + "\n\n".join(ast.get_source_segment(source, n) for n in functions)
    exec(text, namespace)
    return namespace, state


def test_default_wrapper_preserves_legacy_evidence_shape(wrapper):
    ns, _ = wrapper
    result, evidence = ns["_authorized_frozen_operation"](None, [31], None,
        lambda: ns["collect_program_audit_rows"](None, [31]), kind="final_program_audit")
    assert result == {"synthetic_rows": True}
    assert set(evidence["reset_audit"]) == {"resets", "journal_sha256", "actual_reset_count"}
    assert evidence["reset_audit"]["actual_reset_count"] == 1
    assert ns["_ACTIVE_LOCKED_EXECUTION"].get() is None


@pytest.mark.parametrize("persistent", [False, True])
def test_post_operation_source_failure_retains_consumed_reset_and_persistent_journal(wrapper, tmp_path, persistent):
    ns, _ = wrapper
    journal = tmp_path/"nested/resets.jsonl"
    def fail(_):
        raise RuntimeError("synthetic source drift after operation")
    ns["_assert_source_closure"] = fail
    with pytest.raises(RuntimeError, match="source drift") as caught:
        ns["_authorized_frozen_operation"](None, [31], None,
            lambda: ns["collect_program_audit_rows"](None, [31]),
            kind="final_program_audit", **({"journal_path": journal} if persistent else {}))
    audit = caught.value.warehouse_reset_audit
    assert audit["resets"] == [{"seed": 31}]
    assert audit["actual_reset_count"] == 1
    assert audit["journal_sha256"] == sha256(b"31\n").hexdigest()
    if persistent:
        assert audit["journal_sha256"] == sha256(journal.read_bytes()).hexdigest()
        assert audit["journal_path"] == str(journal)
    else:
        assert "journal_path" not in audit
    assert ns["_ACTIVE_LOCKED_EXECUTION"].get() is None


@pytest.mark.parametrize("persistent", [False, True])
def test_keyboard_interrupt_retains_reset_in_default_temporary_mode(wrapper, tmp_path, persistent):
    ns, _ = wrapper
    def operation():
        ns["collect_program_audit_rows"](None, [31])
        raise KeyboardInterrupt("synthetic interruption")
    with pytest.raises(KeyboardInterrupt) as caught:
        ns["_authorized_frozen_operation"](None, [31], None, operation, kind="final_program_audit",
            **({"journal_path": tmp_path/"interrupt.jsonl"} if persistent else {}))
    audit = caught.value.warehouse_reset_audit
    assert audit["resets"] == [{"seed": 31}]
    assert audit["journal_sha256"] == sha256(b"31\n").hexdigest()
    assert ("journal_path" in audit) is persistent
    assert ns["_ACTIVE_LOCKED_EXECUTION"].get() is None


def test_public_collector_optional_journal_preserves_default_options(wrapper, tmp_path):
    ns, state = wrapper
    result, evidence = ns["collect_authorized_program_rows"](None, [31])
    assert state.options == {}
    assert "episode_outcomes" not in evidence
    result, evidence = ns["collect_authorized_program_rows"](None, [31],
        deterministic=True, action_seed_offset=10, include_episode_outcomes=True,
        journal_path=tmp_path/"persistent.jsonl")
    assert state.options["deterministic"]
    assert state.options["action_seed_offset"] == 10
    assert evidence["episode_outcomes"] == [{"seed": 31, "frame_count": 1}]
    assert evidence["sampling_mode"] == "argmax"
    assert evidence["reset_audit"]["journal_path"].endswith("persistent.jsonl")


def test_existing_journal_is_never_overwritten(wrapper, tmp_path):
    ns, _ = wrapper
    journal = tmp_path/"already-claimed.jsonl"
    journal.write_bytes(b"preserved\n")
    with pytest.raises(FileExistsError):
        ns["collect_authorized_program_rows"](None, [31], journal_path=journal)
    assert journal.read_bytes() == b"preserved\n"
