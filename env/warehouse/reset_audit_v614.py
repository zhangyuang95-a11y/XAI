"""No-op-unless-armed audit of successful Warehouse v6.14 resets.

The hook is called after reset state construction. It does not read or
advance the environment RNG and cannot change actions or observations.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
from typing import BinaryIO
from typing import Iterator


_AUDIT_SINK: ContextVar[tuple[list[dict], BinaryIO] | None] = ContextVar(
    "warehouse_v614_reset_sink", default=None)
_AUDIT_PHASE: ContextVar[str] = ContextVar(
    "warehouse_v614_reset_phase", default="unassigned")
_AUDIT_METADATA: ContextVar[dict | None] = ContextVar(
    "warehouse_v614_reset_metadata", default=None)


@contextmanager
def capture_stage1_resets(journal_path: str | Path) -> Iterator[list[dict]]:
    if _AUDIT_SINK.get() is not None:
        raise RuntimeError("Warehouse reset audit is already active")
    rows: list[dict] = []
    with Path(journal_path).open("xb") as handle:
        token = _AUDIT_SINK.set((rows, handle))
        try:
            yield rows
        finally:
            try:
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                _AUDIT_SINK.reset(token)


@contextmanager
def stage1_reset_phase(phase: str, *, start_joint_step: int | None = None,
                       episode_index: int | None = None) -> Iterator[None]:
    if _AUDIT_SINK.get() is None or phase not in (
            "bc", "quality", "pressure", "program", "formal_bc", "bc_quality_screen",
            "formal_raw_rollout", "formal_raw_selection",
            "formal_regularized_rollout", "formal_regularized_selection",
            "formal_regularized_program"):
        raise RuntimeError("Warehouse reset audit phase is not authorized")
    rollout = phase in ("formal_raw_rollout", "formal_regularized_rollout")
    if rollout != (type(start_joint_step) is int and start_joint_step >= 0 and
                   type(episode_index) is int and episode_index >= 0):
        raise RuntimeError("Warehouse formal rollout reset metadata is incomplete")
    token = _AUDIT_PHASE.set(phase)
    metadata_token = _AUDIT_METADATA.set(
        {"start_joint_step": start_joint_step, "episode_index": episode_index}
        if rollout else None)
    try:
        yield
    finally:
        _AUDIT_METADATA.reset(metadata_token)
        _AUDIT_PHASE.reset(token)


def record_successful_reset(seed: int | None, episode_id: int,
                            map_layout_id: str, move_battery_cost: float,
                            charge_per_wait: float) -> None:
    sink = _AUDIT_SINK.get()
    if sink is None:
        return
    if seed is None or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("Warehouse audited reset must have an explicit seed")
    rows, handle = sink
    row = {"phase": _AUDIT_PHASE.get(), "seed": seed,
           "episode_id": int(episode_id),
           "map_layout_id": str(map_layout_id),
           "move_battery_cost": float(move_battery_cost),
           "charge_per_wait": float(charge_per_wait)}
    if _AUDIT_METADATA.get() is not None:
        row.update(_AUDIT_METADATA.get())
    handle.write((json.dumps(row, sort_keys=True,
                             separators=(",", ":"), allow_nan=False) + "\n").encode())
    handle.flush()
    os.fsync(handle.fileno())
    rows.append(row)


__all__ = ["capture_stage1_resets", "stage1_reset_phase",
           "record_successful_reset"]
