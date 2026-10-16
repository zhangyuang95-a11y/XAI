"""Isolated legacy replay-test scope for unregistered synthetic seeds.

The production runner stays fail-closed.  These unit tests deliberately use
the same private evaluation marker as the locked Stage 1 harness, never the
training marker, so they can construct small physical traces for tamper tests.
No registered development, sealed audit, or held-out seed is used here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from backend.training.warehouse_v6_runner import (
    _ACTIVE_LOCKED_EXECUTION, _LOCKED_EVALUATION_MARKER,
)


@contextmanager
def isolated_evaluation_scope() -> Iterator[None]:
    token = _ACTIVE_LOCKED_EXECUTION.set(_LOCKED_EVALUATION_MARKER)
    try:
        yield
    finally:
        _ACTIVE_LOCKED_EXECUTION.reset(token)
