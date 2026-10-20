"""Regression checks for the new spent-data experiment boundary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.training.warehouse_v623_search import ContinuationBudget, METHODS, _tree
from backend.training.warehouse_v619_tree import _size
from backend.training.warehouse_v622_program import program_structure_sha256
from experiments.warehouse import v623_protocol as protocol


def test_prior_six_are_bound_to_closed_audited_batch():
    assert len(protocol.jobs()) == 6
    for job in protocol.jobs():
        refs = protocol.checked_prior(job)
        tree = json.loads(Path(refs["structure"]["path"]).read_bytes())
        assert program_structure_sha256(tree)
        leaves, depth = _size(_tree(tree))
        assert leaves <= 32 and depth <= 6


def test_registered_methods_and_incremental_budget():
    assert set(METHODS) == {"S1", "S2"}
    budget = ContinuationBudget()
    assert budget.maximum_new_leaf_calls == 20000
    assert budget.final_leaf_reserve == 32
    assert budget.wall_seconds - budget.final_wall_reserve == 4800
    with pytest.raises(ValueError):
        ContinuationBudget(maximum_new_leaf_calls=20001)
    with pytest.raises(ValueError):
        ContinuationBudget(final_leaf_reserve=31)


def test_recipe_keeps_pilot_closed_and_new_source_bound():
    assert len(protocol.recipe()) == 64
    recipe = json.loads((protocol.ROOT / protocol.RECIPE).read_bytes())
    assert recipe["future_gates"]["pilot_open"] is False
    files = protocol.source_files()
    assert all(file in files for file in protocol.NEW_FILES)
