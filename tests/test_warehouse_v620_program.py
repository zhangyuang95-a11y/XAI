"""Queued programs remain distribution fits, never runtime controllers."""
from copy import deepcopy

import numpy as np
import pytest

from backend.training.warehouse_v6_program import (
    QUEUED_CONSTRAINED_METHOD, ROBUST_EXPLANATION_METHOD,
    QUEUED_PROGRAM_VERSION, ROBUST_PROGRAM_VERSION,
    WarehouseV6Program, certified_program_actions, fit_program,
)
from tests.test_warehouse_v618_program import _soft_actor, _parts


@pytest.mark.parametrize('method,version', [
    (QUEUED_CONSTRAINED_METHOD, QUEUED_PROGRAM_VERSION),
    (ROBUST_EXPLANATION_METHOD, ROBUST_PROGRAM_VERSION),
])
def test_queued_fit_roundtrip_isolation_and_strict_method_binding(monkeypatch, method, version):
    from backend.training import warehouse_program_search as search
    real = search.fit_legal_structure
    config = search.SearchConfig(initializations=1, max_rounds=1,
        max_leaf_optimization_calls=16, proposals_per_round=2,
        axis_features=1, semantic_candidates_per_node=1)
    monkeypatch.setattr(search, 'fit_legal_structure',
        lambda *args, **kwargs: real(*args, **kwargs, config=config))
    actor, _ = _soft_actor()
    before = {role: {key: value.copy() for key, value in item.weights.items()}
              for role, item in actor.roles.items()}
    parts = _parts()
    program = fit_program('robot_1', actor, *parts, method=method)
    changed = [{**row, 'sampled_action': 'RIGHT'} for row in parts[-1]]
    other = fit_program('robot_1', actor, *parts[:-1], changed, method=method)
    assert program.report['fit_tree'] == other.report['fit_tree']
    np.testing.assert_array_equal(program.leaf_logits, other.leaf_logits)
    assert program.feedback_leaves == other.feedback_leaves
    assert not other.report['validation']['feedback_passed']
    assert program.report['leaf_optimization_calls_outside_search'] == 0
    assert program.report['convex_solver_calls_outside_search'] == 0
    for leaf, actions in certified_program_actions(program).items():
        assert program.leaf_logits[leaf].argmax() in actions
    data = program.to_dict()
    assert data['version'] == version
    restored = WarehouseV6Program.from_dict(data, actor_role_sha256=program.actor_role_sha256)
    assert restored.sha256() == program.sha256()
    tampered = deepcopy(data)
    tampered['version'] = ROBUST_PROGRAM_VERSION if version == QUEUED_PROGRAM_VERSION else QUEUED_PROGRAM_VERSION
    with pytest.raises(RuntimeError, match='version|method'):
        WarehouseV6Program.from_dict(tampered, actor_role_sha256=program.actor_role_sha256)
    for role, item in actor.roles.items():
        for key, value in before[role].items():
            np.testing.assert_array_equal(item.weights[key], value)
