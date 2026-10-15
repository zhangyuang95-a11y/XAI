"""Warehouse A evidence is restricted to the frozen Actor's audited leaves."""
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from domains.warehouse import neural_turnbased as warehouse


def test_formal_human_advice_uses_bound_human_actor_only(monkeypatch):
    monkeypatch.setattr(warehouse, '_restore',
                        lambda state: SimpleNamespace(state=object(),
                            config=SimpleNamespace(map_layout_id='fixture')))
    monkeypatch.setattr(warehouse, '_selection',
                        lambda env, seed, policy_id: ('UP', {'robot_1': 'RIGHT',
                                                             'robot_2': 'UP'}, {}))
    monkeypatch.setattr(warehouse, 'action_masks',
                        lambda state, layout: np.ones((2, len(warehouse.ACTIONS))))
    state = {'terminal': False, 'policy_id': 'warehouse-regularized-v5',
             'scenario_seed': 123}
    assert warehouse.human_advisor(state) == 'right'
    monkeypatch.setattr(warehouse, 'action_masks',
                        lambda state, layout: np.zeros((2, len(warehouse.ACTIONS))))
    with pytest.raises(RuntimeError, match='physically illegal'):
        warehouse.human_advisor(state)


def test_warehouse_program_evidence_requires_final_audit_and_local_agreement(
        tmp_path, monkeypatch):
    role_sha = '2' * 64
    actor_sha = 'a' * 64
    path = tmp_path / 'program.json'
    path.write_text('{}')
    program_sha = sha256(path.read_bytes()).hexdigest()
    metric = {'rows': 600, 'coverage': .25, 'agreement': .92,
              'kl': .2, 'illegal_count': 0}
    audit = {'audit_source': 'final_independent_development_rollouts',
             'development_seeds': list(range(240100, 240124)),
             'actor_role_sha256': role_sha, 'program_sha256': program_sha,
             'program_depth': 5, 'program_leaf_count': 16,
             'sample_count': 2400, 'explanation_eligible': True,
             'validation': {'explanation': metric}}
    report_path = tmp_path / 'report.json'
    receipt_path = tmp_path / 'receipt.json'
    report = {'artifact': {'sha256': actor_sha},
              'selected_role_sha256': {'robot_2': role_sha},
              'selected_program_audit': {'robot_2': audit}}
    report_path.write_text(json.dumps(report))
    receipt_path.write_text(json.dumps({'final_program_audit': {'robot_2': audit}}))
    inventory = {'readiness': True, 'training_report_path': report_path.name,
                 'acceptance_path': receipt_path.name,
                 'actor_sha256': actor_sha,
                 'role_sha256': {'human': '1' * 64, 'ai': role_sha},
                 'program_paths': ['human.json', path.name],
                 'program_sha256': {'robot_2': program_sha}}
    monkeypatch.setattr(warehouse, 'ROOT', tmp_path)
    monkeypatch.setattr(warehouse, 'policy_artifacts',
                        lambda: {'regularized': inventory})
    monkeypatch.setattr(warehouse, '_restore',
                        lambda state: SimpleNamespace(state=object(),
                            config=SimpleNamespace(map_layout_id='fixture')))
    monkeypatch.setattr(warehouse, 'observation',
                        lambda state, role: np.zeros(len(warehouse.observation_feature_names())))
    monkeypatch.setattr(warehouse, 'action_masks',
                        lambda state, layout: np.ones((2, len(warehouse.ACTIONS))))
    monkeypatch.setattr(warehouse, '_selection',
                        lambda env, seed, policy_id: ('UP', {}, {}))

    class Program:
        role = 'robot_2'
        def covered(self, sample, *, actor_action, purpose):
            return ({'leaf_id': 3, 'path': []}
                    if actor_action == 'UP' and purpose == 'explanation' else None)

    monkeypatch.setattr(warehouse.AxisLeafProgram, 'load_json',
                        classmethod(lambda cls, *args, **kwargs: Program()))
    state = {'policy_id': 'warehouse-regularized-v5', 'terminal': False,
             'scenario_seed': 240100}
    evidence = warehouse.audited_program_evidence(state)
    assert evidence['eligible'] is True and evidence['leaf_id'] == 3
    assert '25.0%' in evidence['en'] and '92.0%' in evidence['en']

    metric['agreement'] = .89
    report_path.write_text(json.dumps(report))
    receipt_path.write_text(json.dumps({'final_program_audit': {'robot_2': audit}}))
    assert warehouse.audited_program_evidence(state) is None
