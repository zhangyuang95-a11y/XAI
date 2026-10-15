"""The v5 Pong binding must identify and execute one frozen NN pair."""
from copy import deepcopy
import hashlib
import json

import pytest

from domains.pong import nn_policy, turnbased as pong
from core.program import ExecutableProgram, ProgramNode
from scripts.train_rolling_pong_v5 import HELDOUT_SEEDS, reserve_heldout_once


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _write_local_receipt(manifest, root):
    variants = manifest['variants']
    (root / variants['raw']['acceptance_path']).write_text(json.dumps({
        'release': nn_policy.V5_RELEASE_ID,
        'passed': True, 'heldout_evaluated': True,
        'artifact_sha256': {name: entry['actor_sha256']
                            for name, entry in variants.items()},
        'training_report_sha256': {name: entry['training_report_sha256']
                                   for name, entry in variants.items()},
        'program_sha256': {'regularized': {
            role: variants['regularized']['program_sha256'][path]
            for role, path in zip(('human', 'ai'), variants['regularized']['program_paths'])}},
    }))


@pytest.fixture
def bound_policies(monkeypatch, tmp_path):
    monkeypatch.setattr(nn_policy, 'ROOT', tmp_path)
    monkeypatch.setattr(nn_policy, 'V5_MANIFEST_PATH', tmp_path / 'manifest.json')
    source = json.loads(nn_policy.MODEL_PATH.read_text())
    manifest = {'format': 'pong-paired-policies-v5', 'variants': {}}
    for variant in ('raw', 'regularized'):
        entry = nn_policy._v5_default_artifacts(variant)
        actor_path = tmp_path / entry['actor_path']
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(source).encode()
        actor_path.write_bytes(raw)
        entry['actor_sha256'] = _sha(raw)
        entry['role_sha256'] = {
            role: nn_policy._role_digest(source['actors'][role])
            for role in ('human', 'ai')}
        report = tmp_path / entry['training_report_path']
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{}')
        entry['training_report_sha256'] = _sha(report.read_bytes())
        for path in entry['program_paths']:
            program = tmp_path / path
            program.parent.mkdir(parents=True, exist_ok=True)
            program.write_text('{"audit":{"explanation_eligible":false,"program_depth":1,"program_leaf_count":1}}')
            entry['program_sha256'][path] = _sha(program.read_bytes())
        manifest['variants'][variant] = entry
    nn_policy.V5_MANIFEST_PATH.write_text(json.dumps(manifest))
    _write_local_receipt(manifest, tmp_path)
    nn_policy._cached_bound_payload.cache_clear()
    yield manifest, tmp_path
    nn_policy._cached_bound_payload.cache_clear()


def test_v5_state_and_decision_bind_exact_pair(bound_policies, monkeypatch):
    manifest, _ = bound_policies
    monkeypatch.setattr(pong, 'human_advisor', lambda _: (_ for _ in ()).throw(AssertionError('advisor')))
    for variant in ('raw', 'regularized'):
        policy_id = manifest['variants'][variant]['policy_id']
        state = pong.initial_state(880101, 1, policy_id=policy_id)
        decision = pong.decide(state)
        assert decision['policy_id'] == policy_id
        assert decision['model_sha256'] == manifest['variants'][variant]['actor_sha256']
        assert decision['action'] == nn_policy.choose(state, 'ai', policy_id)
        assert decision['human_action'] == nn_policy.choose(state, 'human', policy_id)
        next_state = pong.step(state, 'wait', decision)
        assert next_state['policy_id'] == policy_id
        assert next_state['actor_sha256'] == state['actor_sha256']
        demo = pong.demonstration(policy_id=policy_id)
        for first, second in zip(demo['frames'], demo['frames'][1:]):
            for role in ('human', 'ai'):
                expected = nn_policy.choose(first, role, policy_id)
                assert second[role]['x'] == first[role]['x'] + pong._DELTAS[expected]
        assert all(frame['policy_id'] == policy_id for frame in demo['frames'])
        assert all(frame['actor_sha256'] == state['actor_sha256'] for frame in demo['frames'])


def test_v5_physical_legality_and_submitted_decision(bound_policies):
    state = pong.initial_state(880105, 3, policy_id='pong-regularized-v5')
    state['ai']['x'] = 0
    decision = pong.decide(state)
    assert decision['action'] in pong.legal_actions(state, 'ai')
    altered = deepcopy(decision)
    altered['action'] = next(action for action in pong.legal_actions(state, 'ai')
                             if action != decision['action'])
    with pytest.raises(ValueError, match='does not match'):
        pong.step(state, 'wait', altered)
    state['human']['x'] = 0
    with pytest.raises(ValueError, match='Illegal human action'):
        pong.step(state, 'left', pong.decide(state))


def test_v5_missing_or_changed_assets_and_state_fail_closed(bound_policies):
    manifest, root = bound_policies
    policy_id = 'pong-raw-v5'
    state = pong.initial_state(880102, 2, policy_id=policy_id)
    altered = deepcopy(state)
    altered['role_sha256']['ai'] = '0' * 64
    with pytest.raises(RuntimeError, match='state policy binding'):
        pong.decide(altered)
    actor = root / manifest['variants']['raw']['actor_path']
    actor.write_bytes(actor.read_bytes() + b' ')
    with pytest.raises(RuntimeError, match='actor hash mismatch'):
        pong.decide(state)


def test_v5_acceptance_cannot_rebind_an_actor(bound_policies):
    manifest, root = bound_policies
    entry = manifest['variants']['raw']
    receipt = root / entry['acceptance_path']
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        'artifact_sha256': {'raw': '0' * 64},
        'training_report_sha256': {'raw': entry['training_report_sha256']},
    }))
    with pytest.raises(RuntimeError, match='acceptance and policy manifest disagree'):
        pong.initial_state(880104, 2, policy_id='pong-raw-v5')


def test_v5_artifact_catalog_has_both_variants(bound_policies):
    artifacts = nn_policy.policy_artifacts()
    assert set(artifacts) == {'raw', 'regularized'}
    assert artifacts['raw']['program_paths'] == []
    assert len(artifacts['regularized']['program_paths']) == 2
    assert artifacts['raw']['role_sha256']['human'] == artifacts['regularized']['role_sha256']['human']
    with pytest.raises(RuntimeError, match='explanation audit'):
        nn_policy.load_bound_program('pong-regularized-v5', 'ai')
    state = pong.initial_state(880103, 2, policy_id='pong-regularized-v5')
    assert pong.audited_program_evidence(state) is None


def test_v5_explanation_evidence_requires_local_program_agreement(bound_policies):
    manifest, root = bound_policies
    state = pong.initial_state(880104, 2, policy_id='pong-regularized-v5')
    chosen = nn_policy.choose(state, 'ai')
    probabilities = tuple(float(action == chosen) for action in nn_policy.ACTIONS)
    program = ExecutableProgram(
        action_names=nn_policy.ACTIONS, feature_names=nn_policy.FEATURE_NAMES,
        root=ProgramNode(probabilities=probabilities))
    path = manifest['variants']['regularized']['program_paths'][1]
    content = json.dumps({
        'program': program.to_dict(),
        'audit': {'explanation_eligible': True, 'program_depth': 1,
                  'program_leaf_count': 1},
        'policy_id': 'pong-regularized-v5', 'role': 'ai',
        'selected_actor_artifact_sha256': manifest['variants']['regularized']['actor_sha256'],
        'selected_role_sha256': manifest['variants']['regularized']['role_sha256']['ai'],
        'selected_actor_development_audit': {'passed': True, 'purpose': 'explanation',
                                             'sample_count': 1000,
                                             'action_fidelity': .95,
                                             'mean_kl_divergence': .10},
    }).encode()
    (root / path).write_bytes(content)
    manifest['variants']['regularized']['program_sha256'][path] = _sha(content)
    nn_policy.V5_MANIFEST_PATH.write_text(json.dumps(manifest))
    _write_local_receipt(manifest, root)
    evidence = pong.audited_program_evidence(state)
    assert evidence['eligible'] is True
    assert evidence['program_sha256'] == _sha(content)
    assert '95.0%' in evidence['en']


def test_human_feedback_program_keeps_its_own_audit_gate(bound_policies):
    manifest, root = bound_policies
    entry = manifest['variants']['regularized']
    path = entry['program_paths'][0]
    evidence = {
        'policy_id': 'pong-regularized-v5', 'role': 'human',
        'selected_actor_artifact_sha256': entry['actor_sha256'],
        'selected_role_sha256': entry['role_sha256']['human'],
        'audit': {'program_depth': 5, 'program_leaf_count': 16},
        'selected_actor_development_audit': {
            'purpose': 'training_feedback', 'passed': True,
            'sample_count': 1920, 'action_fidelity': .849,
            'mean_kl_divergence': .370},
    }
    content = json.dumps(evidence).encode()
    (root / path).write_bytes(content)
    entry['program_sha256'][path] = _sha(content)
    nn_policy.V5_MANIFEST_PATH.write_text(json.dumps(manifest))
    assert nn_policy.load_bound_program('pong-regularized-v5', 'human') == evidence
    with pytest.raises(RuntimeError, match='explanation audit'):
        nn_policy.load_bound_program('pong-regularized-v5', 'ai')


def test_heldout_start_receipt_is_exclusive_before_any_rollout(tmp_path):
    acceptance = tmp_path / 'rolling_actor_v5.acceptance.json'
    reports = {251016: {
        'raw': {'checkpoint_sha256': 'a' * 64},
        'regularized': {'checkpoint_sha256': 'b' * 64},
    }}
    receipt = reserve_heldout_once(acceptance, (251016,), 251016, reports)
    saved = receipt.read_bytes()
    content = json.loads(saved)
    assert content['status'] == 'started'
    assert content['selected_pair_seed'] == 251016
    assert content['heldout_seeds'] == list(HELDOUT_SEEDS)
    assert content['checkpoint_sha256'] == {'raw': 'a' * 64, 'regularized': 'b' * 64}
    with pytest.raises(RuntimeError, match='already started'):
        reserve_heldout_once(acceptance, (251016,), 251016, reports)
    assert receipt.read_bytes() == saved
