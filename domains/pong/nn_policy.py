"""Versioned, dependency-free inference for the rolling Pong actors.

Only currently visible balls and paddle positions enter the policy. The model
file is immutable at runtime; a missing or altered file is a startup error.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).with_name('rolling_actor_v1.json')
ACCEPTANCE_PATH = Path(__file__).with_name('rolling_actor_v1.acceptance.json')
EVALUATION_PATH = Path(__file__).with_name('rolling_actor_v1.evaluation.json')
MODEL_VERSION = 'pong-rolling-actors.v1'
ACTIONS = ('wait', 'left', 'right')
MAX_BALLS = 10
FEATURE_SIZE = 2 + MAX_BALLS * 13
BALL_FIELDS = ('visible', 'cooperative', 'first_contact_lane', 'second_contact_lane',
               'arrival_steps', 'first_own_gap', 'second_own_gap', 'first_partner_gap',
               'second_partner_gap', 'first_own_slack', 'second_own_slack',
               'first_partner_slack', 'second_partner_slack')
FEATURE_NAMES = ('own_lane', 'partner_lane') + tuple(
    f'ball_{slot}_{field}' for slot in range(MAX_BALLS) for field in BALL_FIELDS)
assert len(FEATURE_NAMES) == FEATURE_SIZE
V5_MANIFEST_PATH = Path(__file__).with_name('rolling_actor_v5.manifest.json')
V5_RELEASE_ID = 'policylens-three-domain-20261014.v5.0'
V5_POLICY_IDS = {
    'pong-regularized-v5': 'regularized',
    'pong-raw-v5': 'raw',
}


def _role_digest(layers):
    return hashlib.sha256(json.dumps(layers, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _v5_default_artifacts(variant):
    prefix = f'domains/pong/rolling_actor_v5_{variant}'
    return {'policy_id': f'pong-{variant}-v5', 'actor_path': prefix + '.json',
            'actor_sha256': None, 'role_sha256': {'human': None, 'ai': None},
            'program_paths': ([prefix + '_human.program.json', prefix + '_ai.program.json']
                              if variant == 'regularized' else []),
            'program_sha256': {},
            'training_report_path': prefix + '.training.json',
            'training_report_sha256': None,
            'acceptance_path': 'domains/pong/rolling_actor_v5.acceptance.json'}


def policy_artifacts(release_id=None):
    """Return the pinned v5 artifacts; absent assets remain visibly unready."""
    if release_id is not None:
        from study_v3.release_catalog import RELEASE_ID, policy_artifacts as bundled
        if release_id != RELEASE_ID:
            raise RuntimeError('unknown Pong release catalog')
        return bundled('pong')
    entries = {name: _v5_default_artifacts(name) for name in ('regularized', 'raw')}
    if V5_MANIFEST_PATH.exists():
        manifest = json.loads(V5_MANIFEST_PATH.read_text())
        if manifest.get('format') != 'pong-paired-policies-v5':
            raise RuntimeError('invalid Pong v5 policy manifest')
        for variant, item in manifest.get('variants', {}).items():
            if variant not in entries:
                raise RuntimeError('unknown Pong v5 variant')
            entries[variant].update(item)
    for variant, entry in entries.items():
        if entry['policy_id'] != f'pong-{variant}-v5':
            raise RuntimeError('Pong v5 policy ID mismatch')
    return entries


def _read_pinned(path, digest, label):
    if not digest or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f'Pong v5 {label} is not pinned')
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f'Pong v5 {label} path escapes the release root')
    if not target.is_file():
        raise RuntimeError(f'Pong v5 {label} is missing')
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError(f'Pong v5 {label} hash mismatch')
    return raw


@lru_cache(maxsize=4)
def _cached_bound_payload(policy_id, actor_path, actor_digest, human_digest, ai_digest):
    payload = _validate_payload(json.loads((ROOT / actor_path).read_text()))
    if (_role_digest(payload['actors']['human']) != human_digest
            or _role_digest(payload['actors']['ai']) != ai_digest):
        raise RuntimeError('Pong v5 role weight hash mismatch')
    payload['_loaded_sha256'] = actor_digest
    payload['_policy_id'] = policy_id
    return payload


def load_bound_model(policy_id, release_id=None):
    """Verify every v5 artifact before returning its frozen two-Actor payload."""
    variant = V5_POLICY_IDS.get(policy_id)
    if variant is None:
        raise ValueError('unknown Pong policy_id')
    entry = policy_artifacts(release_id)[variant]
    _read_pinned(entry['actor_path'], entry['actor_sha256'], 'actor')
    _read_pinned(entry['training_report_path'], entry.get('training_report_sha256'), 'training report')
    if variant == 'regularized':
        if len(entry['program_paths']) != 2:
            raise RuntimeError('Pong v5 regularized pair needs two role programs')
        for path in entry['program_paths']:
            _read_pinned(path, entry.get('program_sha256', {}).get(path), 'program')
    acceptance = (ROOT / entry['acceptance_path']).resolve()
    if not acceptance.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Pong v5 acceptance path escapes the release root')
    try:
        receipt = json.loads(acceptance.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError('Pong v5 acceptance receipt is missing or malformed') from exc
    if not isinstance(receipt, dict):
        raise RuntimeError('Pong v5 acceptance receipt is malformed')
    if ((receipt.get('artifact_sha256') or {}).get(variant) != entry['actor_sha256']
            or (receipt.get('training_report_sha256') or {}).get(variant)
            != entry['training_report_sha256']):
        raise RuntimeError('Pong v5 acceptance and policy manifest disagree')
    if variant == 'regularized':
        expected = (receipt.get('program_sha256') or {}).get('regularized') or {}
        for role, path in zip(('human', 'ai'), entry['program_paths']):
            if expected.get(role) != entry['program_sha256'].get(path):
                raise RuntimeError('Pong v5 program acceptance hash mismatch')
    if (receipt.get('release') != V5_RELEASE_ID or receipt.get('passed') is not True
            or receipt.get('heldout_evaluated') is not True):
        raise RuntimeError('Pong v5 acceptance receipt is not approved for execution')
    roles = entry.get('role_sha256', {})
    return _cached_bound_payload(policy_id, entry['actor_path'], entry['actor_sha256'],
                                 roles.get('human'), roles.get('ai'))


def load_bound_program(policy_id, actor='ai', release_id=None):
    """Read a verified role program with the audit required for its use."""
    if policy_id != 'pong-regularized-v5':
        return None
    entry = policy_artifacts(release_id)['regularized']
    paths = entry['program_paths']
    if len(paths) != 2:
        raise RuntimeError('Pong v5 role programs are missing')
    path = paths[0 if actor == 'human' else 1 if actor == 'ai' else -1]
    if actor not in ('human', 'ai'):
        raise ValueError('unknown Pong actor')
    evidence = json.loads(_read_pinned(path, entry['program_sha256'].get(path), 'program'))
    if not isinstance(evidence, dict):
        raise RuntimeError('Pong v5 program evidence is malformed')
    complexity = evidence.get('audit') or {}
    if (not isinstance(complexity, dict)
            or type(complexity.get('program_depth')) is not int
            or type(complexity.get('program_leaf_count')) is not int
            or not 1 <= complexity['program_depth'] <= 5
            or not 1 <= complexity['program_leaf_count'] <= 16):
        raise RuntimeError('Pong v5 program exceeds the bounded protocol')
    audit = evidence.get('selected_actor_development_audit') or {}
    if (not isinstance(audit, dict)
            or type(audit.get('sample_count')) is not int
            or type(audit.get('action_fidelity')) not in (int, float)
            or type(audit.get('mean_kl_divergence')) not in (int, float)
            or not math.isfinite(audit['action_fidelity'])
            or not math.isfinite(audit['mean_kl_divergence'])):
        raise RuntimeError('Pong v5 program explanation audit is malformed')
    audit_purpose = 'explanation' if actor == 'ai' else 'training_feedback'
    minimum_fidelity = .95 if actor == 'ai' else .75
    maximum_kl = .10 if actor == 'ai' else .60
    if (evidence.get('policy_id') != policy_id or evidence.get('role') != actor
            or evidence.get('selected_actor_artifact_sha256') != entry['actor_sha256']
            or evidence.get('selected_role_sha256') != entry['role_sha256'][actor]
            or audit.get('purpose') != audit_purpose or audit.get('passed') is not True
            or audit.get('sample_count', 0) < 1000
            or audit.get('action_fidelity', 0) < minimum_fidelity
            or audit.get('mean_kl_divergence', math.inf) > maximum_kl):
        raise RuntimeError(f'Pong v5 program lacks selected-Actor {audit_purpose} audit')
    return evidence


def features(state, actor):
    if actor not in ('human', 'ai'):
        raise ValueError('unknown Pong actor')
    other = 'ai' if actor == 'human' else 'human'
    values = [state[actor]['x'] / 8.0, state[other]['x'] / 8.0]
    balls = sorted(state['balls'], key=lambda b: (b['remaining'], b['kind'], tuple(b['contacts']), b['id']))
    if len(balls) > MAX_BALLS:
        raise ValueError('too many visible Pong balls for actor')
    for ball in balls:
        contacts = ball['contacts']
        first, second = contacts[0], contacts[-1]
        own, partner = state[actor]['x'], state[other]['x']
        remaining = ball['remaining']
        values.extend((1.0, float(ball['kind'] == 'cooperative'), first / 8.0,
                       second / 8.0, remaining / 12.0,
                       (first - own) / 8.0, (second - own) / 8.0,
                       (first - partner) / 8.0, (second - partner) / 8.0,
                       (remaining - abs(first - own)) / 12.0,
                       (remaining - abs(second - own)) / 12.0,
                       (remaining - abs(first - partner)) / 12.0,
                       (remaining - abs(second - partner)) / 12.0))
    values.extend([0.0] * (FEATURE_SIZE - len(values)))
    return values


def legal_mask(state, actor):
    x = state[actor]['x']
    return (True, x > 0, x < 8)


def semantic_features(state, actor):
    """The program's named, visible, pre-action inputs."""
    values = dict(zip(FEATURE_NAMES, features(state, actor)))
    values['legal_left'] = float(legal_mask(state, actor)[1])
    values['legal_right'] = float(legal_mask(state, actor)[2])
    return values


def _check_layer(layer, inputs):
    if not isinstance(layer, dict) or len(layer.get('weight', ())) != len(layer.get('bias', ())):
        raise RuntimeError('invalid Pong actor layer')
    if not layer['weight'] or any(len(row) != inputs for row in layer['weight']):
        raise RuntimeError('invalid Pong actor dimensions')
    if any(not math.isfinite(float(v)) for row in layer['weight'] for v in row):
        raise RuntimeError('nonfinite Pong actor weight')
    if any(not math.isfinite(float(v)) for v in layer['bias']):
        raise RuntimeError('nonfinite Pong actor bias')
    return len(layer['bias'])


def _validate_payload(payload):
    if (payload.get('version') != MODEL_VERSION or payload.get('feature_size') != FEATURE_SIZE
            or payload.get('actions') != list(ACTIONS)):
        raise RuntimeError('Pong actor version mismatch')
    digest = hashlib.sha256(json.dumps(payload['actors'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != payload.get('actors_sha256'):
        raise RuntimeError('Pong actor integrity check failed')
    for actor in ('human', 'ai'):
        layers = payload['actors'][actor]
        if len(layers) != 3:
            raise RuntimeError('invalid Pong actor architecture')
        width = FEATURE_SIZE
        for layer in layers:
            width = _check_layer(layer, width)
        if width != len(ACTIONS):
            raise RuntimeError('invalid Pong action count')
    return payload


@lru_cache(maxsize=1)
def load_model():
    raw = MODEL_PATH.read_bytes()
    receipt = json.loads(ACCEPTANCE_PATH.read_text())
    if (receipt.get('model_sha256') != hashlib.sha256(raw).hexdigest()
            or not receipt.get('development_passed') or not receipt.get('heldout_passed')):
        raise RuntimeError('Pong frozen actor lacks matching acceptance receipt')
    evidence_raw = EVALUATION_PATH.read_bytes()
    evidence = json.loads(evidence_raw)
    if (receipt.get('evaluation_sha256') != hashlib.sha256(evidence_raw).hexdigest()
            or evidence.get('model_sha256') != receipt['model_sha256']
            or not evidence.get('runtime_parity')
            or not evidence.get('development', {}).get('passed')
            or not evidence.get('heldout', {}).get('passed')):
        raise RuntimeError('Pong frozen actor lacks matching evaluation evidence')
    payload = _validate_payload(json.loads(raw))
    payload['_loaded_sha256'] = hashlib.sha256(raw).hexdigest()
    return payload


def model_sha256(policy_id=None, release_id=None):
    return (load_bound_model(policy_id, release_id) if policy_id else load_model())['_loaded_sha256']


def logits_from_payload(state, actor, payload):
    values = features(state, actor)
    for index, layer in enumerate(payload['actors'][actor]):
        values = [sum(weight * value for weight, value in zip(row, values)) + bias
                  for row, bias in zip(layer['weight'], layer['bias'])]
        if index < 2:
            values = [math.tanh(value) for value in values]
    return values


def logits(state, actor, policy_id=None):
    selected = policy_id or state.get('policy_id')
    return logits_from_payload(state, actor, load_bound_model(selected, state.get('release_id')) if selected else load_model())


def choose_from_payload(state, actor, payload):
    scores = logits_from_payload(state, actor, payload)
    mask = legal_mask(state, actor)
    return ACTIONS[max((i for i in range(len(ACTIONS)) if mask[i]), key=lambda i: (scores[i], -i))]


def choose(state, actor, policy_id=None):
    selected = policy_id or state.get('policy_id')
    return choose_from_payload(state, actor, load_bound_model(selected, state.get('release_id')) if selected else load_model())
