"""Content bindings, exclusive claims and append-before-action experiment journal."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from .contracts import MethodSpec, digest, file_sha, write_new
from .model import FrozenRole


def append(path, event):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    with path.open('a') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def save_checkpoint(path, payload):
    """Atomic replacement; interrupted writes cannot replace a complete snapshot."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, 'rb') as f: os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return dict(path=str(path.resolve()), sha256=file_sha(path))


def export_actor(model, path):
    arrays = {role + '/' + key: v.detach().cpu().numpy()
              for role, actor in model.actors.items() for key, v in actor.state_dict().items()}
    metadata = dict(version='three-domain-unified-actor.v1', domain=asdict(model.domain),
                    domain_sha256=model.domain.sha256, method_sha256=MethodSpec().sha256,
                    role_sha256=model.role_hashes(), runtime_action_overrides=0)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez(f, metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **arrays)
        f.flush(); os.fsync(f.fileno())
    return dict(path=str(path.resolve()), sha256=file_sha(path), role_sha256=model.role_hashes())


def load_frozen(path, expected_sha256, domain, role_sha256):
    if file_sha(path) != expected_sha256: raise ValueError('actor file hash differs')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'].item()))
        if (metadata['version'] != 'three-domain-unified-actor.v1' or metadata['domain_sha256'] != domain.sha256
                or metadata['method_sha256'] != MethodSpec().sha256 or metadata['role_sha256'] != role_sha256
                or metadata['domain'] != json.loads(json.dumps(asdict(domain)))):
            raise ValueError('actor architecture, domain or method binding differs')
        from .model import RoleActor, tensor_hash
        result = {}
        expected_keys = {'metadata_json'}
        for role in domain.roles:
            # Validate all tensor names, shapes, dtypes and numerical values.
            # Shape validation uses a forked RNG so reading a frozen model
            # cannot advance the study/training Torch random stream.
            with torch.random.fork_rng():
                template = RoleActor(role)
            state = {}
            for key, expected in template.state_dict().items():
                name = role.name + '/' + key; expected_keys.add(name)
                value = archive[name]
                if value.shape != tuple(expected.shape) or value.dtype != expected.numpy().dtype or not np.isfinite(value).all():
                    raise ValueError('actor tensor shape, dtype or finite check failed')
                state[key] = torch.from_numpy(value.copy())
            if tensor_hash(state) != role_sha256[role.name]: raise ValueError('independent role tensor hash differs')
            template.load_state_dict(state)
            support = template.tree.support_mask.numpy()
            from .contracts import source_support
            if any(len(source_support(role.features, np.flatnonzero(row).tolist())) > 3 for row in support):
                raise ValueError('Actor sparse split exceeds public source limit')
            result[role.name] = FrozenRole.from_actor(template)
        if set(archive.files) != expected_keys: raise ValueError('unknown actor payload fields')
    return result


class Run:
    """One finite owner; seed audit and plan are checked before every reset."""
    def __init__(self, folder, plan, clearance):
        self.folder = Path(folder); self.plan = plan; self.plan_hash = digest(plan)
        self.clearance = Path(clearance); self.clearance_hash = file_sha(clearance)
        self.used = set(); self.journal = self.folder / 'events.jsonl'
        self._checked_clearance(full=True)
        write_new(self.folder / 'claim.json', dict(plan_sha256=self.plan_hash, clearance_sha256=self.clearance_hash,
                  started_utc=datetime.now(timezone.utc).isoformat(), model_acceptance_passed=False, pilot_opened=False))

    def ensure_open(self):
        if (self.clearance.parent/'CLOSED.json').exists(): raise ValueError('shared candidate has been closed')
        if digest(self.plan) != self.plan_hash: raise ValueError('finite case plan was mutated')

    def _checked_clearance(self, *, full=False):
        self.ensure_open()
        if file_sha(self.clearance) != self.clearance_hash: raise ValueError('seed clearance changed')
        value = json.loads(self.clearance.read_text())
        if (value.get('version') != 'three-domain-unified-seed-clearance.v2'
                or value.get('plan_sha256') != self.plan_hash or value.get('passed') is not True):
            raise ValueError('actual independent metadata clearance required before a reset')
        if file_sha(self.clearance.parent/'seed-proposal.json') != value['source_proposal_sha256']:
            raise ValueError('audited seed proposal changed')
        if full:
            for ref in value.get('evidence', []):
                if file_sha(ref['path']) != ref['sha256']: raise ValueError('seed audit evidence changed')
        if len(value.get('evidence', [])) < 3: raise ValueError('missing inventory, triage or final metadata delta')
        delta_ref = value['final_delta']
        if file_sha(delta_ref['path']) != delta_ref['sha256']: raise ValueError('final seed delta changed')
        delta = json.loads(Path(delta_ref['path']).read_text())
        if (delta['version'] != 'three-domain-unified-final-seed-delta.v1' or delta['passed'] is not True
                or not delta['selection_unchanged'] or not delta['opaque_exclusions_unchanged']
                or delta['changed'] or delta['missing']):
            raise ValueError('incomplete or failed final metadata delta')
        ref = value['registration']
        if file_sha(ref['path']) != ref['sha256']: raise ValueError('final clearance registration changed')
        return value

    def finish(self):
        value = self._checked_clearance(full=True)
        write_new(self.folder/'verification-bracket.json', dict(version='three-domain-unified-static-input-bracket.v1',
            plan_sha256=self.plan_hash, clearance_sha256=self.clearance_hash, evidence=value['evidence'],
            full_before_and_after=True, actual_resets=len(self.used), pilot_opened=False))

    def reset(self, adapter, case, *, purpose):
        self._checked_clearance()
        identity = case['case_id']
        if purpose == 'ppo':
            ordinal = sum(key.startswith(identity + ':restart:') for key in self.used)
            if ordinal >= case['maximum_resets']: raise ValueError('training block reset ceiling exhausted')
            identity += ':restart:' + str(ordinal)
        if case not in self.plan['cases'] or case['purpose'] != purpose or identity in self.used:
            raise ValueError('unregistered, repeated or wrong-partition reset')
        append(self.journal, dict(event='reset_reserved', case=case, reset_identity=identity))
        self.used.add(identity)
        state = adapter.initial_state(case['seed'], case['task'])
        append(self.journal, dict(event='reset_completed', case_id=case['case_id']))
        return state


def source_files(root):
    root = Path(root)
    paths = [*root.joinpath('training_v7').glob('*.py'), *root.joinpath('scripts').glob('*.py')]
    # Include the actual transitive domain physics and teacher dependencies.
    for directory in ('env/warehouse', 'domains/pong', 'domains/kitchen', 'backend/training', 'core', 'study_v3', 'experiments/warehouse'):
        paths.extend(root.joinpath(directory).rglob('*.py'))
    return {str(p.relative_to(root)): file_sha(p) for p in sorted(set(paths)) if p.is_file()}


def verify_sources(root, expected):
    if source_files(root) != expected: raise ValueError('registered scientific source closure changed')
