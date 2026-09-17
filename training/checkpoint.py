"""Atomic trusted-local checkpoints. Never load untrusted pickle checkpoints."""
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    if state['mps'] is not None and torch.backends.mps.is_available(): torch.mps.set_rng_state(state['mps'].cpu())


def atomic_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent,prefix='.checkpoint-')
    try:
        with os.fdopen(fd,'wb') as handle:
            torch.save(payload,handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def save(path, model, actor_opt, critic_opt, collector, feedback, rng, counters, config):
    env = collector.env
    restorable = callable(getattr(env,'snapshot',None)) and callable(getattr(env,'restore',None))
    atomic_save(dict(version='core-ppo-1',signature=env.spec.signature(),hidden=model.hidden,
                     model=model.state_dict(),actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),
                     config=config,rng=rng_state(),minibatch_rng=rng.bit_generator.state,counters=counters,
                     feedback=feedback.state_dict(),restorable=restorable,
                     environment=env.snapshot() if restorable else None,frame=collector.frame,
                     neural_samples=collector.samples),path)


def load(path,model,actor_opt,critic_opt,collector,feedback,rng,config,initialize=False):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state['version'] != 'core-ppo-1' or state['signature'] != collector.env.spec.signature() or state['hidden'] != model.hidden:
        raise ValueError('Network, observation, feature or action signature mismatch')
    if not initialize:
        comparable=lambda c:{k:v for k,v in c.items() if k not in ('max_steps','max_minutes','output')}
        if comparable(config) != comparable(state['config']): raise ValueError('Resume configuration mismatch; use explicit --initialize')
        if not state['restorable'] or not callable(getattr(collector.env,'restore',None)):
            raise ValueError('Environment has no snapshot/restore; use --initialize in a new output directory')
    model.load_state_dict(state['model'])
    if initialize: return dict(joint_steps=0,actor_updates=0,critic_updates=0)
    actor_opt.load_state_dict(state['actor_optimizer']);critic_opt.load_state_dict(state['critic_optimizer'])
    collector.env.restore(state['environment']);collector.frame=state['frame'];collector.samples=state['neural_samples']
    feedback.load_state_dict(state['feedback']);rng.bit_generator.state=state['minibatch_rng'];restore_rng(state['rng'])
    return state['counters']
