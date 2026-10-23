"""Replay the actual PPO action stream from immutable per-rollout Actors.

This verifies execution, complete-episode costs and stored counters; it does
not claim to rerun all optimizer arithmetic or independently estimate quality.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .bc import choose
from .contracts import MethodSpec,digest,file_sha,write_new
from .losses import risk_config
from .model import RoleActor
from .storage import load_frozen,verify_sources


def replay(adapter,folder,plan,*,budget,variant,final_asset,reference_role_sha256,sources,root,output,
           extension_sources=None,require_learning_gates=True):
    from backend.training.warehouse_v6_shutdown_cost import ShutdownRiskDual
    folder=Path(folder);journal=folder/'events.jsonl';journal_sha=file_sha(journal)
    extensions=extension_sources or {}
    if set(extensions)&set(sources):raise ValueError('reader extensions cannot replace frozen training sources')
    current_sources=dict(sources,**extensions)
    verify_sources(root,current_sources)
    if variant not in ('raw','regularized'):raise ValueError('unknown training treatment')
    torch.set_num_threads(1)
    states=[None]*8;pending=[0]*8;initials=[None]*8;reset_pending={};used=set()
    cases={c['case_id']:c for c in plan['cases'] if c['purpose']=='ppo'}
    risk=ShutdownRiskDual(risk_config());completed=[];model=None;rng=torch.Generator()
    count=0;reserved=None;initial_hash=None;bound_hash=None;actor_refs=[];checkpoint_refs=[];reset_counts={}
    role_actions=0;last_updates=None
    ignored={'solver_predebit','current_epoch_program_check'}
    with journal.open() as f:
        for line in f:
            e=json.loads(line);kind=e['event']
            if kind=='rollout_actor_binding':
                if reserved is not None or e['joint_steps']!=count:raise ValueError('training rollout boundary differs')
                asset=e['actor'];frozen=load_frozen(asset['path'],asset['sha256'],adapter.spec,asset['role_sha256'])
                if not Path(asset['path']).resolve().is_relative_to(folder.resolve()):raise ValueError('rollout Actor outside finite owner')
                with torch.random.fork_rng():
                    actors=torch.nn.ModuleDict({r.name:RoleActor(r) for r in adapter.spec.roles})
                for role in adapter.spec.roles:
                    actors[role.name].load_state_dict({k:torch.from_numpy(v.copy()) for k,v in frozen[role.name].arrays.items()})
                model=SimpleNamespace(actors=actors);rng.set_state(torch.tensor(e['action_rng'],dtype=torch.uint8))
                bound_hash=asset['role_sha256'];initial_hash=initial_hash or bound_hash
                if initial_hash!=reference_role_sha256:raise ValueError('training did not start from its own paired BC')
                if count<=MethodSpec().critic_warmup and bound_hash!=initial_hash:
                    raise ValueError('Actor changed during critic warmup')
                actor_refs.append(dict(path=asset['path'],sha256=asset['sha256']))
            elif kind=='reset_reserved':
                case=e['case'];ident=e['reset_identity']
                if cases.get(case['case_id'])!=case or ident in used:raise ValueError('training reset differs or repeats')
                ordinal=reset_counts.get(case['case_id'],0)
                if ident!=case['case_id']+':restart:'+str(ordinal) or ordinal>=case['maximum_resets']:
                    raise ValueError('training restart exceeds finite declaration')
                reset_counts[case['case_id']]=ordinal+1
                used.add(ident);slot=case['slot']
                block=(count//8)//adapter.spec.horizon
                if case['block']!=block or (states[slot] is not None and not adapter.terminal(states[slot])):
                    raise ValueError('unregistered scenario block or premature restart')
                states[slot]=adapter.initial_state(case['seed'],case['task']);pending[slot]=0
                initials[slot]=adapter.snapshot(states[slot]);reset_pending[case['case_id']]=True
            elif kind=='reset_completed':
                if not reset_pending.pop(e['case_id'],False):raise ValueError('reset completion without reservation')
            elif kind=='NN_joint_frame_reserved':
                slot=count%8
                if (reserved is not None or model is None or e['joint_step']!=count or e['slot']!=slot
                        or states[slot] is None or e['actor_role_sha256']!=bound_hash):
                    raise ValueError('training physical frame identity differs')
                case=cases.get(e['case_id'])
                if case is None or case['slot']!=slot or case['block']!=(count//8)//adapter.spec.horizon:
                    raise ValueError('training scenario exposure differs')
                public=adapter.public(states[slot])
                if digest(public)!=e['public_sha256']:raise ValueError('training public observation differs')
                indices,_=choose(model,adapter.observe_public(public),rng)
                if indices!=e['indices']:raise ValueError('training executed action differs from bound NN')
                states[slot],_=adapter.step(states[slot],indices);pending[slot]+=1
                term,trunc=adapter.ending(states[slot])
                cut=((count//8+1)%adapter.spec.horizon==0 and not adapter.terminal(states[slot]))
                reserved=dict(slot=slot,indices=indices,terminated=term,truncated=trunc,training_cut=cut,
                    public_sha256=e['public_sha256'],case_id=e['case_id'],cut_seen=False)
            elif kind=='incomplete_training_block_cut':
                if (reserved is None or not reserved['training_cut'] or e['slot']!=reserved['slot']
                        or e['frames']!=pending[e['slot']] or e['counted_as_complete'] is not False):
                    raise ValueError('training truncation treated as a complete cost episode')
                reserved['cut_seen']=True
            elif kind=='NN_joint_frame':
                if (reserved is None or e['joint_step']!=count or e['actor_role_sha256']!=bound_hash
                        or e['physical_action_overrides']!=0 or any(e[k]!=reserved[k] for k in
                        ('slot','indices','terminated','truncated','training_cut','public_sha256','case_id'))):
                    raise ValueError('training frame completion differs')
                slot=reserved['slot']
                if adapter.terminal(states[slot]):
                    completed.append((True,bool(adapter.outcome(states[slot],adapter.restore(initials[slot]))['failed'])))
                    pending[slot]=0
                elif reserved['training_cut']:
                    if not reserved['cut_seen']:raise ValueError('unrecorded partial episode cut')
                    states[slot]=None;pending[slot]=0
                count+=1;role_actions+=2;reserved=None
            elif kind=='ppo_rollout_completed':
                if (e['variant']!=variant or e['joint_steps']!=count or e['completed_episodes']!=len(completed)):
                    raise ValueError('actual complete cost episodes differ')
                risk.update(completed);completed=[]
                if e['risk']!=risk.snapshot():raise ValueError('risk window or multiplier is not reproducible')
                if variant=='raw' and (any(e['program_updates'].values()) or any(e['program_gradient_norm_sum'].values())):
                    raise ValueError('raw received external program feedback')
                last_updates=e
            elif kind=='checkpoint_quiescent':
                ref=e['checkpoint']
                if (e['joint_steps']!=count or e['plan_sha256']!=digest(plan) or file_sha(ref['path'])!=ref['sha256']
                        or not Path(ref['path']).resolve().is_relative_to(folder.resolve())):
                    raise ValueError('training checkpoint source differs')
                checkpoint_refs.append(ref)
            elif kind not in ignored:raise ValueError('unknown training provenance event: '+str(kind))
    if count!=budget or reserved is not None or reset_pending or completed or last_updates is None:
        raise ValueError('incomplete finite PPO execution')
    load_frozen(final_asset['path'],final_asset['sha256'],adapter.spec,final_asset['role_sha256'])
    changed=all(final_asset['role_sha256'][n]!=initial_hash[n] for n in initial_hash)
    if not changed and require_learning_gates:raise ValueError('Actor did not change')
    final=next((r for r in reversed(checkpoint_refs) if Path(r['path']).name==f'checkpoint-{budget}.pt'),None)
    if final is None:raise ValueError('actual final checkpoint missing')
    cp=torch.load(final['path'],map_location='cpu',weights_only=False)
    if (cp['method_sha256']!=MethodSpec().sha256 or cp['plan_sha256']!=digest(plan)
            or cp['joint_steps']!=budget or cp['model_role_sha256']!=final_asset['role_sha256']
            or cp['reference_role_sha256']!=initial_hash or cp['source_files']!=sources
            or cp['program_updates']!=last_updates['program_updates']
            or cp['program_norm']!=last_updates['program_gradient_norm_sum']):
        raise ValueError('actual final model, counters, source or BC reference differs')
    if variant=='raw':
        if any(cp['program_feedback_frames'].values()) or any(cp['program_updates'].values()) or any(cp['program_norm'].values()):
            raise ValueError('raw external program counters nonzero')
    both_gradient=(all(cp['program_feedback_frames'].values()) and all(cp['program_updates'].values()) and all(cp['program_norm'].values()))
    if variant=='regularized' and not both_gradient and require_learning_gates:
        raise ValueError('regularized lacks both Actors actual program gradient')
    if file_sha(journal)!=journal_sha:raise ValueError('training journal changed during replay')
    verify_sources(root,current_sources)
    result=dict(version='three-domain-unified-actual-training-replay.v1',method_sha256=MethodSpec().sha256,
        plan_sha256=digest(plan),variant=variant,joint_steps=count,replayed_role_actions=role_actions,
        journal=dict(path=str(journal.resolve()),sha256=journal_sha),actors=actor_refs,checkpoints=checkpoint_refs,
        initial_role_sha256=initial_hash,final_asset=final_asset,
        learning_gates=dict(both_Actors_changed=changed,raw_zero_or_regularized_both_actual_gradient=variant=='raw' or bool(both_gradient)),
        actual_program_feedback_frames=cp['program_feedback_frames'],actual_program_updates=cp['program_updates'],
        actual_program_gradient_norm_sum=cp['program_norm'],
        physical_action_overrides=0,optimizer_arithmetic_reexecuted=False,model_acceptance_passed=False,pilot_opened=False)
    write_new(output,result);return result
