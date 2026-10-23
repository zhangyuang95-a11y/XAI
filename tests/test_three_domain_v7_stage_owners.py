"""Software fixtures only: existing spent Pong seed, never v7 acceptance."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from training_v7.contracts import MethodSpec,digest,file_sha
from training_v7.domains import domain
from training_v7.model import DualActorCritic
from training_v7.storage import append,export_actor,source_files

torch.set_num_threads(1)


@pytest.mark.parametrize('name',['warehouse','pong','kitchen'])
def test_role_binding_roundtrip_is_independent_of_JSON_dictionary_order(tmp_path,name):
    from training_v7.evaluation_run import binding_for
    d=domain(name);model=DualActorCritic(d.spec);asset=export_actor(model,tmp_path/'fixture.actor.npz')
    original=binding_for(d,'regularized',asset)
    decoded=json.loads(json.dumps(asset,sort_keys=True))
    assert binding_for(d,'regularized',decoded)==original
    assert [name for name,h in original.role_sha256]==[r.name for r in d.spec.roles]


@pytest.mark.parametrize('name',['warehouse','pong','kitchen'])
def test_finite_stage_plans_reuse_BC_but_not_confirmation_scenes(name):
    from training_v7.stage_plans import pair_plans,stage_seed_proposal
    d=domain(name);plans=pair_plans(d.spec,phase='development',pair=0,budgets=MethodSpec().budgets,has_proxy=hasattr(d,'proxy'))
    assert len(plans['bc']['cases'])==1270
    seen=set();first=None
    for budget in MethodSpec().budgets:
        p=plans[str(budget)];rows=p['cases']
        assert tuple(sum(c.get('partition')==part for c in rows) for part in
            ('fit','calibration','eligibility','validation'))==(900,150,300,300)
        quality={c['seed'] for c in rows if c['purpose']=='quality'}
        assert len(quality)==75 and not quality&seen;seen|=quality
        prefix=[(c['seed'],c['task'],c['slot'],c['block']) for c in rows if c['purpose']=='ppo']
        if first is None:first=prefix
        else:assert prefix[:len(first)]==first
        pressure=[c for c in rows if c['purpose']=='pressure' and c['policy']=='raw' and c['pairing']=='nn_nn']
        assert len(pressure)==216 and len(set(c['seed'] for c in pressure))==72
    proposal=stage_seed_proposal(plans)
    from experiments.warehouse.seed_metadata_audit_indexed_v642 import read_proposal
    # Structural interval checks are also applied by the real seed auditor.
    values=[s for f in proposal['families'] if f['kind']!='init' for s in f['values']]
    assert len(values)==len(set(values))


def test_formal_quality_set_identical_across_pairs_checkpoints():
    from training_v7.stage_plans import pair_plans
    d=domain('pong');expected=None
    for pair in range(3):
        p=pair_plans(d.spec,phase='formal',pair=pair,budgets=(200000,),has_proxy=False)['200000']
        for cp in ('84','93','final'):
            cases=[c for c in p['cases'] if c['purpose']=='quality' and c['checkpoint']==cp and c['policy']=='raw']
            current=[(c['seed'],c['task'],c['action_seed']) for c in cases]
            if expected is None:expected=current
            else:assert current==expected


def test_generic_evaluation_replays_actions_and_rejects_forged_quality(tmp_path):
    from training_v7.evaluation_run import execute,verify
    from training_v7.stage_plans import evaluation_cases
    d=domain('pong');model=DualActorCritic(d.spec);asset=export_actor(model,tmp_path/'fixture.actor.npz')
    cases=evaluation_cases('pong','development',0,200000,'quality',880100,3,
        policies=('reference','raw','regularized'),pairings=('nn_nn',))
    # Every case deliberately uses one spent seed. Shortened archived-style
    # states are software probes, not real task-quality observations.
    for c in cases:c['seed']=880100
    class Owner:
        plan=dict(phase='development',cases=cases)
        plan_hash=digest(plan)
        def reset(self,adapter,case,purpose):
            s=adapter.initial_state(880100,case['task']);s['max_turns']=2
            s['balls']=[dict(id='software-fixture',kind='ordinary',remaining=1,contacts=[s['human']['x']],y=1,vy=1)]
            return s
    owner=Owner();refs=execute(d,owner,purpose='quality',assets={p:asset for p in ('reference','raw','regularized')},
        directory=tmp_path/'evaluation')
    result=json.loads(Path(refs['result']['path']).read_text())
    verified=json.loads(Path(refs['verification']['path']).read_text())
    assert verified['replayed_role_actions']==36 and verified['complete_episodes']==9
    assert verified['model_acceptance_passed'] is False
    result['task_metrics']['raw']['score']=1234
    changed=tmp_path/'changed.json';changed.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='summary'):
        verify(d,changed,expected_cases=cases,expected_plan=owner.plan_hash)


def test_calibration_quality_collector_roundtrips_sorted_role_hashes(tmp_path):
    from training_v7.quality_run import collect_quality,audit_quality
    from training_v7.program import Program,certify,FitBudget
    from training_v7.stage_plans import evaluation_cases
    d=domain('pong');model=DualActorCritic(d.spec);asset=export_actor(model,tmp_path/'fixture.actor.npz')
    cases=evaluation_cases('pong','calibration',0,20000,'calibration_quality',880100,3,
        policies=('reference','raw','regularized'),pairings=('nn_nn',))
    for c in cases:c['seed']=880100
    class Owner:
        plan=dict(cases=cases)
        def reset(self,adapter,case,purpose):
            s=adapter.initial_state(880100,case['task']);s['max_turns']=2
            s['balls']=[dict(id='software-fixture',kind='ordinary',remaining=1,contacts=[s['human']['x']],y=1,vy=1)]
            return s
    programs={v:{} for v in ('raw','regularized')}
    for variant in programs:
        for role in d.spec.roles:
            leaf=dict(support=[],bias=[0. if a==role.wait_index else -20. for a in range(len(role.actions))],
                weights=[[] for a in role.actions],witness=role.wait_index)
            programs[variant][role.name]=Program(role,dict(leaf=leaf,proof=certify(role,leaf,[],FitBudget())),asset['role_sha256'][role.name])
    collect_quality(d,Owner(),{p:asset for p in ('reference','raw','regularized')},programs,output=tmp_path/'quality')
    audited=audit_quality(d,tmp_path/'quality/result.json')
    assert audited['replayed_role_actions']==36 and audited['full_scientific_acceptance_passed'] is False


def test_actual_training_action_replay_including_cost_window(tmp_path):
    from training_v7.collector import Collector
    from training_v7.ppo import PPO
    from training_v7.training_audit import replay
    d=domain('pong');s=d.initial_state(880100,1);model=DualActorCritic(d.spec)
    reference=model.role_hashes();obs=d.observe_public(d.public(s))
    reference_rows={r.name:{k:np.repeat(obs[r.name][k][None],8,axis=0) for k in ('x','mask','action_features')} for r in d.spec.roles}
    cases=[dict(case_id=f'software-spent-{block}-{slot}',seed=880100,task=1,purpose='ppo',block=block,slot=slot,
                maximum_resets=d.spec.horizon) for block in range(32) for slot in range(8)]
    class Owner:
        folder=tmp_path;journal=tmp_path/'events.jsonl';used=set();plan=dict(cases=cases)
        def reset(self,adapter,case,purpose):
            ordinal=sum(x.startswith(case['case_id']+':restart:') for x in self.used)
            ident=case['case_id']+':restart:'+str(ordinal);self.used.add(ident)
            append(self.journal,dict(event='reset_reserved',case=case,reset_identity=ident))
            state=adapter.initial_state(case['seed'],case['task'])
            append(self.journal,dict(event='reset_completed',case_id=case['case_id']));return state
    owner=Owner();engine=PPO(model,optimizer_seed=17,reference_rows=reference_rows)
    collector=Collector(d,model,owner,action_seed=19,budget=20000)
    while collector.joint_steps<5024:
        asset=export_actor(model,tmp_path/f'rollout-{collector.joint_steps}.npz')
        append(owner.journal,dict(event='rollout_actor_binding',joint_steps=collector.joint_steps,actor=asset,
            action_rng=collector.generator.get_state().tolist()))
        rows,episodes=collector.collect();engine.joint_steps=collector.joint_steps
        engine.update(rows,episodes,variant='raw',budget=20000,program_provider=lambda *a:pytest.fail('raw feedback'))
        append(owner.journal,dict(event='ppo_rollout_completed',variant='raw',joint_steps=engine.joint_steps,
            completed_episodes=len(episodes),risk=engine.risk.snapshot(),program_updates=engine.program_updates,
            program_gradient_norm_sum=engine.program_norm))
    root=Path(__file__).resolve().parents[1];sources=source_files(root)
    final=export_actor(model,tmp_path/'final.actor.npz')
    cp=engine.checkpoint(tmp_path/f'checkpoint-{collector.joint_steps}.pt',collector_state=collector.snapshot(),
        programs={},plan_sha256=digest(owner.plan),sources=sources)
    append(owner.journal,dict(event='checkpoint_quiescent',checkpoint=cp,plan_sha256=digest(owner.plan),joint_steps=collector.joint_steps))
    result=replay(d,tmp_path,owner.plan,budget=collector.joint_steps,variant='raw',final_asset=final,
        reference_role_sha256=reference,sources=sources,root=root,output=tmp_path/'replay.json')
    assert result['joint_steps']==5024 and result['replayed_role_actions']==10048
    assert result['optimizer_arithmetic_reexecuted'] is False and result['model_acceptance_passed'] is False
    events=[json.loads(l) for l in owner.journal.read_text().splitlines()]
    frame=next(e for e in events if e['event']=='NN_joint_frame_reserved')
    frame['indices']['ai']=(frame['indices']['ai']+1)%3
    owner.journal.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError,match='differs from bound NN'):
        replay(d,tmp_path,owner.plan,budget=collector.joint_steps,variant='raw',final_asset=final,
            reference_role_sha256=reference,sources=sources,root=root,output=tmp_path/'tamper.json')
