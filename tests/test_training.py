from copy import deepcopy
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from core import RCPD, ExecutableProgram
from training.networks import ActorCritic
from training.rollout import Collector, advantages
from training.ppo import epoch_minibatches, update
from training.feedback import Feedback
from training.checkpoint import save, load
from tests.tiny_env import TinyEnv


def setup(external=False,snapshots=True):
    torch.manual_seed(4)
    cfg=yaml.safe_load(open('configs/default.yaml'))
    env=TinyEnv(external=external,snapshots=snapshots)
    model=ActorCritic(env.spec,16)
    a=torch.optim.Adam(model.actor.parameters(),lr=.001);c=torch.optim.Adam(model.critic.parameters(),lr=.001)
    coll=Collector(env,model,'cpu',4)
    f=Feedback(cfg['feedback'],env.spec.action_names)
    rng=np.random.default_rng(4)
    return cfg,model,a,c,coll,f,rng


def test_permutation_and_truncation():
    batches=epoch_minibatches(37,8,np.random.default_rng(1))
    assert sorted(np.concatenate(batches))==list(range(37))
    r=np.array([[1.],[2.]],dtype=np.float32);v=np.zeros_like(r);n=np.full_like(r,10)
    term=np.array([[False],[True]]);trunc=np.array([[True],[False]])
    adv,_=advantages(r,v,n,term,trunc,.9,.95)
    np.testing.assert_allclose(adv[:,0],[10,2]) # truncated bootstrap; no cross-episode carry

@pytest.mark.parametrize('external',[False,True])
def test_actor_gradient_filter(external):
    cfg,m,a,c,coll,f,rng=setup(external)
    batch=coll.collect(10)
    before=deepcopy(m.state_dict())
    stats=update(m,a,c,batch,cfg['ppo'],rng)
    changed=any(not torch.equal(before[k],m.state_dict()[k]) for k in before if k.startswith('actor.'))
    assert changed != external
    assert stats['actor_updates']==0 if external else stats['actor_updates']>0
    assert any(not torch.equal(before[k],m.state_dict()[k]) for k in before if k.startswith('critic.'))


def test_resume_exact(tmp_path):
    cfg,m,a,c,coll,f,rng=setup()
    update(m,a,c,coll.collect(12),cfg['ppo'],rng)
    counters=dict(joint_steps=12,actor_updates=4,critic_updates=4)
    save(tmp_path/'last.pt',m,a,c,coll,f,rng,counters,cfg)
    expected=coll.collect(8)
    _,m2,a2,c2,col2,f2,rng2=setup()
    assert load(tmp_path/'last.pt',m2,a2,c2,col2,f2,rng2,cfg)==counters
    actual=col2.collect(8)
    for k in expected:
        if k!='features': np.testing.assert_array_equal(actual[k],expected[k])
    assert actual['features']==expected['features']
    update(m,a,c,expected,cfg['ppo'],rng)
    update(m2,a2,c2,actual,cfg['ppo'],rng2)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)


def test_no_snapshot_and_signature(tmp_path):
    cfg,m,a,c,col,f,rng=setup(snapshots=False)
    save(tmp_path/'last.pt',m,a,c,col,f,rng,dict(joint_steps=8),cfg)
    with pytest.raises(ValueError,match='snapshot'):load(tmp_path/'last.pt',m,a,c,col,f,rng,cfg)
    assert load(tmp_path/'last.pt',m,a,c,col,f,rng,cfg,initialize=True)['joint_steps']==0
    from dataclasses import replace
    col.env.spec=replace(col.env.spec,action_names=('choice_b','choice_a'))
    with pytest.raises(ValueError,match='signature'):load(tmp_path/'last.pt',m,a,c,col,f,rng,cfg,initialize=True)


def test_feedback_labels_freeze_and_roundtrip(tmp_path):
    cfg,m,a,c,col,f,rng=setup()
    f=Feedback(dict(enabled=True,warmup_steps=0,interval=100,minimum_partition_samples=8,
                    extraction=dict(max_depth=3,max_leaf_nodes=8,min_samples_leaf=4,regularization_lambda=.01)),col.env.spec.action_names)
    batch=col.collect(160)
    seen=[]
    original=f.engine.fit
    def audited(train,oracle,encoder,**kwargs):
        valid=kwargs['validation_states']
        train_keys={json.dumps(encoder(i),sort_keys=True) for i in train}
        valid_keys={json.dumps(encoder(i),sort_keys=True) for i in valid}
        assert not train_keys & valid_keys
        for i in train[:10]:
            with torch.no_grad(): expected=m.probabilities(torch.from_numpy(batch['obs'][i:i+1]),torch.from_numpy(batch['roles'][i:i+1]))[0].numpy()
            np.testing.assert_allclose(list(oracle(i).values()),expected,rtol=1e-5)
        seen.append(True)
        return original(train,oracle,encoder,**kwargs)
    f.engine.fit=audited
    report=f.maybe_extract(m,batch,0)
    assert seen and f.weight>0
    digest=f.program_hash
    loss=f.loss(m.logits(torch.from_numpy(batch['obs']),torch.from_numpy(batch['roles'])),batch['features'])
    a.zero_grad();loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.actor.parameters())>0
    a.step()
    assert f.maybe_extract(m,batch,1) is None and f.program_hash==digest
    p=tmp_path/'program.json';f.engine.program.save_json(p)
    loaded=ExecutableProgram.load_json(p)
    for row in batch['features']:
        assert loaded.execute(row)==f.engine.program.execute(row)
    f2=Feedback(f.config,f.actions);f2.load_state_dict(f.state_dict())
    assert f2.program_hash==digest and f2.weight==f.weight


def test_disabled_feedback_equivalent():
    cfg,m,a,c,col,f,rng=setup()
    b=col.collect(10)
    _,m2,a2,c2,_,_,rng2=setup();m2.load_state_dict(m.state_dict())
    update(m,a,c,b,cfg['ppo'],rng,None)
    update(m2,a2,c2,b,cfg['ppo'],rng2,f)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)


def test_finite_kl_and_detached_program():
    logits=torch.tensor([[2.,-1.]],requires_grad=True)
    target=torch.tensor([[0.,1.]],requires_grad=True)
    loss=RCPD.regularization_loss(logits,target)
    assert torch.isfinite(loss)
    loss.backward();assert logits.grad is not None and target.grad is None


def test_quality_failure_disables_feedback():
    cfg,m,a,c,col,f,rng=setup()
    f.config.update(warmup_steps=0,minimum_partition_samples=10000)
    f.weight=.01
    assert f.maybe_extract(m,col.collect(10),10000)['reason']=='insufficient_disjoint_states'
    assert f.weight==0


def test_cli(tmp_path):
    cfg=yaml.safe_load(open('configs/default.yaml'))
    cfg.update(max_steps=48,rollout_steps=48,hidden=16)
    cfg['feedback'].update(warmup_steps=0,minimum_partition_samples=4)
    cfg['feedback']['extraction'].update(minimum_overall_fidelity_for_feedback=0.,max_depth=2,min_samples_leaf=2)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    output=tmp_path/'run'
    base=[sys.executable,'-m','training.runner']
    def run(args,ok=True):
        result=subprocess.run(base+args,capture_output=True,text=True)
        assert (result.returncode==0)==ok,result.stderr
        return result
    assert 'requires --factory' in run(['train'],False).stderr
    command=['train','--factory','tests.tiny_env:make_env','--config',str(config),'--output',str(output),'--device','cpu']
    run(command)
    run(command+['--resume',str(output/'last.pt'),'--max-steps','56'])
    run(command,False)
    run(['export','--checkpoint',str(output/'last.pt'),'--output',str(tmp_path/'program.json')])
    features=tmp_path/'features.json';features.write_text('{"signal":0.1,"context":0.2}')
    run(['explain','--program',str(tmp_path/'program.json'),'--features',str(features)])
    run(command+['--initialize',str(output/'last.pt'),'--output',str(tmp_path/'new'),'--max-steps','2'])


def test_single_agent_and_neural_sample_count():
    torch.manual_seed(1)
    for count in [1,3]:
        env=TinyEnv(agents=count)
        model=ActorCritic(env.spec,16)
        collector=Collector(env,model,'cpu',1)
        data=collector.collect(5)
        assert collector.samples==5*count
        assert data['mask'].all() and len(data['actions'])==5*count


def test_actual_quality_gate_and_complexity():
    cfg,m,a,c,col,f,rng=setup()
    f=Feedback(dict(enabled=True,warmup_steps=0,interval=1,minimum_partition_samples=4,
                    extraction=dict(max_depth=1,max_leaf_nodes=2,min_samples_leaf=4,
                                    maximum_mean_kl_for_feedback=0.,regularization_lambda=.01)),col.env.spec.action_names)
    report=f.maybe_extract(m,col.collect(128),0)
    assert 'metrics' in report and report['metrics']['mean_kl_divergence']>0
    assert f.weight==0 and report['complexity']['depth']<=1
    assert report['complexity']['leaf_nodes']<=2


def test_no_application_dependencies():
    import ast
    from pathlib import Path
    for root in ['core','training','examples']:
        for path in Path(root).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                modules=([a.name for a in node.names] if isinstance(node,ast.Import)
                         else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                assert not any(m.split('.')[0] in ['env','backend','ui'] for m in modules)
