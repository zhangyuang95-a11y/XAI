"""Synthetic extraction plus ONE feedback gradient update, not RL training."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from core.interfaces import EnvSpec
from training.networks import ActorCritic
from training.feedback import Feedback


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='output/demo');args=parser.parse_args(argv)
    torch.manual_seed(7)
    spec=EnvSpec(2,2,('choice_a','choice_b'),('signal','context'),('signal','context'),('signal','context'))
    model=ActorCritic(spec,hidden=16)
    x=np.random.default_rng(7).normal(size=(512,2)).astype(np.float32)
    batch=dict(obs=x,roles=np.zeros(len(x),dtype=np.int64),features=[dict(zip(spec.feature_names,map(float,row))) for row in x])
    feedback=Feedback(dict(enabled=True,warmup_steps=0,interval=10,minimum_partition_samples=16,
                          extraction=dict(max_depth=3,max_leaf_nodes=8,min_samples_leaf=8,regularization_lambda=.01,
                                          maximum_regularization_lambda=.05,minimum_overall_fidelity_for_feedback=.8,
                                          maximum_mean_kl_for_feedback=.1)),spec.action_names)
    report=feedback.maybe_extract(model,batch,0)
    if not feedback.weight: raise RuntimeError(f'Demo extraction gate failed: {report}')
    before=torch.cat([p.detach().flatten() for p in model.actor.parameters()]).clone()
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=.001)
    loss=feedback.loss(model.logits(torch.from_numpy(x),torch.from_numpy(batch['roles'])),batch['features'])
    optimizer.zero_grad();loss.backward();optimizer.step()
    change=float((torch.cat([p.detach().flatten() for p in model.actor.parameters()])-before).norm())
    if not change>0: raise RuntimeError('No Actor parameter change')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    program=feedback.engine.program;program.save_json(output/'program.json')
    (output/'features.json').write_text(json.dumps(batch['features'][0],indent=2))
    trace=asdict(program.execute(batch['features'][0]))
    (output/'trace.json').write_text(json.dumps(trace,indent=2))
    (output/'metrics.json').write_text(json.dumps(dict(report=report,feedback_loss=float(loss.detach()),parameter_change=change),indent=2))
    print('接口演示：合成数据 → 当前 NN 标签 → 有界程序 → 一次 KL 更新。不是强化学习结果。')
    print(json.dumps(dict(fidelity=report['metrics'].get('action_fidelity'),loss=float(loss.detach()),parameter_change=change),ensure_ascii=False))
    for step in program.trace(batch['features'][0]):
        print(f'{step.feature} {step.operator} {step.threshold:.4f}; observed={step.observed_value:.4f}, result={step.result}')
    return 0

if __name__=='__main__': raise SystemExit(main())
