"""Run with python -m training.runner; environment factory is mandatory for training."""
import argparse
import importlib
import json
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import yaml
from core import ExecutableProgram
from .networks import ActorCritic, select_device
from .rollout import Collector
from .ppo import update
from .feedback import Feedback
from .checkpoint import save, load


def log(event, **values):
    print(json.dumps(dict(event=event,**values),ensure_ascii=False),flush=True)


def validate_config(c):
    for key in ['max_steps','rollout_steps','checkpoint_interval','hidden']:
        if c[key] <= 0: raise ValueError(f'{key} must be positive')
    for key in ['epochs','minibatch_size','max_grad_norm']:
        if c['ppo'][key] <= 0: raise ValueError(f'{key} must be positive')
    if c['feedback']['interval'] <= 0 or c['feedback']['minimum_partition_samples'] < 2:
        raise ValueError('Invalid feedback interval/sample count')
    extraction = c['feedback']['extraction']
    if not 0 <= extraction['regularization_lambda'] <= extraction['maximum_regularization_lambda'] <= .1:
        raise ValueError('Require 0 <= feedback weight <= ceiling <= 0.1')


def train(args):
    log('loading_config')
    config = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps is not None: config['max_steps']=args.max_steps
    if args.max_minutes is not None: config['max_minutes']=args.max_minutes
    if args.seed is not None: config['seed']=args.seed
    config['factory']=args.factory
    validate_config(config)
    output=Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('New experiments require an empty output directory; use --resume for exact continuation')
    output.mkdir(parents=True,exist_ok=True)
    device=select_device(args.device)
    config['device']=str(device)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    module, name=args.factory.split(':',1)
    env=getattr(importlib.import_module(module),name)()
    env.spec.signature()
    model=ActorCritic(env.spec,config['hidden']).to(device)
    actor_opt=torch.optim.Adam(model.actor.parameters(),lr=config['actor_lr'])
    critic_opt=torch.optim.Adam(model.critic.parameters(),lr=config['critic_lr'])
    collector=Collector(env,model,device,config['seed'])
    feedback=Feedback(config['feedback'],env.spec.action_names)
    rng=np.random.default_rng(config['seed'])
    counters=dict(joint_steps=0,actor_updates=0,critic_updates=0)
    if args.resume or args.initialize:
        counters=load(args.resume or args.initialize,model,actor_opt,critic_opt,collector,feedback,rng,config,initialize=bool(args.initialize))
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'signature.json').write_text(json.dumps(env.spec.signature(),indent=2))
    stopped=[False]
    def stop(signum,frame):
        stopped[0]=True
        log('stop_requested',message='Will save after this rollout/update')
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    started=time.monotonic();last_saved=counters['joint_steps']
    log('training_ready',device=str(device),**counters)
    try:
        while counters['joint_steps'] < config['max_steps'] and not stopped[0]:
            if config['max_minutes']>0 and time.monotonic()-started >= config['max_minutes']*60: break
            steps=min(config['rollout_steps'],config['max_steps']-counters['joint_steps'])
            batch=collector.collect(steps,config['ppo']['gamma'],config['ppo']['gae_lambda'])
            metrics=update(model,actor_opt,critic_opt,batch,config['ppo'],rng,feedback)
            counters['joint_steps']+=steps
            for key in ['actor_updates','critic_updates']: counters[key]+=metrics[key]
            report=feedback.maybe_extract(model,batch,counters['joint_steps'])
            if report is not None:
                log('extraction',**report)
                (output/'extraction.json').write_text(json.dumps(report,indent=2))
                if feedback.engine.program: feedback.engine.program.save_json(output/'program.json')
            log('ppo_update',**counters,neural_samples=collector.samples,feedback_updates=feedback.updates,**{k:v for k,v in metrics.items() if k not in counters})
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(**counters,metrics=metrics,feedback_weight=feedback.weight))+'\n')
            if counters['joint_steps']-last_saved >= config['checkpoint_interval']:
                save(output/'last.pt',model,actor_opt,critic_opt,collector,feedback,rng,counters,config)
                last_saved=counters['joint_steps']
        save(output/'last.pt',model,actor_opt,critic_opt,collector,feedback,rng,counters,config)
        log('saved',joint_steps=counters['joint_steps'],restorable=callable(getattr(env,'snapshot',None)))
    finally:
        for s,handler in handlers.items(): signal.signal(s,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Generic PPO + extracted program feedback; supply your own environment')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train')
    p.add_argument('--factory',help='Your importable module:function returning an environment')
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--output',default='output/run')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--seed',type=int);p.add_argument('--max-steps',type=int);p.add_argument('--max-minutes',type=float)
    group=p.add_mutually_exclusive_group();group.add_argument('--resume');group.add_argument('--initialize')
    p=sub.add_parser('export');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('explain');p.add_argument('--program',required=True);p.add_argument('--features',required=True,help='JSON file with named numeric features')
    args=parser.parse_args(argv)
    if args.command=='train':
        if not args.factory: parser.error('Training requires --factory your_module:make_env. Implement examples/adapter_template.py first; no environment is bundled.')
        train(args)
    elif args.command=='export':
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        program=state['feedback']['engine']['program']
        if not program: parser.error('No program extracted yet; checkpoint cannot export a program')
        ExecutableProgram.from_dict(program).save_json(args.output)
        log('program_exported',output=args.output,quality=state['feedback']['report'])
    else:
        program=ExecutableProgram.load_json(args.program)
        features=json.loads(Path(args.features).read_text())
        missing=set(program.feature_names)-set(features)
        if missing: parser.error(f'Missing features: {sorted(missing)}')
        print(json.dumps(asdict(program.execute(features)),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
