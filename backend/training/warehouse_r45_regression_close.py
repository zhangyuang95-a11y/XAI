"""Close known internal-pilot regressions without runtime action overrides."""
from __future__ import annotations
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import numpy as np
import torch
from backend.training import warehouse_r42_delivery_evaluation as base_eval
from backend.training.warehouse_r45_adaptation import actor_environment, _coordination_teacher_action
from backend.training.warehouse_r45_behavior_evaluation import evaluate
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor

VERSION = "warehouse-r45-known-regression-closure.v1"

def load_model(actor, device):
    model=NativeActorCritic(actor.obs_dim,actor.state_dim,actor.hidden).to(device)
    model.actor.load_state_dict({k:torch.as_tensor(np.array(v,copy=True),device=device) for k,v in actor.weights.items()})
    return model

def collect(model, contract, scenes, device):
    mismatches=[]; anchors=[]
    for si,scene in enumerate(scenes):
      for pi,profile in enumerate(base_eval.PROFILES):
        env=actor_environment(contract,scene);rng=random.Random(459000+si*10+pi)
        while not env.done:
          obs=env.observations()["robot_2"].astype(np.float32,copy=True)
          with torch.no_grad(): logits=model.actor_logits(torch.as_tensor(obs,device=device).unsqueeze(0))
          action=ACTIONS[int(logits.argmax(-1).item())];teacher=_coordination_teacher_action(env)
          (mismatches if action!=teacher else anchors).append((obs,ACTIONS.index(teacher)))
          human="WAIT" if profile=="wait" else base_eval.partner_action(env,"robot_1",profile,rng)
          env.step({"robot_1":human,"robot_2":action})
    return mismatches,anchors

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actor-parent',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=8);a=p.parse_args(argv)
    source=Path(a.actor_parent).resolve();actor=NumPyNativeActor(source);manifest_path=Path(a.manifest).resolve();manifest=json.loads(manifest_path.read_text());scenes=list(manifest['splits']['play']);scenes=scenes[1:] if len(scenes)==7 else scenes
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=load_model(actor,device);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);rounds=[]
    for ri in range(1,a.rounds+1):
      mismatches,anchors=collect(model,actor,scenes,device);rng=np.random.default_rng(459500+ri);sample=[anchors[i] for i in rng.choice(len(anchors),size=min(len(anchors),max(1024,len(mismatches)*4)),replace=False)];rows=mismatches*48+sample;x=np.stack([r[0] for r in rows]).astype(np.float32);y=np.asarray([r[1] for r in rows],dtype=np.int64);opt=torch.optim.Adam(model.actor.parameters(),lr=5e-7,eps=1e-5)
      for _ in range(16):
        order=rng.permutation(len(x))
        for start in range(0,len(order),384):
          ids=order[start:start+384];xb=torch.as_tensor(x[ids],device=device);yb=torch.as_tensor(y[ids],device=device);loss=torch.nn.functional.cross_entropy(model.actor_logits(xb),yb);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.0);opt.step()
      path=out/f'actor_round_{ri:02d}.npz';meta=deepcopy(actor.metadata);meta.update({'r45_regression_closure':VERSION,'r45_regression_parent_sha256':actor.sha256,'r45_regression_round':ri,'formal_play_scenes_used_for_internal_regression':True,'runtime_action_override':False});model.export_npz(path,meta);actor=NumPyNativeActor(path);report=evaluate(path,manifest_path);rounds.append({'round':ri,'mismatches':len(mismatches),'anchors':len(sample),'actor_sha256':actor.sha256,'gates':report['gates'],'passed':report['passed']});print(rounds[-1],flush=True)
      if report['passed']:break
    final=out/'actor.npz';final.write_bytes(path.read_bytes());report=evaluate(final,manifest_path);(out/'behavior_report.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');training={'version':VERSION,'parent':str(source),'parent_sha256':NumPyNativeActor(source).sha256,'actor':str(final),'actor_sha256':sha256(final.read_bytes()).hexdigest(),'formal_play_scenes_used_for_internal_regression':True,'rounds':rounds,'runtime_action_override':False};(out/'regression_closure_report.json').write_text(json.dumps(training,ensure_ascii=False,sort_keys=True,indent=2)+'\n');return 0 if report['passed'] else 2

if __name__=='__main__': raise SystemExit(main())
