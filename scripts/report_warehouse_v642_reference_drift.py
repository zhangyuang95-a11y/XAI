#!/usr/bin/env python3
"""Read-only reference drift on the original common-fit training buffer.

No environment, optimizer, stochastic draw or held-out data is used. This
descriptive training diagnostic cannot pass any program or quality gate.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse.v626_protocol import require,pointer,bound,file_sha
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v625_actor_training import ReferenceTrainingBuffer
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-reference-training-buffer-drift.v1'
BUFFER=ROOT/'output/warehouse_v629/training-preparation/reference-buffer-01/reference-training.npz'
BUFFER_SHA='2d4b2e593ed3d16a60e90564f8ee9c58a8026a722c4786f1e9ccff574b125dfa'


def reference_drift(reference_logits,current_logits,masks):
    old=np.asarray(reference_logits,dtype=np.float64);new=np.asarray(current_logits,dtype=np.float64);m=np.asarray(masks)
    require(old.shape==new.shape==m.shape and old.shape[-1]==5 and np.isfinite(old).all()
        and np.isfinite(new).all() and np.isin(m,(0,1)).all() and (m[...,4]==1).all(),'invalid complete masked NN distribution inputs')
    def logp(x):
        x=np.where(m==1,x,-np.inf);x=x-x.max(axis=-1,keepdims=True)
        return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
    a,b=logp(old),logp(new)
    kl=np.maximum(0.,(np.exp(a)*(np.where(m==1,a,0.)-np.where(m==1,b,0.))).sum(-1))
    return kl,np.argmax(np.where(m==1,old,-np.inf),-1)==np.argmax(np.where(m==1,new,-np.inf),-1)


def run(calibration,output):
    calibration=Path(calibration).resolve();descriptor_ref=pointer(calibration/'descriptor.json');descriptor=bound(descriptor_ref)
    require(descriptor['version']=='warehouse-v642-finite-paired-calibration-owner.v1'
        and descriptor['stage']=='paired_20k_calibration_not_formal_training','explicit actual calibration descriptor required')
    anchor=descriptor['reference_actor'];reference=FrozenWarehouseV6Actor.load(anchor['path'],expected_sha256=anchor['sha256'])
    buffer=ReferenceTrainingBuffer.load(BUFFER,BUFFER_SHA,anchor['sha256'])
    variants=('raw','regularized');available={}
    for variant in variants:
        available[variant]={int(p.parent.name):p for p in (calibration/variant/'rollouts').glob('*/current-NN.npz')
            if p.parent.name.isdigit() and not p.stat().st_mode&0o222}
    common=set(available['raw'])&set(available['regularized']);require(common,'no matching immutable collection Actor exports yet')
    step=max(common);refs={v:pointer(available[v][step]) for v in variants}
    models={v:FrozenWarehouseV6Actor.load(r['path'],expected_sha256=r['sha256']) for v,r in refs.items()}
    measures={v:{role:dict(kl=[],same=[]) for role in ('robot_1','robot_2')} for v in variants}
    n=len(buffer.encoded)
    for start in range(0,n,512):
        stop=min(start+512,n)
        for i,role in enumerate(('robot_1','robot_2')):
            encoded=buffer.encoded[start:stop,i];mask=buffer.masks[start:stop,i]
            old=reference.roles[role].logits_encoded(encoded)
            for variant in variants:
                kl,same=reference_drift(old,models[variant].roles[role].logits_encoded(encoded),mask)
                measures[variant][role]['kl'].append(kl);measures[variant][role]['same'].append(same)
    reports={}
    for variant,roles in measures.items():
        reports[variant]={}
        for role,values in roles.items():
            kl=np.concatenate(values['kl']);same=np.concatenate(values['same'])
            reports[variant][role]=dict(mean_reference_KL_nat=float(kl.mean()),p95_reference_KL_nat=float(np.quantile(kl,.95)),
                maximum_reference_KL_nat=float(kl.max()),reference_argmax_agreement=float(same.mean()),physical_rows=len(kl))
    require(pointer(descriptor_ref['path'])==descriptor_ref and file_sha(BUFFER)==BUFFER_SHA
        and all(pointer(ref['path'])==ref for ref in (anchor,*refs.values())),'training diagnostic inputs changed while reading')
    result=dict(version=VERSION,descriptor=descriptor_ref,reference_actor=anchor,current_actors=refs,
        joint_step_of_collection_actors=step,reference_buffer=pointer(BUFFER),reports=reports,
        scope='original_common_fit_training_buffer_only_not_current_state_mix_or_independent_validation',
        physical_joint_rows=n,source_sha256=file_sha(__file__),new_environment_resets=0,new_stochastic_draws=0,
        optimizer_updates=0,program_fidelity_claimed=False,task_quality_claimed=False,model_acceptance_passed=False,pilot_opened=False)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);write_new(output,result);output.chmod(0o444)
    return pointer(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--calibration',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(run(args.calibration,args.output)),flush=True)
