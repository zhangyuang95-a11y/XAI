"""Bounded 20k-per-arm lower-rate feedback experiment, with durable receipts."""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import time
import numpy as np

from . import warehouse_family_stable_trainer as learner
from . import warehouse_family_stable_evaluation as evaluation
from . import warehouse_family_branch_warm_run as source_io
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_native_cycle_budget import CycleBudget
from .warehouse_native import atomic_torch_save
from .warehouse_native_partner_mix_run import write_json, bound, decode, save_batch

VERSION='warehouse-family-stable-feedback-run.v1'
TEACHER=ROOT/'output/warehouse_native/branch_grouped_fit_388m_20260910/fit_result.json'
TEACHER_SHA='606a4d3d39ae47a5a43d4cdb50db2ffad092ed8267af68a6585de9ee6526e591'
BRANCHES=('control','feedback')
BATCH=2000


def sources():
    value={**learner.execution_sources(),**evaluation.execution_sources()}
    for module in (source_io,):value[str(Path(module.__file__).relative_to(ROOT))]=file_hash(module.__file__)
    value[str(Path(__file__).relative_to(ROOT))]=file_hash(__file__)
    return value


def _json(path):return json.loads(Path(path).read_bytes())


def _save(path,value):
    if path.exists():raise ValueError('Immutable checkpoint already exists')
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(path,value)


def _material(inputs):
    source,scenes,parameters=source_io._restore_source(inputs,'mps',False)
    if file_hash(TEACHER)!=TEACHER_SHA:raise ValueError('Frozen grouped teacher differs')
    fit=_json(TEACHER)
    report=_json(source_io._external(inputs['source_report']))
    metadata,_=source_io._actor_equal(source,source_io._external(inputs['source_actor']))
    bindings=report['actor_bindings']
    if (bindings['actor_sha256']!=inputs['source_actor']['sha256'] or
            bindings['actor_parameters_sha256']!=parameters or
            any(digest(v)!=digest(metadata.get(k)) for k,v in bindings.items() if k!='actor_sha256') or
            report['identity']['actor_metadata_sha256']!=digest(metadata)):
        raise ValueError('Initial qualification does not belong to actual source Actor')
    baseline=_json(source_io._external(inputs['baselines']))
    with np.load(source_io._external(inputs['auxiliary']),allow_pickle=False) as arrays:
        if set(arrays.files)!={'observations','weights'} or not np.all(arrays['weights']==1):
            raise ValueError('Expected original unit-weight TRAIN auxiliary pairs')
        obs=arrays['observations'].copy();weights=arrays['weights'].copy()
    aux_binding=_json(source_io._external(inputs['auxiliary_binding']))
    expected=learner.source_io.auxiliary_binding(obs,weights,
        source_actor_sha256=inputs['source_actor']['sha256'],source_actor_parameters_sha256=parameters,
        fit_step=inputs['source_cumulative_step'],evidence_sha256=inputs['auxiliary_origin']['collection_manifest_sha256'])
    if digest(expected)!=digest(aux_binding) or fit['fit_report']['observed197_bindings']['training_data_sha256']!=inputs['auxiliary_origin']['training_data_sha256']:
        raise ValueError('Auxiliary pool is not bound to original TRAIN data')
    return source,scenes,fit,report,baseline,obs


def _trainer(protocol,source,inputs,branch,fit,obs):
    return learner.StableFeedbackTrainer(protocol,source,
        source_checkpoint_sha256=inputs['source_checkpoint']['sha256'],
        expected_source_state_sha256=inputs['source_native_state_sha256'],branch=branch,
        teacher_fit=fit if branch=='feedback' else None,
        auxiliary_observations=obs if branch=='feedback' else None,device='mps')


def prepare(output,source_inputs,source_sha256):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new versioned run directory')
    frozen=sources()
    inputs=source_io._read_inputs(source_inputs,source_sha256,False)
    source,scenes,fit,report,baseline,obs=_material(inputs)
    protocol=learner.make_protocol(source,source_checkpoint_sha256=inputs['source_checkpoint']['sha256'],
        source_state_sha256=inputs['source_native_state_sha256'],source_report=report,
        team_reference=baseline['reference'])
    output.mkdir(parents=True)
    write_json(output/'inputs.json',inputs);write_json(output/'protocol.json',protocol)
    identity=dict(version=VERSION,cycle_id=protocol['cycle_id'],protocol_sha256=digest(protocol),
        runtime_sources=frozen,budget_caps={b:dict(ppo=20000,evaluation=36000) for b in BRANCHES},
        source_inputs=dict(path=str(Path(source_inputs).resolve()),sha256=source_sha256),
        teacher=dict(path=str(TEACHER),sha256=TEACHER_SHA),test_fixture=False)
    ledger=CycleBudget.create(output,identity)
    equal={}
    with ledger.lease():
        for branch in BRANCHES:
            trainer=_trainer(protocol,source,inputs,branch,fit,obs)
            if not trainer.set_capability(report):raise ValueError('Source no longer passes original capability')
            state=trainer.state_dict()
            # Exercise the actual new full loader before granting any sampling.
            trainer.load_state_dict(state)
            if semantic(trainer.state_dict())!=semantic(state):raise ValueError('Initial full MPS restore differs')
            equal[branch]=trainer.initial_learning_state_sha256
            path=output/'branches'/branch/'checkpoints'/'initial.pt'
            _save(path,dict(version=VERSION,branch=branch,identity_sha256=digest(identity),trainer=state,
                actual_steps=0,source_actor_sha256=inputs['source_actor']['sha256']))
            for kind in ('ppo','evaluation'):ledger.initialize_head(kind,branch,str(path.relative_to(output)),file_hash(path))
        if len(set(equal.values()))!=1:raise ValueError('Arms do not start with identical learning state')
        if sources()!=frozen:raise ValueError('Source changed during preparation')
        write_json(output/'prepared.json',dict(version=VERSION,identity=identity,
            inputs=bound(output,output/'inputs.json'),protocol=bound(output,output/'protocol.json'),
            same_initial_learning_state=equal,source_PPO_steps=3880000,new_PPO_steps=0,
            new_environment_steps=0,qualification_granted=False))
    print(json.dumps(dict(event='prepared',output=str(output),PPO_cap=40000,evaluation_cap=72000)),flush=True)


def _evaluate(output,ledger,branch,trainer,actor,protocol,inputs):
    step=trainer.joint_steps;folder=output/'branches'/branch/'validation'/f'step_{step:07d}'
    baseline=_json(source_io._external(inputs['baselines']))
    sha=file_hash(actor)
    def before(context):
        if context['feedback_branch']!=branch or context['actor_bindings']['actor_sha256']!=sha:
            raise ValueError('Evaluation branch or frozen Actor differs')
        reserved=ledger.reserve('evaluation',branch,120,context['operation_id'])
        if not reserved['execution_permitted']:raise ValueError('Cannot repeat reserved episode')
    def after(context):
        row=Path(context['row_path']);trace=Path(context['trace_path'])
        if (not row.is_relative_to(folder) or not trace.is_relative_to(folder) or
                file_hash(row)!=context['row_sha256'] or file_hash(trace)!=context['trace_sha256']):
            raise ValueError('Evaluation evidence mismatch')
        path=output/'branches'/branch/'evaluation_receipts'/(context['operation_id']+'.json')
        write_json(path,dict(version=VERSION,context=context,actor_sha256=sha))
        ledger.ack(context['operation_id'],str(path.relative_to(output)),file_hash(path),context['actual_steps'])
    confirmed={key for key,op in ledger.read()['operations'].items()
        if op['request']['kind']=='evaluation' and op['request']['branch']==branch and op['status']=='acknowledged'}
    kwargs=dict(expected_actor_sha256=sha,reference_report=baseline['reference'],
        random_report=baseline['random'],confirmed_operation_ids=confirmed)
    if (folder/'report.json').exists():
        report=evaluation.read_existing(actor,trainer.scenarios['splits']['validation'],protocol,folder,**kwargs)
    else:
        report=evaluation.evaluate(actor,trainer.scenarios['splits']['validation'],protocol,folder,
            before_episode=before,on_episode=after,**kwargs)
    passed=trainer.set_capability(report)
    print(json.dumps(dict(event='capability',branch=branch,step=step,passed=passed,
        summary=report['summary'],strict=trainer.last_capability)),flush=True)
    return report,passed


def run(output):
    output=Path(output).resolve();prepared=_json(output/'prepared.json');identity=prepared['identity']
    if sources()!=identity['runtime_sources']:raise ValueError('Frozen experiment source changed')
    if (output/'terminal.json').exists():raise ValueError('This bounded experiment is already terminal')
    inputs=_json(source_io._check(output,prepared['inputs']))
    protocol=_json(source_io._check(output,prepared['protocol']))
    source,scenes,fit,report,baseline,obs=_material(inputs)
    ledger=CycleBudget(output,identity)
    started=time.monotonic();trainers={}
    with ledger.lease():
        for branch in BRANCHES:
            if any(ledger.read()['branches'][branch][kind]['pending'] for kind in ('ppo','evaluation')):
                raise ValueError('Unconfirmed operation remains consumed; no automatic replay')
            trainer=_trainer(protocol,source,inputs,branch,fit,obs)
            head=ledger.head('ppo',branch)
            payload=decode(output/head['path'],head['sha256'])
            if payload.get('version')!=VERSION or payload.get('identity_sha256')!=digest(identity):
                raise ValueError('Saved checkpoint identity differs')
            trainer.load_state_dict(payload['trainer'])
            if semantic(trainer.state_dict())!=semantic(payload['trainer']):raise ValueError('Actual MPS resume differs')
            trainers[branch]=trainer
        if len({t.joint_steps for t in trainers.values()})!=1 and not all(t.joint_steps% BATCH==0 for t in trainers.values()):
            raise ValueError('Invalid saved pair progress')
        try:
            for endpoint in (10000,20000):
                gates={}
                for branch in BRANCHES:
                    trainer=trainers[branch]
                    marker=output/'branches'/branch/'boundaries'/f'step_{endpoint:07d}.json'
                    if marker.exists():
                        saved=_json(marker)
                        actor=source_io._check(output,saved['actor'])
                        report_path=source_io._check(output,saved['report'])
                        confirmed={key for key,op in ledger.read()['operations'].items()
                            if op['request']['kind']=='evaluation' and op['request']['branch']==branch and op['status']=='acknowledged'}
                        checked=evaluation.read_existing(actor,scenes['splits']['validation'],protocol,report_path.parent,
                            expected_actor_sha256=saved['actor']['sha256'],reference_report=baseline['reference'],
                            random_report=baseline['random'],confirmed_operation_ids=confirmed,
                            expected_report_sha256=saved['report']['sha256'])
                        if trainer.joint_steps==endpoint:
                            trainer.load_state_dict(decode(source_io._check(output,saved['checkpoint']),saved['checkpoint']['sha256'])['trainer'])
                        if bool(checked['strict_capability_eligible']) is not saved['passed']:
                            raise ValueError('Saved boundary gate differs from original acknowledged evidence')
                        gates[branch]=saved['passed'];continue
                    if trainer.joint_steps>endpoint:raise ValueError('Missing required capability boundary')
                    while trainer.joint_steps<endpoint:
                        if sources()!=identity['runtime_sources']:raise ValueError('Source changed before sampling')
                        step=trainer.joint_steps+BATCH;opid=f'ppo_{branch}_{step:07d}'
                        reservation=ledger.reserve('ppo',branch,BATCH,opid,expected_step=trainer.joint_steps)
                        if not reservation['execution_permitted']:raise ValueError('Duplicate PPO request')
                        before=semantic(trainer.state_dict())
                        batch,metrics=trainer.train_chunk(BATCH//16)
                        if len(batch['transition_records'])!=BATCH or batch['audit']['neural_overrides']!=0:
                            raise ValueError('NN execution or step count differs')
                        evidence=save_batch(output,branch,opid,batch)
                        path=output/'branches'/branch/'checkpoints'/f'step_{step:07d}.pt'
                        _save(path,dict(version=VERSION,branch=branch,identity_sha256=digest(identity),
                            trainer=trainer.state_dict(),before_state_sha256=before,actual_steps=BATCH,
                            evidence=evidence,metrics=metrics,audit=batch['audit']))
                        ledger.ack(opid,str(path.relative_to(output)),file_hash(path),BATCH)
                        write_json(output/'progress.json',dict(version=VERSION,branch=branch,step=step,
                            lambda_value=trainer.current_lambda,totals=ledger.read()['totals']),replace=True)
                        print(json.dumps(dict(event='PPO_ack',branch=branch,step=step,lambda_value=trainer.current_lambda)),flush=True)
                    actor=output/'branches'/branch/'actors'/f'actor_{endpoint:07d}.npz'
                    if actor.exists():
                        source_io._actor_equal(trainer,actor)
                        from env.warehouse_native.policy import NumPyNativeActor
                        evaluation.validate_actor(NumPyNativeActor(actor),protocol,file_hash(actor))
                    else:trainer.export(actor)
                    result,passed=_evaluate(output,ledger,branch,trainer,actor,protocol,inputs)
                    trainer.last_evaluated_joint_steps=endpoint
                    path=marker.with_suffix('.pt')
                    _save(path,dict(version=VERSION,branch=branch,identity_sha256=digest(identity),trainer=trainer.state_dict(),actual_steps=0))
                    write_json(marker,dict(version=VERSION,branch=branch,step=endpoint,passed=passed,
                        checkpoint=bound(output,path),actor=bound(output,actor),report=bound(output,
                        output/'branches'/branch/'validation'/f'step_{endpoint:07d}'/'report.json')))
                    gates[branch]=passed
                if not all(gates.values()) or endpoint==20000:
                    status='capability_passed_at_fixed_endpoint' if all(gates.values()) else 'capability_failed_stop_at_matched_endpoint'
                    write_json(output/'terminal.json',dict(version=VERSION,status=status,endpoint=endpoint,
                        gates=gates,totals=ledger.read()['totals'],elapsed_seconds=time.monotonic()-started,
                        tree_refresh_executed=False,explanation_qualified=False,whole_goal_completed=False))
                    print(json.dumps(dict(event='terminal',status=status,endpoint=endpoint,totals=ledger.read()['totals'])),flush=True)
                    break
        except BaseException as error:
            write_json(output/'failure.json',dict(version=VERSION,error=repr(error),
                totals=ledger.read()['totals'],automatic_retry=False),replace=True)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--source-inputs',type=Path)
    parser.add_argument('--source-sha256')
    args=parser.parse_args()
    if args.prepare:
        if args.source_inputs is None or args.source_sha256 is None:parser.error('Preparation needs externally bound source inputs')
        prepare(args.output,args.source_inputs,args.source_sha256)
    else:run(args.output)


if __name__=='__main__':main()
