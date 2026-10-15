"""Fixed 50k-per-arm continuation with original capability and teacher receipts.

Each arm resumes its own actual stable 20k learner. Capability failure never
shortens this fixed comparison; feedback failure permanently closes its lambda.
No teacher collection, extraction, model release or action controller is added.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import time
import numpy as np

from . import warehouse_family_alignment_trainer as learner
from . import warehouse_family_alignment_evaluation as evaluation
from . import warehouse_family_stable_run as old
from . import warehouse_family_stable_runtime_collection as collection
from . import warehouse_family_branch_teacher_fit as data_validator
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_native_cycle_budget import CycleBudget
from .warehouse_native import atomic_torch_save
from .warehouse_native_partner_mix_run import write_json, bound, decode, save_batch
from env.warehouse_native.policy import NumPyNativeActor

VERSION = 'warehouse-family-alignment-run.v1'
SOURCE = ROOT/'output/warehouse_native/stable_feedback_20k_pair_v2_20260910'
DATA = ROOT/'output/warehouse_native/stable_teacher_data_390m_20260910'
ALIGNMENT = ROOT/'output/warehouse_native/frozen_teacher_alignment_390m_20260910/report.json'
ALIGNMENT_SHA = '3e3c2419277d39fb8dafc80f46b1996e9fd9a667809cd34d1b7851bbc8ca01d0'
DATA_SHA = '65f2134ab4eb9b78fe7d70f89651ec359d40a28fa2fda79f51333a7c66a5c61f'
BRANCHES = ('control','feedback')
ENDPOINTS = (10000,20000,30000,40000,50000)
PPO_CAP, EVALUATION_CAP, BATCH = 50000,90000,2000
SOURCE_ACTORS = {'control':'9fc12abf7f9caf7bcc2941c634b33ff7be2da6e21cf77bb2e6fb8527485f6ddf',
    'feedback':'88a7e744dcc5e435ce05c0c06547f02a5f17b88dadd88dc2ab47725a10647a74'}
SOURCE_CHECKPOINTS = {'control':'3f553d236f842cd5f70e09131b2cbaa86d9cdbd5928eb49e214f4964be0811ea',
    'feedback':'ff76fcc5aefe8dab96a90bde7842284e2d02236803a81caaf572edcbadf8c253'}
source_io = old.source_io


def _json(path): return json.loads(Path(path).read_bytes())


def sources():
    value = {**old.sources(), **learner.execution_sources(), **evaluation.execution_sources()}
    # The read-only codec verifies these exact original collection producers.
    value.update(_json(DATA/'plan.json')['source_files'])
    for module in (old, source_io, collection, data_validator):
        value[str(Path(module.__file__).relative_to(ROOT))] = file_hash(module.__file__)
    for name in ('warehouse_native_cycle_budget.py','warehouse_native_partner_mix_run.py'):
        path=Path(__file__).with_name(name); value[str(path.relative_to(ROOT))]=file_hash(path)
    value[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    if any(file_hash(ROOT/name)!=expected for name,expected in value.items()):
        raise ValueError('An original collection or execution source changed')
    return value


def _external(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=file_hash(path),size=path.stat().st_size)


def _inputs():
    prepared=_json(SOURCE/'prepared.json')
    if prepared['version']!=old.VERSION or prepared['identity']['runtime_sources']!=old.sources():
        raise ValueError('Original prepared stable producer differs')
    terminal=_json(SOURCE/'terminal.json')
    if terminal['endpoint']!=20000:
        raise ValueError('Original stable run must have terminated at its actual 20k boundary')
    if file_hash(ALIGNMENT)!=ALIGNMENT_SHA or file_hash(DATA/'manifest.json')!=DATA_SHA:
        raise ValueError('Frozen teacher alignment or current collection differs')
    result=dict(version=VERSION,source_root=str(SOURCE),source_prepared=_external(SOURCE/'prepared.json'),
        source_terminal=_external(SOURCE/'terminal.json'),
        source_inputs=_external(source_io._check(SOURCE,prepared['inputs'])),
        source_protocol=_external(source_io._check(SOURCE,prepared['protocol'])),
        source_teacher=_external(old.TEACHER),alignment=_external(ALIGNMENT),
        collection_root=str(DATA),collection_manifest=_external(DATA/'manifest.json'),
        collection_plan=_external(DATA/'plan.json'),branches={},test_fixture=False)
    for branch in BRANCHES:
        marker=SOURCE/'branches'/branch/'boundaries'/'step_0020000.json'
        saved=_json(marker)
        if (saved['version']!=old.VERSION or saved['branch']!=branch or saved['step']!=20000
                or saved['actor']['sha256']!=SOURCE_ACTORS[branch]
                or saved['checkpoint']['sha256']!=SOURCE_CHECKPOINTS[branch]):
            raise ValueError('Wrong actual stable branch boundary')
        result['branches'][branch]={'marker':_external(marker),
            **{key:_external(source_io._check(SOURCE,saved[key])) for key in ('checkpoint','actor','report')}}
    return result


def _material(inputs):
    """Real old constructors/loaders only; this function executes no training."""
    if inputs['version']!=VERSION or inputs['source_root']!=str(SOURCE) or inputs['collection_root']!=str(DATA):
        raise ValueError('Alignment inputs point to another experiment')
    for key in ('source_prepared','source_terminal','source_inputs','source_protocol','source_teacher',
                'alignment','collection_manifest','collection_plan'):
        source_io._external(inputs[key])
    prepared=_json(source_io._external(inputs['source_prepared']))
    if prepared['identity']['runtime_sources']!=old.sources():
        raise ValueError('Actual old restore implementation changed')
    original_inputs=_json(source_io._external(inputs['source_inputs']))
    old_protocol=_json(source_io._external(inputs['source_protocol']))
    origin,scenes,fit,_,baseline,old_auxiliary=old._material(original_inputs)
    ledger=CycleBudget(SOURCE,prepared['identity']); source_ledger=ledger.read()
    if any(source_ledger['branches'][b][k]['pending'] for b in BRANCHES for k in ('ppo','evaluation')):
        raise ValueError('Old stable source has an unresolved operation')
    restored={}; entries={}
    for branch in BRANCHES:
        record=inputs['branches'][branch]
        for key in ('marker','checkpoint','actor','report'): source_io._external(record[key])
        payload=decode(source_io._external(record['checkpoint']),record['checkpoint']['sha256'])
        if (payload.get('version')!=old.VERSION or payload.get('branch')!=branch
                or payload.get('identity_sha256')!=digest(prepared['identity'])):
            raise ValueError('Stable boundary checkpoint identity differs')
        instance=old._trainer(old_protocol,origin,original_inputs,branch,fit,old_auxiliary)
        instance.load_state_dict(payload['trainer'])
        actual=instance.state_dict(); state_sha=semantic(actual)
        if state_sha!=semantic(payload['trainer']) or instance.joint_steps!=20000:
            raise ValueError('Actual old full MPS learner restore differs')
        if 'source_state_sha256' in record and record['source_state_sha256']!=state_sha:
            raise ValueError('Frozen source learner semantic state differs')
        record['source_state_sha256']=state_sha
        actor=source_io._external(record['actor'])
        source_io._actor_equal(instance,actor)
        confirmed={key for key,op in source_ledger['operations'].items()
            if op['request']['kind']=='evaluation' and op['request']['branch']==branch and op['status']=='acknowledged'}
        report=old.evaluation.read_existing(actor,scenes['splits']['validation'],old_protocol,
            source_io._external(record['report']).parent,expected_actor_sha256=record['actor']['sha256'],
            reference_report=baseline['reference'],random_report=baseline['random'],
            confirmed_operation_ids=confirmed,expected_report_sha256=record['report']['sha256'])
        if bool(report['strict_capability_eligible']) is not _json(source_io._external(record['marker']))['passed']:
            raise ValueError('Original source boundary decision differs from real acknowledged episodes')
        restored[branch]=instance; entries[branch]=report
    pools,plan,manifest=collection.read_data(DATA)
    contracts={pool:data_validator.validate_data(pools[pool],pool=pool) for pool in ('train','selection')}
    if (plan['actor_training_clock']!=3900000 or plan['actor_bindings']['actor_sha256']!=SOURCE_ACTORS['feedback']
            or plan['actor_bindings']['actor_parameters_sha256']!=learner._actor_parameters(restored['feedback'])
            or file_hash(DATA/'manifest.json')!=DATA_SHA):
        raise ValueError('Current saved TRAIN labels do not belong to the actual feedback source')
    indices=np.asarray([(p['baseline_index'],p['changed_index']) for p in pools['train']['pairs']
        if p['physical_effect'] and p['nn_changed']],np.int64)
    if indices.shape!=(824,2): raise ValueError('Preserve all 824 current TRAIN changed pairs')
    auxiliary=pools['train']['observations'][indices].copy()
    if auxiliary.dtype!=np.float32 or auxiliary.shape!=(824,2,197): raise ValueError('Wrong auxiliary public observations')
    auxiliary.setflags(write=False)
    binding=dict(version=learner.AUXILIARY_VERSION,source_collection_manifest_sha256=DATA_SHA,
        source_train_data_sha256=contracts['train']['data_sha256'],
        source_selection_data_sha256=contracts['selection']['data_sha256'],
        actor_sha256=plan['actor_bindings']['actor_sha256'],actor_parameters_sha256=plan['actor_bindings']['actor_parameters_sha256'],
        observations_sha256=semantic(auxiliary),pair_count=824,source_pool='train',
        pair_filter='physical_effect_and_nn_changed',selection_excluded=True,
        row_index_pairs_sha256=semantic(indices),feature_names=list(plan['feature_names']))
    alignment=_json(source_io._external(inputs['alignment']))
    if file_hash(source_io._external(inputs['alignment']))!=ALIGNMENT_SHA:
        raise ValueError('Fixed teacher alignment evidence differs')
    return restored,scenes,entries,baseline,auxiliary,binding,pools['selection'],alignment


def _trainer(protocol,source,inputs,branch,auxiliary):
    record=inputs['branches'][branch]
    return learner.AlignmentTrainer(protocol,source,
        source_checkpoint_sha256=record['checkpoint']['sha256'],
        expected_source_state_sha256=record['source_state_sha256'],
        auxiliary_observations=auxiliary if branch=='feedback' else None,device='mps')


def _save(path,value):
    if path.exists():
        if semantic(decode(path,file_hash(path)))!=semantic(value):
            raise ValueError('Existing immutable checkpoint differs')
        return
    path.parent.mkdir(parents=True,exist_ok=True); atomic_torch_save(path,value)


def prepare(output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new versioned alignment run directory')
    frozen=sources(); inputs=_inputs()
    restored,_,entries,_,auxiliary,binding,_,alignment=_material(inputs)
    protocols={}
    for branch in BRANCHES:
        record=inputs['branches'][branch]
        protocols[branch]=learner.make_protocol(restored[branch],
            source_checkpoint_sha256=record['checkpoint']['sha256'],source_state_sha256=record['source_state_sha256'],
            entry_actor_sha256=record['actor']['sha256'],entry_report=entries[branch],
            teacher_alignment=alignment if branch=='feedback' else None,
            auxiliary_binding=binding if branch=='feedback' else None)
    output.mkdir(parents=True)
    write_json(output/'inputs.json',inputs); write_json(output/'protocols.json',protocols)
    identity=dict(version=VERSION,cycle_id='alignment_50k_20260910',protocol_sha256=digest(protocols),
        branch_protocol_sha256={b:digest(p) for b,p in protocols.items()},runtime_sources=frozen,
        budget_caps={b:dict(ppo=PPO_CAP,evaluation=EVALUATION_CAP) for b in BRANCHES},
        input_sha256=digest(inputs),teacher_check=dict(feedback_boundaries=list(ENDPOINTS),
            maximum_NN_batches=5,maximum_NN_query_rows=5*9289,environment_steps=0),test_fixture=False)
    ledger=CycleBudget.create(output,identity); verified={}
    with ledger.lease():
        for branch in BRANCHES:
            trainer=_trainer(protocols[branch],restored[branch],inputs,branch,auxiliary)
            state=trainer.state_dict(); trainer.load_state_dict(state)
            if semantic(trainer.state_dict())!=semantic(state): raise ValueError('New full MPS learner round-trip differs')
            source_io._actor_equal(trainer,source_io._external(inputs['branches'][branch]['actor']))
            verified[branch]=dict(source_state_sha256=inputs['branches'][branch]['source_state_sha256'],
                initial_learning_state_sha256=trainer.initial_learning_state_sha256,round_trip_equal=True,
                initial_Actor_equals_own_source=True,entry_capability_passed=protocols[branch]['entry_capability']['passed'])
            path=output/'branches'/branch/'checkpoints'/'initial.pt'
            _save(path,dict(version=VERSION,branch=branch,identity_sha256=digest(identity),trainer=state,actual_steps=0))
            write_json(output/'branches'/branch/'protocol.json',protocols[branch])
            for kind in ('ppo','evaluation'): ledger.initialize_head(kind,branch,str(path.relative_to(output)),file_hash(path))
        if sources()!=frozen: raise ValueError('Source changed during preparation')
        write_json(output/'prepared.json',dict(version=VERSION,identity=identity,
            inputs=bound(output,output/'inputs.json'),protocols=bound(output,output/'protocols.json'),
            actual_source_restoration=verified,arms_share_initial_learning_state=False,
            source_cumulative_PPO_steps_per_arm=3900000,new_PPO_steps=0,new_environment_steps=0,
            new_NN_forward_calls=0,qualification_granted=False))
    print(json.dumps(dict(event='prepared',output=str(output),PPO_cap=2*PPO_CAP,
        evaluation_cap=2*EVALUATION_CAP,arms_resume_their_own_source=True)),flush=True)


def _confirmed(ledger,branch):
    return {key for key,op in ledger.read()['operations'].items() if op['request']['kind']=='evaluation'
        and op['request']['branch']==branch and op['status']=='acknowledged'}


def _evaluate(output,ledger,branch,trainer,actor,protocol,baseline):
    step=trainer.joint_steps; folder=output/'branches'/branch/'validation'/f'step_{step:07d}'; sha=file_hash(actor)
    def before(context):
        if context['feedback_branch']!=branch or context['actor_bindings']['actor_sha256']!=sha:
            raise ValueError('Evaluation branch or Actor differs')
        if not ledger.reserve('evaluation',branch,120,context['operation_id'])['execution_permitted']:
            raise ValueError('Cannot repeat a reserved episode')
    def after(context):
        row,trace=Path(context['row_path']),Path(context['trace_path'])
        if (not row.is_relative_to(folder) or not trace.is_relative_to(folder)
                or file_hash(row)!=context['row_sha256'] or file_hash(trace)!=context['trace_sha256']):
            raise ValueError('Evaluation evidence differs')
        receipt=output/'branches'/branch/'evaluation_receipts'/(context['operation_id']+'.json')
        write_json(receipt,dict(version=VERSION,context=context,actor_sha256=sha))
        ledger.ack(context['operation_id'],str(receipt.relative_to(output)),file_hash(receipt),context['actual_steps'])
    kwargs=dict(expected_actor_sha256=sha,reference_report=baseline['reference'],
        random_report=baseline['random'],confirmed_operation_ids=_confirmed(ledger,branch))
    if (folder/'report.json').exists():
        report=evaluation.read_existing(actor,trainer.scenarios['splits']['validation'],protocol,folder,**kwargs)
    else:
        report=evaluation.evaluate(actor,trainer.scenarios['splits']['validation'],protocol,folder,
            before_episode=before,on_episode=after,**kwargs)
    passed=trainer.set_capability(report)
    print(json.dumps(dict(event='capability',branch=branch,step=step,passed=passed,
        summary=report['summary'],strict=trainer.last_capability)),flush=True)
    return report,passed


def _restore_boundary(output,ledger,branch,trainer,protocol,baseline,selection,marker,identity):
    saved=_json(marker); endpoint=saved['step']
    if saved['version']!=VERSION or saved['branch']!=branch or endpoint not in ENDPOINTS:
        raise ValueError('Saved boundary identity differs')
    actor=source_io._check(output,saved['actor']); report_path=source_io._check(output,saved['report'])
    report=evaluation.read_existing(actor,trainer.scenarios['splits']['validation'],protocol,report_path.parent,
        expected_actor_sha256=saved['actor']['sha256'],reference_report=baseline['reference'],
        random_report=baseline['random'],confirmed_operation_ids=_confirmed(ledger,branch),
        expected_report_sha256=saved['report']['sha256'])
    teacher=None
    if branch=='feedback':
        teacher_path=source_io._check(output,saved['teacher_report'])
        teacher=evaluation.read_teacher_check(actor,protocol,selection,trainer.program,teacher_path.parent,
            expected_actor_sha256=saved['actor']['sha256'],
            selection_data_sha256=protocol['auxiliary_binding']['source_selection_data_sha256'],
            expected_report_sha256=saved['teacher_report']['sha256'])
    elif saved['teacher_report'] is not None: raise ValueError('Control cannot have a teacher check')
    payload=decode(source_io._check(output,saved['checkpoint']),saved['checkpoint']['sha256'])
    if (payload['version']!=VERSION or payload['branch']!=branch or payload['identity_sha256']!=digest(identity)
            or payload['trainer']['joint_steps']!=endpoint):
        raise ValueError('Saved full boundary checkpoint differs')
    if trainer.joint_steps==endpoint:
        trainer.load_state_dict(payload['trainer'])
        if semantic(trainer.state_dict())!=semantic(payload['trainer']): raise ValueError('Boundary full-state restore differs')
    if (bool(report['strict_capability_eligible']) is not saved['capability_passed']
            or (teacher['passed'] if teacher else None) != saved['teacher_passed']):
        raise ValueError('Saved boundary gates differ from acknowledged evidence')
    for key,expected in (('capability_history',report),('teacher_reliability_history',teacher)):
        if expected is None: continue
        entries=[x for x in getattr(trainer,key) if x['joint_steps']==endpoint]
        if len(entries)!=1 or entries[0]['report_sha256']!=digest(expected):
            raise ValueError('Resumed learner omits or changes a completed admission boundary')
    return saved


def run(output):
    output=Path(output).resolve(); prepared=_json(output/'prepared.json'); identity=prepared['identity']
    if prepared['version']!=VERSION or sources()!=identity['runtime_sources']:
        raise ValueError('Frozen alignment execution changed')
    if (output/'terminal.json').exists(): raise ValueError('This fixed alignment experiment is already terminal')
    inputs=_json(source_io._check(output,prepared['inputs'])); protocols=_json(source_io._check(output,prepared['protocols']))
    if digest(inputs)!=identity['input_sha256'] or digest(protocols)!=identity['protocol_sha256']:
        raise ValueError('Prepared source inputs or per-arm protocols changed')
    restored,_,_,baseline,auxiliary,_,selection,_=_material(inputs)
    ledger=CycleBudget(output,identity); started=time.monotonic(); trainers={}
    with ledger.lease():
        for branch in BRANCHES:
            if any(ledger.read()['branches'][branch][kind]['pending'] for kind in ('ppo','evaluation')):
                raise ValueError('Unconfirmed operation remains consumed; no automatic replay')
            trainer=_trainer(protocols[branch],restored[branch],inputs,branch,auxiliary)
            head=ledger.head('ppo',branch); payload=decode(output/head['path'],head['sha256'])
            if payload['version']!=VERSION or payload['branch']!=branch or payload['identity_sha256']!=digest(identity):
                raise ValueError('Current learner checkpoint identity differs')
            trainer.load_state_dict(payload['trainer'])
            if semantic(trainer.state_dict())!=semantic(payload['trainer']): raise ValueError('Current MPS resume differs')
            if trainer.joint_steps!=head['step'] or trainer.joint_steps%BATCH or not 0<=trainer.joint_steps<=PPO_CAP:
                raise ValueError('Actual learner step differs from permanently acknowledged PPO budget')
            trainers[branch]=trainer
        try:
            for endpoint in ENDPOINTS:
                boundaries={}
                for branch in BRANCHES:
                    trainer=trainers[branch]; protocol=protocols[branch]
                    marker=output/'branches'/branch/'boundaries'/f'step_{endpoint:07d}.json'
                    if marker.exists():
                        boundaries[branch]=_restore_boundary(output,ledger,branch,trainer,protocol,
                            baseline,selection,marker,identity); continue
                    if trainer.joint_steps>endpoint: raise ValueError('Missing required completed boundary')
                    while trainer.joint_steps<endpoint:
                        if sources()!=identity['runtime_sources']: raise ValueError('Source changed before sampling')
                        step=trainer.joint_steps+BATCH; opid=f'ppo_{branch}_{step:07d}'
                        if not ledger.reserve('ppo',branch,BATCH,opid,expected_step=trainer.joint_steps)['execution_permitted']:
                            raise ValueError('Duplicate PPO request')
                        before=semantic(trainer.state_dict()); batch,metrics=trainer.train_chunk(BATCH//16)
                        if (trainer.joint_steps!=step or len(batch['transition_records'])!=BATCH
                                or batch['audit']['neural_overrides']!=0):
                            raise ValueError('Raw NN execution or exact step count differs')
                        evidence=save_batch(output,branch,opid,batch)
                        path=output/'branches'/branch/'checkpoints'/f'step_{step:07d}.pt'
                        _save(path,dict(version=VERSION,branch=branch,identity_sha256=digest(identity),trainer=trainer.state_dict(),
                            before_state_sha256=before,actual_steps=BATCH,evidence=evidence,metrics=metrics,audit=batch['audit']))
                        ledger.ack(opid,str(path.relative_to(output)),file_hash(path),BATCH)
                        write_json(output/'progress.json',dict(version=VERSION,branch=branch,step=step,
                            lambda_value=trainer.current_lambda,totals=ledger.read()['totals']),replace=True)
                        print(json.dumps(dict(event='PPO_ack',branch=branch,step=step,lambda_value=trainer.current_lambda)),flush=True)
                    actor=output/'branches'/branch/'actors'/f'actor_{endpoint:07d}.npz'
                    if actor.exists():
                        source_io._actor_equal(trainer,actor)
                        evaluation.validate_actor(NumPyNativeActor(actor),protocol,file_hash(actor))
                    else: trainer.export(actor)
                    _,passed=_evaluate(output,ledger,branch,trainer,actor,protocol,baseline)
                    teacher_report=None; teacher_passed=None
                    if branch=='feedback':
                        folder=output/'branches'/branch/'teacher_checks'/f'step_{endpoint:07d}'
                        teacher_report=evaluation.check_teacher(actor,protocol,selection,trainer.program,folder,
                            expected_actor_sha256=file_hash(actor),
                            selection_data_sha256=protocol['auxiliary_binding']['source_selection_data_sha256'])
                        teacher_passed=trainer.set_teacher_reliability(teacher_report)
                        print(json.dumps(dict(event='teacher_reliability',step=endpoint,passed=teacher_passed,
                            lambda_value=trainer.current_lambda,ordinary=teacher_report['ordinary_metrics']['overall'])),flush=True)
                    trainer.last_evaluated_joint_steps=endpoint
                    path=marker.with_suffix('.pt')
                    _save(path,dict(version=VERSION,branch=branch,identity_sha256=digest(identity),trainer=trainer.state_dict(),actual_steps=0))
                    saved=dict(version=VERSION,branch=branch,step=endpoint,capability_passed=passed,teacher_passed=teacher_passed,
                        feedback_allowed=trainer.feedback_allowed,lambda_value=trainer.current_lambda,
                        checkpoint=bound(output,path),actor=bound(output,actor),
                        report=bound(output,output/'branches'/branch/'validation'/f'step_{endpoint:07d}'/'report.json'),
                        teacher_report=bound(output,folder/'report.json') if teacher_report is not None else None)
                    write_json(marker,saved); boundaries[branch]=saved
                if endpoint==PPO_CAP:
                    write_json(output/'terminal.json',dict(version=VERSION,status='fixed_50k_pair_completed',endpoint=endpoint,
                        capability_gates={b:v['capability_passed'] for b,v in boundaries.items()},
                        teacher_gate=boundaries['feedback']['teacher_passed'],feedback_allowed=trainers['feedback'].feedback_allowed,
                        totals=ledger.read()['totals'],elapsed_seconds=time.monotonic()-started,
                        teacher_checks=dict(completed_NN_batches=5,completed_NN_query_rows=5*9289,environment_steps=0),
                        tree_refresh_executed=False,explanation_qualified=False,whole_goal_completed=False))
                    print(json.dumps(dict(event='terminal',status='fixed_50k_pair_completed',totals=ledger.read()['totals'])),flush=True)
        except BaseException as error:
            write_json(output/'failure.json',dict(version=VERSION,error=repr(error),totals=ledger.read()['totals'],
                automatic_retry=False),replace=True)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.output)
    else: run(args.output)


if __name__=='__main__': main()
