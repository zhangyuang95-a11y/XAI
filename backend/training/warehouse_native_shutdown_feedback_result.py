"""Fixed-endpoint retained-beta feedback comparison from saved evidence.

An external completed-report SHA is mandatory. No live progress/ledger, ancestor
checkpoint, model construction, forward, environment or final-test execution is
used. Optional CPU decoding is limited to two final boundary states and the
first recorded positive-feedback update. Saved metrics are not recomputed
gradients, and this reader never selects or publishes an Actor.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from . import warehouse_native_shutdown_feedback_run as runner
from . import warehouse_native_shutdown_feedback_trainer as trainer
from . import warehouse_native_shutdown_stage_evaluation as evaluation
from . import warehouse_native_cycle_budget as budget
from .warehouse_native_shutdown_result import _Inputs, _json_bytes, _same, _int
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from env.warehouse_native.policy import ACTIONS

VERSION = 'warehouse-native-own-shutdown-feedback-result.v1'
BRANCHES = ('control', 'feedback')


def _head(read, root, item, branch):
    if not isinstance(item, dict) or set(item) != {'path', 'sha256', 'step'}:
        raise ValueError('A complete original ledger head is required')
    if Path(item['path']).parts[:2] != ('branches', branch):
        raise ValueError('Checkpoint belongs to another branch')
    _int(item['step']); read.raw(root, item['path'], item['sha256'])
    return item


def _ledger(root, read, p, account):
    """Rebuild only the immutable completed snapshot; never opens cycle_budget.json."""
    caps = p['budget_caps']
    for key, expected in [('version', budget.VERSION), ('root', str(root)), ('identity', p['identity']),
            ('identity_sha256', budget._digest(p['identity'])), ('caps', caps), ('caps_sha256', budget._digest(caps))]:
        _same(account.get(key), expected, 'Completed budget binding differs: '+key)
    if set(account['branches']) != set(BRANCHES): raise ValueError('Missing paired arm')
    rebuilt = {arm: budget._cell(caps[arm]) for arm in BRANCHES}
    for arm in BRANCHES:
        for kind in budget.KINDS:
            first = _head(read, root, account['branches'][arm][kind]['initial_head'], arm)
            if first['step'] != 0: raise ValueError('Initial ledger head is not zero')
            rebuilt[arm][kind].update(initial_head=deepcopy(first), head=deepcopy(first))
    operations = sorted(account['operations'].items(), key=lambda pair: pair[1]['sequence'])
    for seq, (opid, op) in enumerate(operations, 1):
        budget.CycleBudget._opid(opid)
        if op['opid'] != opid or op['sequence'] != seq or op['status'] != 'acknowledged' or op['abandon_reason'] is not None:
            raise ValueError('Unfinished or nonsequential operation cannot supply a completed comparison')
        req = op['request']
        if set(req) != {'kind', 'branch', 'steps', 'expected_step'} or req['kind'] not in budget.KINDS or req['branch'] not in BRANCHES:
            raise ValueError('Invalid original reservation')
        _same(op['request_sha256'], budget._digest(req), 'Reservation digest differs')
        cell = rebuilt[req['branch']][req['kind']]
        _same(op['old_head'], cell['head'], 'Original reservation predecessor differs')
        if req['expected_step'] is not None and req['expected_step'] != cell['head']['step']:
            raise ValueError('Original expected step differs')
        amount = _int(req['steps'], 1); done = op['completion']; actual = _int(done['actual_steps'], 1)
        if set(done) != {'actual_steps', 'checkpoint'} or actual > amount:
            raise ValueError('Actual sampling exceeds the permanent reservation')
        cp = _head(read, root, done['checkpoint'], req['branch'])
        if cp['step'] != cell['head']['step']+actual or (req['kind'] == 'ppo' and actual != amount):
            raise ValueError('Acknowledged cumulative steps differ')
        cell.update(reserved=cell['reserved']+amount, acknowledged=cell['acknowledged']+actual,
            remaining=cell['remaining']-amount, head=deepcopy(cp))
        if cell['remaining'] < 0: raise ValueError('Permanent budget exceeded')
    _same(account['branches'], rebuilt, 'Recorded branch totals differ')
    _same(account['totals'], budget.CycleBudget._totals(rebuilt, caps), 'Recorded global totals differ')
    if account['revision'] != 4+2*len(operations): raise ValueError('Completed ledger revision differs')
    for arm in BRANCHES:
        for kind in budget.KINDS:
            cell = rebuilt[arm][kind]
            if cell['remaining'] or cell['reserved'] != cell['cap'] or (kind == 'ppo' and cell['acknowledged'] != cell['cap']):
                raise ValueError('A complete fixed comparison must exhaust each frozen allocation')
    return operations


def _prepare(root, read, completion, fixture):
    p = read.json(root, 'prepared.json')
    if p.get('version') != runner.VERSION or p.get('test_fixture') is not fixture:
        raise ValueError('Feedback preparation version or fixture boundary differs')
    expected_identity = {k: v for k, v in p.items() if k not in ('identity', 'created_unix', 'formal_ready', 'explanation_qualified')}
    _same(p['identity'], expected_identity, 'Prepared identity differs')
    _same(p['runtime_sources'], runner.sources(), 'Frozen execution sources differ')
    for name, expected in p['runtime_sources'].items(): read.raw(root, 'source_snapshot/'+name, expected)
    protocol = read.json(root, 'protocol.json'); scenes = read.json(root, 'scenarios.json')
    for value, key in [(protocol, 'protocol_sha256'), (scenes, 'scenario_manifest_sha256')]:
        _same(digest(value), p[key], 'Prepared configuration differs: '+key)
    for name, key in [('authorization.json','authorization_record_sha256'), ('baselines.json','baselines_sha256'),
            ('source_validation.json','source_report_sha256')]: read.raw(root, name, p[key])
    if p['authorization_record_sha256'] != runner.AUTHORIZATION_SHA: raise ValueError('Authorization binding differs')
    endpoints = p['validation_endpoints']; cap = p['primary_endpoint']
    if (protocol['version'] != trainer.PROTOCOL_VERSION or protocol['feedback'] != trainer.feedback_contract()
            or protocol['budget']['maximum_ppo_joint_steps_per_arm'] != cap
            or protocol['evaluation']['checkpoints_ppo_steps'] != endpoints or endpoints[-1] != cap
            or endpoints != sorted(set(endpoints)) or any(type(s) is not int or s <= 0 for s in endpoints)):
        raise ValueError('Fixed registered feedback protocol differs')
    if not fixture and (cap != 250000 or endpoints != [50000,100000,150000,200000,250000]):
        raise ValueError('Production comparison is fixed at 250000 with all five registered boundaries')
    trainer.validated_feedback_config(protocol['feedback_config'], fixture)
    initial = read.json(root, 'initialization_check.json')
    if (initial.get('identical_native_learning_state') is not True or initial.get('new_environment_steps') != 0
            or initial.get('program_runtime_controller') is not False or initial['source_state_sha256'] != p['source_state_sha256']):
        raise ValueError('Missing same-source complete-state initialization record')
    required = {'model','optimizers','envs','rng','python_rng','numpy_rng','torch_rng','partner_kinds','program_roles',
        'scenario_ids','episode_context','episode_returns','episode_reward_components'}
    if set(initial['matched']) != required or any(not evaluation.compact._sha(v) for v in initial['matched'].values()):
        raise ValueError('Initial NN, dual Adam, environment and RNG evidence is incomplete')
    fit = read.json(root, 'initial_fit.json')
    if digest(fit) != p['initial_tree']['fit_sha256'] or fit.get('reliable') is not True or fit.get('version') != runner.expanded.VERSION:
        raise ValueError('Reliable initial expanded tree evidence differs')
    pools = read.json(root, 'refresh_pools.json'); contexts = read.json(root, 'refresh_contexts.json')
    _same(digest(pools), p['refresh_pools_sha256'], 'Refresh pools changed')
    _same(contexts, runner._contexts(pools,scenes['configuration']['horizon']), 'Refresh ordering or seed changed')
    _same(digest(contexts), p['refresh_contexts_sha256'], 'Refresh contexts changed')
    if not fixture and (len(pools['train']),len(pools['selection'])) != (192,32): raise ValueError('Expanded pool counts differ')
    per_eval = 3*len(scenes['splits']['validation'])*scenes['configuration']['horizon']
    per_aux = sum(c['horizon'] for c in contexts)
    _same(p['budget_caps'], {a:{'ppo':cap,'evaluation':per_eval*len(endpoints)} for a in BRANCHES}, 'Paired finite caps differ')
    _same(p['auxiliary_caps'], {f'step_{s:07d}':per_aux for s in endpoints}, 'Fixed refresh caps differ')
    if p['maximum_auxiliary_steps'] != per_aux*len(endpoints): raise ValueError('Auxiliary total differs')
    if (completion.get('status') != 'fixed_endpoint_completed' or completion.get('until') != cap
            or completion.get('formal_ready') is not False or completion.get('explanation_qualified') is not False
            or completion.get('total_interaction_equal') is not False
            or completion.get('initial_tree_budget_is_separate') is not True
            or completion.get('control_terminal_tree_extracted') is not False):
        raise ValueError('Only a completed fixed development comparison is accepted')
    return p, protocol, scenes, initial


def _ppo_records(root, read, p, protocol, operations):
    """Verify saved sampled commands; no logits or gradient is recomputed."""
    n = protocol['training']['environments']; summaries = {}; first_positive = None
    for arm in BRANCHES:
        summary = {'joint_steps':0,'updates':0,'trainable_neural_rows':0,'raw_neural_overrides':0,
            'positive_lambda_joint_steps':0,'lambda_min':None,'lambda_max':0.,'operations':[]}
        for opid, op in operations:
            req = op['request']
            if req['kind'] != 'ppo' or req['branch'] != arm: continue
            prefix = f'branches/{arm}/rollouts/{opid}'
            records = [_json_bytes(line) for line in gzip.decompress(read.raw(root,prefix+'.jsonl.gz')).splitlines() if line]
            with np.load(io.BytesIO(read.raw(root,prefix+'.npz')),allow_pickle=False) as arrays:
                amount = req['steps']; tsteps = amount//n
                if amount%n or len(records) != amount or arrays['actions'].shape != (tsteps,n,2) or arrays['trainable'].shape != (tsteps,n,2):
                    raise ValueError('Original PPO rows and reserved count differ')
                lambdas = set()
                for index, record in enumerate(records):
                    t,i = divmod(index,n)
                    if (record['rollout_index'] != t or record['environment_index'] != i
                            or record['version'] != trainer.VERSION or record['feedback_branch'] != arm
                            or record['cycle_id'] != p['cycle_id'] or record['shutdown_arm'] != p['shutdown_arm']
                            or record['own_shutdown_beta'] != p['own_shutdown_beta']):
                        raise ValueError('Original PPO condition or indexing differs')
                    strength = record['feedback_lambda']
                    if type(strength) not in (int,float) or not np.isfinite(strength) or not 0 <= strength <= protocol['feedback_config']['lambda_max']:
                        raise ValueError('Invalid recorded KL coefficient')
                    if arm == 'control' and strength != 0: raise ValueError('Control applied feedback')
                    lambdas.add(float(strength))
                    _same(record['trainable'],arrays['trainable'][t,i].tolist(),'Neural training mask differs')
                    _same(record['neural_sampled_actions'],[ACTIONS[int(a)] for a in arrays['actions'][t,i]],'Sampled actions differ from PPO arrays')
                    for j,key in enumerate(('robot_1','robot_2')):
                        if record['trainable'][j]:
                            summary['trainable_neural_rows'] += 1
                            if record['requested_actions'][key] != record['neural_sampled_actions'][j]:
                                raise ValueError('A trainable neural action was overwritten')
                if len(lambdas) != 1: raise ValueError('KL gate changed inside an original rollout')
            strength = lambdas.pop(); summary['joint_steps'] += amount; summary['updates'] += 1
            summary['positive_lambda_joint_steps'] += amount if strength > 0 else 0
            summary['lambda_min'] = strength if summary['lambda_min'] is None else min(summary['lambda_min'],strength)
            summary['lambda_max'] = max(summary['lambda_max'],strength)
            item = {'operation_id':opid,'step':op['completion']['checkpoint']['step'],'steps':amount,'lambda':strength,
                'checkpoint':deepcopy(op['completion']['checkpoint']),
                'arrays':{'path':prefix+'.npz',**read.records[str(root/(prefix+'.npz'))]},
                'trace':{'path':prefix+'.jsonl.gz',**read.records[str(root/(prefix+'.jsonl.gz'))]}}
            summary['operations'].append(item)
            if arm == 'feedback' and strength > 0 and first_positive is None: first_positive=item
        summaries[arm] = summary
    return summaries,first_positive


def _boundary(root, read, p, arm, step, account, report):
    prefix = f'branches/{arm}/boundaries/step_{step:07d}'
    marker = read.json(root,prefix+'.json')
    operations = [op for op in account['operations'].values() if op['request']['kind']=='ppo'
        and op['request']['branch']==arm and op['completion']['checkpoint']['step']==step]
    if len(operations)!=1: raise ValueError('Boundary has no unique acknowledged PPO predecessor')
    _same(marker['predecessor_head'],operations[0]['completion']['checkpoint'],'Boundary predecessor differs')
    if marker.get('version')!=runner.VERSION or marker['branch']!=arm or marker['new_training_steps']!=0:
        raise ValueError('Boundary changes the learner identity or training count')
    if marker['checkpoint']['path']!=prefix+'.pt': raise ValueError('Effective checkpoint must be the completed boundary state')
    read.bound(root,marker['checkpoint'])
    for item in marker['evidence'].values(): read.bound(root,item)
    validation=f'branches/{arm}/validation/step_{step:07d}'
    for key,name in [('validation_report','report.json'),('validation_manifest','manifest.json')]:
        if marker['evidence'][key]['path']!=validation+'/'+name: raise ValueError('Boundary report belongs elsewhere')
    _same(read.json(root,validation+'/report.json'),report,'Recomputed boundary report differs')
    if arm=='control':
        if marker['gate'].get('lambda')!=0 or set(marker['evidence'])!={'validation_report','validation_manifest'}:
            raise ValueError('Control boundary contains feedback evidence')
        return marker,None
    folder=f'branches/feedback/refresh/step_{step:07d}'
    saved={}
    for name in ('fit_result.json','fit_request.json','manifest.json','plan.json','auxiliary_budget.json'):
        item=marker['evidence']['refresh_'+name.replace('.','_')]
        if item['path']!=folder+'/'+name: raise ValueError('Refresh evidence belongs to another endpoint')
        saved[name]=_json_bytes(read.bound(root,item))
    plan,manifest,fit,request,allocation=(saved[n] for n in ('plan.json','manifest.json','fit_result.json','fit_request.json','auxiliary_budget.json'))
    if (plan.get('version')!=runner.VERSION or plan['pair_identity_sha256']!=digest(p['identity'])
            or manifest.get('status')!='completed' or manifest['plan_sha256']!=digest(plan)
            or len(manifest['episodes'])!=len(plan['contexts']) or manifest['ppo_steps']!=0):
        raise ValueError('Refresh is incomplete or belongs to another paired source')
    _same(plan['contexts'],read.json(root,'refresh_contexts.json'),'Refresh changed its matrix')
    _same(plan['actor_bindings'],report['actor_bindings'],'Refresh does not use the evaluated endpoint Actor')
    _same(plan['expanded_fit_contract'],runner.expanded.contract(),'Refresh extraction contract differs')
    if plan['maximum_rcpd_calls']!=13 or plan['maximum_sklearn_fits']!=104:
        raise ValueError('Expanded refresh search limits differ')
    reserved=actual=0
    for entry,context in zip(manifest['episodes'],plan['contexts']):
        _same(entry['context'],context,'Refresh episode order differs')
        if entry['status']!='completed': raise ValueError('Unconfirmed extraction cannot be ignored')
        raw=_json_bytes(gzip.decompress(read.bound(root/folder,entry['record'])))
        read.bound(root/folder,entry['arrays'])
        if raw['actor_bindings']!=plan['actor_bindings'] or raw['joint_transitions']<=0 or raw['joint_transitions']>context['horizon']:
            raise ValueError('Stored extraction Actor or actual count differs')
        if digest(raw['trace'])!=raw['episode']['trace_sha256']: raise ValueError('Saved extraction trace differs')
        reserved+=context['horizon'];actual+=raw['joint_transitions']
    if (reserved!=manifest['reserved_auxiliary_steps'] or actual!=manifest['actual_auxiliary_steps']
            or reserved!=allocation['reserved_joint_steps'] or reserved!=p['auxiliary_caps'][f'step_{step:07d}']
            or allocation['cap']!=reserved or plan['maximum_auxiliary_steps']!=reserved):
        raise ValueError('Refresh acknowledgments or permanent budget differs')
    _same(_json_bytes(read.bound(root/folder,manifest['fit_result'])),fit,'Cached fit differs')
    _same(_json_bytes(read.bound(root/folder,manifest['fit_request'])),request,'Cached fit request differs')
    if (fit['version']!=runner.expanded.VERSION or fit.get('test_fixture') is not p['test_fixture']
            or request['plan_sha256']!=digest(plan) or request['acknowledged_episodes_sha256']!=digest(manifest['episodes'])
            or request['maximum_candidate_fits']!=13 or request['maximum_sklearn_fits']!=104
            or request['extraction_contract_sha256']!=digest(runner.expanded.contract())):
        raise ValueError('Refreshed fit identity differs')
    if fit['reliable'] is False and marker['gate']['lambda']!=0: raise ValueError('Failed refresh retained active feedback')
    if fit['reliable'] is True:
        binding=fit['fit_report']['observed197_bindings']
        for key in ('actor_sha256','actor_parameters_sha256'):
            if binding[key]!=report['actor_bindings'][key]: raise ValueError('Reliable refresh labels another Actor')
        if fit['evidence_sha256']!=digest({'binding':binding,'fit_report':fit['fit_report']}):
            raise ValueError('Reliable refresh evidence digest differs')
    return marker,{'reserved':reserved,'acknowledged':actual,'reliable':fit['reliable'],
        'gate':deepcopy(marker['gate']),'fit_result_sha256':marker['evidence']['refresh_fit_result_json']['sha256']}


def _facts(report):
    result={}
    for partner in evaluation.PARTNERS:
        s=report['summary'][partner];rows=[r for r in report['rows'] if r['partner']==partner]
        result[partner]={'nn_deliveries':s['mean_ai_deliveries'],'team_deliveries':s['mean_team_deliveries'],
            'program_deliveries':s['mean_program_deliveries'],'nn_survival':s['ai_active_end_rate'],
            'collisions_per_step':s['collisions_per_step'],'longest_consecutive_collisions':max(r['longest_consecutive_collisions'] for r in rows),
            'longest_steps_without_pickup_or_delivery':max(r['longest_steps_without_task_progress'] for r in rows),
            'nn_wall_invalid':sum(r['wall_invalid_by_role'][1] for r in rows),
            'nn_blocked_moves':sum(r['blocked_moves'][1] for r in rows),
            'nn_full_battery_waits':s['full_battery_waits_by_role'][1],'both_active_wait_steps':s['double_wait_steps']}
    return {'primary_value':report['primary_value'],'partners':result,
        'full_capability':report['capability'],'warmup_capability':report['warmup_capability']}


def _comparison(reports):
    a,b=reports['control'],reports['feedback'];differences=[]
    for left,right in zip(a['rows'],b['rows']):
        for key in ('partner','scenario_id','scenario_index','initial_fingerprint','seed'):
            _same(left[key],right[key],'The two conditions use different validation episodes')
        differences.append({**{k:left[k] for k in ('partner','scenario_id','initial_fingerprint','seed')},
            'nn_delivery_difference':right['ai_deliveries']-left['ai_deliveries'],
            'team_delivery_difference':right['team_deliveries']-left['team_deliveries'],
            'survival_difference':int(right['ai_active_end'])-int(left['ai_active_end']),
            'collisions_difference':right['collisions']-left['collisions']})
    return {'control':_facts(a),'feedback':_facts(b),'feedback_minus_control':b['primary_value']-a['primary_value'],
        'paired_rows':differences}


def _decode_cpu(read,root,item):
    import torch
    # No model is constructed, and no GPU storage is restored.
    raw=read.raw(root,item['path'],item['sha256'],item.get('size'))
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)


def _checkpoint_audit(root,read,p,protocol,boundaries,reports,first_positive,completion):
    inspected=[]
    for arm in BRANCHES:
        marker=boundaries[(arm,p['primary_endpoint'])];payload=_decode_cpu(read,root,marker['checkpoint'])
        state=payload['trainer'];native=state['native_state'];binding=reports[p['primary_endpoint']][arm]['actor_bindings']
        if (payload['version']!=runner.VERSION or payload['branch']!=arm or payload['predecessor_head']!=marker['predecessor_head']
                or payload['evidence']!=marker['evidence'] or payload['new_training_steps']!=0
                or state['version']!=trainer.VERSION or state['feedback_branch']!=arm
                or native['joint_steps']!=p['primary_endpoint']): raise ValueError('Final effective checkpoint identity differs')
        _same(state['protocol'],protocol,'Final checkpoint protocol differs')
        _same(native['protocol'],protocol['native_protocol'],'Final native protocol differs')
        _same(state['execution_sources'],trainer.execution_sources(),'Final trainer sources differ')
        if arm=='control':
            if any(state.get(k) is not None for k in ('feedback_state','feedback_evidence','capability_evidence','refresh_failure')) or state.get('refresh_pending') is not False:
                raise ValueError('Control checkpoint carries feedback state')
        else:
            actor_path=f'branches/{arm}/actors/actor_{p["primary_endpoint"]:07d}.npz'
            with np.load(io.BytesIO(read.raw(root,actor_path,binding['actor_sha256'])),allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata_json']))
            manager=runner.expanded.ExactProgramManager(metadata['feature_names'],trainer.validated_feedback_config(protocol['feedback_config'],p['test_fixture']))
            manager.load_state_dict(state['feedback_state']);trainer.validate_program(manager,p['test_fixture'])
            trainer.algorithm._validate_refresh_state(manager,state['capability_evidence'],state['refresh_pending'],state['refresh_failure'])
            evidence=state['feedback_evidence'];fit=manager.last_fit_report;fit_binding=fit.get('observed197_bindings',{})
            if (not evidence or evidence['manager_fit_sha256']!=digest(fit)
                    or evidence['evidence_sha256']!=digest({'binding':fit_binding,'fit_report':fit})
                    or evidence['source_actor_parameters_sha256']!=fit_binding.get('actor_parameters_sha256')
                    or evidence['source_actor_sha256']!=fit_binding.get('actor_sha256')):
                raise ValueError('Final retained program evidence differs')
            if marker['gate']['lambda']!=manager.current_lambda:raise ValueError('Final boundary gate differs from saved manager')
        weights={k[len('actor.'):]:v for k,v in native['model'].items() if k.startswith('actor.')}
        if len(weights)!=6 or semantic(weights)!=binding['actor_parameters_sha256']:
            raise ValueError('Final checkpoint Actor parameters differ from actual exported metadata')
        actor_path=f'branches/{arm}/actors/actor_{p["primary_endpoint"]:07d}.npz'
        with np.load(io.BytesIO(read.raw(root,actor_path,binding['actor_sha256'])),allow_pickle=False) as actor:
            for key,value in weights.items():
                if not np.array_equal(value.numpy(),actor[key]):raise ValueError('Final Actor export differs from the retained checkpoint')
        counters={key:native[key] for key in ('joint_steps','optimizer_updates','actor_optimizer_steps','critic_optimizer_steps')}
        _same({'ppo_steps':counters.pop('joint_steps'),**counters},completion['counts'][arm],'Final actual counters differ from completion')
        for role in ('actor','critic'):
            steps={int(value['step'].item()) for value in native['optimizers'][role]['state'].values()}
            if steps!={native['source_counters'][role+'_optimizer_steps']+native[role+'_optimizer_steps']}:
                raise ValueError('Final dual Adam counters differ')
        inspected.append({'branch':arm,'kind':'final_effective_boundary','checkpoint':marker['checkpoint'],
            'actor_parameters_sha256':semantic(weights),'actor_export_equal':True,'dual_adam_steps_verified':True})
        del payload,state,native,weights
    if first_positive is not None:
        payload=_decode_cpu(read,root,first_positive['checkpoint']);metrics=payload['metrics']
        if (payload['version']!=runner.VERSION or payload['cycle_id']!=p['cycle_id']
                or payload['operation_id']!=first_positive['operation_id'] or payload['branch']!='feedback' or payload['audit']['neural_overrides']!=0
                or payload['trainer']['version']!=trainer.VERSION or payload['trainer']['feedback_branch']!='feedback'
                or payload['trainer']['native_state']['joint_steps']!=first_positive['step']):
            raise ValueError('Positive feedback checkpoint belongs to another update')
        _same(payload['trainer']['protocol'],protocol,'Positive update used another paired protocol')
        for kind in ('arrays','trace'):_same(payload['evidence'][kind],first_positive[kind],'Positive update rollout evidence differs')
        for value in metrics.values():
            if not np.isfinite(value):raise ValueError('Nonfinite recorded update metric')
        if (metrics['feedback_lambda']!=first_positive['lambda'] or metrics['feedback_lambda']<=0
                or metrics['feedback_gradient_norm']<0 or metrics['feedback_rows']<0
                or not np.isclose(metrics['feedback_loss'],metrics['feedback_lambda']*metrics['feedback_kl'],rtol=1e-5,atol=1e-8)):
            raise ValueError('Positive KL update lacks consistent committed loss/gradient evidence')
        inspected.append({'branch':'feedback','kind':'first_positive_lambda_update','checkpoint':first_positive['checkpoint'],
            'metrics':metrics,'nonzero_feedback_gradient_observed':metrics['feedback_gradient_norm']>0,
            'scope':'Original committed metrics only; gradients and minibatches were not rerun'})
    return inspected


def compare_fixed(output, *, expected_completion_sha256, decode_checkpoints=False, allow_test_fixture=False):
    """Inspect a complete immutable endpoint; default zero PT decodes, optional at most three."""
    if type(decode_checkpoints) is not bool or type(allow_test_fixture) is not bool or not evaluation.compact._sha(expected_completion_sha256):
        raise ValueError('An external completion SHA and explicit inspection scope are required')
    root=Path(output).expanduser().resolve();read=_Inputs()
    # This is the sole external entry anchor. No live source/ledger is opened.
    filename='completion_0250000.json'
    if allow_test_fixture:
        fixture_prepared=read.json(root,'prepared.json');filename=f'completion_{fixture_prepared["primary_endpoint"]:07d}.json'
    completion=read.json(root,filename,expected_completion_sha256)
    p,protocol,scenes,initial=_prepare(root,read,completion,allow_test_fixture)
    account=completion['ledger'];operations=_ledger(root,read,p,account)
    training,positive=_ppo_records(root,read,p,protocol,operations)
    for arm in BRANCHES:
        c=completion['counts'][arm]
        if c['ppo_steps']!=p['primary_endpoint'] or c['optimizer_updates']!=training[arm]['updates']:
            raise ValueError('Completed update count differs from acknowledged PPO batches')
    _same(completion['counts']['control'],completion['counts']['feedback'],'PPO and update allocations differ')
    reports={};boundaries={};aux={};actual_eval=0
    for step in p['validation_endpoints']:
        reports[step]={}
        for arm in BRANCHES:
            report=runner._report_record(root,arm,step,p,protocol,scenes,account,allow_test_fixture)
            reports[step][arm]=report;actual_eval+=report['environment_steps']
            marker,refresh=_boundary(root,read,p,arm,step,account,report);boundaries[(arm,step)]=marker
            if refresh is not None:aux[f'step_{step:07d}']=refresh
            folder=f'branches/{arm}/validation/step_{step:07d}'
            manifest=read.json(root,folder+'/manifest.json')
            for entry in manifest['episodes']:
                for kind in ('row','trace'):read.bound(root/folder,entry[kind])
            read.raw(root,f'branches/{arm}/actors/actor_{step:07d}.npz',report['actor_bindings']['actor_sha256'])
    if actual_eval!=account['totals']['evaluation']['acknowledged']:raise ValueError('Validation reports omit acknowledged evaluation')
    auxiliary={'cap':p['maximum_auxiliary_steps'],'reserved':sum(v['reserved'] for v in aux.values()),
        'acknowledged':sum(v['acknowledged'] for v in aux.values()),
        'boundaries':{k:{'reserved':v['reserved'],'acknowledged':v['acknowledged']} for k,v in aux.items()}}
    _same(auxiliary,completion['auxiliary'],'Original completed extraction accounting differs')
    decoded=_checkpoint_audit(root,read,p,protocol,boundaries,reports,positive,completion) if decode_checkpoints else []
    if len(decoded)>3:raise RuntimeError('Checkpoint inspection limit exceeded')
    read.unchanged()
    counts=read.counts();counts.update(checkpoint_decodes=len(decoded),numpy_actor_loads=len(p['validation_endpoints'])*2,
        actor_constructions_scope='NumPy validation objects loaded without forward; zero Torch NN constructions',torch_nn_constructions=0)
    counts.pop('actor_constructions',None)
    return {'version':VERSION,'status':'fixed_endpoint_complete','cycle_id':p['cycle_id'],'primary_endpoint':p['primary_endpoint'],
        'primary':_comparison(reports[p['primary_endpoint']]),
        'descriptive_endpoints':{str(s):_comparison(r) for s,r in reports.items() if s!=p['primary_endpoint']},
        'training':training,'checkpoint_audits':decoded,'first_positive_feedback_operation':positive,
        'feedback_gradient_scope':'Only explicitly decoded committed metrics; no complete minibatch gradient history or causal proof',
        'same_source_initialization':initial,'initialization_scope':'Original bound same-state preparation record; initial PTs hashed, not independently decoded',
        'budget':account['totals'],'auxiliary':auxiliary,'refresh_status':aux,'total_interaction_equal':False,
        'completion_sha256':expected_completion_sha256,'inspection_counts':counts,'input_bindings':read.records,
        'result_source_sha256':file_hash(Path(__file__)),'source_selection':None,'test_fixture':allow_test_fixture,
        'scope':'Saved reports/trajectories and local completion receipts; not new physics, NN, fitting or statistical causal replication',
        'release_ready':False,'formal_ready':False,'explanation_qualified':False,'control_terminal_tree_extracted':False}
