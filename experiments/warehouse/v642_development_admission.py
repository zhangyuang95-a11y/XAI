"""Independently replay three real development streams before locking budget.

No caller-supplied `passed` flag or symbolic protocol declaration grants
formal authority. Failed program pools remain failed without invented fresh
task scores. A successful lock authorizes only formal preparation/training.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
from pathlib import Path
import json
from . import v642_long_training_protocol as protocol
from .v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

ROOT=protocol.ROOT
VERSION='warehouse-v642-independently-replayed-development-budget-lock.v1'
OUTPUT=ROOT/'output/warehouse_v642/development-admission'
_TOKEN=object()
FILES=('experiments/warehouse/v642_development_admission.py',
       'tests/test_warehouse_v642_development_admission.py','docs/warehouse_v642_development_admission.md')


def source_files():
    from . import v642_long_training_quality as quality
    files=quality.source_files();files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def _asset(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'explicit hashed evidence pointer required')
    path=Path(ref['path'])
    require(pointer(path)==ref and not path.is_symlink() and not path.stat().st_mode&0o222,
            'development evidence is missing, writable or changed')
    return path


def _quality_totals(outcomes,variant):
    rows=[r for r in outcomes if r['variant']==variant]
    require(len(rows)==75 and all(r['complete'] is True and type(r['shutdown']) is bool
        and type(r['deliveries']) is type(r['robot_collisions']) is int for r in rows),
        'budget lock requires the exact complete-episode integer outcomes')
    return dict(episodes=len(rows),deliveries=sum(r['deliveries'] for r in rows),
        collisions=sum(r['robot_collisions'] for r in rows),shutdown_episodes=sum(r['shutdown'] for r in rows))


def _program_metrics(row):
    f,e=row['metrics']['feedback'],row['metrics']['explanation']
    return dict(feedback_coverage=f['coverage'],explanation_coverage=e['coverage'],
        expected_feedback_coverage=f['expected_actionable_coverage'],feedback_agreement=f['agreement'],
        explanation_agreement=e['agreement'],feedback_KL=f['kl'],explanation_KL=e['kl'],
        illegal_actions=max(f['whole_tree_raw_illegal_count'],e['whole_tree_raw_illegal_count']),
        unproved_paths=max(f['whole_tree_path_uncertified_count'],e['whole_tree_path_uncertified_count']))


def initial_identity(context):
    p=context['plan'];starts=[e for e in context['events'] if e['kind']=='initialized']
    require(len(starts)==2 and {e['variant'] for e in starts}==set(protocol.VARIANTS),
            'budget lock is missing actual paired initialization events')
    cases=bound(p.value['train_case_inventory'])['cases']
    sequence=digest([{k:c[k] for k in ('scenario_seed','action_seed','slot')} for c in cases])
    result={}
    for row in starts:
        result[row['variant']]=dict(reference_actor_sha256=protocol.REFERENCE_SHA256,
            actor_role_sha256=row['roles'],critic_sha256=row['critic_sha256'],
            Actor_optimizer_sha256=row['actor_optimizer_sha256'],critic_optimizer_sha256=row['critic_optimizer_sha256'],
            initialization_random_stream_sha256=digest(dict(constructor='torch.manual_seed',seed=p.value['initialization_seed'])),
            optimizer_random_stream_sha256=digest(dict(constructor='np.random.default_rng',seed=p.value['optimizer_seed'])),
            scenario_sequence_sha256=sequence)
    require(result['raw']==result['regularized'],'actual paired initialization differs')
    return result


def pair_declaration(context,quality_report):
    """Translate independently replayed native facts into strict gate arithmetic."""
    p=context['plan'];collected=bound(quality_report['quality_result'])
    return dict(stream_id=p.value['stream_id'],checkpoint_id=f'final-{p.config.joint_step_budget}',
        joint_step_budget=p.config.joint_step_budget,config=asdict(p.config),start_joint_step=0,
        end_joint_steps={v:p.config.joint_step_budget for v in protocol.VARIANTS},initialization=initial_identity(context),
        quality_cases_sha256=quality_report['quality_cases_sha256'],
        quality={v:_quality_totals(collected['outcomes'],v) for v in protocol.VARIANTS},
        reference_quality=_quality_totals(collected['outcomes'],'reference'),
        programs={v:{r:_program_metrics(quality_report['program_metrics']['roles'][v][r]) for r in protocol.ROLES}
                  for v in protocol.VARIANTS},
        program_gradient_updates={v:context['receipt_audit'][v]['accepted_program_updates'] for v in protocol.VARIANTS})


def audit_attempts(attempts):
    from . import v642_long_training_assets as assets,v642_long_training_final_programs as final
    from . import v642_long_training_quality as quality
    require(type(attempts) is list and 0<len(attempts)<=len(protocol.BUDGETS),'explicit ordered development attempts required')
    sources=source_files();groups=[];inputs=[];common_starts={};selected=None
    for index,group in enumerate(attempts):
        budget=protocol.BUDGETS[index]
        require(type(group) is dict and set(group)=={'joint_step_budget','streams'}
            and type(group['joint_step_budget']) is int and group['joint_step_budget']==budget
            and set(group['streams'])==set(protocol.DEVELOPMENT_STREAMS),
            'each budget requires all three complete streams in registered order')
        contexts={};program_results={};checks={};declarations={};refs={}
        for stream in protocol.DEVELOPMENT_STREAMS:
            item=group['streams'][stream]
            require(type(item) is dict and set(item)=={'completed_assets','final_programs','quality_gate'},
                    'unknown development outcome schema')
            _asset(item['completed_assets']);_asset(item['final_programs'])
            asset_report=bound(item['completed_assets'])
            c=assets.finalization_context(item['completed_assets'],asset_report['reference_directory'])
            p=c['plan'];require(p.value['phase']=='development' and p.value['stream_id']==stream
                and p.config.joint_step_budget==budget,'attempt contains another stream, phase or budget')
            final_read=final._read_result(item['final_programs'],item['completed_assets'],require_passed=False)
            program_results[stream]=bound(item['final_programs'])['program_precheck_passed']
            contexts[stream]=c;refs[stream]=item
            start=initial_identity(c)['raw'];stable={k:v for k,v in start.items() if k!='scenario_sequence_sha256'}
            require(stream not in common_starts or common_starts[stream]==stable,
                    'higher budget did not restart the same Actor, critic, optimizers and random streams')
            common_starts[stream]=stable
            inputs.extend([item['completed_assets'],item['final_programs'],*c['input_asset_refs'],*final_read['inputs']])
        for key in ('initialization_random_stream_sha256','optimizer_random_stream_sha256','scenario_sequence_sha256'):
            require(len({initial_identity(c)['raw'][key] for c in contexts.values()})==3,
                    'development continuation streams are not independently registered')
        if not all(program_results.values()):
            require(all(refs[s]['quality_gate'] is None for s in protocol.DEVELOPMENT_STREAMS),
                    'failed group program precheck must not allocate a fresh quality cohort')
            pool_reports={}
            for stream in protocol.DEVELOPMENT_STREAMS:
                item=refs[stream];reference=bound(item['completed_assets'])['reference_directory']
                replay=final.audit_program_pool(item['final_programs'],item['completed_assets'],reference_directory=reference)
                require(replay['program_pool_passed'] is program_results[stream],'final failed-program evidence did not replay')
                pool_reports[stream]=replay;inputs.extend(replay['input_asset_refs'])
            group_passed=False
            detail=dict(scope='program_pool_failure_no_fresh_task_scores',program_pool_reports=pool_reports,
                        task_quality=None,passed=False)
        else:
            cohort=set();cohort_cases=set()
            for stream in protocol.DEVELOPMENT_STREAMS:
                item=refs[stream];gate_ref=item['quality_gate'];_asset(gate_ref);recorded=bound(gate_ref)
                replay=quality.audit(recorded['quality_result'],recorded['supervisor_exit'])
                require(replay==recorded and replay['completed_assets']==item['completed_assets']
                    and replay['seed_plan']==contexts[stream]['plan'].ref and replay['phase']=='development'
                    and replay['stream_id']==stream and replay['joint_step_budget']==budget,
                    'independent quality admission did not reproduce this exact training pair')
                cohort.add(digest(replay['cohort']));cohort_cases.add(replay['quality_cases_sha256'])
                declaration=pair_declaration(contexts[stream],replay)
                undefined=[]
                for variant in protocol.VARIANTS:
                    for role in protocol.ROLES:
                        for purpose in ('feedback','explanation'):
                            metric=replay['program_metrics']['roles'][variant][role]['metrics'][purpose]
                            if metric['kl'] is None:
                                require(metric['rows']==0 and metric['coverage']==0,
                                        'undefined KL without a genuinely empty qualified region')
                                undefined.append((variant,role,purpose))
                if undefined:
                    require(replay['development_pair_gate_passed'] is False,
                            'an empty qualified region cannot pass the development gate')
                    checked=dict(all_declared_gates_passed=False,undefined_KL_regions=undefined)
                else:
                    checked=protocol.validate_pair_declaration(declaration,budget=budget,phase='development')
                require(checked['all_declared_gates_passed'] is replay['development_pair_gate_passed'],
                        'native independent quality and strict development arithmetic disagree')
                checks[stream]=replay;declarations[stream]=declaration
                inputs.extend([gate_ref,*replay['input_asset_refs']])
            require(len(cohort)==len(cohort_cases)==1,'all three streams require the same predeclared 75-scene cohort')
            group_passed=all(r['development_pair_gate_passed'] for r in checks.values())
            detail=dict(scope='actual_independently_replayed_three_stream_quality',quality_reports=checks,
                        declarations=declarations,passed=group_passed)
        groups.append(dict(joint_step_budget=budget,streams=deepcopy(group['streams']),**detail))
        if group_passed:
            require(index==len(attempts)-1,'cannot continue searching after the first all-three passing budget')
            selected=budget
    unique={}
    for ref in inputs:
        _asset(ref)
        require(ref['path'] not in unique or unique[ref['path']]==ref,'one evidence path has conflicting hashes')
        unique[ref['path']]=ref
    require(source_files()==sources,'development admission source changed during independent replay')
    return dict(version=VERSION,attempts=deepcopy(attempts),source_files=sources,groups=groups,
        selected_joint_step_budget=selected,all_three_streams_passed=selected is not None,
        next_joint_step_budget=(None if selected is not None or len(attempts)==3 else protocol.BUDGETS[len(attempts)]),
        input_asset_refs=[unique[k] for k in sorted(unique)],
        formal_preparation_authorized=selected is not None,sealed_authority=False,
        model_acceptance_passed=False,pilot_opened=False,new_environment_resets=0,NN_updates=0)


def issue_budget_lock(attempts):
    report=audit_attempts(attempts)
    require(report['all_three_streams_passed'],'no budget lock: one or more development gates failed')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    target=OUTPUT/'budget-lock.json';claim=OUTPUT/'budget-lock.claim.json'
    write_new(claim,dict(version=VERSION,report_sha256=digest(report),single_use_even_if_interrupted=True,
                         model_acceptance_passed=False,pilot_opened=False));claim.chmod(0o444)
    write_new(target,dict(**report,claim=pointer(claim)));target.chmod(0o444)
    return pointer(target)


@dataclass(frozen=True,init=False)
class BudgetLock:
    value:dict
    ref:dict
    refs:tuple
    payload_sha256:str
    _token:object

    def __init__(self,*args,**kwargs):raise RuntimeError('read the actual independently replayed budget lock')

    def verify(self):
        require(self._token is _TOKEN and digest(self.value)==self.payload_sha256
            and self.value['source_files']==source_files() and self.value['all_three_streams_passed'] is True
            and self.value['formal_preparation_authorized'] is True
            and self.value['selected_joint_step_budget'] in protocol.BUDGETS
            and self.value['model_acceptance_passed'] is self.value['pilot_opened'] is self.value['sealed_authority'] is False,
            'budget lock changed or has another scope')
        for ref in self.refs:_asset(ref)
        return self


def read_budget_lock(ref):
    _asset(ref);value=bound(ref)
    require(ref==pointer(OUTPUT/'budget-lock.json') and value['version']==VERSION,
            'unknown or relocated development budget lock')
    replay=audit_attempts(value['attempts']);claim_ref=value['claim'];_asset(claim_ref)
    require(value==dict(**replay,claim=claim_ref) and replay['all_three_streams_passed'] is True
        and claim_ref==pointer(OUTPUT/'budget-lock.claim.json')
        and bound(claim_ref)==dict(version=VERSION,report_sha256=digest(replay),single_use_even_if_interrupted=True,
                                  model_acceptance_passed=False,pilot_opened=False),
        'budget lock is not the first independently verified all-three passing budget')
    token=object.__new__(BudgetLock)
    for key,item in dict(value=deepcopy(value),ref=deepcopy(ref),refs=(ref,claim_ref,*replay['input_asset_refs']),
                         payload_sha256=digest(value),_token=_TOKEN).items():object.__setattr__(token,key,item)
    return token.verify()
