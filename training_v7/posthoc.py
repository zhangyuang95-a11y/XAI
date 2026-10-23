"""Read-only scientific diagnosis after a calibration has been closed.

No training, seed claims, environment resets, leaf refits or reopen authority.
The normalized trace view only adds the explicit trajectory=0 format field;
every original byte and CLOSED receipt remains independently hash-bound.
"""
import gzip
import json
from pathlib import Path
from .contracts import MethodSpec,digest,file_sha,write_new
from .domains import domain
from .evaluation_run import verify
from .program import Program
from .storage import source_files
from .training_audit import replay
from .workflow import registration_at


def audit_closed(root,name,output):
    root=Path(root).resolve();output=Path(output).resolve()
    closed=root/'CLOSED.json';closed_sha=file_sha(closed)
    reg_path,reg=registration_at(root)
    if reg['method_sha256']!=MethodSpec().sha256:raise ValueError('closed calibration method differs')
    working=Path(__file__).resolve().parents[1];current=source_files(working)
    if any(current.get(p)!=h for p,h in reg['source_files'].items()):
        raise ValueError('closed NN/physics/training source differs; reader extensions may only add files')
    extra={p:h for p,h in current.items() if p not in reg['source_files']}
    run=root/name/'calibration-0';plan=reg['calibration_plans'][name]
    original_path=run/'fresh-20k-quality/result.json';original_sha=file_sha(original_path)
    original=json.loads(original_path.read_text());adapter=domain(name)
    if adapter.spec.sha256!=plan['domain_sha256']:raise ValueError('closed physical configuration differs')
    output.mkdir(exist_ok=False,parents=True)
    bc_path=run/'bc/result.json';bc=json.loads(bc_path.read_text())
    training={};programs={};references=[]
    for variant in ('raw','regularized'):
        result_path=run/variant/variant/'result.json';actual=json.loads(result_path.read_text())
        if actual['joint_steps']!=20000:raise ValueError('post-20k diagnosis requires an actual 20k model')
        audited=replay(adapter,run/variant,plan,budget=20000,variant=variant,
            final_asset=original['assets'][variant],reference_role_sha256=bc['actor']['role_sha256'],
            sources=reg['source_files'],root=working,extension_sources=extra,require_learning_gates=False,
            output=output/f'{variant}-actual-training-replay.json')
        training[variant]=dict(result=dict(path=str(result_path),sha256=file_sha(result_path)),
            gates=audited['learning_gates'],joint_steps=audited['joint_steps'],
            replayed_role_actions=audited['replayed_role_actions'],
            program_feedback_frames=audited['actual_program_feedback_frames'],
            program_updates=audited['actual_program_updates'],
            program_gradient_norm_sum=audited['actual_program_gradient_norm_sum'],timing=actual['timing'])
        programs[variant]={}
        for role in adapter.spec.roles:
            path=run/variant/variant/'final-programs'/f'{role.name}.json'
            p=json.loads(path.read_text())['program']
            program=Program.from_dict(role,p,actor_sha256=original['assets'][variant]['role_sha256'][role.name])
            programs[variant][role.name]=program.as_dict()
            references.append(dict(path=str(path),sha256=file_sha(path)))
    traces=[]
    for ordinal,ref in enumerate(original['traces']):
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('closed physical trace changed')
        with gzip.open(ref['path'],'rt') as f:trace=json.load(f)
        trace['outcome']={'trajectory':0,**trace['outcome']}
        path=output/f'normalized-{ordinal:04d}.json.gz'
        with gzip.open(path,'xt') as f:json.dump(trace,f,separators=(',',':'),allow_nan=False)
        traces.append(dict(path=str(path),sha256=file_sha(path),original=ref))
    normalized=dict(version='three-domain-unified-finite-evaluation.v1',domain=name,phase='calibration',
        purpose='calibration_quality',formal_thresholds=False,quality_required=True,method_sha256=MethodSpec().sha256,
        plan_sha256=digest(plan),cases=[c for c in plan['cases'] if c['purpose']=='calibration_quality'],
        assets=original['assets'],rows={p:[{'trajectory':0,**r} for r in rows] for p,rows in original['rows'].items()},
        traces=traces,task_metrics=original['task_metrics'],reference_evaluated=True,programs=programs,
        programs_control_actions=False,human_participants=0,model_acceptance_passed=False,pilot_opened=False,
        original=dict(path=str(original_path),sha256=original_sha),
        normalization='add_explicit_zero_trajectory_only; binding reader follows DomainSpec role order')
    path=output/'normalized-quality.json';write_new(path,normalized)
    verified=verify(adapter,path,expected_cases=normalized['cases'],expected_plan=digest(plan))
    if verified['program_metrics']!=original['program_metrics']:
        raise ValueError('original program metrics do not match freshly recomputed NN frame evidence')
    write_new(output/'quality-verification.json',verified)
    if file_sha(closed)!=closed_sha or file_sha(original_path)!=original_sha:
        raise ValueError('closed receipt or original quality changed during diagnosis')
    distinct=all(original['assets']['raw']['role_sha256'][r.name]!=original['assets']['regularized']['role_sha256'][r.name]
                 for r in adapter.spec.roles)
    scientific=(distinct and verified['quality_passed'] and verified['programs_passed']
                and all(all(t['gates'].values()) for t in training.values()))
    result=dict(version='three-domain-unified-closed-calibration-diagnosis.v1',domain=name,
        candidate_closed=True,closed_receipt=dict(path=str(closed),sha256=closed_sha),
        registration=dict(path=str(reg_path),sha256=file_sha(reg_path)),original_quality=normalized['original'],
        program_assets=references,frozen_training_sources=reg['source_files'],additive_reader_sources=extra,
        task_metrics=verified['task_metrics'],program_metrics=verified['program_metrics'],
        quality_intervals=verified['quality_intervals'],program_descriptive=verified['program_descriptive'],
        training=training,raw_regularized_role_weights_distinct=distinct,
        scientific_calibration_requirements_met=bool(scientific),replayed_quality_role_actions=verified['replayed_role_actions'],
        normalization_is_not_new_evaluation=True,new_scientific_resets=0,new_NN_updates=0,
        reopened=False,formal_training_started=False,sealed_audit_started=False,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',result);return result
