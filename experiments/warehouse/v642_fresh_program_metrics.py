"""Score fixed final-NN programs on fresh, already NN-replayed trajectories.

Eligibility sets are held fixed. No fitting/calibration/qualification routine,
environment or execution controller is called from this module.
"""
import numpy as np
from .v626_protocol import require,pointer,bound
from .v642_calibration_quality import validate_cases
from backend.training.warehouse_v642_program import Program,measure
from backend.training.warehouse_v629_online_evidence import gate
from backend.training.warehouse_v6_actor import FEATURE_DIM
from backend.training.warehouse_v622_program import MASK_INDICES
from scripts.audit_warehouse_v642_program import forward

VERSION='warehouse-v642-frozen-program-fresh-calibration-metrics.v1'
ROLES=('robot_1','robot_2')


def evaluate(program_manifest,actor_role_sha256,entries,cases):
    validate_cases(cases);byid={c['case_id']:c for c in cases};selected={};seen=set()
    for entry in entries:
        require(entry['variant'] in ('reference','raw','regularized'),'unknown fresh policy variant')
        if entry['variant']=='reference':continue
        key=(entry['variant'],entry['case']['case_id'])
        require(key not in seen and key[1] in byid and entry['case']==byid[key[1]],'fresh program scenes are duplicated or differently bound')
        seen.add(key);selected[key]=entry
    require(seen=={(v,c['case_id']) for v in ('raw','regularized') for c in cases},'both final policies must finish every fresh scene')
    require(set(program_manifest['programs'])=={'raw','regularized'},'raw and regularized final programs required')
    reports={};assets=[]
    for variant in ('raw','regularized'):
        parts={role:{k:[] for k in ('x','m','p','sampled','rows')} for role in ROLES}
        for case in cases:
            entry=selected[(variant,case['case_id'])];ref=entry['arrays']
            require(pointer(ref['path'])==ref,'fresh NN arrays changed')
            with np.load(ref['path'],allow_pickle=False) as z:
                require(set(z.files)=={'x','m','p','sampled'},'unknown fresh NN array schema')
                a={key:z[key].copy() for key in z.files}
            n=len(a['x']);require(0<n<=120 and a['x'].shape==(n,2,FEATURE_DIM)
                and a['m'].shape==a['p'].shape==(n,2,5) and a['sampled'].shape==(n,2)
                and all(a[k].dtype==np.float32 and np.isfinite(a[k]).all() for k in ('x','m','p'))
                and a['sampled'].dtype==np.int64 and np.isin(a['m'],(0,1)).all() and (a['m'][...,4]==1).all()
                and np.array_equal(a['x'][...,MASK_INDICES],a['m'])
                and ((a['p']>=0)&(a['p']<=1)).all() and np.allclose(a['p'].sum(-1),1.,atol=2e-6,rtol=0.)
                and (a['p'][a['m']==0]==0).all() and ((a['sampled']>=0)&(a['sampled']<5)).all()
                and (np.take_along_axis(a['m'],a['sampled'][...,None],axis=-1)==1).all(), 'invalid complete masked-NN physical arrays')
            rows=[dict(episode=case['case_id'],base_seed=str(case['base']),task=case['task'],frame=i) for i in range(n)]
            for i,role in enumerate(ROLES):
                for key in ('x','m','p','sampled'):parts[role][key].append(a[key][:,i])
                parts[role]['rows'].extend(rows)
            assets.append(ref)
        require(set(program_manifest['programs'][variant])==set(ROLES),'both final actor program roles required')
        reports[variant]={}
        for role in ROLES:
            ref=program_manifest['programs'][variant][role]
            program=Program.from_dict(bound(ref),actor_role_sha256[variant][role]);program.validate()
            require(program.role==role and program.target_actor_role_sha256==actor_role_sha256[variant][role],
                'fresh programs must bind the actual final NN role')
            feedback,explanation=program.feedback_leaves,program.explanation_leaves
            a={key:np.concatenate(values,axis=0) if key!='rows' else values for key,values in parts[role].items()}
            require(all(np.array_equal(x,y) for x,y in zip(program.distributions(a['x'],a['m']),forward(program,a['x'],a['m']),strict=True)),
                'independent fresh-program routing/distribution differs')
            metrics=measure(program,a);verdict=gate(metrics)
            require(program.feedback_leaves==feedback and program.explanation_leaves==explanation
                and program.to_dict()==bound(ref),'fresh scoring changed the frozen qualification/program')
            reports[variant][role]=dict(program=ref,target_actor_role_sha256=program.target_actor_role_sha256,
                metrics=metrics,gate=verdict,physical_frames=len(a['x']),independent_scenes=len(cases))
            assets.append(ref)
    for ref in assets:require(pointer(ref['path'])==ref,'fresh program metric inputs changed during scoring')
    return dict(version=VERSION,roles=reports,passed=all(r['gate']['passed'] for roles in reports.values() for r in roles.values()),
        scope='fixed_final_programs_on_fresh_calibration_frames_requires_bound_NN_collection_replay',
        program_fit_calls=0,qualification_changes=0,NN_updates=0,new_environment_resets=0,
        model_acceptance_passed=False,pilot_opened=False)
