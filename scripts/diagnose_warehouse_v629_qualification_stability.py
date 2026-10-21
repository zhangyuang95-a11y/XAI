#!/usr/bin/env python3
"""Fixed-program, fit-only whole-scene qualification stability diagnosis.

No new randomness, fit, LP, environment or NN update. Deterministic repeated
subsets are correlated descriptive diagnostics, not confidence intervals,
independent validations or a program/capacity selection rule.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v629_protocol as v
from backend.training.warehouse_v629_program import Program,from_parent,protected_indices
from backend.training.warehouse_v628_program import ParentProgram
from backend.training.warehouse_v622_program import _wilson
from backend.training.warehouse_v626_domain import transform
from backend.training.warehouse_v628_certificate import region
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import basic,same_metrics,eligible
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v629-fit-only-qualification-stability.v1'


def scene_subsets(cases,per_task,repeats=30):
    v.require(len({c['case_id'] for c in cases})==len(cases) and all(c['sampling_mode']=='sampled' for c in cases),
        'diagnostic pools must have unique sampled scenes')
    groups={t:[i for i,c in enumerate(cases) if c['task']==t] for t in (1,2,3)}
    v.require(all(len(g)>=per_task for g in groups.values()),'insufficient whole scenes per task')
    return [np.array([i for t in (1,2,3) for i in sorted(groups[t],key=lambda i:
        sha256(f'{VERSION}:{repeat}:{cases[i]["case_id"]}'.encode()).hexdigest())[:per_task]],int)
        for repeat in range(repeats)]


def aggregate(program,arrays,case_ids):
    pred=program.distributions(arrays['x'],arrays['m']);independent=forward(program,arrays['x'],arrays['m'])
    v.require(all(np.array_equal(x,y) for x,y in zip(pred,independent,strict=True)),'qualification diagnostic forward differs')
    route,z,q=pred;n=len(program.leaves);index={c:i for i,c in enumerate(case_ids)}
    scene=np.array([index[r['episode']] for r in arrays['rows']],int);key=scene*n+route
    proofs=program.certificates();raw=z.argmax(1);mask=arrays['m'];target=arrays['p']
    illegal=mask[np.arange(len(mask)),raw]!=1
    unproved=np.array([int(k) not in proofs[int(i)] for k,i in zip(raw,route,strict=True)])
    loss=np.sum(target*(np.log(np.maximum(target,np.float32(1e-8)))-np.log(np.maximum(q,np.float32(1e-8)))),axis=1)
    out={}
    for name,weight in [('frames',None),('correct',(q.argmax(1)==target.argmax(1)).astype(int)),('KL_sum',loss),
                        ('illegal',illegal.astype(int)),('unproved',unproved.astype(int))]:
        out[name]=np.bincount(key,weights=weight,minlength=len(case_ids)*n).reshape(len(case_ids),n)
    return out,pred


def leaf_metrics(stats,selected,leaf):
    counts=stats['frames'][selected,leaf];n=int(counts.sum());scenes=int(np.count_nonzero(counts))
    correct=int(stats['correct'][selected,leaf].sum());agreement=correct/n if n else 0.
    return dict(rows=n,agreement=agreement,agreement_wilson_lower_90=_wilson(agreement,n),
        kl=float(stats['KL_sum'][selected,leaf].sum()/n) if n else None,
        distinct_episode_count=scenes,distinct_scenario_seed_count=scenes,
        max_episode_fraction=float(counts.max()/n) if n else 0.,
        illegal_count=int(stats['illegal'][selected,leaf].sum()),path_uncertified_count=int(stats['unproved'][selected,leaf].sum()))


def sampled_fit(plan,receipt,fold,role):
    ids=plan['folds'][fold]['outer']['fit'];byid={c['case_id']:c for c in plan['cases']}
    cases=[byid[i] for i in ids if byid[i]['sampling_mode']=='sampled'];v.require(len(cases)==600,'expected 600 sampled fit scenes')
    v.require(all(len([c for c in cases if c['task']==t])==200 for t in (1,2,3)),'fit tasks are not balanced')
    index=list(v.old.ROLES).index(role);out={k:[] for k in ('x','m','p','sampled')};rows=[];refs=[]
    for case in cases:
        ref=receipt['cases'][case['case_id']]['arrays'];v.require(v.pointer(ref['path'])==ref,'sampled fit source changed')
        refs.append(ref)
        with np.load(ref['path'],allow_pickle=False) as z:
            for key in out:out[key].append(z[key][:,index].copy())
            rows.extend(dict(episode=case['case_id'],base_seed=str(case['base']),task=case['task']) for _ in range(len(z['x'])))
    result={k:np.concatenate(a) for k,a in out.items()};result['rows']=rows
    return cases,result,refs


def unit(program,arrays,cases):
    ids=[c['case_id'] for c in cases];stats,pred=aggregate(program,arrays,ids);all_scenes=np.arange(len(ids))
    subsets={n:scene_subsets(cases,n//3) for n in (180,300)};protected=protected_indices(program)
    leaf_rows=[];total_frames=int(stats['frames'].sum())
    for i in range(len(program.leaves)):
        all_metric=leaf_metrics(stats,all_scenes,i)
        support=program.leaves[i]['support'];u=transform(arrays['x'][pred[0]==i],support).astype(float)
        centered=u-u.mean(0) if len(u) else u
        singular=np.linalg.svd(centered,compute_uv=False) if centered.size else np.array([],float)
        v.require(np.isfinite(singular).all(),'nonfinite feature support singular values')
        rank=int(np.count_nonzero(singular>singular.max()*1e-7)) if np.any(singular) else 0
        # An independent frame-level implementation checks the aggregation.
        direct=basic(program,arrays,pred,{i},'explanation')
        same_metrics(all_metric,direct)
        task_metrics={str(t):leaf_metrics(stats,np.array([j for j,c in enumerate(cases) if c['task']==t]),i) for t in (1,2,3)}
        repeated={}
        for n,subsets_n in subsets.items():
            measures=[leaf_metrics(stats,s,i) for s in subsets_n]
            repeated[str(n)]=dict(repeats=len(measures),feedback_pass_count=sum(eligible(m,.93) for m in measures),
                explanation_pass_count=sum(eligible(m,.95) for m in measures),
                agreement_min=min(m['agreement'] for m in measures),agreement_max=max(m['agreement'] for m in measures),
                frames_min=min(m['rows'] for m in measures),frames_max=max(m['rows'] for m in measures))
        leaf_rows.append(dict(leaf=i,protected=i in protected,fit_frame_fraction=all_metric['rows']/total_frames,
            pooled_sampled_fit=all_metric,by_task=task_metrics,repeated_subsets=repeated,
            depth=len(region(program.structure,i,program.maximum_leaves)['path']),
            support_diagnostic=dict(indices=list(support),names=[FEATURE_NAMES[j] for j in support],
                centered_fit_rank_relative_1e_7=rank,singular_values=singular.tolist(),
                rank_is_empirical_not_a_global_feature_identity_proof=True)))
    summaries={}
    for n in (180,300):
        for purpose,threshold in [('feedback',.93),('explanation',.95)]:
            values=[]
            for selected in subsets[n]:
                accepted=[i for i in range(len(program.leaves)) if eligible(leaf_metrics(stats,selected,i),threshold)]
                # Coverage uses the common 600-scene fit frame distribution.
                values.append(float(stats['frames'][:,accepted].sum()/total_frames))
            summaries[f'{n}_{purpose}']=dict(min=min(values),mean=float(np.mean(values)),max=max(values))
    return dict(leaves=leaf_rows,descriptive_requalified_fit_coverage=summaries,fit_scenes=len(cases),fit_frames=total_frames,
        protected_fit_fraction=sum(row['fit_frame_fraction'] for row in leaf_rows if row['protected']),
        protected_below_95_pooled_fit_fraction=sum(row['fit_frame_fraction'] for row in leaf_rows
            if row['protected'] and row['pooled_sampled_fit']['agreement']<.95),
        data_overlap_with_program_fit_and_search=True,independent_validation=False)


def run(output):
    source_sha=v.file_sha(__file__)
    path,plan=v.checked_plan();old=v.bound(plan['previous_plan']);receipt=v.bound(plan['collection'])
    refs=[v.pointer(path),plan['previous_plan'],plan['collection']];rows=[]
    for fold in range(3):
        for role in v.old.ROLES:
            key=f'fold-{fold}-{role}';ref=old['parents'][key]
            parent=ParentProgram.from_dict(v.bound(ref),plan['actor_role_sha256'][role]);p=from_parent(parent,ref,64)
            cases,arrays,assets=sampled_fit(old,receipt,fold,role);refs.extend([ref,*assets])
            rows.append(dict(name='v626_parent',fold=fold,role=role,program=ref,metrics=unit(p,arrays,cases)))
            if fold==0 and role=='robot_1':
                for name in ('G64','G128'):
                    result_ref=v.pointer(v.OUTPUT/'jobs'/f'{name}-fold-0-robot_1'/'result.json');r=v.bound(result_ref)
                    v.require(r['status']=='completed' and v.bound(v.pointer(Path(result_ref['path']).with_name('completed.json')))
                        ==dict(result_sha256=result_ref['sha256']),'only settled fixed diagnostic programs may be used')
                    p=Program.from_dict(v.bound(r['program']),parent.actor_role_sha256);refs.extend([result_ref,r['program']])
                    rows.append(dict(name=name,fold=fold,role=role,program=r['program'],metrics=unit(p,arrays,cases)))
    for ref in refs:v.require(v.pointer(ref['path'])==ref,'diagnostic input changed')
    v.require(v.file_sha(__file__)==source_sha,'diagnostic source changed during measurement')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=output/'source.py';source.write_bytes(Path(__file__).read_bytes());source.chmod(0o444)
    result=dict(version=VERSION,source_sha256=source_sha,source_snapshot=v.pointer(source),scientific_source_sha256=plan['source_sha256'],inputs=refs,
        units=rows,whole_scene_subsets=[180,300],repeats=30,task_balanced=True,
        subset_selection='fixed_SHA256_ranking_of_repeat_and_original_scene_ID_no_RNG',
        scope='fit_only_correlated_in_sample_diagnostic_not_method_selection_or_admission',
        environment_resets=0,NN_updates=0,solver_calls=0,registered_candidate_changed=False,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    return v.pointer(output/'result.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.output),ensure_ascii=False))
