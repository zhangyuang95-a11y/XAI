"""Fit-only stable confidence partitions with frozen surrogate outputs.

No collection or NN training. A registered runner must bind all sources,
whole-scene partitions, budgets and a single-use output before calling this.
Final qualification/validation is separate from this fit-only selection.
"""
from copy import deepcopy
from hashlib import sha256
import numpy as np
from . import warehouse_v630_partition_proposals as proposals
from .warehouse_v630_program import Program,from_previous
from .warehouse_v630_certificate import extend
from .warehouse_v626_search import Node,decode,encode,replace
from .warehouse_v628_certificate import region,structure_info
from .warehouse_v628_program import qualify,measure
from .warehouse_v629_search import score_metrics
from .warehouse_v622_program import _route,_wilson,digest,plain,_eligible
from experiments.warehouse import v630_semantic_protection as semantic
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-stable-frozen-output-partition-search.v1'


def lane_parts(arrays,cases):
    """600 sampled outer.fit scenes -> 300 train / 150 risk / 150 select."""
    require(len(cases)==600 and len({c['case_id'] for c in cases})==600
        and all(c['sampling_mode']=='sampled' for c in cases),'confidence search needs 600 unique sampled fit scenes')
    lanes={s:set() for s in ('fit','eligibility','validation')}
    for task in (1,2,3):
        ids=sorted(c['case_id'] for c in cases if c['task']==task);require(len(ids)==200,'confidence tasks must be balanced')
        lanes['fit'].update(ids[:100]);lanes['eligibility'].update(ids[100:150]);lanes['validation'].update(ids[150:])
    ids={str(r['episode']) for r in arrays['rows']}
    require(ids=={c['case_id'] for c in cases},'frame/case inventory differs from sampled fit scenes')
    out={}
    for name,episodes in {**lanes,'calibration':set()}.items():
        mask=np.array([str(r['episode']) in episodes for r in arrays['rows']],bool)
        out[name]={k:np.asarray(arrays[k])[mask] for k in ('x','m','p','sampled')}
        out[name]['rows']=[r for r,keep in zip(arrays['rows'],mask,strict=True) if keep]
    return out


def risk_subsets(rows):
    episodes={}
    for row in rows:
        episode=str(row['episode']);task=row['task']
        require(episode not in episodes or episodes[episode]==task,'one risk scene has multiple tasks')
        episodes[episode]=task
    require(len(episodes)==150 and all(sum(t==task for t in episodes.values())==50 for task in (1,2,3)),
        'stability risk lane needs 50 whole scenes per task')
    result=[]
    for repeat in range(30):
        selected=set()
        for task in (1,2,3):
            ids=[i for i,t in episodes.items() if t==task]
            selected.update(sorted(ids,key=lambda i:sha256(f'{VERSION}:{repeat}:{i}'.encode()).hexdigest())[:30])
        result.append(np.array([str(row['episode']) in selected for row in rows],bool))
    return result


def frame_signals(program,arrays,pred):
    route,z,q=pred;p=arrays['p'];top=q.argmax(1);nn=p.argmax(1);raw=z.argmax(1);proofs=program.certificates()
    return dict(correct=top==nn,illegal=arrays['m'][np.arange(len(raw)),raw]==0,
        unproved=np.array([int(a) not in proofs[int(i)] for a,i in zip(raw,route,strict=True)]),
        losses=np.sum(p*(np.log(np.maximum(p,np.float32(1e-8)))-np.log(np.maximum(q,np.float32(1e-8)))),axis=1),
        expected_F=p[np.arange(len(p)),top]*(top==nn),
        episodes=np.array([str(r['episode']) for r in arrays['rows']]),scenes=np.array([str(r['base_seed']) for r in arrays['rows']]))


def row_metrics(signals,chosen):
    n=int(chosen.sum());_,counts=np.unique(signals['episodes'][chosen],return_counts=True)
    agreement=float(np.mean(signals['correct'][chosen])) if n else 0.
    return dict(rows=n,agreement=agreement,agreement_wilson_lower_90=_wilson(agreement,n),
        kl=float(signals['losses'][chosen].mean()) if n else None,illegal_count=int(np.sum(chosen&signals['illegal'])),
        path_uncertified_count=int(np.sum(chosen&signals['unproved'])),distinct_episode_count=len(counts),
        distinct_scenario_seed_count=len(np.unique(signals['scenes'][chosen])),
        max_episode_fraction=float(counts.max()/n) if n else 0.)


def local_table(program,source_leaf,options,parts,predictions,signals,risk_masks):
    """Freeze score table before DP; all targets come from the fit-only lanes."""
    output=[];details={}
    parent={s:predictions[s][0]==source_leaf for s in ('eligibility','validation')}
    local_x={s:parts[s]['x'][parent[s]] for s in parent}
    local_signals={s:{k:value[parent[s]] for k,value in signals[s].items()} for s in parent}
    local_risk_masks=[mask[parent['eligibility']] for mask in risk_masks]
    for option in options:
        t,info=structure_info(option,64);key=digest(plain(t));routes={}
        for name in ('eligibility','validation'):
            routes[name]=_route(t,local_x[name],info['depth'])
        total=dict(E=0,F=0,expected_F=0.)
        children=[]
        for child in range(t['leaf_count']):
            risk_take=routes['eligibility']==child;score_take=routes['validation']==child
            risk=row_metrics(local_signals['eligibility'],risk_take)
            score=row_metrics(local_signals['validation'],score_take)
            repeated=[row_metrics(local_signals['eligibility'],risk_take&mask) for mask in local_risk_masks]
            selected={}
            for purpose,threshold,score_threshold in [('feedback',.93,.95),('explanation',.95,.97)]:
                passes=sum(_eligible(r,threshold) for r in repeated)
                selected[purpose]=bool(_eligible(risk,threshold) and passes>=27
                    and score['rows']>0 and score['agreement']>=score_threshold and score['kl']<=.20)
                if selected[purpose]:
                    total['F' if purpose=='feedback' else 'E']+=score['rows']
                if purpose=='feedback' and selected[purpose]:
                    total['expected_F']+=float(local_signals['validation']['expected_F'][score_take].sum())
            children.append(dict(leaf=child,risk=risk,selection=score,
                risk_feedback_passes=sum(_eligible(r,.93) for r in repeated),
                risk_explanation_passes=sum(_eligible(r,.95) for r in repeated),reliable_selected=selected))
        output.append(dict(id=key,leaves=int(t['leaf_count']),score=[total['E'],total['expected_F'],total['F']]))
        details[key]=dict(structure=plain(t),children=children,totals=total)
    return output,details


def assemble(source,partitions):
    """Compose disjoint supplied refinements, certify once, inherit no eligibility."""
    source.validate();tree=decode(source.structure);_,old_paths=encode(tree)
    for leaf,subtree in sorted(partitions.items()):tree=replace(tree,old_paths[leaf],decode(subtree))
    structure,new_paths=encode(tree);structure=plain(structure);structure_info(structure,256)
    out=Program(source.role,source.actor_role_sha256,structure,[],'C3',256,protection=deepcopy(source.protection))
    new_legal=out.certificates();old_legal=source.certificates();derivations=[]
    for i,path in enumerate(new_paths):
        j=next(j for j,prefix in enumerate(old_paths) if path[:len(prefix)]==prefix)
        leaf,proof=extend(source.leaves[j],region(source.structure,j,source.maximum_leaves),region(structure,i,256),old_legal[j],new_legal[i])
        out.leaves.append(leaf);derivations.append(dict(source_leaf=j,target_leaf=i,proof=proof))
    out.validate();return out,derivations


def qualify_current(program,parts):
    qualify(program,parts)
    program.report['protected_refit_and_calibration_skipped']=semantic.verify(program)['protected_descendants']
    program.report['qualification_only_no_leaf_fitting']=True
    return program.validate()


def run(source,parts,budget,*,scope):
    require(scope in ('control','protected','all'),'unknown registered confidence scope')
    source.validate();base=from_previous(source);base.maximum_leaves=256;base.validate()
    require({len({str(r['episode']) for r in parts[s]['rows']}) for s in ('eligibility','validation')}=={150}
        and len({str(r['episode']) for r in parts['fit']['rows']})==300,'incorrect confidence lane sizes')
    initial=qualify_current(deepcopy(base),parts)
    if scope=='control':return initial,dict(version=VERSION,scope=scope,mandatory_complete=True,source_leaf_count=len(source.leaves),
        inspected_leaves=[],candidates=[],coefficient_updates=0,solver_calls=0,all_distributions_preserved=True)
    allowed=set(semantic.verify(base)['protected_descendants']) if scope=='protected' else set(range(len(base.leaves)))
    predictions={s:base.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
    signals={s:frame_signals(base,parts[s],pred) for s,pred in predictions.items()}
    subsets=risk_subsets(parts['eligibility']['rows']);groups={};local={};checked=[];proposal_records=[]
    for leaf in range(len(base.leaves)):
        budget.check()
        proposal=proposals.propose(base,parts['fit'],leaf,budget) if leaf in allowed else dict(source_leaf=leaf,
            proposals=[plain(encode(None)[0])],fit_called=False,reason='outside_registered_scope')
        options,details=local_table(base,leaf,proposal['proposals'],parts,predictions,signals,subsets)
        group=f'{leaf:04d}';groups[group]=options;local[group]=details;checked.append(leaf);proposal_records.append(proposal)
        budget.checkpoint(dict(stage='local_tables',scope=scope,inspected_leaves=checked,groups=groups,local=local,proposal_records=proposal_records))
    n=len(parts['validation']['rows']);candidates=[];best=None
    for weight in (0.,.5,1.,2.):
        budget.check();weighted={}
        for group,options in groups.items():
            weighted[group]=[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)
                +weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options]
        allocation=proposals.allocate(weighted,256)
        partitions={int(group):local[group][option]['structure'] for group,option in allocation['selected'].items()}
        program,derivation=assemble(base,partitions);qualify_current(program,parts)
        metrics=measure(program,parts['validation'])
        reliable={key:sum(local[group][option]['totals'][key] for group,option in allocation['selected'].items())
            for key in ('E','F','expected_F')}
        reliable_pass=reliable['E']/n>=.45 and reliable['F']/n>=.45 and reliable['expected_F']/n>=.40
        for split,a in parts.items():
            if split=='calibration':continue
            actual=program.distributions(a['x'],a['m']);reference=predictions[split]
            require(np.array_equal(actual[1],reference[1]) and np.array_equal(actual[2],reference[2]),
                'confidence partition changed frozen surrogate output')
        whole_KL=metrics['explanation']['whole_tree_kl']
        score=(int(reliable_pass),*score_metrics(metrics,whole_KL,len(program.leaves)))
        record=dict(weight=weight,allocation=allocation,reliable_selection_counts=reliable,
            reliable_selection_passed=reliable_pass,metrics=metrics,score=list(score),derivations=derivation)
        candidates.append(record)
        if best is None or score>best[0]:best=(score,program,len(candidates)-1)
        budget.checkpoint(dict(stage='allocation',scope=scope,inspected_leaves=checked,groups=groups,local=local,
            proposal_records=proposal_records,candidates=candidates,best_program=best[1].to_dict(),best_candidate=best[2]))
    return best[1],dict(version=VERSION,scope=scope,mandatory_complete=len(checked)==len(base.leaves),
        source_leaf_count=len(source.leaves),inspected_leaves=checked,proposal_records=proposal_records,
        fixed_option_tables_sha256=digest(groups),candidates=candidates,best_candidate=best[2],
        option_groups=groups,local_details=local,
        reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],
        coefficient_updates=0,all_distributions_preserved=True,
        optimizer_scope='one_CART_per_eligible_source_leaf_then_fixed_prunings_and_four_additive_allocations',
        nonlinear_global_optimality_gap='unknown')
