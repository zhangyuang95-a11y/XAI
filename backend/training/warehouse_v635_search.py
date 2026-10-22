"""Finite whole-program choice over learned protected-free root options."""
from copy import deepcopy
import numpy as np
from . import warehouse_v635_program as language,warehouse_v635_subtrees as build
from . import warehouse_v630_partition_search as metrics
from .warehouse_v630_partition_proposals import allocate
from .warehouse_v626_search import decode,encode
from .warehouse_v622_program import plain,digest,_eligible
from .warehouse_v629_search import score_metrics
from scripts.diagnose_warehouse_v634_free_roots import free_roots
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-bounded-guard-root-option-search.v6.35'


def subset(a,take):
    out={k:np.asarray(a[k])[take] for k in ('x','m','p','sampled')}
    out['rows']=[row for row,yes in zip(a['rows'],take,strict=True) if yes];return out


def root_parts(source,root,parts):
    _,paths=encode(decode(source.structure));ids=[i for i,path in enumerate(paths) if path[:len(root)]==root]
    masks={s:np.isin(source.leaf_indices(a['x']),ids) for s,a in parts.items()}
    return {s:subset(a,masks[s]) for s,a in parts.items()},masks


def option_score(option,parts,repeated):
    predictions={s:build.local_predictions(option,a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
    class LogicalMap:
        def certificates(self):return tuple(frozenset(x['legal']) for x in option['paths'])
    signals={s:metrics.frame_signals(LogicalMap(),parts[s],p) for s,p in predictions.items()}
    total=dict(E=0,F=0,expected_F=0.);children=[]
    fit_counts=np.bincount(predictions['fit'][0],minlength=len(option['models']))
    for i in range(len(option['models'])):
        risk=predictions['eligibility'][0]==i;chosen=predictions['validation'][0]==i
        risk_metric=metrics.row_metrics(signals['eligibility'],risk);selected=metrics.row_metrics(signals['validation'],chosen)
        repeats=[metrics.row_metrics(signals['eligibility'],risk&mask) for mask in repeated];accepted={};passes={}
        for purpose,threshold,select_threshold in (('feedback',.93,.95),('explanation',.95,.97)):
            passes[purpose]=sum(_eligible(r,threshold) for r in repeats)
            accepted[purpose]=bool(fit_counts[i] and _eligible(risk_metric,threshold) and passes[purpose]>=27 and
                selected['rows']>0 and selected['agreement']>=select_threshold and selected['kl']<=.20)
            if accepted[purpose]:total['F' if purpose=='feedback' else 'E']+=selected['rows']
            if purpose=='feedback' and accepted[purpose]:total['expected_F']+=float(signals['validation']['expected_F'][chosen].sum())
        children.append(dict(leaf=i,fit_count=int(fit_counts[i]),risk=risk_metric,selection=selected,
                             repeated_passes=passes,reliable_selected=accepted))
    return plain(dict(leaves=len(option['models']),score=[total['E'],total['expected_F'],total['F']],totals=total,children=children))


def run(previous,outer,lanes,budget):
    """No outer eligibility/validation enters predicate or option selection."""
    previous.validate();roots,paths=free_roots(previous);source=language.import_previous(previous)
    train,_=build.internal_split(outer['fit']['rows']);risk_masks=metrics.risk_subsets(lanes['eligibility']['rows'])
    fixed=language.protection(source)['protected_descendants'];groups={};tables={};objects={};records=[]
    # Fixed current protected descendants retain the existing confidence
    # partition, coefficients and original parent outputs.
    for i in fixed:
        path=paths[i];parts,masks=root_parts(source,path,lanes)
        option=build.original_option(source,path);score=option_score(option,parts,[m[masks['eligibility']] for m in risk_masks])
        key=f'p{i:04d}';groups[key]=[dict(id='fixed',leaves=1,score=score['score'])];tables[key]={'fixed':score}
    for index,root in enumerate(roots):
        budget.check();key=f'r{index:04d}';parts,masks=root_parts(source,root,lanes)
        options={'old':build.original_option(source,root)};records_for_root=[]
        tr_parts,tr_mask=root_parts(source,root,{'fit':outer['fit']});a=outer['fit'];indices=np.flatnonzero(tr_mask['fit']&train)
        training={k:a[k][indices] for k in ('x','m','p')}
        first=options['old']['paths'][0]['path']['path'][:len(root)];known={4}
        for item in first:known.update(build.guards.evidence(item['weight'],item['bias'])[item['branch']+'_proven'])
        for purity in build.PRECISIONS:
            if 9-len(root)<1 or len(indices)<80:
                records_for_root.append(dict(precision_target=purity,skipped='no_remaining_depth_or_fit_exposure'));continue
            tree,record=build.learn_tree(training,known,9-len(root),budget,precision=purity,root=root)
            option=build.fit_option(source,root,tree,outer,budget);name=f'precision-{purity:.2f}'
            options[name]=option;record['option_id']=name;records_for_root.append(record)
        objects[key]=dict(root=list(root),options=options);groups[key]=[];tables[key]={}
        for name,option in options.items():
            score=option_score(option,parts,[mask[masks['eligibility']] for mask in risk_masks])
            groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
        records.append(dict(group=key,root=list(root),root_training_frames=len(indices),known_actions=sorted(known),templates=records_for_root))
        budget.checkpoint(dict(stage='root_options',completed_roots=len(records),total_roots=len(roots),records=records))
    n=len(lanes['validation']['x']);candidates=[];best=None
    for weight in (0.,.5,1.,2.):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256)
        selected={tuple(objects[g]['root']):objects[g]['options'][name] for g,name in allocation['selected'].items() if g in objects}
        program=build.assemble(source,selected);language.qualify(program,lanes);observed=language.measure(program,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*score_metrics(observed,observed['explanation']['whole_tree_kl'],len(program.leaves)))
        candidates.append(dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,
                               metrics=observed,score=list(score)))
        if best is None or score>best[0]:best=(score,program,len(candidates)-1)
    return best[1],plain(dict(version=VERSION,root_records=records,groups=groups,tables=tables,option_objects=objects,
        candidates=candidates,selected=best[2],reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],
        original_protected_outputs_unchanged=True,unprotected_outputs_may_change=True,NN_updates=0,training_authority=False,pilot_opened=False))
