"""Joint fixed-distribution/partition options, never execution actions.

Retain v636 OLD leaves and all already fitted v637 final candidate models.
Compare their fit-only confidence partitions BEFORE discarding zero-score
whole leaves. All new path certificates are exact zero-dual extensions.
"""
from copy import deepcopy
import numpy as np
from . import warehouse_v635_subtrees as build,warehouse_v635_search as choices,warehouse_v635_program as language
from . import warehouse_v632_partitions as sparse
from .warehouse_v630_certificate import extend
from .warehouse_v630_partition_proposals import allocate
from .warehouse_v626_search import encode,decode,replace
from .warehouse_v628_certificate import region,structure_info
from .warehouse_v622_program import plain,digest
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-fixed-leaf-distribution-and-confidence-joint-options.v6.38'


def compile_option(source,leaf,model,subtree):
    _,old_paths=encode(decode(source.structure));root=old_paths[leaf]
    structure,paths=encode(replace(decode(source.structure),root,subtree));structure=plain(structure)
    structure_info(structure,256);legal=build.guards.leaf_actions(structure,256)
    old_path=region(source.structure,leaf,256);old_legal=source.certificates()[leaf]
    ids=[i for i,path in enumerate(paths) if path[:len(root)]==root];models=[];evidence=[]
    for i in ids:
        path=region(structure,i,256);fitted,_=extend(model,old_path,path,old_legal,legal[i]);models.append(fitted)
        evidence.append(dict(path=path,legal=sorted(legal[i])))
    local,relative=encode(subtree);require([paths[i][len(root):] for i in ids]==relative,'joint option leaf order differs')
    return plain(dict(structure=local,models=models,paths=evidence,root=list(root),training_only_no_eligibility=True))


def run(source,outer,lanes,budget,candidate_report):
    source.validate();base=language.import_previous(source);_,paths=encode(decode(base.structure))
    risk=choices.metrics.risk_subsets(lanes['eligibility']['rows']);groups={};tables={};objects={};records=[]
    require(len(candidate_report['records'])==len(base.leaves),'candidate source report has incomplete leaves')
    for i,model in enumerate(base.leaves):
        budget.check();root=paths[i];parts,masks=choices.root_parts(base,root,lanes);repeats=[r[masks['eligibility']] for r in risk]
        old=build.original_option(base,root);oldscore=choices.option_score(old,parts,repeats);row=candidate_report['records'][i]
        require(row['leaf']==i and row['old_score']==oldscore,'candidate report is not from the uniform v636 source')
        frozen=row['frozen_reliable_E_or_core'];options={'old':old};proposals=[]
        if not frozen:
            outputs={'old':model}
            if 'candidate_model'in row:outputs['refit']=row['candidate_model']
            for label,payload in outputs.items():
                current=deepcopy(base);current.leaves[i]=deepcopy(payload);current.validate()
                proposal=sparse.propose(current,lanes['fit'],i,budget,oblique=True)
                proposals.append(dict(output=label,model_sha256=digest(language.payload(payload)),proposal=proposal))
                for tree in proposal['proposals']:
                    name=label+'-'+digest(tree)
                    options[name]=compile_option(base,i,payload,decode(tree))
        key=f'l{i:04d}';objects[key]=dict(root=list(root),options=options);groups[key]=[];tables[key]={}
        for name,option in options.items():
            score=choices.option_score(option,parts,repeats);groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
        records.append(dict(leaf=i,frozen_original_core_or_reliable_E=frozen,proposals=proposals))
        budget.checkpoint(dict(stage='joint_leaf_options',completed_leaves=len(records),total_leaves=len(base.leaves),option_count=sum(len(x['options']) for x in objects.values())))
    n=len(lanes['validation']['x']);candidates=[];best=None
    for weight in (0.,.5,1.,2.):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256);selected={tuple(objects[g]['root']):objects[g]['options'][name] for g,name in allocation['selected'].items()}
        program=build.assemble(base,selected);language.qualify(program,lanes);observed=language.measure(program,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*choices.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(program.leaves)))
        candidates.append(dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=observed,score=list(score)))
        if best is None or score>best[0]:best=(score,program,len(candidates)-1)
    final=best[1]
    for i,row in enumerate(candidate_report['records']):
        if row['frozen_reliable_E_or_core']:
            selected=candidates[best[2]]['allocation']['selected'][f'l{i:04d}'];require(selected=='old','frozen reliable leaf gained a new option')
    return final,plain(dict(version=VERSION,records=records,groups=groups,tables=tables,option_objects=objects,candidates=candidates,selected=best[2],
        reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],coefficient_fit_calls=0,proof_optimizer_calls=0,
        candidate_models_never_refitted=True,source_original_core_and_fit_reliable_E_unchanged=True,NN_updates=0,training_authority=False,pilot_opened=False))
