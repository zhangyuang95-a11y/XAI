"""High-precision joint fixed-distribution/partition union; no execution actions.

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

VERSION='warehouse-fixed-leaf-distribution-and-confidence-joint-options.v6.39'


from .warehouse_v638_joint_options import compile_option
from . import warehouse_v634_partitions as precision


def merge_proposals(previous,extras):
    require([t.get('precision_target') for t in extras]==[.98,.99],'need both registered high-precision templates')
    templates=[deepcopy(previous),*deepcopy(extras)]
    for t in templates:
        require(t['source_leaf']==previous['source_leaf'] and t['fit_frames']==previous['fit_frames'],'confidence template source/exposure mismatch')
    options={digest(tree):tree for t in templates for tree in t['proposals']}
    return plain(dict(source_leaf=previous['source_leaf'],fit_frames=previous['fit_frames'],templates=templates,
        proposals=[options[k] for k in sorted(options)],entropy_template_reused=True,classifier_outputs_discarded=True,execution_actions_produced=False))


def run(source,outer,lanes,budget,candidate_report,parent_report):
    source.validate();base=language.import_previous(source);_,paths=encode(decode(base.structure))
    risk=choices.metrics.risk_subsets(lanes['eligibility']['rows']);groups={};tables={};objects={};records=[]
    require(len(candidate_report['records'])==len(base.leaves),'candidate source report has incomplete leaves')
    for i,model in enumerate(base.leaves):
        budget.check();root=paths[i];parts,masks=choices.root_parts(base,root,lanes);repeats=[r[masks['eligibility']] for r in risk]
        old=build.original_option(base,root);oldscore=choices.option_score(old,parts,repeats);row=candidate_report['records'][i]
        require(row['leaf']==i and row['old_score']==oldscore,'candidate report is not from the uniform v636 source')
        key=f'l{i:04d}';parent=parent_report['option_objects'][key]
        require(parent['root']==list(root) and parent['options']['old']==old,'v638 source root or old option changed')
        frozen=row['frozen_reliable_E_or_core'];options=deepcopy(parent['options']);proposals=[]
        if not frozen:
            outputs={'old':model}
            if 'candidate_model'in row:outputs['refit']=row['candidate_model']
            for label,payload in outputs.items():
                current=deepcopy(base);current.leaves[i]=deepcopy(payload);current.validate()
                entropy=next(p for p in parent_report['records'][i]['proposals'] if p['output']==label)
                require(entropy['model_sha256']==digest(language.payload(payload)),'fixed v638 proposal model changed')
                extras=[precision.propose(current,lanes['fit'],i,budget,p) for p in (.98,.99)]
                proposal=merge_proposals(entropy['proposal'],extras)
                proposals.append(dict(output=label,model_sha256=digest(language.payload(payload)),proposal=proposal))
                for tree in proposal['proposals']:
                    name=label+'-'+digest(tree)
                    options[name]=compile_option(base,i,payload,decode(tree))
        require(set(parent['options'])<=set(options),'union lost registered v638 options')
        options=dict(sorted(options.items()))
        objects[key]=dict(root=list(root),options=options);groups[key]=[];tables[key]={}
        for name,option in options.items():
            score=choices.option_score(option,parts,repeats)
            if name in parent['options']:require(score==parent_report['tables'][key][name],'inherited v638 finite table changed')
            groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
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
        reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],coefficient_fit_calls=0,proof_optimizer_calls=0,inherited_all_v638_options=True,new_entropy_fits=0,
        candidate_models_never_refitted=True,source_original_core_and_fit_reliable_E_unchanged=True,NN_updates=0,training_authority=False,pilot_opened=False))
