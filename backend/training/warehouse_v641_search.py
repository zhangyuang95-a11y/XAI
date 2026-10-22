"""Rebuild only subtrees disjoint from original core and reliable E leaves."""
import numpy as np
from . import warehouse_v641_contexts as context,warehouse_v635_subtrees as build,warehouse_v635_search as old,warehouse_v635_program as language
from .warehouse_v626_search import encode,decode
from .warehouse_v622_program import plain
from .warehouse_v630_partition_proposals import allocate
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-frozen-free-margin-context-rebuild.v6.41'


def scan(source,lanes):
    _,paths=encode(decode(source.structure));risk=old.metrics.risk_subsets(lanes['eligibility']['rows'])
    core=set(language.protection(source)['protected_descendants']);fixed={};scores={}
    for i,path in enumerate(paths):
        parts,masks=old.root_parts(source,path,lanes);option=build.original_option(source,path)
        score=old.option_score(option,parts,[m[masks['eligibility']] for m in risk]);scores[i]=score
        if i in core or score['children'][0]['reliable_selected']['explanation']:fixed[i]=dict(path=path,option=option,score=score)
    frozen=set(fixed);tree=decode(source.structure);lookup={p:i for i,p in enumerate(paths)};memo={}
    def leaves(node,path):
        ids=[lookup[path]] if node is None else leaves(node.left,path+(0,))+leaves(node.right,path+(1,));memo[path]=ids;return ids
    leaves(tree,());roots=[]
    def walk(node,path):
        if not frozen.intersection(memo[path]):roots.append(path);return
        if node is not None:walk(node.left,path+(0,));walk(node.right,path+(1,))
    walk(tree,());return fixed,roots,paths,risk


def run(previous,outer,lanes,budget):
    previous.validate();source=language.import_previous(previous);fixed,roots,paths,risk=scan(source,lanes)
    tr,_=build.internal_split(outer['fit']['rows']);groups={};tables={};objects={};records=[]
    for i,item in fixed.items():
        key=f'p{i:04d}';score=item['score'];groups[key]=[dict(id='fixed',leaves=1,score=score['score'])];tables[key]={'fixed':score}
    for index,root in enumerate(roots):
        budget.check();key=f'r{index:04d}';parts,masks=old.root_parts(source,root,lanes)
        options={'old':build.original_option(source,root)};_,takes=old.root_parts(source,root,{'fit':outer['fit']})
        a=outer['fit'];idx=np.flatnonzero(takes['fit']&tr);training={k:a[k][idx] for k in ('x','m','p')}
        prefix=options['old']['paths'][0]['path']['path'][:len(root)];known={4}
        for item in prefix:known.update(build.guards.evidence(item['weight'],item['bias'])[item['branch']+'_proven'])
        bank=context.borrowed_bank(source,root,training['x']) if len(idx) else [];templates=[]
        for depth in context.CONTEXT_DEPTHS:
            if 9-len(root)<1 or len(idx)<80:
                templates.append(dict(context_depth=depth,skipped='no_remaining_depth_or_minimum80_fit_frames'));continue
            tree,record=context.learn_tree(training,known,9-len(root),budget,context_depth=depth,root=root,bank=bank)
            option=build.fit_option(source,root,tree,outer,budget);name=f'context-{depth}'
            options[name]=option;record['option_id']=name;templates.append(record)
        objects[key]=dict(root=list(root),options=dict(sorted(options.items())));groups[key]=[];tables[key]={}
        for name,option in objects[key]['options'].items():
            score=old.option_score(option,parts,[m[masks['eligibility']] for m in risk]);groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
        records.append(dict(group=key,root=list(root),training_frames=len(idx),known_actions=sorted(known),bank=bank,templates=templates))
        budget.checkpoint(dict(stage='frozen_free_contexts',completed_roots=len(records),total_roots=len(roots)))
    n=len(lanes['validation']['x']);candidates=[];best=None
    for weight in (0.,.5,1.,2.):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256);selected={tuple(objects[g]['root']):objects[g]['options'][name] for g,name in allocation['selected'].items() if g in objects}
        out=build.assemble(source,selected);require(len(out.leaves)==allocation['leaves'],'compiled tree differs from allocated leaf count')
        language.qualify(out,lanes);observed=language.measure(out,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*old.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(out.leaves)))
        candidates.append(dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=observed,score=list(score)))
        if best is None or score>best[0]:best=(score,out,len(candidates)-1)
    return best[1],plain(dict(version=VERSION,frozen_leaves=sorted(fixed),root_records=records,groups=groups,tables=tables,option_objects=objects,
        candidates=candidates,selected=best[2],reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],
        original_core_and_current_reliable_E_unchanged=True,NN_updates=0,training_authority=False,pilot_opened=False))
