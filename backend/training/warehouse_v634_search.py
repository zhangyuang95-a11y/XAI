"""Fit-only union of entropy and high-precision confidence templates."""
import numpy as np
from . import warehouse_v634_partitions as precision
from . import warehouse_v632_partitions as legacy
from . import warehouse_v630_partition_search as old
from .warehouse_v630_partition_proposals import allocate
from .warehouse_v622_program import digest,plain
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-precision-template-union.v6.34'


def run(source,parts,budget,*,selective):
    base,originals=legacy.collapse_protected(source)
    pred={s:base.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
    for split,reference in pred.items():
        before=source.distributions(parts[split]['x'],parts[split]['m'])
        require(all(np.array_equal(x,y) for x,y in zip(reference[1:],before[1:])),'collapsing changed full q')
    signals={s:old.frame_signals(base,parts[s],p) for s,p in pred.items()};subsets=old.risk_subsets(parts['eligibility']['rows'])
    groups={};local={};records=[]
    for leaf in range(len(base.leaves)):
        budget.check();templates=[legacy.propose(base,parts['fit'],leaf,budget,oblique=True)]
        if selective:templates.extend(precision.propose(base,parts['fit'],leaf,budget,p) for p in precision.PRECISIONS)
        options={digest(t):t for t in [originals[leaf],*(t for proposal in templates for t in proposal['proposals'])]}
        g,d=old.local_table(base,leaf,[options[k] for k in sorted(options)],parts,pred,signals,subsets)
        groups[f'{leaf:04d}']=g;local[f'{leaf:04d}']=d
        records.append(dict(source_leaf=leaf,original_option_sha256=digest(originals[leaf]),templates=templates))
    n=len(parts['validation']['x']);candidates=[];best=None
    for weight in (0.,.5,1.,2.):
        budget.check()
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(
            o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256)
        p,_=old.assemble(base,{int(g):local[g][i]['structure'] for g,i in allocation['selected'].items()});old.qualify_current(p,parts)
        metrics=old.measure(p,parts['validation']);counts={k:sum(local[g][i]['totals'][k] for g,i in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*old.score_metrics(metrics,metrics['explanation']['whole_tree_kl'],len(p.leaves)))
        for split,reference in pred.items():
            actual=p.distributions(parts[split]['x'],parts[split]['m'])
            require(all(np.array_equal(x,y) for x,y in zip(actual[1:],reference[1:])),'new confidence predicates changed q')
        candidates.append(dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=metrics,score=list(score)))
        if best is None or score>best[0]:best=(score,p,len(candidates)-1)
    return best[1],plain(dict(version=VERSION,selective=selective,collapsed_base=base.to_dict(),
        original_subtrees={str(k):v for k,v in originals.items()},proposal_records=records,groups=groups,local=local,
        candidates=candidates,selected=best[2],reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],
        distributions_preserved=True,NN_updates=0,training_authority=False,pilot_opened=False))
