"""Uniform six-unit confidence experiment; all method choice stays in old fit.

Each protected prefix is an exhaustive constant-payload group, not a new
controller. The original subtree is always a candidate; all leaf counts and
legality guards count against the unchanged 256/depth9 language budget.
"""
from copy import deepcopy
import numpy as np
from . import warehouse_v632_partitions as sparse
from . import warehouse_v630_partition_search as old
from .warehouse_v630_partition_proposals import allocate
from .warehouse_v622_program import digest, plain
from experiments.warehouse.v626_protocol import require

VERSION = 'warehouse-frozen-output-axis-oblique-comparison.v6.32'


def run(source, parts, budget, *, oblique):
    base, originals = sparse.collapse_protected(source)
    pred = {s:base.distributions(a['x'], a['m']) for s,a in parts.items() if s != 'calibration'}
    for s, values in pred.items():
        reference = source.distributions(parts[s]['x'], parts[s]['m'])
        require(all(np.array_equal(x,y) for x,y in zip(values[1:], reference[1:])),
                'protected collapse changed the complete surrogate distribution')
    signals = {s:old.frame_signals(base, parts[s], p) for s,p in pred.items()}
    subsets = old.risk_subsets(parts['eligibility']['rows'])
    groups, local, records = {}, {}, []
    for leaf in range(len(base.leaves)):
        budget.check()
        record = sparse.propose(base, parts['fit'], leaf, budget, oblique=oblique)
        options = {digest(t):t for t in [*record['proposals'], originals[leaf]]}
        options, details = old.local_table(base, leaf, [options[k] for k in sorted(options)],
                                          parts, pred, signals, subsets)
        key = f'{leaf:04d}'; groups[key], local[key] = options, details
        record['original_option_sha256'] = digest(originals[leaf]); records.append(record)
    n = len(parts['validation']['x']); candidates = []; best = None
    for weight in (0., .5, 1., 2.):
        budget.check()
        weighted = {g:[dict(id=o['id'], leaves=o['leaves'], score=[o['score'][0]/(.45*n)+
            weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)), *o['score']]) for o in options]
            for g,options in groups.items()}
        allocation = allocate(weighted, 256)
        refinements = {int(g):local[g][i]['structure'] for g,i in allocation['selected'].items()}
        program, _ = old.assemble(base, refinements)
        old.qualify_current(program, parts)
        metrics = old.measure(program, parts['validation'])
        counts = {k:sum(local[g][i]['totals'][k] for g,i in allocation['selected'].items())
                  for k in ('E','F','expected_F')}
        reliable = counts['E']/n >= .45 and counts['F']/n >= .45 and counts['expected_F']/n >= .40
        score = (int(reliable), *old.score_metrics(metrics, metrics['explanation']['whole_tree_kl'], len(program.leaves)))
        for s, reference in pred.items():
            current = program.distributions(parts[s]['x'], parts[s]['m'])
            require(all(np.array_equal(x,y) for x,y in zip(current[1:], reference[1:])),
                    'confidence search changed fixed complete surrogate outputs')
        candidates.append(dict(weight=weight, allocation=allocation, reliable_counts=counts,
                               reliable_selection_passed=reliable, metrics=metrics, score=list(score)))
        if best is None or score > best[0]:
            best = (score, program, len(candidates)-1)
    report = dict(version=VERSION, oblique=oblique, source_program_sha256=digest(source.to_dict()),
                  collapsed_base=base.to_dict(), original_subtrees={str(k):v for k,v in originals.items()},
                  groups=groups, local=local, proposal_records=records, candidates=candidates, selected=best[2],
                  reliable_selection_passed=candidates[best[2]]['reliable_selection_passed'],
                  distributions_preserved=True, NN_updates=0, training_authority=False, pilot_opened=False)
    return best[1], plain(report)
