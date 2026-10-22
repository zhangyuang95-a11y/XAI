#!/usr/bin/env python3
"""Inventory maximal subtrees disjoint from every immutable protected prefix.

Structural roots are enumerated without NN labels. Old validation exposure
is descriptive only, never an admission result or a root selection rule.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v630_certificate import path_actions
from backend.training.warehouse_v626_search import decode,encode
from backend.training.warehouse_v628_certificate import region
from experiments.warehouse.v630_semantic_protection import verify
from scripts.run_warehouse_v625 import write_new


def free_roots(program):
    protected=[tuple(0 if x['branch']=='left' else 1 for x in e['path']) for e in program.protection['entries']]
    roots=[]
    def walk(node,path):
        if any(path[:len(p)]==p for p in protected):return
        if any(p[:len(path)]==path for p in protected):
            v.require(node is not None,'protected prefix exceeds the current tree')
            walk(node.left,path+(0,));walk(node.right,path+(1,))
        else:roots.append(path)
    tree=decode(program.structure);walk(tree,());_,paths=encode(tree)
    expected=set(range(len(program.leaves)))-set(verify(program)['protected_descendants'])
    covered={i for i,p in enumerate(paths) if any(p[:len(a)]==a for a in roots)}
    v.require(covered==expected and all(not a[:len(b)]==b for a in roots for b in roots if a!=b),
              'free roots overlap or omit an unprotected leaf')
    return roots,paths


def run(summary_path,output):
    summaryref=v.pointer(summary_path);summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and len(summary['units'])==6,'need one uniform six-program baseline')
    _,old=v.checked_plan();dp=v.bound(v.bound(old['previous_plan'])['previous_plan'])
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    config=dict(source=source,summary=summaryref,inputs=summary['units'],root_rule='maximal_prefix_disjoint_from_all_original_protected_prefixes',
                NN_labels_not_used_to_choose_roots=True,validation_is_spent_diagnostic_only=True,new_fits=0)
    write_new(output/'plan.json',config);(output/'plan.json').chmod(0o444);rows=[]
    for ref in summary['units']:
        r=v.bound(ref);spec=r['spec'];program=Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']])
        roots,paths=free_roots(program);a=data.load_parts(dp,spec['fold'],spec['role'])['validation'];route=program.leaf_indices(a['x']);nn=a['p'].argmax(1)
        entries=[]
        for anchor in roots:
            leaves=[i for i,p in enumerate(paths) if p[:len(anchor)]==anchor];take=np.isin(route,leaves)
            prefix=region(program.structure,leaves[0],256)['path'][:len(anchor)]
            remaining=9-len(anchor);entries.append(dict(path=list(anchor),depth=len(anchor),remaining_depth=remaining,
                current_descendant_leaves=leaves,current_relative_depth=max(len(paths[i])-len(anchor) for i in leaves),
                inherited_path_legal_actions=sorted(path_actions(dict(path=prefix))),
                validation_frames=int(take.sum()),validation_fraction=float(take.mean()),
                validation_NN_top_counts=np.bincount(nn[take],minlength=5).tolist(),
                optimistic_action_list_rule_slots=min(4,remaining),all_protected_prefixes_and_outputs_outside_anchor=True))
        row=dict(job=spec['id'],input=ref,program=r['program'],roots=entries,unprotected_leaf_count=sum(len(e['current_descendant_leaves']) for e in entries),
            unprotected_validation_fraction=sum(e['validation_fraction'] for e in entries),
            fraction_with_at_least_one_rule_slot=sum(e['validation_fraction'] for e in entries if e['remaining_depth']>=1),
            fraction_with_at_least_three_rule_slots=sum(e['validation_fraction'] for e in entries if e['remaining_depth']>=3),
            fraction_with_four_rule_slots=sum(e['validation_fraction'] for e in entries if e['remaining_depth']>=4))
        rows.append(row);print(spec['id'],'roots',len(entries),'unprotected',row['unprotected_validation_fraction'],
                               'three_slots',row['fraction_with_at_least_three_rule_slots'],flush=True)
    for ref in [*summary['units'],source]:v.require(v.pointer(ref['path'])==ref,'root inventory source/input changed')
    result=dict(schema='warehouse-maximal-protected-free-root-inventory.v1',plan=v.pointer(output/'plan.json'),units=rows,
        structural_partition_verified=True,root_rebuild_tested=False,new_fits=0,NN_updates=0,new_environment_resets=0,
        actual_programs_or_gates_changed=False,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444);print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.summary.resolve(),a.output.resolve())
