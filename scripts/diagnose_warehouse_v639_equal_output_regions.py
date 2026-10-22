#!/usr/bin/env python3
"""Read-only structural inventory of redundant equal-output confidence trees.

No qualification, model fitting, proof LP, new environment or source edit.
A region is an opportunity, not evidence that contraction is certified or
that any new partition can meet fidelity/coverage gates.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v635_program as language
from backend.training.warehouse_v626_search import decode,encode
from backend.training.warehouse_v628_certificate import region
from scripts.run_warehouse_v625 import write_new


def inventory(program):
    tree=decode(program.structure);_,paths=encode(tree);lookup={p:i for i,p in enumerate(paths)}
    core=set(language.protection(program)['protected_descendants']);memo={}
    def inspect(node,path):
        if node is None:
            i=lookup[path];value=({v.digest(language.payload(program.leaves[i]))},[i])
        else:
            left,li=inspect(node.left,path+(0,));right,ri=inspect(node.right,path+(1,));value=(left|right,li+ri)
        memo[path]=value;return value
    inspect(tree,());out=[];legal=program.certificates()
    def choose(node,path):
        signatures,ids=memo[path]
        if node is not None and len(signatures)==1 and not core.intersection(ids):
            terms=region(program.structure,ids[0],256)['path'][:len(path)];known={4}
            for item in terms:known.update(language.guards.evidence(item['weight'],item['bias'])[item['branch']+'_proven'])
            out.append(dict(root=list(path),root_depth=len(path),source_leaves=ids,leaf_count=len(ids),
                descendant_depths=[len(paths[i]) for i in ids],remaining_root_depth=9-len(path),
                payload_sha256=next(iter(signatures)),root_legal=sorted(known),
                all_descendants_have_same_root_legal=all(set(legal[i])==known for i in ids),
                witness_known_at_root=program.leaves[ids[0]]['witness'] in known,
                contraction_certificate_checked=False,new_coverage_claimed=False))
            return
        if node is not None:choose(node.left,path+(0,));choose(node.right,path+(1,))
    choose(tree,());return out


def main(folder,output):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref);auditref=v.pointer(folder/'independent-replay.json')
    v.require(summary['all_six_settled'] and v.bound(auditref)['integrity_passed'],'need six completed replayed source assets')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    plan=dict(source=source,summary=summaryref,audit=auditref,created_utc=datetime.now(timezone.utc).isoformat(),
        selection='maximal_equal_payload_subtrees_disjoint_original_core_no_NN_labels',data='old_fit_exposure_only',
        optimizer_calls=0,new_environment_resets=0,maximum_workers=1)
    write_new(output/'plan.json',plan);_,old=v.checked_plan();dp=v.bound(v.bound(old['previous_plan'])['previous_plan']);rows=[]
    for ref in summary['units']:
        r=v.bound(ref);spec=r['spec'];program=language.Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']])
        reference=language.Program.from_dict(v.bound(v.bound(spec['source_result'])['program']),old['actor_role_sha256'][spec['role']])
        reference_paths=encode(decode(reference.structure))[1]
        found=inventory(program)
        for group in found:
            route=tuple(group['root'])
            group['same_root_already_proposed_in_v639']=route in reference_paths
            group['strictly_broader_than_v639_leaf_roots']=any(len(path)>len(route) and path[:len(route)]==route for path in reference_paths)
        a=data.load_parts(dp,spec['fold'],spec['role'])['fit'];route=program.leaf_indices(a['x'])
        counts=np.bincount(route,minlength=len(program.leaves));frames=len(a['x'])
        for group in found:group['fit_frames']=int(counts[group['source_leaves']].sum());group['fit_frame_fraction']=group['fit_frames']/frames
        row=dict(job=spec['id'],input=ref,source_leaf_count=len(program.leaves),source_maximum_depth=max(len(p) for p in encode(decode(program.structure))[1]),
            groups=found,equal_output_root_count=len(found),covered_fit_frames=sum(g['fit_frames'] for g in found),fit_frames=frames,
            possible_leaf_savings=sum(g['leaf_count']-1 for g in found),
            same_root_legal_roots=sum(g['all_descendants_have_same_root_legal'] for g in found),
            broader_roots=sum(g['strictly_broader_than_v639_leaf_roots'] for g in found),
            model_outputs_or_qualification_changed=False)
        rows.append(row);print(row['job'],'equal-output roots',len(found),'same legality',row['same_root_legal_roots'],flush=True)
    v.require(v.pointer(__file__)==source,'inventory source changed')
    result=dict(schema='warehouse-v639-equal-output-structural-inventory.v1',plan=v.pointer(output/'plan.json'),units=rows,
        new_fits=0,proof_LPs=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',result);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
