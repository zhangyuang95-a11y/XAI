"""Exact region/output protection under finer surrogate-only partitions.

This checks a structural invariant, not numerical legality certificates,
current-NN fidelity, eligibility, or candidate admission. No program parser,
fitter, execution adapter or training capability is supplied here.
"""
from pathlib import Path
from hashlib import sha256
import json
from backend.training.warehouse_v628_program import ParentProgram,make_protection,payload
from backend.training.warehouse_v628_certificate import structure_info,region
from backend.training.warehouse_v629_guards import leaf_actions
from experiments.warehouse.v626_protocol import require
from backend.training.warehouse_v622_program import digest

VERSION='warehouse-protected-region-complete-descendants-identical-payload.v1'


def verify(program):
    """Every old protected route prefix remains, all its descendants agree.

    A complete finite binary tree partitions each prefix exhaustively. Every
    descendant retaining the exact serialized output payload therefore gives
    the same float32 leaf arithmetic and masked distribution throughout the
    original region. Empty branches still count toward normal tree limits.
    The caller must separately validate every child's new numerical proof.
    """
    contract=program.protection
    require(isinstance(contract,dict) and set(contract)=={'parent','parent_version','entries'},
        'missing original complete protection contract')
    ref=contract['parent'];raw=Path(ref['path']).read_bytes()
    require(sha256(raw).hexdigest()==ref['sha256'],'protected parent bytes changed')
    parent=ParentProgram.from_dict(json.loads(raw),program.actor_role_sha256)
    require(parent.role==program.role and contract==make_protection(parent,ref),
        'semantic protection changed the original anchor or protected region set')
    tree,info=structure_info(program.structure,program.maximum_leaves)
    require(len(program.leaves)==tree['leaf_count'],'semantic protection requires every counted descendant leaf')
    paths=[region(tree,i,program.maximum_leaves)['path'] for i in range(tree['leaf_count'])]
    legal=program.certificates();require(isinstance(legal,tuple) and legal==leaf_actions(tree,program.maximum_leaves),
        'semantic protection lacks a complete logical legality map')
    entries=[];all_descendants=set()
    for entry in contract['entries']:
        prefix=entry['path'];depth=len(prefix)
        descendants=[i for i,path in enumerate(paths) if len(path)>=depth and digest(path[:depth])==digest(prefix)]
        require(descendants,'protected original route prefix was removed or modified')
        expected=digest(entry['leaf_payload'])
        for i in descendants:
            require(digest(payload(program.leaves[i]))==expected,
                'a protected descendant changed original float32 output payload')
            require(set(entry['legal'])<=set(legal[i]),'protected descendant lost its inherited explicit legal proof')
        require(not all_descendants.intersection(descendants),'original protected regions overlap')
        all_descendants.update(descendants)
        entries.append(dict(parent_leaf=entry['parent_leaf'],original_path_sha256=digest(prefix),
            original_leaf_payload_sha256=expected,descendant_leaves=descendants,
            complete_prefix_partition=True,all_descendant_payloads_identical=True))
    return dict(version=VERSION,parent=ref,role=program.role,actor_role_sha256=program.actor_role_sha256,
        tree_sha256=digest(tree),maximum_leaves=program.maximum_leaves,leaf_count=tree['leaf_count'],depth=info['depth'],
        entries=entries,protected_descendants=sorted(all_descendants),
        statement='all_original_protected_states_retain_identical_leaf_arithmetic_and_masked_distribution',
        numerical_child_certificates_checked=False,current_NN_fidelity_checked=False,
        descendant_qualification_inherited=False,model_acceptance_passed=False,training_authority=False,pilot_opened=False)
