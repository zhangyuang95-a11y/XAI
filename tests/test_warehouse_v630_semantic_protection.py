"""Logical region preservation only; refined fixtures are not program assets."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from experiments.warehouse import v630_semantic_protection as s
from experiments.warehouse import v628_protocol as old
from backend.training.warehouse_v629_program import from_parent
from backend.training.warehouse_v629_guards import leaf_actions
from backend.training.warehouse_v626_search import Node,decode,encode,replace
from backend.training.warehouse_v622_program import plain
from backend.training.warehouse_v6_actor import FEATURE_NAMES


def parent(fold=0,role='robot_1'):
    ref=old.parents()[f'fold-{fold}-{role}']
    p=s.ParentProgram.from_dict(json.loads(Path(ref['path']).read_bytes()),old.old.ROLE_SHA256[role])
    return from_parent(p,ref,128)


@pytest.mark.parametrize('fold,role',[(f,r) for f in range(3) for r in ('robot_1','robot_2')])
def test_all_six_real_anchors_preserve_the_same_complete_region_contract(fold,role):
    p=parent(fold,role);proof=s.verify(p)
    assert len(proof['entries'])==len(p.protection['entries'])
    assert all(len(e['descendant_leaves'])==1 for e in proof['entries'])
    assert not proof['numerical_child_certificates_checked'] and not proof['descendant_qualification_inherited']
    assert not proof['training_authority'] and not proof['model_acceptance_passed']


@pytest.fixture
def split_anchor():
    p=parent();tree=decode(p.structure);_,paths=encode(tree)
    entry=p.protection['entries'][0];leaf=next(i for i in range(len(p.leaves))
        if s.region(p.structure,i,p.maximum_leaves)['path']==entry['path'])
    path=paths[leaf];weights=[0.]*len(FEATURE_NAMES);weights[FEATURE_NAMES.index('self_battery_fraction')]=1.
    structure,newpaths=encode(replace(tree,path,Node(tuple(weights),-.5)))
    # Output payloads are copied, but their old numerical certificates are
    # deliberately NOT upgraded. This fixture can only test the new logic.
    leaves=[deepcopy(p.leaves[next(i for i,oldpath in enumerate(paths) if newpath[:len(oldpath)]==oldpath)]) for newpath in newpaths]
    result=SimpleNamespace(role=p.role,actor_role_sha256=p.actor_role_sha256,structure=plain(structure),
        leaves=leaves,maximum_leaves=128,protection=deepcopy(p.protection))
    result.certificates=lambda:leaf_actions(result.structure,result.maximum_leaves)
    return p,result,leaf,path


def test_finer_complete_partition_keeps_both_children_and_inherits_no_qualification(split_anchor):
    before,p,leaf,path=split_anchor;proof=s.verify(p)
    changed=next(e for e in proof['entries'] if len(e['descendant_leaves'])==2)
    assert proof['leaf_count']==len(before.leaves)+1 and len(changed['descendant_leaves'])==2
    assert all(s.digest(s.payload(p.leaves[i]))==s.digest(s.payload(before.leaves[leaf])) for i in changed['descendant_leaves'])
    assert not proof['numerical_child_certificates_checked'] and not proof['current_NN_fidelity_checked']


def test_changing_just_one_child_output_is_rejected(split_anchor):
    before,p,leaf,path=split_anchor;proof=s.verify(p)
    child=next(e['descendant_leaves'][0] for e in proof['entries'] if len(e['descendant_leaves'])==2)
    p.leaves[child]['bias'][0]+=1e-5
    with pytest.raises(RuntimeError,match='float32 output payload'):s.verify(p)


def test_changing_original_route_prefix_is_rejected(split_anchor):
    _,p,_,_=split_anchor;p.structure['split_bias'][0]+=.125
    with pytest.raises(RuntimeError,match='route prefix'):s.verify(p)


def test_original_protected_leaf_cannot_be_silently_dropped(split_anchor):
    _,p,_,_=split_anchor;p.protection['entries']=p.protection['entries'][1:]
    with pytest.raises(RuntimeError,match='region set'):s.verify(p)


def test_ancestral_legal_facts_cannot_be_removed(split_anchor):
    _,p,_,_=split_anchor;p.certificates=lambda:tuple(frozenset() for i in range(len(p.leaves)))
    with pytest.raises(RuntimeError,match='logical legality map'):s.verify(p)


def test_every_empty_or_fallback_child_still_counts_toward_capacity(split_anchor):
    _,p,_,_=split_anchor;p.leaves.pop()
    with pytest.raises(RuntimeError,match='counted descendant'):s.verify(p)
