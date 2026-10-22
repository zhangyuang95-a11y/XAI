"""Union source integrity, fixed proposal preservation and no hidden action."""
from copy import deepcopy
import pytest
from backend.training import warehouse_v639_joint_precision as s


def test_union_keeps_old_tree_and_both_precision_proposals_without_mutation():
    stump=s.plain(s.encode(None)[0]);d=stump['feature_dim'];w=[0.]*d;w[0]=1.
    tree=s.plain(s.encode(s.build.Node(tuple(w),-.5,None,None))[0])
    old=dict(source_leaf=3,fit_frames=120,proposals=[stump]);extras=[dict(source_leaf=3,fit_frames=120,precision_target=p,proposals=[tree,stump]) for p in (.98,.99)]
    before=deepcopy((old,extras));out=s.merge_proposals(old,extras)
    assert (old,extras)==before and len(out['proposals'])==2
    assert {s.digest(t) for t in old['proposals']}<={s.digest(t) for t in out['proposals']}
    assert out['entropy_template_reused'] and out['execution_actions_produced'] is False
    assert 'models' not in out and len(out['templates'])==3
    extras[1]['source_leaf']=4
    with pytest.raises(RuntimeError,match='mismatch'):s.merge_proposals(old,extras)


def test_union_rejects_missing_precision_target():
    old=dict(source_leaf=0,fit_frames=0,proposals=[])
    with pytest.raises(RuntimeError,match='both'):s.merge_proposals(old,[])
