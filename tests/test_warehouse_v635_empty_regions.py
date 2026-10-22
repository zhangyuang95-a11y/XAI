"""Regression boundaries from startup interruption; no scientific retry."""
import numpy as np
from backend.training import warehouse_v635_search as s,warehouse_v635_subtrees as b
from backend.training.warehouse_v626_search import encode


def test_empty_qualification_region_has_zero_score_and_no_boolean_dtype_failure():
    # No coefficient evaluation occurs because every region is empty. The
    # fake leaf is not a deployable/certified asset and cannot authorize action.
    tree=s.plain(encode(None)[0]);option=dict(structure=tree,models=[{}],paths=[dict(legal=[4])])
    class EmptyArrays:
        pass
    a=dict(x=np.empty((0,173),np.float32),m=np.empty((0,5),np.float32),p=np.empty((0,5),np.float32),
           sampled=np.empty(0,int),rows=[])
    # The public prediction API still validates real leaves, even on zero
    # rows; supply an ordinary complete-domain constant with no proof needed
    # by this purely arithmetic local scoring function.
    from backend.training.warehouse_v626_domain import domain_sha
    from backend.training.warehouse_v626_program import VERSION
    option['models']=[dict(version=VERSION,domain_sha256=domain_sha(),normalized=True,support=[],bias=[0.]*5,weights=[[] for _ in range(5)])]
    result=s.option_score(option,{k:a for k in ('fit','eligibility','validation')},[np.empty(0,bool) for _ in range(30)])
    assert result['score']==[0,0.,0] and not any(result['children'][0]['reliable_selected'].values())


def test_json_stump_normalization_routes_every_row_to_the_only_leaf():
    from backend.training.warehouse_v628_certificate import structure_info
    from backend.training.warehouse_v622_program import _route
    encoded=s.plain(encode(None)[0]);tree,info=structure_info(encoded,256)
    assert tree['split_weights'].shape==(0,173) and info['depth']==0
    assert np.array_equal(_route(tree,np.zeros((5,173),np.float32),0),np.zeros(5,int))
