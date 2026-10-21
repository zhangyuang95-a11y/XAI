"""Synthetic component tests, no new scientific program/search or NN."""
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from backend.training import warehouse_v630_partition_proposals as c
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v626_search import Node,encode


class Budget:
    def __init__(self,limit=1):self.calls=[];self.limit=limit
    def check(self):pass
    def solve(self,kind,metadata,callback):
        if len(self.calls)>=self.limit:raise RuntimeError('synthetic predebit exhausted')
        self.calls.append((kind,metadata));return callback()


def toy():
    n=240;x=np.zeros((n,len(FEATURE_NAMES)),np.float32);feature=FEATURE_NAMES.index('self_battery_fraction')
    x[:,feature]=np.linspace(.01,.99,n,dtype=np.float32);m=np.ones((n,5),np.float32)
    target=np.zeros_like(m);target[x[:,feature]<.5,0]=1.;target[x[:,feature]>=.5,4]=1.
    def distributions(x,m):
        q=np.zeros((len(x),5),np.float32);q[:,4]=1.;return np.zeros(len(x),int),q.copy(),q
    program=SimpleNamespace(structure=c.plain(encode(None)[0]),leaves=[{}],maximum_leaves=256,
        validate=lambda:None,leaf_indices=lambda rows:np.zeros(len(rows),int),distributions=distributions)
    return program,dict(x=x,m=m,p=target),feature


def test_public_predicate_is_learned_once_under_node_predebit_and_no_outputs_escape():
    program,a,feature=toy();b=Budget();out=c.propose(program,a,0,b)
    assert len(b.calls)==1 and b.calls[0][0]=='node' and out['fit_called'] and out['fit_correct_count']==120
    assert out['classifier_outputs_discarded'] and out['no_qualification_or_action_output']
    assert any(t['leaf_count']>1 and any(float(w[feature])==1 for w in t['split_weights']) for t in out['proposals'])
    assert not any('probabilities' in t or 'qualification' in t for t in out['proposals'])


def test_classifier_fit_cannot_start_without_budget_predebit(monkeypatch):
    program,a,_=toy();b=Budget(limit=0)
    monkeypatch.setattr(c.DecisionTreeClassifier,'fit',lambda *a:pytest.fail('unbudgeted classifier fit'))
    with pytest.raises(RuntimeError,match='predebit exhausted'):c.propose(program,a,0,b)


def test_no_split_with_insufficient_frames_has_no_fit_call():
    program,a,_=toy();a={k:v[:20] for k,v in a.items()};b=Budget();out=c.propose(program,a,0,b)
    assert not out['fit_called'] and not b.calls and out['proposals'][0]['leaf_count']==1


def test_pruning_keeps_both_branches_and_counts_empty_leaves():
    weights=(1.,)+(0.,)*(len(FEATURE_NAMES)-1)
    tree=Node(weights,-.5,Node(weights,-.25),Node(weights,-.75))
    options=c.prunings(tree);assert len(options)==5 and sorted(t['leaf_count'] for t in options)==[1,2,3,3,4]


def test_global_leaf_allocation_matches_brute_force_and_is_input_order_independent():
    groups={'b':[dict(id='b0',leaves=1,score=[1.,0.]),dict(id='b1',leaves=3,score=[4.,1.])],
        'a':[dict(id='a0',leaves=1,score=[1.,2.]),dict(id='a1',leaves=2,score=[3.,1.]),dict(id='a2',leaves=3,score=[4.,0.])]}
    result=c.allocate(groups,4);choices=[]
    for a,b in product(groups['a'],groups['b']):
        if a['leaves']+b['leaves']<=4:choices.append((tuple(x+y for x,y in zip(a['score'],b['score'])),a['leaves']+b['leaves'],(a['id'],b['id'])))
    expected=min(choices,key=lambda x:(tuple(-v for v in x[0]),x[1],x[2]))
    assert tuple(result['score'])==expected[0] and result['leaves']==expected[1]
    reordered={k:list(reversed(groups[k])) for k in reversed(groups)};other=c.allocate(reordered,4)
    assert result['selected']==other['selected'] and result['score']==other['score']
    assert result['optimality_scope']=='only_supplied_fixed_options_and_additive_scores'


@pytest.mark.parametrize('bad',[{'a':[]},{'a':[dict(id='x',leaves=0,score=[1.])]},
    {'a':[dict(id='x',leaves=1,score=[float('nan')])]},
    {'a':[dict(id='x',leaves=1,score=[1.]),dict(id='x',leaves=1,score=[2.])]}])
def test_ambiguous_or_nonfinite_allocation_tables_are_rejected(bad):
    with pytest.raises(RuntimeError):c.allocate(bad,256)


def test_infeasible_leaf_budget_has_no_partial_success():
    groups={'a':[dict(id='a',leaves=2,score=[1.])],'b':[dict(id='b',leaves=2,score=[1.])]}
    with pytest.raises(RuntimeError,match='cannot fit'):c.allocate(groups,3)
