"""Hand-built receipt/table fixtures: no search, optimizer, model or environment."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.warehouse import v622_capacity_audit_math as a

NAMES=tuple('legal_'+v.lower() for v in a.ACTIONS)+('public_scalar',)
ROLE='a'*64


def arrays():
    masks=np.ones((12,5),np.float32);masks[:6,0]=0
    x=np.zeros((12,6),np.float32);x[:,:5]=masks;x[:,5]=np.arange(12)/12
    p=np.zeros_like(masks);p[:6,4]=1;p[6:,0]=1
    return x,masks,p,np.asarray(['ep'+str(i) for i in range(12)]),NAMES,ROLE


def resign(search):
    search.pop('report_sha256',None);search['report_sha256']=a.digest(search);return search


def attempt():
    return {'call':1,'rows_sha256':'b'*64,'rows':12,'path_legal_actions':[4],
        'cache_miss':True,'success':True,'actual_convex_solver_calls':1}


def final():
    return {'leaf_count':1,'leaf_optimization_calls':1,'leaf_optimization_calls_outside_search':1,
        'convex_solver_calls':1,'leaf_fits':{'0':{'enumerated_action_solves':1}}}


def search():
    starts=[]
    for i,(response,quantiles,contrast) in enumerate(a.RECIPES):
        starts.append({'initialization':{'id':i,'response':response,'quantiles':list(quantiles),'contrast':contrast},
            'initialization_started':True,'leaf_optimization_calls':0,'actual_convex_solver_calls':0,'rounds':[],
            'global_rounds_before_start':0,'global_rounds_after_start':0,'round_budget_available_at_start':((i+1)*60)//8,
            'reserved_cumulative_round_limit':((i+1)*60)//8,'reserved_cumulative_leaf_call_limit':((i+1)*19968)//8})
    return resign({'leaf_optimization_calls':1,'actual_convex_solver_calls':1,'leaf_api_attempts':[attempt()],
        'bootstrap_leaf_optimization_calls':1,'bootstrap_actual_convex_solver_calls':1,'search_runs':starts,
        'search_uses_fit_only':True,'calibration_eligibility_validation_used':False,'complete_incumbent':True,
        'execution_action_override':False,'global_search_rounds':0,'search_budget':{'initializations':8,
        'max_rounds':60,'max_leaf_optimization_calls':19968,'max_depth':6,'max_leaves':32,
        'template_count':4,'template_max_leaf_optimization_calls':20000,'table_max_leaf_optimization_calls':508},
        'input_identity':a._identity(arrays()),'actor_role_sha256':ROLE,'fit_rows':12})


def test_a32_complete_ledgers_and_bound_inputs():
    report=a.audit_search_budget(search(),'A32',final(),arrays())
    assert report['passed'] and report['total_leaf_api_calls']==2
    changed=list(arrays());changed[0]=changed[0].copy();changed[0][0,5]+=.1
    with pytest.raises(RuntimeError,match='input_identity'):
        a.audit_search_budget(search(),'A32',final(),changed)


@pytest.mark.parametrize('change',[
    lambda r:r.update(leaf_optimization_calls=True),
    lambda r:r.update(actual_convex_solver_calls=-1),
    lambda r:r['leaf_api_attempts'].clear(),
    lambda r:r['search_runs'].pop(),
    lambda r:r['search_runs'][0].update(leaf_optimization_calls=1),
    lambda r:r['leaf_api_attempts'][0].update(actual_convex_solver_calls=0),
    lambda r:r.update(global_search_rounds=1),
    lambda r:r.update(complete_incumbent=False),
    lambda r:r.update(calibration_eligibility_validation_used=True),
])
def test_a32_rejects_incomplete_or_forged_ledgers(change):
    r=search();change(r);resign(r)
    with pytest.raises(RuntimeError):a.audit_search_budget(r,'A32',final(),arrays())


def test_a32_report_digest_and_unstarted_wall_receipts():
    r=search();r['fit_rows']=11
    with pytest.raises(RuntimeError,match='digest'):a.audit_search_budget(r,'A32',final())
    r=search();r['search_runs'][-1]={'initialization':r['search_runs'][-1]['initialization'],
        'initialization_started':False,'rounds':[],'stop_reason':'wall_time_budget',
        'leaf_optimization_calls':0,'actual_convex_solver_calls':0,'remaining_queue':0}
    resign(r)
    with pytest.raises(RuntimeError,match='actual wall exhaustion'):a.audit_search_budget(r,'A32',final())
    r['wall_time_budget_exhausted']=True;resign(r)
    assert a.audit_search_budget(r,'A32',final())['passed']


def d_budget():
    r=search();r['search_runs']=[]
    r['generation']={'leaf_optimization_calls':1,'actual_convex_solver_calls':1,'leaf_api_attempts':[attempt()]}
    events=[]
    for i,legal in enumerate([[4],[4],[0,4]]):
        events.append({**attempt(),'call':i+1,'path_legal_actions':legal,'actual_convex_solver_calls':len(legal)})
    r['terminal_table']={'leaf_optimization_calls':3,'actual_convex_solver_calls':4,'leaf_api_attempts':events}
    r.update(leaf_optimization_calls=4,actual_convex_solver_calls=5,
        template_generation_leaf_optimization_calls=1,frozen_table_leaf_optimization_calls=3,
        final_refit_reserved_calls=0,unused_final_refit_allowance=32,
        final_leaf_outputs_frozen_from_selected_table=True,
        template_generation_reports=[{'template':0,'generation_calls':1,'completed_structure':True}])
    f={'leaf_count':2,'leaf_optimization_calls':0,'leaf_optimization_calls_outside_search':0,
        'convex_solver_calls':0,'leaf_fits':{'0':{},'1':{}}}
    return resign(r),f


def test_d32_budget_phase_calls_are_not_missing_zero_defaults():
    r,f=d_budget()
    assert a.audit_search_budget(r,'D32',f,arrays())['total_leaf_api_calls']==4
    for key,value in [('template_generation_leaf_optimization_calls',True),
            ('frozen_table_leaf_optimization_calls',-1),('actual_convex_solver_calls',6)]:
        bad=deepcopy(r);bad[key]=value;resign(bad)
        with pytest.raises(RuntimeError):a.audit_search_budget(bad,'D32',f)
    del r['terminal_table']['leaf_api_attempts'];resign(r)
    with pytest.raises(RuntimeError):a.audit_search_budget(r,'D32',f)


def compiler_fixture(guards):
    # A complete old 16-leaf tree in breadth-first node order. Pure scalar
    # splits prove no movement, so precisely the chosen movement leaves guard.
    weights=np.zeros((15,6),np.float32);weights[:,5]=1
    bias=np.full(15,-.5,np.float32)
    refs=lambda offset:np.asarray([2*i+offset if 2*i+offset<15 else -1-(2*i+offset-15) for i in range(15)])
    z=np.zeros((16,5),np.float32)
    for i in range(16):z[i,0 if i<guards else 4]=1
    actor=SimpleNamespace(feature_names=NAMES,role_sha256=ROLE,weights={'tree.node_count':np.asarray([15]),
        'tree.leaf_count':np.asarray([16]),'tree.split.weight':weights,'tree.split.bias':bias,
        'tree.left_refs':refs(1),'tree.right_refs':refs(2),'tree.leaf_logits':z})
    old,cap=a.old.structure({'root_ref':0,'split_weights':weights,'split_bias':bias,'left_refs':refs(1),
        'right_refs':refs(2),'leaf_count':16},NAMES,enforce_limits=False)
    # Fixture construction is independent of the helper under test: retain
    # each original split and append a legal-UP split to selected terminals.
    w,b,l,r,origins,decisions=[],[],[],[],[],[]
    def leaf(i,action,branch):
        k=len(origins);origins.append({'compiled_leaf':k,'original_leaf':i,'guard_branch':branch,'frozen_preferred_action':a.ACTIONS[action]});return -1-k
    def build(ref):
        if ref<0:
            i=-1-ref;guard=i<guards;action=0 if guard else 4
            d={'original_leaf':i,'frozen_preferred_action':a.ACTIONS[action],'guard_added':guard,
                'original_path':cap['leaf_paths'][i],'original_path_legal_actions':[4]};decisions.append(d)
            if not guard:return leaf(i,4,'unchanged')
            k=len(w);row=np.zeros(6,np.float32);row[0]=1;w.append(row);b.append(-.5)
            l.append(leaf(i,4,'illegal_false_wait'));r.append(leaf(i,0,'legal_true'))
            d.update(guard_node=k,guard_feature='legal_up',guard_bias=-.5,guard_coefficient=1.);return k
        k=len(w);w.append(weights[ref]);b.append(bias[ref]);l.append(0);r.append(0)
        l[k]=build(int(old['left_refs'][ref]));r[k]=build(int(old['right_refs'][ref]));return k
    build(0)
    tree,capacity=a.old.structure({'root_ref':0,'split_weights':w,'split_bias':b,'left_refs':l,'right_refs':r,'leaf_count':16+guards},NAMES,enforce_limits=False)
    proof=a.old.certificates(tree,NAMES,explicit_crossing=True)
    for i,o in enumerate(origins):o.update(path_legal_actions=sorted(proof[i]),compiled_path=capacity['leaf_paths'][i])
    report={'actor_role_sha256':ROLE,'guard_count':guards,'leaf_guard_decisions':decisions,'compiled_leaf_origins':origins,
        'schema':'warehouse-v622-internal-topology-legal-guards.v1',
        'limits':{'maximum_depth':6,'maximum_leaves':32,'maximum_source_features_per_node':3,'guards_and_empty_branches_count':True},
        'guard_policy':'frozen_raw_float32_leaf_argmax_then_explicit_legal_guard_else_WAIT',
        'original_structure':a._plain(old),'required_guarded_structure':a._plain(tree),
        'original_structure_sha256':a.digest(old),'required_guarded_structure_sha256':a.digest(tree),
        'original_capacity':a._plain(cap),'required_capacity':a._plain(capacity),'feasible':True,'status':'feasible',
        'infeasibility_reasons':[],'guard_policy_uses_observations':False,'all_guards_and_empty_children_counted':True,
        'input_weights_modified':False,'full_actor_policy_equivalence_claimed':False,
        'leaf_outputs_reused_as_fitted_program':False,'new_leaf_preferred_action_requires_new_path_proof':True}
    return report,actor,a._plain(tree)


@pytest.mark.parametrize('guards',[9,11])
def test_b32_rebuilds_entire_25_or_27_leaf_tree(guards):
    report,actor,tree=compiler_fixture(guards)
    result=a.audit_compiler(report,actor,tree)
    assert result['capacity']['leaf_count']==16+guards and result['guard_count']==guards
    tree['split_bias'][1]+=.1
    with pytest.raises(RuntimeError):a.audit_compiler(report,actor,tree)


def frozen_fixture():
    data=arrays();x,m,p,episodes,names,role=data
    tree={'root_ref':0,'split_weights':[[1.,0.,0.,0.,0.,0.]],'split_bias':[-.5],
        'left_refs':[-1],'right_refs':[-2],'leaf_count':2,'feature_dim':6}
    records=[];episode_order=sorted(set(episodes),key=lambda e:sha256(e.encode()).digest())
    groups={e:i%3 for i,e in enumerate(episode_order)}
    for node,(path,rows,legal,z) in enumerate([((),np.arange(12),[4],[0,0,0,0,2]),
            ((0,),np.arange(6),[4],[0,0,0,0,20]),((1,),np.arange(6,12),[0,4],[20,0,0,0,0])]):
        z=np.asarray(z,np.float32);scores=np.where(m[rows]>0,z.astype(float),-np.inf)
        logq=scores-np.log(np.exp(scores).sum(-1,keepdims=True))
        kl=-logq[np.arange(len(rows)),p[rows].argmax(-1)];matches=p[rows].argmax(-1)==z.argmax()
        stability=[]
        for g in range(3):
            take=np.asarray([groups[e]==g for e in episodes[rows]])
            stability.append({'group':g,'rows':int(take.sum()),'episodes':int(take.sum()),
                'agreement':float(matches[take].mean()) if take.any() else None,
                'masked_kl':float(kl[take].mean()) if take.any() else None})
        records.append({'node_id':node,'path':list(path),'rows':len(rows),'rows_sha256':sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest(),
            'path_legal_actions':legal,'logits':z.tolist(),'score':[0.,0.,-float(kl.sum()),-1.],
            'optimizer':{'enumerated_action_solves':len(legal),'success':True,'float32_feasible':True,
                'chosen_action':int(z.argmax()),'row_count':len(rows),'solver':'SLSQP','solver_status':0},'fit_qualified':False,
            'fit_stable_explanation_qualified':False,'episode_stability_groups':stability,'agreement':float(matches.mean())})
    score=[sum(r['score'][i] for r in records[1:]) for i in range(4)]
    cert={'objective_vector':score,'reachable_leaf_budgets':[1,2],'complete_DP':True,'combinations_evaluated':1,
        'selected_terminal_paths':[[0],[1]],'allowed_moves':'terminal_at_node_or_keep_both_children_no_child_promotion'}
    table={'template':0,'template_structure_sha256':a.digest(tree),'records':records,'frozen_terminal_table_sha256':a.digest(records),'certificate':cert}
    logits=np.asarray([r['logits'] for r in records[1:]],np.float32)
    binding={'selected_table_sha256':a.digest(records),'selected_terminal_paths':[[0],[1]],'selected_table_node_ids':[1,2],
        'selected_leaf_logits_sha256':sha256(logits.astype('<f4').tobytes()).hexdigest(),'restricted_table_gap':0.,'global_function_class_optimality_gap':None}
    structure={**deepcopy(tree),'initial_leaf_logits':logits.tolist(),'frozen_table_certificate':binding,
        'fit_target_sha256':a._identity(data)['fit_target_sha256']}
    selected=[]
    for i,rec in enumerate(records[1:]):
        selected.append({'leaf':i,'rows':rec['rows'],'depth':1,'path_legal_actions':rec['path_legal_actions'],
            'raw_action':int(np.asarray(rec['logits']).argmax()),'fit_qualified':False,'fit_stable_explanation_qualified':False,
            'table_node_id':rec['node_id'],'selected_terminal_path':rec['path'],'agreement':rec['agreement'],
            'expected_feedback_mass':rec['score'][1],'masked_kl':-rec['score'][2]/rec['rows'],'optimizer':rec['optimizer']})
    structure['fit_leaf_reports']=selected
    family={'selected_table_sha256':binding['selected_table_sha256'],'selected_terminal_paths':[[0],[1]],
        'selected_leaf_logits_sha256':binding['selected_leaf_logits_sha256'],'restricted_objective_lower_bound_vector':score,
        'restricted_objective_upper_bound_vector':score,'restricted_table_gap':0.,'completed_template_family_combination_gap':0.,
        'global_function_class_optimality_gap':None,'continuous_leaf_output_optimality_claimed':False,
        'final_refit_and_calibrated_program_covered_by_DP_certificate':False,'all_requested_templates_completed_and_enumerated':False}
    search={'templates_requested':4,'complete_frozen_tables':1,'selected_template':0,'selected_fit_score':score,
        'fit_leaf_reports':selected,
        'frozen_terminal_tables':[table],'certificate':family,'template_generation_reports':[{'template':0,
        'completed_structure':True,'frozen_table_completed':True,'dp_completed':True,'structure':tree,'structure_sha256':a.digest(tree)}]}
    raw={'method':'fixed_template_leaf_budget_dp_v622','actor_role_sha256':ROLE,'structure':tree,
        'leaf_logits':logits.tolist(),'report':{'precalibration_leaf_logits':logits.tolist(),
        'frozen_fit_leaf_reports':selected,'frozen_fit_leaf_reports_sha256':a.digest(selected),
        'fit_leaf_row_counts':[6,6],'leaf_fits':{'0':records[1]['optimizer'],'1':records[2]['optimizer']},
        'frozen_table_certificate':binding,'fit_target_sha256':structure['fit_target_sha256'],
        'leaf_calibrations':{'0':{'temperature':1.,'chosen_action':4},'1':{'temperature':1.,'chosen_action':0}}}}
    return search,raw,structure,data


def test_frozen_dp_recalculates_all_nodes_and_rejects_forged_zero_gap():
    search,raw,structure,data=frozen_fixture()
    result=a.audit_frozen_table(search,raw,structure,data)
    assert result['passed'] and result['DP_nodes_checked']==3
    search['frozen_terminal_tables'][0]['certificate']['selected_terminal_paths']=[[]]
    with pytest.raises(RuntimeError):a.audit_frozen_table(search,raw,structure,data)


def test_frozen_scores_recomputed_even_if_record_hash_is_resigned():
    s,r,t,data=frozen_fixture()
    records=s['frozen_terminal_tables'][0]['records'];records[0]['score'][2]=0.
    s['frozen_terminal_tables'][0]['frozen_terminal_table_sha256']=a.digest(records)
    with pytest.raises(RuntimeError,match='score'):a.audit_frozen_table(s,r,t,data)


def test_completed_dp_template_cannot_be_omitted_from_family():
    s,r,t,data=frozen_fixture();g=deepcopy(s['template_generation_reports'][0]);g['template']=1
    s['template_generation_reports'].append(g)
    with pytest.raises(RuntimeError,match='omitted'):a.audit_frozen_table(s,r,t,data)


@pytest.mark.parametrize('change',[
    lambda s,r,t:s['frozen_terminal_tables'][0]['records'].pop(),
    lambda s,r,t:s['template_generation_reports'][0].update(dp_completed=False),
    lambda s,r,t:s['certificate'].update(global_function_class_optimality_gap=0.),
    lambda s,r,t:r['leaf_logits'][0].__setitem__(4,19.),
    lambda s,r,t:t.update(fit_target_sha256='0'*64),
])
def test_frozen_table_rejects_missing_evidence_or_continuous_refit(change):
    s,r,t,data=frozen_fixture();change(s,r,t)
    with pytest.raises(RuntimeError):a.audit_frozen_table(s,r,t,data)


def test_no_production_search_program_or_solver_imports():
    source=Path(a.__file__).read_text();tree=ast.parse(source)
    imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert all('warehouse_v622_program' not in name and 'warehouse_v622_search' not in name for name in imports)
    names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not names.intersection({'fit_program','program_metrics','fit_structure','minimize','_prune_frozen_tree'})
