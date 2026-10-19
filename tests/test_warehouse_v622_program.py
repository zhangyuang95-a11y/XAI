"""Synthetic 32-leaf/schema/physical-mask tests; no environment or real model."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_v622_program as p
from backend.training.warehouse_v619_leaf_fit import ConstrainedLeafFitError
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import MOVE_DELTAS
from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES


class Role:
    feature_names=p.FEATURE_NAMES
    feature_dim=p.FEATURE_DIM
    model_version=p.MODEL_VERSION
    def __init__(self, identity="a"*64, probabilities=(.1,.1,.1,.1,.6)):
        self.role_sha256=identity
        self.logits=np.log(np.asarray(probabilities,np.float32))
    def logits_encoded(self,x):
        return np.tile(self.logits,(len(x),1))


def actor():
    return SimpleNamespace(roles={"robot_1":Role(),"robot_2":Role("b"*64)})


def rows(split,n=48,*,sampled="WAIT",masks=None):
    offset={"fit":100,"calibration":200,"eligibility":300,"validation":400,"new":500}[split]
    # Public observations are assembled from a fixed map, without constructing
    # or resetting an environment. This cell has legal UP, RIGHT and WAIT.
    obs=np.zeros(p.OBSERVATION_DIM,np.float32)
    index={name:i for i,name in enumerate(OBSERVATION_FEATURE_NAMES)}
    layout=STUDY_MAP_LAYOUT;cells=layout.passable_positions;own=(5,2)
    for r,tiles in enumerate(layout.tiles):
        for c,tile in enumerate(tiles):obs[index[f"wall_row_{r}_column_{c}"]]=float(tile=="#")
    positions={"self":own,"partner":cells[-1],"charger":layout.charger_position,
        "order_0_pickup":cells[0],"order_0_delivery":cells[1],
        "order_1_pickup":cells[-2],"order_1_delivery":cells[-1]}
    for name,(r,c) in positions.items():
        obs[index[name+"_row"]]=r/(layout.rows-1)
        obs[index[name+"_column"]]=c/(layout.cols-1)
    obs[index["self_active"]]=obs[index["partner_active"]]=1
    obs[index["self_battery_fraction"]]=obs[index["partner_battery_fraction"]]=1
    physical=np.asarray([float(layout.is_passable((own[0]+MOVE_DELTAS[a][0],own[1]+MOVE_DELTAS[a][1])))
        if a in MOVE_DELTAS else 1. for a in p.ACTIONS],np.float32)
    return [{"obs":obs.copy(),
        "mask":physical.copy() if masks is None else np.asarray(masks,np.float32).copy(),
        "episode":f"{split}-{i//12}","base_seed":offset+i//12,"frame":i%12,
        "sampled_action":sampled if i%2==0 else "UP"} for i in range(n)]


def parts():return [rows(name) for name in p.SPLITS]


def singleton():
    return {"root_ref":-1,"split_weights":np.zeros((0,p.FEATURE_DIM),np.float32),
        "split_bias":np.zeros(0,np.float32),"left_refs":np.zeros(0,np.int64),
        "right_refs":np.zeros(0,np.int64),"leaf_count":1,"feature_dim":p.FEATURE_DIM}


def fit(**kwargs):
    model=kwargs.pop("actor",actor());data=kwargs.pop("data",parts());tree=kwargs.pop("structure",singleton())
    return p.fit_program("robot_1",model,*data,structure=tree,
        structure_sha256=p.program_structure_sha256(tree),**kwargs)


def binary_shape(depth):
    if depth==0:return None
    return (binary_shape(depth-1),binary_shape(depth-1))


def shape_structure(shape):
    w,b,left,right=[],[],[],[];leaves=[]
    def visit(tree):
        if tree is None:
            index=len(leaves);leaves.append(index);return -1-index
        index=len(w);row=np.zeros(p.FEATURE_DIM,np.float32)
        row[p.FEATURE_NAMES.index("self_row")]=1
        w.append(row);b.append(-.5);left.append(0);right.append(0)
        left[index]=visit(tree[0]);right[index]=visit(tree[1]);return index
    root=visit(shape)
    return {"root_ref":root,"split_weights":np.asarray(w,np.float32).reshape(-1,p.FEATURE_DIM),
        "split_bias":np.asarray(b,np.float32),"left_refs":np.asarray(left,np.int64),
        "right_refs":np.asarray(right,np.int64),"leaf_count":len(leaves),"feature_dim":p.FEATURE_DIM}


def compiler_role(tree):
    return SimpleNamespace(feature_names=p.FEATURE_NAMES,feature_dim=p.FEATURE_DIM,role_sha256="c"*64,
        weights={"tree.node_count":np.array([len(tree["split_weights"])],np.float32),
            "tree.leaf_count":np.array([tree["leaf_count"]],np.float32),
            "tree.split.weight":tree["split_weights"],"tree.split.bias":tree["split_bias"],
            "tree.left_refs":tree["left_refs"],"tree.right_refs":tree["right_refs"],
            "tree.leaf_logits":np.tile(np.array([2,0,0,0,0],np.float32),(tree["leaf_count"],1))})


def test_limits_are_explicit_and_old_constants_remain_unchanged():
    from backend.training import warehouse_v6_program as old
    assert old.PROGRAM_MAX_LEAVES==16 and old.PROGRAM_MAX_DEPTH==5
    assert p.DEFAULT_LIMITS==p.ProgramLimits(6,32,3)
    for values in [(7,32,3),(6,33,3),(6,32,4),(True,32,3)]:
        with pytest.raises(ValueError):p.ProgramLimits(*values)
    assert p.ProgramLimits.from_dict(p.DEFAULT_LIMITS.to_dict())==p.DEFAULT_LIMITS


def test_b32_compilation_counts_all_32_guarded_leaves_and_depth6():
    # Sixteen leaves with one fifth-level branch and one shallower branch.
    half=binary_shape(3)
    deep=(None,None)
    def replace_first(s,replacement):
        if s is None:return replacement
        return (replace_first(s[0],replacement),s[1])
    def prune_first(s):
        if s==(None,None):return None
        return (prune_first(s[0]),s[1])
    shape=(replace_first(half,deep),prune_first(half))
    original=shape_structure(shape)
    assert original['leaf_count']==16
    role=compiler_role(original);before={k:v.copy() for k,v in role.weights.items()}
    tree,report=p.compile_actor_structure(role)
    assert tree is not None and tree['leaf_count']==32
    assert report['original_capacity']['depth']==5
    assert report['required_capacity']['depth']==6
    assert report["guard_count"]==original["leaf_count"]
    assert report["required_capacity"]["leaf_count"]==2*original["leaf_count"]
    assert report["all_guards_and_empty_children_counted"] is True
    assert report["full_actor_policy_equivalence_claimed"] is False
    assert all(np.array_equal(role.weights[k],v) for k,v in before.items())
    if tree is not None:
        proof=p._certificates(tree)
        assert all(p.ACTIONS.index(item["frozen_preferred_action"]) in proof[i]
                   for i,item in enumerate(report["compiled_leaf_origins"]))


def test_b32_full16_shallow_tree_compiles_to32_and_old16_rejects_capacity():
    role=compiler_role(shape_structure(binary_shape(4)))
    tree,report=p.compile_actor_structure(role)
    assert tree["leaf_count"]==32 and report["guard_count"]==16
    assert report["required_capacity"]["depth"]==5
    none,small=p.compile_actor_structure(role,limits=p.ProgramLimits(5,16,3))
    assert none is None and small["status"]=="structurally_infeasible"
    assert "maximum_leaf_count_exceeded" in small["infeasibility_reasons"]


def test_structural_depth_cycle_unreachable_and_source_cost_rejected():
    tree=shape_structure(binary_shape(5));p.normalize_structure(tree)
    with pytest.raises(ValueError):p.normalize_structure(shape_structure(binary_shape(6)))
    bad=deepcopy(tree);bad["left_refs"][0]=0
    with pytest.raises(ValueError):p.normalize_structure(bad)
    bad=shape_structure(binary_shape(1));bad["split_weights"][0,:4]=1
    with pytest.raises(ValueError):p.normalize_structure(bad)
    row=np.zeros(p.FEATURE_DIM,np.float32);row[p.FEATURE_NAMES.index("public_hard_risk_context")]=1
    row[0]=1;row[1]=1
    assert p.source_feature_cost(row)==4


def test_compiler_rejects_fractional_counts_and_unsparsified_internal_weights():
    role=compiler_role(shape_structure(binary_shape(1)))
    role.weights['tree.node_count'][0]=.9
    with pytest.raises(ValueError,match='counts'):p.compile_actor_structure(role)
    role=compiler_role(shape_structure(binary_shape(1)))
    role.weights['tree.split.weight'][0,:4]=1
    with pytest.raises(ValueError,match='hard-route parity'):p.compile_actor_structure(role)


def test_target_arithmetic_is_bit_exact_legacy_float32_and_rejects_illegal_sentinel_mass():
    from backend.training.warehouse_v6_program import _masked_probabilities
    logits=np.array([[1,-2,3,.5,-1],[-3,9,0,-.1,2]],np.float32)
    masks=np.array([[1,0,1,0,1],[0,1,0,1,1]],np.float32)
    assert np.array_equal(p.masked_probabilities(logits,masks),_masked_probabilities(logits,masks))
    assert p.masked_probabilities(logits,masks).dtype==np.float32
    with pytest.raises(RuntimeError,match="illegal"):
        p.masked_probabilities(np.array([[0,-2e9,-2e9,-2e9,-2e9]],np.float32),np.array([[0,1,1,1,1]],np.float32))


def test_full_NN_distributions_fit_metrics_and_actual_sample_denominator():
    model=actor();program=fit(actor=model)
    m=p.program_metrics(program,model,rows("validation"))
    f,e=m["feedback"],m["explanation"]
    assert program.feedback_leaves==program.explanation_leaves==frozenset({0})
    assert f["physical_frame_count"]==48 and f["coverage"]==1
    assert f["actionable_rows"]==24 and f["actionable_coverage"]==.5
    assert f["expected_actionable_coverage"]==pytest.approx(.75,abs=2e-7)
    assert f["agreement"]==1 and f["whole_tree_raw_illegal_count"]==f["whole_tree_path_uncertified_count"]==0
    assert f["kl"]<.01 and e["kl"]<.01
    assert set(f["public_strata"])=={"ordinary","conflict","near_robots","low_return_margin"}
    assert f["per_nn_action"]["WAIT"]["agreement"]==1
    assert f["per_nn_action"]["LEFT"]["qualified_coverage"] is None
    assert program.report["leaf_optimization_calls_outside_search"]==1
    assert program.report["convex_solver_calls"]==1
    assert program.report["validation"]["gate"]["passed"] is True


def test_every_32_leaf_is_really_fitted_including_empty_children():
    program=fit(structure=shape_structure(binary_shape(5)))
    assert program.report["leaf_optimization_calls"]==32
    assert program.report["leaf_optimization_calls_outside_search"]==32
    assert len(program.report["leaf_fits"])==32
    assert sum(program.report["fit_leaf_row_counts"])==48
    assert sum(v["empty_leaf"] for v in program.report["leaf_fits"].values())==31
    assert len(program.feedback_leaves)==1


def test_four_partitions_are_episode_and_scene_disjoint_before_solver(monkeypatch):
    data=parts();data[1][0]["episode"]=data[0][0]["episode"]
    monkeypatch.setattr(p,"fit_constrained_leaf_distribution",lambda *a,**kw:pytest.fail("solver reached"))
    with pytest.raises(ValueError,match="independent"):fit(data=data)
    data=parts();data[1][0]["base_seed"]=data[0][0]["base_seed"]
    with pytest.raises(ValueError,match="independent"):fit(data=data)


def test_raw_program_top_must_be_reproved_after_actual_fit(monkeypatch):
    real=p.fit_constrained_leaf_distribution
    def corrupt(*args,**kwargs):
        z,r=real(*args,**kwargs);z[0]=100;return z,r
    monkeypatch.setattr(p,"fit_constrained_leaf_distribution",corrupt)
    with pytest.raises(p.ProgramFitError,match="path proof") as caught:fit()
    assert caught.value.report["leaf_optimization_calls"]==1


def test_sampled_actions_are_NN_proposals_and_illegal_input_not_repaired():
    data=parts();data[0][0]["sampled_action"]="DOWN"
    with pytest.raises(ValueError,match="physically illegal"):fit(data=data)


def test_roundtrip_unknown_method_role_hash_limits_and_missing_legality_fail_closed(tmp_path):
    program=fit();path=tmp_path/'program.json';h=program.save(path)
    loaded=p.WarehouseV622Program.load(path,expected_sha256=h,actor_role_sha256="a"*64,role="robot_1")
    assert loaded.sha256()==program.sha256()
    with pytest.raises(FileExistsError):program.save(path)
    for change in ("method","role","schema","hash","limits","qualification","whole_tree"):
        value=deepcopy(program.to_dict())
        if change=="method":value["method"]="legacy_unknown"
        if change=="role":value["role"]="robot_2"
        if change=="schema":value["feature_schema_sha256"]="b"*64
        if change=="hash":value["actor_role_sha256"]="b"*64
        if change=="limits":value["limits"]["maximum_leaves"]=16
        if change=="qualification":value["feedback_leaves"]=[]
        if change=="whole_tree":del value["report"]["validation"]["feedback"]["whole_tree_raw_illegal_count"]
        with pytest.raises((RuntimeError,KeyError,ValueError)):
            p.WarehouseV622Program.from_dict(value,actor_role_sha256="a"*64,role="robot_1")
    with pytest.raises(RuntimeError):p.WarehouseV622Program.load(path,expected_sha256="b"*64,actor_role_sha256="a"*64)


def test_zero_qualification_keeps_none_KL_and_all_physical_frames():
    model=actor();model.roles['robot_1'].logits=np.log(np.asarray([.6,.1,.1,.1,.1],np.float32))
    program=fit(actor=model);m=p.program_metrics(program,model,rows('validation'))
    assert not program.feedback_leaves
    assert m['feedback']['coverage']==0 and m['feedback']['kl'] is None
    assert m['feedback']['physical_frame_count']==48
    evidence=p.feedback_evidence(program,model,rows('validation'))
    assert not evidence['qualified'].any()


def test_development_and_fresh_precheck_keep45_40_gates():
    program=fit();metrics=p.program_metrics(program,actor(),rows('validation'))
    assert p.gate_metrics(metrics,phase='development')==p.gate_metrics(metrics,phase='precheck')
    metrics['feedback']['coverage']=metrics['explanation']['coverage']=.44
    assert p.gate_metrics(metrics,phase='precheck')['passed'] is False


def test_current_fidelity_rebinds_current_NN_but_does_not_reuse_fit_episodes():
    model=actor();program=fit(actor=model);model.roles['robot_1'].role_sha256='c'*64
    with pytest.raises(RuntimeError):p.program_metrics(program,model,rows('validation'))
    with pytest.raises(RuntimeError):p.feedback_evidence(program,model,rows('validation'))
    receipt=p.current_actor_fidelity(program,model,rows('validation'))
    assert receipt['passed'] is True and receipt['current_actor_role_sha256']=='c'*64
    evidence=p.feedback_evidence(program,model,rows('validation'),current_fidelity=receipt)
    assert evidence['qualified'].all() and np.all(evidence['raw_program_top']==4)
    with pytest.raises(ValueError,match='reuse'):p.current_actor_fidelity(program,model,rows('fit'))
    receipt['program_sha256']='e'*64
    with pytest.raises(RuntimeError):p.feedback_evidence(program,model,rows('validation'),current_fidelity=receipt)


def test_final_API_reservation_checked_before_fits_and_solver_failure_counts(monkeypatch):
    monkeypatch.setattr(p,'fit_constrained_leaf_distribution',lambda *a,**kw:pytest.fail('no reserved API'))
    with pytest.raises(ValueError,match='budget'):fit(fit_budget={'maximum_leaf_api_calls':0})
    def fail(*a,**kw):
        raise ConstrainedLeafFitError('synthetic fail',candidate_reports=[{'action':4}])
    monkeypatch.setattr(p,'fit_constrained_leaf_distribution',fail)
    with pytest.raises(p.ProgramFitError) as caught:fit()
    assert caught.value.report['leaf_optimization_calls']==1
    assert caught.value.report['convex_solver_calls']==1
    assert caught.value.report['fit_budget']['actual_final_leaf_optimization_calls']==1


def test_wall_deadline_before_fit_and_after_completed_leaf_has_typed_scientific_receipt(monkeypatch):
    clock=[2.];monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    with pytest.raises(p.ProgramBudgetExhausted) as caught:fit(fit_budget={'deadline_monotonic':1.})
    assert caught.value.report['scientific_failure'] is True
    assert caught.value.report['leaf_optimization_calls']==0
    clock[0]=0;real=p.fit_constrained_leaf_distribution
    def late(*a,**kw):
        out=real(*a,**kw);clock[0]=2.;return out
    monkeypatch.setattr(p,'fit_constrained_leaf_distribution',late)
    with pytest.raises(p.ProgramBudgetExhausted) as caught:fit(fit_budget={'deadline_monotonic':1.})
    assert caught.value.report['failure_stage']=='after_leaf_fit_0'
    assert caught.value.report['leaf_optimization_calls']==1


def frozen_structure(model,data):
    tree=singleton();x,m,_=p._arrays(data[0]);target=p.masked_probabilities(model.roles['robot_1'].logits_encoded(x),m)
    z,r=p.fit_constrained_leaf_distribution(target,m,{4})
    tree.update({'initial_leaf_logits':z[None],
        'fit_target_sha256':sha256(np.ascontiguousarray(target,dtype='<f4').tobytes()).hexdigest(),
        'fit_leaf_reports':[{'leaf':0,'rows':len(x),'raw_action':4,'path_legal_actions':[4],
            'table_node_id':0,'selected_terminal_path':[],'optimizer':r}],
        'frozen_table_certificate':{'selected_table_sha256':'d'*64,'selected_terminal_paths':[[]],
            'selected_table_node_ids':[0],
            'selected_leaf_logits_sha256':sha256(np.ascontiguousarray(z[None],dtype='<f4').tobytes()).hexdigest(),
            'restricted_table_gap':0,'global_function_class_optimality_gap':None}})
    return tree


def test_D32_preserves_frozen_table_with_zero_refits_and_limits_certificate_to_precalibration(monkeypatch):
    model=actor();data=parts();tree=frozen_structure(model,data)
    monkeypatch.setattr(p,'fit_constrained_leaf_distribution',lambda *a,**kw:pytest.fail('D32 cannot refit frozen table'))
    program=fit(actor=model,data=data,structure=tree,fit_method=p.D32_METHOD,fit_budget={'maximum_leaf_api_calls':0})
    assert program.report['leaf_optimization_calls_outside_search']==0
    assert program.report['convex_solver_calls']==0
    assert program.report['fit_budget']['actual_final_leaf_optimization_calls']==0
    assert program.report['DP_certificate_scope']=='frozen_precalibration_table_only_not_final_continuous_optimum'
    value=program.to_dict();loaded=p.WarehouseV622Program.from_dict(value,actor_role_sha256='a'*64)
    assert loaded.sha256()==program.sha256()
    value['report']['precalibration_leaf_logits'][0][0]+=1
    with pytest.raises(RuntimeError,match='table'):p.WarehouseV622Program.from_dict(value,actor_role_sha256='a'*64)


def test_D32_missing_or_wrong_frozen_target_table_rejected_without_refit(monkeypatch):
    model=actor();data=parts();tree=frozen_structure(model,data);tree['fit_target_sha256']='e'*64
    monkeypatch.setattr(p,'fit_constrained_leaf_distribution',lambda *a,**kw:pytest.fail('D32 fallback fit'))
    with pytest.raises(ValueError,match='frozen-table'):fit(actor=model,data=data,structure=tree,fit_method=p.D32_METHOD)


@pytest.mark.parametrize('field',['selected_terminal_paths','selected_table_node_ids','leaf_path','leaf_node'])
def test_D32_selected_paths_and_table_rows_cannot_be_relabelled(field):
    model=actor();data=parts();tree=frozen_structure(model,data)
    if field=='selected_terminal_paths':tree['frozen_table_certificate'][field]=[[0]]
    elif field=='selected_table_node_ids':tree['frozen_table_certificate'][field]=[1]
    elif field=='leaf_path':tree['fit_leaf_reports'][0]['selected_terminal_path']=[0]
    else:tree['fit_leaf_reports'][0]['table_node_id']=1
    with pytest.raises(ValueError,match='frozen-table'):fit(actor=model,data=data,structure=tree,fit_method=p.D32_METHOD)


def test_real_new_D32_search_exports_a_directly_compatible_frozen_program():
    from backend.training import warehouse_v622_search as search
    model=actor();data=parts();x,m,_=p._arrays(data[0])
    target=p.masked_probabilities(model.roles['robot_1'].logits_encoded(x),m)
    cfg=search.SearchConfig(initializations=1,max_rounds=1,max_leaf_optimization_calls=8,
        template_count=1,template_max_leaf_optimization_calls=4,table_max_leaf_optimization_calls=2,
        template_candidates_per_growth=1,axis_features=1,semantic_candidates_per_node=1,
        proposals_per_round=1,beam_width=1,max_nonimproving_steps=1,max_depth=1,max_leaves=2)
    tree,report=search.fit_structure('D32',x,m,target,[r['episode'] for r in data[0]],
        feature_names=p.FEATURE_NAMES,actor_role_sha256='a'*64,config=cfg)
    program=fit(actor=model,data=data,structure=tree,fit_method=p.D32_METHOD)
    assert program.report['leaf_optimization_calls_outside_search']==0
    assert program.report['frozen_table_certificate']['selected_table_sha256']==report['certificate']['selected_table_sha256']
    assert program.report['fit_target_sha256']==report['fit_target_sha256']


def test_new_method_does_not_reinterpret_old_program_payload():
    with pytest.raises(RuntimeError):p.WarehouseV622Program.from_dict({'version':'warehouse-role-program.v6.20-robust-explanation'},actor_role_sha256='a'*64)
