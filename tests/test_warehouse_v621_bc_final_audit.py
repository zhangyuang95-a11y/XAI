"""Only synthetic completion/probe semantics; no model/result assets opened."""
from copy import deepcopy
import ast
from pathlib import Path
import numpy as np
import pytest
from scripts import audit_warehouse_v621_bc_results as audit


def fixture(open_gate=True):
    p=np.asarray([[.6,.1,.1,.1,.1]]*2);q=np.asarray([[.8,.05,.05,.05,.05]]*2)
    rows=[{'encoded':np.zeros(1),'mask':np.ones(5),'nn_probabilities':p[i],
           'sampled_action':'UP','episode':'synthetic','base_seed':0} for i in range(2)]
    f={'coverage':1. if open_gate else .1,'actionable_coverage':1. if open_gate else .1,'agreement':1.,'kl':.05,
       **{k:0 for k in ('illegal_count','path_uncertified_count','whole_tree_raw_illegal_count','whole_tree_path_uncertified_count','core_KL_bound_violations')}}
    context={'nn_top':np.zeros(2,int),'raw_top':np.zeros(2,int),'leaf_indices':np.zeros(2,int),
       'path_certified':np.ones(2,bool),'program_probabilities':q}
    loss=.01*(.8*np.log(.8/.6)+.2*np.log(.05/.1)) if open_gate else 0.
    record={'current_fidelity':{**f,'passed':open_gate,'program_sha256':'b'*64,'current_actor_role_sha256':'a'*64},
      'program_coefficient':.01,'clone_virtual_frozen_at_step':150000,'post_75_percent_structure_and_core_logits_frozen_clone':True,
      'real_PPO_feedback_or_human_benefit_claimed':False,'optimizer_steps':0,'actual_PPO_joint_steps':0,'actor_parameters_unchanged':True,
      'variants':{'raw':{'feedback_rows':0,'weighted_loss':0.,'maximum_gradient_norm':0.,'nonzero_gradient_batches':0},
       'regularized':{'feedback_rows':2 if open_gate else 0,'weighted_loss':loss,'maximum_gradient_norm':.3 if open_gate else 0.,'nonzero_gradient_batches':1 if open_gate else 0}}}
    return record,{'feedback':f},context,rows,{'feedback_leaves':[0],'actor_role_sha256':'a'*64},'b'*64


def test_completion_preflight_fails_before_any_audit_asset_read(monkeypatch,tmp_path):
    import scripts.report_warehouse_v621_bc_metrics as reporter
    def incomplete(**kwargs):raise RuntimeError('entire batch incomplete')
    monkeypatch.setattr(reporter,'prepare_report',incomplete)
    monkeypatch.setattr(audit.Evidence,'__init__',lambda *a,**k:(_ for _ in ()).throw(AssertionError('asset read before completion')))
    with pytest.raises(RuntimeError,match='entire batch incomplete'):audit.audit(root=tmp_path,specification_path=tmp_path/'not-opened.json')


def test_global_gate_closed_is_distinguished_from_local_KL_zero():
    out=audit.probe_audit(*fixture(False))
    assert out['interpretation']=='independent_validation_gate_closed' and out['potential_matching_rows']==2
    assert out['actual_probe_rows']==0 and out['independent_weighted_q_to_p_KL']==0
    assert out['native_autograd_reexecuted'] is False and out['actual_PPO_joint_steps']==0


def test_probe_independent_direction_coefficient_and_raw_exact_zero():
    args=fixture();out=audit.probe_audit(*args)
    assert out['independent_weighted_q_to_p_KL']==pytest.approx(.0009151622107541)
    assert out['interpretation']=='positive_native_autograd_probe_bound_not_PPO'
    bad=deepcopy(args);bad[0]['variants']['regularized']['weighted_loss']*=100
    with pytest.raises(RuntimeError,match='samples/KL'):audit.probe_audit(*bad)
    bad=deepcopy(args);bad[0]['variants']['raw']['nonzero_gradient_batches']=1
    with pytest.raises(RuntimeError,match='raw external'):audit.probe_audit(*bad)


def test_auditor_source_has_no_execution_or_production_metric_calls():
    tree=ast.parse(Path(audit.__file__).read_text())
    forbidden={'reset','step','fit','backward','autograd','program_metrics','_metric','_metrics','differentiable_feedback_loss','load_and_authorize','create_model','build_model_lock'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
            assert name not in forbidden
