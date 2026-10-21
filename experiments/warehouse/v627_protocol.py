"""Finite v627 admission. Programs are never execution controllers."""
from pathlib import Path
import json
import sklearn
import torch
from experiments.warehouse import v626_protocol as parent
from experiments.warehouse.v626_protocol import old,require,pointer,bound,file_sha,digest,ZERO

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE='warehouse-v627-semantic-compiled-01'
OUTPUT=ROOT/'output/warehouse_v627'/CANDIDATE
FILES=('backend/training/warehouse_v627_program.py','backend/training/warehouse_v627_search.py',
       'experiments/warehouse/v627_protocol.py','scripts/run_warehouse_v627.py',
       'scripts/audit_warehouse_v627.py','scripts/close_warehouse_v627.py',
       'configs/warehouse_v627_recipe.json','docs/warehouse_v627_plan.md')


def source_files():
    from scripts.run_warehouse_v626_search import files
    result=files()
    require(digest(result)=='b85f3b0464164eb220d5fe9ed4e5fd4456915c552994de68727328f5db32c1f1',
            'immutable v626 source changed')
    result.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(result.items()))


def recipe():
    return dict(schema='warehouse-v627-semantic-compiled-recipe.v1',candidate_id=CANDIDATE,
        stages=['S32','S64'],capacities={'S32':dict(semantic_depth=5,semantic_leaves=16,compiled_depth=6,compiled_leaves=32),
        'S64':dict(semantic_depth=6,semantic_leaves=32,compiled_depth=7,compiled_leaves=64)},
        initialization=['S1_guard_discarded_recompiled','NN_distribution_greedy_tree','random_extra_tree_2','random_extra_tree_3'],
        support_source_limit=3,leaf_coefficient_limit=20,
        inner_fit_partition=dict(fit=.55,calibration=.10,eligibility=.15,score=.20,unit='whole_scenario',stratify='task'),
        leaf_eligibility='unchanged_v622_eligible_with_counts_Wilson_episodes_KL_and_proofs',
        selection=['qualified_explanation_coverage','qualified_expected_feedback','negative_full_KL','negative_leaf_count'],
        rounds=14,beam_width=4,stagnation_stop=False,cache_entries=512,
        mandatory_per_start=['whole_tree_gradient','root_subtree_rebuild'],
        limits=dict(leaf=20000,node=1024,proof=100000),gradient_steps_total=20000,gradient_steps_per_call=200,
        gradient_optimizer=dict(name='Adam',lr=.015,batch_frames=512,sigmoid_temperature=.5,
            forward='same_raw_fp32_coefficients_shared_fp64_hard_router',backward='sparse_forward_dense_support_straight_through'),
        search_seconds=4800,settlement_reserve=10,final_seconds=600,
        final_leaf_calls_per_leaf=8,final_proof_calls_per_leaf=52,
        maximum_workers=3,BLAS_threads=1,Torch_threads=1,automatic_retry=False,
        complete_NN_distribution_target=True,reuse_only_pure_frozen_targets=True,Actor_unchanged=True,
        performance=dict(maximum_checkpoint_fraction=.10,blocks_science=False,blocks_release=True),
        development_gate=dict(coverage=.45,expected_feedback=.40,feedback_agreement=.95,
            explanation_agreement=.97,KL=.20,illegal=0,unproved=0),
        design_targets=dict(coverage=.55,expected_feedback=.50,feedback_agreement=.98,explanation_agreement=.99,KL=.10),
        release_id='policylens-three-domain-20261016.v6.0',**ZERO)


def checked_recipe():
    p=ROOT/'configs/warehouse_v627_recipe.json'
    require(json.loads(p.read_bytes())==recipe(),'v627 recipe changed')
    return pointer(p)


def runtime():
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    return {**runtime_identity(),'torch':torch.__version__,'sklearn':sklearn.__version__}


def jobs(stage):
    require(stage in ('S32','S64'),'unknown v627 stage')
    return [dict(job_id=f'fold-{fold}-{role}',fold=fold,role=role,stage=stage,arm='C3')
            for fold in range(3) for role in old.ROLES]


def gate(metrics):
    from backend.training.warehouse_v622_program import gate_metrics
    return gate_metrics(metrics,phase='development')
