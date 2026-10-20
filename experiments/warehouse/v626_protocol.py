"""Source-bound staged v6.26 protocol. Scientific and performance gates split."""
from pathlib import Path
import json
from experiments.warehouse import v625_protocol as previous
from experiments.warehouse.v625_protocol import old,require,pointer,file_sha,bound,digest

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE='warehouse-v626-domain-complete-01'
OUTPUT=ROOT/'output/warehouse_v626'/CANDIDATE
ZERO=dict(environment_resets=0,actor_updates=0,PPO_joint_steps=0,pilot_opened=False,
          publishable=False,fresh_data_used=False,sealed_audit_used=False,heldout_used=False)
FILES=('backend/training/warehouse_v626_domain.py','backend/training/warehouse_v626_certificate.py',
       'backend/training/warehouse_v626_program.py','backend/training/warehouse_v626_budget.py',
       'experiments/warehouse/v626_protocol.py','scripts/run_warehouse_v626.py',
       'scripts/audit_warehouse_v626.py','configs/warehouse_v626_recipe.json','docs/warehouse_v626_plan.md')


def source_files():
    files=previous.source_files()
    require(digest(files)=='f2183b27612b6acefe74c0ce894ea96cf997fede9a4152972655fbcc287521c6','historical v625 source changed')
    files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def source_sha():return digest(source_files())


def recipe():
    from backend.training.warehouse_v626_domain import domain_sha
    return dict(schema='warehouse-v626-domain-complete-recipe.v1',candidate_id=CANDIDATE,
        domain_sha256=domain_sha(),release_id='policylens-three-domain-20261016.v6.0',
        arms=['C0','C1','C2','C3'],roles=list(old.ROLES),folds=3,spent_episodes=275,
        limits_A0=dict(leaf=2304,node=0,proof=10000),A0_wall_seconds=1200,
        final_leaf_reserve=256,final_wall_seconds=600,maximum_workers=3,BLAS_threads=1,
        source_features_per_node=3,source_features_per_leaf=3,coefficients_per_leaf=20,
        internal_fit_split=dict(train=.8,selection=.2,unit='whole_scenario',stratify='task',
            hash_recipe=CANDIDATE+':inner-fit-v1'),
        gate=dict(feedback_coverage=.45,explanation_coverage=.45,expected_feedback_coverage=.40,
            feedback_agreement=.95,explanation_agreement=.97,qualified_KL=.20,illegal=0,uncertified=0),
        performance_gate=dict(maximum_checkpoint_fraction=.10,blocks_science=False,blocks_release=True),
        correctness_and_budget_remain_hard_gates=True,
        A1=dict(order=['TAO32','TAO64'],rounds=14,leaf_calls=20000,node_calls=1024,proof_calls=100000,
            search_seconds=4800,final_seconds=600,reserved_final_calls_per_leaf=8,
            capacities=[dict(leaves=32,depth=6),dict(leaves=64,depth=7)]),
        B=dict(fit_episodes=100,argmax_fit_episodes=50,calibration_episodes=25,eligibility_episodes=75,
            validation_episodes=75,pressure_scenarios=72,pressure_trajectories=3,
            requires_scientific_method_lock=True,bootstrap_lower=.9875,primary_endpoints=4),
        actor_initialization='exact_complete_frozen_v620',full_NN_masked_distribution_targets=True,
        no_diagnostic_cache_import=True,automatic_retry=False,**ZERO)


def checked_recipe():
    path=ROOT/'configs/warehouse_v626_recipe.json'
    require(json.loads(path.read_bytes())==recipe(),'registered v626 recipe differs')
    return pointer(path)


def jobs():
    return [dict(job_id=f'fold-{fold}-{role}-{arm}',fold=fold,role=role,arm=arm)
        for fold in range(3) for role in old.ROLES for arm in ('C0','C1','C2','C3')]


def gate(metrics):
    from backend.training.warehouse_v622_program import gate_metrics
    return gate_metrics(metrics,phase='development')
