"""Registered v628 capacity study, with finite development collection authority."""
from pathlib import Path
import json
from experiments.warehouse import v626_protocol as parent
from experiments.warehouse.v626_protocol import require,pointer,bound,file_sha,digest,old

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE='warehouse-v628-v626-capacity-01'
OUTPUT=ROOT/'output/warehouse_v628'/CANDIDATE
FILES=('backend/training/warehouse_v628_certificate.py','backend/training/warehouse_v628_program.py',
    'backend/training/warehouse_v628_checkpoint.py','backend/training/warehouse_v628_search.py',
    'experiments/warehouse/v628_protocol.py','experiments/warehouse/v628_data.py',
    'scripts/run_warehouse_v628.py','scripts/audit_warehouse_v628.py',
    'configs/warehouse_v628_recipe.json','docs/warehouse_v628_plan.md',
    'tests/test_warehouse_v628.py')


def recipe():
    return dict(schema='warehouse-v628-protected-capacity-recipe.v1',candidate_id=CANDIDATE,
        method='protected_region_subtree_refinement_v1',capacities=[64,128,256],maximum_depth=9,
        source_features_per_node=3,source_features_per_leaf=3,maximum_leaf_coefficients=20,
        parent='v626_TAO64_final_programs',protected='union_parent_feedback_and_explanation',
        protected_refit=False,protected_temperature_calibration=False,inherited_qualification=False,
        data=dict(episodes=1650,argmax=300,sampled=1350,outer=dict(fit=900,calibration=150,eligibility=300,validation=300),
            inner=dict(fit=450,calibration=90,eligibility=180,validation=180),
            fold_rotation_sampled_per_task=100,inner_argmax_fit_only=True,scene_disjoint_within_fold=True),
        limits=dict(leaf=80000,node=4096,proof=400000),search_seconds=4800,final_seconds=1200,
        common_final_reserve=dict(leaf=2048,node=0,proof=13312),
        rounds=14,beam_width=3,cache_entries=512,automatic_retry=False,stagnation_stop=False,
        node_initialization_rng=0,node_rng_freshness_claimed=False,mandatory_all_editable_regions=True,
        independent_capacity_initializations=True,shared_learned_caches=False,maximum_workers=3,BLAS_threads=1,
        gate=dict(coverage=.45,expected_feedback=.40,feedback_agreement=.95,explanation_agreement=.97,KL=.20,illegal=0,unproved=0),
        confirmation=dict(episodes=300,pressure_scenes=72,draws_per_scene=3,bootstrap_one_sided_lower=.9875,primary_endpoints=4,
            requires_all_18_complete=True,requires_fixed_six_programs=True),
        performance=dict(maximum_checkpoint_fraction=.10,blocks_science=False,blocks_release=True),
        actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False,sealed_audit_used=False,heldout_used=False,
        release_id='policylens-three-domain-20261016.v6.0')


def source_files():
    from scripts.run_warehouse_v626_search import files
    result=files()
    require(digest(result)=='b85f3b0464164eb220d5fe9ed4e5fd4456915c552994de68727328f5db32c1f1','v626 source changed')
    # All imported collection and seed auditing implementations are explicit.
    for name in (*FILES,'experiments/warehouse/seed_metadata_audit.py','experiments/warehouse/seed_metadata_triage.py',
        'experiments/warehouse/exact_seed_audit.py','experiments/warehouse/seed_context.py'):
        result[name]=file_sha(ROOT/name)
    return dict(sorted(result.items()))


def jobs():
    return [dict(job_id=f'C{n}-fold-{fold}-{role}',capacity=n,fold=fold,role=role)
        for fold in range(3) for role in old.ROLES for n in (64,128,256)]


def parents():
    out={}
    for fold in range(3):
        for role in old.ROLES:
            key=f'fold-{fold}-{role}'
            out[key]=pointer(parent.OUTPUT/'TAO64/jobs'/key/'program.json')
    return out


def checked_plan():
    path=OUTPUT/'plan.json';p=json.loads(path.read_bytes())
    require(json.loads(path.with_name('plan.claim.json').read_bytes())==dict(sha256=file_sha(path),single_use=True),'v628 plan seal changed')
    require(p['candidate_id']==CANDIDATE and p['source_files']==source_files() and p['source_sha256']==digest(source_files())
        and p['recipe']==recipe() and p['jobs']==jobs() and p['parents']==parents(),'v628 source/config/parent binding changed')
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    require(p['runtime']==runtime_identity(),'v628 numerical runtime changed')
    require(p['actor']['sha256']==old.ACTOR_SHA256 and p['actor_role_sha256']==old.ROLE_SHA256
        and pointer(p['actor']['path'])==p['actor'],'v628 Actor binding changed')
    require(json.loads((ROOT/'configs/warehouse_v628_recipe.json').read_bytes())==recipe(),'v628 recipe differs')
    from .v628_data import cases,folds
    require(p['cases']==cases() and p['folds']==folds(),'v628 registered scene/fold allocation changed')
    for name,sha in p['source_files'].items():
        f=OUTPUT/'source'/p['source_sha256']/name
        require(file_sha(f)==sha and not f.stat().st_mode&0o222,'v628 source snapshot changed')
    return path,p
