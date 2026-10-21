"""Pre-registered v629 paired program study, no new reset/training authority."""
from pathlib import Path
import json
from . import v628_protocol as previous
from .v626_protocol import require, pointer, bound, file_sha, digest, old

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = 'warehouse-v629-cumulative-guard-01'
OUTPUT = ROOT/'output/warehouse_v629'/CANDIDATE
FILES = ('backend/training/warehouse_v629_guards.py', 'backend/training/warehouse_v629_program.py',
    'backend/training/warehouse_v629_search.py', 'experiments/warehouse/v629_protocol.py',
    'scripts/run_warehouse_v629.py', 'scripts/audit_warehouse_v629.py',
    'configs/warehouse_v629_recipe.json', 'docs/warehouse_v629_plan.md', 'tests/test_warehouse_v629.py')


def recipe():
    return dict(schema='warehouse-v629-cumulative-recipe.v1', candidate_id=CANDIDATE,
        method='cumulative_local_certified_repair_v1',
        arms=[dict(name='C64', capacity=64, guard_bundles=False),
              *[dict(name=f'G{n}', capacity=n, guard_bundles=True) for n in (64,128,256)]],
        initial='corresponding_v628_committed_inner_search_best_after_batch_settled',
        initial_source_order=['sealed_search_best_file', 'last_committed_CAS_best_only_if_terminal_budget_exhausted'],
        protected='unchanged_v626_feedback_explanation_union',
        maximum_depth=9, source_features_per_node=3, source_features_per_leaf=3, maximum_leaf_coefficients=20,
        local_unchanged_leaf_refit=False, cumulative_acceptance='immediate_in_same_sweep',
        guard_language='canonical_two_three_physical_legal_bit_conjunctions',
        guard_verifier='independent_exact_rational_replay_all_16_movement_masks',
        mandatory='all_maximal_disjoint_wholly_unprotected_roots_first_sweep',
        descendant_repairs='enumerated_each_sweep_fit_residual_priority_new_descendants_next_sweep',
        data='exact_v628_common_1650_development_pool_and_partitions',
        outer_evidence='shared_development_only_fresh_confirmation_required',
        limits=dict(leaf=80000,node=4096,proof=400000), search_seconds=4800, soft_search_seconds=4770,
        hard_search_seconds=4790, final_seconds=1200,
        common_final_reserve=dict(leaf=2048,node=0,proof=13312), rounds=14, beam_width=3,
        frontier='one_alternative_per_repair_path_deduplicated_before_truncation',
        maximum_workers=3, BLAS_threads=1, automatic_retry=False, stoppage_does_not_prove_infeasibility=True,
        gate=previous.recipe()['gate'], selected='smallest_G_capacity_with_all_6_units_pass_after_all_24_settled',
        run_after='verified_v628_driver_exit_and_all_18_settled_no_interruption',
        actor_updates=0, PPO_joint_steps=0, environment_resets=0, sealed_audit_used=False, heldout_used=False,
        pilot_opened=False, publishable=False, release_id=previous.recipe()['release_id'],
        performance=previous.recipe()['performance'])


def source_files():
    result = previous.source_files()
    require(digest(result) == '1fd77d5d7de38a99260e1ee06dddb6104bf0d99f11a3b22027539508e823721b', 'v628 source changed')
    for name in FILES: result[name] = file_sha(ROOT/name)
    return dict(sorted(result.items()))


def jobs():
    return [dict(job_id=f"{arm['name']}-fold-{fold}-{role}", fold=fold, role=role, **arm)
        for fold in range(3) for role in old.ROLES for arm in recipe()['arms']]


def checked_plan():
    path = OUTPUT/'plan.json'; p = json.loads(path.read_bytes())
    require(json.loads((OUTPUT/'plan.claim.json').read_bytes()) == dict(sha256=file_sha(path), single_use=True), 'v629 plan changed')
    require(p['source_files'] == source_files() and p['source_sha256'] == digest(source_files())
        and p['recipe'] == recipe() and p['jobs'] == jobs() and p['candidate_id'] == CANDIDATE, 'v629 source/config changed')
    require(previous.pointer(previous.OUTPUT/'plan.json') == p['previous_plan']
        and previous.pointer(previous.OUTPUT/'collection/result.json') == p['collection'], 'v629 common input changed')
    require(pointer(p['actor']['path']) == p['actor'] and p['actor']['sha256'] == old.ACTOR_SHA256, 'frozen NN changed')
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    require(p['runtime'] == runtime_identity(), 'numerical runtime changed')
    require(json.loads((ROOT/'configs/warehouse_v629_recipe.json').read_bytes()) == recipe(), 'v629 recipe differs')
    for name, sha in p['source_files'].items():
        snapshot = OUTPUT/'source'/p['source_sha256']/name
        require(file_sha(snapshot) == sha and not snapshot.stat().st_mode & 0o222, 'v629 source snapshot changed')
    return path, p
