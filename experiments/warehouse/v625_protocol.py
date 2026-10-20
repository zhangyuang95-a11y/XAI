"""v6.25 spent-data capability. No environment or training authority."""
from __future__ import annotations
import json
from pathlib import Path

from experiments.warehouse import v624_protocol as previous
from experiments.warehouse import v622_recovery_protocol as old
from experiments.warehouse.v624_protocol import (
    require, digest, file_sha, pointer, bound, checked_baseline, program_gate,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = 'warehouse-v625-sparse-logistic-01'
RECIPE = 'configs/warehouse_v625_recipe.json'
ENGINEERING_RECEIPT = 'native-04/receipt.json'
NATIVE_RECIPE = 'configs/warehouse_v625_native_engineering.json'
OUTPUT = ROOT / 'output/warehouse_v625' / CANDIDATE / 'stage-A' / 'paired-01'
ROLES, SPLITS, ARMS = old.ROLES, old.SPLITS, ('P0', 'P1')
ZERO = dict(previous.ZERO)
GATES = dict(previous.GATES)
BUDGET = dict(previous.BUDGET)
HISTORICAL_SOURCE = '04e895cea7f687df17db18248413b33d2888b5ed6d1b0c5518b267f557d59986'
NEW_FILES = (
    RECIPE, NATIVE_RECIPE, 'docs/warehouse_v625_plan.md', 'experiments/warehouse/v625_protocol.py',
    'backend/training/warehouse_v625_program.py',
    'backend/training/warehouse_v625_checkpoint.py',
    'backend/training/warehouse_v625_search.py',
    'scripts/run_warehouse_v625.py', 'scripts/audit_warehouse_v625.py',
    'scripts/verify_warehouse_v625_engineering.py',
    'scripts/verify_warehouse_v625_native.py',
    'scripts/finalize_warehouse_v625.py',
    'scripts/report_warehouse_v625.py',
)


def jobs():
    return previous.jobs()


def source_files(root=ROOT):
    root = Path(root)
    files = previous.source_files(root)
    require(digest(files) == HISTORICAL_SOURCE, 'historical v624 source was altered')
    files.update({rel: file_sha(root / rel) for rel in NEW_FILES})
    return dict(sorted(files.items()))


def source_sha(root=ROOT):
    return digest(source_files(root))


def expected_recipe():
    from dataclasses import asdict
    from backend.training.warehouse_v625_search import UpgradeBudget, search_configuration
    return dict(ZERO, **BUDGET,
        schema='warehouse-v625-sparse-logistic-recipe.v1', candidate_id=CANDIDATE,
        release_id='policylens-three-domain-20261016.v6.0', stage='A',
        arms=list(ARMS), roles=list(ROLES), folds=3, spent_development_episodes=275,
        eligible_upgrade_arm='P1', P0_is_control_only=True,
        all_twelve_complete_before_selection=True, automatic_retry=False,
        maximum_depth=6, maximum_leaves=32, maximum_public_source_features_per_node=3,
        maximum_public_source_features_per_leaf=3, maximum_leaf_coefficients=20,
        maximum_total_leaf_coefficients=640, all_guards_and_empty_branches_count=True,
        initialization='paired_fresh_search_from_identical_pinned_S1_topology',
        target='complete_frozen_NN_physical_masked_five_action_distribution',
        objective='KL_NN_to_program_plus_L2', maximum_workers=3, BLAS_threads_per_worker=1,
        search_configuration=search_configuration(), production_budget=asdict(UpgradeBudget()),
        leaf_fit=dict(L2=1e-4, maxiter=200, tolerance=1e-6,
            temperature_candidates=[.8,.9,1.,1.1,1.2], candidate_features=8,
            maximum_full_support_witness_trials=8, failed_trials_charged=True,
            final_fixed_support_and_witness=True),
        certificate='public_bounded_box_corners_fixed_path_legal_witness_float32_margin',
        checkpoint='content_addressed_objects_safe_explicit_pause_only_no_crash_resume',
        engineering=dict(maximum_checkpoint_fraction=.10, actual_large_state=True,
            slow_io_and_crash_injection=True, hard_deadline_required=True,
            required_receipt=ENGINEERING_RECEIPT,
            measurement='native_actual_runner_both_languages_absolute_IO_and_paired_extra_time',
            legacy_routing_only_preflights_passed=False,
            legacy_routing_only_failures_must_be_preserved=True,
            native_recipe=NATIVE_RECIPE),
        precheck=GATES, future_stages_authorized=False,
        future_confirmation=dict(authorized_by_stage_A_recipe=False,
            fresh_episodes=dict(fit=100, calibration=25, eligibility=75, validation=75),
            fit_argmax_episodes=50, pressure_scenarios=72, pressure_trajectories_per_scenario=3,
            pressure_gate='strict_development', primary_comparisons=[
                'robot_1_explanation_coverage','robot_2_explanation_coverage',
                'robot_1_expected_feedback_coverage','robot_2_expected_feedback_coverage'],
            individual_one_sided_lower_confidence_level=.9875,
            paired_bootstrap_unit='complete_scenario_stratified_by_task',
            estimand='pooled_physical_frame_numerator_over_denominator'),
        future_training=dict(authorized_by_stage_A_recipe=False,
            independent_pilot_steps=20000, independent_restarted_budgets=[200000,300000,400000],
            continuation_streams_per_budget=3, environments=8, rollout_steps_per_environment=128,
            epochs=4, minibatch_joint_frames=256, actor_lr=3e-5, critic_lr=3e-4,
            gamma=.99, GAE=.95, clip=.2, entropy=.005, gradient_clip=.5,
            critic_only_steps=4000, reference_beta_initial=1., reference_KL_target=.01,
            reference_beta_bounds=[.001,100.], reference_mix=[.5,.5],
            extraction_fractions=[.75,.84,.93], lambda_ramp=[.80,.85], lambda_target=.01,
            checkpoint_fractions=[.85,.93,1.], checkpoint_score='min_raw_reg_D_minus_.25C_minus_5S',
            program_episodes_per_variant_run_maximum=1100,
            raw_program_feedback_and_gradient=0, external_topology_support_fixed_after_B=True))


def recipe(root=ROOT):
    path = Path(root) / RECIPE
    require(json.loads(path.read_bytes()) == expected_recipe(), 'v625 scientific recipe differs')
    return file_sha(path)


def comparison(results):
    # The old comparator independently enforces the exact twelve unique jobs,
    # per-unit metrics, finite probabilities and all-six gates.
    value = previous.comparison(results)
    value['selected_arm'] = 'P1' if value['arms']['P1']['all_six_pass'] else None
    value['next_stage'] = 'fresh_confirmation' if value['selected_arm'] else 'closed_failed_program_precheck'
    value['P0_is_control_only'] = True
    return value


def require_stage_A_pass():
    require(not (OUTPUT/'execution/failed.json').exists()
        and not (OUTPUT/'finalizations/engineering-interruption-01').exists(),
        'interrupted or failed candidate cannot grant a new-stage capability')
    seal_path=OUTPUT/'finalizations/completed-development-01/receipt.json'
    seal_ref=pointer(seal_path);seal=bound(seal_ref,seal_path)
    require(seal.get('status')=='stage_A_passed' and seal.get('candidate_closed') is False
        and seal.get('selected_arm')=='P1' and seal.get('complete_scientific_comparison') is True
        and seal.get('candidate_id')==CANDIDATE and seal.get('source_sha256')==source_sha(),
        'completed independently audited stage A seal required')
    manifest=bound(seal['evidence_manifest'],seal_path.parent/'evidence-manifest.json')
    require(manifest.get('source_sha256')==source_sha() and manifest.get('candidate_id')==CANDIDATE,
            'sealed manifest identity changed')
    for asset in manifest['files']:
        path=Path(asset['path'])
        require(path.is_relative_to(OUTPUT) and path.stat().st_size==asset['size']
            and file_sha(path)==asset['sha256'],'sealed stage A evidence missing or changed')
    summary_ref = pointer(OUTPUT / 'summary.json')
    audit_ref = pointer(OUTPUT / 'independent-audit.json')
    summary, audit = bound(summary_ref), bound(audit_ref)
    require(summary.get('candidate_id') == CANDIDATE and summary.get('selected_arm') == 'P1'
        and summary.get('arms', {}).get('P1', {}).get('all_six_pass') is True
        and summary.get('source_sha256') == source_sha()
        and audit.get('passed') is True and audit.get('summary_sha256') == summary_ref['sha256'],
        'fresh data/training requires completed independently audited six-pass P1')
    return dict(summary=summary_ref, independent_audit=audit_ref,seal=seal_ref,
                evidence_manifest=seal['evidence_manifest'])
