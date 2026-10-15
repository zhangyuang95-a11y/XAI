"""Warehouse v5 paired actor and runtime binding regressions."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from backend.training.warehouse_dual_v5 import (
    DualNN, collect_episode, export_actor, train_variant, _weight_hashes,
    _scenario_seed, _scenario_schedule_sha256, _validate_scenario_schedule,
)
from backend.training import warehouse_dual_v5 as trainer
from domains.warehouse import neural_turnbased as warehouse
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.pure_nn import FrozenDualPureNN, observations


def _valid_heldout_evidence():
    summary = {'mean_deliveries': 5.0, 'mean_collisions': 3.0,
               'shutdown_episode_rate': 0.0, 'quality': 4.25,
               'passed': True}
    tasks, combined = {}, []
    for task in (1, 2, 3):
        episodes = []
        for base_seed in range(250000, 250100):
            scenario_seed = base_seed * 10 + task
            episodes.append({'seed': scenario_seed, 'deliveries': 5,
                             'collisions': 3, 'shutdown_episode': False})
            combined.append({'task': task, 'seed': base_seed,
                             'scenario_seed': scenario_seed, 'deliveries': 5,
                             'collisions': 3, 'shutdown_episode': False})
        tasks[str(task)] = {'summary': summary.copy(), 'episodes': episodes}
    return {variant: {'summary': summary.copy(), 'tasks': deepcopy(tasks),
                      'episodes': deepcopy(combined)}
            for variant in ('raw', 'regularized')}


def _fixture_actor_hash_only(monkeypatch, config):
    """Let unaccepted test pairs exercise mechanics without a release claim."""
    def verify(policy_id):
        if policy_id == warehouse.LEGACY_POLICY_ID:
            return
        record = config['variants'][warehouse._V5_POLICY_IDS[policy_id]]
        path = warehouse.ROOT / record['actor_path']
        try:
            digest = sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError as exc:
            raise RuntimeError('Warehouse dual NN artifact missing') from exc
        if digest != record['actor_sha256']:
            raise RuntimeError('Warehouse dual NN artifact hash mismatch')
    monkeypatch.setattr(warehouse, '_verify_v5_artifact', verify)


def test_dual_actor_export_and_role_hashes_are_independent(tmp_path):
    torch.manual_seed(318)
    model = DualNN(hidden=32)
    output = export_actor(model, tmp_path / "pair.npz", pair_seed=318, variant="raw")
    assert output["role_sha256"]["robot_1"] != output["role_sha256"]["robot_2"]
    frozen = FrozenDualPureNN.load(output["path"],
                                   expected_sha256=output["sha256"],
                                   expected_role_sha256=output["role_sha256"])
    environment = WarehouseMultiAgentEnv(participant_study_config())
    environment.reset(seed=100)
    features = observations(environment.state)
    with torch.no_grad():
        expected = np.stack([model.actors[agent_id](torch.from_numpy(features[row])).numpy()
                             for row, agent_id in enumerate(("robot_1", "robot_2"))])
    np.testing.assert_allclose(frozen.logits(features), expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError, match="role weight hash mismatch"):
        FrozenDualPureNN.load(output["path"], expected_sha256=output["sha256"],
                              expected_role_sha256={"robot_1": "0" * 64,
                                                    "robot_2": output["role_sha256"]["robot_2"]})


def test_policy_inventory_exposes_both_immutable_role_bindings():
    inventory = warehouse.policy_artifacts()
    assert set(inventory) == {"raw", "regularized"}
    assert inventory["raw"]["policy_id"] == "warehouse-raw-v5"
    assert inventory["regularized"]["policy_id"] == "warehouse-regularized-v5"
    assert set(inventory["raw"]["role_sha256"]) == {"human", "ai"}
    assert inventory["regularized"]["program_paths"]


def test_formal_dual_selfplay_uses_both_actor_actions_without_override(monkeypatch):
    torch.set_num_threads(1)
    observed = []
    observed_rewards = []
    environment = trainer.WarehouseMultiAgentEnv
    def same_scored_environment(config):
        observed.append(config)
        env = environment(config)
        original_step = env.step
        def recorded_step(actions):
            result = original_step(actions)
            observed_rewards.append(result[1].copy())
            return result
        env.step = recorded_step
        return env
    monkeypatch.setattr(trainer, "WarehouseMultiAgentEnv", same_scored_environment)
    model = DualNN(hidden=32)
    episode, summary = collect_episode(model, seed=813, generator=torch.Generator().manual_seed(813),
                                       max_steps=20)
    assert episode["records"]
    assert summary["actor_decisions"] == 2 * summary["joint_steps"]
    assert summary["proxy_human_overrides"] == 0
    assert summary["post_policy_interventions"] == 0
    assert model.critic(torch.zeros(1, 2 * trainer.OBSERVATION_DIM)).shape == (1, 2)
    assert np.shape(episode["bootstrap"]) == (2,)
    assert len(observed_rewards) == len(episode["records"])
    for record, rewards in zip(episode["records"], observed_rewards, strict=True):
        np.testing.assert_allclose(record["reward"],
                                   [rewards["robot_1"], rewards["robot_2"]])
        assert np.shape(record["value"]) == (2,)
    assert len(observed) == 1
    assert observed[0].delivery_points == 10.0
    assert observed[0].robot_collision_points == -100.0
    assert observed[0].shutdown_points == -200.0


def test_matched_scenario_blocks_survive_policy_specific_early_shutdowns():
    seed = 6001
    raw = [{'seed': _scenario_seed(seed, 0), 'joint_steps': 40},
           {'seed': _scenario_seed(seed, 40), 'joint_steps': 80},
           {'seed': _scenario_seed(seed, 120), 'joint_steps': 120}]
    regularized = [{'seed': _scenario_seed(seed, 0), 'joint_steps': 120},
                   {'seed': _scenario_seed(seed, 120), 'joint_steps': 50},
                   {'seed': _scenario_seed(seed, 170), 'joint_steps': 70}]
    _validate_scenario_schedule(raw, seed, 240)
    _validate_scenario_schedule(regularized, seed, 240)
    assert _scenario_schedule_sha256(seed, 240) != _scenario_schedule_sha256(seed + 1, 240)
    with pytest.raises(RuntimeError, match='schedule drifted'):
        _validate_scenario_schedule(raw[:-1] +
            [{'seed': _scenario_seed(seed, 0), 'joint_steps': 120}], seed, 240)


def test_heldout_task_seeds_follow_formal_enrollment_mapping(monkeypatch):
    observed = []
    def fake_evaluate(_model, seeds):
        observed.append(tuple(seeds))
        return {'summary': {}, 'episodes': [
            {'seed': seed, 'deliveries': 5, 'collisions': 3,
             'shutdown_episode': False} for seed in seeds]}
    monkeypatch.setattr(trainer, 'evaluate', fake_evaluate)
    result = trainer.evaluate_heldout(object())
    assert len(observed) == 3
    for task, seeds in enumerate(observed, start=1):
        assert seeds == tuple(base * 10 + task for base in range(250000, 250100))
    assert len(result['episodes']) == 300
    assert result['episodes'][0]['seed'] == 250000
    assert result['episodes'][0]['scenario_seed'] == 2500001


def test_repeated_warehouse_run_cannot_touch_heldout_again(tmp_path):
    (tmp_path / 'warehouse_dual_v5_acceptance.json').write_text(
        json.dumps({'heldout_evaluated': True, 'passed': False}))
    with pytest.raises(RuntimeError, match='already evaluated'):
        trainer.run_experiment(output_dir=tmp_path)


def test_completed_warehouse_run_rechecks_full_receipts_and_can_install_later(
        tmp_path, monkeypatch):
    role_hashes = {variant: {'robot_1': char * 64, 'robot_2': char.upper() * 64}
                   for variant, char in [('raw', 'a'), ('regularized', 'b')]}
    acceptance = {
        'release': 'policylens-three-domain-20261014.v5.0',
        'heldout_evaluated': True, 'passed': True,
        'selected_pair_seed': 6001,
        'candidate_training': [{'pair_seed': 6001, 'variants': {
            variant: {'selected_role_sha256': role_hashes[variant],
                      'scenario_schedule_sha256': 'c' * 64}
            for variant in ('raw', 'regularized')}}],
        'role_sha256': role_hashes,
        'program_sha256': {'regularized': {
            'robot_1': 'd' * 64, 'robot_2': 'e' * 64}},
        'heldout_scenario_seed_rule': 'enrollment_seed_times_10_plus_task',
        'heldout_seed_ranges': {str(task): [250000, 250099]
                                for task in (1, 2, 3)},
        'heldout': _valid_heldout_evidence(),
        'final_program_audit': {'robot_1': {}, 'robot_2': {}},
    }
    selection = trainer._selection_from_acceptance(acceptance)
    (tmp_path / 'warehouse_dual_v5_acceptance.json').write_text(json.dumps(acceptance))
    heldout = acceptance['heldout']
    (tmp_path / 'warehouse_dual_v5_heldout_receipt.json').write_text(json.dumps({
        'status': 'completed', 'selection': selection, 'heldout': heldout,
        'heldout_sha256': sha256(json.dumps(heldout, sort_keys=True).encode()).hexdigest()}))
    audit = acceptance['final_program_audit']
    (tmp_path / 'warehouse_dual_v5_final_program_audit_receipt.json').write_text(
        json.dumps({'status': 'completed', 'selection': selection | {
            'final_program_audit_seeds': list(trainer.FINAL_PROGRAM_AUDIT_SEEDS)},
            'audit': audit, 'audit_sha256': sha256(
                json.dumps(audit, sort_keys=True).encode()).hexdigest()}))
    monkeypatch.setattr(trainer, '_validate_final_program_audit', lambda *_: None)
    installed = []
    monkeypatch.setattr(trainer, '_install_accepted_assets',
                        lambda receipt, directory: installed.append((receipt, directory)))
    assert trainer.run_experiment(output_dir=tmp_path, install=True) == acceptance
    assert installed == [(acceptance, tmp_path)]
    acceptance['candidate_training'][0]['variants']['raw'][
        'scenario_schedule_sha256'] = 'f' * 64
    (tmp_path / 'warehouse_dual_v5_acceptance.json').write_text(json.dumps(acceptance))
    with pytest.raises(RuntimeError, match='receipt and acceptance disagree'):
        trainer.run_experiment(output_dir=tmp_path)


def test_warehouse_heldout_receipt_is_claimed_before_evaluation_and_reusable_once(tmp_path):
    path = tmp_path / 'heldout_receipt.json'
    selection = {'release': 'v5', 'selected_pair_seed': 6001}
    trainer._start_heldout_once(path, selection)
    assert json.loads(path.read_text()) == {'status': 'started', 'selection': selection}
    with pytest.raises(RuntimeError, match='already exists'):
        trainer._start_heldout_once(path, selection)
    with pytest.raises(RuntimeError, match='incomplete'):
        trainer._reuse_completed_heldout(path, selection)
    evidence = _valid_heldout_evidence()
    trainer._complete_heldout(path, selection, evidence)
    assert trainer._reuse_completed_heldout(path, selection) == evidence
    with pytest.raises(RuntimeError, match='selected models changed'):
        trainer._reuse_completed_heldout(path, {'selected_pair_seed': 6002})
    corrupted = json.loads(path.read_text())
    corrupted['heldout']['raw']['episodes'][0] = {'collisions': 999}
    path.write_text(json.dumps(corrupted))
    with pytest.raises(RuntimeError, match='hash mismatch'):
        trainer._reuse_completed_heldout(path, selection)


def test_completed_heldout_receipt_rejects_self_consistent_wrong_task_seed(tmp_path):
    path = tmp_path / 'heldout_receipt.json'
    selection = {'selected_pair_seed': 6001}
    trainer._start_heldout_once(path, selection)
    evidence = _valid_heldout_evidence()
    evidence['raw']['tasks']['1']['episodes'][0]['seed'] = 999
    with pytest.raises(RuntimeError, match='task episode mapping'):
        trainer._complete_heldout(path, selection, evidence)
    assert json.loads(path.read_text())['status'] == 'started'


def test_started_warehouse_heldout_receipt_blocks_retraining(tmp_path):
    (tmp_path / 'warehouse_dual_v5_heldout_receipt.json').write_text(
        json.dumps({'status': 'started', 'selection': {'selected_pair_seed': 6001}}))
    with pytest.raises(RuntimeError, match='started receipt exists'):
        trainer.run_experiment(output_dir=tmp_path)


def test_final_program_audit_claims_fresh_development_block_once(tmp_path):
    assert set(trainer.FINAL_PROGRAM_AUDIT_SEEDS).isdisjoint(trainer.DEVELOPMENT_SEEDS)
    assert set(trainer.FINAL_PROGRAM_AUDIT_SEEDS).isdisjoint(
        set().union(*map(set, trainer.HELDOUT_SEEDS_BY_TASK.values())))
    path = tmp_path / 'final_program_audit.json'
    selection = {'selected_pair_seed': 6001,
                 'final_program_audit_seeds': list(trainer.FINAL_PROGRAM_AUDIT_SEEDS),
                 'selected_role_sha256': {'regularized': {
                     'robot_1': 'a' * 64, 'robot_2': 'b' * 64}},
                 'selected_program_sha256': {
                     'robot_1': 'c' * 64, 'robot_2': 'd' * 64}}
    trainer._start_final_program_audit_once(path, selection)
    with pytest.raises(RuntimeError, match='already exists'):
        trainer._start_final_program_audit_once(path, selection)
    with pytest.raises(RuntimeError, match='incomplete'):
        trainer._reuse_final_program_audit(path, selection)
    def evidence(role):
        global_metrics = {'rows': 1200, 'coverage': 1.0,
                          'agreement': .7, 'kl': .5, 'illegal_count': 0}
        feedback = {'rows': 300, 'coverage': .25,
                    'agreement': .9, 'kl': .2, 'illegal_count': 0}
        explanation = ({'rows': 300, 'coverage': .25,
                        'agreement': .95, 'kl': .2, 'illegal_count': 0}
                       if role == 'robot_2' else
                       {'rows': 0, 'coverage': 0.0, 'agreement': 0.0,
                        'kl': None, 'illegal_count': 0})
        validation = {'global': global_metrics, 'feedback': feedback,
                      'explanation': explanation, 'feedback_passed': True,
                      'explanation_passed': role == 'robot_2'}
        return {'audit_source': 'final_independent_development_rollouts',
                'role': role,
                'development_seeds': list(trainer.FINAL_PROGRAM_AUDIT_SEEDS),
                'actor_role_sha256': selection['selected_role_sha256']['regularized'][role],
                'program_sha256': selection['selected_program_sha256'][role],
                'sample_count': 1200, 'program_depth': 5,
                'program_leaf_count': 16, 'global': global_metrics,
                'feedback': feedback, 'explanation': explanation,
                'feedback_eligible': True,
                'explanation_eligible': role == 'robot_2',
                'passed': True, 'validation': validation}
    audit = {role: evidence(role) for role in trainer.AGENTS}
    trainer._complete_final_program_audit(path, selection, audit)
    assert trainer._reuse_final_program_audit(path, selection) == audit
    with pytest.raises(RuntimeError, match='selected assets changed'):
        trainer._reuse_final_program_audit(path, {'selected_pair_seed': 6002})
    corrupted = json.loads(path.read_text())
    corrupted['audit']['robot_1']['feedback_eligible'] = False
    path.write_text(json.dumps(corrupted))
    with pytest.raises(RuntimeError, match='hash mismatch'):
        trainer._reuse_final_program_audit(path, selection)


def test_final_program_receipt_rejects_claimed_pass_without_numeric_gate(tmp_path):
    path = tmp_path / 'audit.json'
    selection = {'final_program_audit_seeds': list(trainer.FINAL_PROGRAM_AUDIT_SEEDS),
                 'selected_role_sha256': {'regularized': {
                     'robot_1': 'a' * 64, 'robot_2': 'b' * 64}},
                 'selected_program_sha256': {
                     'robot_1': 'c' * 64, 'robot_2': 'd' * 64}}
    trainer._start_final_program_audit_once(path, selection)
    with pytest.raises(RuntimeError, match='incomplete'):
        trainer._complete_final_program_audit(path, selection, {})
    assert json.loads(path.read_text())['status'] == 'started'


def test_pair_selection_uses_passing_model_quality_before_program_audit():
    candidates = [
        {'seed': 6001, 'development_passed': True, 'min_quality': 4.0,
         'selected_program_audit': {'robot_2': {'explanation_eligible': True}}},
        {'seed': 6002, 'development_passed': True, 'min_quality': 6.0,
         'selected_program_audit': {'robot_2': {'explanation_eligible': False}}},
        {'seed': 6003, 'development_passed': False, 'min_quality': 9.0,
         'selected_program_audit': {'robot_2': {'explanation_eligible': True}}},
    ]
    selected, passed = trainer._choose_quality_pair(candidates)
    assert passed and selected['seed'] == 6002


def test_every_regularized_pair_needs_both_role_feedback_before_final_audit():
    candidates = [
        {"regularized": {"report": {"feedback_rows": {
            "robot_1": 10, "robot_2": 2}}}},
        {"regularized": {"report": {"feedback_rows": {
            "robot_1": 3, "robot_2": 1}}}},
        {"regularized": {"report": {"feedback_rows": {
            "robot_1": 8, "robot_2": 0}}}},
    ]
    assert not trainer._all_candidates_received_regularized_feedback(candidates)
    candidates[-1]["regularized"]["report"]["feedback_rows"]["robot_2"] = 1
    assert trainer._all_candidates_received_regularized_feedback(candidates)


def test_fresh_warehouse_audit_cannot_use_arbitrary_output_directory(tmp_path):
    with pytest.raises(RuntimeError, match="canonical deployment directory"):
        trainer.run_experiment(output_dir=tmp_path, enable_heldout=True)
    assert not any(tmp_path.iterdir())


def test_raw_ppo_changes_each_actor_and_has_zero_program_feedback():
    torch.set_num_threads(1)
    torch.manual_seed(902)
    model = DualNN(hidden=32)
    initial = deepcopy(model.state_dict())
    before = _weight_hashes(model)
    after, programs, report = train_variant(initial, pair_seed=902, variant="raw",
                                            joint_steps=100, development_seeds=(240000,))
    assert programs == {}
    assert report["joint_steps"] == 100
    assert report["feedback_rows"] == {"robot_1": 0, "robot_2": 0}
    assert all(before[agent_id] != _weight_hashes(after)[agent_id]
               for agent_id in ("robot_1", "robot_2"))
    assert all(report["actor_weight_delta_l2"][role] > 0
               and report["actor_weight_changed_parameter_count"][role] > 0
               for role in trainer.AGENTS)


def test_warehouse_checkpoint_rejects_old_collision_reward_contract(tmp_path):
    torch.manual_seed(902)
    model = DualNN(hidden=32)
    checkpoint = tmp_path / "old_reward.pt"
    torch.save({"pair_seed": 902, "variant": "raw", "joint_steps": 100,
                "scenario_schedule_version": trainer.SCENARIO_SCHEDULE_VERSION,
                "training_reward_version": "participant-score-v3-role-credit-shutdown-minus200",
                "bc_architecture_version": trainer.BC_ARCHITECTURE_VERSION,
                "initial_role_sha256": _weight_hashes(model),
                "complete": False}, checkpoint)
    with pytest.raises(RuntimeError, match="another experiment"):
        train_variant(deepcopy(model.state_dict()), pair_seed=902, variant="raw",
                      joint_steps=100, development_seeds=(240000,),
                      checkpoint_path=checkpoint)


def test_regularized_checkpoint_rejects_previous_program_method(tmp_path):
    model = DualNN(hidden=32)
    checkpoint = tmp_path / "old_program.pt"
    torch.save({"pair_seed": 902, "variant": "regularized", "joint_steps": 100,
                "scenario_schedule_version": trainer.SCENARIO_SCHEDULE_VERSION,
                "training_reward_version": trainer.TRAINING_REWARD_VERSION,
                "bc_architecture_version": trainer.BC_ARCHITECTURE_VERSION,
                "program_feedback_version": "selective-rcpd-fixed-role-presets-124-scenarios.v4",
                "initial_role_sha256": _weight_hashes(model),
                "complete": False}, checkpoint)
    with pytest.raises(RuntimeError, match="another experiment"):
        train_variant(deepcopy(model.state_dict()), pair_seed=902,
                      variant="regularized", joint_steps=100,
                      development_seeds=(240000,), checkpoint_path=checkpoint)


def test_program_feedback_needs_current_and_sampled_nn_action_agreement():
    class Program:
        def covered(self, sample, *, actor_action, purpose):
            assert purpose == 'feedback'
            assert actor_action == 'UP'
            return {'probabilities': {'UP': 1., 'DOWN': 0., 'LEFT': 0.,
                                      'RIGHT': 0., 'WAIT': 0.}}

    logits = torch.tensor([[3., 1., 0., 0., 0.]], requires_grad=True)
    mask = torch.ones(1, 5)
    obs = torch.zeros(1, trainer.OBSERVATION_DIM)
    blocked, rows, _ = trainer._program_loss(
        logits, mask, obs, torch.tensor([1]), Program(), 0.9)
    assert rows == 0 and float(blocked) == 0.0
    allowed, rows, kl = trainer._program_loss(
        logits, mask, obs, torch.tensor([0]), Program(), 0.9)
    assert rows == 1 and kl >= 0.0 and float(allowed.detach()) >= 0.0


def test_program_extraction_uses_disjoint_fixed_training_scenarios(monkeypatch):
    class Program:
        report = {"validation": {"feedback_passed": True},
                  "split_episode_sha256": {}}

        def sha256(self):
            return "a" * 64

    captured = {}
    def fit(role, fit_rows, calibration_rows, eligibility_rows, validation_rows,
            oracle, **kwargs):
        captured[role] = (fit_rows, calibration_rows, eligibility_rows,
                          validation_rows, kwargs)
        assert all(np.isclose(sum(oracle(row)), 1.0)
                   for rows in (fit_rows, calibration_rows,
                                eligibility_rows, validation_rows)
                   for row in rows)
        return Program()

    monkeypatch.setattr(trainer, "fit_selective_program", fit)
    model = DualNN(hidden=16)
    # Buffer can contain older scenarios; only the most recent 124 may enter
    # extraction, and every scenario must stay in exactly one split.
    rows = [{"episode": str(82000 + index),
             "obs": np.zeros(trainer.OBSERVATION_DIM, dtype=np.float32),
             "mask": np.ones(len(trainer.ACTIONS), dtype=np.float32)}
            for index in range(150)]
    program, extracted = trainer._extract_selective_program(
        model, "robot_2", rows, pair_seed=6001, joint_steps=180000)
    assert program is not None
    fit_rows, calibration_rows, eligibility_rows, validation_rows, options = captured["robot_2"]
    parts = (fit_rows, calibration_rows, eligibility_rows, validation_rows)
    assert tuple(map(len, parts)) == trainer.PROGRAM_SPLIT_EPISODES == (50, 25, 25, 24)
    episode_sets = [set(row["episode"] for row in part) for part in parts]
    assert len(set.union(*episode_sets)) == 124
    assert all(not (left & right) for index, left in enumerate(episode_sets)
               for right in episode_sets[index + 1:])
    assert min(set.union(*episode_sets)) == str(82026)
    assert max(set.union(*episode_sets)) == str(82149)
    assert options["hard_weight"] == 0.0
    assert extracted["training_split_episode_counts"] == dict(zip(
        ("fit", "calibration", "eligibility", "validation"),
        trainer.PROGRAM_SPLIT_EPISODES, strict=True))


def test_zero_action_coverage_counts_optimizer_row_presentations():
    torch.set_num_threads(1)
    model = DualNN(hidden=16)
    episode, _ = collect_episode(model, seed=813,
                                 generator=torch.Generator().manual_seed(813),
                                 max_steps=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    count = len(episode["records"])
    update = trainer.ppo_update(model, optimizer, [episode],
                                {role: None for role in trainer.AGENTS},
                                progress=.90, epochs=2)
    assert update["feedback_window_actor_rows"] == {
        role: 2 * count for role in trainer.AGENTS}
    assert update["zero_action_coverage_rows"] == {
        role: 2 * count for role in trainer.AGENTS}
    assert update["physical_no_legal_action_rows"] == {
        role: 0 for role in trainer.AGENTS}
    assert update["feedback_rows"] == {role: 0 for role in trainer.AGENTS}


def test_selected_program_report_matches_selected_checkpoint(monkeypatch):
    class Program:
        def __init__(self, marker):
            self.report = {"marker": marker}

    counter = {"extraction": 0}
    def fake_episode(_model, *, seed, generator, max_steps):
        return {"records": [], "bootstrap": np.zeros(2)}, {
            "seed": seed, "joint_steps": 1, "actor_decisions": 2}

    def fake_update(_model, _optimizer, _episodes, _programs, *, progress):
        return {"feedback_rows": {role: 0 for role in trainer.AGENTS},
                "feedback_kl": {role: 0.0 for role in trainer.AGENTS}}

    def fake_extract(_model, role, _rows, *, pair_seed, joint_steps):
        if role == "robot_1":
            counter["extraction"] += 1
        return Program(counter["extraction"]), {"joint_steps": joint_steps}

    def fake_evaluate(_model, _seeds):
        quality = 10.0 if counter["extraction"] == 6 else 5.0
        return {"summary": {"quality": quality, "passed": True}, "episodes": []}

    monkeypatch.setattr(trainer, "collect_episode", fake_episode)
    monkeypatch.setattr(trainer, "ppo_update", fake_update)
    monkeypatch.setattr(trainer, "_extract_selective_program", fake_extract)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    model = DualNN(hidden=16)
    _, programs, report = train_variant(deepcopy(model.state_dict()),
                                        pair_seed=902, variant="regularized",
                                        joint_steps=100,
                                        development_seeds=(240000,))
    assert report["selected_development"]["joint_steps"] == 96
    assert report["rcpd"]["robot_1"] == programs["robot_1"].report == {"marker": 6}
    assert report["rcpd"]["robot_2"] == programs["robot_2"].report == {"marker": 6}
    assert counter["extraction"] == 7


def test_explicit_v5_binding_drives_human_play_and_demo(monkeypatch, tmp_path):
    torch.manual_seed(1301)
    model = DualNN(hidden=32)
    artifact = export_actor(model, tmp_path / "pair.npz", pair_seed=1301, variant="raw")
    config = deepcopy(warehouse.V5_CONFIG)
    record = config["variants"]["raw"]
    record["actor_path"] = artifact["path"]
    record["actor_sha256"] = artifact["sha256"]
    record["role_sha256"] = artifact["role_sha256"]
    monkeypatch.setattr(warehouse, "V5_CONFIG", config)
    _fixture_actor_hash_only(monkeypatch, config)
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    try:
        state = warehouse.initial_state(77, 1, policy_id="warehouse-raw-v5")
        assert state["scenario_seed"] == 771
        assert warehouse.initial_state(78, 1, policy_id="warehouse-raw-v5")["scenario_seed"] == 781
        assert warehouse.audited_program_evidence(state) is None
        assert state["role_sha256"] == {"human": artifact["role_sha256"]["robot_1"],
                                        "ai": artifact["role_sha256"]["robot_2"]}
        actor = warehouse._actor(state["policy_id"])
        env = warehouse._restore(state)
        assert env.state.participant_controlled_agent_id == "robot_1"
        actions, _ = actor.act(env.state, base_seed=state["scenario_seed"])
        decision = warehouse.decide(state)
        assert decision["policy_id"] == state["policy_id"]
        assert decision["actor_sha256"] == artifact["sha256"]
        assert decision["action"] == actions["robot_2"].lower()
        after = warehouse.step(state, "wait", decision)
        assert after["last_actions"]["ai"] == actions["robot_2"].lower()
        tampered = deepcopy(state)
        tampered["role_sha256"]["ai"] = "0" * 64
        with pytest.raises(RuntimeError, match="binding mismatch"):
            warehouse.decide(tampered)
        demo = warehouse.demonstration("warehouse-raw-v5")
        assert demo["provenance"]["policy_id"] == "warehouse-raw-v5"
        assert demo["provenance"]["role_sha256"] == state["role_sha256"]
        assert len(demo["frames"]) > 1
        (tmp_path / "pair.npz").unlink()
        with pytest.raises(RuntimeError, match="artifact missing"):
            warehouse.decide(state)
    finally:
        warehouse._actor.cache_clear()
        warehouse.demonstration.cache_clear()


def test_explicit_v5_missing_artifact_fails_closed(monkeypatch, tmp_path):
    config = deepcopy(warehouse.V5_CONFIG)
    config["variants"]["raw"]["actor_path"] = str(tmp_path / "absent.npz")
    config["variants"]["raw"]["actor_sha256"] = "a" * 64
    monkeypatch.setattr(warehouse, "V5_CONFIG", config)
    warehouse._actor.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="artifact missing"):
            warehouse.initial_state(77, 1, policy_id="warehouse-raw-v5")
    finally:
        warehouse._actor.cache_clear()


def test_v5_collision_physics_keeps_neural_proposal(monkeypatch, tmp_path):
    torch.manual_seed(1601)
    model = DualNN(hidden=32)
    with torch.no_grad():
        for parameter in model.actors["robot_2"].parameters():
            parameter.zero_()
        model.actors["robot_2"][4].bias[2] = 40.0  # LEFT
    artifact = export_actor(model, tmp_path / "collision_pair.npz",
                            pair_seed=1601, variant="raw")
    config = deepcopy(warehouse.V5_CONFIG)
    config["variants"]["raw"].update(actor_path=artifact["path"],
                                      actor_sha256=artifact["sha256"],
                                      role_sha256=artifact["role_sha256"])
    monkeypatch.setattr(warehouse, "V5_CONFIG", config)
    _fixture_actor_hash_only(monkeypatch, config)
    warehouse._actor.cache_clear()
    try:
        state = warehouse.initial_state(77, 1, policy_id="warehouse-raw-v5")
        env = warehouse._restore(state)
        env.state.by_id("robot_1").position = (1, 2)
        env.state.by_id("robot_2").position = (1, 0)
        masked_actions, masked_distributions = warehouse._actor("warehouse-raw-v5").act(
            env.state, base_seed=state["scenario_seed"])
        assert masked_distributions["robot_2"].action_mask[2] == 0.0
        assert masked_actions["robot_2"] != "LEFT"
        env.state.by_id("robot_1").position = (1, 0)
        env.state.by_id("robot_2").position = (1, 2)
        state = warehouse._state(env, 1, 77, policy_id="warehouse-raw-v5")
        decision = warehouse.decide(state)
        assert decision["action"] == "left"
        after = warehouse.step(state, "right", decision)
        assert after["last_actions"] == {"human": "right", "ai": "left"}
        assert after["collision_events"] > state["collision_events"]
    finally:
        warehouse._actor.cache_clear()
