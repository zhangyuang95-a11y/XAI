"""Full-length manual task playability; not scientific model-quality evaluation.

Only the public manual-preview seeds are used. The human waits while the bound
frozen Actor controls its teammate. Reaching a terminal frame here means the UI
can finish a task, not that the team achieved a research quality threshold.
"""
from time import perf_counter

import pytest

from runtime_v642.contracts import RuntimeUnavailable
from ui.model_preview_server import Preview, SEEDS


@pytest.fixture(scope='module')
def preview(tmp_path_factory):
    instance = Preview(tmp_path_factory.mktemp('three-manual-tasks') / 'manual.sqlite3')
    yield instance
    instance.close()


@pytest.mark.parametrize('domain', ['warehouse', 'pong', 'kitchen'])
@pytest.mark.parametrize('variant', ['raw', 'regularized'])
def test_frozen_teammate_can_finish_all_three_manual_tasks(preview, domain, variant, record_property):
    began = perf_counter()
    binding = preview.catalog.binding(domain, variant)
    adapter = preview.adapters[domain]
    task_lengths = []
    for task in (1, 2, 3):
        initial = preview.start(dict(domain=domain, variant=variant, task=task, seed=SEEDS[domain][0]))
        token = initial['session_token']
        assert initial['state']['task'] == task
        assert initial['state']['turn'] == 0
        assert initial['binding'] == binding.to_dict()

        empty_replay = preview.export(token)
        assert empty_replay['independently_replayed'] is True
        assert empty_replay['replay']['records'] == []
        assert empty_replay['replay']['final_state'] == empty_replay['replay']['initial_state']

        actual = initial
        checked_nn_frames = {}
        horizon = initial['state']['max_turns']
        for turn in range(horizon):
            assert actual['state']['terminal'] is False
            # Recompute the bound NN at the start and the halfway point. The
            # check uses exact saved pre-action state, never a display snapshot.
            if turn in (0, horizon // 2):
                with preview.runtime.request():
                    _, before, _, saved_binding = preview._session(token)
                    assert saved_binding == binding
                    checked_nn_frames[turn] = adapter.decide(before, binding)['action']

            actual = preview.step(token, dict(command_id=f'three-task-{task}-{turn}',
                expected_turn=turn, human_action='wait'))
            assert actual['state']['turn'] == turn + 1
            assert actual['state']['task'] == task
            assert actual['binding'] == binding.to_dict()
            assert actual['state']['binding_identity_sha256'] == binding.identity_sha256
            assert actual['last_action']['human'] == 'wait'
            assert actual['last_action']['two_NN'] is False
            assert actual['last_action']['ai_action_overrides'] == 0
            assert actual['last_action']['execution_source'] == 'bound_frozen_NN_with_physical_mask'
            if turn in checked_nn_frames:
                assert actual['last_action']['ai'] == checked_nn_frames[turn]
            if actual['state']['terminal']:
                break

        assert actual['state']['terminal'] is True
        assert 1 <= actual['state']['turn'] <= horizon
        restored = preview.view(token)
        assert restored['state'] == actual['state']
        assert restored['binding'] == binding.to_dict()

        terminal_replay = preview.export(token)
        assert terminal_replay['independently_replayed'] is True
        assert terminal_replay['human_study_sample'] is False
        assert terminal_replay['scientific_evaluation'] is False
        replay = terminal_replay['replay']
        assert replay['binding'] == binding.to_dict()
        assert replay['final_state']['terminal'] is True
        assert replay['final_state']['turn'] == actual['state']['turn']
        assert len(replay['records']) == actual['state']['turn']
        for turn, expected_action in checked_nn_frames.items():
            assert replay['records'][turn]['decision']['action'] == expected_action
        task_lengths.append(actual['state']['turn'])

    record_property('manual_episode_joint_steps', ','.join(map(str, task_lengths)))
    record_property('elapsed_seconds', round(perf_counter() - began, 3))
    assert len(task_lengths) == 3


@pytest.mark.parametrize('domain', ['pong', 'kitchen'])
def test_invalid_human_command_does_not_damage_session_and_wait_can_follow(preview, domain):
    initial = preview.start(dict(domain=domain, variant='regularized', task=1, seed=SEEDS[domain][0]))
    token = initial['session_token']
    if domain == 'pong':
        actual = initial
        # Reach the physical left boundary with legal participant commands.
        while actual['state']['human']['x'] > 0:
            turn = actual['state']['turn']
            actual = preview.step(token, dict(command_id=f'reach-left-boundary-{turn}',
                expected_turn=turn, human_action='left'))
        illegal = 'left'
    else:
        assert initial['state']['interaction']['available'] is False
        illegal = 'interact'

    before = preview.view(token)
    turn = before['state']['turn']
    _, private_before, _, binding_before = preview._session(token)
    with pytest.raises(RuntimeUnavailable, match='illegal carried human command'):
        preview.step(token, dict(command_id='rejected-human-input', expected_turn=turn, human_action=illegal))
    assert preview.view(token) == before
    _, private_after, _, binding_after = preview._session(token)
    assert private_after == private_before
    assert binding_after == binding_before
    after = preview.step(token, dict(command_id='legal-wait-after-rejection', expected_turn=turn, human_action='wait'))
    assert after['state']['turn'] == turn + 1
    assert after['binding'] == before['binding']
    assert after['last_action']['ai_action_overrides'] == 0
