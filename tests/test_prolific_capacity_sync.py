from scripts.sync_prolific_capacity import plan
import pytest


def pair(n, domain='kitchen', group='A', status='RETURNED', completed=None):
    return ({'instance_id': str(n), 'submission_id': str(n), 'prolific_pid': 'p'+str(n),
             'domain': domain, 'group_code': group, 'stage': 'completed' if completed else 'task2',
             'completed': completed, 'released': None},
            {'id': str(n), 'participant_id': 'p'+str(n), 'status': status})


def test_terminal_incomplete_only_released_and_completed_timeouts_retained():
    pairs = [pair(1), pair(2,status='TIMED-OUT'), pair(3,status='ACTIVE'),
             pair(4,status='AWAITING REVIEW'), pair(5,status='TIMED-OUT',completed=123),
             pair(6,status='RETURNED',completed=456)]
    result = plan([r for r,s in pairs], [s for r,s in pairs])
    assert {r['instance_id'] for r in result['releases']} == {'1','2'}
    assert result['complete'] == 2
    assert result['unfinished'] == 2
    assert result['vacancies'] == 56


def test_release_is_not_counted_again_and_identity_is_checked():
    r,s = pair(1);r['released']=123
    result=plan([r],[s])
    assert result['releases']==[] and result['vacancies']==60
    with pytest.raises(ValueError):plan([r],[])
    with pytest.raises(ValueError):plan([r],[{**s,'participant_id':'wrong'}])


def test_each_cell_capped_independently():
    pairs=[pair(n,status='ACTIVE') for n in range(11)]
    with pytest.raises(ValueError):plan([r for r,s in pairs],[s for r,s in pairs])
