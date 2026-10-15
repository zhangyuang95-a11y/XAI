"""Money boundaries, authoritative scores, frozen promises, and private payout data."""
import json
import pytest
from study_v3 import rewards, prolific
from study_v3.config import Settings
from study_v3.store import Store
from tests.test_study_v3_store import Flow
from tests.test_prolific_enrolment import store, payload

@pytest.mark.parametrize('domain',['warehouse','pong','kitchen'])
def test_bonus_bounds_and_rounding(domain):
    rule=rewards.policy(domain);low,high=rewards.BOUNDS[domain];span=high-low
    for score,expected in [(low-100,0),(low,0),(low+span/2,10),(high,20),(high+100,20),
                           (low+span*.024,0),(low+span*.025,1)]:
        assert rewards.task_bonus(score,rule)==expected
    for invalid in [float('nan'),float('inf')]:
        with pytest.raises(ValueError):rewards.task_bonus(invalid,rule)

@pytest.mark.parametrize('group',['A','B'])
def test_saved_scores_restart_and_legacy(tmp_path,group):
    settings=Settings(database=str(tmp_path/'rewards.db'));s=Store(settings);f=Flow(s,'pong',group)
    assert f.view['rewards']['base_pence']==300
    assert f.view['rewards']['earned_bonus_pence']==0
    f.finish_demo()
    assert f.view['rewards']['completed_tasks']==[]
    # Server-owned fixture: finished scores across all tasks, plus the active
    # score must never contribute. Client-supplied scores are not used.
    with s.db.transaction() as db:
        db.execute('UPDATE pl3_runs SET score_json=? WHERE id=?',(json.dumps({'task_score':65}),f.view['run_id']))
    assert s.view(f.token,f.view['instance_id'])['rewards']['earned_bonus_pence']==0
    f.finish_task()
    f.command('next');f.finish_task();f.command('next');f.finish_task()
    with s.db.transaction() as db:
        for task,score in [(1,45),(2,55),(3,65)]:
            db.execute('UPDATE pl3_runs SET score_json=? WHERE instance_id=? AND task=?',
                       (json.dumps({'task_score':score}),f.view['instance_id'],task))
    expected=s.view(f.token,f.view['instance_id'])['rewards']
    assert [t['bonus_pence'] for t in expected['completed_tasks']]==[0,10,20]
    assert expected['total_pence']==330
    s.db.close();s=Store(settings)
    assert s.view(f.token,f.view['instance_id'])['rewards']==expected
    assert len(s.export()['reward_settings'])==1
    with s.db.transaction() as db:db.execute('DELETE FROM pl3_reward_settings WHERE instance_id=?',(f.view['instance_id'],))
    # Pre-feature sessions lack a snapshot and must keep their old promise,
    # even when create/recovery is called again under the new release.
    _,restored=s.create({'participant_id':f.name,'domain':'pong','mode':'test','consent':True},f.token,admin=True)
    assert restored['rewards'] is None
    s.db.close()

def test_private_payment_export_and_immutable_policy(store):
    token,view=prolific.enrol(store,payload(1))
    before=prolific.payment_records(store)[0]
    assert before['rewards']['currency']=='GBP' and before['rewards']['maximum_total_pence']==360
    assert before['rewards']['payment_status']=='not_tracked_here'
    with store.db.transaction() as db:
        row=db.one('SELECT policy_json FROM pl3_reward_settings WHERE instance_id=?',(view['instance_id'],))
        rule=json.loads(row['policy_json']);rule['base_pence']=280;rule['version']='gbp-280-base-20-per-task-v1'
        db.execute('UPDATE pl3_reward_settings SET policy_json=? WHERE instance_id=?',(json.dumps(rule),view['instance_id']))
    assert prolific.resume(store,token)['rewards']['base_pence']==280
    assert prolific.payment_records(store)[0]['rewards']['base_pence']==280
    with store.db.transaction() as db:db.execute('DELETE FROM pl3_reward_settings WHERE instance_id=?',(view['instance_id'],))
    assert 'rewards' not in prolific.payment_records(store)[0]
