"""Synthetic KitchenStore export validation; these are not participant results."""
import copy
import csv
import json
from pathlib import Path

import pytest
from sqlalchemy import insert
from env.cooperative_kitchen import CooperativeKitchen,program_decision,ACTOR_IDS
from ui.cooperative_kitchen_store import KitchenStore,encode,runs,surveys,questions
from scripts.cooperative_kitchen.analyze_events import analyze_exports,write_outputs,ExportValidationError


def synthetic_export(wait_only=False):
    store=KitchenStore('sqlite:///:memory:',namespace='test',allow_sqlite=True)
    version={'fixture':'synthetic-analysis-v1-never-human-data'}
    rid='synthetic-run-0'
    run={'id':rid,'participant_id':'SYNTHETIC-PERSON','mode':'pilot','namespace':'test','phase':'complete','language':'zh','condition':'A','task_order':'XY','version':900,'episode_index':6,'episode_id':rid+'-ep6','retry_id':0,'versions':version,'created':1,'preset':'supply'}
    with store.transaction() as db:
        db.execute(insert(runs).values(id=rid,namespace='test',token_hash='synthetic-token',document=encode(run),version=900,created=1,updated=900))
        for index,phase in enumerate(['practice']+['task1']*3+['task2']*3):
            env=CooperativeKitchen();eid=f'{rid}-ep{index}'
            first=env.snapshot()
            ep={'id':eid,'run_id':rid,'index':index,'phase':phase,'scenario_id':'base_empty','scenario':{'scenario_id':'base_empty','seed':0},'done':False,'summary':None,'snapshot':first,'attempt_id':f'synthetic-attempt-{index}','created':index+2,'versions':version}
            store.save_episode(db,ep);store.save_frame(db,eid,first,env.public_view())
            while not env.state['done']:
                before=env.snapshot();t=env.state['turn']
                human=['INTERACT','LEFT','LEFT','WAIT'][t] if t<4 else program_decision(env,'human')['action']
                ai='WAIT' if t<4 else program_decision(env,'ai')['action']
                if wait_only:human=ai='WAIT'
                result=env.step({'human':human,'ai':ai})
                after=env.snapshot();public=env.public_view()
                store.save_frame(db,eid,after,public)
                store.event(db,rid,eid,f'{eid}-step-{t}','joint_step',{'before':before,'after':after,'human_command':human,'proposed_actions':result['proposed_actions'],'actual_actions':result['actual_actions'],'distributions':{'ai':{'chosen_action':ai,'policy_kind':'synthetic_program_fixture'}},'events':result['events'],'versions':version})
            ep.update(done=True,snapshot=env.snapshot(),summary={'orders':env.state['orders'],'steps':env.state['turn'],'score':env.public_view()['score'],'completed':env.state['orders']>=2,'reason':env.state['reason'],'first_delivery':env.state['_first_serve_turn']})
            store.save_episode(db,ep)
            if index==1:
                qid='synthetic-question'
                db.execute(insert(questions).values(id=qid,run_id=rid,episode_id=eid,frame=0,status='complete',attempts=1,created=100,updated=101,document=encode({'versions':version,'snapshot':first,'kind':'why','language':'zh','question':'Synthetic fixture','answer':{'verified':True,'text':'SYNTHETIC ONLY'}})))
                for number,event,time in [(0,'shown',100.),(1,'closed',104.),(2,'shown',105.)]:
                    store.event(db,rid,eid,f'exposure-{number}','answer_exposure',{'question_id':qid,'event':event,'frame':0})
                    # Deterministic server receipt timestamps for the duration assertion.
                    from ui.cooperative_kitchen_store import events
                    from sqlalchemy import update
                    db.execute(update(events).where(events.c.operation_id==f'exposure-{number}').values(created=time))
        answers={f'p{i}':('UP' if i in (0,4,5) else 'WAIT') for i in range(8)}
        answers.update(cooperation_understanding='5',predictability='4',difficulty='3')
        db.execute(insert(surveys).values(run_id=rid,submitted=1000,document=encode({'answers':answers,'prediction_item_accuracy':.75,'counterfactual_item_accuracy':.5,'prediction_accuracy':.625,'versions':version})))
    text=store.export('jsonl');store.engine.dispose();return text


@pytest.fixture(scope='module')
def fixture_text():return synthetic_export()


def saved(tmp_path,text,name='events.jsonl'):
    path=tmp_path/name;path.write_text(text);return path


def records(text):return [json.loads(line) for line in text.splitlines()]
def dump(rows):return '\n'.join(encode(r) for r in rows)+'\n'


def test_real_store_export_analysis_has_rich_metrics_and_default_filter(tmp_path,fixture_text):
    path=saved(tmp_path,fixture_text)
    assert analyze_exports([path])[0]==[]
    participants,episodes,report=analyze_exports([path],include_test=True)
    assert len(participants)==1 and len(episodes)==7 and report['include_test']
    p=participants[0]
    assert p['six_rounds_complete'] and p['primary_eligible'] and p['research_data'] is False
    assert p['task2_mean_score']==92 and p['task1_mean_score']==92 and p['task2_minus_task1_descriptive']==0
    assert p['task2_orders']==6 and p['task2_steps_per_delivered_soup']==54
    assert p['prediction_accuracy']==.75 and p['counterfactual_accuracy']==.5
    assert p['qa_exposure_count']==2 and p['qa_exposure_seconds']==4
    assert p['qa_exposure_closed_intervals']==1 and p['qa_exposure_unclosed']==1
    assert p['human_wait_count']>0 and p['human_blocked_count']==6
    assert p['human_turn_in_place_count']>0 and p['human_invalid_interaction_count']==6
    assert p['task_handoff_count']==48 and p['task_handoff_latency_mean_steps']>0
    assert all(e['first_delivery_step'] is not None and e['handoff_count']==8 for e in episodes)


@pytest.mark.parametrize('damage',['summary_score','missing_frame','missing_step','mixed_versions','item_id','question_frame'])
def test_corrupted_records_are_rejected(tmp_path,fixture_text,damage):
    rows=records(fixture_text)
    if damage=='summary_score':next(r for r in rows if r['type']=='episode')['document']['summary']['score']+=1
    elif damage=='missing_frame':rows.remove(next(r for r in rows if r['type']=='frame' and r['turn']==3))
    elif damage=='missing_step':rows.remove(next(r for r in rows if r['type']=='event' and r['kind']=='joint_step'))
    elif damage=='mixed_versions':next(r for r in rows if r['type']=='event' and r['kind']=='joint_step')['document']['versions']={'different':True}
    elif damage=='question_frame':next(r for r in rows if r['type']=='question')['frame']=1
    else:
        event=next(r for r in rows if r['type']=='event' and r['kind']=='joint_step' and any(e['type']=='pickup' for e in r['document']['events']))
        next(e for e in event['document']['events'] if e['type']=='pickup')['item_id']='wrong-item'
    with pytest.raises(ExportValidationError):analyze_exports([saved(tmp_path,dump(rows))],include_test=True)


def test_partial_run_never_emits_primary_task2_score(tmp_path,fixture_text):
    rows=records(fixture_text);remove='synthetic-run-0-ep6'
    rows=[r for r in rows if r.get('episode_id')!=remove and not(r['type']=='episode' and r['id']==remove)]
    run=rows[0]['document'];run.update(phase='task2',episode_index=5,episode_id='synthetic-run-0-ep5')
    rows=[r for r in rows if r['type']!='survey']
    p=analyze_exports([saved(tmp_path,dump(rows))],include_test=True)[0][0]
    assert p['task2_rounds_complete']==2 and p['task2_mean_score'] is None and not p['primary_eligible']


def test_runs_and_retries_never_merge_even_for_same_participant(tmp_path,fixture_text):
    first=records(fixture_text);first[0]['document']['phase']='technical_retry_closed'
    second=records(fixture_text.replace('synthetic-run-0','synthetic-run-1'))
    second[0]['document']['retry_id']=1;second[0]['document']['previous_run_id']='synthetic-run-0'
    participants,_,_=analyze_exports([saved(tmp_path,dump(first+second))],include_test=True)
    assert len(participants)==2 and len({p['participant_id'] for p in participants})==1
    assert participants[0]['task2_mean_score'] is None and participants[1]['task2_mean_score']==92
    assert participants[0]['primary_exclusion_reason']=='technical_retry_closed'


def test_different_versions_and_duplicate_exports_are_rejected(tmp_path,fixture_text):
    other=fixture_text.replace('synthetic-run-0','synthetic-run-1').replace('synthetic-analysis-v1-never-human-data','synthetic-analysis-v2-never-human-data')
    with pytest.raises(ExportValidationError,match='Mixed frozen versions'):
        analyze_exports([saved(tmp_path,fixture_text+other)],include_test=True)
    with pytest.raises(ExportValidationError,match='Duplicate exported run'):
        analyze_exports([saved(tmp_path,fixture_text+fixture_text,'duplicate.jsonl')],include_test=True)


def test_csv_outputs_preserve_raw_input_and_refuse_overwrite(tmp_path,fixture_text):
    source=saved(tmp_path,fixture_text);destination=tmp_path/'analysis'
    report=write_outputs([source],destination,include_test=True)
    with (destination/'participants.csv').open() as handle:p=next(csv.DictReader(handle))
    assert p['task2_mean_score']=='92' and p['namespace']=='test'
    assert source.read_text()==fixture_text and report['inputs'][0]['sha256']
    with pytest.raises(ExportValidationError,match='already exists'):write_outputs([source],destination,include_test=True)


def test_no_soup_has_null_delivery_latency_and_efficiency(tmp_path):
    p,episodes,_=analyze_exports([saved(tmp_path,synthetic_export(wait_only=True))],include_test=True)
    assert p[0]['task2_mean_score']==-180 and p[0]['task2_orders']==0
    assert p[0]['task2_steps_per_delivered_soup'] is None and p[0]['task2_mean_first_delivery_step'] is None
    assert all(e['first_delivery_step'] is None and e['steps_per_delivered_soup'] is None and e['handoff_latency_mean_steps'] is None for e in episodes)


def test_namespace_keys_keep_equal_run_ids_separate(tmp_path,fixture_text):
    other=records(fixture_text);other[0]['namespace']='development';other[0]['document']['namespace']='development'
    participants,_,_=analyze_exports([saved(tmp_path,fixture_text+dump(other))],include_test=True)
    assert len(participants)==2 and {p['namespace'] for p in participants}=={'test','development'}


def test_complete_versioned_research_block_is_default_included(tmp_path,fixture_text):
    # Re-label only this isolated unit-test input; no research database is touched.
    rows=records(fixture_text);rows[0]['namespace']='pilot';rows[0]['document']['namespace']='pilot'
    participants,_,_=analyze_exports([saved(tmp_path,dump(rows))])
    assert len(participants)==1 and participants[0]['research_data'] and participants[0]['task2_mean_score']==92
