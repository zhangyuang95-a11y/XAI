"""Validate KitchenStore JSONL and export separate participant/run and episode CSVs.

Only pilot-mode records in pilot/confirmatory namespaces are included by default.
No statistics pool retries, participants, namespaces, modes, or code versions.
This script uses persisted evidence; it never advances or replays a live game.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

RESEARCH_NAMESPACES={'pilot','confirmatory'}
TASKS={'task1','task2'}
ACTIONS={'UP','DOWN','LEFT','RIGHT','INTERACT','WAIT'}
COUNTERS={'2,4','4,4'}
GAME_KEYS=('turn','orders','maxSteps','targetOrders','done','reason','map','actors','pot','counters','scenario_id','seed','preset')
BEHAVIOR_TYPES=('move','wait','blocked','turn_in_place','invalid_interaction')
ID_COLUMNS=['namespace','mode','participant_id','run_id','retry_id','condition','task_order','language','versions_sha256']
BEHAVIOR_COLUMNS=[f'{actor}_{kind}_count' for actor in ('human','ai') for kind in BEHAVIOR_TYPES]
QA_COLUMNS=['qa_requested_count','qa_exposure_count','qa_exposure_closed_intervals','qa_exposure_seconds','qa_exposure_unclosed','qa_exposure_unmatched_closes']
EPISODE_COLUMNS=ID_COLUMNS+['episode_id','attempt_id','episode_index','phase','scenario_id','scenario_seed','done','reason','orders','steps','score','completed','first_delivery_step','steps_per_delivered_soup','handoff_count','handoff_latency_mean_steps','onion_handoff_count','onion_handoff_latency_mean_steps','soup_handoff_count','soup_handoff_latency_mean_steps','initial_counter_pickups']+BEHAVIOR_COLUMNS+QA_COLUMNS
PARTICIPANT_COLUMNS=ID_COLUMNS+['phase','previous_run_id','research_data','task1_rounds_complete','task2_rounds_complete','six_rounds_complete','primary_eligible','primary_exclusion_reason','task1_mean_score','task2_mean_score','task2_minus_task1_descriptive','task1_orders','task2_orders','task2_completion_rate','task2_steps','task2_steps_per_delivered_soup','task2_mean_first_delivery_step','task2_rounds_with_delivery','prediction_accuracy','counterfactual_accuracy','all_prediction_items_accuracy','cooperation_understanding','predictability','difficulty','survey_submitted','task_handoff_count','task_handoff_latency_mean_steps']+BEHAVIOR_COLUMNS+QA_COLUMNS


class ExportValidationError(ValueError):
    pass


def require(condition,message):
    if not condition:raise ExportValidationError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def average(values):return mean(values) if values else None


def integer(value,minimum=0):return type(value) is int and value>=minimum


def public_projection(snapshot):
    result={key:snapshot[key] for key in GAME_KEYS}
    result['actors']=[{key:value for key,value in actor.items() if not key.startswith('_')} for actor in snapshot['actors']]
    return result


def _blocks(paths):
    for path in paths:
        current=None
        with Path(path).open(encoding='utf-8') as source:
            for number,line in enumerate(source,1):
                if not line.strip():continue
                try:row=json.loads(line)
                except (ValueError,TypeError) as error:raise ExportValidationError(f'{path}:{number}: invalid JSON') from error
                require(isinstance(row,dict),f'{path}:{number}: expected JSON object')
                if row.get('type')=='run':
                    if current is not None:yield current
                    current={'run':row.get('document'),'namespace':row.get('namespace'),'records':[]}
                else:
                    require(current is not None,f'{path}:{number}: record appears before its run block')
                    require(row.get('type') in {'episode','event','question','survey','frame'},f'{path}:{number}: unsupported record type')
                    current['records'].append(row)
        if current is not None:yield current


def identity(run,namespace):
    require(isinstance(run,dict),'Missing run document')
    for key in ('id','participant_id','mode','phase','versions','retry_id'):
        require(key in run,f'Missing run field {key}')
    require(namespace==run.get('namespace'),'Run namespace mismatch')
    require(namespace in {'pilot','confirmatory','development','test'},'Unknown namespace')
    require(run['mode'] in {'pilot','freeplay'},'Unknown run mode')
    require(integer(run['retry_id']),'Invalid retry ID')
    require(isinstance(run['versions'],dict) and bool(run['versions']),'Missing frozen run versions')
    return dict(namespace=namespace,mode=run['mode'],participant_id=run['participant_id'],run_id=run['id'],retry_id=run['retry_id'],condition=run.get('condition'),task_order=run.get('task_order'),language=run.get('language'),versions_sha256=digest(run['versions']))


def check_versions(document,run,where,*,required=False):
    if required:require('versions' in document,f'{where}: versions missing')
    if 'versions' in document:require(document['versions']==run['versions'],f'{where}: mixed versions')


def exposure_metrics(question_rows,event_rows):
    known={q['id']:q for q in question_rows}
    opened={};intervals=[];shown=unclosed=unmatched=0
    exposure=[e for e in event_rows if e.get('kind')=='answer_exposure']
    for event in sorted(exposure,key=lambda e:e['id']):
        doc=event['document'];qid=doc.get('question_id');timestamp=event.get('created')
        require(qid in known,'Exposure references unknown question')
        require(known[qid]['episode_id']==event.get('episode_id'),'Exposure crosses episode boundary')
        require(doc.get('frame')==known[qid]['frame'],'Exposure frame differs from question frame')
        require(isinstance(timestamp,(int,float)) and not isinstance(timestamp,bool) and math.isfinite(timestamp),'Invalid exposure timestamp')
        if doc.get('event')=='shown':
            shown+=1
            if qid in opened:unclosed+=1
            opened[qid]=float(timestamp)
        elif doc.get('event')=='closed':
            if qid not in opened:unmatched+=1;continue
            start=opened.pop(qid)
            require(timestamp>=start,'Answer exposure time runs backwards')
            intervals.append(float(timestamp)-start)
        else:raise ExportValidationError('Unknown answer exposure event')
    return dict(qa_requested_count=len(question_rows),qa_exposure_count=shown,qa_exposure_closed_intervals=len(intervals),qa_exposure_seconds=sum(intervals) if intervals else None,qa_exposure_unclosed=unclosed+len(opened),qa_exposure_unmatched_closes=unmatched)


def validate_episode(run,episode,frame_rows,joint_rows,question_rows,event_rows,base):
    ep=episode['document'];eid=ep['id']
    require(episode.get('id')==eid and episode.get('run_id')==run['id']==ep.get('run_id'),'Episode ownership mismatch')
    require(episode.get('episode_index')==ep.get('index') and episode.get('phase')==ep.get('phase'),'Episode header mismatch')
    require(ep.get('phase') in {'practice','task1','task2','freeplay'},'Unknown episode phase')
    require(integer(ep.get('index')) and isinstance(ep.get('attempt_id'),str),'Episode index/attempt missing')
    check_versions(ep,run,f'Episode {eid}',required=True)
    final=ep.get('snapshot',{});last_turn=final.get('turn')
    require(integer(last_turn),'Invalid final episode turn')
    require(len(frame_rows)==last_turn+1,f'{eid}: missing or duplicated frames')
    frames={}
    for frame in frame_rows:
        turn=frame.get('turn');require(integer(turn) and turn not in frames,f'{eid}: duplicate/invalid frame')
        snapshot,public=frame.get('snapshot',{}),frame.get('public',{})
        require(snapshot.get('turn')==public.get('turn')==turn,f'{eid}: inconsistent frame turn')
        require(isinstance(snapshot.get('environment_version'),str) and bool(snapshot['environment_version']),f'{eid}: environment version missing')
        require(integer(snapshot.get('orders')) and integer(snapshot.get('maxSteps'),1) and integer(snapshot.get('targetOrders'),1),f'{eid}: invalid score inputs')
        require(0<=snapshot['orders']<=snapshot['targetOrders'] and turn<=snapshot['maxSteps'],f'{eid}: progress exceeds limits')
        require(public.get('score')==100*snapshot['orders']-turn,f'{eid}: frame score mismatch')
        require(public_projection(snapshot)=={key:public.get(key) for key in GAME_KEYS},f'{eid}: public/snapshot state mismatch')
        frames[turn]=snapshot
    require(set(frames)==set(range(last_turn+1)),f'{eid}: frame sequence has gaps')
    require(frames[last_turn]==final,f'{eid}: final snapshot differs from final frame')
    for snapshot in frames.values():
        require(all(snapshot[key]==frames[0][key] for key in ('map','scenario_id','seed','preset')),f'{eid}: fixed scenario changed mid-round')
    require(ep.get('scenario_id')==frames[0]['scenario_id'],f'{eid}: episode scenario differs from frames')
    for question in question_rows:
        require(question['document'].get('snapshot')==frames[question['frame']],f'{eid}: question snapshot differs from its bound frame')
    environment_versions={(s.get('environment_version'),s['maxSteps'],s['targetOrders']) for s in frames.values()}
    require(len(environment_versions)==1,f'{eid}: environment version/limits changed mid-round')
    if base['namespace'] in RESEARCH_NAMESPACES and run['mode']=='pilot':
        require(final['maxSteps']==180 and final['targetOrders']==2,f'{eid}: research limits differ from frozen protocol')
        require(all(s['preset']=='supply' for s in frames.values()),f'{eid}: research roles were swapped')
    ordered=sorted(joint_rows,key=lambda row:row['document'].get('after',{}).get('turn',-1))
    require(len(ordered)==last_turn,f'{eid}: missing or duplicated joint steps')
    behavior=Counter();handoffs=[];pending={};deliveries=[];initial_pickups=0
    for turn,row in enumerate(ordered,1):
        doc=row['document'];check_versions(doc,run,f'Joint step {row["id"]}',required=True)
        require(doc.get('before')==frames[turn-1] and doc.get('after')==frames[turn],f'{eid}: joint-step state chain has gaps or differs from frames')
        for field in ('proposed_actions','actual_actions'):
            actions=doc.get(field,{})
            require(set(actions)=={'human','ai'} and all(a in ACTIONS for a in actions.values()),f'{eid}: invalid {field}')
        require(doc.get('human_command')==doc['proposed_actions']['human'],f'{eid}: human command mismatch')
        require(doc.get('events')==frames[turn].get('_last_events'),f'{eid}: frame/event mismatch')
        for event in doc['events']:
            kind,actor=event.get('type'),event.get('actor')
            if actor in {'human','ai'} and kind in BEHAVIOR_TYPES:behavior[f'{actor}_{kind}_count']+=1
            if kind=='serve':deliveries.append(turn)
            if kind not in {'drop','pickup'}:continue
            item_id=event.get('item_id');target=event.get('target',[]);counter=','.join(map(str,target))
            require(isinstance(item_id,str) and actor in {'human','ai'} and counter in COUNTERS,f'{eid}: handoff provenance missing')
            if kind=='drop':
                require(frames[turn]['_counter_item_ids'].get(counter)==item_id,f'{eid}: drop item ID mismatch')
                require(frames[turn-1]['counters'].get(counter) is None,f'{eid}: item dropped onto occupied counter')
                pending[counter]={'item_id':item_id,'actor':actor,'turn':turn,'kind':event.get('item')}
            else:
                require(frames[turn-1]['_counter_item_ids'].get(counter)==item_id,f'{eid}: pickup item ID mismatch')
                prior=pending.pop(counter,None)
                if prior:
                    require(prior['item_id']==item_id,f'{eid}: item identity changed during handoff')
                    if prior['actor']!=actor:handoffs.append({'steps':turn-prior['turn'],'kind':prior['kind']})
                else:
                    require(frames[0]['_counter_item_ids'].get(counter)==item_id,f'{eid}: pickup lacks a prior drop or initial placement')
                    initial_pickups+=1
    require(len(deliveries)==final['orders']-frames[0]['orders'],f'{eid}: delivered-order count differs from serve events')
    first=deliveries[0] if deliveries else None
    require(final.get('_first_serve_turn')==first,f'{eid}: first-delivery state mismatch')
    done=ep.get('done');require(type(done) is bool,f'{eid}: invalid completion flag')
    if ep['phase'] in TASKS:require(done==final['done'],f'{eid}: formal terminal state mismatch')
    if done:
        summary=ep.get('summary') or {}
        expected={'orders':final['orders'],'steps':last_turn,'score':100*final['orders']-last_turn,'completed':final['orders']>=final['targetOrders']}
        require(all(summary.get(key)==value for key,value in expected.items()),f'{eid}: episode summary/score mismatch')
        require(summary.get('first_delivery')==first,f'{eid}: summary first-delivery mismatch')
        if ep['phase'] in TASKS:require(summary.get('reason')==final['reason'],f'{eid}: terminal reason mismatch')
    else:require(ep.get('summary') is None,f'{eid}: unfinished episode has a final summary')
    qa=exposure_metrics(question_rows,event_rows)
    row={**base,'episode_id':eid,'attempt_id':ep['attempt_id'],'episode_index':ep['index'],'phase':ep['phase'],
         'scenario_id':ep.get('scenario_id'),'scenario_seed':final.get('seed'),'done':done,'reason':(ep.get('summary') or {}).get('reason'),
         'orders':final['orders'],'steps':last_turn,'score':100*final['orders']-last_turn,'completed':final['orders']>=final['targetOrders'],
         'first_delivery_step':first,'steps_per_delivered_soup':last_turn/final['orders'] if final['orders'] else None,
         'handoff_count':len(handoffs),'handoff_latency_mean_steps':average([h['steps'] for h in handoffs]),'initial_counter_pickups':initial_pickups,
         **{name:behavior[name] for name in BEHAVIOR_COLUMNS},**qa,'_handoffs':handoffs}
    for kind in ('onion','soup'):
        latency=[h['steps'] for h in handoffs if h['kind']==kind]
        row[f'{kind}_handoff_count']=len(latency);row[f'{kind}_handoff_latency_mean_steps']=average(latency)
    return row


def analyze_block(block):
    run=block['run'];base=identity(run,block['namespace']);records=block['records']
    grouped=defaultdict(list)
    for row in records:
        grouped[row['type']].append(row)
        if row['type']!='frame':require(row.get('run_id')==run['id'],'Record crosses run boundary')
        if 'document' in row:check_versions(row['document'],run,f'{row["type"]} record')
    episode_by_id={row.get('id'):row for row in grouped['episode']}
    require(len(episode_by_id)==len(grouped['episode']),'Duplicate episode IDs')
    indices=sorted(row['document'].get('index',-1) for row in grouped['episode'])
    require(indices==list(range(len(indices))),'Episode sequence has gaps or duplicates')
    require(run.get('episode_index')==len(indices)-1,'Run episode index differs from exported history')
    if indices:
        final_id=next(row['id'] for row in grouped['episode'] if row['document']['index']==indices[-1])
        require(run.get('episode_id')==final_id,'Run current episode differs from exported history')
    for kind in ('event','question'):
        require(len({row['id'] for row in grouped[kind]})==len(grouped[kind]),f'Duplicate {kind} IDs')
    require(len(grouped['survey'])<=1,'Duplicate survey')
    for row in [*grouped['event'],*grouped['question'],*grouped['frame']]:
        require(row.get('episode_id') is None or row['episode_id'] in episode_by_id,'Record references missing episode')
    for question in grouped['question']:
        check_versions(question['document'],run,'Question',required=True)
        require(integer(question.get('frame')) and question['frame']<=episode_by_id[question['episode_id']]['document']['snapshot']['turn'],'Question frame is outside episode')
    episode_rows=[]
    for eid,episode in sorted(episode_by_id.items(),key=lambda pair:pair[1]['document']['index']):
        events=[row for row in grouped['event'] if row.get('episode_id')==eid]
        episode_rows.append(validate_episode(run,episode,[row for row in grouped['frame'] if row['episode_id']==eid],
            [row for row in events if row['kind']=='joint_step'],[q for q in grouped['question'] if q['episode_id']==eid],events,base))
    tasks=[ep for ep in episode_rows if ep['phase'] in TASKS]
    expected_phases=['task1']*min(3,len(tasks))+['task2']*max(0,len(tasks)-3)
    require([ep['phase'] for ep in tasks]==expected_phases,'Formal task phases are out of order')
    for phase in TASKS:require(sum(ep['phase']==phase for ep in tasks)<=3,f'Formal phase {phase} has more than three rounds')
    one=[ep for ep in tasks if ep['phase']=='task1' and ep['done']]
    two=[ep for ep in tasks if ep['phase']=='task2' and ep['done']]
    six=len(one)==len(two)==3
    require(not two or len(one)==3,'Task2 exists without three completed Task1 rounds')
    primary=six and run['mode']=='pilot' and run['phase']!='technical_retry_closed'
    exclusion='' if primary else 'technical_retry_closed' if run['phase']=='technical_retry_closed' else 'non_research_mode' if run['mode']!='pilot' else 'six_complete_rounds_required'
    submitted=False;ratings={};accuracy={}
    if grouped['survey']:
        survey=grouped['survey'][0];check_versions(survey['document'],run,'Survey',required=True)
        submitted=survey.get('submitted') is not None
        if submitted:
            doc=survey['document'];answers=doc.get('answers',{})
            for field in ('prediction_item_accuracy','counterfactual_item_accuracy','prediction_accuracy'):
                value=doc.get(field)
                require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1,f'Submitted survey missing/invalid {field}')
                accuracy[field]=float(value)
            require(abs(accuracy['prediction_accuracy']-(accuracy['prediction_item_accuracy']+accuracy['counterfactual_item_accuracy'])/2)<1e-9,'Questionnaire subset scores disagree with eight-item score')
            for key in ('cooperation_understanding','predictability','difficulty'):
                value=answers.get(key,answers.get('understanding') if key=='cooperation_understanding' else None)
                require(str(value) in {str(n) for n in range(1,8)},f'Submitted survey invalid {key}')
                ratings[key]=int(value)
    task_handoffs=[h for ep in tasks for h in ep['_handoffs']]
    qa=exposure_metrics(grouped['question'],grouped['event'])
    task1_score=average([ep['score'] for ep in one]) if len(one)==3 else None
    task2_score=average([ep['score'] for ep in two]) if primary else None
    participant={**base,'phase':run['phase'],'previous_run_id':run.get('previous_run_id'),'research_data':base['namespace'] in RESEARCH_NAMESPACES and run['mode']=='pilot',
        'task1_rounds_complete':len(one),'task2_rounds_complete':len(two),'six_rounds_complete':six,'primary_eligible':primary,'primary_exclusion_reason':exclusion,
        'task1_mean_score':task1_score,'task2_mean_score':task2_score,'task2_minus_task1_descriptive':task2_score-task1_score if primary else None,
        'task1_orders':sum(e['orders'] for e in one),'task2_orders':sum(e['orders'] for e in two),
        'task2_completion_rate':average([float(e['completed']) for e in two]) if len(two)==3 else None,
        'task2_steps':sum(e['steps'] for e in two),'task2_steps_per_delivered_soup':sum(e['steps'] for e in two)/sum(e['orders'] for e in two) if sum(e['orders'] for e in two) else None,
        'task2_mean_first_delivery_step':average([e['first_delivery_step'] for e in two if e['first_delivery_step'] is not None]),'task2_rounds_with_delivery':sum(e['first_delivery_step'] is not None for e in two),
        'prediction_accuracy':accuracy.get('prediction_item_accuracy'),'counterfactual_accuracy':accuracy.get('counterfactual_item_accuracy'),'all_prediction_items_accuracy':accuracy.get('prediction_accuracy'),
        'survey_submitted':submitted,**ratings,'task_handoff_count':len(task_handoffs),'task_handoff_latency_mean_steps':average([h['steps'] for h in task_handoffs]),
        **{name:sum(e[name] for e in tasks) for name in BEHAVIOR_COLUMNS},**qa}
    return participant,episode_rows


def analyze_exports(paths,*,include_test=False):
    participants=[];episodes=[];seen=set();versions=set();excluded=[]
    for block in _blocks(paths):
        run=block['run'];base=identity(run,block['namespace'])
        key=(base['namespace'],base['mode'],base['run_id'],base['retry_id'])
        require(key not in seen,'Duplicate exported run; use one complete export snapshot per run')
        seen.add(key)
        if not include_test and not (base['namespace'] in RESEARCH_NAMESPACES and base['mode']=='pilot'):
            excluded.append({'namespace':base['namespace'],'run_id':base['run_id'],'mode':base['mode'],'reason':'not_research_data'});continue
        versions.add(base['versions_sha256'])
        require(len(versions)==1,'Mixed frozen versions across included runs; analyze each version separately')
        participant,rows=analyze_block(block);participants.append(participant);episodes.extend(rows)
    participants.sort(key=lambda r:(r['namespace'],r['mode'],r['participant_id'],r['run_id'],r['retry_id']))
    episodes.sort(key=lambda r:(r['namespace'],r['run_id'],r['episode_index']))
    report={'schema':'cooperative_kitchen_event_analysis_v1','included_runs':len(participants),'included_episodes':len(episodes),
        'primary_eligible_runs':sum(r['primary_eligible'] for r in participants),'include_test':include_test,'excluded_runs':excluded,
        'versions_sha256':next(iter(versions),None),'primary_metric':'Task2 three-round mean score, only after all six formal rounds validate',
        'data_label':'includes development/test/freeplay only by explicit request; inspect namespace/mode before research use' if include_test else 'pilot/confirmatory namespaces with pilot mode only',
        'missing_exposure_policy':'Only complete shown/closed intervals contribute seconds; unclosed intervals are censored',
        'survey_accuracy_source':'Server-persisted scoring against the frozen question bank; subset and overall consistency validated'}
    return participants,episodes,report


def _csv_value(value):
    if isinstance(value,str) and value[:1] in {'=','+','-','@'}:return "'"+value
    return value


def write_outputs(paths,output,*,include_test=False):
    participants,episodes,report=analyze_exports(paths,include_test=include_test)
    output=Path(output);targets=[output/name for name in ('participants.csv','episodes.csv','analysis_report.json')]
    require(not any(path.exists() for path in targets),'Analysis output already exists; choose a new output directory')
    report['inputs']=[]
    for path in paths:
        with Path(path).open('rb') as source:
            report['inputs'].append({'path':str(Path(path).resolve()),'sha256':hashlib.file_digest(source,'sha256').hexdigest()})
    output.mkdir(parents=True,exist_ok=True)
    for target,rows,columns in ((targets[0],participants,PARTICIPANT_COLUMNS),(targets[1],episodes,EPISODE_COLUMNS)):
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
            for row in rows:writer.writerow({key:_csv_value(row.get(key)) for key in columns})
    fd=os.open(targets[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2);handle.write('\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('exports',nargs='+',help='KitchenStore JSONL exports')
    p.add_argument('--output-dir',required=True);p.add_argument('--include-test',action='store_true',help='Explicitly include development/test/freeplay data; never treat it as participant evidence')
    a=p.parse_args()
    try:report=write_outputs(a.exports,a.output_dir,include_test=a.include_test)
    except (ExportValidationError,KeyError,TypeError,OSError) as error:p.exit(1,f'Analysis rejected: {error}\n')
    print(json.dumps({key:report[key] for key in ('included_runs','included_episodes','primary_eligible_runs','include_test')},ensure_ascii=False))

if __name__=='__main__':main()
