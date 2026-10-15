"""Bounded replacement-cohort capacity sync; no payments or new paid places."""
import argparse
import datetime
import json
from pathlib import Path
import time
from scripts.sync_prolific_capacity import Sync, TERMINAL

ROOT=Path(__file__).resolve().parents[1]

def cycle(sync, settings, directory):
    sid=settings['study_id']
    study=sync.request(sync.study_url)
    if study['reward']!=300 or study['total_available_places']!=6:
        raise ValueError('Replacement study budget changed')
    submissions={s['id']:s for s in sync.submissions()}
    url=sync.origin+'/api/prolific/admin/payments?study_id='+sid
    rows=sync.request(url,site=True)['records']
    for row in rows:
        sub=submissions.get(row['submission_id'])
        if not sub or sub['participant_id']!=row['prolific_pid']:
            raise ValueError('Missing or mismatched submission')
        if row['completed'] or row.get('released') or sub['status'] not in TERMINAL:continue
        sub=sync.request('https://api.prolific.com/api/v1/submissions/'+sub['id']+'/')
        if sub['status'] not in TERMINAL:continue
        sync.request(sync.origin+'/api/prolific/admin/release-slot','POST',{
            'study_id':sid,'instance_id':row['instance_id'],'submission_id':row['submission_id'],
            'submission_status':TERMINAL[sub['status']],
            'reason':'Verified terminal Prolific status during replacement-cohort reconciliation'},site=True)
    rows=sync.request(url,site=True)['records']
    cells=[]
    for q in settings['quotas']:
        matching=[r for r in rows if r['domain']==q['domain'] and r['group_code']==q['group'] and not r.get('released')]
        if len(matching)>q['places']:raise ValueError('Cell capacity exceeded')
        cells.append({**q,'occupied':len(matching),'completed':sum(bool(r['completed']) for r in matching)})
    occupied=sum(c['occupied'] for c in cells)
    completed=sum(c['completed'] for c in cells)
    if occupied==6 and study['status']=='ACTIVE':
        sync.request(sync.study_url+'transition/','POST',{'action':'PAUSE'})
    elif occupied<6 and study['status']=='PAUSED' and study.get('places_taken',6)<6:
        sync.request(sync.study_url+'transition/','POST',{'action':'START'})
    result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'study_id':sid,'cells':cells,'completed':completed,
        'platform_status':sync.request(sync.study_url)['status']}
    (directory/'sync-status.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return completed==6

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--hours',type=float,default=8);args=parser.parse_args()
    if not 0<args.hours<=8:raise SystemExit('Maximum eight-hour operational window')
    directory=Path(args.directory);settings=json.loads((directory/'cohort.json').read_text())
    sync=Sync(ROOT/'output/main-study');sync.sid=settings['study_id']
    sync.study_url='https://api.prolific.com/api/v1/studies/'+sync.sid+'/'
    until=time.monotonic()+args.hours*3600
    while time.monotonic()<until:
        try:
            if cycle(sync,settings,directory):break
        except Exception as error:
            print(type(error).__name__+': '+str(error),flush=True)
        time.sleep(30)
