"""One-game Prolific enrolment. Private identity mapping is not a research export.

Selection and session creation share the existing global enrollment transaction,
which serializes across processes on PostgreSQL and writers on SQLite.
"""
import re
import json
import secrets
import time
from . import rewards
from .store import StudyError, uid

CONSENT_VERSION='policylens-prolific-consent-20260924.v5-main'
CELLS=tuple((domain,group) for domain in ('warehouse','pong','kitchen') for group in ('A','B'))

def register_cohort(store,payload):
    """Immutable researcher-only quotas; never alter earlier enrollment records."""
    study=payload.get('study_id','')
    code=payload.get('completion_code','')
    quotas=payload.get('quotas')
    if (not isinstance(study,str) or not re.fullmatch(r'[0-9a-f]{24}',study)
        or study==store.settings.prolific_study_id
        or not isinstance(code,str) or not re.fullmatch(r'[A-Za-z0-9]{4,64}',code)
        or not isinstance(quotas,list) or not 1<=len(quotas)<=6):
        raise StudyError('invalid_prolific_cohort')
    targets={}
    for row in quotas:
        if not isinstance(row,dict):raise StudyError('invalid_prolific_cohort')
        cell=(row.get('domain'),row.get('group'))
        n=row.get('places')
        if cell not in CELLS or cell in targets or type(n) is not int or not 1<=n<=100:
            raise StudyError('invalid_prolific_cohort')
        targets[cell]=n
    normalized=[{'domain':d,'group':g,'places':targets[d,g]} for d,g in CELLS if (d,g) in targets]
    encoded=json.dumps(normalized,sort_keys=True)
    with store.db.transaction() as db:
        old=db.one('SELECT * FROM pl3_prolific_cohorts WHERE study_id=?',(study,))
        if old:
            if old['completion_code']!=code or old['quotas_json']!=encoded:
                raise StudyError('prolific_cohort_immutable',409)
        else:
            db.execute('INSERT INTO pl3_prolific_cohorts VALUES(?,?,?,?)',(study,code,encoded,time.time()))
    return {'study_id':study,'quotas':normalized,'places':sum(targets.values())}

def cohort(db,settings,study):
    if study==settings.prolific_study_id:
        return settings.prolific_completion_code,{cell:settings.prolific_places//6 for cell in CELLS}
    row=db.one('SELECT * FROM pl3_prolific_cohorts WHERE study_id=?',(study,))
    if not row:raise StudyError('wrong_prolific_study',403)
    return row['completion_code'],{(q['domain'],q['group']):q['places'] for q in json.loads(row['quotas_json'])}

def ready(store):
    s=store.settings
    return bool(store.ready and s.prolific_launch_confirmed
        and re.fullmatch(r'[0-9a-f]{24}',s.prolific_study_id)
        and re.fullmatch(r'[A-Za-z0-9]{4,64}',s.prolific_completion_code)
        and s.prolific_places>0 and s.prolific_places%6==0)

def identifiers(payload):
    values=[]
    for key in ('PROLIFIC_PID','STUDY_ID','SESSION_ID'):
        value=payload.get(key,'')
        if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{24}',value):
            raise StudyError('invalid_prolific_link')
        values.append(value)
    return values

def enrol(store,payload,token=None):
    if payload.get('consent') is not True or payload.get('age_21') is not True:
        raise StudyError('consent_required')
    if payload.get('consent_version')!=CONSENT_VERSION:
        raise StudyError('consent_version_changed',409)
    pid,study,submission=identifiers(payload)
    with store.db.transaction() as db:
        _,targets=cohort(db,store.settings,study)
        linked=db.one('SELECT * FROM pl3_prolific_links WHERE prolific_pid=?',(pid,))
        if linked:
            if linked['study_id']!=study or linked['submission_id']!=submission:
                raise StudyError('prolific_already_participated',409)
            # A copied URL or known Prolific ID is not a session credential.
            participant=store._participant(db,token)
            if participant['id']!=linked['participant_id']:raise StudyError('session_required',401)
            instance=store._instance(db,token,linked['instance_id'])
            return token,store._view(db,instance)
        if not ready(store):raise StudyError('study_not_ready',503)
        if token:
            try: current=store._participant(db,token)
            except StudyError: current=None
            if current and db.one('SELECT participant_id FROM pl3_prolific_links WHERE participant_id=?',(current['id'],)):
                raise StudyError('prolific_already_participated',409)
        if db.one('SELECT submission_id FROM pl3_prolific_links WHERE submission_id=?',(submission,)):
            raise StudyError('invalid_prolific_link',409)
        rows=db.all('SELECT i.domain,i.group_code,COUNT(*) AS n FROM pl3_prolific_links l JOIN pl3_instances i ON i.id=l.instance_id LEFT JOIN pl3_prolific_releases r ON r.instance_id=i.id WHERE l.study_id=? AND r.instance_id IS NULL GROUP BY i.domain,i.group_code',(study,))
        counts={(r['domain'],r['group_code']):r['n'] for r in rows}
        allocated=sum(counts.values())
        available=[cell for cell,n in targets.items() if counts.get(cell,0)<n]
        if not available:raise StudyError('prolific_full',409)
        least=min(counts.get(cell,0) for cell in available)
        domain,group=secrets.choice([cell for cell in available if counts.get(cell,0)==least])
        # Keep the original allocation history unique after a returned slot is
        # released. Releases affect capacity, never IDs or earlier consent.
        allocation_index=db.one('SELECT COALESCE(MAX(allocation_index),0)+1 AS next_index FROM pl3_prolific_links WHERE study_id=?',(study,))['next_index']
        # Only internal IDs enter game data or the semantic provider context.
        name='pl-'+uid()
        new_token,view=store.create({'domain':domain,'group':group,'participant_id':name,
            'mode':'pilot','consent':True,'language':'en'},token=None,_db=db,_prolific=True)
        db.execute('INSERT INTO pl3_prolific_links VALUES(?,?,?,?,?,?,?,?)',
            (pid,study,submission,name,view['instance_id'],allocation_index,CONSENT_VERSION,time.time()))
        source='prolific_randomized_vacancy' if allocation_index>allocated+1 else 'prolific_randomized_block_6'
        if study!=store.settings.prolific_study_id:source='prolific_replacement_quota'
        db.execute('UPDATE pl3_enrollments SET assignment_source=? WHERE instance_id=?',(source,view['instance_id']))
        instance=store._instance(db,new_token,view['instance_id'])
        return new_token,store._view(db,instance)

def resume(store,token,expected=None):
    with store.db.transaction(read_only=True) as db:
        participant=store._participant(db,token)
        linked=db.one('SELECT * FROM pl3_prolific_links WHERE participant_id=?',(participant['id'],))
        if not linked:raise StudyError('session_required',401)
        if expected:
            pid,study,submission=identifiers(expected)
            if (pid,study,submission)!=(linked['prolific_pid'],linked['study_id'],linked['submission_id']):
                raise StudyError('prolific_identity_mismatch',409)
        return store._view(db,store._instance(db,token,linked['instance_id']))

def payment_records(store,study_id=None):
    """Restricted operational view. Never included in the ordinary research export."""
    with store.db.transaction(read_only=True) as db:
        study_id=study_id or store.settings.prolific_study_id
        cohort(db,store.settings,study_id)
        records=db.all('SELECT l.*,i.domain,i.group_code,i.stage,i.completed,r.submission_status,r.reason AS release_reason,r.released FROM pl3_prolific_links l JOIN pl3_instances i ON i.id=l.instance_id LEFT JOIN pl3_prolific_releases r ON r.instance_id=i.id WHERE l.study_id=? ORDER BY l.allocation_index',(study_id,))
        for record in records:
            record['scores']=[{'task':r['task'],'score_json':r['score_json']} for r in db.all('SELECT task,score_json FROM pl3_runs WHERE instance_id=? AND status=? ORDER BY task',(record['instance_id'],'completed'))]
            reward=rewards.summary(db,record['instance_id'])
            if reward is not None:record['rewards']=reward
        return records

def release_slot(store,payload):
    """Researcher-confirmed terminal submission; never infer from inactivity.

    The instance lock also protects gameplay/completion. Enrollment is globally
    serialized and only counts committed releases. No research rows are removed.
    This does not change the submission or payment status on Prolific.
    """
    iid=payload.get('instance_id','')
    submission=payload.get('submission_id','')
    status=payload.get('submission_status')
    reason=payload.get('reason','')
    if (not isinstance(iid,str) or not re.fullmatch(r'[0-9a-f]{32}',iid)
        or not isinstance(submission,str) or not re.fullmatch(r'[0-9a-f]{24}',submission)
        or status not in ('RETURNED','TIMED_OUT')
        or not isinstance(reason,str) or not 1<=len(reason.strip())<=1000):
        raise StudyError('invalid_slot_release')
    with store.db.transaction(scope=iid) as db:
        study=payload.get('study_id',store.settings.prolific_study_id)
        cohort(db,store.settings,study)
        linked=db.one('SELECT l.instance_id,i.completed,i.stage FROM pl3_prolific_links l JOIN pl3_instances i ON i.id=l.instance_id WHERE l.instance_id=? AND l.submission_id=? AND l.study_id=?',
            (iid,submission,study))
        if not linked:raise StudyError('prolific_submission_not_found',404)
        if linked['completed'] is not None or linked['stage']=='completed':
            raise StudyError('completed_slot_cannot_be_released',409)
        prior=db.one('SELECT * FROM pl3_prolific_releases WHERE instance_id=?',(iid,))
        if prior:
            if prior['submission_status']!=status:raise StudyError('slot_release_conflict',409)
            return prior
        released=time.time()
        db.execute('INSERT INTO pl3_prolific_releases VALUES(?,?,?,?)',(iid,status,reason.strip(),released))
        return {'instance_id':iid,'submission_status':status,'reason':reason.strip(),'released':released}
