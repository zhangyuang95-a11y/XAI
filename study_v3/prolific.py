"""One-game Prolific enrolment. Private identity mapping is not a research export.

Selection and session creation share the existing global enrollment transaction,
which serializes across processes on PostgreSQL and writers on SQLite.
"""
import re
import secrets
import time
from .store import StudyError, uid

CONSENT_VERSION='policylens-prolific-consent-20260922.v2'
CELLS=tuple((domain,group) for domain in ('warehouse','pong','kitchen') for group in ('A','B'))

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
    if study!=store.settings.prolific_study_id:raise StudyError('wrong_prolific_study',403)
    with store.db.transaction() as db:
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
        rows=db.all('SELECT i.domain,i.group_code,COUNT(*) AS n FROM pl3_prolific_links l JOIN pl3_instances i ON i.id=l.instance_id WHERE l.study_id=? GROUP BY i.domain,i.group_code',(study,))
        counts={(r['domain'],r['group_code']):r['n'] for r in rows}
        allocated=sum(counts.values())
        if allocated>=store.settings.prolific_places:raise StudyError('prolific_full',409)
        least=min(counts.get(cell,0) for cell in CELLS)
        domain,group=secrets.choice([cell for cell in CELLS if counts.get(cell,0)==least])
        # Only internal IDs enter game data or the semantic provider context.
        name='pl-'+uid()
        new_token,view=store.create({'domain':domain,'group':group,'participant_id':name,
            'mode':'pilot','consent':True,'language':'en'},token=None,_db=db,_prolific=True)
        db.execute('INSERT INTO pl3_prolific_links VALUES(?,?,?,?,?,?,?,?)',
            (pid,study,submission,name,view['instance_id'],allocated+1,CONSENT_VERSION,time.time()))
        db.execute("UPDATE pl3_enrollments SET assignment_source='prolific_randomized_block_6' WHERE instance_id=?",(view['instance_id'],))
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

def payment_records(store):
    """Restricted operational view. Never included in the ordinary research export."""
    with store.db.transaction(read_only=True) as db:
        records=db.all('SELECT l.*,i.domain,i.group_code,i.stage,i.completed FROM pl3_prolific_links l JOIN pl3_instances i ON i.id=l.instance_id WHERE l.study_id=? ORDER BY l.allocation_index',(store.settings.prolific_study_id,))
        for record in records:
            record['scores']=[{'task':r['task'],'score_json':r['score_json']} for r in db.all('SELECT task,score_json FROM pl3_runs WHERE instance_id=? AND status=? ORDER BY task',(record['instance_id'],'completed'))]
        return records
