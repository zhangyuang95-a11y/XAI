"""Authoritative study flow and Task-2-only explanation authorization."""
import hashlib
import hmac
import json
from contextlib import closing
import re
import secrets
import threading
import time
import uuid

from . import RELEASE_ID, SUPPORTED_RELEASE_IDS
from .database import Database
from .registry import engine, demonstration, MODULES, scenario_config

def encode(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(value.encode()).hexdigest()
def uid(): return uuid.uuid4().hex
def public_answer(answer):
    return {k:answer[k] for k in ('status','answer','evidence_ids','language') if k in answer}

class StudyError(Exception):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)

COMMON_ITEMS = [
 ('predictable','I could predict what my teammate would do.','我能预测队友接下来会做什么。'),
 ('understood','I understood when my teammate waited or changed its plan.','我理解队友何时等待或改变计划。'),
 ('coordinate','I knew how to coordinate my actions with my teammate.','我知道怎样安排自己的动作来配合队友。'),
 ('workload','The task required a lot of mental effort.','任务需要我投入很多思考。'),
 ('smooth','Working together felt smooth.','协作过程感觉顺畅。'),
]
EXPLANATION_ITEMS = [
 ('relevant','The Task 2 answers addressed my questions.','Task 2 的回答切中了我的问题。'),
 ('clear','The Task 2 answers were easy to understand.','Task 2 的回答容易理解。'),
 ('helpful','The Task 2 answers helped me choose my next action.','Task 2 的回答帮助我选择下一步动作。'),
]

EXPORT_TABLES = ('participants','enrollments','instances','runs','frames',
                 'questions','questionnaires','timings','releases')

class Store:
    def __init__(self, settings, explainer=None):
        self.settings, self.db, self.explainer = settings, Database(settings.database), explainer
        self._qa_health_lock = threading.Lock()
        self._qa_healthy, self._qa_health_version = settings.verified, 0

    @property
    def qa_healthy(self):
        with self._qa_health_lock:
            return self._qa_healthy

    @qa_healthy.setter
    def qa_healthy(self, healthy):
        self.set_qa_health(healthy)

    def qa_health_snapshot(self):
        with self._qa_health_lock:
            return self._qa_healthy, self._qa_health_version

    def set_qa_health(self, healthy, *, expected_version=None):
        """An older probe cannot overwrite a newer participant QA result."""
        with self._qa_health_lock:
            if expected_version is not None and expected_version != self._qa_health_version:
                return False
            self._qa_healthy = bool(healthy)
            self._qa_health_version += 1
            return True

    @property
    def ready(self):
        return self.settings.ready and self.qa_healthy

    def _participant(self, db, token):
        if not token: raise StudyError('session_required',401)
        row=db.one('SELECT p.* FROM pl3_sessions s JOIN pl3_participants p ON p.id=s.participant_id WHERE s.token_hash=?',(digest(token),))
        if row is None: raise StudyError('session_required',401)
        return row

    def _instance(self, db, token, instance_id):
        participant=self._participant(db,token)
        row=db.one('SELECT * FROM pl3_instances WHERE id=? AND participant_id=?',(instance_id,participant['id']))
        if row is None: raise StudyError('study_not_found',404)
        if row['release_id'] not in SUPPORTED_RELEASE_IDS: raise StudyError('release_changed',409)
        return row

    def _ask_allowed(self, db, instance):
        if instance['group_code']!='A' or instance['stage']!='task2' or not instance['current_run']: return False
        row=db.one('SELECT status FROM pl3_runs WHERE id=?',(instance['current_run'],))
        return bool(row and row['status']=='active')

    def create(self, payload, token=None, admin=False):
        if payload.get('consent') is not True: raise StudyError('consent_required')
        domain=payload.get('domain')
        if domain not in MODULES: raise StudyError('unknown_domain')
        name=str(payload.get('participant_id','')).strip().casefold()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,63}',name): raise StudyError('invalid_participant_id')
        mode=payload.get('mode','pilot')
        if mode not in ('pilot','preview','test'): raise StudyError('invalid_mode')
        if mode!='pilot' and not admin: raise StudyError('researcher_access_required',403)
        if mode=='pilot' and not self.ready: raise StudyError('study_not_ready',503)
        requested_group=payload.get('group')
        if requested_group is not None and requested_group not in ('A','B'):
            raise StudyError('invalid_group')
        language='zh' if payload.get('language')=='zh' else 'en'
        recovery=None
        with self.db.transaction() as db:
            existing=db.one('SELECT * FROM pl3_participants WHERE id=?',(name,))
            authenticated=None
            if token:
                try: authenticated=self._participant(db,token)
                except StudyError: pass
            if existing:
                code=str(payload.get('recovery_code',''))
                if not (authenticated and authenticated['id']==name) and not (code and hmac.compare_digest(digest(code),existing['recovery_hash'])):
                    raise StudyError('participant_exists_use_recovery',409)
                if db.one('SELECT id FROM pl3_instances WHERE participant_id=? AND mode!=?',(name,mode)):
                    raise StudyError('participant_mode_conflict',409)
                # The participant row retains its original assignment. A new
                # domain may have a different explicit, per-instance choice.
                group=requested_group or existing['group_code']
            else:
                counts={x['group_code']:x['n'] for x in db.all('SELECT group_code,COUNT(*) AS n FROM pl3_instances WHERE domain=? AND mode=? AND release_id=? GROUP BY group_code',(domain,mode,RELEASE_ID))}
                group=secrets.choice(['A','B']) if counts.get('A',0)==counts.get('B',0) else min(['A','B'],key=lambda g:counts.get(g,0))
                if requested_group: group=requested_group
                recovery=secrets.token_urlsafe(18)
                db.execute('INSERT INTO pl3_participants VALUES(?,?,?,?)',(name,group,digest(recovery),time.time()))
            if not authenticated or authenticated['id']!=name:
                token=secrets.token_urlsafe(32)
                db.execute('INSERT INTO pl3_sessions VALUES(?,?,?)',(digest(token),name,time.time()))
            # Switching domains never rewrites or finishes another instance.
            # A rules revision starts a separate enrollment. Old runs, frames,
            # answers and questionnaires remain intact under their release ID.
            candidates=db.all('SELECT * FROM pl3_instances WHERE participant_id=? AND domain=? ORDER BY created DESC,id DESC',(name,domain))
            instance=next((r for r in candidates if r['release_id'] in SUPPORTED_RELEASE_IDS),None)
            if not instance:
                iid=uid()
                config=scenario_config(domain)
                seeds=config.get('heldout_seeds') or config['held_out_seeds']
                allocations=db.all('SELECT scenario_seed,group_code,COUNT(*) AS n FROM pl3_instances WHERE domain=? AND mode=? AND release_id=? GROUP BY scenario_seed,group_code',(domain,mode,RELEASE_ID))
                usage={(r['scenario_seed'],r['group_code']):r['n'] for r in allocations}
                opposite='B' if group=='A' else 'A'
                seed=min(seeds,key=lambda candidate:(usage.get((candidate,group),0),-usage.get((candidate,opposite),0),seeds.index(candidate)))
                db.execute('INSERT INTO pl3_instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (iid,name,domain,RELEASE_ID,group,mode,'demo',0,None,0,language,seed,time.time(),None))
                if requested_group:
                    assignment='researcher_override' if admin and mode!='pilot' else 'participant_choice'
                else:
                    assignment='existing_participant' if existing else 'randomized_balanced'
                db.execute('INSERT INTO pl3_enrollments VALUES(?,?,?,?,?,?)',(iid,1,language,assignment,config.get('scenario_version',config['version']),time.time()))
                instance=db.one('SELECT * FROM pl3_instances WHERE id=?',(iid,))
            result=self._view(db,instance)
            result['recovery_code']=recovery
        return token,result

    def recover_view(self, token, domain=None):
        with self.db.transaction(read_only=True) as db:
            p=self._participant(db,token)
            rows=db.all('SELECT * FROM pl3_instances WHERE participant_id=? ORDER BY created DESC,id DESC',(p['id'],))
            if domain:
                instance=next((r for r in rows if r['domain']==domain and r['release_id'] in SUPPORTED_RELEASE_IDS),None)
            else:
                compatible=[r for r in rows if r['release_id'] in SUPPORTED_RELEASE_IDS]
                instance=next((r for r in compatible if r['stage']!='completed'),compatible[0] if compatible else None)
            if not instance: return {'participant_id':p['id'],'stage':'welcome',
                'previous_version_saved':any(r['release_id'] not in SUPPORTED_RELEASE_IDS and (not domain or r['domain']==domain) for r in rows)}
            if instance['release_id'] not in SUPPORTED_RELEASE_IDS: raise StudyError('release_changed',409)
            return self._view(db,instance)

    def view(self, token, instance_id):
        with self.db.transaction(read_only=True) as db: return self._view(db,self._instance(db,token,instance_id))

    def _view(self,db,instance):
        eng=engine(instance['domain'])
        language=instance['language']
        enrollment=db.one('SELECT assignment_source FROM pl3_enrollments WHERE instance_id=?',(instance['id'],))
        result={k:instance[k] for k in ('id','domain','stage','revision','language','mode','release_id')}
        result.update(instance_id=instance['id'],participant_id=instance['participant_id'],
            can_ask=self._ask_allowed(db,instance),rules=eng.rules(language),
            task2_explanation_notice=instance['stage']=='task2' and instance['group_code']=='A' and not self._ask_allowed(db,instance),
            task_runs=[],questions=[],state=None,run_id=instance['current_run'],
            group=instance['group_code'],group_selection_locked=True,
            group_assignment_source=enrollment['assignment_source'] if enrollment else None)
        runs=db.all('SELECT id,task,status,score_json,state_json FROM pl3_runs WHERE instance_id=? ORDER BY task',(instance['id'],))
        result['task_runs']=[{'id':r['id'],'task':r['task'],'status':r['status'],'score':eng.public_state(json.loads(r['state_json']))['score'],'turn':json.loads(r['state_json'])['turn']} for r in runs]
        if instance['stage']=='demo':
            demo=demonstration(instance['domain'])
            result['demo']={'index':instance['demo_index'],'captions':demo['captions'],'frames':demo['frames']}
        if instance['current_run']:
            run=db.one('SELECT * FROM pl3_runs WHERE id=?',(instance['current_run'],))
            state=json.loads(run['state_json'])
            result.update(state=eng.public_state(state),actions=eng.legal_actions(state) if not state['terminal'] else [],run_status=run['status'])
            if hasattr(eng,'action_label'):
                result['action_labels']={a:eng.action_label(a,language) for a in result['actions']}
        if result['can_ask']:
            result['questions']=[{'id':q['id'],'question':q['question'],'language':q['language'],
                'result':public_answer(json.loads(q['result_json'])),'status':q['status'],'target_turn':q['target_turn'],'target_run':q['target_run']}
                for q in db.all("SELECT * FROM pl3_questions WHERE authorized_run=? AND status IN ('answered','clarification','unavailable') ORDER BY requested",(instance['current_run'],))]
        if instance['stage']=='questionnaire':
            items=COMMON_ITEMS+(EXPLANATION_ITEMS if instance['group_code']=='A' else [])
            result['questionnaire']={'items':[{'id':i[0],'text':i[2 if language=='zh' else 1],'allow_na':i in EXPLANATION_ITEMS} for i in items],
                'comprehension':[]}
        return result

    def command(self,token,kind,payload):
        command_id=payload.get('command_id')
        if not isinstance(command_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}',command_id): raise StudyError('command_id_required')
        fingerprint=digest(encode({'kind':kind,'payload':payload}))
        with self.db.transaction(scope=payload.get('instance_id')) as db:
            instance=self._instance(db,token,payload.get('instance_id'))
            prior=db.one('SELECT * FROM pl3_commands WHERE session_hash=? AND command_id=?',(digest(token),command_id))
            if prior:
                if prior['request_hash']!=fingerprint: raise StudyError('idempotency_conflict',409)
                # Never return a cached response containing now-revoked answers.
                return self._view(db,instance)
            if payload.get('revision')!=instance['revision']: raise StudyError('stale_state',409)
            timings=payload.get('timings',[])
            if not isinstance(timings,list) or len(timings)>4:raise StudyError('invalid_timing')
            for measurement in timings:
                if not isinstance(measurement,dict):raise StudyError('invalid_timing')
                seconds=measurement.get('seconds');time_kind=measurement.get('kind')
                if time_kind not in ('active','reading','replay') or type(seconds) not in (int,float) or not 0<=seconds<=86400:raise StudyError('invalid_timing')
                task=int(instance['stage'][-1]) if instance['stage'].startswith('task') else None
                db.execute('INSERT INTO pl3_timings VALUES(?,?,?,?,?,?)',(uid(),instance['id'],task,time_kind,seconds,time.time()))
            if kind=='demo_next':
                if instance['stage']!='demo': raise StudyError('wrong_stage',409)
                end=len(demonstration(instance['domain'])['captions'])
                db.execute('UPDATE pl3_instances SET demo_index=? WHERE id=?',(min(instance['demo_index']+1,end),instance['id']))
            elif kind in ('demo_finish','demo_skip'):
                if instance['stage']!='demo': raise StudyError('wrong_stage',409)
                end=len(demonstration(instance['domain'])['captions'])
                db.execute('UPDATE pl3_instances SET demo_index=? WHERE id=?',(end,instance['id']))
                if kind=='demo_skip':
                    # Skip is an explicit participant choice, recorded as its
                    # own idempotent command, never a simulated task action.
                    self._next(db,{**instance,'demo_index':end})
            elif kind=='next': self._next(db,instance)
            elif kind=='action': self._action(db,instance,payload)
            elif kind=='language':
                if payload.get('language') not in ('en','zh'): raise StudyError('invalid_language')
                db.execute('UPDATE pl3_instances SET language=? WHERE id=?',(payload['language'],instance['id']))
            elif kind=='questionnaire': self._questionnaire(db,instance,payload)
            elif kind=='timing':
                seconds=payload.get('seconds')
                if not isinstance(seconds,(int,float)) or not 0<=seconds<=86400: raise StudyError('invalid_timing')
                timing_kind=payload.get('kind')
                if timing_kind not in ('reading','replay','active','explanation_wait'):raise StudyError('invalid_timing')
                task=int(instance['stage'][-1]) if instance['stage'].startswith('task') else None
                db.execute('INSERT INTO pl3_timings VALUES(?,?,?,?,?,?)',(uid(),instance['id'],task,timing_kind,seconds,time.time()))
            else: raise StudyError('unknown_command',404)
            db.execute('UPDATE pl3_instances SET revision=revision+1 WHERE id=?',(instance['id'],))
            updated=db.one('SELECT * FROM pl3_instances WHERE id=?',(instance['id'],))
            db.execute('INSERT INTO pl3_commands VALUES(?,?,?,?,?)',(digest(token),command_id,fingerprint,'{}',time.time()))
            return self._view(db,updated)

    def _next(self,db,instance):
        stage=instance['stage']
        if stage=='demo':
            if instance['demo_index']<len(demonstration(instance['domain'])['captions']): raise StudyError('finish_demo',409)
            task=1
        elif stage in ('task1','task2','task3'):
            run=db.one('SELECT * FROM pl3_runs WHERE id=?',(instance['current_run'],))
            if run['status']!='completed': raise StudyError('finish_task',409)
            if stage=='task3':
                db.execute("UPDATE pl3_instances SET stage='questionnaire' WHERE id=?",(instance['id'],))
                return
            task=int(stage[-1])+1
        else: raise StudyError('wrong_stage',409)
        eng=engine(instance['domain']); state=eng.initial_state(instance['scenario_seed'],task)
        rid=uid(); now=time.time()
        db.execute('INSERT INTO pl3_runs VALUES(?,?,?,?,?,?,?,?,?)',
            (rid,instance['id'],task,instance['scenario_seed'],encode(state),'active',encode(eng.score(state)),now,None))
        db.execute('INSERT INTO pl3_frames VALUES(?,?,?,?,?,?,?)',
            (rid,0,encode(state),encode(eng.public_state(state)),encode(eng.decide(state)),None,now))
        db.execute('UPDATE pl3_instances SET stage=?,current_run=? WHERE id=?',('task'+str(task),rid,instance['id']))

    def _action(self,db,instance,payload):
        if instance['stage'] not in ('task1','task2','task3'): raise StudyError('wrong_stage',409)
        if payload.get('run_id')!=instance['current_run']:raise StudyError('wrong_run',409)
        run=db.one('SELECT * FROM pl3_runs WHERE id=?',(instance['current_run'],))
        if run['status']!='active':raise StudyError('task_finished',409)
        state=json.loads(run['state_json']); eng=engine(instance['domain'])
        if payload.get('turn')!=state['turn']:raise StudyError('stale_state',409)
        action=payload.get('action')
        if action not in eng.legal_actions(state):raise StudyError('illegal_action')
        decision=eng.decide(state)
        nxt=eng.step(state,action,decision)
        db.execute('UPDATE pl3_frames SET human_action=?,decision_json=? WHERE run_id=? AND turn=?',
            (action,encode(decision),run['id'],state['turn']))
        now=time.time(); terminal=nxt['terminal']
        db.execute('UPDATE pl3_runs SET state_json=?,status=?,score_json=?,ended=? WHERE id=?',
            (encode(nxt),'completed' if terminal else 'active',encode(eng.score(nxt)),now if terminal else None,run['id']))
        db.execute('INSERT INTO pl3_frames VALUES(?,?,?,?,?,?,?)',
            (run['id'],nxt['turn'],encode(nxt),encode(eng.public_state(nxt)),encode(eng.decide(nxt)) if not terminal else '{}',None,now))
        if terminal:
            db.execute("UPDATE pl3_questions SET status='revoked' WHERE authorized_run=? AND status='pending'",(run['id'],))

    def _questionnaire(self,db,instance,payload):
        if instance['stage']!='questionnaire':raise StudyError('wrong_stage',409)
        answers=payload.get('answers',{})
        if not isinstance(answers,dict):raise StudyError('invalid_questionnaire')
        for q in COMMON_ITEMS+(EXPLANATION_ITEMS if instance['group_code']=='A' else []):
            value=answers.get(q[0])
            if q in EXPLANATION_ITEMS and value=='na':continue
            if type(value) is not int or not 1<=value<=7:raise StudyError('incomplete_questionnaire')
        graded=[]
        feedback=str(payload.get('feedback',''))[:4000]
        db.execute('INSERT INTO pl3_questionnaires VALUES(?,?,?,?)',(instance['id'],encode({'ratings':answers,'feedback':feedback}),encode(graded),time.time()))
        db.execute("UPDATE pl3_instances SET stage='completed',completed=? WHERE id=?",(time.time(),instance['id']))

    def frame(self,token,instance_id,run_id,turn):
        with self.db.transaction(read_only=True) as db:
            instance=self._instance(db,token,instance_id)
            row=db.one('SELECT f.public_json FROM pl3_frames f JOIN pl3_runs r ON r.id=f.run_id WHERE r.instance_id=? AND f.run_id=? AND f.turn=?',(instance['id'],run_id,turn))
            if not row:raise StudyError('frame_not_found',404)
            return {'state':json.loads(row['public_json']),'can_ask':self._ask_allowed(db,instance)}

    def ask(self,token,payload):
        question=str(payload.get('question','')).strip()
        if not question or len(question)>2000:raise StudyError('invalid_question')
        qid=payload.get('question_id') or uid()
        if not isinstance(qid,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}',qid):raise StudyError('invalid_question_id')
        with self.db.transaction(scope=payload.get('instance_id')) as db:
            instance=self._instance(db,token,payload.get('instance_id'))
            if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
            if payload.get('authorized_run')!=instance['current_run']:raise StudyError('wrong_run',403)
            target=payload.get('target_run',instance['current_run']);turn=payload.get('turn')
            row=db.one('SELECT f.*,r.task FROM pl3_frames f JOIN pl3_runs r ON r.id=f.run_id WHERE r.instance_id=? AND f.run_id=? AND f.turn=? AND r.task<=2',(instance['id'],target,turn))
            if not row:raise StudyError('frame_not_found',404)
            # Expire interrupted requests after both provider attempts plus a margin.
            for expired_language in ('en','zh'):
                expired={'status':'unavailable','answer':'上一次请求已中断，请重新提问。' if expired_language=='zh' else 'The previous request was interrupted. Please ask again.','evidence_ids':[],
                    'audit':{'error':'request_lease_expired','displayed':False}}
                db.execute("UPDATE pl3_questions SET status='unavailable',result_json=?,finished=? WHERE authorized_run=? AND status='pending' AND requested<? AND language=?",
                    (encode(expired),time.time(),instance['current_run'],time.time()-120,expired_language))
            prior=db.one('SELECT * FROM pl3_questions WHERE id=?',(qid,))
            if prior:
                if prior['instance_id']!=instance['id'] or prior['question']!=question or prior['target_run']!=target or prior['target_turn']!=turn:raise StudyError('idempotency_conflict',409)
                return {'id':qid,'status':prior['status'],'result':public_answer(json.loads(prior['result_json']))}
            if db.one("SELECT id FROM pl3_questions WHERE authorized_run=? AND status='pending'",(instance['current_run'],)):
                raise StudyError('question_busy',409)
            language=payload.get('language',instance['language'])
            if language not in ('en','zh'):raise StudyError('invalid_language')
            history=db.all("SELECT id,question,result_json FROM pl3_questions WHERE authorized_run=? AND status IN ('answered','clarification') ORDER BY requested DESC LIMIT 6",(instance['current_run'],))
            previous=[{'question':h['question'],'answer':json.loads(h['result_json']).get('answer','')} for h in reversed(history)]
            db.execute('INSERT INTO pl3_questions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (qid,instance['id'],instance['current_run'],target,turn,question,language,'{}','pending',time.time(),None,None))
            # A displayed frame is an AFTER-action snapshot. Its own stored
            # decision predicts the next transition; the previous row contains
            # the actual simultaneous actions that produced the displayed one.
            earlier=db.all('SELECT public_json FROM pl3_frames WHERE run_id=? AND turn<=? ORDER BY turn DESC LIMIT 16',(target,turn))
            state=json.loads(row['state_json']);decision=json.loads(row['decision_json'])
            incoming=db.one('SELECT state_json,decision_json,human_action FROM pl3_frames WHERE run_id=? AND turn=?',
                (target,turn-1)) if turn>0 else None
            action_context={'state':json.loads(incoming['state_json']) if incoming else None,
                'decision':json.loads(incoming['decision_json']) if incoming else None,
                'human_action':incoming['human_action'] if incoming else None}
        try:
            if not self.explainer:raise RuntimeError('model_unavailable')
            answer=self.explainer.answer(engine(instance['domain']),state,decision,question,language,previous,
                [json.loads(f['public_json']) for f in reversed(earlier)],action_context=action_context)
            status=answer.get('status','answered')
            if status not in ('answered','clarification','unavailable'):status='unavailable'
        except Exception:
            status='unavailable';answer={'status':status,'answer':'问答暂时不可用，请稍后重试。' if language=='zh' else 'Questions are temporarily unavailable. Please try again.','evidence_ids':[]}
        answer.setdefault('audit',{}).update(context_question_ids=[h['id'] for h in reversed(history)],context_sha256=digest(encode(previous)),authorized_run=instance['current_run'],target_run=target,target_turn=turn,target_display_turn=turn,
            performed_decision_turn=turn-1 if incoming else None,
            incoming_transition_sha256=digest(encode(action_context)),authorization_at_request=True)
        self.qa_healthy = status in ('answered','clarification')
        with self.db.transaction(scope=payload.get('instance_id')) as db:
            current=self._instance(db,token,instance['id'])
            permitted=self._ask_allowed(db,current) and current['current_run']==instance['current_run']
            answer['audit']['authorization_at_completion']=permitted
            db.execute('UPDATE pl3_questions SET result_json=?,status=?,finished=? WHERE id=?',
                (encode(answer),status if permitted else 'revoked',time.time(),qid))
        if not permitted:raise StudyError('explanations_unavailable',403)
        return {'id':qid,'status':status,'result':public_answer(answer),'target_run':target,'target_turn':turn}

    def acknowledge_answer(self,token,instance_id,qid):
        with self.db.transaction(scope=instance_id) as db:
            instance=self._instance(db,token,instance_id)
            if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
            row=db.one('SELECT * FROM pl3_questions WHERE id=? AND instance_id=? AND authorized_run=?',(qid,instance_id,instance['current_run']))
            if not row or row['status'] not in ('answered','clarification','unavailable'):raise StudyError('answer_unavailable',403)
            db.execute('UPDATE pl3_questions SET displayed=? WHERE id=?',(time.time(),qid))
        return {'ok':True}

    def iter_export(self, release_id=None, mode=None, batch_size=64):
        """Read one consistent export without loading unrelated historical rows.

        SQL predicates select related rows before transfer. Each query uses a
        bounded cursor; callers must consume/close this iterator before sending
        a slow client response so no database transaction follows that client.
        """
        clauses, values = [], []
        if release_id:
            clauses.append('i.release_id=?'); values.append(release_id)
        if mode:
            clauses.append('i.mode=?'); values.append(mode)
        selected = ' AND '.join(clauses)
        relationships = {
            'participants': 'i.participant_id=r.id',
            'enrollments': 'i.id=r.instance_id',
            'instances': 'i.id=r.id',
            'runs': 'i.id=r.instance_id',
            'questions': 'i.id=r.instance_id',
            'questionnaires': 'i.id=r.instance_id',
            'timings': 'i.id=r.instance_id',
            'releases': 'i.release_id=r.id',
        }
        ordering = {'participants':'r.id','enrollments':'r.instance_id','instances':'r.id',
                    'runs':'r.id','frames':'r.run_id,r.turn','questions':'r.id',
                    'questionnaires':'r.instance_id','timings':'r.id','releases':'r.id'}
        with self.db.transaction(read_only=True) as db:
            for name in EXPORT_TABLES:
                # Recovery/session hashes never enter the export cursor.
                fields = 'r.id,r.group_code,r.created' if name == 'participants' else 'r.*'
                query = 'SELECT ' + fields + ' FROM pl3_' + name + ' r'
                if selected:
                    if name == 'frames':
                        query += (' WHERE EXISTS (SELECT 1 FROM pl3_runs u JOIN pl3_instances i'
                                  ' ON i.id=u.instance_id WHERE u.id=r.run_id AND ' + selected + ')')
                    else:
                        query += (' WHERE EXISTS (SELECT 1 FROM pl3_instances i WHERE '
                                  + relationships[name] + ' AND ' + selected + ')')
                query += ' ORDER BY ' + ordering[name]
                with closing(db.iterate(query, tuple(values), batch_size=batch_size)) as rows:
                    for row in rows:
                        yield {'table':name,'record':row}

    def export(self, release_id=None, mode=None):
        """Compatibility for local callers that explicitly request a dictionary.

        The HTTP endpoint uses iter_export and a disk file instead, so this
        materialized convenience API cannot grow production response memory.
        """
        result = {name:[] for name in EXPORT_TABLES}
        for item in self.iter_export(release_id, mode):
            result[item['table']].append(item['record'])
        return result
