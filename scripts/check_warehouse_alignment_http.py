#!/usr/bin/env python3
"""Finite real-HTTP flow check; does not start a server or grant qualification.

Default is plan-only. --execute requires an already running genuine service and
an empty, separate *_http_qa.sqlite3. Only public HTTP and read-only SQLite are
used. Four technical participants execute fixed public controls, never a model
ability test. No runtime imports, mock Store, private question keys or NN calls.
"""
import argparse
from contextlib import closing
from copy import copy, deepcopy
import gzip
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

VERSION='warehouse-alignment-real-http-flow-check.v1'
FAMILY='warehouse_alignment197'
COOKIE='warehouse_alignment_local_pilot_session_v1'
CAP={'registered_participants':4,'practice_actions_per_participant':5,
     'study_rounds_per_participant':6,'horizon':120,'gameplay_steps':2900,
     'ordinary_questions':4,'historical_answer_replay_step_upper_bound':4,
     'total_environment_step_upper_bound':2904,'http_requests':3600}
ROOT=Path(__file__).resolve().parents[1]


def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise AssertionError(message)


def save(path,value):
    path=Path(path);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())


def database(path):
    path=Path(path).resolve(strict=True)
    require(path.name.endswith('_http_qa.sqlite3'),'Use an explicit separate *_http_qa.sqlite3, never the participant database')
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=15)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');return db


def db_identity(path,manifest_sha,empty=False):
    with closing(database(path)) as db:
        meta={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}
        require(meta.get('service_family')==FAMILY and meta.get('namespace')=='local_pilot','Wrong service database identity')
        contexts=[v for k,v in meta.items() if k.startswith('service_context:')]
        require(len(contexts)==1 and contexts[0]['manifest_sha256']==manifest_sha,'QA database must bind exactly the declared genuine release')
        if empty:
            for table in ('sessions','runs','frames','questions','operations','events','blocks'):
                require(db.execute('SELECT count(*) FROM '+table).fetchone()[0]==0,'QA database is not empty: '+table)
        return contexts[0]


def allocation(path,sid):
    with closing(database(path)) as db:
        row=db.execute('SELECT participant_id,condition,task_order,position,stage FROM sessions WHERE id=?',(sid,)).fetchone()
        require(row is not None,'HTTP session is not in the explicit QA database');return dict(row)


def public_only(value):
    if isinstance(value,dict):
        require(not ({'probabilities','logits','decision','actor_weights','snapshot'} & set(value)), 'Private policy or authoritative snapshot leaked')
        for x in value.values():public_only(x)
    elif isinstance(value,list):
        for x in value:public_only(x)


def physical(view):
    return {'run_id':view['run_id'],'state':view['state'],'metrics':view['metrics'],'ended':view['ended']}


class Harness:
    def __init__(self,base,output):
        self.base=base;self.output=Path(output);self.requests=0;self.gameplay_steps=0;self.questions=0
        self.answers=[];self.checks=[];self.pending=None
        self.journal=(self.output/'http_journal.jsonl').open('x');os.chmod(self.output/'http_journal.jsonl',0o600)

    def record(self,value):
        self.journal.write(canonical(value)+'\n');self.journal.flush();os.fsync(self.journal.fileno())

    def request(self,client,path,payload=None,expected=200,error=None):
        require(self.requests<CAP['http_requests'],'Fixed HTTP request budget exhausted')
        self.requests+=1;rid=f'{self.requests:05d}';request={'id':rid,'client':client.name,'path':path,'payload':payload}
        self.pending=request;self.record({'event':'before','request':request})
        raw=None;status=None;started=time.monotonic()
        req=urllib.request.Request(self.base+path,data=None if payload is None else canonical(payload).encode(),
            headers={'Content-Type':'application/json','Origin':self.base})
        try:
            response=client.opener.open(req,timeout=60)
        except urllib.error.HTTPError as exc:response=exc
        with response:
            status=response.code;headers=dict(response.headers);raw=response.read();value=json.loads(raw)
        target=self.output/f'response_{rid}.json.gz'
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as out:
            with gzip.GzipFile(fileobj=out,mode='wb',compresslevel=1,mtime=0) as gz:gz.write(raw)
            out.flush();os.fsync(out.fileno())
        self.record({'event':'after','id':rid,'status':status,'elapsed_seconds':time.monotonic()-started,
            'response_file':target.name,'response_sha256':sha(target),'cache_control':headers.get('Cache-Control')})
        self.pending=None
        require(headers.get('Cache-Control')=='no-store','HTTP response permits caching')
        require(status==expected,f'Unexpected status {status}, expected {expected}: {value}')
        if error:require(value.get('error')==error,f'Unexpected API rejection: {value}')
        if status==200:public_only(value)
        return value

    def check(self,name):self.checks.append(name)


class Client:
    def __init__(self,harness,name,jar=None):
        self.h=harness;self.name=name;self.jar=jar or http.cookiejar.CookieJar()
        self.opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar));self.view=None;self.old_question=None;self.question_id=None

    def get(self):
        self.view=self.h.request(self,'/api/view');return self.view

    def command(self,kind,expected=200,error=None,**fields):
        payload={'kind':kind,'operation_id':uuid.uuid4().hex,'expected_version':self.view['version'],**fields}
        result=self.h.request(self,'/api/command',payload,expected,error)
        if expected==200:self.view=result
        return payload,result

    def refresh(self):
        old=deepcopy(self.view)
        # CookieJar owns an RLock on Python 3.13 and therefore cannot be
        # deep-copied.  Copy the immutable cookie records into a fresh jar to
        # model a new HTTP client with the same browser session.
        jar=http.cookiejar.CookieJar()
        for cookie in self.jar: jar.set_cookie(copy(cookie))
        fresh=Client(self.h,self.name,jar);new=fresh.get()
        require(new==old,'Fresh HTTP client/cookie does not restore current state')
        self.jar,self.opener,self.view=fresh.jar,fresh.opener,new

    def action(self,action,duplicate=False):
        require(self.h.gameplay_steps<CAP['gameplay_steps'],'Gameplay step cap exhausted')
        before=deepcopy(self.view);payload,result=self.command('action',action=action)
        require(result['state']['frame']==before['state']['frame']+1,'Action did not advance exactly one joint step')
        self.h.gameplay_steps+=1
        if duplicate:
            require(self.h.request(self,'/api/command',payload)==result,'Identical action replay changed state')
            self.h.request(self,'/api/command',{**payload,'action':'LEFT' if action!='LEFT' else 'RIGHT'},409,'operation_id_reused')
            self.h.request(self,'/api/command',{**payload,'operation_id':uuid.uuid4().hex},409,'state_version_conflict')

    def forbidden_question(self):
        self.command('question',403,'explanation_forbidden',run_id=self.view['run_id'],frame=0,
            question='为什么选择这个动作？',language='zh',focus='executed')
        require(not self.view['explain_allowed'] and not self.view['answers'],'Unauthorized view exposes explanations')

    def question(self,language):
        require(self.h.questions<CAP['ordinary_questions'],'Question budget exhausted')
        before=physical(self.view)
        question='为什么选择这个动作？' if language=='zh' else 'Why did you choose this action?'
        payload,result=self.command('question',run_id=self.view['run_id'],frame=self.view['state']['frame'],
            question=question,language=language,focus='executed')
        self.h.questions+=1
        # The submitted question is the only new row matching this unique frame/text pair.
        matches=[q for q in result['answers'] if q['question']==question and q['frame']==payload['frame']]
        require(len(matches)==1,'Question identity is ambiguous');qid=matches[0]['id']
        for _ in range(50):
            answer=next(q for q in self.view['answers'] if q['id']==qid)
            if answer['status'] not in ('pending','running'):break
            time.sleep(.2);self.get()
        require(answer['status']=='complete' and answer['text'].strip(),'Real explanation failed or exceeded the bounded polling window')
        require(answer['run_id']==payload['run_id'] and answer['frame']==payload['frame'],'Answer frame/run binding differs')
        require(physical(self.view)==before,'Reading/explaining advanced the real run')
        self.command('answer_seen',answer_id=qid)
        self.h.answers.append({'session_id':self.view['session_id'],'id':qid,'language':language,'run_id':payload['run_id'],'frame':payload['frame']})
        if self.old_question is None:self.old_question,self.question_id=payload,qid


def history(client):
    before=physical(client.view);result=client.h.request(client,'/api/history')
    require(result['run_id']==client.view['run_id'],'History belongs to another run')
    require([x['state']['frame'] for x in result['frames']]==list(range(client.view['state']['frame']+1)),'History frames have gaps or duplicates')
    require(result['frames'][-1]['state']==client.view['state'],'History head differs from live view')
    client.get();require(physical(client.view)==before,'History read advanced the run')


def final_database_check(path,clients,harness):
    rows=[];total=0
    with closing(database(path)) as db:
        require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','SQLite integrity check failed')
        for c in clients:
            sid=c.view['session_id'];s=dict(db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone())
            runs=[dict(x) for x in db.execute('SELECT * FROM runs WHERE session_id=? ORDER BY rowid',(sid,))]
            require(s['stage']=='completed' and len(runs)==7,'Missing practice or six fixed study rounds')
            require([(r['stage'],r['round_index']) for r in runs]==[('practice',0),('task1',0),('task1',1),('task1',2),('task2',0),('task2',1),('task2',2)],'Study order/slots differ')
            require(all(r['ended']==1 for r in runs),'Unended recorded round')
            require(len(json.loads(s['questionnaire']))==11 and s['questionnaire_scores'] is not None,'Incomplete 4+4+3 questionnaire or grading record')
            q=[dict(x) for x in db.execute('SELECT * FROM questions WHERE session_id=?',(sid,))]
            require(len(q)==(2 if s['condition']=='A' else 0),'Questions crossed an A/B permission boundary')
            require(all(x['stage']=='task1' and x['status']=='complete' and x['shown'] for x in q),'Answer completion/show records incomplete')
            n=0;delivery=[]
            for run in runs:
                metrics=json.loads(run['metrics']);steps=metrics['steps'];n+=steps
                frames=db.execute('SELECT count(*),min(frame),max(frame) FROM frames WHERE run_id=?',(run['id'],)).fetchone()
                require(tuple(frames)==(steps+1,0,steps),'Saved frame count differs from confirmed actions')
                require(metrics['overrides']==0,'Saved run reports a neural action override')
                for frame in db.execute('SELECT frame,internal FROM frames WHERE run_id=? AND frame>0',(run['id'],)):
                    internal=json.loads(frame['internal'])
                    require(internal['decision']['policy_actions']['robot_2']==internal['submitted_actions']['robot_2'],
                        'Saved actual NN proposal was not submitted unchanged')
                if run['stage']=='task2':delivery.append(metrics['deliveries'])
            operation_actions=db.execute("SELECT count(*) FROM operations WHERE session_id=? AND kind='action'",(sid,)).fetchone()[0]
            require(operation_actions==n,'Duplicate/failed requests caused an unaccounted action')
            total+=n;rows.append({'participant_id':s['participant_id'],'condition':s['condition'],'task_order':s['task_order'],
                'session_id':sid,'run_ids':[r['id'] for r in runs],'confirmed_steps':n,'task2_deliveries':delivery,
                'task2_mean_deliveries':sum(delivery)/3,'technical_flow_only':True})
        require(total==harness.gameplay_steps,'HTTP and database confirmed-step counts differ')
    return rows


def execute(args):
    base=args.base.rstrip('/');parsed=urllib.parse.urlsplit(base)
    require(parsed.scheme=='http' and parsed.hostname=='127.0.0.1' and parsed.path=='','Only an explicitly started local loopback service is supported')
    require(re.fullmatch('[0-9a-f]{64}',args.expected_manifest_sha256 or ''),'Explicit release manifest SHA required')
    require(args.database is not None and args.output is not None,'Explicit QA database and new output directory required')
    context=db_identity(args.database,args.expected_manifest_sha256,empty=True)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False);os.chmod(output,0o700)
    plan={'version':VERSION,'base':base,'database':str(Path(args.database).resolve()),'expected_manifest_sha256':args.expected_manifest_sha256,
        'script_sha256':sha(__file__),'budget':CAP,'technical_flow_only':True,'formal_sample':False,
        'answer_assessment':'Complete nonempty real renderer response and frame binding; not a new factual-quality acceptance',
        'auxiliary_accounting':'At most one isolated historical replay per ordinary answer; not directly metered by public HTTP',
        'autoplay':'Fixed WAIT actions after five practice direction/WAIT inputs; never a performance evaluation'}
    save(output/'plan.json',plan);h=Harness(base,output);clients=[]
    try:
        token=uuid.uuid4().hex[:12]
        for i in range(4):
            c=Client(h,f'qa_http_{token}_{i+1}');v=c.get();require(v['flow']['stage']=='registration','Unexpected initial stage')
            require(v['release']['study_ready'] is True and v['release']['test_fixture'] is False and v['study_only'] is True,'Service is not genuinely admitted')
            require(v['horizon']==120,'The preregistered HTTP budget assumes horizon 120')
            require(all(k not in v['flow'] for k in ('condition','task_order')),'Participant sees allocation')
            a=allocation(args.database,v['session_id']);require(a['participant_id'] is None,'Request reached another database or existing participant')
            c.command('start',403,'study_only_no_freeplay',mode='freeplay')
            c.command('action',403,'actions_not_allowed_in_stage',action='WAIT')
            c.command('start',mode='study',participant_id=c.name,consent=True)
            require(c.view['flow']['stage']=='consent' and c.view['run_id'] is None,'Registration skipped consent')
            c.command('next',400,'explicit_consent_required');c.refresh();clients.append(c)
        cells=[allocation(args.database,c.view['session_id']) for c in clients]
        require({(x['condition'],x['task_order']) for x in cells}=={('A','XY'),('A','YX'),('B','XY'),('B','YX')},'Four-person allocation block is unbalanced')
        duplicate=Client(h,'unregistered_duplicate_browser');duplicate.get()
        duplicate.command('start',409,'participant_id_taken',mode='study',participant_id=clients[0].name.upper())
        require(duplicate.view['flow']['stage']=='registration','Second browser took over an ID')
        for c in clients:
            c.command('next',consent=True);c.forbidden_question()
            for j,action in enumerate(('UP','RIGHT','DOWN','LEFT','WAIT')):
                if c.view['ended']:break
                c.action(action,duplicate=j==0)
            history(c);c.refresh()
            if not c.view['ended']:c.command('end')
            c.command('next')
        h.check('four_cells_consent_practice_duplicate_id_action_idempotency_stale_tab_refresh')
        for index,c in enumerate(clients):
            group=cells[index]['condition'];language='zh' if cells[index]['task_order']=='XY' else 'en'
            for stage in ('task1','task2'):
                for round_index in range(1,4):
                    require((c.view['flow']['stage'],c.view['flow']['round_index'])==(stage,round_index),'Stage progression skipped or reordered a round')
                    if stage=='task2' or group=='B':c.forbidden_question()
                    if stage=='task2' and round_index==1 and c.old_question:
                        old_run=c.old_question['run_id']
                        require(not h.request(c,'/api/command',c.old_question)['answers'],'Old cached Task1 request replay exposed an answer')
                        c.command('answer_seen',403,'explanation_forbidden',answer_id=c.question_id)
                        h.request(c,'/api/history?'+urllib.parse.urlencode({'run_id':old_run}),
                            None,403,'history_current_run_only')
                    c.action('WAIT')
                    if stage=='task1' and round_index==1:
                        other=clients[(index+1)%4]
                        h.request(other,'/api/history?'+urllib.parse.urlencode({'run_id':c.view['run_id']}),None,
                            403 if other.view['run_id'] else 404,
                            'history_current_run_only' if other.view['run_id'] else 'run_not_found')
                        if group=='A':
                            if other.view['run_id']:
                                c.command('question',404,'run_not_found',run_id=other.view['run_id'],frame=0,
                                    question='为什么选择这个动作？',language='zh',focus='executed')
                            c.question(language)
                    while not c.view['ended']:
                        require(c.view['state']['frame']<120,'Unterminated round exceeded horizon');c.action('WAIT')
                    history(c)
                    if stage=='task1' and round_index==1 and group=='A':c.question('en' if language=='zh' else 'zh')
                    c.command('action',409,'round_ended',action='WAIT');c.command('next')
            require(c.view['flow']['stage']=='questionnaire','Questionnaire is not after all six rounds')
            items=c.view['questionnaire']['items'];kinds=[x.get('prediction_kind') for x in items]
            require(len(items)==11 and kinds.count('next_action')==4 and kinds.count('wait_three')==4 and sum(x['type']=='scale' for x in items)==3,'Frozen questionnaire is not 4+4+3')
            answers={x['id']:(4 if x['type']=='scale' else x['options'][0]['value']) for x in items}
            first=dict(list(answers.items())[:4]);c.command('questionnaire',answers=first,submit=False);c.refresh()
            require(c.view['questionnaire']['draft']==first,'Questionnaire draft did not survive refresh')
            c.command('questionnaire',400,'questionnaire_incomplete',answers={},submit=True)
            c.command('questionnaire',answers=answers,submit=True);c.refresh()
            require(c.view['flow']['stage']=='completed' and not c.view['answers'],'Completion or final answer suppression failed')
        h.check('six_fixed_rounds_real_bilingual_A_answers_B_rejection_Task2_old_new_denial_history_and_questionnaire')
        rows=final_database_check(args.database,clients,h)
        checkpoint=[]
        for c in clients:
            cookie=next((x for x in c.jar if x.name==COOKIE),None)
            require(cookie and cookie.has_nonstandard_attr('HttpOnly'),'Dedicated HttpOnly session cookie missing')
            checkpoint.append({'name':c.name,'cookie':cookie.value,'view_sha256':digest(c.view),'session_id':c.view['session_id']})
        save(output/'restart_checkpoint.json',{'plan':plan,'clients':checkpoint,'participants':rows,'confirmed_gameplay_steps':h.gameplay_steps})
        report={'version':VERSION,'status':'passed_real_http_flow','checks':h.checks,'participants':rows,'http_requests':h.requests,
            'confirmed_gameplay_steps':h.gameplay_steps,'questions_submitted':h.questions,'answers':h.answers,
            'historical_replay_step_upper_bound':h.questions,'all_steps_upper_bound':h.gameplay_steps+h.questions,
            'model_capability_claimed':False,'answer_quality_revalidated':False,'browser_verified':False,'process_restart_verified':False,'formal_sample':False}
        save(output/'report.json',report);return report
    except BaseException as error:
        save(output/'failure.json',{'version':VERSION,'error':repr(error),'pending':h.pending,'http_requests':h.requests,
            'confirmed_gameplay_steps':h.gameplay_steps,'questions_submitted':h.questions,'automatic_retry':False,'passed':False})
        raise
    finally:h.journal.close()


def verify_restart(args):
    """Read-only HTTP after an operator has actually restarted the same service."""
    root=Path(args.verify_restart).resolve();saved=json.loads((root/'restart_checkpoint.json').read_bytes());p=saved['plan']
    db_identity(p['database'],p['expected_manifest_sha256'])
    output=root/'post_restart';output.mkdir(exist_ok=False);h=Harness(p['base'],output);clients=[]
    try:
        for record in saved['clients']:
            jar=http.cookiejar.CookieJar();jar.set_cookie(http.cookiejar.Cookie(0,COOKIE,record['cookie'],None,False,'127.0.0.1',False,False,'/',True,False,None,True,None,None,{'HttpOnly':None},False))
            c=Client(h,record['name'],jar);c.get()
            require(digest(c.view)==record['view_sha256'],'Confirmed state changed across the operator restart');clients.append(c)
        h.gameplay_steps=saved['confirmed_gameplay_steps']
        require(final_database_check(p['database'],clients,h)==saved['participants'],'Persisted experiment records changed')
        result={'status':'saved_sessions_and_records_restored','new_gameplay_steps':0,'new_questions':0,
            'restart_performed_by_operator':True,'process_lifecycle_independently_observed':False}
        save(output/'report.json',result);return result
    finally:h.journal.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',default='http://127.0.0.1:8009')
    parser.add_argument('--database',type=Path);parser.add_argument('--expected-manifest-sha256');parser.add_argument('--output',type=Path)
    group=parser.add_mutually_exclusive_group();group.add_argument('--execute',action='store_true');group.add_argument('--verify-restart',type=Path)
    args=parser.parse_args()
    result=verify_restart(args) if args.verify_restart else execute(args) if args.execute else {'version':VERSION,'execution':False,'budget':CAP,
        'requires':'Already started genuine service, empty separate *_http_qa.sqlite3 and explicit release SHA; --execute'}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
