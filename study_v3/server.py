"""Single same-origin HTTP service for all three turn-based domains."""
import argparse
from contextlib import closing, contextmanager
import csv
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

from . import RELEASE_ID, SUPPORTED_RELEASE_IDS, KITCHEN_SUPPORTED_RELEASE_IDS
from .config import Settings
from .registry import MODULES, engine, demonstration
from .kitchen_tutorial import VERSION as KITCHEN_TUTORIAL_VERSION
from .store import Store, StudyError, encode
from . import prolific, automatic_explanations, understanding, rewards

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'study_v3/web'
COOKIE='policylens_study_v3'

@contextmanager
def export_file(store, release_id=None, mode=None, csv_format=False):
    """Serialize a consistent export to disk, releasing DB before HTTP output.

    Both formats retain the existing logical API. A row, one database batch and
    small I/O buffers are the only resident export data, regardless of history.
    The anonymous temporary file is private and is deleted even on disconnect.
    """
    with tempfile.TemporaryFile(mode='w+b') as output:
        text = io.TextIOWrapper(output, encoding='utf-8', newline='')
        try:
            writer = csv.writer(text) if csv_format else None
            if writer:
                writer.writerow(['table','record_json'])
            with closing(store.iter_export(release_id, mode)) as records:
                for item in records:
                    if writer:
                        writer.writerow([item['table'],encode(item['record'])])
                    else:
                        text.write(encode(item) + '\n')
            text.flush()
        finally:
            # Leave the binary file alive for its bounded HTTP read loop.
            text.detach()
        output.seek(0)
        yield output

def manifest(settings):
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): commit=os.environ.get('RENDER_GIT_COMMIT','unknown')
    source=hashlib.sha256()
    paths=list((ROOT/'study_v3').rglob('*.py'))+list(WEB.glob('*'))
    paths+=[ROOT/(p.replace('.','/')+'.py') for p in MODULES.values()]
    # Warehouse runs the preserved physical engine and the pinned historical
    # controller. Include its actual transitive runtime sources and Actor.
    paths+=list((ROOT/'env/warehouse').rglob('*.py'))
    paths+=[ROOT/'backend/adapters/warehouse_explanations.py',
            ROOT/'output/deployment/warehouse_mappo_v68_6x7_actor.npz']
    paths+=list((ROOT/'configs').glob('study_v3_*.json'))
    paths+=[ROOT/'ui/domain_hub_server.py',ROOT/'requirements-render.txt',ROOT/'.python-version']
    paths+=[ROOT/'docs/consent_review_content.json']
    for path in sorted(set(paths)):
        if path.is_file():source.update(str(path.relative_to(ROOT)).encode()+b'\0'+path.read_bytes())
    return {'release_id':RELEASE_ID,'commit':commit,'source_sha256':source.hexdigest(),
        'kitchen_rules':engine('kitchen').rule_metadata(),
        'kitchen_tutorial_version':KITCHEN_TUTORIAL_VERSION,
        'enrollment':{'group_selection':'participant_choice_at_new_domain',
            'resume_preserves_group':True,'parallel_domains':True,
            'compatible_session_releases':[RELEASE_ID],
            'compatible_session_releases_by_domain':{
                domain:sorted(KITCHEN_SUPPORTED_RELEASE_IDS if domain=='kitchen' else SUPPORTED_RELEASE_IDS)
                for domain in MODULES}},
        'domains':{k:{'url':'/'+k+'/','version':engine(k).VERSION} for k in MODULES},
        'explanations':{'group':'A','task':2,'active_only':True,
            'automatic_for_new_enrollments':settings.automatic_explanations,'automatic_version':automatic_explanations.VERSION},
        'understanding_ratings':{'enabled_for_new_enrollments':settings.understanding_ratings,
            'version':understanding.VERSION,'tasks':list(understanding.TASKS),'groups':list(understanding.GROUPS),
            'checkpoints':list(understanding.CHECKPOINTS),'scale':[1,5],
            'progress_basis':'turn_budget_with_end_rating_on_early_completion'},
        'understanding_ratings_required':True,
        'optional_prompts_for_new_enrollments':settings.optional_prompts,
        'rewards':{domain:rewards.policy(domain) for domain in MODULES},
        'default_language':'en','mode':settings.mode,'storage_persistent':settings.persistent,
        'prolific':{'entry_path':'/prolific/','assignment':'randomized_block_6',
            'one_game_per_participant':True,'consent_version':prolific.CONSENT_VERSION},
        'semantic_qa_configured':settings.llm_configured,'deployment_validation_complete':settings.verified,'study_ready':settings.ready,
        'human_effect_status':'not_measured','target_task2_relative_gain':0.5}

class QAReadinessMonitor:
    """Retry failed real QA checks without holding HTTP or database locks.

    Only one background probe runs at a time, at most once per interval. The
    provider has its normal bounded request timeout. Closing the server cancels
    future probes and discards an in-flight result without waiting for its I/O.
    """
    def __init__(self, store, probe, *, interval=30.0, clock=time.monotonic):
        if interval <= 0:
            raise ValueError('positive_qa_probe_interval_required')
        self.store, self.probe, self.interval, self.clock = store, probe, interval, clock
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None
        self._busy = False
        self._last_attempt = None

    def check(self):
        with self._lock:
            if self._stopped.is_set() or self._busy or not self.store.settings.ready:
                return False
            healthy, version = self.store.qa_health_snapshot()
            now = self.clock()
            if healthy or (self._last_attempt is not None and now - self._last_attempt < self.interval):
                return False
            self._busy, self._last_attempt = True, now
        try:
            result = self.probe()
            healthy = isinstance(result, dict) and result.get('status') in ('answered', 'clarification')
            outcome = 'validated' if healthy else 'unavailable_or_invalid_result'
        except Exception:
            healthy, outcome = False, 'probe_exception'
        with self._lock:
            self._busy = False
            if self._stopped.is_set():
                return False
            updated = self.store.set_qa_health(healthy, expected_version=version)
        if updated and healthy:
            logging.info('QA readiness probe validated; study admission available')
        elif updated:
            # Fixed outcome labels only: never log provider responses, URLs,
            # questions, credentials or exception text.
            logging.warning('QA readiness probe %s; retry in %.0f seconds', outcome, self.interval)
        return updated and healthy

    def _run(self):
        while not self._stopped.is_set():
            self.check()
            if self._stopped.wait(self.interval):
                break

    def start(self):
        with self._lock:
            if self._thread is not None or self._stopped.is_set() or not self.store.settings.ready:
                return
            self._thread = threading.Thread(target=self._run, name='study-qa-readiness', daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._stopped.set()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=0.1)


class StudyHTTPServer(ThreadingHTTPServer):
    def shutdown(self):
        if hasattr(self, 'qa_monitor'):
            self.qa_monitor.stop()
        super().shutdown()

    def server_close(self):
        if hasattr(self, 'qa_monitor'):
            self.qa_monitor.stop()
        super().server_close()
        if hasattr(self,'store'):self.store.db.close()

def make_server(settings,host='127.0.0.1',port=8010,explainer=None):
    # Cold demonstrations perform complete simulated trajectories. Build them
    # before opening the database or accepting requests so enrollment/recovery
    # only read the cache and never hold a transaction while doing this work.
    for domain in MODULES:
        if domain!='kitchen':demonstration(domain)
    if explainer is None:
        try:
            from .qa import Explainer
            explainer=Explainer(settings)
        except ImportError: pass
    store=Store(settings,explainer)
    # A configured service is not ready until a real, validated answer succeeds.
    # Run this check after opening the HTTP server so a transient model failure
    # cannot permanently lock enrollment or delay liveness requests.
    store.qa_healthy=False
    def qa_probe():
        probe_engine=engine('pong');probe_state=probe_engine.initial_state(26092000,2)
        return explainer.answer(probe_engine,probe_state,probe_engine.decide(probe_state),
            'What is your next action?', 'en', [], [probe_engine.public_state(probe_state)])
    release=manifest(settings)
    release['study_ready']=store.ready
    with store.db.transaction() as db:
        previous=db.one('SELECT manifest_json FROM pl3_releases WHERE id=?',(RELEASE_ID,))
        conflict=bool(previous and json.loads(previous['manifest_json'])['source_sha256']!=release['source_sha256'])
        if not previous:
            db.execute('INSERT INTO pl3_releases VALUES(?,?,?)',(RELEASE_ID,encode(release),time.time()))
    if conflict and settings.mode=='pilot':
        store.db.close()
        raise RuntimeError('release_id_reused_with_changed_source')

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        server_version='PolicyLens'
        def handle(self):
            try:super().handle()
            except (ConnectionResetError,BrokenPipeError):pass
        def log_message(self,format,*args):
            # Do not log query strings, anonymous participant codes, questions or secrets.
            logging.info('%s %s',self.command,urlsplit(self.path).path)
        def reply(self,status,payload,content_type='application/json; charset=utf-8',token=None):
            data=payload if isinstance(payload,bytes) else encode(payload).encode()
            self.reply_headers(status,len(data),content_type,token)
            self.wfile.write(data)
        def reply_headers(self,status,length,content_type,token=None):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(length))
            self.send_header('Cache-Control','no-store, private')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Permissions-Policy','camera=(), microphone=(), geolocation=()')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if token:
                secure='; Secure' if settings.origin.startswith('https:') else ''
                self.send_header('Set-Cookie',f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000{secure}')
            self.end_headers()
        def reply_file(self,stream,content_type):
            stream.seek(0,os.SEEK_END);length=stream.tell();stream.seek(0)
            self.reply_headers(200,length,content_type)
            try:
                while chunk := stream.read(64 * 1024):
                    self.wfile.write(chunk)
            except (BrokenPipeError,ConnectionResetError):
                self.close_connection=True
        def token(self):
            try:
                c=SimpleCookie();c.load(self.headers.get('Cookie',''))
                return c[COOKIE].value if COOKIE in c else None
            except Exception:return None
        def admin(self):
            supplied=self.headers.get('Authorization','')
            return bool(settings.admin_token) and hmac.compare_digest(supplied,'Bearer '+settings.admin_token)
        def body(self):
            try:size=int(self.headers.get('Content-Length','0'))
            except ValueError:raise StudyError('invalid_request')
            if not 0<size<=100_000:raise StudyError('request_too_large',413)
            origin=self.headers.get('Origin')
            if origin and origin.rstrip('/')!=settings.origin:raise StudyError('origin_not_allowed',403)
            if self.headers.get('Sec-Fetch-Site')=='cross-site':raise StudyError('origin_not_allowed',403)
            if not self.headers.get('Content-Type','').startswith('application/json'):raise StudyError('json_required',415)
            try:payload=json.loads(self.rfile.read(size))
            except (ValueError,UnicodeDecodeError):raise StudyError('invalid_json')
            if not isinstance(payload,dict):raise StudyError('invalid_request')
            return payload
        def failure(self,exc):
            self.close_connection=True
            if isinstance(exc,StudyError):self.reply(exc.status,{'error':exc.code})
            else:
                logging.error('Request failed: %s',type(exc).__name__)
                self.reply(500,{'error':'server_error'})
        def do_GET(self):
            try:
                parts=urlsplit(self.path);path=parts.path;q=parse_qs(parts.query)
                if path.startswith(('/warehouse/api/','/pong/api/')) or path in ('/api/view','/api/study/reference-trajectory'):
                    self.reply(410,{'error':'release_changed','message_en':'This earlier study version cannot continue here. Please contact the researcher before beginning a new task.','message_zh':'旧版本任务无法在此继续，请先联系研究者，再开始新任务。'});return
                if path in ('/','/warehouse/','/pong/','/kitchen/','/prolific/'):
                    self.reply(200,(WEB/'index.html').read_bytes(),'text/html; charset=utf-8');return
                if path in ('/warehouse','/pong','/kitchen'):
                    self.send_response(308);self.send_header('Location',path+'/');self.send_header('Content-Length','0');self.end_headers();return
                assets={'/study-assets/app.js':('app.js','text/javascript; charset=utf-8'),
                    '/study-assets/prolific.js':('prolific.js','text/javascript; charset=utf-8'),
                    '/study-assets/board.js':('board.js','text/javascript; charset=utf-8'),
                    '/study-assets/styles.css':('styles.css','text/css; charset=utf-8'),
                    '/study-assets/favicon.svg':('favicon.svg','image/svg+xml')}
                if path in assets:
                    name,mime=assets[path];self.reply(200,(WEB/name).read_bytes(),mime);return
                if path=='/health':
                    with store.db.transaction(read_only=True) as db:db.execute('SELECT 1')
                    self.reply(200,{'status':'ok','service':'policylens-three-domain',
                        'release_id':RELEASE_ID,'study_ready':store.ready,
                        'domains':{d:'/'+d+'/' for d in MODULES}});return
                if path=='/api/release':self.reply(200,{**release,'study_ready':store.ready});return
                if path=='/api/prolific/info':
                    consent=json.loads((ROOT/'docs/consent_review_content.json').read_text())
                    self.reply(200,{'ready':prolific.ready(store),'consent':consent,
                        'consent_version':prolific.CONSENT_VERSION});return
                if path=='/api/prolific/session':
                    self.reply(200,prolific.resume(store,self.token(),{k:v[0] for k,v in q.items()} if q else None));return
                if path=='/api/prolific/admin/payments':
                    if not self.admin():raise StudyError('researcher_access_required',403)
                    self.reply(200,{'records':prolific.payment_records(store)});return
                if path=='/api/study/view':
                    iid=q.get('instance_id',[None])[0]
                    result=store.view(self.token(),iid) if iid else store.recover_view(self.token(),q.get('domain',[None])[0])
                    self.reply(200,result);return
                if path=='/api/study/frame':
                    try:turn=int(q.get('turn',['-1'])[0])
                    except ValueError:raise StudyError('invalid_turn')
                    self.reply(200,store.frame(self.token(),q.get('instance_id',[None])[0],q.get('run_id',[None])[0],turn));return
                if path=='/api/study/admin/export':
                    if not self.admin():raise StudyError('researcher_access_required',403)
                    csv_format=q.get('format',['jsonl'])[0]=='csv'
                    content_type='text/csv; charset=utf-8' if csv_format else 'application/x-ndjson; charset=utf-8'
                    with export_file(store,q.get('release_id',[None])[0],q.get('mode',[None])[0],csv_format) as output:
                        self.reply_file(output,content_type)
                    return
                raise StudyError('not_found',404)
            except Exception as exc:self.failure(exc)
        def do_POST(self):
            try:
                path=urlsplit(self.path).path;payload=self.body()
                if path=='/api/prolific/admin/release-slot':
                    if not self.admin():raise StudyError('researcher_access_required',403)
                    self.reply(200,prolific.release_slot(store,payload));return
                if path.startswith(('/warehouse/api/','/pong/api/')) or path=='/api/study/command':
                    self.reply(410,{'error':'release_changed','message_en':'This earlier study version cannot continue here. Please contact the researcher.','message_zh':'旧版本任务无法在此继续，请联系研究者。'});return
                if path=='/api/study/session':
                    token,result=store.create(payload,self.token(),self.admin());self.reply(200,result,token=token);return
                if path=='/api/prolific/enrol':
                    token,result=prolific.enrol(store,payload,self.token());self.reply(200,result,token=token);return
                if path=='/api/study/ask':self.reply(200,store.ask(self.token(),payload));return
                if path=='/api/study/automatic-explanation-displayed':
                    self.reply(200,store.acknowledge_automatic(self.token(),payload.get('instance_id'),payload.get('explanation_id')));return
                if path=='/api/study/answer-displayed':
                    self.reply(200,store.acknowledge_answer(self.token(),payload.get('instance_id'),payload.get('question_id')));return
                prefix='/api/study/'
                if path.startswith(prefix):self.reply(200,store.command(self.token(),path[len(prefix):],payload));return
                raise StudyError('not_found',404)
            except Exception as exc:self.failure(exc)
    server=StudyHTTPServer((host,port),Handler)
    server.daemon_threads=True
    server.store=store
    server.release=release
    server.qa_monitor=QAReadinessMonitor(store, qa_probe)
    server.qa_monitor.start()
    return server

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8010')))
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    server=make_server(Settings.from_env(),args.host,args.port)
    logging.info('PolicyLens %s ready on port %s; study_ready=%s',RELEASE_ID,server.server_port,server.store.ready)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
