"""Single same-origin HTTP service for all three turn-based domains."""
import argparse
import csv
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

from . import RELEASE_ID
from .config import Settings
from .registry import MODULES, engine
from .store import Store, StudyError, encode

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'study_v3/web'
COOKIE='policylens_study_v3'

def manifest(settings):
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): commit=os.environ.get('RENDER_GIT_COMMIT','unknown')
    source=hashlib.sha256()
    paths=list((ROOT/'study_v3').rglob('*.py'))+list(WEB.glob('*'))
    paths+=[ROOT/(p.replace('.','/')+'.py') for p in MODULES.values()]
    paths+=list((ROOT/'configs').glob('study_v3_*.json'))
    paths+=[ROOT/'ui/domain_hub_server.py',ROOT/'requirements-render.txt',ROOT/'.python-version']
    for path in sorted(set(paths)):
        if path.is_file():source.update(str(path.relative_to(ROOT)).encode()+b'\0'+path.read_bytes())
    return {'release_id':RELEASE_ID,'commit':commit,'source_sha256':source.hexdigest(),
        'domains':{k:{'url':'/'+k+'/','version':engine(k).VERSION} for k in MODULES},
        'explanations':{'group':'A','task':2,'active_only':True},
        'default_language':'en','mode':settings.mode,'storage_persistent':settings.persistent,
        'semantic_qa_configured':settings.llm_configured,'deployment_validation_complete':settings.verified,'study_ready':settings.ready,
        'human_effect_status':'not_measured','target_task2_relative_gain':0.5}

class StudyHTTPServer(ThreadingHTTPServer):
    def server_close(self):
        super().server_close()
        if hasattr(self,'store'):self.store.db.close()

def make_server(settings,host='127.0.0.1',port=8010,explainer=None):
    if explainer is None:
        try:
            from .qa import Explainer
            explainer=Explainer(settings)
        except ImportError: pass
    store=Store(settings,explainer)
    if settings.ready:
        try:
            probe_engine=engine('pong');probe_state=probe_engine.initial_state(26092000,2)
            probe=explainer.answer(probe_engine,probe_state,probe_engine.decide(probe_state),
                'What is your next action?', 'en', [], [probe_engine.public_state(probe_state)])
            store.qa_healthy=probe.get('status') in ('answered','clarification')
        except Exception:store.qa_healthy=False
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
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store, private')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Permissions-Policy','camera=(), microphone=(), geolocation=()')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if token:
                secure='; Secure' if settings.origin.startswith('https:') else ''
                self.send_header('Set-Cookie',f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000{secure}')
            self.end_headers(); self.wfile.write(data)
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
                if path in ('/','/warehouse/','/pong/','/kitchen/'):
                    self.reply(200,(WEB/'index.html').read_bytes(),'text/html; charset=utf-8');return
                if path in ('/warehouse','/pong','/kitchen'):
                    self.send_response(308);self.send_header('Location',path+'/');self.send_header('Content-Length','0');self.end_headers();return
                assets={'/study-assets/app.js':('app.js','text/javascript; charset=utf-8'),
                    '/study-assets/styles.css':('styles.css','text/css; charset=utf-8'),
                    '/study-assets/favicon.svg':('favicon.svg','image/svg+xml')}
                if path in assets:
                    name,mime=assets[path];self.reply(200,(WEB/name).read_bytes(),mime);return
                if path=='/health':
                    with store.db.transaction() as db:db.execute('SELECT 1')
                    self.reply(200,{'status':'ok','service':'policylens-three-domain',
                        'release_id':RELEASE_ID,'study_ready':store.ready,
                        'domains':{d:'/'+d+'/' for d in MODULES}});return
                if path=='/api/release':self.reply(200,{**release,'study_ready':store.ready});return
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
                    data=store.export(q.get('release_id',[None])[0],q.get('mode',[None])[0])
                    if q.get('format',['jsonl'])[0]=='csv':
                        output=io.StringIO();writer=csv.writer(output);writer.writerow(['table','record_json'])
                        for table,rows in data.items():
                            for row in rows:writer.writerow([table,encode(row)])
                        self.reply(200,output.getvalue().encode(),'text/csv; charset=utf-8')
                    else:
                        lines=[encode({'table':table,'record':row}) for table,rows in data.items() for row in rows]
                        self.reply(200,('\n'.join(lines)+'\n').encode(),'application/x-ndjson; charset=utf-8')
                    return
                raise StudyError('not_found',404)
            except Exception as exc:self.failure(exc)
        def do_POST(self):
            try:
                path=urlsplit(self.path).path;payload=self.body()
                if path.startswith(('/warehouse/api/','/pong/api/')) or path=='/api/study/command':
                    self.reply(410,{'error':'release_changed','message_en':'This earlier study version cannot continue here. Please contact the researcher.','message_zh':'旧版本任务无法在此继续，请联系研究者。'});return
                if path=='/api/study/session':
                    token,result=store.create(payload,self.token(),self.admin());self.reply(200,result,token=token);return
                if path=='/api/study/ask':self.reply(200,store.ask(self.token(),payload));return
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
    return server

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8010')))
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    server=make_server(Settings.from_env(),args.host,args.port)
    logging.info('PolicyLens %s ready on port %s; study_ready=%s',RELEASE_ID,server.server_port,server.release['study_ready'])
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
