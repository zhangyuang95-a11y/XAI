"""Explicit local/staging research server. Scientific and operational gates stay closed."""
from contextlib import nullcontext
import sqlite3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import os
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from runtime_v642.catalog import decode
from runtime_v642.contracts import RuntimeUnavailable
from runtime_v642.store import StudyError, check, encode
from runtime_v642.registry import RuntimeRegistry
from .store import StudyStore, CONSENT_VERSION, PROTOCOL_ID, RATINGS
from .storage import StorageError
from .qa import EvidenceError, EvidenceQuestionComposer, OpenAICompatibleSelector

ROOT=Path(__file__).resolve().parents[1]
WEB=Path(__file__).parent/'web'


def make_server(store, *, admin_token, host='127.0.0.1', port=0):
    check(isinstance(store, StudyStore) and type(admin_token) is str and len(admin_token)>=32, 'explicit_study_store_admin_required')
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*_): pass
        def admin(self):
            offered=self.headers.get('X-Admin-Token','')
            return bool(offered) and hmac.compare_digest(offered.encode(),admin_token.encode())
        def token(self):
            value=self.headers.get('Authorization','');check(value.startswith('Bearer '),'session_required',401)
            return value[7:]
        def reply(self,value,status=200,mime='application/json; charset=utf-8'):
            from runtime_v642.server import _plain
            raw=value if type(value) is bytes else encode(_plain(value)).encode()
            self.send_response(status)
            for k,v in {'Content-Type':mime,'Content-Length':str(len(raw)),'Cache-Control':'no-store',
                'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
                'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"}.items():self.send_header(k,v)
            if status>=400:self.send_header('Connection','close');self.close_connection=True
            self.end_headers();self.wfile.write(raw)
        def handle_request(self,post=False):
            try:
                url=urlsplit(self.path);path=url.path
                if not post and path=='/health':return self.reply(dict(alive=True,pilot_opened=False))
                if not post and path in ('/','/study/','/warehouse/','/pong/','/kitchen/'):
                    return self.reply((WEB/'index.html').read_bytes(),mime='text/html; charset=utf-8')
                static={'/study-assets/app.js':(WEB/'app.js','application/javascript'),
                    '/study-assets/styles.css':(WEB/'styles.css','text/css'),
                    '/study-assets/board.js':(ROOT/'study_v3/web/board.js','application/javascript'),
                    '/study-assets/preview-board.js':(ROOT/'ui/model_preview/board.js','application/javascript'),
                    '/study-assets/input-queue.js':(ROOT/'ui/model_preview/input_queue.js','application/javascript')}
                if not post and path in static:
                    file,mime=static[path];return self.reply(file.read_bytes(),mime=mime)
                check(path.startswith('/study-api/'),'not_found',404)
                store._gate()
                if not post and path=='/study-api/status':
                    return self.reply(dict(protocol_id=PROTOCOL_ID,consent_version=CONSENT_VERSION,mode=store.mode,
                        scientific_sample=False,recruitment_opened=False,questionnaire_fields=list(RATINGS),
                        primary_endpoint='Task3',domains=list(store.scenarios),
                        question_service_mode='template_only' if store.question_composer is None or store.question_composer.provider is None else 'configured_not_production_verified'))
                payload=None
                if post:
                    origin=self.headers.get('Origin')
                    check(not origin or urlsplit(origin).netloc==self.headers.get('Host'),'cross_origin_request',403)
                    check(self.headers.get_content_type()=='application/json','json_required',415)
                    length=self.headers.get('Content-Length','');check(length.isdigit() and 0<int(length)<=65536,'invalid_body',413)
                    payload=decode(self.rfile.read(int(length)))
                if post and path=='/study-api/enroll':
                    # This preview endpoint is researcher-only, never a substitute
                    # for a scientifically admitted participant recruitment route.
                    check(self.admin(),'researcher_preview_only',403)
                    allowed={'domain','participant_reference','consent','consent_version','language','group','question_transmission_consent'}
                    check(set(payload)<=allowed and {'domain','participant_reference','consent','consent_version'}<=set(payload),'invalid_enrollment')
                    return self.reply(store.enroll(**payload,admin=True))
                parts=path.split('/')[2:];check(len(parts)==2,'not_found',404)
                iid,action=parts;token=self.token()
                if action=='view' and not post:value=store.view(token,iid)
                elif action=='demo' and post:value=store.demonstration(token,iid)
                elif action=='demo-complete' and post:value=store.acknowledge_demonstration(token,iid)
                elif action=='preparation' and post:value=store.complete_preparation(token,iid,**payload)
                elif action=='next-task' and post:value=store.start_task(token,iid)
                elif action=='action' and post:
                    check(set(payload)=={'command_id','expected_turn','human_action'},'invalid_action')
                    value=store.action(token,iid,**payload)
                elif action=='question' and post:
                    check({'question'}<=set(payload)<= {'question','decision_turn','turn','target_task','language','human_actions'},'invalid_question')
                    value=store.ask(token,iid,**payload)
                elif action=='answers' and not post:value=store.answers(token,iid)
                elif action=='survey' and post:
                    check({'answers'}<=set(payload)<= {'answers','feedback'},'invalid_questionnaire')
                    value=store.submit_survey(token,iid,**payload)
                elif action=='understanding' and not post:value=store.understanding(token,iid)
                elif action=='understanding' and post:
                    check(set(payload)=={'answers','version','sha256'},'invalid_understanding')
                    value=store.submit_understanding(token,iid,**payload)
                elif action=='event' and post:value=store.event(token,iid,**payload)
                elif action=='replay' and not post:
                    query=parse_qs(url.query);check(set(query)=={'task'} and len(query['task'])==1,'invalid_query')
                    value=store.replay(token,iid,task=int(query['task'][0]))
                elif action=='export' and not post:
                    check(self.admin(),'admin_required',403);return self.reply(store.export_instance(token,iid,admin=True))
                else:raise StudyError('not_found',404)
                return self.reply(store.participant_view(token,iid,value))
            except StudyError as e:self.reply(dict(error=e.code),e.status)
            except RuntimeUnavailable as e:self.reply(dict(error=str(e),pilot_opened=False),503)
            except (ValueError,TypeError,KeyError,EvidenceError):self.reply(dict(error='invalid_request'),400)
            except (sqlite3.Error,StorageError):self.reply(dict(error='study_storage_unavailable'),503)
        def dispatch(self,post=False):
            context=getattr(store,'request_context',nullcontext) if self.path.startswith('/study-api/') else nullcontext
            # Request-scoped asset validation happens before routing. Failure
            # must produce a closed API response rather than an empty socket.
            try:
                with context():self.handle_request(post)
            except RuntimeUnavailable as exc:
                self.reply(dict(error=str(exc),pilot_opened=False),503)
        def do_GET(self):self.dispatch()
        def do_POST(self):self.dispatch(True)
    server=ThreadingHTTPServer((host,port),Handler);server.daemon_threads=True
    return server


def compose_preview(database, *, postgres_dsn=None, postgres_schema=None):
    # Use exact inference-only copies and request-scoped immutable asset checks.
    # The manual catalog explicitly cannot grant pilot admission.
    from ui.model_preview_server import Preview, SEEDS
    from .request_physics import StudyWarehouseAdapter
    preview=Preview(':memory:')
    preview.adapters['warehouse']=StudyWarehouseAdapter(preview.catalog,preview.runtime,
        preloaded_actors=preview.adapters['warehouse']._actors)
    registry=RuntimeRegistry(preview.catalog,adapters=preview.adapters)
    kwargs=dict(registry=registry,scenario_pools={d:tuple(s[:6]) for d,s in SEEDS.items()},
        demonstration_seeds={d:s[-1] for d,s in SEEDS.items()},mode='preview')
    store=StudyStore(database,**kwargs) if postgres_dsn is None else StudyStore.from_postgres(
        dsn=postgres_dsn,schema=postgres_schema,**kwargs)
    store.question_composer=EvidenceQuestionComposer()
    store.request_context=preview.runtime.request
    return store,preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8014)
    parser.add_argument('--database',default=None)
    parser.add_argument('--postgres-env',default=None,help='Explicit new-study DSN environment variable name')
    parser.add_argument('--postgres-schema',default=None)
    parser.add_argument('--enable-consented-qa-provider',action='store_true')
    args=parser.parse_args();admin=os.environ.get('POLICYLENS_STUDY_PREVIEW_ADMIN','')
    check(len(admin)>=32,'set_private_preview_admin_environment')
    check(bool(args.database)!=bool(args.postgres_env),'choose_exactly_one_database_backend')
    dsn=None
    if args.postgres_env:
        check(args.postgres_env.startswith('POLICYLENS_STUDY_') and args.postgres_schema,'explicit_isolated_study_database_required')
        dsn=os.environ.get(args.postgres_env);check(bool(dsn),'study_database_not_configured')
    store,preview=compose_preview(args.database,postgres_dsn=dsn,postgres_schema=args.postgres_schema)
    if args.enable_consented_qa_provider:
        check(os.environ.get('POLICYLENS_STUDY_QA_TRANSMISSION_APPROVED')=='yes','explicit_provider_transmission_approval_required')
        store.question_composer=EvidenceQuestionComposer(OpenAICompatibleSelector(
            base_url=os.environ.get('POLICYLENS_STUDY_QA_BASE_URL',''),model=os.environ.get('POLICYLENS_STUDY_QA_MODEL',''),
            api_key=os.environ.get('POLICYLENS_STUDY_QA_API_KEY',''),consented=True))
    server=make_server(store,admin_token=admin,host=args.host,port=args.port)
    try:server.serve_forever()
    finally:server.server_close();store.close();preview.close()

if __name__=='__main__':main()
