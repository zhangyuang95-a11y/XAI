"""Isolated, disposable Task 2 demo; never loads production database or Prolific settings."""
import json
import os
import secrets
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .config import Settings
from .registry import MODULES, engine
from .server import make_server, WEB
from .store import StudyError


def make_preview_server(*, host='127.0.0.1', port=9131, origin=None, database=None):
    # Ignore production storage/Prolific settings. Only explicitly configured LLM settings are used.
    path=Path(database or '/tmp/policylens-task2-preview/preview.sqlite3')
    path.parent.mkdir(parents=True, exist_ok=True)
    settings=Settings(database=str(path), mode='preview', automatic_explanations=True, understanding_ratings=True, optional_prompts=True,
                      admin_token=secrets.token_urlsafe(32),
                      llm_api_key=os.environ.get('POLICYLENS_LLM_API_KEY',''),
                      llm_base_url=(os.environ.get('POLICYLENS_LLM_BASE_URL','https://api.deepseek.com') if os.environ.get('POLICYLENS_LLM_API_KEY') else ''),
                      llm_model=os.environ.get('POLICYLENS_LLM_MODEL','deepseek-chat'),
                      origin=origin or f'http://127.0.0.1:{port}')
    server=make_server(settings, host=host, port=port)
    store=server.store

    def prepare(domain, token, language, restart=False):
        name='friend-preview-'+uuid.uuid4().hex
        if token and not restart:
            try:
                with store.db.transaction(read_only=True) as db:
                    participant=store._participant(db, token)
                if participant['id'].startswith('friend-preview-'):
                    name=participant['id']
                else:
                    token=None
            except StudyError:
                token=None
        if restart:
            token=None
        token, view=store.create(dict(participant_id=name, domain=domain, mode='preview',
                                     group='A', language=language, consent=True), token=token, admin=True)
        if view['stage']!='demo':
            return token
        def command(kind, **fields):
            nonlocal view
            view=store.command(token, kind, dict(instance_id=view['instance_id'],
                revision=view['revision'], command_id=uuid.uuid4().hex, **fields))
        command('demo_skip')
        # This synthetic baseline exists only in this disposable demo database.
        with store.db.transaction() as db:
            db.execute("UPDATE pl3_runs SET status='completed' WHERE id=?", (view['run_id'],))
        command('next')
        # Begin at Task 2 turn zero; the participant reaches the first natural node.
        return token

    original=server.RequestHandlerClass
    class ShareHandler(original):
        def reply(self, status, payload, content_type='application/json; charset=utf-8', token=None):
            if isinstance(payload, dict) and ('release_id' in payload or 'instance_id' in payload):
                payload={**payload, 'share_preview':True, 'share_qa_available':settings.llm_configured}
                if 'task_runs' in payload:
                    payload['task_runs']=[r for r in payload['task_runs'] if r['task']==2]
            return super().reply(status, payload, content_type, token)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ('/', '/try/', '/try/pong', '/try/warehouse', '/try/kitchen'):
                return self.reply(200, (WEB/'share-preview.html').read_bytes(), 'text/html; charset=utf-8')
            if path=='/share-preview.js':
                return self.reply(200, (WEB/'share-preview.js').read_bytes(), 'text/javascript; charset=utf-8')
            allowed={'/health','/api/release','/api/study/view','/api/study/frame',
                     '/pong/','/warehouse/','/kitchen/'}
            if path not in allowed and not path.startswith('/study-assets/'):
                return self.failure(StudyError('not_found',404))
            super().do_GET()

        def do_POST(self):
            path=urlsplit(self.path).path
            if path=='/api/try/start':
                try:
                    payload=self.body()
                    domain=payload.get('domain')
                    if domain not in MODULES:
                        raise StudyError('unknown_domain')
                    language='zh' if payload.get('language')=='zh' else 'en'
                    token=prepare(domain, self.token(), language, payload.get('restart') is True)
                    self.reply(200, {'url':'/'+domain+'/'}, token=token)
                except Exception as exc:
                    self.failure(exc)
                return
            allowed={'open_question_guide','ask','answer-displayed','action','request_explanation','confirm_explanation','automatic-explanation-displayed','language','timing','understanding_rating','skip_prompts'}
            if not path.startswith('/api/study/') or path.removeprefix('/api/study/') not in allowed:
                return self.failure(StudyError('preview_only',403))
            super().do_POST()

    server.RequestHandlerClass=ShareHandler
    return server


if __name__=='__main__':
    port=int(os.environ.get('PORT','9131'))
    origin=os.environ.get('RENDER_EXTERNAL_URL') or f'http://127.0.0.1:{port}'
    server=make_preview_server(host='0.0.0.0', port=port, origin=origin)
    print('Task 2 share preview ready', flush=True)
    server.serve_forever()
