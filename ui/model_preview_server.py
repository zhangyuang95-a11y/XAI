"""Manual play of exact v6.42/v5 Actors, isolated from participant enrollment."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading
from urllib.parse import urlsplit

from runtime_v642.catalog import PolicyCatalog
from runtime_v642.contracts import RuntimeUnavailable, require
from runtime_v642.legacy_domain import PongAdapter, KitchenAdapter
from .model_preview_warehouse import WarehouseAdapter

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'policylens-manual-model-preview-20261024.v1'
CATALOG_SHA256 = 'ec7b6370454321748f9767c5cbbce4f630d539afb5763da742f6da6f7f543d43'
PACKAGE = ROOT / 'releases' / RELEASE
WEB = ROOT / 'ui' / 'model_preview'
SEEDS = dict(warehouse=list(range(40786, 40810)), pong=list(range(880101, 880125)), kitchen=list(range(1000, 1024)))


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_catalog():
    path = PACKAGE / 'catalog.json'
    require(sha256(path.read_bytes()).hexdigest() == CATALOG_SHA256, 'preview catalog identity differs')
    manifest = json.loads(path.read_text())
    # Git preserves content, not the read-only bit; verify before sealing checkout copies.
    for key, item in manifest['assets'].items():
        target = PACKAGE / item['path']
        require(target.resolve().is_relative_to(PACKAGE.resolve()) and not target.is_symlink()
                and sha256(target.read_bytes()).hexdigest() == key, 'preview asset identity differs')
        target.chmod(0o444)
    path.chmod(0o444)
    return PolicyCatalog.load(path, expected_sha256=CATALOG_SHA256)


class Preview:
    def __init__(self, database):
        self.catalog = load_catalog()
        self.adapters = dict(warehouse=WarehouseAdapter(self.catalog), pong=PongAdapter(self.catalog), kitchen=KitchenAdapter(self.catalog))
        for domain, adapter in self.adapters.items():
            for variant in ('raw', 'regularized'):
                binding = self.catalog.binding(domain, variant)
                adapter._actor(binding) if domain == 'warehouse' else adapter._core(binding)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS manual_sessions (
                token_sha TEXT PRIMARY KEY, domain TEXT NOT NULL, variant TEXT NOT NULL,
                state TEXT NOT NULL, initial_state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS manual_commands (
                token_sha TEXT NOT NULL, command_id TEXT NOT NULL, payload TEXT NOT NULL,
                response TEXT NOT NULL, transition TEXT NOT NULL,
                PRIMARY KEY(token_sha, command_id));
        ''')

    def close(self):
        self.db.close()

    def metadata(self):
        models = {}
        for domain in self.adapters:
            models[domain] = dict(version='v6.42 / 20k' if domain == 'warehouse' else 'v5 / 200k',
                acceptance='calibration_pair_failed' if domain == 'warehouse' else 'historical_v5_passed',
                seeds=SEEDS[domain], policies={v:self.catalog.binding(domain, v).to_dict() for v in ('raw','regularized')})
        return dict(release_id=RELEASE, commit=os.environ.get('RENDER_GIT_COMMIT', 'local'),
            mode='manual_preview', pilot_opened=False, study_ready=False, human_study_sample=False,
            storage='isolated_ephemeral_preview_sqlite', database_is_participant_database=False,
            explanations_available=False, models=models)

    def _public(self, state, binding, last=None):
        public = self.adapters[binding.domain].public_state(state, binding)
        if binding.domain == 'warehouse':
            from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
            public['walls'] = [[p[1],p[0]] for p in sorted(layout.blocked_positions)]
            charger = layout.charger_position
            public['stations'] = [dict(id='charger',kind='charger',x=charger[1],y=charger[0])]
        return dict(state=public, binding=binding.to_dict(), last_action=last,
                    mode='manual_preview', pilot_opened=False, scientific_acceptance_claimed=False)

    def start(self, payload):
        require(set(payload) == {'domain','variant','task','seed'}, 'invalid preview selection')
        domain, variant, task, seed = (payload[k] for k in ('domain','variant','task','seed'))
        require(domain in SEEDS and variant in ('raw','regularized') and type(task) is int and task in (1,2,3)
                and type(seed) is int and seed in SEEDS[domain], 'unknown manual scene or model')
        with self.lock:
            binding = self.catalog.binding(domain, variant)
            state = self.adapters[domain].initial_state(seed, task, binding)
            result = self._public(state, binding)
            token = secrets.token_urlsafe(32)
            self.db.execute('INSERT INTO manual_sessions VALUES (?,?,?,?,?)',
                (sha256(token.encode()).hexdigest(),domain,variant,encode(state),encode(state)))
            self.db.commit()
            return dict(result, session_token=token)

    def _session(self, token):
        require(type(token) is str and 20 <= len(token) <= 100, 'session_required')
        key = sha256(token.encode()).hexdigest()
        row = self.db.execute('SELECT domain,variant,state,initial_state FROM manual_sessions WHERE token_sha=?',(key,)).fetchone()
        require(row is not None, 'preview_session_unavailable')
        domain, variant, state, initial = row
        binding = self.catalog.binding(domain, variant)
        state = json.loads(state)
        self.adapters[domain].restore(state, binding)
        return key, state, json.loads(initial), binding

    def view(self, token):
        with self.lock:
            _, state, _, binding = self._session(token)
            return self._public(state, binding)

    def step(self, token, payload):
        require(set(payload) == {'command_id','expected_turn','human_action'}, 'invalid preview command')
        command_id = payload['command_id']
        require(type(command_id) is str and 8 <= len(command_id) <= 80
                and type(payload['expected_turn']) is int, 'invalid command identity')
        require(payload['human_action'] is None or type(payload['human_action']) is str, 'invalid human action')
        with self.lock:
            key, state, _, binding = self._session(token)
            prior = self.db.execute('SELECT payload,response FROM manual_commands WHERE token_sha=? AND command_id=?',(key,command_id)).fetchone()
            if prior:
                require(prior[0] == encode(payload), 'command_id_reused_with_different_payload')
                return json.loads(prior[1])
            require(state['turn'] == payload['expected_turn'] and not state['terminal'], 'stale_or_terminal_frame')
            adapter = self.adapters[binding.domain]
            decision = adapter.decide(state, binding)
            # None is the explicit two-NN demonstration; otherwise only the human role is replaced.
            after, transition = adapter._advance(state, binding, payload['human_action'], decision)
            last = dict(ai=decision['action'], two_NN=payload['human_action'] is None,
                human=payload['human_action'] if payload['human_action'] is not None else
                    decision.get('human_action', decision['policy_actions'].get('robot_1')),
                ai_action_overrides=decision['ai_action_overrides'], execution_source=decision['execution_source'])
            require(last['ai_action_overrides'] == 0, 'AI action was overridden')
            result = self._public(after, binding, last)
            with self.db:
                self.db.execute('UPDATE manual_sessions SET state=? WHERE token_sha=?',(encode(after),key))
                self.db.execute('INSERT INTO manual_commands VALUES (?,?,?,?,?)',
                    (key,command_id,encode(payload),encode(result),encode(transition)))
            return result

    def export(self, token):
        with self.lock:
            key, state, initial, binding = self._session(token)
            records = [json.loads(r[0]) for r in self.db.execute('SELECT transition FROM manual_commands WHERE token_sha=? ORDER BY rowid',(key,))]
            from .model_preview_warehouse import REPLAY_VERSION as W_REPLAY
            from runtime_v642.legacy_domain import REPLAY_VERSION as L_REPLAY
            bundle = dict(version=W_REPLAY if binding.domain == 'warehouse' else L_REPLAY,
                binding=binding.to_dict(), initial_state=initial, records=records, final_state=state, two_frozen_NN=False)
            self.adapters[binding.domain].replay(bundle, binding)
            return dict(release_id=RELEASE, mode='manual_preview', human_study_sample=False,
                scientific_evaluation=False, independently_replayed=True, replay=bundle)


def make_server(preview, host='127.0.0.1', port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value, status=200, mime='application/json; charset=utf-8'):
            raw = value if type(value) is bytes else encode(value).encode()
            self.send_response(status)
            for k,v in {'Content-Type':mime,'Content-Length':str(len(raw)),'Cache-Control':'no-store',
                'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
                'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"}.items():
                self.send_header(k,v)
            self.end_headers()
            self.wfile.write(raw)

        def token(self):
            value = self.headers.get('Authorization','')
            require(value.startswith('Bearer '), 'session_required')
            return value[7:]

        def do_GET(self):
            try:
                path = urlsplit(self.path).path
                if path in ('/','/warehouse/','/pong/','/kitchen/','/try/'):
                    return self.reply((WEB/'index.html').read_bytes(),mime='text/html; charset=utf-8')
                if path == '/health':
                    return self.reply(dict(alive=True,release_id=RELEASE,mode='manual_preview',pilot_opened=False))
                if path in ('/api/release','/manual-api/models'):
                    return self.reply(preview.metadata())
                if path == '/manual-api/session':
                    return self.reply(preview.view(self.token()))
                if path == '/manual-api/export':
                    return self.reply(preview.export(self.token()))
                static = {'/preview-assets/app.js':(WEB/'app.js','application/javascript'),
                    '/preview-assets/styles.css':(WEB/'styles.css','text/css'),
                    '/preview-assets/board.js':(ROOT/'study_v3/web/board.js','application/javascript')}
                if path in static:
                    file,mime = static[path]
                    return self.reply(file.read_bytes(),mime=mime)
                return self.reply(dict(error='not_found',pilot_opened=False),404)
            except (RuntimeUnavailable,ValueError,KeyError):
                return self.reply(dict(error='preview_binding_or_session_unavailable'),409)

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin:
                    require(urlsplit(origin).netloc == self.headers.get('Host'), 'cross_origin_request')
                require(self.headers.get_content_type() == 'application/json', 'json_required')
                length = self.headers.get('Content-Length','')
                require(length.isdigit() and 0 < int(length) <= 65536, 'invalid_body')
                payload = json.loads(self.rfile.read(int(length)))
                require(type(payload) is dict, 'invalid_body')
                path = urlsplit(self.path).path
                if path == '/manual-api/start':
                    return self.reply(preview.start(payload))
                if path == '/manual-api/action':
                    return self.reply(preview.step(self.token(),payload))
                return self.reply(dict(error='not_found',pilot_opened=False),404)
            except (RuntimeUnavailable,ValueError,KeyError,TypeError) as exc:
                return self.reply(dict(error=str(exc)),409)

    server = ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8011')))
    parser.add_argument('--database',default='/tmp/policylens-manual-model-preview-v1.sqlite3')
    args = parser.parse_args()
    preview = Preview(args.database)
    server = make_server(preview,args.host,args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        preview.close()


if __name__ == '__main__':
    main()
