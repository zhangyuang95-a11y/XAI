"""Explicit JSON core API factory; no automatic server or deployment startup."""
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import hmac
import json
import sqlite3

from .catalog import PolicyCatalog, decode
from .contracts import RuntimeUnavailable, require
from .registry import RuntimeRegistry
from .store import Store, StudyError, ReadOnlyLegacyReader, encode, check

VERSION = 'runtime-v642-core-HTTP.v2-committed-history'
MAX_BODY_BYTES = 65536


def compose_store(*, database, catalog_path, catalog_sha256, scenario_pools,
                  demonstration_seeds, mode='pilot', legacy_database=None):
    """Compose only actual versioned adapters; no fake acceptance injection.

    Unknown Warehouse formal schema remains closed. The caller supplies a
    separately created immutable catalog; this function never publishes one.
    Constructors are lazy and do not reset or run an environment.
    """
    from .warehouse import WarehouseAdapter
    from .legacy_domain import PongAdapter, KitchenAdapter
    catalog = PolicyCatalog.load(catalog_path, expected_sha256=catalog_sha256)
    legacy = None if legacy_database is None else ReadOnlyLegacyReader(legacy_database)
    registry = RuntimeRegistry(catalog, adapters=dict(warehouse=WarehouseAdapter(catalog),
        pong=PongAdapter(catalog), kitchen=KitchenAdapter(catalog)), legacy_reader=legacy)
    return Store(database, registry, scenario_pools=scenario_pools,
                 demonstration_seeds=demonstration_seeds, mode=mode)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _fields(payload, required=(), optional=()):
    check(type(payload) is dict and set(required) <= set(payload)
          and set(payload) <= set(required) | set(optional), 'unknown_or_missing_fields')
    return payload


class CoreHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_close(self):
        super().server_close()
        if getattr(self, '_close_store', False):
            self.store.close()


def make_server(store, *, host='127.0.0.1', port=0, admin_token, close_store=False):
    """Return an unstarted server. Caller must explicitly serve/shutdown it.

    This is a new API, not a drop-in replacement for the old UI/Prolific flow.
    Participant Bearer tokens scope every read/write to one persisted instance.
    Administrative capabilities use a separate X-Admin-Token header.
    """
    require(type(store) is Store and type(admin_token) is str and len(admin_token) >= 32
            and type(close_store) is bool, 'explicit core Store and private admin capability required')

    class Handler(BaseHTTPRequestHandler):
        server_version = 'PolicyLensCoreV642'

        def log_message(self, *_args):
            # Never log participant tokens, questions, private URLs or headers.
            pass

        def _admin(self):
            offered = self.headers.get('X-Admin-Token', '')
            return bool(offered) and hmac.compare_digest(offered.encode(), admin_token.encode())

        def _token(self):
            value = self.headers.get('Authorization', '')
            check(value.startswith('Bearer '), 'session_required', 401)
            return value[7:]

        def _body(self):
            check(self.headers.get_content_type() == 'application/json', 'json_required', 415)
            length = self.headers.get('Content-Length', '')
            check(length.isdigit() and 0 < int(length) <= MAX_BODY_BYTES, 'invalid_body_length', 413)
            try:
                return decode(self.rfile.read(int(length)))
            except (ValueError, UnicodeError, RuntimeUnavailable) as exc:
                raise StudyError('invalid_JSON') from exc

        def _reply(self, value, *, status=200):
            raw = encode(_plain(value)).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def _route(self, *, post=False):
            url = urlsplit(self.path)
            parts = [p for p in url.path.split('/') if p]
            if not post and parts == ['health']:
                return dict(server=VERSION, alive=True, scientific_acceptance_claimed=False)
            if not post and len(parts) == 2 and parts[0] == 'archive':
                query = parse_qs(url.query, strict_parsing=bool(url.query))
                check(set(query) <= {'export'}, 'invalid_query')
                return store.archived_instance(parts[1], export=query.get('export') == ['1'], admin=self._admin())
            check(parts and parts[0] == 'v642', 'not_found', 404)
            store._gate()  # Every new pilot route fails before auth/body/assignment when unadmitted.
            if not post and parts == ['v642', 'status']:
                return store.status()
            if not post and parts == ['v642', 'research', 'instances']:
                check(not url.query, 'invalid_query')
                return store.research_index(admin=self._admin())
            if not post and len(parts) == 5 and parts[:3] == ['v642', 'research', 'instances'] and parts[4] == 'export':
                check(not url.query, 'invalid_query')
                return store.research_export(parts[3], admin=self._admin())
            payload = self._body() if post else None
            if post and parts == ['v642', 'instances']:
                p = _fields(payload, ('domain', 'participant_reference'), ('group', 'scenario_seed'))
                return store.enroll(**p, admin=self._admin())
            check(len(parts) >= 3 and parts[:2] == ['v642', 'instances'], 'not_found', 404)
            instance_id, tail, token = parts[2], parts[3:], self._token()
            self._public_context = (token, instance_id)
            if not post:
                query = parse_qs(url.query, strict_parsing=bool(url.query))
                if not tail:
                    check(not query, 'invalid_query')
                    return store.view(token, instance_id)
                if tail == ['replay']:
                    check(set(query) == {'task'} and len(query['task']) == 1, 'invalid_query')
                    return store.replay(token, instance_id, task=int(query['task'][0]))
                if tail == ['frame']:
                    check(set(query) == {'task', 'turn'} and all(len(v) == 1 for v in query.values()), 'invalid_query')
                    return store.frame(token, instance_id, task=int(query['task'][0]), turn=int(query['turn'][0]))
                check(not query, 'invalid_query')
                if tail == ['export']:
                    self._public_context = None
                    return store.export_instance(token, instance_id, admin=self._admin())
                if tail == ['answers']:
                    return store.answers(token, instance_id)
                if tail == ['kitchen-practice']:
                    return store.kitchen_practice_metadata(token, instance_id)
            else:
                check(not url.query, 'invalid_query')
                if tail == ['demonstration']:
                    _fields(payload)
                    return store.demonstration(token, instance_id)
                if tail == ['demonstration', 'complete']:
                    _fields(payload)
                    return store.acknowledge_demonstration(token, instance_id)
                if tail == ['tasks']:
                    _fields(payload)
                    return store.start_task(token, instance_id)
                if tail == ['actions']:
                    return store.action(token, instance_id, **_fields(payload,
                        ('command_id', 'expected_turn', 'human_action')))
                if tail == ['questions']:
                    return store.ask(token, instance_id, **_fields(payload, ('question',), ('turn', 'decision_turn', 'target_task', 'language', 'human_actions')))
            raise StudyError('not_found', 404)

        def _handle(self, post):
            try:
                self._public_context = None
                value = self._route(post=post)
                if self._public_context is not None:
                    value = store.participant_view(*self._public_context, value)
                self._reply(value)
            except StudyError as exc:
                self._reply(dict(error=exc.code), status=exc.status)
            except RuntimeUnavailable:
                self._reply(dict(error='runtime_unavailable', pilot_opened=False), status=503)
            except (ValueError, TypeError):
                self._reply(dict(error='invalid_request'), status=400)
            except (KeyError, FileNotFoundError, sqlite3.Error):
                self._reply(dict(error='bound_asset_or_state_unavailable', pilot_opened=False), status=503)

        def do_GET(self):
            self._handle(False)

        def do_POST(self):
            self._handle(True)

    server = CoreHTTPServer((host, port), Handler)
    server.store, server._close_store = store, close_store
    return server
