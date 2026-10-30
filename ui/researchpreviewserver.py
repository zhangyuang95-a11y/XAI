"""One isolated deployment for the manual playground and researcher preview.

The original handlers run on private loopback ports. The public router does
not share their stores, authorizations or handler methods, and never opens the
old participant database. No automatic retry is performed for a mutation.
"""
import os

# Must precede all numerical-runtime imports, including transitive imports.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from contextlib import contextmanager
import gzip
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
from time import perf_counter
from urllib.parse import urlsplit

VERSION = 'research-preview-composition.v2-public-play'
HOP_HEADERS = frozenset(('connection', 'keep-alive', 'proxy-authenticate',
    'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade'))
MAX_BODY = 65536
MAX_RESPONSE = 32 * 1024 * 1024


def deployment_identity(commit=None):
    return dict(deployment_runtime=VERSION,
        deployment_commit=commit or os.environ.get('RENDER_GIT_COMMIT', 'local'),
        researcher_preview_available=True, researcher_preview_path='/research/',
        public_play_path='/study/', public_play_requires_access_key=False,
        pilot_opened=False, recruitment_opened=False,
        researcher_storage='isolated_ephemeral_preview_sqlite',
        database_is_participant_database=False)


def private_admin(environment=None):
    """An explicit existing operator credential may authorize this preview."""
    env = os.environ if environment is None else environment
    admin = env.get('POLICYLENS_STUDY_PREVIEW_ADMIN') or env.get('POLICYLENS_ADMIN_TOKEN', '')
    if not isinstance(admin, str) or len(admin) < 32:
        raise ValueError('private_researcher_preview_admin_required')
    return admin


def _gzip_allowed(header):
    for item in header.split(','):
        parts = [v.strip() for v in item.split(';')]
        if parts[0].lower() != 'gzip':
            continue
        try:
            q = next((float(v[2:]) for v in parts[1:] if v.startswith('q=')), 1.)
        except ValueError:
            return False
        return q > 0
    return False


def make_server(manual_server, study_server, *, host='127.0.0.1', port=0, commit=None):
    """Route to private, already-constructed backends without mutating them."""
    identity = deployment_identity(commit)
    addresses = {'manual': manual_server.server_address, 'study': study_server.server_address}
    if any(address[0] not in ('127.0.0.1', '::1') for address in addresses.values()):
        raise ValueError('private_loopback_backends_required')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(180)

        def log_message(self, *_):
            pass

        def reject(self, status, code):
            body = json.dumps(dict(error=code, pilot_opened=False), separators=(',', ':')).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            self.wfile.write(body)

        def dispatch(self):
            began = perf_counter()
            response_started = False
            target = urlsplit(self.path)
            if target.scheme or target.netloc or not self.path.startswith('/') or self.path.startswith('//'):
                return self.reject(400, 'origin_form_target_required')
            if len(self.headers.get_all('Host', [])) != 1:
                return self.reject(400, 'one_host_required')
            if self.headers.get('Transfer-Encoding') is not None:
                return self.reject(400, 'transfer_encoding_not_supported')
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) > 1 or lengths and (not lengths[0].isdigit() or int(lengths[0]) > MAX_BODY):
                return self.reject(413, 'invalid_body_size')
            length = int(lengths[0]) if lengths else 0
            if self.command == 'GET' and length:
                return self.reject(400, 'get_body_not_supported')
            body = self.rfile.read(length) if length else None
            if body is not None and len(body) != length:
                return self.reject(400, 'incomplete_body')
            # Public play never enrolls a research participant or bypasses its
            # protected routes. The public /study/ link is a game chooser.
            researcher_page = target.path == '/research/'
            backend = 'study' if researcher_page or target.path.startswith(('/study-api/', '/study-assets/')) else 'manual'
            forwarded_path = '/study/' if researcher_page else self.path
            # Forward the original Host: both original handlers enforce their
            # own same-origin rule, and their private ports must not replace it.
            connection_headers = {part.strip().lower() for part in self.headers.get('Connection', '').split(',')}
            headers = {key: value for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS | connection_headers}
            headers['Connection'] = 'close'
            connection = HTTPConnection(*addresses[backend], timeout=180)
            try:
                connection.request(self.command, forwarded_path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read(MAX_RESPONSE + 1)
                if len(raw) > MAX_RESPONSE:
                    return self.reject(502, 'preview_response_too_large')
                response_headers = dict(response.getheaders())
                encoding = response.getheader('Content-Encoding')
                # Keep the old release response shape, adding deployment facts.
                if self.command == 'GET' and target.path in ('/health', '/api/release') and response.status == 200:
                    plain = gzip.decompress(raw) if encoding == 'gzip' else raw
                    value = json.loads(plain)
                    value.update(identity)
                    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
                    response_headers = {k: v for k, v in response_headers.items() if k.lower() != 'content-encoding'}
                    encoding = None
                if not encoding and len(raw) >= 512 and _gzip_allowed(self.headers.get('Accept-Encoding', '')):
                    raw = gzip.compress(raw, compresslevel=1, mtime=0)
                    response_headers['Content-Encoding'] = 'gzip'
                self.send_response(response.status)
                response_started = True
                for key, value in response_headers.items():
                    if key.lower() not in HOP_HEADERS | {'content-length', 'server', 'date', 'vary', 'server-timing'}:
                        self.send_header(key, value)
                timing = response.getheader('Server-Timing')
                self.send_header('Server-Timing', (timing + ', ' if timing else '') +
                    f'preview_router;dur={(perf_counter() - began) * 1000:.3f}')
                self.send_header('Vary', 'Accept-Encoding')
                self.send_header('Content-Length', str(len(raw)))
                if response.status >= 400:
                    self.send_header('Connection', 'close')
                    self.close_connection = True
                self.end_headers()
                self.wfile.write(raw)
            except (OSError, HTTPException, ValueError, TypeError):
                # In particular: never silently replay an uncertain POST.
                if response_started:
                    self.close_connection = True
                    return
                return self.reject(503, 'preview_backend_unavailable')
            finally:
                connection.close()

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


@contextmanager
def running_backends(*servers):
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        yield
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8015')))
    parser.add_argument('--manual-database', default='/tmp/policylens-manual-model-preview-v1.sqlite3')
    parser.add_argument('--study-database', default='/tmp/policylens-research-preview-20261030-v1.sqlite3')
    args = parser.parse_args()
    admin = private_admin()
    # Explicit paths only; ignore all legacy participant/PostgreSQL variables.
    manual_path, study_path = (Path(value).expanduser().resolve() for value in (args.manual_database, args.study_database))
    if manual_path == study_path:
        raise ValueError('separate_preview_databases_required')
    from ui.model_preview_server import Preview, make_server as manual_server
    from study_vnext.server import compose_preview, make_server as study_server
    manual = Preview(str(manual_path))
    study = preview = public = None
    try:
        study, preview = compose_preview(str(study_path))
        manual_backend = manual_server(manual, '127.0.0.1', 0)
        study_backend = study_server(study, admin_token=admin, host='127.0.0.1', port=0)
        with running_backends(manual_backend, study_backend):
            public = make_server(manual_backend, study_backend, host=args.host, port=args.port)
            public.serve_forever()
    finally:
        if public is not None:
            public.server_close()
        if study is not None:
            study.close()
        if preview is not None:
            preview.close()
        manual.close()


if __name__ == '__main__':
    main()
