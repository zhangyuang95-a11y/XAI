"""One Render entry point for the Warehouse and Cooperative Pong studies.

The Warehouse release stays hash-bound to its existing portable package.  This
module only mounts it below ``/warehouse/`` and mounts Pong below ``/pong/``;
it deliberately does not change the Warehouse release sources that the package
verifies at startup.
"""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from domains.pong.web.server import CONFIG_PATH as PONG_CONFIG_PATH
from domains.pong.web.server import COOKIE_NAME as PONG_COOKIE
from domains.pong.web.server import PongApplication
from ui import warehouse_alignment_r42_server as warehouse


ROOT = Path(__file__).resolve().parents[1]
HUB_ASSETS = ROOT / "ui" / "domain_hub"
PONG_ASSETS = ROOT / "domains" / "pong" / "web"
MAX_PONG_BODY = 100_000


def _hub_asset(name: str) -> bytes:
    return (HUB_ASSETS / name).read_bytes()


def _pong_asset(name: str) -> bytes:
    return (PONG_ASSETS / name).read_bytes()


def _mounted_warehouse_index(asset: bytes) -> bytes:
    """Keep the existing Warehouse page intact below its explicit path."""
    return asset.replace(b'"/assets/', b'"/warehouse/assets/')


def _mounted_warehouse_script(asset: bytes) -> bytes:
    # The Warehouse client uses three root-relative API endpoints.  Its
    # controller, UI and artifact bindings are otherwise left unchanged.
    return asset.replace(b'"/api/', b'"/warehouse/api/')


class DomainHubHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, store: Any,
                 pong: PongApplication, public_origin: str) -> None:
        self.store = store
        self.pong = pong
        self.public_origin = public_origin.rstrip("/")
        super().__init__(address, domain_hub_handler(store, pong, self.public_origin))


def domain_hub_handler(store: Any, pong: PongApplication, public_origin: str):
    """Build the one router used by production and by the focused HTTP tests."""
    expected_origin = public_origin.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def _headers(self, status: int, length: int, content_type: str,
                     cookies: tuple[str, ...] = ()) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            for cookie in cookies:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()

        def _reply(self, status: int, payload: Any, *,
                   content_type: str = "application/json; charset=utf-8",
                   cookies: tuple[str, ...] = ()) -> None:
            body = payload if isinstance(payload, bytes) else warehouse._canonical(payload).encode("utf-8")
            self._headers(status, len(body), content_type, cookies)
            self.wfile.write(body)

        def _cookies(self) -> SimpleCookie:
            result = SimpleCookie()
            try:
                result.load(self.headers.get("Cookie", ""))
            except Exception:
                pass
            return result

        def _warehouse_session(self) -> str:
            cookie = self._cookies()
            return store.session(cookie[store.cookie_name].value if store.cookie_name in cookie else None)

        def _pong_session(self) -> str | None:
            cookie = self._cookies()
            return cookie[PONG_COOKIE].value if PONG_COOKIE in cookie else None

        def _warehouse_cookie(self, sid: str) -> str:
            secure = "; Secure" if expected_origin.startswith("https://") else ""
            return f"{store.cookie_name}={sid}; Path=/warehouse; HttpOnly{secure}; SameSite=Strict"

        def _pong_cookie(self, sid: str) -> str:
            secure = "; Secure" if expected_origin.startswith("https://") else ""
            return f"{PONG_COOKIE}={sid}; Path=/pong; HttpOnly{secure}; SameSite=Lax"

        def _warehouse_assets(self, suffix: str) -> tuple[bytes, str] | None:
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/assets/favicon.svg": ("favicon.svg", "image/svg+xml"),
            }
            item = assets.get(suffix)
            if item is None:
                return None
            name, content_type = item
            body = store.web_assets[name]
            if name == "index.html":
                body = _mounted_warehouse_index(body)
            elif name == "app.js":
                body = _mounted_warehouse_script(body)
            return body, content_type

        def _warehouse_view(self, suffix: str) -> None:
            asset = self._warehouse_assets(suffix)
            if asset is not None:
                body, content_type = asset
                self._reply(HTTPStatus.OK, body, content_type=content_type)
                return
            sid = self._warehouse_session()
            try:
                if suffix == "/api/view":
                    result = store.view(sid)
                elif suffix == "/api/history":
                    run_id = parse_qs(urlparse(self.path).query).get("run_id", [None])[0]
                    result = store.history(sid, run_id)
                else:
                    raise warehouse.CommandError("not_found", 404)
                self._reply(HTTPStatus.OK, result, cookies=(self._warehouse_cookie(sid),))
            except warehouse.CommandError as exc:
                self._reply(exc.status, {"error": exc.reason, "view": exc.view},
                            cookies=(self._warehouse_cookie(sid),))

        def _pong_view(self, suffix: str) -> None:
            if suffix == "/api/view":
                self._reply(HTTPStatus.OK, pong.view(self._pong_session()))
                return
            if suffix == "/api/review":
                value = parse_qs(urlparse(self.path).query).get("frame", [None])[0]
                try:
                    result = pong.review(self._pong_session(), None if value is None else int(value))
                    self._reply(HTTPStatus.OK, result)
                except (ValueError, TypeError, RuntimeError) as exc:
                    self._reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            item = assets.get(suffix)
            if item is None:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            name, content_type = item
            self._reply(HTTPStatus.OK, _pong_asset(name), content_type=content_type)

        def _pong_command(self, suffix: str) -> None:
            routes = {
                "/api/start": "start", "/api/input": "input", "/api/pause": "pause",
                "/api/tick": "tick", "/api/step": "tick", "/api/ask": "ask", "/api/task2": "task2",
            }
            operation = routes.get(suffix)
            if operation is None:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= MAX_PONG_BODY:
                    raise ValueError("request_too_large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request_body_must_be_object")
                session_id = self._pong_session()
                if operation == "start":
                    result = pong.start(payload)
                    self._reply(HTTPStatus.OK, result, cookies=(self._pong_cookie(result["session_id"]),))
                elif operation == "input":
                    self._reply(HTTPStatus.OK, pong.set_input(session_id, payload))
                elif operation == "pause":
                    self._reply(HTTPStatus.OK, pong.pause(session_id, payload))
                elif operation == "tick":
                    self._reply(HTTPStatus.OK, pong.tick(session_id, payload))
                elif operation == "ask":
                    self._reply(HTTPStatus.OK, pong.ask(session_id, payload))
                else:
                    self._reply(HTTPStatus.OK, pong.task2(session_id))
            except (ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def _warehouse_command(self, suffix: str) -> None:
            sid = self._warehouse_session()
            try:
                if suffix not in ("/api/study/command", "/api/command"):
                    raise warehouse.CommandError("not_found", 404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= warehouse.MAX_BODY:
                    raise warehouse.CommandError("invalid_body_size", 413)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise warehouse.CommandError("invalid_body")
                origin = self.headers.get("Origin")
                local = expected_origin.startswith(("http://127.0.0.1", "http://localhost"))
                if origin != expected_origin or not (origin.startswith("https://") or local):
                    raise warehouse.CommandError("https_origin_required", 403)
                self._reply(HTTPStatus.OK, store.command(sid, payload),
                            cookies=(self._warehouse_cookie(sid),))
            except warehouse.CommandError as exc:
                self._reply(exc.status, {"error": exc.reason, "view": exc.view},
                            cookies=(self._warehouse_cookie(sid),))
            except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                self._reply(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"},
                            cookies=(self._warehouse_cookie(sid),))

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/health", "/api/health"):
                self._reply(HTTPStatus.OK, {
                    "status": "ok", "service": warehouse.SERVICE_FAMILY,
                    "version": store.service_version,
                    "release_version": store.public_release_version,
                    "pilot_class": store.pilot_class,
                    "data_persistent": store.data_persistent,
                    "formal_ready": False, "formal_sample_eligible": False,
                    "domains": {"warehouse": "/warehouse/", "pong": "/pong/"},
                    "pong_version": pong.view(None)["version"],
                })
                return
            if path in ("/", "/index.html"):
                self._reply(HTTPStatus.OK, _hub_asset("index.html"), content_type="text/html; charset=utf-8")
                return
            if path == "/domain-hub.css":
                self._reply(HTTPStatus.OK, _hub_asset("styles.css"), content_type="text/css; charset=utf-8")
                return
            if path == "/warehouse" or path.startswith("/warehouse/"):
                self._warehouse_view(path[len("/warehouse"):] or "/")
                return
            if path == "/pong":
                self.send_response(HTTPStatus.PERMANENT_REDIRECT)
                self.send_header("Location", "/pong/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path.startswith("/pong/"):
                self._pong_view(path[len("/pong"):] or "/")
                return
            # Keep existing bookmarked Warehouse API URLs readable after the
            # hub becomes the root page.
            if path in ("/api/view", "/api/history") or path.startswith("/assets/"):
                self._warehouse_view(path)
                return
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/pong" or path.startswith("/pong/"):
                self._pong_command(path[len("/pong"):] or "/")
                return
            if path == "/warehouse" or path.startswith("/warehouse/"):
                self._warehouse_command(path[len("/warehouse"):] or "/")
                return
            if path in ("/api/study/command", "/api/command"):
                self._warehouse_command(path)
                return
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--base64", type=Path, default=os.environ.get("WAREHOUSE_RELEASE_BASE64"))
    parser.add_argument("--expected-package-sha256", default=os.environ.get("WAREHOUSE_RELEASE_PACKAGE_SHA256"))
    parser.add_argument("--expected-manifest-sha256", default=os.environ.get("WAREHOUSE_RELEASE_MANIFEST_SHA256"))
    parser.add_argument("--release-module", choices=warehouse.RELEASE_MODULES,
                        default=os.environ.get("WAREHOUSE_RELEASE_MODULE", warehouse.R42_RELEASE_MODULE))
    parser.add_argument("--database", type=Path, default=warehouse.DEFAULT_DATABASE)
    parser.add_argument("--storage-mode", choices=("auto", "persistent", "ephemeral"),
                        default=os.environ.get("WAREHOUSE_STORAGE_MODE", "auto"))
    parser.add_argument("--allow-internal-manual-groups", action="store_true")
    parser.add_argument("--public-origin", default=os.environ.get("WAREHOUSE_PUBLIC_ORIGIN", warehouse.DEFAULT_ORIGIN))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", warehouse.DEFAULT_PORT)))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.expected_package_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.expected_package_sha256):
        raise SystemExit("expected package SHA256 is required")
    if not args.expected_manifest_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.expected_manifest_sha256):
        raise SystemExit("expected manifest SHA256 is required")
    if bool(args.package) == bool(args.base64):
        raise SystemExit("provide exactly one of --package or --base64")
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    if not args.public_origin.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise SystemExit("public origin must use HTTPS (except loopback development)")
    if not PONG_CONFIG_PATH.is_file():
        raise SystemExit("Pong configuration is missing")
    context = warehouse.load_online_context(
        expected_package_sha256=args.expected_package_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        package_path=args.package, base64_path=args.base64,
        release_module=args.release_module,
    )
    store = warehouse.OnlineAlignmentStudyStore(
        context, database=args.database, storage_mode=args.storage_mode,
        allow_internal_manual_groups=args.allow_internal_manual_groups or None,
    )
    server = DomainHubHTTPServer((args.host, args.port), store=store,
                                 pong=PongApplication(), public_origin=args.public_origin)
    handlers: dict[int, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop)
        print(warehouse._canonical(warehouse._startup_identity(store, base64_path=args.base64)), flush=True)
        print(f"PolicyLens domain hub: {args.public_origin}", flush=True)
        server.serve_forever(poll_interval=.25)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        store.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
