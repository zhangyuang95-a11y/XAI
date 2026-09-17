"""One Render entry point for the existing Warehouse and Cooperative Pong studies.

The deployed Warehouse implementation stays intact and is mounted at
``/warehouse/``. Cooperative Pong is mounted independently at ``/pong/``.
The root page only lets a participant choose which study to open.
"""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from domains.pong.web.server import CONFIG_PATH as PONG_CONFIG_PATH
from domains.pong.web.server import COOKIE_NAME as PONG_COOKIE
from domains.pong.web.server import PongApplication
from ui import development_preview_server as warehouse


ROOT = Path(__file__).resolve().parents[1]
HUB_ASSETS = ROOT / "ui" / "domain_hub"
PONG_ASSETS = ROOT / "domains" / "pong" / "web"
MAX_PONG_BODY = 100_000


def _hub_asset(name: str) -> bytes:
    return (HUB_ASSETS / name).read_bytes()


def _pong_asset(name: str) -> bytes:
    return (PONG_ASSETS / name).read_bytes()


def _mounted_warehouse_index(asset: bytes) -> bytes:
    return asset.replace(b'"/assets/', b'"/warehouse/assets/')


def _mounted_warehouse_script(asset: bytes) -> bytes:
    return asset.replace(b'"/api/', b'"/warehouse/api/')


class DomainHubHTTPServer(ThreadingHTTPServer):
    """Threaded server with one Warehouse session pool and one Pong app."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        warehouse_sessions: Any,
        pong: PongApplication,
        public_origin: str,
    ) -> None:
        self.warehouse_sessions = warehouse_sessions
        self.pong = pong
        self.public_origin = public_origin.rstrip("/")
        super().__init__(address, domain_hub_handler(warehouse_sessions, pong, self.public_origin))


def domain_hub_handler(warehouse_sessions: Any, pong: PongApplication, public_origin: str):
    """Create the router; dependencies stay injectable for focused tests."""
    expected_origin = public_origin.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def _headers(
            self,
            status: int,
            length: int,
            content_type: str,
            cookies: tuple[str, ...] = (),
        ) -> None:
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

        def _reply(
            self,
            status: int,
            payload: Any,
            *,
            content_type: str = "application/json; charset=utf-8",
            cookies: tuple[str, ...] = (),
        ) -> None:
            body = (
                payload
                if isinstance(payload, bytes)
                else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            self._headers(status, len(body), content_type, cookies)
            self.wfile.write(body)

        def _cookies(self) -> SimpleCookie:
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
            except Exception:
                pass
            return cookies

        def _pong_session(self) -> str | None:
            cookies = self._cookies()
            return cookies[PONG_COOKIE].value if PONG_COOKIE in cookies else None

        def _pong_cookie(self, session_id: str) -> str:
            secure = "; Secure" if expected_origin.startswith("https://") else ""
            return f"{PONG_COOKIE}={session_id}; Path=/pong; HttpOnly{secure}; SameSite=Lax"

        def _warehouse_state(self) -> Any:
            # Warehouse already keys its preview state using this client page id.
            return warehouse_sessions.state_for(self.headers.get("X-Warehouse-Page"))

        def _warehouse_asset(self, suffix: str) -> tuple[bytes, str] | None:
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/assets/favicon.svg": ("favicon.svg", "image/svg+xml"),
            }
            item = assets.get(suffix)
            if item is None:
                return None
            name, content_type = item
            body = (warehouse.WEB / name).read_bytes()
            if name == "index.html":
                body = _mounted_warehouse_index(body)
            elif name == "app.js":
                body = _mounted_warehouse_script(body)
            return body, content_type

        def _warehouse_view(self, suffix: str) -> None:
            asset = self._warehouse_asset(suffix)
            if asset is not None:
                body, content_type = asset
                self._reply(HTTPStatus.OK, body, content_type=content_type)
                return
            state = self._warehouse_state()
            with state.lock:
                if suffix == "/api/view":
                    self._reply(HTTPStatus.OK, state.view())
                    return
                if suffix == "/api/study/reference-trajectory":
                    self._reply(HTTPStatus.OK, state.reference_trajectory())
                    return
                if suffix == "/api/fixture-metrics":
                    self._reply(
                        HTTPStatus.OK,
                        {
                            "command_requests": state.command_requests,
                            "reference_requests": state.reference_requests,
                        },
                    )
                    return
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def _warehouse_command(self, suffix: str) -> None:
            if suffix != "/api/study/command":
                self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= warehouse.MAX_BODY:
                    raise ValueError("invalid_body_size")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, Mapping):
                    raise ValueError("request_body_must_be_object")
                state = self._warehouse_state()
                with state.lock:
                    self._reply(HTTPStatus.OK, state.command(payload))
            except (ValueError, KeyError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def _pong_view(self, suffix: str) -> None:
            if suffix == "/api/view":
                self._reply(HTTPStatus.OK, pong.view(self._pong_session()))
                return
            if suffix == "/api/review":
                value = parse_qs(urlparse(self.path).query).get("frame", [None])[0]
                try:
                    self._reply(HTTPStatus.OK, pong.review(self._pong_session(), None if value is None else int(value)))
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
                "/api/start": "start",
                "/api/input": "input",
                "/api/pause": "pause",
                "/api/tick": "tick",
                "/api/step": "tick",
                "/api/ask": "ask",
                "/api/task2": "task2",
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

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/health", "/api/health"):
                self._reply(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "policylens-domain-hub",
                        "warehouse_service": "development-preview",
                        "domains": {"warehouse": "/warehouse/", "pong": "/pong/"},
                        "pong_version": pong.view(None)["version"],
                    },
                )
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
            if path == "/pong" or path.startswith("/pong/"):
                self._pong_view(path[len("/pong"):] or "/")
                return
            # Keep previously bookmarked Warehouse API and static links valid.
            if path in ("/api/view", "/api/study/reference-trajectory", "/api/fixture-metrics") or path.startswith("/assets/"):
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
            if path == "/api/study/command":
                self._warehouse_command(path)
                return
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument(
        "--public-origin",
        default=os.environ.get("WAREHOUSE_PUBLIC_ORIGIN", "https://policylens-warehouse-study.onrender.com"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    if not PONG_CONFIG_PATH.is_file():
        raise SystemExit("Pong configuration is missing")
    sessions = warehouse._require_preview_sessions()
    server = DomainHubHTTPServer(
        (args.host, args.port),
        warehouse_sessions=sessions,
        pong=PongApplication(),
        public_origin=args.public_origin,
    )
    print(f"PolicyLens domain hub: {args.public_origin}", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
