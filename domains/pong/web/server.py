from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from ..config import PongConfig
from ..study import PongStudySession
from study_three_tasks import PATH as STUDY_PROTOCOL_PATH, task as study_task

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.parent / "configs" / "default.json"
COOKIE_NAME = "pong_session"


class PongApplication:
    """Small multi-session runner for local play and future route mounting."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, PongStudySession] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = PongConfig.from_json(CONFIG_PATH)
        session = PongStudySession(
            group=str(payload.get("group", "A")).upper(),
            participant_id=str(payload.get("participant_id", "local")),
            seed=int(study_task("pong", 1)["seed"]),
            condition_source=str(payload.get("assignment_source", "manual_self_select")),
            config=config,
        )
        session_id = uuid4().hex
        session.session_id = session_id
        with self._lock:
            self._sessions[session_id] = session
            self._ensure_runner(session_id)
        return {"session_id": session_id, **self.view(session_id)}

    def _ensure_runner(self, session_id: str) -> None:
        """Start exactly one active worker for the current task."""
        thread = self._threads.get(session_id)
        session = self._sessions[session_id]
        if thread is not None and thread.is_alive() and getattr(thread, "pong_task", None) == session.task:
            return
        thread = threading.Thread(target=self._run, args=(session_id, session.task), daemon=True,
                                  name=f"pong-{session_id[:8]}")
        thread.pong_task = session.task  # type: ignore[attr-defined]
        self._threads[session_id] = thread
        thread.start()

    def _session(self, session_id: str | None) -> PongStudySession:
        if not session_id or session_id not in self._sessions:
            raise ValueError("start the Pong study first")
        return self._sessions[session_id]

    def _run(self, session_id: str, task: int) -> None:
        next_tick = time.monotonic()
        while True:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None or session.task != task or session.environment.terminal:
                    return
                if not session.paused:
                    session.tick()
                interval = session.config.fixed_dt
            # Use a monotonic deadline instead of "work, then sleep(dt)".
            # The latter slows the simulation by its own computation time and
            # creates visible periodic stutter as the worker drifts.
            next_tick += interval
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # Do not run an unbounded catch-up burst: it would make a
                # delayed browser miss a series of visible contact events.
                next_tick = time.monotonic()

    def view(self, session_id: str | None) -> dict[str, Any]:
        with self._lock:
            if not session_id or session_id not in self._sessions:
                return {"started": False, "domain_id": "pong", "version": "pong-continuous-24x14-v7"}
            return {"started": True, **self._sessions[session_id].summary()}

    def set_input(self, session_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._session(session_id).set_input(str(payload.get("action", "stay")))

    def pause(self, session_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._session(session_id).set_paused(bool(payload.get("paused", True)))

    def tick(self, session_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._session(session_id).tick(payload.get("action"))

    def review(self, session_id: str | None, frame: int | None) -> dict[str, Any]:
        with self._lock:
            return self._session(session_id).review(frame)

    def ask(self, session_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._session(session_id).ask(
                str(payload.get("question", "")), payload.get("frame"), payload.get("ball_id"))

    def task2(self, session_id: str | None) -> dict[str, Any]:
        with self._lock:
            result = self._session(session_id).advance_task()
            assert session_id is not None
            self._ensure_runner(session_id)
            return result

    def task3(self, session_id: str | None) -> dict[str, Any]:
        with self._lock:
            result = self._session(session_id).advance_task()
            assert session_id is not None
            self._ensure_runner(session_id)
            return result


class PongHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], application: PongApplication):
        super().__init__(address, PongRequestHandler)
        self.application = application


class PongRequestHandler(BaseHTTPRequestHandler):
    server: PongHTTPServer

    def log_message(self, *_args: Any) -> None:
        return

    def _session_id(self) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        morsel = cookie.get(COOKIE_NAME)
        return morsel.value if morsel else None

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/pong":
            self.send_response(HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Location", "/pong/")
            self.end_headers()
            return
        if path == "/pong" or path.startswith("/pong/"):
            path = path[len("/pong"):] or "/"
        if path == "/api/view":
            self._json(HTTPStatus.OK, self.server.application.view(self._session_id()))
            return
        if path == "/api/review":
            value = parse_qs(urlparse(self.path).query).get("frame", [None])[0]
            try:
                frame = None if value is None else int(value)
                result = self.server.application.review(self._session_id(), frame)
            except (ValueError, TypeError, RuntimeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)
            return
        if path == "/study_protocol.json":
            content = STUDY_PROTOCOL_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/coordinated.js": ("coordinated.js", "text/javascript; charset=utf-8"),
            "/explain.js": ("explain.js", "text/javascript; charset=utf-8"),
            "/nn_model.json": ("nn_model.json", "application/json; charset=utf-8"),
            "/controller_config.json": ("controller_config.json", "application/json; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if path in assets:
            name, content_type = assets[path]
            content = (ROOT / name).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/pong" or path.startswith("/pong/"):
            path = path[len("/pong"):] or "/"
        routes = {
            "/api/start": "start", "/api/input": "input", "/api/pause": "pause",
            "/api/tick": "tick", "/api/step": "tick", "/api/ask": "ask", "/api/task2": "task2", "/api/task3": "task3",
        }
        operation = routes.get(path)
        if operation is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100_000:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            session_id = self._session_id()
            if operation == "start":
                result = self.server.application.start(body)
                self._json(HTTPStatus.OK, result, cookie=result["session_id"])
                return
            if operation == "input":
                result = self.server.application.set_input(session_id, body)
            elif operation == "pause":
                result = self.server.application.pause(session_id, body)
            elif operation == "tick":
                result = self.server.application.tick(session_id, body)
            elif operation == "ask":
                result = self.server.application.ask(session_id, body)
            elif operation == "task2":
                result = self.server.application.task2(session_id)
            else:
                result = self.server.application.task3(session_id)
            self._json(HTTPStatus.OK, result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _json(self, status: HTTPStatus, payload: dict[str, Any], cookie: str | None = None) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", f"{COOKIE_NAME}={cookie}; Path=/; SameSite=Lax")
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local continuous Cooperative Pong preview")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = PongHTTPServer((args.host, args.port), PongApplication())
    print(f"Pong preview: http://{args.host}:{args.port}/pong", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
