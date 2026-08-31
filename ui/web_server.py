"""Dependency-free HTTP server for the modern Warehouse XAI interface."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from typing import Any, Mapping
import webbrowser

from .web_runtime import WarehouseWebApplication
from .study_store import StudyConflict
from backend.artifacts import CollaborativeArtifactPaths
from env.warehouse.contracts import ARTIFACT_NAMESPACE


STATIC_ROOT = Path(__file__).resolve().parent / "web"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = CollaborativeArtifactPaths.under(
    PROJECT_ROOT,
    ARTIFACT_NAMESPACE,
)
DEFAULT_ARTIFACT_ROOT = DEFAULT_ARTIFACTS.root
DEFAULT_POLICY = (
    PROJECT_ROOT / "output" / "deployment" / "warehouse_mappo_v68_6x7.pt"
)
DEFAULT_PROGRAM = DEFAULT_ARTIFACTS.rcpd_program
SESSION_COOKIE = "warehouse_xai_session"
MAX_REQUEST_BODY_BYTES = 1_000_000
# Draining a small over-limit request before closing prevents TCP resets for
# ordinary clients, while the cap prevents a forged Content-Length from
# turning rejection into an unbounded read.
MAX_REJECT_DRAIN_BYTES = 1_048_576
API_ROUTES = {
    "/api/view": "view",
    "/api/study/command": "study_command",
    "/api/study/reference-trajectory": "reference_trajectory",
}
STATIC_ROUTES = {
    "/": STATIC_ROOT / "index.html",
    "/index.html": STATIC_ROOT / "index.html",
    "/assets/styles.css": STATIC_ROOT / "styles.css",
    "/assets/app.js": STATIC_ROOT / "app.js",
    "/assets/favicon.svg": STATIC_ROOT / "favicon.svg",
    "/favicon.ico": STATIC_ROOT / "favicon.svg",
}
INDEX_STYLESHEET_TAG = '<link rel="stylesheet" href="/assets/styles.css">'
INDEX_SCRIPT_TAG = '<script src="/assets/app.js" defer></script>'


def bundled_index_html(nonce: str) -> bytes:
    """Build one self-contained first response for unreliable edge tunnels.

    Cloudflare Quick Tunnels can occasionally deliver the HTML while a
    following CSS or JavaScript request fails.  Serving those two critical
    resources in the same response makes page rendering atomic: either the
    browser receives the usable application or it receives no page at all.
    """

    template = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    stylesheet = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    if INDEX_STYLESHEET_TAG not in template or INDEX_SCRIPT_TAG not in template:
        raise RuntimeError("The index asset placeholders are missing.")
    if "</style" in stylesheet.lower() or "</script" in script.lower():
        raise RuntimeError("A bundled asset contains an unsafe closing tag.")
    document = template.replace(
        INDEX_STYLESHEET_TAG,
        f'<style data-bundled-asset="styles.css">\n{stylesheet}\n</style>',
        1,
    ).replace(
        INDEX_SCRIPT_TAG,
        f'<script nonce="{nonce}" data-bundled-asset="app.js">\n{script}\n</script>',
        1,
    )
    return document.encode("utf-8")


class WarehouseHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        application: WarehouseWebApplication,
    ) -> None:
        super().__init__(address, WarehouseRequestHandler)
        self.application = application


class WarehouseRequestHandler(BaseHTTPRequestHandler):
    """Small JSON/static handler with one opaque session cookie."""

    server: WarehouseHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # The stdlib handler logs synchronously on every request.  In a hosted
        # process stdout may be a bounded pipe; once that pipe fills, otherwise
        # healthy request threads block before sending CSS/JS responses.  Study
        # commands already have durable transactional audit events, while the
        # reverse proxy owns operational access logs, so per-request stdout
        # logging is deliberately disabled here.
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {"status": "ok", "service": "warehouse-xai"},
            )
            return
        if path == "/api/view":
            self._dispatch("view", {})
            return
        if path == "/api/study/reference-trajectory":
            self._dispatch("reference_trajectory", {})
            return
        file_path = STATIC_ROUTES.get(path)
        if file_path is None or not file_path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        nonce = secrets.token_urlsafe(24)
        is_index = path in {"/", "/index.html"}
        content = bundled_index_html(nonce) if is_index else file_path.read_bytes()
        content_type = "text/html" if is_index else mimetypes.guess_type(file_path.name)[0]
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            (content_type or "application/octet-stream")
            + ("; charset=utf-8" if file_path.suffix in {".html", ".css", ".js"} else ""),
        )
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            f"script-src 'self' 'nonce-{nonce}'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = self.path.split("?", 1)[0]
        operation = API_ROUTES.get(path)
        if operation is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must not be negative.")
            if length > MAX_REQUEST_BODY_BYTES:
                if length <= MAX_REJECT_DRAIN_BYTES:
                    self.rfile.read(length)
                self.close_connection = True
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {
                        "error": "Request body is too large.",
                        "error_type": "ValueError",
                        "code": "request_too_large",
                    },
                )
                return
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("JSON request body must be an object.")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        self._dispatch(operation, payload)

    def _dispatch(self, operation: str, payload: Mapping[str, Any]) -> None:
        page_id = self.headers.get("X-Warehouse-Page", "").strip()
        routed_payload = dict(payload)
        if page_id:
            routed_payload["__page_id"] = page_id
        try:
            session_id, result = self.server.application.dispatch(
                self._session_id(),
                operation,
                routed_payload,
            )
        except StudyConflict as exc:
            self._json(
                HTTPStatus.CONFLICT,
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "code": exc.code,
                    "view": exc.current,
                },
            )
            return
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        except Exception as exc:  # Operational failures are recoverable and structured.
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "code": "service_temporarily_unavailable",
                },
            )
            return
        self._json(
            HTTPStatus.OK,
            result,
            session_id=session_id,
        )

    def _session_id(self) -> str | None:
        cookie_header = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        value = cookie.get(SESSION_COOKIE)
        return value.value if value is not None else None

    def _json(
        self,
        status: HTTPStatus,
        payload: Mapping[str, Any],
        *,
        session_id: str | None = None,
    ) -> None:
        content = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        if session_id is not None:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the browser-based Warehouse XAI interface."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_POLICY))
    parser.add_argument("--program", default=str(DEFAULT_PROGRAM))
    parser.add_argument("--transformer-model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--study-steps", type=int, default=120)
    parser.add_argument(
        "--parallel-seed-library",
        default=None,
        help=(
            "Calibrated seed-pair library. By default, use "
            "parallel_seed_pairs.json next to --checkpoint."
        ),
    )
    parser.add_argument(
        "--reference-trajectory",
        default=None,
        help=(
            "Frozen reference_trajectory.json manifest. By default, use the "
            "file next to --checkpoint."
        ),
    )
    parser.add_argument(
        "--study-randomization-seed",
        type=int,
        default=41000,
    )
    parser.add_argument(
        "--study-phase",
        choices=("pilot", "confirmatory"),
        default="pilot",
    )
    parser.add_argument(
        "--test-condition-selector",
        action="store_true",
        help=(
            "Enable the development-only condition selector and use the "
            "development namespace by default. Never enable for formal data collection."
        ),
    )
    parser.add_argument(
        "--study-db",
        default=None,
        help=(
            "Study database. By default, use collaborative_study.sqlite3 "
            "next to --checkpoint."
        ),
    )
    parser.add_argument(
        "--study-namespace",
        default=None,
        help="Isolation namespace (defaults to the study phase).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--maximum-sessions", type=int, default=16)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    application = WarehouseWebApplication(
        checkpoint=args.checkpoint,
        transformer_model=args.transformer_model,
        program_path=args.program,
        device=args.device,
        local_files_only=args.local_files_only,
        seed=args.seed,
        study_steps=args.study_steps,
        study_db=args.study_db,
        study_namespace=args.study_namespace,
        study_randomization_seed=args.study_randomization_seed,
        study_phase=args.study_phase,
        test_condition_selector=args.test_condition_selector,
        maximum_sessions=args.maximum_sessions,
        parallel_seed_library=args.parallel_seed_library,
        reference_trajectory=args.reference_trajectory,
    )
    server = WarehouseHTTPServer((args.host, args.port), application)
    visible_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{visible_host}:{server.server_port}/"
    print(f"[Warehouse Web UI] {url}")
    print("Press Ctrl+C to stop the server.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[Warehouse Web UI] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
