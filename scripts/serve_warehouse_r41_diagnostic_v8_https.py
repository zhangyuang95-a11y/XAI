#!/usr/bin/env python3
"""Run the r4.1-diagnostic v8 service over loopback HTTPS for acceptance.

The production Render service terminates TLS outside Python.  Browser
acceptance still needs a same-origin HTTPS endpoint locally because the
authoritative server rejects every state-changing request from an HTTP
origin.  This wrapper loads the exact portable release in the same way as the
production entry point, then wraps only the loopback listening socket in TLS.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import ssl
import stat
import subprocess
from urllib.parse import urlsplit

from ui import warehouse_alignment_online_server as online


VERSION = "warehouse-r41-diagnostic-v8-loopback-https.v1"
RELEASE_MODULE = online.R41_DIAGNOSTIC_RELEASE_MODULE_V8
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _regular(path: str | Path, label: str) -> Path:
    requested = Path(path).expanduser().absolute()
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"{label} must be a canonical regular file")
    return requested.resolve()


def generate_certificate(directory: str | Path) -> tuple[Path, Path]:
    """Create a one-day localhost certificate without touching a keychain."""
    output = Path(directory).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    config = output / "openssl.cnf"
    cert = output / "localhost.crt"
    key = output / "localhost.key"
    config.write_text(
        """[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = localhost
[ext]
subjectAltName = @alt
basicConstraints = critical,CA:TRUE
keyUsage = critical,digitalSignature,keyEncipherment,keyCertSign
extendedKeyUsage = serverAuth
[alt]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
""",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-days", "1", "-keyout", str(key), "-out", str(cert),
             "-config", str(config), "-extensions", "ext"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("openssl could not create the loopback certificate") from error
    finally:
        config.unlink(missing_ok=True)
    os.chmod(key, 0o600)
    os.chmod(cert, 0o644)
    _regular(key, "TLS private key")
    _regular(cert, "TLS certificate")
    return cert, key


def https_server(store, *, host: str, port: int, public_origin: str,
                 certificate: str | Path, private_key: str | Path):
    parsed = urlsplit(public_origin.rstrip("/"))
    if (parsed.scheme != "https" or parsed.hostname not in
            {"127.0.0.1", "localhost", "::1"} or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.username
            or parsed.password):
        raise ValueError("public origin must be a credential-free loopback HTTPS origin")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("local HTTPS acceptance must bind only to loopback")
    cert = _regular(certificate, "TLS certificate")
    key = _regular(private_key, "TLS private key")
    if stat.S_IMODE(key.stat().st_mode) & 0o077:
        raise ValueError("TLS private key must not be group/world accessible")
    server = online.ThreadingHTTPServer(
        (host, port), online.handler_class(store, public_origin=public_origin))
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(certfile=cert, keyfile=key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    server.daemon_threads = False
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", type=Path)
    source.add_argument("--base64", type=Path)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--public-origin")
    args = parser.parse_args(argv)
    for value, label in ((args.expected_package_sha256, "package"),
                         (args.expected_manifest_sha256, "manifest")):
        if _HEX.fullmatch(value) is None:
            parser.error(f"expected {label} SHA-256 is required")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    origin = args.public_origin or f"https://{args.host}:{args.port}"
    context = online.load_online_context(
        expected_package_sha256=args.expected_package_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        package_path=args.package, base64_path=args.base64,
        release_module=RELEASE_MODULE,
    )
    store = None
    server = None
    handlers = {}
    try:
        store = online.OnlineAlignmentStudyStore(
            context, database=args.database, storage_mode="ephemeral")
        server = https_server(
            store, host=args.host, port=args.port, public_origin=origin,
            certificate=args.certificate, private_key=args.private_key,
        )
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: (_ for _ in ()).throw(
                KeyboardInterrupt()))
        identity = online._startup_identity(store, base64_path=args.base64)
        print(json.dumps({
            "version": VERSION, "status": "ready", "origin": origin,
            "certificate_sha256": sha256(
                _regular(args.certificate, "TLS certificate").read_bytes()).hexdigest(),
            "deployment_identity": identity,
        }, sort_keys=True, separators=(",", ":")), flush=True)
        server.serve_forever(poll_interval=0.25)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if server is not None:
            server.server_close()
        if store is not None:
            store.close()
        else:
            context.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
