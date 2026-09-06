"""Local researcher CLI. Reads KITCHEN_URL and KITCHEN_ADMIN_KEY from the environment.

Participants enter their own pseudonymous user ID. Invitation issuance is
retired. Exports and technical-retry receipts use new owner-only local files.
This tool never registers participants or sends messages to them.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import uuid
from urllib.parse import urlsplit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, help='Optional owner-only literal dotenv file; never executed or printed.')
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    retry = commands.add_parser('retry', help='Create a recorded technical retry for an existing run.')
    retry.add_argument('--run-id', required=True)
    retry.add_argument('--reason', required=True)
    retry.add_argument('--operation-id', default=None, help='Reuse this ID and the identical run/reason after an uncertain response.')
    retry.add_argument('--output', required=True)
    export = commands.add_parser("export")
    export.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from remote_acceptance import private_environment, AcceptanceError
    try: config = private_environment(args.env_file)
    except (OSError, ValueError, AcceptanceError): parser.error('Private configuration is missing or unsafe; no credentials were logged.')
    base = config.get("KITCHEN_URL", "").rstrip("/")
    admin_key = config.get("KITCHEN_ADMIN_KEY", "")
    parsed = urlsplit(base)
    if not admin_key or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
        parser.error("Set KITCHEN_URL and KITCHEN_ADMIN_KEY; do not put credentials in the URL.")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
        parser.error("Remote researcher requests require HTTPS.")
    destination = Path(args.output).expanduser().resolve() if hasattr(args, "output") else None
    if destination and destination.exists(): parser.error("Output already exists; choose a new filename.")
    import httpx
    try:
        with httpx.Client(timeout=30, follow_redirects=False, headers={"X-Kitchen-Admin-Key": admin_key}) as client:
            if args.command == "retry":
                if not args.reason.strip() or len(args.reason.strip()) > 1000: parser.error('A 1–1000 character audit reason is required.')
                operation = args.operation_id or str(uuid.uuid4())
                print(json.dumps({"operation_id": operation}))
                response = client.post(base + '/api/admin/retry', json={'operation_id': operation,
                    'run_id': args.run_id, 'reason': args.reason})
            elif args.command == "export": response = client.get(base + "/api/admin/export", params={"format": args.format})
            else: response = client.get(base + "/api/admin/status")
        if response.status_code != 200:
            parser.exit(1, f"Researcher request failed (HTTP {response.status_code}). No credentials were logged.\n")
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as out: out.write(response.content)
            print(json.dumps({"saved": str(destination), "bytes": destination.stat().st_size}))
        else: print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except httpx.HTTPError:
        parser.exit(1, 'Network request is unconfirmed. Repeat technical retries with the same printed operation ID, run ID and reason.\n')
    except OSError:
        parser.exit(1, 'The output file could not be saved. Repeat technical retries with the same printed operation ID, run ID and reason.\n')


if __name__ == "__main__": main()
