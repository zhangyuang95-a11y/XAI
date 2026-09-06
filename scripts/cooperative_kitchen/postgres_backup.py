"""Create a consistent PostgreSQL archive and a verifiable, credential-free manifest.

Credentials are read from KITCHEN_DIRECT_DATABASE_URL (preferred) or DATABASE_URL,
never command-line arguments. Run this from a trusted local computer, not the
ephemeral Render filesystem. Requires psycopg and PostgreSQL client utilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ENV_NAMES = {
    "host": "PGHOST", "hostaddr": "PGHOSTADDR", "port": "PGPORT", "dbname": "PGDATABASE",
    "user": "PGUSER", "password": "PGPASSWORD", "sslmode": "PGSSLMODE",
    "sslcert": "PGSSLCERT", "sslkey": "PGSSLKEY", "sslrootcert": "PGSSLROOTCERT",
    "channel_binding": "PGCHANNELBINDING", "connect_timeout": "PGCONNECT_TIMEOUT",
    "options": "PGOPTIONS", "application_name": "PGAPPNAME", "target_session_attrs": "PGTARGETSESSIONATTRS",
}


def connection(value):
    from psycopg.conninfo import conninfo_to_dict
    if not value:
        raise ValueError("Set a PostgreSQL connection URL in the required environment variable.")
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value[len("postgresql+psycopg://"):]
    if not value.startswith(("postgres://", "postgresql://")):
        raise ValueError("A PostgreSQL URL is required; SQLite files are not research backups.")
    try:
        info = conninfo_to_dict(value)
    except Exception as exc:
        raise ValueError("Invalid PostgreSQL connection URL; no credentials were logged.") from None
    if "-pooler" in info.get("host", ""):
        raise ValueError("Use the database's direct, non-pooler connection URL for backup and restore.")
    unsupported = set(info) - set(ENV_NAMES)
    if unsupported:
        raise ValueError("Unsupported connection options: " + ", ".join(sorted(unsupported)))
    if not info.get("dbname") or not info.get("host"):
        raise ValueError("The PostgreSQL URL must specify a host and database.")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    env.update({ENV_NAMES[key]: str(item) for key, item in info.items()})
    env.setdefault("PGCONNECT_TIMEOUT", "15")
    return value, info, env


def fingerprint(info):
    identity = {key: info.get(key, "") for key in ("host", "port", "dbname")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def table_counts(conn):
    from psycopg import sql
    tables = conn.execute("SELECT schemaname,tablename FROM pg_catalog.pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema') AND schemaname NOT LIKE 'pg_toast%' ORDER BY 1,2").fetchall()
    return {f"{schema}.{table}": conn.execute(sql.SQL("SELECT COUNT(*) FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(table))).fetchone()[0] for schema, table in tables}


def executable(name):
    found = shutil.which(name)
    if not found:
        raise ValueError(f"{name} is not installed or not on PATH. Install PostgreSQL client utilities matching the server major version.")
    return found


def run_client(command, env):
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        # Database client stderr may contain connection strings. Never relay it.
        raise RuntimeError(f"{Path(command[0]).name} exited with status {result.returncode}; check connectivity, permissions and PostgreSQL client/server versions. Credentials were not logged.")
    return result


def backup(destination):
    import psycopg
    url, info, env = connection(os.environ.get("KITCHEN_DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL"))
    tool = executable("pg_dump")
    destination = Path(destination).expanduser().resolve()
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ValueError("The output or manifest already exists. Choose a new backup filename.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".kitchen-backup-", dir=destination.parent)
    os.close(handle)
    try:
        with psycopg.connect(url) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            snapshot = conn.execute("SELECT pg_export_snapshot()").fetchone()[0]
            counts = table_counts(conn)
            server_version = conn.execute("SHOW server_version").fetchone()[0]
            if not any(name.startswith("public.kitchen_") for name in counts):
                raise ValueError("This database contains no kitchen research tables.")
            run_client([tool, "--format=custom", "--no-owner", "--no-acl", "--snapshot", snapshot, "--file", temporary], env)
        os.chmod(temporary, 0o600)
        # Hard-link publication refuses to overwrite a concurrently created file.
        os.link(temporary, destination)
        document = {"schema": "cooperative_kitchen_pg_backup_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "archive": destination.name, "sha256": file_hash(destination), "bytes": destination.stat().st_size,
                    "source_fingerprint": fingerprint(info), "postgresql_version": server_version,
                    "pg_dump_version": run_client([tool, "--version"], env).stdout.strip(), "table_counts": counts,
                    "scope": "Entire source database, including all kitchen namespaces. Not a participant-level export."}
        fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(document, target, ensure_ascii=False, indent=2); target.write("\n")
        return {"archive": str(destination), "manifest": str(manifest_path), "sha256": document["sha256"], "tables": len(counts)}
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="New local .dump archive filename; existing files are never overwritten.")
    args = parser.parse_args()
    try:
        print(json.dumps(backup(args.output), indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    except Exception:
        parser.exit(1, "Backup failed; check database connectivity and privileges. No connection credentials were logged.\n")


if __name__ == "__main__": main()
