"""Read-only Neon archive verification against a newly created local database.

The source is loaded from a private environment file. Only this script's own
temporary local database is dropped. Source rows, credentials and session tokens
are never printed. The existing backup/restore utilities perform the archive IO.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid

import psycopg
from psycopg import sql

from postgres_backup import backup, connection, fingerprint, table_counts
import postgres_backup
from postgres_restore import restore
from remote_acceptance import private_environment


def content_hashes(conn, counts):
    hashes = {}
    for qualified in counts:
        schema, table = qualified.split(".", 1)
        query = sql.SQL("SELECT row_to_json(t)::text FROM {}.{} AS t").format(sql.Identifier(schema), sql.Identifier(table))
        rows = [json.dumps(json.loads(row[0]), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in conn.execute(query)]
        hashes[qualified] = hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()
    return hashes


def verified_backup(env_file, private_dir, report_path):
    values = private_environment(env_file)
    source = values.get("KITCHEN_DIRECT_DATABASE_URL")
    if not source:
        raise ValueError("A direct database URL is required")
    source_url, info, _ = connection(source)
    if not info.get("host", "").endswith(".neon.tech"):
        raise ValueError("Expected the authorized Neon direct endpoint")
    if info.get("sslmode") not in {"require", "verify-ca", "verify-full"}:
        raise ValueError("A TLS-requiring source configuration is required")
    report_path, private_dir = Path(report_path), Path(private_dir)
    if report_path.exists():
        raise ValueError("Choose a new report filename")
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(private_dir.stat().st_mode) & 0o077:
        raise ValueError("Private backup directory must have owner-only permissions")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dbname = "kitchen_neon_restore_" + stamp.lower() + "_" + uuid.uuid4().hex[:8]
    archive = private_dir / (dbname + ".dump")
    local = "postgresql://zhangyuang@/postgres?host=/tmp/policylens-kitchen-pg&port=55432"
    target = local.replace("/postgres?", "/" + dbname + "?")
    report = {"schema": "cooperative_kitchen_neon_backup_restore_v1", "mode": "real_neon_to_local",
        "passed": False, "started_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": fingerprint(info), "source_operations": "Read-only exported snapshot and pg_dump; no cloud writes or deletion.",
        "destination": "New isolated local Unix-socket PostgreSQL database", "source_sslmode": info["sslmode"],
        "archive_at_rest": "Unencrypted PostgreSQL custom archive with owner-only filesystem permissions; disk encryption was not independently verified.",
        "cloud_storage_encryption": "Not independently verified in this test.",
        "local_transport": "Unix-domain socket; no network TLS connection is involved.",
        "archive": str(archive.resolve()), "checks": {}, "local_temporary_database_removed": False}
    old_table_counts = postgres_backup.table_counts
    old_run_client = postgres_backup.run_client
    source_connection = []
    original_direct = os.environ.get("KITCHEN_DIRECT_DATABASE_URL")
    original_target = os.environ.get("KITCHEN_RESTORE_DATABASE_URL")
    created = False
    try:
        def snapshot_capture(conn):
            source_connection.append(conn)
            counts = old_table_counts(conn)
            report["source_table_counts"] = counts
            report["source_table_sha256"] = content_hashes(conn, counts)
            tls = conn.execute("SELECT ssl,version,cipher,bits FROM pg_stat_ssl WHERE pid=pg_backend_pid()").fetchone()
            report["backend_pg_stat_ssl"] = dict(zip(("enabled", "version", "cipher", "bits"), tls)) if tls else None
            report["observed_source_connection_tls"] = {"enabled": bool(conn.pgconn.ssl_in_use), "observation": "Client libpq connection; Neon proxy may terminate TLS before the PostgreSQL backend."}
            if not conn.pgconn.ssl_in_use:
                raise RuntimeError("Source TLS was not observed")
            report["source_transaction_read_only"] = conn.execute("SHOW transaction_read_only").fetchone()[0] == "on"
            report["source_namespaces"] = dict(conn.execute("SELECT namespace,count(*) FROM kitchen_runs GROUP BY namespace ORDER BY namespace").fetchall())
            report["confirmed_joint_steps"] = conn.execute("SELECT count(*) FROM kitchen_events WHERE kind='joint_step'").fetchone()[0]
            return counts
        postgres_backup.table_counts = snapshot_capture
        def bounded_client(command, env):
            if Path(command[0]).name != "pg_dump" or "--version" in command:
                return old_run_client(command, env)
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=300)
                if result.returncode:
                    raise RuntimeError("pg_dump failed; client diagnostics were not printed")
                return result
            except BaseException:
                # A vanished network can otherwise leave the context manager
                # blocked indefinitely trying to roll back a read-only snapshot.
                for connection in source_connection:
                    connection.close()
                raise
        postgres_backup.run_client = bounded_client
        os.environ["KITCHEN_DIRECT_DATABASE_URL"] = source
        saved = backup(archive)
        postgres_backup.table_counts = old_table_counts
        report["archive_sha256"] = saved["sha256"]
        manifest = json.loads(archive.with_suffix(".dump.manifest.json").read_text())
        report["source_postgresql_version"] = manifest["postgresql_version"]
        report["pg_dump_version"] = manifest["pg_dump_version"]
        with psycopg.connect(local, autocommit=True) as control:
            control.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(dbname)))
            created = True
        os.environ["KITCHEN_RESTORE_DATABASE_URL"] = target
        report["restore"] = restore(archive)
        with psycopg.connect(target) as restored:
            restored.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            counts = table_counts(restored)
            hashes = content_hashes(restored, counts)
            report["restored_table_sha256"] = hashes
            report["target_postgresql_version"] = restored.execute("SHOW server_version").fetchone()[0]
            report["restored_confirmed_joint_steps"] = restored.execute("SELECT count(*) FROM kitchen_events WHERE kind='joint_step'").fetchone()[0]
        report["checks"] = {"same_snapshot_full_row_hashes": hashes == report["source_table_sha256"],
            "table_counts_match": counts == report["source_table_counts"],
            "confirmed_step_count_matches": report["confirmed_joint_steps"] == report["restored_confirmed_joint_steps"],
            "archive_and_manifest_owner_only": all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in (archive, archive.with_suffix(".dump.manifest.json"))),
            "source_read_only": report["source_transaction_read_only"],
            "source_connection_tls_observed": bool(report["observed_source_connection_tls"]["enabled"])}
        report["scope"] = ("All actual rows present in the Neon exported snapshot were restored and matched by whole-row hashes. This does not test remote HTTP acknowledgement/restart recovery."
            if report["confirmed_joint_steps"] else "Neon contains no confirmed gameplay steps yet. Verified the actual schema/existing rows, not recovery of participant progress or a remote HTTP acknowledgement.")
        report["passed"] = all(report["checks"].values()) and report["restore"]["row_counts_verified"]
    except (Exception, KeyboardInterrupt) as exc:
        report["error_type"] = type(exc).__name__
    finally:
        postgres_backup.table_counts = old_table_counts
        postgres_backup.run_client = old_run_client
        for source_conn in source_connection:
            source_conn.close()
        report["local_temporary_database_created"] = created
        if created:
            try:
                with psycopg.connect(local, autocommit=True) as control:
                    control.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname)))
                report["local_temporary_database_removed"] = True
            except Exception as exc:
                report["cleanup_error_type"] = type(exc).__name__
                report["passed"] = False
        for key, value in (("KITCHEN_DIRECT_DATABASE_URL", original_direct), ("KITCHEN_RESTORE_DATABASE_URL", original_target)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = verified_backup(args.env_file, args.private_dir, args.report)
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({key: report.get(key) for key in ("mode", "passed", "source_postgresql_version", "pg_dump_version", "source_table_counts", "confirmed_joint_steps", "observed_source_connection_tls", "checks", "local_temporary_database_removed", "error_type", "scope")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
