"""Verify and restore a trusted kitchen archive into a separate EMPTY database.

Set KITCHEN_RESTORE_DATABASE_URL to a dedicated recovery database or empty Neon
branch. This utility refuses the source database and any non-empty target; it
never drops tables. Credentials are never printed or passed as process arguments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from postgres_backup import connection, executable, file_hash, fingerprint, run_client, table_counts


def restore(archive, *, check_only=False):
    import psycopg
    archive = Path(archive).expanduser().resolve()
    manifest_path = archive.with_suffix(archive.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cooperative_kitchen_pg_backup_v1" or manifest.get("archive") != archive.name:
        raise ValueError("Archive and backup manifest do not match.")
    if archive.stat().st_size != manifest.get("bytes") or file_hash(archive) != manifest.get("sha256"):
        raise ValueError("Backup checksum verification failed. No database was changed.")
    url, info, env = connection(os.environ.get("KITCHEN_RESTORE_DATABASE_URL"))
    if fingerprint(info) == manifest.get("source_fingerprint"):
        raise ValueError("The target is the original source database. Use a separate, empty recovery database.")
    with psycopg.connect(url) as conn:
        if table_counts(conn):
            raise ValueError("The recovery target contains tables. Use a new, empty database; this script never deletes existing data.")
    tool = executable("pg_restore")
    run_client([tool, "--list", str(archive)], env)
    report = {"schema": "cooperative_kitchen_pg_recovery_v1", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "backup_sha256": manifest["sha256"], "target_fingerprint": fingerprint(info), "archive_verified": True,
              "empty_target_verified": True, "restored": False, "row_counts_verified": False}
    if check_only:
        return report
    # --single-transaction rolls back the complete restore on an SQL error.
    run_client([tool, "--exit-on-error", "--single-transaction", "--no-owner", "--no-acl", "--dbname", info["dbname"], str(archive)], env)
    with psycopg.connect(url) as conn:
        counts = table_counts(conn)
    report.update(restored=True, table_counts=counts, row_counts_verified=counts == manifest.get("table_counts"))
    if not report["row_counts_verified"]:
        raise RuntimeError("Restore completed, but table row counts differ from the consistent backup snapshot. Keep the recovery database isolated for diagnosis.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Trusted .dump file with its adjacent .manifest.json.")
    parser.add_argument("--check-only", action="store_true", help="Check the checksum, archive directory and empty target without restoring.")
    parser.add_argument("--report", help="Write recovery report to a new local JSON file.")
    args = parser.parse_args()
    if args.report and Path(args.report).exists(): parser.error("Report already exists; choose a new filename.")
    try:
        result = restore(args.archive, check_only=args.check_only)
        if args.report:
            fd = os.open(Path(args.report), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as out: json.dump(result, out, indent=2); out.write("\n")
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    except Exception:
        parser.exit(1, "Recovery failed; check archive integrity, database privileges and client/server versions. No connection credentials were logged.\n")


if __name__ == "__main__": main()
