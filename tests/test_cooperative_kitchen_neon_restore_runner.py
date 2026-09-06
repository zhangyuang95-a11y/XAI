"""Network-failure cleanup of the optional backup verification utility."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def test_dump_deadline_closes_snapshot_and_preserves_failure_without_cloud_io(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/cooperative_kitchen"))
    verify = importlib.import_module("verify_neon_restore")
    source = "postgresql://fixture:DO_NOT_LOG@ep-fixture.ap-southeast-1.aws.neon.tech/kitchen?sslmode=require"
    monkeypatch.setattr(verify, "private_environment", lambda path: {"KITCHEN_DIRECT_DATABASE_URL": source})
    class Rows(list):
        def fetchone(self):
            return self[0]
        def fetchall(self):
            return list(self)
    class Connection:
        pgconn = SimpleNamespace(ssl_in_use=True)
        closed = False
        def close(self):
            self.closed = True
        def execute(self, query):
            text = str(query)
            if "row_to_json" in text:
                return Rows([('{"namespace":"development","created":0}',), ('{"namespace":"pilot","created":0}',)])
            if "pg_stat_ssl" in text:
                return Rows([(False, None, None, None)])
            if "transaction_read_only" in text:
                return Rows([("on",)])
            if "GROUP BY namespace" in text:
                return Rows([])
            if "kind='joint_step'" in text:
                return Rows([(0,)])
            raise AssertionError("Unexpected fixture query")
    connection = Connection()
    monkeypatch.setattr(verify.postgres_backup, "table_counts", lambda conn: {"public.kitchen_namespace_locks": 2})
    old_runner = verify.postgres_backup.run_client
    def fake_backup(archive):
        verify.postgres_backup.table_counts(connection)
        return verify.postgres_backup.run_client(["/fixture/pg_dump", "--format=custom"], {})
    monkeypatch.setattr(verify, "backup", fake_backup)
    def expired(command, **kwargs):
        assert kwargs["timeout"] == 300
        raise subprocess.TimeoutExpired(command, 300)
    monkeypatch.setattr(verify.subprocess, "run", expired)
    report_path = tmp_path / "report.json"
    report = verify.verified_backup(tmp_path / "private.env", tmp_path / "private", report_path)
    assert not report["passed"] and report["error_type"] == "TimeoutExpired"
    assert report["local_temporary_database_created"] is False
    assert connection.closed
    assert report["source_transaction_read_only"]
    assert report["observed_source_connection_tls"]["enabled"] is True
    assert report["backend_pg_stat_ssl"]["enabled"] is False
    assert "DO_NOT_LOG" not in report_path.read_text()
    assert json.loads(report_path.read_text())["source_table_sha256"]
    assert verify.postgres_backup.run_client is old_runner
