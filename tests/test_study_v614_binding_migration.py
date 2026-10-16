"""The v6.14 physics binding is additive to existing study storage."""
from __future__ import annotations

import sqlite3

from study_v3.database import Database


def test_existing_policy_metadata_gets_nullable_physics_column(tmp_path):
    path = tmp_path / "historical.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE pl3_policy_binding_metadata (
            instance_id TEXT PRIMARY KEY, actor_model_version TEXT NOT NULL,
            feature_schema_sha256 TEXT NOT NULL, created DOUBLE PRECISION NOT NULL
        )""")
        connection.execute(
            "INSERT INTO pl3_policy_binding_metadata VALUES(?,?,?,?)",
            ("historical-instance", "warehouse-structured-v54", "a" * 64, 1.0))
    database = Database(str(path))
    try:
        with database.transaction(read_only=True) as db:
            row = db.one("SELECT * FROM pl3_policy_binding_metadata "
                         "WHERE instance_id=?", ("historical-instance",))
        assert row["actor_model_version"] == "warehouse-structured-v54"
        assert row["feature_schema_sha256"] == "a" * 64
        assert row["action_physics_config_sha256"] is None
    finally:
        database.close()
