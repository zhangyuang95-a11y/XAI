"""The v6 Prolific block includes the raw-model C arm in every domain."""
from types import SimpleNamespace
from dataclasses import replace

from study_v3 import prolific
from study_v3.config import Settings
from study_v3.store import Store


def test_default_prolific_cohort_has_nine_balanced_cells():
    settings = Settings(database=":memory:", prolific_study_id="a" * 24,
                        prolific_completion_code="TESTONLY",
                        prolific_launch_confirmed=True,
                        prolific_places=18)
    _, targets = prolific.cohort(None, settings, settings.prolific_study_id)
    assert len(prolific.CELLS) == 9
    assert set(targets) == {(domain, group)
                            for domain in ("warehouse", "pong", "kitchen")
                            for group in ("A", "B", "C")}
    assert set(targets.values()) == {2}
    assert prolific.ready(SimpleNamespace(settings=settings, ready=True))
    assert not prolific.ready(SimpleNamespace(
        settings=replace(settings, prolific_places=12),
        ready=True))


def test_registered_cohort_accepts_c_and_preserves_its_quota(tmp_path):
    settings = Settings(database=str(tmp_path / "study.sqlite3"),
                        prolific_study_id="a" * 24,
                        prolific_completion_code="TESTONLY")
    store = Store(settings)
    try:
        quota = {"study_id": "b" * 24, "completion_code": "NEWC01",
                 "quotas": [{"domain": "kitchen", "group": "C", "places": 3}]}
        assert prolific.register_cohort(store, quota)["places"] == 3
        with store.db.transaction(read_only=True) as db:
            code, targets = prolific.cohort(db, settings, quota["study_id"])
        assert code == "NEWC01"
        assert targets == {("kitchen", "C"): 3}
    finally:
        store.db.close()
