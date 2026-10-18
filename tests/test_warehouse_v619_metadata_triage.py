import json
from pathlib import Path
import pickle

from scripts import triage_warehouse_v619_seed_metadata as t
from scripts.audit_warehouse_v619_seed_ledger import extract


class SerializedMetadata:
    def __init__(self):
        self.seed = 41
        self.other = {"action_seeds": [42, 43]}


def test_inert_pickle_decodes_state_without_original_class():
    value = t.inert_pickle_fields(pickle.dumps(SerializedMetadata(), protocol=5))
    assert extract(value)[0] == {41, 42, 43}
    assert isinstance(value, dict)


class CallDuringNormalUnpickle:
    def __reduce__(self):
        return (Path, ("/never-opened-by-this-test",))


def test_pickle_reducer_global_becomes_an_inert_record():
    value = t.inert_pickle_fields(pickle.dumps(CallDuringNormalUnpickle(), protocol=5))
    assert value["constructor_args"] == ["/never-opened-by-this-test"]
    assert not isinstance(value, Path)


def test_complete_streaming_json_checks_seed_keys_intervals_and_candidate_values(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps({"schedule": {"scenario_seeds": [10, 11]},
        "reserved_intervals": [{"start": 100, "end_exclusive": 200}],
        "unrelated_numeric_candidate_is_conservatively_flagged": 1700000000}))
    targets = tmp_path / "targets.json"
    targets.write_text('{"1700000000":true}')
    seeds, intervals, numeric, count = t.stream_seeds(source, targets, tmp_path / "stderr")
    assert seeds == {10, 11}
    assert intervals == {(100, 200)}
    assert 1700000000 in numeric
    assert count >= 5
