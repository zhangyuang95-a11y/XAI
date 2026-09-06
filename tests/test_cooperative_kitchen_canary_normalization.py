"""Synthetic evidence consistency and write-recovery tests; no remote/API use."""
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cooperative_kitchen"))
import normalize_canary as tool

CODE = hashlib.sha256(b"synthetic-canary-runtime").hexdigest()
URL = "https://synthetic-canary.onrender.com"


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def binding(case):
    manifest = json.loads(case["manifest"].read_text())
    return {"runtime_sha256": CODE, "manifest_sha256": tool.sha(case["manifest"].read_bytes()),
            "actor_sha256": h("actor"), "program_sha256": h("program"),
            "qa_configuration_sha256": tool.sha(tool.canonical(manifest["qa_configuration"]))}


def evidence(case, kind="closed_restart"):
    bind = binding(case)
    closed = kind == "closed_restart"
    result = {"schema": "kitchen_canary_evidence_v1", "kind": kind, "endpoint": URL,
              "binding": bind, "started_at": 1000, "finished_at": 1300,
              "enrollment_mode": "closed" if closed else "internal_pilot",
              "freeplay_qa_enabled": closed, "sessions": [], "questions": []}
    if closed:
        result["restart"] = {"provider": "render", "operation": "restart", "operator_confirmed": True,
                             "evidence_sha256": h("restart-evidence"), "dashboard_event": {
                                 "service_id": "srv-synthetic", "dashboard_url": "https://dashboard.render.com/web/srv-synthetic/events",
                                 "event_text": "Service restarted by you January 1, 1970 at 12:20 AM",
                                 "displayed_timezone": "UTC", "displayed_event_minute": "1970-01-01T00:20:00+00:00",
                                 "time_precision": "minute", "source": "authenticated_render_dashboard_accessibility_tree"}}
    for i in range(1 if closed else 4):
        run, episode = "synthetic_run_" + str(i), "synthetic_ep_" + str(i)
        condition = None if closed else ["A", "B", "A", "B"][i]
        capture = {"endpoint": URL, "binding": copy.deepcopy(bind), "run_id": run, "episode_id": episode,
                   "phase": "freeplay" if closed else "task1", "version": 9, "turn": 3,
                   "state_sha256": h("state-" + str(i))}
        session = {"run_id": run, "episode_id": episode, "namespace": "development" if closed else "pilot",
                   "mode": "freeplay" if closed else "pilot", "condition": condition,
                   "task_order": None if closed else ["XY", "XY", "YX", "YX"][i],
                   "before": {**capture, "captured_at": 1190, "capture_sha256": h("before-capture"), "record_index": 17},
                   "after": {**capture, "captured_at": 1280, "capture_sha256": h("after-capture"), "record_index": 1},
                   "replay": {**capture, "captured_at": 1280, "capture_sha256": h("after-capture"), "record_index": 4},
                   "operation": {"operation_id_sha256": h("op-" + str(i)), "replayed_operation_id_sha256": h("op-" + str(i)),
                                 "request_sha256": h("request-" + str(i)), "replayed_request_sha256": h("request-" + str(i)),
                                 "first_response_sha256": h("response-" + str(i)), "replayed_response_sha256": h("response-" + str(i)),
                                 "request_version": 8, "first_response_version": 9, "replayed_response_version": 9,
                                 "first_state_sha256": h("state-" + str(i)), "replayed_state_sha256": h("state-" + str(i)),
                                 "first_http_status": 200, "replayed_http_status": 200,
                                 "database": {"evidence_source": "postgresql", "record_sha256": h("database-" + str(i)),
                                              "operation_receipt_count": 1, "joint_step_event_count": 1,
                                              "request_sha256": h("request-" + str(i)), "recorded_response_version": 9},
                                 "first_ack_seconds": .25, "replay_ack_seconds": .18}}
        result["sessions"].append(session)
        if closed or condition == "A":
            for j, lang in enumerate(("zh", "en")):
                result["questions"].append({"question_id": "synthetic_q_" + str(i) + "_" + lang,
                    "run_id": run, "episode_id": episode, "frame": 0, "language": lang, "provider": "deepseek",
                    "status": "complete", "llm_success": True, "verified": True,
                    "evidence_source": "postgresql", "record_sha256": h("private-record-" + str(i) + lang),
                    "request_version": 2 + j, "completed_version": 3 + j,
                    "game_before_sha256": h("qa-state"), "game_after_sha256": h("qa-state"), "elapsed_seconds": 1.5})
    return result


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "runtime_hash", lambda: CODE)
    release = tmp_path / "output/cooperative_kitchen/v3-id-pilot"
    recovery = {"schema": "kitchen_recovery_audit_v3", "passed": False, "runtime_sha256": CODE,
                "remote_restart_tested": False, "local_postgresql": {"passed": True, "report_sha256": h("local")}}
    remote = {"schema": "kitchen_remote_load_audit_v3", "passed": False, "mode": "pending_xai_render_canary",
              "runtime_sha256": CODE, "previous_20_session_result": {"passed": False,
                  "action_p95_seconds": 6.76, "question_p95_seconds": 37.97,
                  "zero_duplicate_steps": True}}
    targets = {name: release / f"acceptance/{name}_report.json" for name in ("recovery", "remote_load")}
    write(targets["recovery"], recovery); write(targets["remote_load"], remote)
    manifest = {"schema": "cooperative_kitchen_release_manifest_v1", "status": "candidate",
                "runtime_sha256": CODE, "qa_configuration": {"provider": "deepseek", "model": "synthetic-fixture-model"},
                "artifacts": {"actor": {"path": "selected/actor.npz", "sha256": h("actor")},
                              "program": {"path": "artifacts/program.json", "sha256": h("program")}}}
    for name, target in targets.items():
        manifest["artifacts"][name + "_report"] = {"path": target.relative_to(release).as_posix(), "sha256": tool.sha(target.read_bytes())}
    manifest_path = release / "manifest.json"; write(manifest_path, manifest)
    result = {"release": release, "manifest": manifest_path, "targets": targets,
              "evidence": tmp_path / "deidentified-evidence.json"}
    write(result["evidence"], evidence(result))
    return result


def refreeze_fixture(case):
    manifest = json.loads(case["manifest"].read_text())
    for name, target in case["targets"].items():
        manifest["artifacts"][name + "_report"]["sha256"] = tool.sha(target.read_bytes())
    write(case["manifest"], manifest)


def test_default_dry_run_has_no_writes_and_never_passes_twenty_session_gate(case, capsys):
    before = {p: p.read_bytes() for p in case["release"].rglob("*") if p.is_file()}
    assert tool.main(["--release", str(case["release"]), "--evidence", str(case["evidence"]), "--expected-url", URL]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["validated"] and not result["written"] and not result["remote_load_passed"]
    assert {p: p.read_bytes() for p in case["release"].rglob("*") if p.is_file()} == before
    assert not (case["release"] / "provenance").exists()


def test_explicit_write_backs_up_originals_and_preserves_all_historical_metrics(case):
    originals = {p.name: p.read_bytes() for p in [case["manifest"], *case["targets"].values()]}
    result = tool.normalize(case["release"], case["evidence"], URL, write=True)
    assert result["written"] and result["manifest_requires_refreeze"]
    backup = Path(result["backup_directory"])
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    for name, data in originals.items():
        assert (backup / name).read_bytes() == data
        assert stat.S_IMODE((backup / name).stat().st_mode) == 0o600
    assert case["manifest"].read_bytes() == originals["manifest.json"]
    for path in case["targets"].values(): assert stat.S_IMODE(path.stat().st_mode) == 0o600
    remote = json.loads(case["targets"]["remote_load"].read_text())
    assert remote["passed"] is False and remote["mode"] == "real_remote"
    assert remote["previous_20_session_result"] == json.loads(originals["remote_load_report.json"])["previous_20_session_result"]
    assert remote["closed_canary"]["sessions"] == 1
    assert remote["closed_canary"]["languages"] == ["en", "zh"]
    recovery = json.loads(case["targets"]["recovery"].read_text())
    assert recovery["passed"] and recovery["remote_restart_tested"]
    assert recovery["local_postgresql"] == json.loads(originals["recovery_report.json"])["local_postgresql"]
    assert json.loads((backup / "transaction.json").read_text())["status"] == "complete"
    with pytest.raises(tool.EvidenceError, match="report_binding_mismatch"):
        tool.normalize(case["release"], case["evidence"], URL, write=True)


def test_four_internal_sessions_are_nested_and_keep_actual_old_restart_manifest(case):
    tool.normalize(case["release"], case["evidence"], URL, write=True)
    old = json.loads(case["targets"]["recovery"].read_text())["render_restart_canary"]
    refreeze_fixture(case)
    current = evidence(case, "internal_four")
    assert current["binding"]["manifest_sha256"] != old["binding"]["manifest_sha256"]
    write(case["evidence"], current)
    tool.normalize(case["release"], case["evidence"], URL, write=True)
    remote = json.loads(case["targets"]["remote_load"].read_text())
    recovery = json.loads(case["targets"]["recovery"].read_text())
    assert remote["passed"] is False and remote["four_session_canary"]["sessions"] == 4
    assert remote["four_session_canary"]["passed"] is True
    assert recovery["render_restart_canary"] == old == remote["closed_canary"]


@pytest.mark.parametrize("path,value", [
    (("endpoint",), "https://other.onrender.com"),
    (("binding", "runtime_sha256"), h("other-runtime")),
    (("binding", "manifest_sha256"), h("other-manifest")),
    (("binding", "qa_configuration_sha256"), h("other-config")),
    (("sessions", 0, "after", "run_id"), "another-session"),
    (("sessions", 0, "after", "episode_id"), "another-episode"),
    (("sessions", 0, "after", "version"), 10),
    (("sessions", 0, "replay", "turn"), 4),
    (("sessions", 0, "after", "state_sha256"), h("mutated-game")),
    (("sessions", 0, "operation", "request_version"), 1),
    (("sessions", 0, "operation", "replayed_operation_id_sha256"), h("different-op")),
    (("sessions", 0, "operation", "replayed_request_sha256"), h("different-request")),
    (("sessions", 0, "operation", "replayed_state_sha256"), h("different-game-state")),
    (("sessions", 0, "operation", "database", "operation_receipt_count"), 2),
    (("sessions", 0, "operation", "database", "joint_step_event_count"), 2),
    (("sessions", 0, "operation", "database", "request_sha256"), h("different-database-request")),
    (("sessions", 0, "operation", "database", "recorded_response_version"), 10),
    (("restart", "operation"), "local-restart"),
    (("restart", "dashboard_event", "displayed_event_minute"), "1970-01-01T00:18:00+00:00"),
    (("restart", "dashboard_event", "dashboard_url"), "https://dashboard.render.com/web/another/events"),
    (("restart", "dashboard_event", "time_precision"), "second"),
    (("questions", 0, "llm_success"), False),
    (("questions", 0, "llm_success"), 1),
    (("questions", 0, "verified"), False),
    (("questions", 0, "provider"), "program"),
    (("questions", 0, "evidence_source"), "public_verified_flag"),
    (("questions", 0, "game_after_sha256"), h("changed-by-question")),
    (("questions", 0, "completed_version"), 10),
    (("questions", 1, "language"), "zh"),
    (("enrollment_mode",), "formal"),
    (("freeplay_qa_enabled",), False),
])
def test_bad_binding_replay_or_cloud_evidence_never_writes(case, path, value):
    value_before = {p: p.read_bytes() for p in case["targets"].values()}
    document = evidence(case)
    selected = document
    for part in path[:-1]: selected = selected[part]
    selected[path[-1]] = value
    write(case["evidence"], document)
    with pytest.raises(tool.EvidenceError): tool.normalize(case["release"], case["evidence"], URL, write=True)
    assert {p: p.read_bytes() for p in case["targets"].values()} == value_before
    assert not (case["release"] / "provenance").exists()


@pytest.mark.parametrize("name", ["cookie", "answer", "api_key", "question"])
def test_unexpected_sensitive_fields_are_rejected_without_echo(case, capsys, name):
    document = evidence(case); document["questions"][0][name] = "SECRET-SENTINEL-NEVER-ECHO"
    write(case["evidence"], document)
    assert tool.main(["--release", str(case["release"]), "--evidence", str(case["evidence"]), "--expected-url", URL, "--write"]) == 1
    output = capsys.readouterr()
    assert "SECRET-SENTINEL" not in output.out + output.err
    assert json.loads(output.out)["error"] == "evidence_fields_mismatch"


def test_partial_write_failure_rolls_back_both_reports(case, monkeypatch):
    originals = {p: p.read_bytes() for p in case["targets"].values()}
    original = tool.atomic_write; failed = [False]
    def fail_once(path, value):
        if path == case["targets"]["remote_load"] and not failed[0]:
            failed[0] = True
            raise OSError("synthetic-write-failure")
        return original(path, value)
    monkeypatch.setattr(tool, "atomic_write", fail_once)
    with pytest.raises(OSError): tool.normalize(case["release"], case["evidence"], URL, write=True)
    assert {p: p.read_bytes() for p in case["targets"].values()} == originals
    assert len(list((case["release"] / "provenance").glob("canary_*"))) == 1


def test_four_session_input_cannot_unlock_missing_restart(case):
    write(case["evidence"], evidence(case, "internal_four"))
    with pytest.raises(tool.EvidenceError, match="validated_closed_restart_required_first"):
        tool.normalize(case["release"], case["evidence"], URL, write=True)


def test_slow_measured_canary_is_reported_failed_without_hiding_functional_recovery(case):
    document = evidence(case); document["questions"][0]["elapsed_seconds"] = 35
    write(case["evidence"], document)
    result = tool.normalize(case["release"], case["evidence"], URL, write=True)
    assert result["canary_passed"] is False and result["remote_load_passed"] is False
    remote = json.loads(case["targets"]["remote_load"].read_text())
    assert remote["closed_canary"]["question_p95_seconds"] == 35
    assert json.loads(case["targets"]["recovery"].read_text())["passed"] is True


def test_symlink_and_duplicate_json_keys_are_rejected(case):
    link = case["evidence"].with_name("linked-evidence.json"); link.symlink_to(case["evidence"])
    with pytest.raises(tool.EvidenceError, match="symlink"):
        tool.normalize(case["release"], link, URL)
    case["evidence"].write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(tool.EvidenceError, match="duplicate_json_key"):
        tool.normalize(case["release"], case["evidence"], URL)


def test_concurrent_report_change_is_not_overwritten(case, monkeypatch):
    original = tool.prepare
    changed = b'{"concurrent_writer":"preserve-this-file"}\n'
    def concurrent_change(*args):
        result = original(*args)
        case["targets"]["recovery"].write_bytes(changed)
        return result
    monkeypatch.setattr(tool, "prepare", concurrent_change)
    with pytest.raises(tool.EvidenceError, match="release_changed_before_write"):
        tool.normalize(case["release"], case["evidence"], URL, write=True)
    assert case["targets"]["recovery"].read_bytes() == changed
    assert not list((case["release"] / "provenance").glob("canary_*"))


def test_current_view_replay_after_questions_can_have_newer_version_and_response(case):
    document = evidence(case)
    session = document["sessions"][0]
    for name in ("before", "after", "replay"): session[name]["version"] = 5
    # The original action committed version 2. Two questions and one language
    # change then reached version 5 without advancing the game a second time.
    op = session["operation"]
    op.update(request_version=1, first_response_version=2, replayed_response_version=5,
              replayed_response_sha256=h("different-current-view-with-answers"))
    op["database"]["recorded_response_version"] = 2
    document["questions"][0].update(request_version=2, completed_version=3)
    document["questions"][1].update(request_version=4, completed_version=5)
    # A real dashboard's minute may overlap the before capture. Do not invent
    # the second of the request just to make the comparison artificially exact.
    session["before"]["captured_at"] = 1203
    write(case["evidence"], document)
    result = tool.normalize(case["release"], case["evidence"], URL)
    assert result["validated"] and not result["written"]


def test_failed_local_recovery_cannot_be_overridden_by_a_small_remote_canary(case):
    recovery = json.loads(case["targets"]["recovery"].read_text())
    recovery["local_postgresql"]["passed"] = False
    write(case["targets"]["recovery"], recovery); refreeze_fixture(case)
    write(case["evidence"], evidence(case))
    with pytest.raises(tool.EvidenceError, match="local_recovery_evidence_required"):
        tool.normalize(case["release"], case["evidence"], URL, write=True)
