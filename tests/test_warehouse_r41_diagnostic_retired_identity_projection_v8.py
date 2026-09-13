from copy import deepcopy
import json
from pathlib import Path
import reprlib

import pytest

from backend.training import (
    warehouse_r41_diagnostic_retired_identity_projection_v8 as subject,
)


VERSIONS = sorted(subject.SOURCE_FILE_SHA256)


def _candidates() -> list[dict]:
    return [
        {
            "id": f"candidate-{index}",
            "seed": 100_000 + index,
            "fingerprint": subject.digest({"candidate": index}),
            "snapshot": {"public_but_not_projected": index},
        }
        for index in range(2160)
    ]


def _manifest(candidates: list[dict]) -> dict:
    return {
        "authentication": {
            "full_manifest_json_parsed": False,
            "replay_scope": "none",
            "development_projection_sha256": (
                subject.manifest_binding.EXPECTED_DEVELOPMENT_PROJECTION_SHA256),
            "candidate_scenes_disjoint_from_all_base_splits_authenticated": True,
        },
        "candidate_batches": [
            candidates[0:720], candidates[720:1440], candidates[1440:2160],
        ],
    }


def _source(
    version: str, candidates: list[dict], *, accepted: range, exposed: range,
) -> dict:
    scenes = []
    for index in accepted:
        row = candidates[index]
        scenes.append({
            "id": f"private-{version}-{index}",
            "seed": row["seed"],
            "fingerprint": row["fingerprint"],
            "snapshot": {"must_not_escape": index},
            "workload_screen": {"private_metric": index / 64},
        })
    trace = []
    for index in exposed:
        trace.append({
            "fingerprint": candidates[index]["fingerprint"],
            "accepted": index in accepted,
            "rejection_reason": None if index in accepted else "private reason",
            "workload_receipt": {"private_action": "RIGHT"},
            "public_observation_hashes_sha256": subject.digest({"obs": index}),
        })
    return {
        "version": version,
        "scenes": scenes,
        "selection_trace": trace,
        "statistics": {"accepted": 64, "private_score": 1.0},
        "formal_ready": False,
    }


def _fixture_inputs(tmp_path: Path, monkeypatch):
    candidates = _candidates()
    manifest_value = _manifest(candidates)
    actor = tmp_path / "actor.npz"
    manifest = tmp_path / "manifest.json"
    validation = tmp_path / "validation.json"
    actor.write_bytes(b"fixture actor\n")
    manifest.write_text("{}\n", encoding="utf-8")
    validation.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(subject, "EXPECTED_ACTOR_SHA256", subject.file_hash(actor))
    monkeypatch.setattr(
        subject, "EXPECTED_MANIFEST_SHA256", subject.file_hash(manifest))
    monkeypatch.setattr(
        subject, "EXPECTED_MANIFEST_VALIDATION_SHA256",
        subject.file_hash(validation),
    )
    def identity_only_manifest_reader(*args, **kwargs):
        assert kwargs.get("actor_path") is None
        assert kwargs.get("replay_scope") == "none"
        return deepcopy(manifest_value)

    monkeypatch.setattr(
        subject.manifest_binding,
        "read_saved_manifest",
        identity_only_manifest_reader,
    )
    monkeypatch.setattr(
        subject,
        "_regenerate_development_candidate_identity_batches",
        lambda: [[
            {"seed": row["seed"], "fingerprint": row["fingerprint"]}
            for row in batch
        ] for batch in deepcopy(manifest_value["candidate_batches"])],
    )

    source_values = {
        VERSIONS[0]: _source(
            VERSIONS[0], candidates,
            accepted=range(0, 64), exposed=range(0, 68)),
        # Five trace-touched identities (63..67) overlap v1, while the 64
        # accepted identities (68..131) stay disjoint.
        VERSIONS[1]: _source(
            VERSIONS[1], candidates,
            accepted=range(68, 132), exposed=range(63, 139)),
    }
    paths = []
    expected = {}
    for index, version in enumerate(VERSIONS, start=1):
        path = tmp_path / f"retired-v{index}.json"
        path.write_text(
            subject.canonical(source_values[version]) + "\n", encoding="utf-8")
        paths.append(path)
        expected[version] = subject.file_hash(path)
    monkeypatch.setattr(subject, "SOURCE_FILE_SHA256", expected)
    return actor, manifest, validation, paths, expected, manifest_value


def _build_fixture(tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, expected, manifest_value = _fixture_inputs(
        tmp_path, monkeypatch)
    output = tmp_path / "projection"
    report = subject.build(
        actor_path=actor,
        manifest_path=manifest,
        retired_holdout_paths=paths,
        output=output,
    )
    projection_path = output / "retired_identity_projection.json"
    report_path = output / "report.json"
    return (
        paths, expected, output, projection_path, report_path, report,
        manifest_value,
    )


def test_build_releases_full_trace_exposure_union_without_private_semantics(
        tmp_path: Path, monkeypatch):
    (_, expected, _, projection_path, report_path, report,
     _) = _build_fixture(tmp_path, monkeypatch)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert set(projection) == {
        "version", "sources", "exposed_identities", "content_sha256",
    }
    assert [row["version"] for row in projection["sources"]] == VERSIONS
    assert all(set(row) == {
        "version", "source_file_sha256", "accepted_count",
        "exposed_count",
    } for row in projection["sources"])
    assert [row["accepted_count"] for row in projection["sources"]] == [64, 64]
    assert [row["exposed_count"] for row in projection["sources"]] == [68, 76]
    identities = projection["exposed_identities"]
    assert len(identities) == 139
    assert len({row["seed"] for row in identities}) == 139
    assert len({row["fingerprint"] for row in identities}) == 139
    assert all(set(identity) == {"seed", "fingerprint"}
               for identity in identities)
    serialized = subject.canonical(projection)
    for forbidden in (
            "snapshot", "workload_screen", "selection_trace", "statistics",
            "private_metric", "private_action", "private reason", '"accepted":',
            "rejection_reason", "public_observation_hashes_sha256"):
        assert forbidden not in serialized
    assert saved_report == report
    assert saved_report["declassification"] == subject._DECLASSIFICATION
    assert saved_report["source_file_sha256"] == expected
    assert saved_report["identity_counts"] == {
        VERSIONS[0]: {"accepted": 64, "exposed": 68},
        VERSIONS[1]: {"accepted": 64, "exposed": 76},
    }
    assert saved_report["global_identity_count"] == 139
    assert saved_report["supersedes_projection_sha256"] == (
        subject.SUPERSEDES_PROJECTION_SHA256)


def test_builder_never_loads_actor_or_replays_workloads(
        tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, _, _ = _fixture_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject.manifest_binding, "load_frozen_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("identity projection must not load Actor")))
    monkeypatch.setattr(
        subject.manifest_binding, "replay_workload_and_compare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("identity projection must not replay workloads")))
    report = subject.build(
        actor_path=actor,
        manifest_path=manifest,
        retired_holdout_paths=paths,
        output=tmp_path / "projection",
    )
    assert report["status"] == subject.STATUS


def test_candidate_identity_generator_never_observes_or_runs_workloads(
        monkeypatch):
    def forbidden(label):
        return lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(label))

    monkeypatch.setattr(
        subject.R41DiagnosticConflictWarehouseEnv,
        "reset",
        forbidden("identity generation must not call the public reset API"),
    )
    monkeypatch.setattr(
        subject.R41DiagnosticConflictWarehouseEnv,
        "observations",
        forbidden("identity generation must not calculate observations"),
    )
    monkeypatch.setattr(
        subject.R41DiagnosticConflictWarehouseEnv,
        "_info",
        forbidden("identity generation must not calculate environment info"),
    )
    monkeypatch.setattr(
        subject.R41DiagnosticConflictWarehouseEnv,
        "step",
        forbidden("identity generation must not create a trajectory"),
    )
    monkeypatch.setattr(
        subject.manifest_binding,
        "load_frozen_actor",
        forbidden("identity generation must not load the Actor"),
    )
    monkeypatch.setattr(
        subject.manifest_binding,
        "replay_workload_and_compare",
        forbidden("identity generation must not replay a workload"),
    )
    monkeypatch.setattr(
        subject.manifest_binding.scenes_api,
        "load_frozen_actor",
        forbidden("identity generation must not load the Actor"),
    )
    monkeypatch.setattr(
        subject.manifest_binding.scenes_api,
        "screen_scene",
        forbidden("identity generation must not screen a workload"),
    )
    monkeypatch.setattr(
        subject.manifest_binding.scenes_api,
        "replay_workload_and_compare",
        forbidden("identity generation must not replay a workload"),
    )

    first, first_report = subject._candidate_identity_batch(
        0, per_family=1, maximum_draws=10_000)
    second, second_report = subject._candidate_identity_batch(
        0, per_family=1, maximum_draws=10_000)
    assert first == second
    assert first_report == second_report
    assert len(first) == len(subject.manifest_binding.scenes_api.FAMILY_IDS)
    assert all(set(row) == {"seed", "fingerprint"} for row in first)
    assert len({row["seed"] for row in first}) == len(first)
    assert len({row["fingerprint"] for row in first}) == len(first)


def test_private_build_failure_has_no_sensitive_traceback_locals(
        tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, _, _ = _fixture_inputs(tmp_path, monkeypatch)
    source = json.loads(paths[0].read_text(encoding="utf-8"))
    source["scenes"][0]["snapshot"]["must_not_escape"] = (
        "retired-private-traceback-sentinel-4ea328")
    paths[0].write_text(subject.canonical(source) + "\n", encoding="utf-8")
    subject.SOURCE_FILE_SHA256[VERSIONS[0]] = subject.file_hash(paths[0])

    def fail_with_private_values(values, *, candidate_seeds):
        private_values = values
        private_seed_map = candidate_seeds
        private_message = next(iter(private_values.values()))[1][
            "scenes"][0]["snapshot"]["must_not_escape"]
        raise ValueError(private_message)

    monkeypatch.setattr(subject, "_declassify", fail_with_private_values)
    output = tmp_path / "projection"
    with pytest.raises(RuntimeError, match="private build failed") as caught:
        subject.build(
            actor_path=actor,
            manifest_path=manifest,
            retired_holdout_paths=paths,
            output=output,
        )
    error = caught.value
    assert error.__cause__ is None and error.__context__ is None

    seen: set[int] = set()
    pending: list[BaseException] = [error]
    rendered: list[str] = []
    product_file = Path(subject.__file__).resolve()
    bounded = reprlib.Repr()
    bounded.maxstring = 10_000
    bounded.maxother = 10_000
    bounded.maxlist = 128
    bounded.maxtuple = 128
    bounded.maxset = 128
    bounded.maxdict = 128
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((str(current), bounded.repr(current.args)))
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
        trace = current.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            if Path(frame.f_code.co_filename).resolve() == product_file:
                for value in frame.f_locals.values():
                    rendered.append(bounded.repr(value))
                    if isinstance(value, BaseException):
                        pending.append(value)
                    elif isinstance(value, dict):
                        pending.extend(child for child in value.values()
                                       if isinstance(child, BaseException))
                    elif isinstance(value, (list, tuple, set)):
                        pending.extend(child for child in value
                                       if isinstance(child, BaseException))
            trace = trace.tb_next
    assert "retired-private-traceback-sentinel-4ea328" not in "\n".join(rendered)
    assert not output.exists()
    assert not hasattr(subject, "_sensitive_worker")


def test_every_trace_touched_fingerprint_is_projected_with_public_seed(
        tmp_path: Path, monkeypatch):
    (paths, _, _, projection_path, _, _,
     manifest_value) = _build_fixture(tmp_path, monkeypatch)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    candidate_seed = subject._candidate_seed_map(manifest_value)
    expected_union = set()
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        expected_union.update({
            row["fingerprint"] for row in source["selection_trace"]
        } | {row["fingerprint"] for row in source["scenes"]})
    assert {row["fingerprint"] for row in projection["exposed_identities"]} == (
        expected_union)
    assert all(candidate_seed[row["fingerprint"]] == row["seed"]
               for row in projection["exposed_identities"])


def test_strict_reader_round_trip_binds_projection_report_and_sources(
        tmp_path: Path, monkeypatch):
    (_, _, _, projection_path, report_path, _,
     _) = _build_fixture(tmp_path, monkeypatch)
    result = subject.read_saved_projection(
        projection_path,
        expected_projection_sha256=subject.file_hash(projection_path),
        expected_report_sha256=subject.file_hash(report_path),
    )
    assert result["version"] == subject.VERSION
    assert len({identity["fingerprint"]
                for identity in result["exposed_identities"]}) == 139


@pytest.mark.parametrize("location", ["top", "source", "identity"])
def test_strict_reader_rejects_any_extra_projection_field(
        tmp_path: Path, monkeypatch, location: str):
    (_, expected, output, projection_path, report_path, _,
     _) = _build_fixture(tmp_path, monkeypatch)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    if location == "top":
        projection["statistics"] = {"private": True}
    elif location == "source":
        projection["sources"][0]["selection_trace"] = []
    else:
        projection["exposed_identities"][0]["accepted"] = True
    projection["content_sha256"] = subject.digest({
        key: value for key, value in projection.items()
        if key != "content_sha256"
    })
    projection_raw = subject._raw_json(projection)
    projection_path.write_bytes(projection_raw)
    report = subject._report(
        projection,
        subject._bytes_sha256(projection_raw),
        expected,
        subject.producer_sources(),
    )
    report_path.write_bytes(subject._raw_json(report))
    with pytest.raises(ValueError, match="schema|extra fields"):
        subject.read_saved_projection(
            projection_path,
            expected_projection_sha256=subject.file_hash(projection_path),
            expected_report_sha256=subject.file_hash(report_path),
        )
    assert output.is_dir()


def test_strict_reader_rejects_full_retired_holdout_as_projection(
        tmp_path: Path, monkeypatch):
    _, _, _, paths, _, _ = _fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / "projection"
    output.mkdir()
    projection_path = output / "retired_identity_projection.json"
    projection_path.write_bytes(paths[0].read_bytes())
    report_path = output / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level schema"):
        subject.read_saved_projection(
            projection_path,
            expected_projection_sha256=subject.file_hash(projection_path),
            expected_report_sha256=subject.file_hash(report_path),
        )


@pytest.mark.parametrize("mutation", ["count", "seed", "fingerprint"])
def test_projection_validation_enforces_counts_and_global_uniqueness(
        tmp_path: Path, monkeypatch, mutation: str):
    (_, _, _, projection_path, _, _,
     _) = _build_fixture(tmp_path, monkeypatch)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    if mutation == "count":
        projection["exposed_identities"].pop()
    elif mutation == "seed":
        projection["exposed_identities"][5]["seed"] += 1_000_000
    else:
        projection["exposed_identities"][5]["fingerprint"] = (
            projection["exposed_identities"][6]["fingerprint"])
    projection["content_sha256"] = subject.digest({
        key: value for key, value in projection.items()
        if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="count|global|identity"):
        subject._validate_projection(projection)


def test_builder_rejects_trace_fingerprint_missing_from_public_candidates(
        tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, _, manifest_value = _fixture_inputs(
        tmp_path, monkeypatch)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["selection_trace"][-1]["fingerprint"] = "f" * 64
    paths[0].write_text(subject.canonical(value) + "\n", encoding="utf-8")
    subject.SOURCE_FILE_SHA256[VERSIONS[0]] = subject.file_hash(paths[0])
    with pytest.raises(RuntimeError, match="private build failed"):
        subject.build(
            actor_path=actor,
            manifest_path=manifest,
            retired_holdout_paths=paths,
            output=tmp_path / "projection",
        )
    assert len(subject._candidate_seed_map(manifest_value)) == 2160


def test_builder_authenticates_exact_source_bytes_before_any_json_parse(
        tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, _, _ = _fixture_inputs(tmp_path, monkeypatch)
    expected = deepcopy(subject.SOURCE_FILE_SHA256)
    expected[VERSIONS[0]] = "0" * 64
    monkeypatch.setattr(subject, "SOURCE_FILE_SHA256", expected)
    monkeypatch.setattr(
        subject.manifest_binding,
        "read_saved_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manifest rebuilt before source byte authentication")),
    )
    monkeypatch.setattr(
        subject,
        "_parse_json_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("source JSON parsed before byte authentication")),
    )
    with pytest.raises(RuntimeError, match="private build failed"):
        subject.build(
            actor_path=actor,
            manifest_path=manifest,
            retired_holdout_paths=paths,
            output=tmp_path / "projection",
        )
    assert not (tmp_path / "projection").exists()


def test_builder_declassifies_the_bytes_hashed_from_one_descriptor(
        tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, _, manifest_value = _fixture_inputs(
        tmp_path, monkeypatch)
    snapshots = {path: path.read_bytes() for path in paths}
    expected_hashes = {
        path: subject._bytes_sha256(raw) for path, raw in snapshots.items()
    }
    original_read = subject._read_bytes_and_sha
    raw_source_reads = []

    def read_once(path, label):
        path = Path(path)
        if path in snapshots:
            raw_source_reads.append(path)
            # Replace the pathname after returning its authenticated bytes.
            # The builder must parse this returned byte snapshot, never reopen.
            raw = snapshots[path]
            path.write_text('{"attacker":"replacement"}\n', encoding="utf-8")
            return raw, expected_hashes[path]
        return original_read(path, label)

    monkeypatch.setattr(subject, "_read_bytes_and_sha", read_once)
    # The final path-integrity recheck must fail closed after semantic work;
    # reaching it proves parsing used the already authenticated snapshots.
    with pytest.raises(RuntimeError, match="private build failed"):
        subject.build(
            actor_path=actor,
            manifest_path=manifest,
            retired_holdout_paths=paths,
            output=tmp_path / "projection",
        )
    assert sorted(raw_source_reads) == sorted(paths)
    assert len(subject._candidate_seed_map(manifest_value)) == 2160
    assert not (tmp_path / "projection").exists()


def test_builder_rechecks_source_bytes_before_atomic_publish(
        tmp_path: Path, monkeypatch):
    actor, manifest, _, paths, _, _ = _fixture_inputs(tmp_path, monkeypatch)
    original = subject._atomic_output

    def drift_before_publish(output, projection_raw, report_raw, *, before_publish):
        paths[0].write_text('{"changed":true}\n', encoding="utf-8")
        return original(
            output, projection_raw, report_raw,
            before_publish=before_publish,
        )

    monkeypatch.setattr(subject, "_atomic_output", drift_before_publish)
    with pytest.raises(RuntimeError, match="private build failed"):
        subject.build(
            actor_path=actor,
            manifest_path=manifest,
            retired_holdout_paths=paths,
            output=tmp_path / "projection",
        )
    assert not (tmp_path / "projection").exists()
    assert not list(tmp_path.glob(".projection.tmp-*"))
