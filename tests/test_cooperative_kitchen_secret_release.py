"""Synthetic private-release security tests. No model/recruitment claims or uploads."""
import base64
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/cooperative_kitchen"
sys.path.insert(0, str(SCRIPTS))
import materialize_release as materializer
import package_secret_release as packer
import verify_deployment as verifier

CODE = hashlib.sha256(b"synthetic-test-runtime").hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.delenv("KITCHEN_OUTPUT", raising=False)
    source = tmp_path / "source"
    path = source / "output/cooperative_kitchen/v3-id-pilot"
    path.mkdir(parents=True)
    entries = {}
    for kind, suffix in materializer.ALLOWED.items():
        member = ("selected/actor" if kind == "actor" else "artifacts/" + kind) + suffix
        data = (b"synthetic-neural-file" if kind == "actor" else
                json.dumps({"passed": True, "fixture_only": True, "private_answer": kind}).encode())
        target = path / member; target.parent.mkdir(exist_ok=True); target.write_bytes(data)
        entries[kind] = {"path": member, "sha256": materializer.sha(data)}
    manifest = {"status": "candidate", "runtime_sha256": CODE, "artifacts": entries}
    (path / "manifest.json").write_text(json.dumps(manifest))
    descriptor = source / "deployment/cooperative_kitchen/release.json"
    archive = source / "output/cooperative_kitchen/private/bundle.zip"
    secret = archive.with_suffix(".b64")
    return {"root": source, "release": path, "descriptor": descriptor, "archive": archive, "secret": secret,
            "build_root": tmp_path / "build"}


def prepare(case):
    result = packer.prepare(case["release"], case["archive"], case["secret"], case["descriptor"],
        allow_candidate=True, root=case["root"], runtime_sha256=CODE)
    case["build_root"].mkdir()
    return result


def materialize(case, **kwargs):
    return materializer.materialize(case["descriptor"], case["secret"], root=case["build_root"],
        runtime_sha256=kwargs.pop("runtime_sha256", CODE), **kwargs)


def target(case):
    return case["build_root"] / "output/cooperative_kitchen/v3-id-pilot"


def rewrite_zip(case, transform):
    with zipfile.ZipFile(case["archive"]) as source:
        items = [(info, source.read(info)) for info in source.infolist()]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for info, contents in transform(items): archive.writestr(info, contents)
    raw = output.getvalue()
    case["secret"].write_bytes(base64.b64encode(raw))
    document = json.loads(case["descriptor"].read_text())
    document.update(bundle_sha256=materializer.sha(raw), bundle_bytes=len(raw))
    case["descriptor"].write_text(json.dumps(document))


def test_private_pack_public_descriptor_atomic_materialize_repeat_and_verify(release, monkeypatch):
    result = prepare(release)
    assert result["files"] == 13 and result["secret_file_bytes"] < 1_000_000
    assert stat.S_IMODE(release["secret"].stat().st_mode) == 0o600
    assert stat.S_IMODE(release["archive"].stat().st_mode) == 0o600
    assert "private_answer" not in release["descriptor"].read_text()
    assert "synthetic-neural-file" not in release["descriptor"].read_text()
    first = materialize(release)
    assert first["materialized"] and not first["recruitment_gate_changed"]
    assert materialize(release)["already_present"]
    assert stat.S_IMODE(target(release).stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in target(release).rglob("*") if p.is_file())
    from backend.cooperative_kitchen import artifacts
    monkeypatch.setattr(artifacts, "runtime_hash", lambda: CODE)
    monkeypatch.setattr(verifier, "ROOT", release["build_root"])
    report = verifier.verify(descriptor=release["descriptor"])
    assert report["passed"] and report["runtime_artifacts"] == 13 and report["release_status"] == "candidate"


@pytest.mark.parametrize("malformation", [b"not base64!", b" ", b"\t", b"\r", b"=", b"!!!!"])
def test_strict_base64_rejects_malformed_and_noncanonical(release, malformation):
    prepare(release)
    release["secret"].write_bytes(release["secret"].read_bytes().strip() + malformation)
    with pytest.raises(materializer.ReleaseError): materialize(release)
    assert not target(release).exists()


def test_text_editor_crlf_accepted_but_hash_still_bound(release):
    prepare(release)
    data = release["secret"].read_bytes().strip()
    release["secret"].write_bytes(b"\r\n".join(data[i:i+76] for i in range(0, len(data), 76)) + b"\r\n")
    assert materialize(release)["passed"]


def test_valid_base64_wrong_zip_hash_is_rejected(release):
    prepare(release)
    raw = bytearray(release["archive"].read_bytes()); raw[20] ^= 1
    release["secret"].write_bytes(base64.b64encode(raw))
    with pytest.raises(materializer.ReleaseError, match="bundle_checksum_mismatch"): materialize(release)


@pytest.mark.parametrize("change", ["extra", "missing", "duplicate", "symlink", "directory", "unexpected_contents", "zip_extra"])
def test_archive_exact_whitelist_and_types(release, change):
    prepare(release)
    def transform(items):
        if change == "extra": return items + [(zipfile.ZipInfo("leak.txt"), b"private")]
        if change == "missing": return items[:-1]
        if change == "duplicate": return items[:-1] + [items[0]]
        info, content = items[0]
        if change == "symlink": info.external_attr = (stat.S_IFLNK | 0o777) << 16
        if change == "directory": info.external_attr = (stat.S_IFDIR | 0o700) << 16
        if change == "unexpected_contents": content = b"x" * len(content)
        if change == "zip_extra": info.extra = b"\x01\x00\x00\x00"
        return [(info, content)] + items[1:]
    rewrite_zip(release, transform)
    with pytest.raises(materializer.ReleaseError): materialize(release)
    assert not target(release).exists()


@pytest.mark.parametrize("path", ["../escape.json", "/escape.json", "artifacts/../escape.json", "a\\escape.json", "a//b.json", "./file.json", ".hidden/file.json", "a\x00b.json"])
def test_descriptor_paths_cannot_escape_or_normalize(release, path):
    prepare(release)
    document = json.loads(release["descriptor"].read_text()); document["members"][1]["path"] = path
    release["descriptor"].write_text(json.dumps(document))
    with pytest.raises(materializer.ReleaseError): materialize(release)
    assert not target(release).exists()


@pytest.mark.parametrize("field,value", [("runtime_sha256", "0"*64), ("manifest_sha256", "0"*64),
    ("release_path", "ui/cooperative_kitchen_web"), ("release_path", "output/cooperative_kitchen/../escape"),
    ("bundle_bytes", 1_000_001), ("status", "awaiting_freeze")])
def test_public_descriptor_binding_and_prepared_gate(release, field, value):
    prepare(release)
    document = json.loads(release["descriptor"].read_text()); document[field] = value
    release["descriptor"].write_text(json.dumps(document))
    with pytest.raises(materializer.ReleaseError): materialize(release)


def test_manifest_artifact_cannot_disagree_with_public_descriptor(release):
    prepare(release)
    def transform(items):
        info, contents = items[0]
        manifest = json.loads(contents); manifest["artifacts"]["actor"]["path"] = "different.npz"
        return [(info, json.dumps(manifest).encode())] + items[1:]
    rewrite_zip(release, transform)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(release["secret"].read_bytes()))) as archive:
        contents = archive.read("manifest.json")
    document = json.loads(release["descriptor"].read_text())
    document["manifest_sha256"] = materializer.sha(contents)
    document["members"][0].update(sha256=materializer.sha(contents), bytes=len(contents))
    release["descriptor"].write_text(json.dumps(document))
    with pytest.raises(materializer.ReleaseError, match="manifest_artifact_binding_mismatch"): materialize(release)


def test_corrupted_existing_release_is_not_replaced(release):
    prepare(release); materialize(release)
    actor = target(release) / "selected/actor.npz"; actor.write_bytes(b"preserve-existing-corruption")
    with pytest.raises(materializer.ReleaseError): materialize(release)
    assert actor.read_bytes() == b"preserve-existing-corruption"


@pytest.mark.parametrize("where", ["secret", "descriptor", "parent", "destination", "member"])
def test_symlink_paths_are_rejected(release, where):
    prepare(release)
    if where in {"secret", "descriptor"}:
        path = release[where]; copy = path.with_name(path.name + ".real")
        path.rename(copy); path.symlink_to(copy)
    elif where == "parent":
        real = release["build_root"].parent / "outside"; real.mkdir()
        (release["build_root"] / "output").symlink_to(real, target_is_directory=True)
    elif where == "destination":
        target(release).parent.mkdir(parents=True)
        target(release).symlink_to(release["release"], target_is_directory=True)
    else:
        materialize(release)
        path = target(release) / "selected/actor.npz"; path.unlink()
        path.symlink_to(release["release"] / "selected/actor.npz")
    with pytest.raises(materializer.ReleaseError, match="symlink"): materialize(release)


def test_fixed_render_secret_mount_symlink_is_accepted(release, monkeypatch):
    prepare(release)
    secret = release["secret"]
    mounted = secret.with_name(secret.name + ".mounted")
    secret.rename(mounted)
    secret.symlink_to(mounted)
    monkeypatch.setattr(materializer, "SECRET", secret.absolute())
    assert materialize(release)["passed"]


def test_unknown_existing_file_or_empty_directory_rejected(release):
    prepare(release); materialize(release)
    unknown = target(release) / "extra"; unknown.mkdir()
    with pytest.raises(materializer.ReleaseError): materialize(release)
    assert unknown.exists()


def test_failed_rename_cleans_stage_and_does_not_expose_partial_release(release, monkeypatch):
    prepare(release)
    def fail(*args): raise OSError("synthetic rename failure")
    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError): materialize(release)
    assert not target(release).exists()
    assert list(target(release).parent.glob(".kitchen-release-*")) == []


def test_mismatched_configured_output_rejected(release, monkeypatch):
    prepare(release); monkeypatch.setenv("KITCHEN_OUTPUT", "output/cooperative_kitchen/v2-deepseek")
    with pytest.raises(materializer.ReleaseError, match="configured_output_mismatch"): materialize(release)


def test_packer_preserves_existing_outputs_and_requires_private_location(release):
    prepare(release)
    before = release["secret"].read_bytes()
    with pytest.raises(materializer.ReleaseError): prepare(release)
    assert release["secret"].read_bytes() == before
    with pytest.raises(materializer.ReleaseError, match="private_output_location_required"):
        packer.prepare(release["release"], release["root"] / "public.zip", release["root"] / "public.b64",
                       release["descriptor"], allow_candidate=True, root=release["root"], runtime_sha256=CODE)


def test_duplicate_json_keys_rejected(release):
    prepare(release)
    data = release["descriptor"].read_text()
    release["descriptor"].write_text(data.replace('"schema":', '"status":"prepared","schema":', 1))
    with pytest.raises(materializer.ReleaseError): materialize(release)


def test_secret_size_limit_before_decoding(release, monkeypatch):
    prepare(release)
    monkeypatch.setattr(materializer, "MAX_SECRET_BYTES", 100)
    with pytest.raises(materializer.ReleaseError): materialize(release)
    assert not target(release).exists()


def test_public_descriptor_expanded_size_limit(release):
    prepare(release)
    document = json.loads(release["descriptor"].read_text())
    document["members"][0]["bytes"] = materializer.MAX_EXPANDED_BYTES
    release["descriptor"].write_text(json.dumps(document))
    with pytest.raises(materializer.ReleaseError, match="expanded_size_limit"): materialize(release)


def test_explicit_descriptor_replacement_is_required_and_supported(release):
    release["descriptor"].parent.mkdir(parents=True)
    release["descriptor"].write_text('{"schema":"pending"}')
    with pytest.raises(materializer.ReleaseError, match="descriptor_exists"):
        packer.prepare(release["release"], release["archive"], release["secret"], release["descriptor"],
            allow_candidate=True, root=release["root"], runtime_sha256=CODE)
    report = packer.prepare(release["release"], release["archive"], release["secret"], release["descriptor"],
        replace_descriptor=True, allow_candidate=True, root=release["root"], runtime_sha256=CODE)
    assert report["passed"] and materializer.load_descriptor(release["descriptor"])["status"] == "prepared"


def test_cli_validation_errors_do_not_print_private_contents(release):
    prepare(release)
    release["secret"].write_bytes(b"private-test-marker-not-for-logs")
    process = subprocess.run([sys.executable, str(SCRIPTS / "materialize_release.py"),
        "--descriptor", str(release["descriptor"]), "--secret-file", str(release["secret"])],
        capture_output=True, text=True)
    assert process.returncode == 1
    assert "failed validation" in process.stderr
    assert "private-test-marker" not in process.stderr + process.stdout
    assert "Traceback" not in process.stderr and not process.stdout
