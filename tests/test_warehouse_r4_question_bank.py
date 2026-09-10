from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.training import warehouse_r4_question_bank as question


ROOT = Path(__file__).parents[1]


def test_frozen_question_pool_identity_is_exact():
    directory = ROOT / "output/warehouse_native/family_feedback_question_pool_20260910"
    assert question.file_hash(directory / "manifest.json") == question.REGISTERED_POOL_MANIFEST_SHA256
    assert question.file_hash(directory / "pool.json") == question.REGISTERED_POOL_SHA256


def test_registered_pool_reader_binds_manifest_and_pool_bytes(monkeypatch, tmp_path):
    root = tmp_path / "pool"
    root.mkdir()
    pool = root / "pool.json"
    manifest = root / "manifest.json"
    pool.write_text('{"scenes":[]}\n', encoding="utf-8")
    manifest.write_text('{"version":"registered"}\n', encoding="utf-8")
    seen = []
    registered_pool_sha = question.file_hash(pool)
    monkeypatch.setattr(question, "REGISTERED_POOL_MANIFEST_SHA256",
                        question.file_hash(manifest))
    monkeypatch.setattr(question, "REGISTERED_POOL_SHA256", registered_pool_sha)

    def read(path, *, expected_manifest_sha256, allow_test_fixture):
        seen.append((path, expected_manifest_sha256, allow_test_fixture))
        return {"pool_sha256": registered_pool_sha, "test_fixture": False,
                "scenes": []}

    monkeypatch.setattr(question.warehouse_family_question_pool,
                        "read_registered", read)
    registered, manifest_sha = question.read_registered_pool(pool, manifest)
    assert registered["pool_sha256"] == registered_pool_sha
    assert seen == [(root.resolve(), manifest_sha, False)]

    pool.write_text('{"scenes":[{}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="pool bytes differ"):
        question.read_registered_pool(pool, manifest)


def test_registered_pool_reader_rejects_cross_directory_and_links(monkeypatch, tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    (left / "pool.json").write_text("{}\n", encoding="utf-8")
    (right / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one registered"):
        question.read_registered_pool(left / "pool.json", right / "manifest.json")

    (left / "manifest-target.json").write_text("{}\n", encoding="utf-8")
    (left / "manifest.json").symlink_to(left / "manifest-target.json")
    with pytest.raises(ValueError, match="one registered"):
        question.read_registered_pool(left / "pool.json", left / "manifest.json")


def test_private_question_outputs_are_exclusive_and_owner_only(tmp_path):
    target = tmp_path / "question_bank.json"
    question._write_private(target, {"answer": "RIGHT"})
    assert json.loads(target.read_text()) == {"answer": "RIGHT"}
    assert os.stat(target).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        question._write_private(target, {"answer": "WAIT"})
