"""Only the quality-selected formal pair needs two published role programs."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from experiments.warehouse import protocol as p
from experiments.warehouse.acceptance import _training_assets as verify_assets
from scripts import package_warehouse_v6_release as packaging


def _source(path, payload: object) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _variants(output_root, available_roles: tuple[str, ...]) -> dict:
    variants = {}
    for variant in p.VARIANTS:
        role_sha = {role: sha256(f"{variant}:{role}".encode()).hexdigest()
                    for role in p.ROLES}
        actor = _source(output_root / f"{variant}.actor.npz", {"variant": variant})
        programs = {}
        if variant == "regularized":
            programs = {role: {
                **_source(output_root / f"{role}.program.json", {"role": role}),
                "actor_role_sha256": role_sha[role],
            } for role in available_roles}
        row = {
            "actor_artifact": {**actor, "role_sha256": role_sha},
            "selected_role_sha256": role_sha,
            "program_artifacts": programs,
        }
        report = _source(output_root / f"{variant}.training.json", row)
        variants[variant] = {**row, "training_report_path": report["path"],
                             "training_report_sha256": report["sha256"]}
    return variants


@pytest.mark.parametrize("available_roles", ((), ("robot_1",)))
def test_unselected_formal_pair_keeps_partial_fit_receipts_but_selected_rejects_them(
        tmp_path, monkeypatch, available_roles):
    monkeypatch.setattr(packaging, "ROOT", tmp_path)
    output_root = tmp_path / "output"
    assets_root = tmp_path / "releases" / p.INITIAL_RELEASE_ID / "assets"
    variants = _variants(output_root, available_roles)

    with pytest.raises(ValueError, match="role programs incomplete"):
        packaging._training_assets(
            variants, output_root, assets_root=assets_root)
    packaged = packaging._training_assets(
        variants, output_root, allow_missing_regularized_programs=True,
        assets_root=assets_root)
    assert set(packaged["regularized"]["programs"]) == set(available_roles)
    assert verify_assets(
        tmp_path, p.INITIAL_RELEASE_ID, packaged, variants,
        allow_missing_programs=True)
    assert not verify_assets(
        tmp_path, p.INITIAL_RELEASE_ID, packaged, variants,
        allow_missing_programs=False)


def test_selected_pair_requires_both_roles_and_unknown_role_is_never_allowed(
        tmp_path, monkeypatch):
    monkeypatch.setattr(packaging, "ROOT", tmp_path)
    output_root = tmp_path / "output"
    assets_root = tmp_path / "releases" / p.INITIAL_RELEASE_ID / "assets"
    complete = _variants(output_root, p.ROLES)
    packaged = packaging._training_assets(
        complete, output_root, assets_root=assets_root)
    assert verify_assets(tmp_path, p.INITIAL_RELEASE_ID, packaged, complete)

    bogus = _variants(output_root, ("robot_1",))
    bogus["regularized"]["program_artifacts"]["unknown"] = dict(
        bogus["regularized"]["program_artifacts"]["robot_1"])
    with pytest.raises(ValueError, match="role programs incomplete"):
        packaging._training_assets(
            bogus, output_root, allow_missing_regularized_programs=True,
            assets_root=assets_root)
