from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as subject


def _physical(index: int) -> str:
    return f"{index + 1:064x}"


def _arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "action_indices": np.asarray([row["action"] for row in rows], dtype=np.uint8),
        "scene_fingerprints": np.asarray([row["scene"] for row in rows], dtype="S64"),
        "episode_ids": np.asarray([row["episode"] for row in rows], dtype="S180"),
        "group_bits": np.asarray([row.get("group", 0) for row in rows], dtype=np.uint8),
        "kinds": np.asarray([row["kind"] for row in rows], dtype="S16"),
        "anchor_ids": np.asarray([row.get("anchor", "") for row in rows], dtype="S240"),
        "branch_actions": np.asarray([row.get("branch", "") for row in rows], dtype="S8"),
        "physical_hashes": np.asarray([row.get("physical", "") for row in rows], dtype="S64"),
        "split_validation": np.asarray([row.get("validation", False) for row in rows], dtype=np.bool_),
        # A v7 archive already contains weights.  Deliberately absurd values
        # prove that this independent builder does not inherit them.
        "weights": np.full(len(rows), 999.0, dtype=np.float32),
    }


def _row(
    scene: str,
    partner: str,
    *,
    kind: str,
    action: int,
    group: int = 0,
    anchor: str = "",
    branch: str = "",
    physical: str = "",
    validation: bool = False,
) -> dict:
    return {
        "scene": scene,
        "episode": f"episode-{scene}:{partner}",
        "kind": kind,
        "action": action,
        "group": group,
        "anchor": anchor,
        "branch": branch,
        "physical": physical,
        "validation": validation,
    }


def test_shared_wait_accumulates_half_of_every_symmetric_pair_occurrence():
    rows = [
        _row("scene-a", "skilled", kind="ordinary", action=4,
             group=1, anchor="anchor-a"),
        _row("scene-a", "skilled", kind="intervention", action=4,
             group=1, anchor="anchor-a", branch="WAIT", physical=_physical(0)),
        _row("scene-a", "skilled", kind="intervention", action=0,
             group=1, anchor="anchor-a", branch="UP", physical=_physical(1)),
        _row("scene-a", "skilled", kind="intervention", action=2,
             group=1, anchor="anchor-a", branch="LEFT", physical=_physical(2)),
    ]
    result = subject.build_pair_weights(
        _arrays(rows), scene_families={"scene-a": "family-a"}
    )

    assert result["pairs"].tolist() == [[1, 2], [1, 3]]
    # Both pairs occupy the same cell and have the same action factor here, so
    # each retains one pre-multiplier mass unit.  The default x8 pool multiplier
    # gives four units to each branch and two four-unit halves to WAIT.
    np.testing.assert_allclose(
        result["pair_contribution"], [0.0, 8.0, 4.0, 4.0],
        rtol=0.0, atol=1e-7,
    )
    assert result["weights"][1:].tolist() == pytest.approx([8.0, 4.0, 4.0])
    audit = result["audit"]
    assert audit["pair_count"] == 2
    assert audit["paired_endpoint_rows"] == 3
    assert audit["pair_endpoint_occurrences"] == 4
    assert audit["shared_wait_rows"] == 1
    assert audit["maximum_wait_degree"] == 2
    assert audit["wait_degrees"] == {"1": 2}
    assert audit["pair_pre_balance_unit_mass"] == 1.0
    assert audit["pair_pre_multiplier_mass_total"] == pytest.approx(2.0)
    assert audit["pair_pool_multiplier"] == 8.0
    assert audit["pair_occurrence_mass_total"] == pytest.approx(16.0)
    assert audit["weighted_pair_total_mass"] == pytest.approx(16.0)
    assert audit["pair_endpoint_contribution_total"] == pytest.approx(16.0)
    assert all(row["wait_contribution"] == pytest.approx(row["branch_contribution"])
               for row in audit["pair_occurrences"])
    assert all(audit["symmetry_checks"].values())


def test_cross_split_and_cross_anchor_candidates_are_rejected_without_pairing():
    rows = [
        _row("scene-a", "skilled", kind="intervention", action=4,
             anchor="mixed-anchor", branch="WAIT", physical=_physical(10)),
        _row("scene-a", "skilled", kind="intervention", action=0,
             anchor="mixed-anchor", branch="UP", physical=_physical(11),
             validation=True),
        _row("scene-a", "skilled", kind="intervention", action=1,
             anchor="other-anchor", branch="DOWN", physical=_physical(12)),
        _row("scene-a", "skilled", kind="ordinary", action=4),
    ]
    result = subject.build_pair_weights(
        _arrays(rows), scene_families={"scene-a": "family-a"}
    )

    assert result["pairs"].shape == (0, 2)
    assert not result["pair_contribution"].any()
    assert result["weights"][1] == 1.0
    audit = result["audit"]
    assert audit["invalid_intervention_rows"] == [0, 2]
    assert audit["invalid_reasons_by_row"]["0"] == ["anchor_cross_split"]
    assert audit["invalid_reasons_by_row"]["2"] == ["missing_wait"]
    assert audit["rejected_pair_candidates"]["anchor_cross_split"] == 1
    assert audit["rejected_pair_candidates"]["missing_wait"] == 1


@pytest.mark.parametrize(
    ("wait_scene", "wait_partner", "branch_scene", "branch_partner", "reason"),
    (
        ("scene-a", "skilled", "scene-b", "skilled", "anchor_cross_scene"),
        ("scene-a", "skilled", "scene-a", "noisy", "anchor_cross_episode"),
    ),
)
def test_same_anchor_cannot_pair_across_scene_or_episode(
    wait_scene: str,
    wait_partner: str,
    branch_scene: str,
    branch_partner: str,
    reason: str,
):
    rows = [
        _row(wait_scene, wait_partner, kind="intervention", action=4,
             anchor="reused-anchor", branch="WAIT", physical=_physical(20)),
        _row(branch_scene, branch_partner, kind="intervention", action=0,
             anchor="reused-anchor", branch="UP", physical=_physical(21)),
    ]
    result = subject.build_pair_weights(
        _arrays(rows),
        scene_families={"scene-a": "family-a", "scene-b": "family-b"},
    )

    assert result["pairs"].shape == (0, 2)
    assert result["audit"]["rejected_pair_candidates"] == {reason: 2}
    assert all(result["audit"]["invalid_reasons_by_row"][str(index)] == [reason]
               for index in range(2))


def _balanced_fixture() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    rows: list[dict] = []
    families = {"s1": "f1", "s2": "f1", "s3": "f2"}
    serial = 100

    def add_pairs(scene: str, partner: str, group: int, count: int) -> None:
        nonlocal serial
        anchor = f"pair-{scene}-{partner}-{group}-{serial}"
        rows.append(_row(scene, partner, kind="ordinary", action=4,
                         group=group, anchor=anchor))
        rows.append(_row(scene, partner, kind="intervention", action=4,
                         group=group, anchor=anchor, branch="WAIT",
                         physical=_physical(serial)))
        serial += 1
        for offset, branch in enumerate(subject.ACTIONS[:count]):
            rows.append(_row(
                scene, partner, kind="intervention", action=offset,
                group=group, anchor=anchor, branch=branch,
                physical=_physical(serial),
            ))
            serial += 1

    def add_invalid(scene: str, partner: str, group: int, count: int) -> None:
        nonlocal serial
        for _ in range(count):
            anchor = f"invalid-{serial}"
            rows.append(_row(
                scene, partner, kind="intervention", action=4,
                group=group, anchor=anchor, branch="WAIT",
                physical=_physical(serial),
            ))
            serial += 1

    specifications = (
        ("s1", "skilled", 1, 2, 2),
        ("s1", "skilled", 2, 1, 1),
        ("s1", "assertive", 1, 1, 1),
        ("s2", "skilled", 1, 1, 1),
        ("s2", "assertive", 2, 1, 2),
        ("s3", "skilled", 1, 3, 3),
        ("s3", "skilled", 2, 1, 1),
        ("s3", "assertive", 1, 1, 1),
    )
    for scene, partner, group, pair_count, invalid_count in specifications:
        add_pairs(scene, partner, group, pair_count)
        add_invalid(scene, partner, group, invalid_count)

    # Uneven ordinary multiplicities exercise within-cell action factors and
    # the cell renormalization independently of the anchor source rows above.
    for index in range(7):
        rows.append(_row("s1", "skilled", kind="ordinary", action=index % 4,
                         group=1))
    for index in range(2):
        rows.append(_row("s3", "assertive", kind="ordinary", action=index,
                         group=1))
    return _arrays(rows), families


def _assert_hierarchy_is_balanced(pool: dict) -> None:
    assert pool["post_action_cell_renormalized"] is True
    for row in pool["cell_totals"]:
        assert row["actual_mass"] == pytest.approx(row["target_mass"], abs=1e-12)

    family = {row["family"]: row["actual_mass"] for row in pool["family_totals"]}
    assert len(set(round(value, 10) for value in family.values())) == 1

    scenes: dict[str, list[float]] = {}
    for row in pool["scene_totals"]:
        scenes.setdefault(row["family"], []).append(row["actual_mass"])
    assert all(len(set(round(value, 10) for value in values)) == 1
               for values in scenes.values())

    partners: dict[tuple[str, str], list[float]] = {}
    for row in pool["partner_totals"]:
        partners.setdefault((row["family"], row["scene"]), []).append(
            row["actual_mass"]
        )
    assert all(len(set(round(value, 10) for value in values)) == 1
               for values in partners.values())

    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in pool["cell_totals"]:
        groups.setdefault(
            (row["family"], row["scene"], row["partner"]), []
        ).append(row["actual_mass"])
    assert all(len(set(round(value, 10) for value in values)) == 1
               for values in groups.values())


def test_each_occurrence_pool_balances_family_scene_partner_and_group_cells():
    arrays, families = _balanced_fixture()
    result = subject.build_pair_weights(arrays, scene_families=families)
    audit = result["audit"]

    assert audit["action_factor"]["enabled"] is True
    assert audit["action_factor"]["source"] == "fit action_indices only"
    for name in ("effective_pairs", "ordinary", "invalid_intervention"):
        pool = audit["balance"][name]
        assert pool["unit_count"] > 0
        assert pool["actual_total_mass"] == pytest.approx(
            pool["target_total_mass"], abs=1e-12
        )
        _assert_hierarchy_is_balanced(pool)

    # The aggregate audit exposes every requested roll-up as well as the
    # component responsible for its mass.
    combined = audit["balance"]["combined_training"]
    assert combined["family_totals"]
    assert combined["scene_totals"]
    assert combined["partner_totals"]
    assert combined["cell_totals"]
    assert all("components" in row for row in combined["cell_totals"])


def test_validation_label_changes_cannot_change_fit_weights_pairs_or_audit():
    rows = [
        _row("fit-scene", "skilled", kind="ordinary", action=4,
             group=1, anchor="fit-anchor"),
        _row("fit-scene", "skilled", kind="intervention", action=4,
             group=1, anchor="fit-anchor", branch="WAIT", physical=_physical(30)),
        _row("fit-scene", "skilled", kind="intervention", action=0,
             group=1, anchor="fit-anchor", branch="UP", physical=_physical(31)),
        _row("validation-scene", "noisy", kind="ordinary", action=0,
             validation=True),
        _row("validation-scene", "noisy", kind="intervention", action=4,
             anchor="validation-anchor", branch="WAIT", physical=_physical(32),
             validation=True),
        _row("validation-scene", "noisy", kind="intervention", action=1,
             anchor="validation-anchor", branch="DOWN", physical=_physical(33),
             validation=True),
    ]
    first_arrays = _arrays(rows)
    second_arrays = deepcopy(first_arrays)
    second_arrays["action_indices"][3:] = np.asarray([255, 254, 253], dtype=np.uint8)
    families = {"fit-scene": "fit-family"}

    first = subject.build_pair_weights(first_arrays, scene_families=families)
    second = subject.build_pair_weights(second_arrays, scene_families=families)

    np.testing.assert_array_equal(first["weights"], second["weights"])
    np.testing.assert_array_equal(first["pairs"], second["pairs"])
    np.testing.assert_array_equal(
        first["pair_contribution"], second["pair_contribution"]
    )
    assert first["audit"] == second["audit"]
    np.testing.assert_array_equal(first["weights"][3:], np.ones(3, dtype=np.float32))
    assert first["audit"]["validation_labels_accessed"] is False
    assert first["audit"]["validation_labels_used_for_pairs"] is False
    assert first["audit"]["validation_labels_used_for_action_factors"] is False


def test_pair_requires_unique_wait_and_branch_and_both_kinds_of_change():
    rows = [
        _row("s", "skilled", kind="intervention", action=4,
             anchor="duplicate-wait", branch="WAIT", physical=_physical(40)),
        _row("s", "skilled", kind="intervention", action=3,
             anchor="duplicate-wait", branch="WAIT", physical=_physical(41)),
        _row("s", "skilled", kind="intervention", action=0,
             anchor="duplicate-wait", branch="UP", physical=_physical(42)),
        _row("s", "skilled", kind="intervention", action=4,
             anchor="duplicate-branch", branch="WAIT", physical=_physical(43)),
        _row("s", "skilled", kind="intervention", action=0,
             anchor="duplicate-branch", branch="UP", physical=_physical(44)),
        _row("s", "skilled", kind="intervention", action=1,
             anchor="duplicate-branch", branch="UP", physical=_physical(45)),
        _row("s", "skilled", kind="intervention", action=4,
             anchor="unchanged", branch="WAIT", physical=_physical(46)),
        _row("s", "skilled", kind="intervention", action=0,
             anchor="unchanged", branch="UP", physical=_physical(46)),
        _row("s", "skilled", kind="intervention", action=4,
             anchor="same-action", branch="WAIT", physical=_physical(47)),
        _row("s", "skilled", kind="intervention", action=4,
             anchor="same-action", branch="UP", physical=_physical(48)),
    ]
    result = subject.build_pair_weights(
        _arrays(rows), scene_families={"s": "f"}
    )

    assert result["pairs"].shape == (0, 2)
    reasons = result["audit"]["rejected_pair_candidates"]
    assert reasons == {
        "duplicate_branch": 2,
        "duplicate_wait": 1,
        "unchanged_actor_action": 1,
        "unchanged_physical_hash": 1,
    }
    assert result["audit"]["invalid_intervention_fit_rows"] == len(rows)


@pytest.mark.parametrize("value", [0.0, -1.0, 32.01, np.inf, np.nan, True])
def test_pair_pool_multiplier_is_finite_positive_and_capped(value):
    rows = [_row("s", "skilled", kind="ordinary", action=4)]
    with pytest.raises(ValueError, match="pair_pool_multiplier"):
        subject.build_pair_weights(
            _arrays(rows), scene_families={"s": "f"},
            pair_pool_multiplier=value,
        )


def test_explicit_pair_pool_multiplier_scales_only_pair_mass():
    multiplier_contract = subject.contract()["pair_pool_multiplier"]
    assert multiplier_contract["default"] == 8.0
    assert multiplier_contract["minimum_exclusive"] == 0.0
    assert multiplier_contract["maximum_inclusive"] == 32.0

    rows = [
        _row("s", "skilled", kind="ordinary", action=4,
             group=1, anchor="a"),
        _row("s", "skilled", kind="intervention", action=4,
             group=1, anchor="a", branch="WAIT", physical=_physical(60)),
        _row("s", "skilled", kind="intervention", action=0,
             group=1, anchor="a", branch="UP", physical=_physical(61)),
    ]
    result = subject.build_pair_weights(
        _arrays(rows), scene_families={"s": "f"}, pair_pool_multiplier=3.5,
    )

    np.testing.assert_allclose(result["weights"], [1.0, 1.75, 1.75])
    np.testing.assert_allclose(result["pair_contribution"], [0.0, 1.75, 1.75])
    occurrence = result["audit"]["pair_occurrences"][0]
    assert occurrence["pre_balance_unit_mass"] == 1.0
    assert occurrence["balanced_occurrence_mass"] == pytest.approx(1.0)
    assert occurrence["pair_pool_multiplier"] == 3.5
    assert occurrence["weighted_occurrence_mass"] == pytest.approx(3.5)
    assert occurrence["wait_contribution"] == pytest.approx(1.75)
    assert occurrence["branch_contribution"] == pytest.approx(1.75)
    assert result["audit"]["weighted_pair_total_mass"] == pytest.approx(3.5)
