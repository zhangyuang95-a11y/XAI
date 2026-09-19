"""Independent regression checks from the second-revision review.

These are deterministic code/controlled-network tests, not real browser or
human-performance results and not an evaluation of semantic model quality.
"""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess

import pytest

from domains.pong import turnbased as pong
from study_v3.qa import _catalog, simulate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed,turn", [(730113, 0), (731103, 2), (731119, 78)])
def test_unseen_pong_contacts_cannot_change_decision_evidence_or_branch_output(seed, turn):
    state = pong.initial_state(seed, 2)
    for _ in range(turn):
        state = pong.step(state, pong.human_advisor(state))
    altered = deepcopy(state)
    unseen_ids = []
    for entry in altered["_schedule"]:
        if entry["spawn_turn"] <= turn:
            continue
        ball = entry["ball"]
        ball["id"] = "UNANNOUNCED-" + ball["id"]
        unseen_ids.append(ball["id"])
        ball["contacts"] = sorted(8 - x for x in ball["contacts"])
    assert unseen_ids
    before = deepcopy(state)
    original_decision, altered_decision = pong.decide(state), pong.decide(altered)
    assert original_decision == altered_decision
    assert _catalog(pong, state, original_decision, []) == _catalog(pong, altered, altered_decision, [])
    one = simulate(pong, state, original_decision, ["wait"], 12)
    two = simulate(pong, altered, altered_decision, ["wait"], 12)
    # Audit hashes intentionally identify the actual private input state.
    one.pop("input_state_hash")
    two.pop("input_state_hash")
    assert one == two
    assert one["stopped_at_public_boundary"]
    assert state == before
    assert not any(identifier in repr(two) for identifier in unseen_ids)


def test_frontend_network_races_and_key_edges_in_javascript_runtime():
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    node = os.environ.get("STUDY_TEST_NODE") or shutil.which("node") or (str(bundled) if bundled.exists() else None)
    if not node:
        pytest.skip("Node.js is required for the controlled-network frontend regression test")
    result = subprocess.run([node, str(ROOT / "tests/revision_frontend_review.cjs")], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    assert result.returncode == 0, result.stdout
    assert "checks passed" in result.stdout


@pytest.mark.parametrize("trial", [0, 1, 9, 17, 39, 79])
def test_kitchen_physical_lineage_survives_mixed_cooperation_and_mistakes(trial):
    """Every fetched component remains somewhere or has a recorded disposition."""
    from collections import Counter
    import random
    from domains.kitchen import engine as kitchen

    rng = random.Random(trial)
    state = kitchen.initial_state(1000 + trial % 4, 2)
    created, removed = [], []
    while not state["terminal"]:
        action = kitchen.human_advisor(state) if rng.random() > .12 else rng.choice(kitchen.legal_actions(state))
        state = kitchen.step(state, action)
        for event in state["events"]:
            if event["type"] == "ingredient_taken":
                created.extend(event["item"]["components"])
            if event["type"] in ("served", "waste", "pot_cleared") and event.get("item"):
                removed.extend(event["item"]["components"])
        # Burnt food still physically occupies its pan until a later clear.
        active = [item for item in kitchen._all_items(state) if item]
        active.extend(pot["item"] for pot in state["pots"] if pot["status"] == "burnt" and pot["item"])
        present = [component for item in active for component in item["components"]]
        assert len(present) == len(set(present)), (trial, state["turn"], "duplicate component")
        assert Counter(created) == Counter(present + removed), (trial, state["turn"], "lost or invented component")
