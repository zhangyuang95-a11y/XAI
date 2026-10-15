import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/actors/actor_0050000.npz"
PROTOCOL = ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/protocol.json"
SCENARIOS = ROOT / "output/warehouse_native/native_cycle_500k_candidate_20260909/scenarios.json"


def test_online_import_does_not_load_training_frameworks():
    code = "import sys; import backend.warehouse_alignment_online_runtime; " \
           "assert not [m for m in sys.modules if m.startswith(('torch','sklearn','backend.training'))]"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_online_actor_matches_research_runtime_for_120_decisions():
    from backend.training.warehouse_native_common import digest, file_hash
    from backend.warehouse_alignment_runtime import AlignmentRuntime
    from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime

    protocol = json.loads(PROTOCOL.read_text())
    kwargs = dict(protocol=protocol, expected_actor_sha256=file_hash(ACTOR),
                  expected_protocol_sha256=digest(protocol))
    reference, online = AlignmentRuntime(ACTOR, **kwargs), OnlineAlignmentRuntime(ACTOR, **kwargs)
    scenarios = json.loads(SCENARIOS.read_text())["splits"]["play"]
    player_actions = ("UP", "LEFT", "RIGHT", "DOWN", "WAIT")
    count = 0
    for scenario_index, scenario in enumerate(scenarios):
        old_env, new_env = reference.environment(scenario), online.environment(scenario)
        for index in range(10):
            old_actions, old_decision = reference.decision(old_env)
            new_actions, new_decision = online.decision(new_env)
            assert old_actions == new_actions
            for role in old_actions:
                assert np.array_equal(old_decision["probabilities"][role],
                                      new_decision["probabilities"][role])
            action = player_actions[(scenario_index + index) % len(player_actions)]
            old_transition = reference.step(old_env, action)
            new_transition = online.step(new_env, action)
            assert old_transition["after"]["state"] == new_transition["after"]["state"]
            count += 1
    assert count == 120

