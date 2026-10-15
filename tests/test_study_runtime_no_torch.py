"""The deployed three-domain study must start with the Render dependency set."""
from __future__ import annotations

import subprocess
import sys
import json

import numpy as np


def test_study_manifest_imports_without_training_torch():
    script = """
import builtins
from types import SimpleNamespace
real_import = builtins.__import__
def block_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('simulated Render environment has no torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_torch
from study_v3 import server
settings = SimpleNamespace(
    automatic_explanations=False, understanding_ratings=True,
    optional_prompts=False, mode='preview', persistent=False,
    llm_configured=False, verified=False, ready=False)
manifest = server.manifest(settings)
assert set(manifest['domains']) == {'warehouse', 'pong', 'kitchen'}
assert manifest['study_ready'] is False
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", "import sys\n" + script],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_archived_structured_actor_frozen_reader_matches_original():
    import torch

    from backend.inference.warehouse_structured_v5 import (
        FrozenStructuredDualNN as RuntimeFrozen,
    )
    from backend.training.warehouse_structured_v5 import (
        FrozenStructuredDualNN as ArchivedFrozen, StructuredDualNN,
    )
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from env.warehouse.pure_nn import action_masks, observations

    torch.manual_seed(77)
    model = StructuredDualNN()
    runtime = RuntimeFrozen.from_model(model)
    archived = ArchivedFrozen.from_model(model)
    for seed in (811, 812):
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=seed)
        marked = env.get_state()
        marked.participant_controlled_agent_id = "robot_1"
        env.set_state(marked)
        for _ in range(5):
            obs, masks = observations(env.state), action_masks(env.state)
            np.testing.assert_allclose(
                runtime.logits(obs, masks), archived.logits(obs, masks),
                rtol=0.0, atol=0.0,
            )
            actions, _ = runtime.act(env.state, base_seed=seed)
            assert actions == archived.act(env.state, base_seed=seed)[0]
            env.step(actions)


def test_frozen_v6_actor_selects_legal_actions_without_torch(tmp_path):
    import torch

    from backend.training.warehouse_v6_actor import WarehouseV6Actor, export_actor

    torch.manual_seed(78)
    artifact = export_actor(WarehouseV6Actor(), tmp_path / "actor.npz",
                            pair_seed=78, variant="raw", candidate_id="no-torch-test")
    script = """
import builtins
import json
import sys
real_import = builtins.__import__
def block_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('simulated Render environment has no torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_torch
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
actor = FrozenWarehouseV6Actor.load(
    sys.argv[1], expected_sha256=sys.argv[2],
    expected_role_sha256=json.loads(sys.argv[3]))
env = WarehouseMultiAgentEnv(participant_study_config())
env.reset(seed=809)
marked = env.get_state()
marked.participant_controlled_agent_id = AGENTS[0]
env.set_state(marked)
for _ in range(8):
    if env.state.terminated or env.state.truncated:
        break
    legal = action_masks(env.state)
    actions, distributions = actor.act(env.state, base_seed=809)
    assert all(legal[index, ACTIONS.index(actions[role])] == 1
               and distributions[role].proposed_action == actions[role]
               for index, role in enumerate(AGENTS))
    _, _, _, _, info = env.step(actions)
    assert info['proposed_actions'] == actions
    assert not info['action_override_agents']
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, artifact["path"], artifact["sha256"],
         json.dumps(artifact["role_sha256"])],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
