"""Own-shutdown Actor admission and real observed197 RCPD collection.

This is an independent collector, not a release gate or a training entry point.
``DATA_VERSION`` names the already frozen observed197 array container and
``fit_feedback`` reuses its actual nine-tree algorithm; neither identifies the
Actor as an old trainer. Full new Actor bindings remain in every data/result.

The caller reserves one horizon before ``collect_episode`` and saves the entire
return before acknowledging actual steps. A failed episode is never retried
here. No checkpoint deserialization, Critic, training, or final-test access.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from backend.training import warehouse_native_continuation_rcpd as data_api
from backend.training import warehouse_native_shutdown_evaluation as admission
from backend.training.warehouse_native_common import digest, jsonable
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment
from backend.training import warehouse_native_public_feedback_evaluation as physical
from backend.training import warehouse_native_partner_mix_evaluation as compact
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS, NumPyNativeActor
from env.warehouse_native.scenarios import reset_scenario

VERSION = "warehouse-native-own-shutdown-rcpd.v1"
DATA_VERSION = data_api.VERSION
PROFILES, GROUPS = data_api.PROFILES, data_api.GROUPS
REQUIRED_BINDINGS = (*admission.REQUIRED_BINDINGS, "actor_parameters_sha256")
ROOT = Path(__file__).resolve().parents[2]
EpisodeSamplingError = data_api.EpisodeSamplingError


def extraction_config():
    """New admission/collector identity, with an unchanged data/search contract."""
    return {"version": VERSION, "data_version": DATA_VERSION,
        "actor_version": admission.TRAINER_VERSION, "protocol_version": admission.PROTOCOL_VERSION,
        "data_contract": data_api.extraction_config(),
        "collection_reward": "original_shared_r1", "training_credit_applied": False,
        "training_shutdown_cost_applied": False, "actual_numpy_actor_required_in_fixtures": True,
        "release_eligible": False, "explanation_qualified": False}


def execution_sources():
    result = data_api.execution_sources()
    for key, value in admission.execution_sources().items():
        if key in result and result[key] != value:
            raise ValueError("Extraction dependencies disagree")
        result[key] = value
    result[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def validate_actor(actor, *, expected_bindings, protocol, allow_test_fixture=False, config=None):
    """Zero-forward admission; external PT-to-NP semantics remain caller evidence."""
    config = config or collaborative_study_config()
    if type(actor) is not NumPyNativeActor:
        raise ValueError("Only the real NumPy Actor can label extraction rows, including fixtures")
    if type(expected_bindings) is not dict or not set(REQUIRED_BINDINGS) <= set(expected_bindings):
        raise ValueError("Independent artifact, parameter, arm and source bindings are required")
    if not compact._sha(expected_bindings["actor_parameters_sha256"]):
        raise ValueError("Invalid external parameter semantic hash")
    # The current genuine export has no parameter semantic field. Its source
    # reader must establish PT -> NP array equality; do not invent NP metadata.
    known = {k: v for k, v in expected_bindings.items()
             if k != "actor_parameters_sha256" or k in actor.metadata}
    metadata, _, inherited = admission._contract(actor, protocol, known, allow_test_fixture, config)
    if sha256(actor.path.read_bytes()).hexdigest() != expected_bindings["actor_sha256"]:
        raise ValueError("Frozen Actor file bytes changed")
    weights = admission._weights_digest(actor)
    with np.load(actor.path, allow_pickle=False) as archive:
        if any(not np.array_equal(archive[name], value) for name, value in actor.weights.items()):
            raise ValueError("Loaded Actor tensors differ from its bound file")
    return {"metadata": metadata, "feature_names": list(metadata["feature_names"]),
        "configuration": asdict(config), "actor_training_clock": inherited + metadata["joint_steps"],
        "weights_sha256": weights,
        "extraction_config": extraction_config(), "extraction_config_sha256": digest(extraction_config()),
        "actor_parameters_binding": "externally_verified_checkpoint_semantics_not_reconstructed_from_npz",
        "release_eligible": False, "explanation_qualified": False}


def _receipt(data, *, contract, sources, episodes=None):
    value = {"version": VERSION, "data_version": DATA_VERSION,
        "data_sha256": data["data_sha256"], "actor_bindings": deepcopy(data["actor_bindings"]),
        "actor_metadata": deepcopy(contract["metadata"]),
        "actor_training_clock": contract["actor_training_clock"],
        "weights_sha256": contract["weights_sha256"], "execution_sources": sources,
        "configuration": deepcopy(contract["configuration"]),
        "extraction_config_sha256": digest(extraction_config()),
        "actual_joint_steps": data["joint_transitions"], "test_fixture": data["test_fixture"],
        "collection_reward": "original_shared_r1", "release_eligible": False,
        "explanation_qualified": False}
    if episodes is None:
        value["episode"] = deepcopy(data["episode"])
    else:
        value["episode_receipts"] = deepcopy(episodes)
    data["collector_receipt"] = value
    data["collector_receipt_sha256"] = digest(value)
    return data


def _validate_data(data, fixture):
    data_api._validate_data(data, fixture)
    receipt = data.get("collector_receipt", {})
    if (receipt.get("version") != VERSION or receipt.get("data_version") != DATA_VERSION
            or receipt.get("test_fixture") is not fixture
            or receipt.get("extraction_config_sha256") != digest(extraction_config())
            or receipt.get("execution_sources") != execution_sources()
            or data.get("collector_receipt_sha256") != digest(receipt)
            or receipt.get("data_sha256") != data["data_sha256"]
            or receipt.get("actual_joint_steps") != data["joint_transitions"]
            or receipt.get("actor_training_clock") != data["actor_training_clock"]
            or not compact._same(receipt.get("actor_bindings"), data["actor_bindings"])):
        raise ValueError("Independent shutdown collection provenance differs")
    bindings, metadata = data["actor_bindings"], receipt.get("actor_metadata", {})
    if (not set(REQUIRED_BINDINGS) <= set(bindings)
            or bindings["experiment_version"] != admission.TRAINER_VERSION
            or metadata.get("experiment_version") != admission.TRAINER_VERSION
            or bindings["shutdown_arm"] not in admission.ARMS
            or not compact._same(bindings["own_shutdown_beta"], admission.ARMS[bindings["shutdown_arm"]])
            or metadata.get("feature_names") != data["feature_names"]
            or metadata.get("test_fixture") is not fixture):
        raise ValueError("New shutdown Actor identity must remain attached to the data")
    for key, value in bindings.items():
        if key == "actor_sha256" or (key == "actor_parameters_sha256" and key not in metadata):
            continue
        if not compact._same(metadata.get(key), value):
            raise ValueError("Saved metadata and external binding differ: " + key)
    if "episode" in data and not compact._same(receipt.get("episode"), data["episode"]):
        raise ValueError("Episode/trace receipt differs")


def merge_episodes(episodes, *, allow_test_fixture=False):
    """Use the frozen array merge after verifying every new collector receipt."""
    material = list(episodes)
    if not material:
        raise ValueError("No real episodes to merge")
    for data in material:
        _validate_data(data, allow_test_fixture)
    first = material[0]["collector_receipt"]
    for data in material:
        other = data["collector_receipt"]
        if not compact._same({k: first[k] for k in ("actor_metadata", "weights_sha256", "configuration")},
                             {k: other[k] for k in ("actor_metadata", "weights_sha256", "configuration")}):
            raise ValueError("Merged episodes must have the same actual Actor and physics")
    result = data_api.merge_episodes(material, allow_test_fixture=allow_test_fixture)
    contract = {"metadata": first["actor_metadata"], "weights_sha256": first["weights_sha256"],
        "actor_training_clock": first["actor_training_clock"], "configuration": first["configuration"]}
    return _receipt(result, contract=contract, sources=execution_sources(),
        episodes=[{"episode": deepcopy(d["episode"]), "collector_receipt_sha256": d["collector_receipt_sha256"]}
                  for d in material])


def fit_feedback(train, selection, *, step, feature_names, feedback_config=None,
                 prior_manager_state=None, allow_test_fixture=False):
    """Same real nine fits and gates; never claims independent explanations pass."""
    for data in (train, selection):
        _validate_data(data, allow_test_fixture)
    left, right = train["collector_receipt"], selection["collector_receipt"]
    if not compact._same({k: left[k] for k in ("actor_metadata", "weights_sha256", "configuration")},
                         {k: right[k] for k in ("actor_metadata", "weights_sha256", "configuration")}):
        raise ValueError("Train and selection evidence must retain the same frozen Actor")
    sources = execution_sources()
    result = data_api.fit_feedback(train, selection, step=step, feature_names=feature_names,
        feedback_config=feedback_config, prior_manager_state=prior_manager_state,
        allow_test_fixture=allow_test_fixture)
    if sources != execution_sources():
        raise ValueError("Extraction execution sources changed during fitting")
    provenance = {"version": VERSION, "data_version": DATA_VERSION,
        "actor_bindings": deepcopy(train["actor_bindings"]),
        "actor_metadata": deepcopy(left["actor_metadata"]),
        "train_receipt_sha256": train["collector_receipt_sha256"],
        "selection_receipt_sha256": selection["collector_receipt_sha256"],
        "execution_sources": sources, "extraction_config_sha256": digest(extraction_config())}
    # Keep the manager's actual fit report and evidence digest unchanged. The
    # adapter envelope anchors all new-source fields and the full old result.
    return {**result, "version": VERSION, "data_version": DATA_VERSION,
        "fit_algorithm_version": DATA_VERSION, "algorithm_result_sha256": digest(result),
        "shutdown_extraction": provenance,
        "shutdown_evidence_sha256": digest({"provenance": provenance, "algorithm_result": result}),
        "release_eligible": False}


def collect_episode(actor, scene, *, pool, profile, program_role, seed, episode_id,
                    expected_bindings, protocol, sampling_mode="deterministic", allow_test_fixture=False, config=None):
    """One real episode. Reserve horizon first; persist this return before ack.

    ``pool=selection`` accepts only existing extraction_* scenes. Program
    decisions occur before the NN and are never used as tree training labels.
    Local actor/partner generators are separate from training random streams.
    """
    config = config or collaborative_study_config()
    contract = validate_actor(actor, expected_bindings=expected_bindings, protocol=protocol,
                              allow_test_fixture=allow_test_fixture, config=config)
    split = {"train": "train", "selection": "extraction"}.get(pool)
    if (split is None or not isinstance(scene, dict) or not scene.get("id", "").startswith(split + "_")
            or scene.get("split", split) != split or scene.get("test_fixture", False) is not allow_test_fixture
            or not compact._sha(scene.get("fingerprint"))):
        raise ValueError("Only existing train/extraction development scenes may be sampled")
    if (profile not in PROFILES or type(program_role) is not int
            or program_role not in ((-1,) if profile == "selfplay" else (0, 1))
            or type(seed) is not int or seed < 0 or not isinstance(episode_id, str) or not episode_id
            or sampling_mode not in ("deterministic", "stochastic")):
        raise ValueError("A fixed profile, program role, local seed and episode identity are required")
    sources, original_scene = execution_sources(), digest(scene)
    observations, probabilities, ids, groups, row_sources, trace = [], [], [], [], [], []
    completed = attempts = 0
    with data_api._isolated_host_rng():
        env = PublicFeedbackEnvironment(config, physical.REWARD, collision_cost=.05, mode="observed")
        reset_scenario(env, scene)
        if env.done or env.state.frame != 0 or env.public_history()["valid"]:
            raise ValueError("Extraction begins at the registered raw frame-zero state")
        actor_rng = np.random.default_rng(np.random.SeedSequence([seed, 91]))
        partner_rng = np.random.default_rng(np.random.SeedSequence([seed, 92]))
        try:
            while not env.done:
                before = env.snapshot(); public = env.public_view(); before_hash = digest(before)
                obs = env.observations()
                if any(value.shape != (197,) or value[177] != int(env.state.frame > 0) for value in obs.values()):
                    raise ValueError("History features must come from the actual confirmed previous step")
                program = None if program_role == -1 else partner_action(env, f"robot_{program_role+1}", profile, partner_rng)
                if digest(env.snapshot()) != before_hash: raise ValueError("Program decision changed the sampled state")
                proposed, distribution = actor.act(obs, deterministic=sampling_mode == "deterministic", rng=actor_rng)
                if digest(env.snapshot()) != before_hash: raise ValueError("Actor decision changed the sampled state")
                if set(proposed) != set(obs) or set(distribution) != set(obs): raise ValueError("Incomplete NN action output")
                submitted = dict(proposed)
                if program_role >= 0: submitted[f"robot_{program_role+1}"] = program
                indices, labeled_roles, group_by_role = [], [], {}
                for role, agent in enumerate(env.state.agents):
                    key = agent.agent_id; probability = np.asarray(distribution[key], dtype=np.float32)
                    if (probability.shape != (5,) or not np.isfinite(probability).all() or (probability < 0).any()
                            or not np.isclose(probability.sum(), 1., atol=1e-6) or proposed[key] not in ACTIONS
                            or (sampling_mode == "deterministic" and proposed[key] != ACTIONS[int(probability.argmax())])):
                        raise ValueError("Invalid actual neural action/distribution")
                    group_by_role[key] = critical_groups(env, key)
                    if role == program_role or not agent.active: continue
                    if submitted[key] != proposed[key]: raise ValueError("A neural command was overwritten")
                    indices.append(len(observations)); labeled_roles.append(role)
                    observations.append(np.asarray(obs[key], dtype=np.float32).copy()); probabilities.append(probability.copy())
                    ids.append(episode_id); groups.append(group_by_role[key])
                    row_sources.append({"episode_id": episode_id, "scenario_id": scene["id"],
                        "scenario_fingerprint": scene["fingerprint"], "role": role, "frame": env.state.frame,
                        "public_state_sha256": digest(public), "snapshot_sha256": before_hash,
                        "proposed_action": proposed[key], "neural_argmax": ACTIONS[int(probability.argmax())]})
                if completed >= config.horizon: raise ValueError("Episode exceeded its reserved horizon")
                attempts += 1
                _, actual_rewards, terminated, truncated, info = env.step(submitted)
                completed += 1
                if info["requested_actions"] != submitted: raise ValueError("Physical input differs from submitted commands")
                trace.append(jsonable({"frame": before["state"]["frame"], "before": before, "after": env.snapshot(),
                    "public_before": public, "public_after": env.public_view(), "observations": obs,
                    "probabilities": distribution, "proposed": proposed, "submitted": submitted,
                    "executed": info["executed_actions"], "events": info["events"], "groups": group_by_role,
                    "rewards": actual_rewards, "reward_components": info["reward_components"],
                    "terminated": terminated, "truncated": truncated,
                    "neural_row_indices": indices, "neural_roles": labeled_roles}))
            if digest(scene) != original_scene or execution_sources() != sources:
                raise ValueError("Scene or extraction execution sources changed during sampling")
            final_contract = validate_actor(actor, expected_bindings=expected_bindings, protocol=protocol,
                           allow_test_fixture=allow_test_fixture, config=config)
            if not compact._same(final_contract, contract):
                raise ValueError("Actor contract changed during collection")
        except BaseException as error:
            raise EpisodeSamplingError(str(error), completed_steps=completed, attempted_steps=attempts) from error
    data = {"version": DATA_VERSION, "pool": pool, "actor_bindings": deepcopy(expected_bindings),
        "feature_names": contract["feature_names"], "observations": np.asarray(observations, np.float32).reshape(-1, 197),
        "probabilities": np.asarray(probabilities, np.float32).reshape(-1, 5), "episode_ids": ids, "groups": groups,
        "row_sources": row_sources, "scene_fingerprints": [scene["fingerprint"]], "joint_transitions": completed,
        "teacher_rows_in_fit": 0, "neural_submitted_overrides": 0, "test_fixture": allow_test_fixture,
        "actor_training_clock": contract["actor_training_clock"],
        "extraction_config_sha256": digest(data_api.extraction_config()), "trace": trace,
        "episode": {"id": episode_id, "scenario_id": scene["id"], "fingerprint": scene["fingerprint"],
            "profile": profile, "program_role": program_role, "seed": seed, "sampling_mode": sampling_mode,
            "horizon": config.horizon, "actual_joint_steps": completed, "trace_sha256": digest(trace)}}
    data["data_sha256"] = data_api._data_digest(data)
    return _receipt(data, contract=contract, sources=sources)
