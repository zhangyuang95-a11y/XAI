"""Observed197 RCPD data collection and post-hoc fitting; no training or CLI.

Callers reserve a complete episode before collect_episode, durably save its
arrays and trace after return, then acknowledge the external ledger. This
module neither reserves a budget nor silently retries a failed episode.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import random

import numpy as np

from backend.training.warehouse_native_common import canonical, digest, jsonable
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment, HISTORY_FEATURE_NAMES
from backend.training import warehouse_native_public_feedback_evaluation as physical
from backend.training import warehouse_native_partner_mix_evaluation as compact
from core.program import ExecutableProgram
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager
from env.warehouse_native.observations import observation_names
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS, NumPyNativeActor, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION
from env.warehouse_native.scenarios import reset_scenario

VERSION = "warehouse-native-observed197-rcpd.v1"
PROFILES = ("selfplay", "skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
ROOT = Path(__file__).resolve().parents[2]
ACTOR_VERSIONS = ("warehouse-native-delivery-credit-trainer.v1", "warehouse-native-continuation-trainer.v1",
                  "warehouse-native-continuation-feedback-trainer.v1")
REQUIRED_BINDINGS = (*compact.REQUIRED_BINDINGS, "actor_parameters_sha256")


def extraction_config():
    """Hash this new contract in the outer preparation; old gates are unmodified."""
    return {"version": VERSION, "observation_size": 197, "public_feedback_mode": "observed",
        "training_pool": "train", "selection_pool": "extraction", "profiles": list(PROFILES),
        "depths": [4, 6, 8], "leaves": [16, 32, 64], "minimum_fidelity": .90,
        "minimum_critical_fidelity": .85, "minimum_critical_scenarios": 10,
        "minimum_critical_non_wait_fidelity": .85, "require_critical_non_wait_examples": True,
        "critical_groups": list(GROUPS), "overlap_policy": "reject_without_removing_rows",
        "teacher_rows_in_fit": 0, "explanation_qualified": False, "intervention_direction_tested": False}


def execution_sources():
    result = compact.execution_sources()
    result[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__).read_bytes()).hexdigest()
    return result


@contextmanager
def _isolated_host_rng():
    state = random.getstate(), np.random.get_state()
    try: yield
    finally: random.setstate(state[0]); np.random.set_state(state[1])


def validate_actor(actor, *, expected_bindings, protocol, allow_test_fixture=False, config=None):
    """Zero-forward preflight. Parameter semantics are an external checkpoint anchor."""
    fixture = allow_test_fixture
    config = config or collaborative_study_config()
    if type(fixture) is not bool or protocol.get("test_fixture", False) is not fixture:
        raise ValueError("Explicit matching fixture provenance is required")
    if not fixture and type(actor) is not NumPyNativeActor: raise ValueError("Only an actual NumPy Actor may label production data")
    if type(expected_bindings) is not dict or not set(REQUIRED_BINDINGS) <= set(expected_bindings):
        raise ValueError("Actor artifact, parameter and experiment bindings are required")
    for key in REQUIRED_BINDINGS:
        if key.endswith("sha256") and not compact._sha(expected_bindings[key]): raise ValueError("Invalid external hash binding")
    metadata = getattr(actor, "metadata", {})
    if not isinstance(metadata, dict) or not callable(getattr(actor, "act", None)):
        raise ValueError("A metadata-bound Actor is required")
    if (metadata.get("test_fixture", False) is not fixture or metadata.get("experiment_version") not in ACTOR_VERSIONS
            or expected_bindings["protocol_sha256"] != digest(protocol)):
        raise ValueError("Actor/protocol identity differs")
    names = list(observation_names(config)) + list(HISTORY_FEATURE_NAMES)
    required = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": 197, "state_dim": 354, "hidden": 128, "architecture": "two_hidden_layer_tanh",
        "actions": list(ACTIONS), "action_masks": False, "runtime_action_override": False,
        "public_feedback_mode": "observed", "public_feedback_version": physical.OBSERVER_VERSION,
        "feature_names": names}
    for key, value in required.items():
        if not compact._same(metadata.get(key), value): raise ValueError("Observed197 Actor contract differs: " + key)
    for key, value in expected_bindings.items():
        if key == "actor_parameters_sha256" and key not in metadata:
            continue  # The outer source reader binds this semantic Torch-state hash.
        actual = getattr(actor, "artifact_sha256", None) if key == "actor_sha256" else metadata.get(key)
        if not compact._same(actual, value): raise ValueError("Frozen Actor binding differs: " + key)
    if (not compact._same(protocol.get("reward"), physical.REWARD) or protocol.get("collision_training_cost") != .05
            or (not fixture and config != collaborative_study_config()) or not 1 <= config.horizon <= 120):
        raise ValueError("Original public extraction physics cannot change")
    if type(actor) is NumPyNativeActor and sha256(actor.path.read_bytes()).hexdigest() != expected_bindings["actor_sha256"]:
        raise ValueError("Frozen Actor bytes changed")
    inherited, additional = metadata.get("source_counters", {}).get("joint_steps"), metadata.get("joint_steps")
    if any(type(value) is not int or value < 0 for value in (inherited, additional)):
        raise ValueError("The Actor's actual cumulative training clock is required")
    return {"metadata": deepcopy(metadata), "feature_names": names, "configuration": asdict(config),
        "actor_training_clock": inherited + additional,
        "extraction_config": extraction_config(), "extraction_config_sha256": digest(extraction_config()),
        "actor_parameters_binding": "externally_verified_checkpoint_semantics_not_reconstructed_from_npz"}


class EpisodeSamplingError(RuntimeError):
    def __init__(self, message, *, completed_steps, attempted_steps):
        super().__init__(message)
        self.actual_joint_steps = completed_steps
        self.environment_step_calls = attempted_steps


def _data_digest(data):
    arrays = {key: {"shape": list(data[key].shape), "dtype": data[key].dtype.str,
                   "sha256": sha256(data[key].tobytes()).hexdigest()} for key in ("observations", "probabilities")}
    selected = {key: data[key] for key in ("version", "pool", "actor_bindings", "feature_names", "episode_ids",
        "groups", "row_sources", "scene_fingerprints", "joint_transitions", "teacher_rows_in_fit",
        "neural_submitted_overrides", "test_fixture", "extraction_config_sha256", "actor_training_clock")}
    return digest({"arrays": arrays, "records": selected})


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
    with _isolated_host_rng():
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
                _, _, _, _, info = env.step(submitted)
                completed += 1
                if info["requested_actions"] != submitted: raise ValueError("Physical input differs from submitted commands")
                trace.append(jsonable({"frame": before["state"]["frame"], "before": before, "after": env.snapshot(),
                    "public_before": public, "public_after": env.public_view(), "observations": obs,
                    "probabilities": distribution, "proposed": proposed, "submitted": submitted,
                    "executed": info["executed_actions"], "events": info["events"], "groups": group_by_role,
                    "neural_row_indices": indices, "neural_roles": labeled_roles}))
            if digest(scene) != original_scene or execution_sources() != sources:
                raise ValueError("Scene or extraction execution sources changed during sampling")
            validate_actor(actor, expected_bindings=expected_bindings, protocol=protocol,
                           allow_test_fixture=allow_test_fixture, config=config)
        except BaseException as error:
            raise EpisodeSamplingError(str(error), completed_steps=completed, attempted_steps=attempts) from error
    data = {"version": VERSION, "pool": pool, "actor_bindings": deepcopy(expected_bindings),
        "feature_names": contract["feature_names"], "observations": np.asarray(observations, np.float32).reshape(-1, 197),
        "probabilities": np.asarray(probabilities, np.float32).reshape(-1, 5), "episode_ids": ids, "groups": groups,
        "row_sources": row_sources, "scene_fingerprints": [scene["fingerprint"]], "joint_transitions": completed,
        "teacher_rows_in_fit": 0, "neural_submitted_overrides": 0, "test_fixture": allow_test_fixture,
        "actor_training_clock": contract["actor_training_clock"],
        "extraction_config_sha256": contract["extraction_config_sha256"], "trace": trace,
        "episode": {"id": episode_id, "scenario_id": scene["id"], "fingerprint": scene["fingerprint"],
            "profile": profile, "program_role": program_role, "seed": seed, "sampling_mode": sampling_mode,
            "horizon": config.horizon, "actual_joint_steps": completed, "trace_sha256": digest(trace)}}
    data["data_sha256"] = _data_digest(data)
    return data


def _validate_data(data, fixture):
    if (data.get("version") != VERSION or data.get("test_fixture") is not fixture
            or data.get("extraction_config_sha256") != digest(extraction_config())
            or data.get("teacher_rows_in_fit") != 0 or data.get("neural_submitted_overrides") != 0):
        raise ValueError("Dataset provenance or extraction contract differs")
    x, y = data["observations"], data["probabilities"]
    if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 197
            or not isinstance(y, np.ndarray) or y.dtype != np.float32 or y.shape != (len(x), 5)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any()
            or not np.allclose(y.sum(-1), 1., atol=1e-5)
            or any(len(data[key]) != len(x) for key in ("episode_ids", "groups", "row_sources"))):
        raise ValueError("Invalid observed197 neural dataset")
    if data.get("data_sha256") != _data_digest(data): raise ValueError("Dataset arrays or provenance changed")
    for index, source in enumerate(data["row_sources"]):
        if (source["episode_id"] != data["episode_ids"][index] or source["role"] not in (0, 1)
                or source["scenario_fingerprint"] not in data["scene_fingerprints"]
                or source["neural_argmax"] != ACTIONS[int(y[index].argmax())]
                or any(group not in GROUPS for group in data["groups"][index])):
            raise ValueError("Row/scene/role/neural label provenance differs")


def merge_episodes(episodes, *, allow_test_fixture=False):
    """Merge after durable per-episode saving; full traces are not duplicated."""
    material = list(episodes)
    if not material: raise ValueError("No real episodes to merge")
    first = material[0]; seen = set()
    for data in material:
        _validate_data(data, allow_test_fixture)
        if not compact._same({k: data[k] for k in ("pool", "actor_bindings", "feature_names")},
                             {k: first[k] for k in ("pool", "actor_bindings", "feature_names")}):
            raise ValueError("Mixed source Actors or pools cannot be merged")
        episode = data["episode"]
        if episode["id"] in seen or digest(data["trace"]) != episode["trace_sha256"]:
            raise ValueError("Duplicate episode identity or changed trace")
        seen.add(episode["id"])
    merged = {key: deepcopy(first[key]) for key in ("version", "pool", "actor_bindings", "feature_names",
        "teacher_rows_in_fit", "neural_submitted_overrides", "test_fixture", "extraction_config_sha256", "actor_training_clock")}
    for key in ("observations", "probabilities"): merged[key] = np.concatenate([d[key] for d in material], axis=0)
    for key in ("episode_ids", "groups", "row_sources"): merged[key] = [deepcopy(x) for d in material for x in d[key]]
    merged["scene_fingerprints"] = sorted({v for d in material for v in d["scene_fingerprints"]})
    merged["joint_transitions"] = sum(d["joint_transitions"] for d in material)
    merged["episodes"] = [{**deepcopy(d["episode"]), "data_sha256": d["data_sha256"]} for d in material]
    merged["data_sha256"] = _data_digest(merged)
    return merged


def _fidelity(manager, data):
    x, p = data["observations"], data["probabilities"]
    predicted = manager._predict(manager.program, x).argmax(-1)
    neural = p.argmax(-1); agree = predicted == neural
    def subset(indices):
        return {"rows": len(indices), "fidelity": float(agree[indices].mean()) if len(indices) else None}
    groups = {}
    for name in GROUPS:
        indices = [i for i, values in enumerate(data["groups"]) if name in values]
        movement = [i for i in indices if ACTIONS[int(neural[i])] != "WAIT"]
        groups[name] = {**subset(indices), "scenarios": len({data["row_sources"][i]["scenario_fingerprint"] for i in indices}),
            "non_wait": subset(movement), "action_counts": {a: sum(neural[i] == j for i in indices) for j, a in enumerate(ACTIONS)}}
    return jsonable({"overall": subset(list(range(len(x)))), "critical": groups,
        "by_action": {a: subset(np.flatnonzero(neural == i)) for i, a in enumerate(ACTIONS)},
        "non_wait": subset(np.flatnonzero(neural != ACTIONS.index("WAIT"))),
        "by_role": {str(role): subset([i for i, row in enumerate(data["row_sources"]) if row["role"] == role]) for role in (0, 1)}})


def fit_feedback(train, selection, *, step, feature_names, feedback_config=None,
                 prior_manager_state=None, allow_test_fixture=False):
    """Actual RCPD search; step uses the cumulative neural training clock.

    Selection is repeated development evidence, never independent intervention
    evidence. Overlap is reported/rejected, with no observation or WAIT removal.
    """
    if type(step) is not int or step < 0 or type(allow_test_fixture) is not bool:
        raise ValueError("A cumulative fit step and explicit fixture mode are required")
    if list(feature_names) != list(observation_names(collaborative_study_config())) + list(HISTORY_FEATURE_NAMES):
        raise ValueError("Fitting must retain the complete actual197 public feature names")
    config = feedback_config or FeedbackConfig()
    if (config.depths != (4, 6, 8) or config.leaves != (16, 32, 64)
            or config.minimum_fidelity != .9 or config.minimum_critical_fidelity != .85
            or config.critical_groups != GROUPS):
        raise ValueError("The fixed nine-candidate RCPD reliability search cannot change")
    manager = FeedbackManager(feature_names, config)
    if prior_manager_state is not None: manager.load_state_dict(prior_manager_state)
    manager.current_lambda = 0.; manager.reliable = False
    for data in (train, selection): _validate_data(data, allow_test_fixture)
    if (train["pool"] != "train" or selection["pool"] != "selection"
            or not compact._same(train["actor_bindings"], selection["actor_bindings"])
            or step != train["actor_training_clock"] or step != selection["actor_training_clock"]
            or list(feature_names) != train["feature_names"] or list(feature_names) != selection["feature_names"]):
        raise ValueError("The same frozen197 Actor and distinct named pools are required")
    overlap = {"scene_fingerprints": len(set(train["scene_fingerprints"]) & set(selection["scene_fingerprints"])),
        "episode_ids": len(set(train["episode_ids"]) & set(selection["episode_ids"])),
        "public_states": len({s["public_state_sha256"] for s in train["row_sources"]} & {s["public_state_sha256"] for s in selection["row_sources"]}),
        "exact_observations": len({x.tobytes() for x in train["observations"]} & {x.tobytes() for x in selection["observations"]})}
    if any(overlap.values()): raise ValueError("Extraction overlap rejected without deleting rows: " + canonical(overlap))
    actor_sha = train["actor_bindings"]["actor_sha256"]
    binding = {"config_sha256": digest(extraction_config()), "training_data_sha256": train["data_sha256"],
        "selection_data_sha256": selection["data_sha256"], "actor_sha256": actor_sha,
        "actor_parameters_sha256": train["actor_bindings"]["actor_parameters_sha256"], "cumulative_fit_step": step}
    try:
        with _isolated_host_rng():
            base = manager.fit(train["observations"], train["probabilities"], selection["observations"], selection["probabilities"],
                step=step, source_actor_sha256=actor_sha, train_episode_ids=train["episode_ids"], val_episode_ids=selection["episode_ids"],
                train_groups=train["groups"], val_groups=selection["groups"])
        metrics = _fidelity(manager, selection)
        extra = {name: {"scene_coverage": item["scenarios"] >= 10,
            "movement_evidence": item["non_wait"]["rows"] > 0,
            "movement_fidelity": item["non_wait"]["fidelity"] is not None and item["non_wait"]["fidelity"] >= .85}
            for name, item in metrics["critical"].items()}
        reliable = bool(base["reliable"] and all(all(item.values()) for item in extra.values()))
        fit = {**deepcopy(base), "rcpd_version": VERSION, "reliable": reliable,
            "base_manager_report": deepcopy(base), "base_manager_reliable": base["reliable"],
            "additional_coverage_gate": extra, "selection_metrics": metrics, "observed197_bindings": binding,
            "extraction_config": extraction_config(), "test_fixture": allow_test_fixture,
            "explanation_qualified": False, "intervention_direction_not_tested_here": True,
            "rows_removed": 0, "overlap": overlap}
        manager.reliable, manager.current_lambda, manager.last_fit_report = reliable, 0., fit
        meta = deepcopy(dict(manager.program.metadata))
        meta["observed197_bindings"] = binding
        meta["metrics"] = {**meta.get("metrics", {}), "base_manager_reliable": base["reliable"],
            "feedback_eligible": reliable, "reliable": reliable, "explanation_eligible": False,
            "additional_coverage_gate": extra}
        if set(manager.program.feature_names) != set(feature_names):
            raise ValueError("The extracted program does not retain all197 actual observation names")
        # Generic RCPD sorts its schema. Named predicates are unchanged; use the
        # Actor's explicit order in the serialized schema consumed by the trainer.
        meta["observed197_feature_order"] = "actor_input_order_named_predicates_unchanged"
        manager.program = ExecutableProgram(manager.program.action_names, tuple(feature_names), manager.program.root, meta)
    except (ValueError, RuntimeError) as error:
        manager.reliable, manager.current_lambda = False, 0.
        fit = {"rcpd_version": VERSION, "reliable": False, "reason": str(error), "step": step,
            "source_actor_sha256": actor_sha, "observed197_bindings": binding,
            "explanation_qualified": False, "intervention_direction_not_tested_here": True, "test_fixture": allow_test_fixture}
        manager.last_fit_report = fit
    return {"version": VERSION, "reliable": manager.reliable, "fit_report": jsonable(fit),
        "manager_state": manager.state_dict(), "program": manager.program.to_dict() if manager.program else None,
        "evidence_sha256": digest({"binding": binding, "fit_report": jsonable(fit)}),
        "actual_joint_steps": 0, "neural_training_updates": 0, "test_fixture": allow_test_fixture,
        "explanation_qualified": False}
