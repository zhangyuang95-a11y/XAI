"""Fixed observed197 validation for the paired own-shutdown reward contrast.

Only training differs between beta0 and beta1. This evaluator always executes
original shared r1 reward/score physics and unmodified neural commands. The new
identity retains the actual Continuation source, clock, protocol and each arm;
it never relabels a shutdown Actor as an older trainer. No final pool or PT load.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from backend.training.warehouse_native_common import digest
from backend.training import warehouse_native_continuation_evaluation as continuation
from backend.training import warehouse_native_partner_mix_evaluation as compact
from backend.training.warehouse_native_compact_validation import execute_episodes
from backend.training import warehouse_native_public_feedback_evaluation as physical
from backend.training.warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import NumPyNativeActor, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION

VERSION = "warehouse-native-own-shutdown-evaluation.v1"
TRAINER_VERSION = "warehouse-native-own-shutdown-trainer.v1"
PROTOCOL_VERSION = "warehouse-native-own-shutdown-protocol.v1"
CREDIT_REWARD_VERSION = "warehouse-native-centered-delivery-credit.v1"
PARTNERS = physical.PARTNERS
ROOT = Path(__file__).resolve().parents[2]
SHUTDOWN_REWARD_VERSION = "warehouse-native-own-shutdown-reward.v1"
ARMS = {"beta0": 0., "beta1": 1.}
REQUIRED_BINDINGS = (*continuation.REQUIRED_BINDINGS, "shutdown_arm", "own_shutdown_beta")


def execution_sources():
    result = continuation.execution_sources()
    result[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def _gates(protocol):
    return continuation._gates(protocol)


def _weights_digest(actor):
    if type(actor) is not NumPyNativeActor:
        return None  # Explicit lightweight fixture double; never production.
    return digest({name: {"shape": list(value.shape), "dtype": value.dtype.str,
        "sha256": sha256(value.tobytes(order="C")).hexdigest()}
        for name, value in sorted(actor.weights.items())})


def _contract(actor, protocol, expected, fixture, config):
    if (type(fixture) is not bool or type(protocol) is not dict or protocol.get("version") != PROTOCOL_VERSION
            or protocol.get("test_fixture", False) is not fixture):
        raise ValueError("An explicit own-shutdown protocol is required")
    if not fixture and type(actor) is not NumPyNativeActor:
        raise ValueError("Production evaluation requires the actual NumPy Actor")
    if type(expected) is not dict or not set(REQUIRED_BINDINGS) <= set(expected):
        raise ValueError("Independent shutdown Actor bindings are required")
    for key in REQUIRED_BINDINGS:
        if key.endswith("sha256") and not compact._sha(expected[key]): raise ValueError("Malformed expected binding")
    source, ev = protocol.get("source", {}), protocol.get("evaluation", {})
    endpoints = ev.get("checkpoints_ppo_steps")
    cap = protocol.get("budget", {}).get("maximum_ppo_joint_steps_per_arm")
    if (type(cap) is not int or cap <= 0 or type(endpoints) is not list or not endpoints
            or any(type(v) is not int or not 0 < v <= cap for v in endpoints)
            or endpoints != sorted(set(endpoints)) or endpoints[-1] != cap):
        raise ValueError("A finite cap and ordered fixed endpoints ending at that cap are required")
    step, branch, cycle = expected["joint_steps"], expected["branch"], expected["cycle_id"]
    arm, beta = expected["shutdown_arm"], expected["own_shutdown_beta"]
    if (arm not in ARMS or type(beta) not in (int, float) or not compact._same(beta, ARMS[arm])
            or not compact._same(protocol.get("shutdown_arms"), list(ARMS))
            or not compact._same(protocol.get("betas_by_arm"), ARMS)):
        raise ValueError("The fixed beta0/beta1 contrast differs")
    if not fixture and (cap != 250000 or endpoints != [50000, 250000]):
        raise ValueError("Production shutdown validation must use both registered 50k/250k endpoints")
    if (protocol.get("budget", {}).get("maximum_ppo_joint_steps") != cap
            or protocol["budget"].get("requested_total_ppo_joint_steps") != 2*cap
            or protocol.get("own_shutdown_reward_version") != SHUTDOWN_REWARD_VERSION):
        raise ValueError("The finite paired shutdown contract differs")
    if (expected["protocol_sha256"] != digest(protocol) or expected["experiment_version"] != TRAINER_VERSION
            or branch != "own_credit" or branch != source.get("branch") or branch != protocol.get("branch")
            or type(step) is not int or step not in endpoints or not isinstance(cycle, str) or not cycle
            or cycle != protocol.get("cycle_id")):
        raise ValueError("Shutdown source branch, cycle or fixed endpoint differs")
    metadata = getattr(actor, "metadata", None)
    if type(metadata) is not dict or not callable(getattr(actor, "act", None)):
        raise ValueError("A bound Actor is required")
    alpha = protocol.get("delivery_credit_alpha")
    if type(alpha) not in (int, float) or alpha != .5: raise ValueError("The inherited own-credit alpha must remain .5")
    required = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": 197, "state_dim": 354, "hidden": 128, "architecture": "two_hidden_layer_tanh",
        "actions": list(physical.ACTIONS), "action_masks": False, "runtime_action_override": False,
        "public_feedback_mode": "observed", "public_feedback_version": physical.OBSERVER_VERSION,
        "feature_names": list(observation_names(config)) + list(HISTORY_FEATURE_NAMES),
        "delivery_credit_alpha": alpha, "credit_reward_version": CREDIT_REWARD_VERSION, "test_fixture": fixture,
        "shutdown_arm": arm, "own_shutdown_beta": beta, "own_shutdown_reward_version": SHUTDOWN_REWARD_VERSION}
    for key, value in required.items():
        if not compact._same(metadata.get(key), value): raise ValueError("Actor own-shutdown contract differs: " + key)
    for key, value in expected.items():
        got = getattr(actor, "artifact_sha256", None) if key == "actor_sha256" else metadata.get(key)
        if not compact._same(got, value): raise ValueError("Actor binding differs: " + key)
    inherited = metadata.get("source_counters", {}).get("joint_steps")
    if (type(inherited) is not int or inherited < 0 or type(source.get("cumulative_joint_steps")) is not int
            or inherited != source["cumulative_joint_steps"] or type(source.get("joint_steps")) is not int
            or not 0 <= source["joint_steps"] <= inherited
            or source.get("checkpoint_sha256") != expected["source_checkpoint_sha256"]
            or source.get("state_sha256") != expected["initialization_sha256"]):
        raise ValueError("The explicit inherited source counters or checkpoint differ")
    if (source.get("trainer_version") != continuation.TRAINER_VERSION
            or (not fixture and (inherited != 2230000 or source["joint_steps"] != 1000000))):
        raise ValueError("The registered actual Continuation source differs")
    lineage = metadata.get("source_lineage")
    if not isinstance(lineage, list) or not lineage or not all(isinstance(item, dict) for item in lineage):
        raise ValueError("An explicit inherited source lineage is required")
    if not compact._same(lineage, protocol.get("source_lineage")) or not compact._same(lineage[-1], source):
        raise ValueError("The retained source lineage differs from its protocol")
    previous = None
    for item in lineage:
        if (item.get("trainer_version") not in ("warehouse-native-delivery-credit-trainer.v1", continuation.TRAINER_VERSION)
                or item.get("branch") != branch
                or any(not compact._sha(item.get(key)) for key in ("checkpoint_sha256", "state_sha256", "protocol_sha256", "source_sha256"))
                or type(item.get("joint_steps")) is not int or type(item.get("cumulative_joint_steps")) is not int
                or not 0 <= item["joint_steps"] <= item["cumulative_joint_steps"]
                or (previous is not None and previous + item["joint_steps"] != item["cumulative_joint_steps"])):
            raise ValueError("Source lineage identity or cumulative accounting differs")
        previous = item["cumulative_joint_steps"]
    original = protocol.get("source_protocol", {})
    if original.get("version") != continuation.PROTOCOL_VERSION or digest(original) != source["protocol_sha256"]:
        raise ValueError("The immediate inherited protocol hash differs")
    gates = _gates(protocol)
    if not compact._same(protocol.get("reward"), physical.REWARD) or protocol.get("collision_training_cost") != .05:
        raise ValueError("Validation retains the original shared r1 reward")
    if (ev.get("partners") != list(PARTNERS) or ev.get("scenarios_per_partner") != 50
            or ev.get("deterministic") is not True or ev.get("read_final_test") is not False
            or ev.get("horizon") != 120 or ev.get("training_credit_in_evaluation") is not False
            or ev.get("maximum_environment_steps") != 2*len(endpoints)*3*50*120
            or ev.get("training_shutdown_in_evaluation") is not False
            or ev.get("reuse_source_zero_step_validation") is not True):
        raise ValueError("Only the original fixed deterministic validation matrix is allowed")
    if (not fixture and config != collaborative_study_config()) or type(config.horizon) is not int or not 1 <= config.horizon <= 120:
        raise ValueError("The original validation environment cannot change")
    return deepcopy(metadata), gates, inherited


def evaluate(actor_or_path, scenarios, protocol, output, step_budget, *, expected_bindings,
             reference_report, random_report, before_episode=None, on_episode=None,
             confirmed_operation_ids=None, allow_test_fixture=False, config=None):
    """Same durable episode callbacks as partner-mix; a new explicit contract.

    A saved complete episode can only be skipped when its operation is already
    confirmed by the caller's retained ledger. Pending reservations never rerun.
    """
    if type(step_budget) is not int or step_budget < 0: raise ValueError("Integer step budget required")
    if any(c is not None and not callable(c) for c in (before_episode, on_episode)):
        raise ValueError("Episode callbacks must be callable")
    if confirmed_operation_ids is not None and (not isinstance(confirmed_operation_ids, (set, frozenset, tuple, list))
            or any(not isinstance(v, str) for v in confirmed_operation_ids)):
        raise ValueError("Confirmed IDs must come from the retained ledger")
    if type(allow_test_fixture) is not bool:
        raise ValueError("An explicit fixture boundary is required")
    if (reference_report is None) != (random_report is None) or (not allow_test_fixture and reference_report is None):
        raise ValueError("Both frozen reference/random baseline records are required")
    config = config or collaborative_study_config()
    actor_path = Path(actor_or_path).resolve() if isinstance(actor_or_path, (str, Path)) else None
    actor = NumPyNativeActor(actor_path) if actor_path is not None else actor_or_path
    metadata, gates, inherited = _contract(actor, protocol, expected_bindings, allow_test_fixture, config)
    if actor_path is None and type(actor) is NumPyNativeActor: actor_path = Path(actor.path).resolve()
    scenes = physical._scenarios(scenarios, allow_test_fixture, config)
    reference = physical._baseline(reference_report, scenes) if reference_report is not None else None
    random = physical._baseline(random_report, scenes) if random_report is not None else None
    sources, weights_sha256 = execution_sources(), _weights_digest(actor)
    identity = {"version": VERSION, "actor_bindings": deepcopy(expected_bindings), "actor_metadata_sha256": digest(metadata),
        "validation_entries_sha256": digest(scenes), "protocol_sha256": digest(protocol), "sources": sources,
        "configuration": asdict(config), "episode_count": len(scenes)*len(PARTNERS), "step_budget": step_budget,
        "reference_sha256": digest(reference) if reference is not None else None,
        "random_sha256": digest(random) if random is not None else None, "test_fixture": allow_test_fixture,
        "validation_reward": "original_shared_r1", "training_delivery_credit_alpha": metadata["delivery_credit_alpha"], "cycle_id": protocol["cycle_id"],
        "shutdown_arm": expected_bindings["shutdown_arm"], "training_own_shutdown_beta": expected_bindings["own_shutdown_beta"]}
    contexts = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            context = {"evaluation_version": VERSION, "mode": "observed", "branch": expected_bindings["branch"], "cycle_id": protocol["cycle_id"],
                "shutdown_arm": expected_bindings["shutdown_arm"], "own_shutdown_beta": expected_bindings["own_shutdown_beta"],
                "partner": partner, "scenario_id": scene["id"], "scenario_index": index,
                "initial_fingerprint": scene["fingerprint"], "episode_index": len(contexts), "seed": 17000+index,
                "horizon": config.horizon, "maximum_environment_steps": config.horizon,
                "test_fixture": allow_test_fixture, "actor_bindings": deepcopy(expected_bindings)}
            context["operation_id"] = digest({"evaluation": digest(identity), "episode": context})
            contexts.append(context)
    def unchanged():
        if (execution_sources() != sources or not compact._same(actor.metadata, metadata)
                or _weights_digest(actor) != weights_sha256):
            raise ValueError("Shutdown evaluation sources, Actor metadata or loaded weights changed")
        if actor_path is not None and sha256(actor_path.read_bytes()).hexdigest() != expected_bindings["actor_sha256"]:
            raise ValueError("Frozen shutdown Actor bytes changed")
    rows, manifest = execute_episodes(actor, scenes, config=config, contexts=contexts, identity=identity,
        output=output, step_budget=step_budget,
        dynamics={"reward": deepcopy(protocol["reward"]), "initialization": {"source_joint_steps": inherited}},
        validate_unchanged=unchanged, before_episode=before_episode, on_episode=on_episode,
        confirmed_operation_ids=confirmed_operation_ids)
    unchanged()
    # Reuse the pure row statistics/gates, then attach this experiment identity.
    report = compact._summary(rows, metadata, expected_bindings, gates, reference, random,
        allow_test_fixture, identity, manifest["reserved_environment_steps"])
    report.update(version=VERSION, validation_reward="original_shared_r1",
        training_delivery_credit_alpha=metadata["delivery_credit_alpha"],
        remaining_reservable_steps=step_budget-manifest["reserved_environment_steps"], cycle_id=protocol["cycle_id"],
        shutdown_arm=expected_bindings["shutdown_arm"], training_own_shutdown_beta=expected_bindings["own_shutdown_beta"])
    compact._put(Path(output).expanduser().resolve() / "report.json", report)
    return report
