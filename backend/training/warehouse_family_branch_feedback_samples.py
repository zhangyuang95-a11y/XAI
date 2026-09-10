"""Two-role legal branch queries for a future, separately versioned KL learner.

This component does not run PPO, fit a tree, select a checkpoint or grant an
explanation gate. Its caller supplies a durable event sink and registers the
training-only source snapshots. A failed call must not be automatically retried:
attempted and completed query/physics counts are kept separately on the error.
Neither an intervention action nor a program prediction is a PPO action label.
"""
from copy import deepcopy
from hashlib import sha256

import numpy as np

from backend import warehouse_runtime_family as registry
from backend.training.warehouse_native_common import digest
from backend.training.warehouse_family_explanation_audit import physical_projection
from env.warehouse_native.policy import ACTIONS

VERSION = "warehouse-family-two-role-branch-feedback-samples.v1"
ROLES = ("robot_1", "robot_2")


class BranchSamplingError(RuntimeError):
    """Partial execution is evidence, not permission to repeat the operation."""

    def __init__(self, message, *, counts, phase):
        super().__init__(message)
        self.counts = deepcopy(counts)
        self.phase = phase


def collect_branch_pairs(runtime, live_env, *, record_event, allow_test_fixture=False):
    """Return all nonterminal WAIT-versus-direction pairs for both actual roles.

    At most 10 isolated joint steps and 11 two-role NN queries per source frame.
    The target role's current command is fixed before choosing the other role's
    intervention. Each following query sees the real resulting public history.
    ``record_event`` must acknowledge each event by returning exactly True;
    execution starts only after its corresponding before event is acknowledged.
    Tree agreement and NN action changes never filter the training pairs.
    """
    if not callable(record_event):
        raise ValueError("An explicit event sink is required")
    identity = registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    runtime._check_environment(live_env)
    if live_env.done:
        raise ValueError("round_ended")
    if tuple(live_env.agent_ids) != ROLES or len(live_env.feature_names) != 197:
        raise ValueError("Both actual roles and the complete observed197 input are required")
    before = deepcopy(live_env.snapshot())
    before_sha = digest(before)
    counts = dict(environment_steps_attempted=0, environment_steps=0,
        neural_queries_attempted=0, neural_queries=0, neural_rows=0,
        environment_clones=0, terminal_branches=0, pairs=0,
        ppo_steps=0, optimizer_updates=0, tree_fits=0)
    phase, event_index = "begin", 0

    def emit(kind, **payload):
        nonlocal event_index, phase
        phase = kind
        event = dict(version=VERSION, index=event_index, kind=kind,
            before_sha256=before_sha, counts=deepcopy(counts), **deepcopy(payload))
        event_index += 1
        if record_event(event) is not True:
            raise ValueError("Event sink did not acknowledge " + kind)

    def query(env, purpose):
        emit("before_query", purpose=purpose, snapshot_sha256=digest(env.snapshot()))
        counts["neural_queries_attempted"] += 1
        actions, decision = runtime.decision(env)
        counts["neural_queries"] += 1
        counts["neural_rows"] += 2
        observations = {role: np.asarray(env.observations()[role], dtype=np.float32).copy()
                        for role in ROLES}
        for role in ROLES:
            values = observations[role]
            probabilities = np.asarray(decision["probabilities"][role], dtype=np.float32)
            if (values.shape != (197,) or not np.isfinite(values).all()
                    or sha256(values.tobytes()).hexdigest() != decision["observation_hashes"][role]
                    or probabilities.shape != (5,) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)
                    or actions[role] != ACTIONS[int(probabilities.argmax())]):
                raise ValueError("Actual neural query/observation binding differs")
        if decision["post_policy_overrides"] != 0 or decision["masks"] is not False:
            raise ValueError("A branch query must retain all neural actions")
        saved = {role: values.tolist() for role, values in observations.items()}
        emit("after_query", purpose=purpose, decision=decision, observations=saved)
        return deepcopy(actions), deepcopy(decision), saved

    try:
        emit("begin", scope="training_branch_query_component", snapshot=before,
            actor_sha256=runtime.actor_sha256, runtime_signature=identity["runtime_signature"],
            feature_names=list(live_env.feature_names),
            maximum_environment_steps=10, maximum_neural_queries=11,
            test_fixture=allow_test_fixture)
        base_actions, base_decision, base_observations = query(live_env, "base")
        branches, pairs, excluded_pairs = [], [], []
        for target in ROLES:
            other = next(role for role in ROLES if role != target)
            indices = {}
            for action in ACTIONS:
                branch = runtime.from_snapshot(before)
                counts["environment_clones"] += 1
                if digest(branch.snapshot()) != before_sha:
                    raise ValueError("Isolated branch does not preserve the complete before state")
                submitted = {target: base_actions[target], other: action}
                context = dict(target_role=target, other_role=other, other_action=action,
                    target_neural_command=base_actions[target], submitted_actions=submitted,
                    branch_index=len(branches))
                emit("before_step", **context)
                counts["environment_steps_attempted"] += 1
                _, rewards, terminated, truncated, info = branch.step(submitted)
                counts["environment_steps"] += 1
                if info["requested_actions"] != submitted:
                    raise ValueError("Branch physical resolver changed the submitted commands")
                after = deepcopy(branch.snapshot())
                done = bool(terminated or truncated)
                counts["terminal_branches"] += int(done)
                emit("after_step", **context, after=after, info=info, rewards=rewards, done=done)
                decision, observations = None, None
                if not done:
                    _, decision, observations = query(branch, f"{target}:{action}")
                indices[action] = len(branches)
                branches.append(dict(**context, after=after, done=done,
                    next_decision=decision, observations=observations,
                    physical_sha256=digest(physical_projection(after["state"]))))
            wait = branches[indices["WAIT"]]
            for action in ACTIONS[:-1]:
                altered = branches[indices[action]]
                common = dict(target_role=target, other_role=other, other_action=action,
                    branch_indices=[indices["WAIT"], indices[action]])
                if wait["done"] or altered["done"]:
                    excluded_pairs.append(dict(**common, reason="terminal_endpoint",
                        terminal_endpoints=[wait["done"], altered["done"]]))
                    continue
                endpoints = [wait, altered]
                nn_actions = [item["next_decision"]["policy_actions"][target] for item in endpoints]
                pairs.append(dict(**common,
                    observations=[item["observations"][target] for item in endpoints],
                    neural_probabilities=[item["next_decision"]["probabilities"][target] for item in endpoints],
                    neural_actions=nn_actions,
                    physical_effect=wait["physical_sha256"] != altered["physical_sha256"],
                    nn_changed=nn_actions[0] != nn_actions[1],
                    training_weight=1.0, program_labels=False, ppo_action_labels=False))
        counts["pairs"] = len(pairs)
        if digest(live_env.snapshot()) != before_sha:
            raise ValueError("Branch collection changed the live state, history or RNG")
        registry.verify(runtime, allow_test_fixture=allow_test_fixture,
                        expected_signature=identity["runtime_signature"])
        result = dict(version=VERSION, scope="training_branch_query_component",
            actor_sha256=runtime.actor_sha256, runtime_signature=identity["runtime_signature"],
            before_sha256=before_sha, base_decision=base_decision,
            base_observations=base_observations, branches=branches, pairs=pairs,
            excluded_pairs=excluded_pairs, counts=deepcopy(counts),
            pair_selection="all nonterminal WAIT-versus-direction endpoints; no fidelity/change filter",
            feature_names=list(live_env.feature_names), test_fixture=allow_test_fixture,
            qualification_granted=False, connected_to_training=False)
        emit("complete", result_sha256=digest(result), pair_count=len(pairs))
        return result
    except Exception as exc:
        failed_phase = phase
        try:
            emit("failed", failed_phase=failed_phase, error_type=type(exc).__name__, error=str(exc),
                 live_state_unchanged=digest(live_env.snapshot()) == before_sha, retry_allowed=False)
        except Exception:
            pass  # The caller still receives the non-refunded partial counts.
        raise BranchSamplingError(str(exc), counts=counts, phase=failed_phase) from exc
