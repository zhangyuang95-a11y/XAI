from core.policy_contracts import ActionDistribution
from backend.simulation import query_engine, query_evidence


def test_query_engine_uses_shared_distribution_evidence_boundary() -> None:
    """The real explanation path must import its extracted evidence helper."""

    distribution = ActionDistribution(
        agent_id="robot_2",
        actions=("WAIT", "UP"),
        probabilities=(0.25, 0.75),
        logits=(-1.0, 1.0),
        action_mask=(1.0, 1.0),
        proposed_action="UP",
    )

    assert query_engine.distribution_evidence is query_evidence.distribution_evidence
    assert query_engine.distribution_evidence(distribution) == {
        "actions": ("WAIT", "UP"),
        "raw_logits": (-1.0, 1.0),
        "action_mask": (1.0, 1.0),
        "masked_probabilities": {"WAIT": 0.25, "UP": 0.75},
        "proposed_action": "UP",
        "argmax_action": "UP",
    }
