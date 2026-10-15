"""The preserved five-dish Kitchen model and participant reward promise agree."""

from domains.kitchen import engine_v5_frozen
from study_v3 import rewards


def test_formal_kitchen_menu_and_reward_policy_remain_explicitly_versioned():
    rules = engine_v5_frozen.rule_metadata()
    assert rules["orders_per_task"] == 5
    assert rules["max_turns"] == 360
    assert rules["order_deadlines"] == [100, 140, 240, 280, 360]
    assert rules["score"] == {
        "served": 100, "step": -1,
        "single_component_discard": -5,
        "combined_dish_discard": -20,
    }
    payment = rewards.policy("kitchen")
    assert payment["version"] == "gbp-300-base-20-per-task-v2"
    assert (payment["score_lower"], payment["score_upper"]) == (60, 120)
    assert rewards.task_bonus(60, payment) == 0
    assert rewards.task_bonus(90, payment) == 10
    assert rewards.task_bonus(120, payment) == 20
    # The model's unpaid demonstration may exceed the capped bonus range.
    assert rewards.task_bonus(198, payment) == 20
