"""Non-invasive regression checks for the r4.1 energy admission metric."""

from backend.training import warehouse_r4_active_evaluation as evaluation


def _summary(**updates):
    value = {
        "active_non_wait_rate": .90,
        "productive_action_rate": .75,
        "full_battery_non_charger_wait_rate": 0.,
        "first_productive_latency_median": 1.,
        "first_productive_latency_p90": 1.,
        "ai_delivery_share": .50,
        "no_task_progress_streak_p95": 10.,
        "collision_cancellation_rate": .02,
        "mean_longest_collision_streak": 1.,
        "static_wall_command_rate": 0.,
        "shutdown_count": 0,
        "ai_shutdown_count": 0,
        "player_shutdown_count": 0,
        "action_override_count": 0,
        "mean_ai_deliveries": 5.,
    }
    value.update(updates)
    return value


def test_zero_shutdown_gate_is_scoped_to_the_neural_robot():
    """A program-partner shutdown must not be charged to robot_2's Actor."""
    baseline = _summary()
    partner_only = _summary(
        shutdown_count=7, player_shutdown_count=7, ai_shutdown_count=0)
    actor_only = _summary(
        shutdown_count=1, player_shutdown_count=0, ai_shutdown_count=1)

    assert evaluation._absolute_checks(partner_only, baseline)["shutdowns"] is True
    assert evaluation._absolute_checks(actor_only, baseline)["shutdowns"] is False


def test_summary_preserves_team_actor_and_partner_shutdown_counts():
    rows = []
    for ai_shutdown, player_shutdown in ((0, 0), (1, 0), (0, 1), (1, 1)):
        rows.append({
            "steps": 1,
            "active_frames": 1,
            "active_non_wait": 1,
            "productive_actions": 1,
            "action_counts": {action: int(action == "UP")
                              for action in evaluation.ACTIONS},
            "charge_needed_frames": 0,
            "charge_needed_waits": 0,
            "charge_needed_nonproductive": 0,
            "full_noncharger_frames": 1,
            "full_noncharger_waits": 0,
            "first_productive_latency": 1,
            "ai_deliveries": 1,
            "team_deliveries": 2,
            "collision_cancellations": 0,
            "collision_steps": 0,
            "longest_collision_streak": 0,
            "longest_no_task_progress_streak": 1,
            "static_wall_commands": 0,
            "shutdowns": ai_shutdown + player_shutdown,
            "ai_shutdown": ai_shutdown,
            "player_shutdown": player_shutdown,
            "submitted_actions": 1,
            "action_overrides": 0,
            "action_equal": 1,
        })

    summary = evaluation.summarize(rows)
    assert summary["shutdown_count"] == 4
    assert summary["ai_shutdown_count"] == 2
    assert summary["player_shutdown_count"] == 2
    assert summary["ai_shutdown_episode_rate"] == .5

