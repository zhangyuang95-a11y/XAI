"""r4.4 immediate shared-charger consequence.

The rule is authoritative environment physics.  It records resumable state and
never changes either robot's submitted action.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env.warehouse_native.r43_charger import (
    R43SharedChargerMixin,
    R43_OBSERVATION_FEATURE_NAMES,
)


VERSION = "warehouse-r44-shared-charger-rule.v1"
SNAPSHOT_KEY = "r44_shared_charger"
OCCUPANT_BATTERY_AFTER = 60.0
TEAMMATE_BATTERY = 20.0
TEAMMATE_DISTANCE = 2
PENALTY = -50.0
R44_OBSERVATION_FEATURE_NAMES = R43_OBSERVATION_FEATURE_NAMES


class R44SharedChargerMixin(R43SharedChargerMixin):
    """Penalize the first avoidable WAIT that leaves an occupant above 60%."""

    def step(self, actions, *, decision_metadata=None):
        self._require_state()
        if not hasattr(self, "_r43_streaks"):
            self._initialize_r43_charger()
        before = deepcopy(self.state)
        requested = {key: str(actions.get(key, "WAIT"))
                     for key in self.agent_ids}
        evidence = {
            key: self._qualification(before, requested, key)
            for key in self.agent_ids
        }
        # Skip the r4.3 rule while retaining the underlying warehouse step.
        observations, rewards, terminated, truncated, info = super(
            R43SharedChargerMixin, self
        ).step(actions, decision_metadata=decision_metadata)
        collision = bool(info.get("robot_collision"))
        penalties = []
        for agent_id in self.agent_ids:
            item = evidence[agent_id]
            occupant_before = before.by_id(agent_id)
            occupant_after = self.state.by_id(agent_id)
            teammate = before.by_id(item["teammate_id"])
            on_charger_before = (
                occupant_before.position == self.layout.charger_position
            )
            on_charger_after = (
                occupant_after.position == self.layout.charger_position
            )
            # A new occupation begins only after a true departure and re-entry.
            if not on_charger_after:
                self._r43_streaks[agent_id] = 0
                self._r43_penalized[agent_id] = False
                continue
            safe_departures = list(item["safe_departures"])
            qualifies = bool(
                on_charger_before
                and occupant_before.active
                and teammate.active
                and teammate.battery <= TEAMMATE_BATTERY
                and item["teammate_distance"] <= TEAMMATE_DISTANCE
                and safe_departures
                and requested[agent_id] == "WAIT"
                and not collision
                and occupant_after.battery > OCCUPANT_BATTERY_AFTER
            )
            self._r43_streaks[agent_id] = int(qualifies)
            if not qualifies or self._r43_penalized[agent_id]:
                continue
            event = {
                "event": "charger_occupancy_penalty",
                "event_id": (
                    f"charger-r44-{int(before.episode_id)}-"
                    f"{int(before.frame) + 1}-{agent_id}"
                ),
                "agent_id": agent_id,
                "teammate_id": item["teammate_id"],
                "amount": PENALTY,
                "occupant_battery_before": float(occupant_before.battery),
                "occupant_battery_after": float(occupant_after.battery),
                "teammate_battery_before": float(teammate.battery),
                "teammate_distance_before": int(item["teammate_distance"]),
                "safe_departures_before": safe_departures,
                "threshold_after": OCCUPANT_BATTERY_AFTER,
                "rule_version": VERSION,
            }
            self._r43_penalized[agent_id] = True
            self.state.user_score += PENALTY
            self.state.score_breakdown["charger_occupancy_penalty"] = (
                self.state.score_breakdown.get(
                    "charger_occupancy_penalty", 0.0
                ) + PENALTY
            )
            penalties.append(event)
            self._r43_last_event = deepcopy(event)
        if penalties:
            total = sum(float(event["amount"]) for event in penalties)
            info["events"].extend(deepcopy(penalties))
            info["native_score"] = float(self.state.user_score)
            info["participant_score"] = float(self.state.user_score)
            info["score_breakdown"] = dict(self.state.score_breakdown)
            component = total * float(
                self.reward_config.get("native_score_scale", .01)
            )
            info.setdefault("reward_components", {})[
                "charger_occupancy_penalty"
            ] = component
            rewards = {key: float(value) + component
                       for key, value in rewards.items()}
            self.state.last_rewards = dict(rewards)
        info["shared_charger"] = {
            "version": VERSION,
            "penalized": deepcopy(self._r43_penalized),
            "qualification": evidence,
            "penalties": deepcopy(penalties),
        }
        # Return the observation for the actual post-rule state, rather than
        # the pre-counter observation returned by the base warehouse step.
        return self.observations(), rewards, terminated, truncated, info

    def snapshot(self):
        payload = super(R43SharedChargerMixin, self).snapshot()
        if not hasattr(self, "_r43_streaks"):
            self._initialize_r43_charger()
        payload[SNAPSHOT_KEY] = {
            "version": VERSION,
            "streaks": deepcopy(self._r43_streaks),
            "penalized": deepcopy(self._r43_penalized),
            "last_event": deepcopy(self._r43_last_event),
        }
        return payload

    def restore(self, payload, **kwargs):
        if not isinstance(payload, Mapping):
            raise ValueError("r4.4 snapshot must be an object")
        value = deepcopy(payload.get(SNAPSHOT_KEY))
        if (not isinstance(value, dict)
                or value.get("version") != VERSION
                or set(value.get("streaks", {})) != set(self.agent_ids)
                or set(value.get("penalized", {})) != set(self.agent_ids)
                or any(type(v) is not int or v < 0
                       for v in value["streaks"].values())
                or any(type(v) is not bool
                       for v in value["penalized"].values())):
            raise ValueError("r4.4 shared-charger snapshot differs")
        base = deepcopy(dict(payload))
        del base[SNAPSHOT_KEY]
        super(R43SharedChargerMixin, self).restore(base, **kwargs)
        self._r43_streaks = deepcopy(value["streaks"])
        self._r43_penalized = deepcopy(value["penalized"])
        self._r43_last_event = deepcopy(value.get("last_event"))


__all__ = [
    "VERSION", "SNAPSHOT_KEY", "PENALTY",
    "R44_OBSERVATION_FEATURE_NAMES", "R44SharedChargerMixin",
]
