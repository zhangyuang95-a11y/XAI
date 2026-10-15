"""r4.3 shared-charger occupancy rule on the unchanged warehouse map.

The rule is part of authoritative dynamics.  It records its own resumable
state in snapshots and never changes either robot's submitted action.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from env.warehouse.navigation import MOVE_DELTAS, shortest_path_distance


VERSION = "warehouse-r43-shared-charger-rule.v2"
OCCUPANT_BATTERY = 60.0
TEAMMATE_BATTERY = 20.0
TEAMMATE_DISTANCE = 2
GRACE_WAITS = 2
PENALTY = -50.0
SNAPSHOT_KEY = "r43_shared_charger"
R43_OBSERVATION_FEATURE_NAMES = (
    "shared_charger.self.qualifying_wait_streak",
    "shared_charger.self.penalized_this_occupancy",
    "shared_charger.self.on_charger",
    "shared_charger.self.battery_at_least_60",
    "shared_charger.other.active",
    "shared_charger.other.battery_at_most_20",
    "shared_charger.other.distance_at_most_2",
    "shared_charger.self.safe_departure_available",
    "shared_charger.other.qualifying_wait_streak",
    "shared_charger.other.penalized_this_occupancy",
)


class R43SharedChargerMixin:
    """Apply one penalty on the third avoidable active WAIT in an occupation."""

    def _initialize_r43_charger(self) -> None:
        self._r43_streaks = {agent_id: 0 for agent_id in self.agent_ids}
        self._r43_penalized = {agent_id: False for agent_id in self.agent_ids}
        self._r43_last_event = None

    def reset(self, *, seed=None):
        result = super().reset(seed=seed)
        self._initialize_r43_charger()
        return result

    @property
    def observation_size(self):
        return int(super().observation_size) + len(R43_OBSERVATION_FEATURE_NAMES)

    @property
    def feature_names(self):
        return tuple(super().feature_names) + R43_OBSERVATION_FEATURE_NAMES

    def observations(self):
        """Append the exact resumable rule state to each neural observation."""
        base = super().observations()
        if not hasattr(self, "_r43_streaks"):
            self._initialize_r43_charger()
        result = {}
        for role, agent_id in enumerate(self.agent_ids):
            other_id = self.agent_ids[1 - role]
            agent = self.state.by_id(agent_id)
            other = self.state.by_id(other_id)
            distance = shortest_path_distance(
                other.position, self.layout.charger_position,
                self.config.map_layout_id,
            )
            safe = self._safe_departures(self.state, agent_id)
            values = np.asarray((
                min(2, self._r43_streaks[agent_id]) / 2.0,
                float(self._r43_penalized[agent_id]),
                float(agent.position == self.layout.charger_position),
                float(agent.battery >= OCCUPANT_BATTERY),
                float(other.active),
                float(other.battery <= TEAMMATE_BATTERY),
                float(distance <= TEAMMATE_DISTANCE),
                float(bool(safe)),
                min(2, self._r43_streaks[other_id]) / 2.0,
                float(self._r43_penalized[other_id]),
            ), dtype=np.float32)
            result[agent_id] = np.concatenate((base[agent_id], values)).astype(
                np.float32, copy=False)
        return result

    def _safe_departures(self, state: Any, occupant_id: str) -> tuple[str, ...]:
        teammate_id = next(key for key in self.agent_ids if key != occupant_id)
        safe = []
        for action in MOVE_DELTAS:
            raw = {occupant_id: action, teammate_id: "WAIT"}
            _targets, executed, _invalid, collision, _kind, _intended = (
                self._resolve_motion(state, raw)
            )
            if not collision and executed[occupant_id] == action:
                safe.append(action)
        return tuple(safe)

    def _qualification(self, state: Any, actions: Mapping[str, str],
                       occupant_id: str) -> dict[str, Any]:
        teammate_id = next(key for key in self.agent_ids if key != occupant_id)
        occupant = state.by_id(occupant_id)
        teammate = state.by_id(teammate_id)
        distance = shortest_path_distance(
            teammate.position, self.layout.charger_position,
            self.config.map_layout_id,
        )
        safe = self._safe_departures(state, occupant_id)
        qualifies = bool(
            occupant.active
            and occupant.position == self.layout.charger_position
            and occupant.battery >= OCCUPANT_BATTERY
            and teammate.active
            and teammate.battery <= TEAMMATE_BATTERY
            and distance <= TEAMMATE_DISTANCE
            and safe
            and actions.get(occupant_id) == "WAIT"
        )
        return {
            "qualifies": qualifies,
            "occupant_id": occupant_id,
            "teammate_id": teammate_id,
            "occupant_battery": float(occupant.battery),
            "teammate_battery": float(teammate.battery),
            "teammate_distance": int(distance),
            "safe_departures": list(safe),
            "submitted_action": str(actions.get(occupant_id, "WAIT")),
        }

    def step(self, actions, *, decision_metadata=None):
        self._require_state()
        if not hasattr(self, "_r43_streaks"):
            self._initialize_r43_charger()
        before = deepcopy(self.state)
        requested = {key: str(actions.get(key, "WAIT")) for key in self.agent_ids}
        evidence = {
            key: self._qualification(before, requested, key)
            for key in self.agent_ids
        }
        observations, rewards, terminated, truncated, info = super().step(
            actions, decision_metadata=decision_metadata
        )
        collision = bool(info.get("robot_collision"))
        penalties = []
        for agent_id in self.agent_ids:
            item = evidence[agent_id]
            qualifies = bool(item["qualifies"] and not collision)
            if not qualifies:
                self._r43_streaks[agent_id] = 0
                self._r43_penalized[agent_id] = False
                continue
            self._r43_streaks[agent_id] += 1
            if (self._r43_streaks[agent_id] <= GRACE_WAITS
                    or self._r43_penalized[agent_id]):
                continue
            event = {
                "event": "charger_occupancy_penalty",
                # Stable across replay/restore so a retried response cannot
                # produce a second visual notification or audit identity.
                "event_id": (
                    f"charger-{int(before.episode_id)}-"
                    f"{int(before.frame) + 1}-{agent_id}"
                ),
                "agent_id": agent_id,
                "teammate_id": item["teammate_id"],
                "amount": PENALTY,
                "qualifying_wait": self._r43_streaks[agent_id],
                "grace_waits": GRACE_WAITS,
                "occupant_battery": item["occupant_battery"],
                "teammate_battery": item["teammate_battery"],
                "teammate_distance": item["teammate_distance"],
                "safe_departures": item["safe_departures"],
                "rule_version": VERSION,
            }
            self._r43_penalized[agent_id] = True
            self.state.user_score += PENALTY
            self.state.score_breakdown["charger_occupancy_penalty"] = (
                self.state.score_breakdown.get("charger_occupancy_penalty", 0.0)
                + PENALTY
            )
            penalties.append(event)
            self._r43_last_event = deepcopy(event)
        if penalties:
            total = sum(float(event["amount"]) for event in penalties)
            info["events"].extend(deepcopy(penalties))
            info["native_score"] = float(self.state.user_score)
            info["participant_score"] = float(self.state.user_score)
            info["score_breakdown"] = dict(self.state.score_breakdown)
            info.setdefault("reward_components", {})[
                "charger_occupancy_penalty"
            ] = total * float(self.reward_config.get("native_score_scale", .01))
            rewards = {key: float(value) + total * float(
                self.reward_config.get("native_score_scale", .01)
            ) for key, value in rewards.items()}
            self.state.last_rewards = dict(rewards)
        info["shared_charger"] = {
            "version": VERSION,
            "streaks": deepcopy(self._r43_streaks),
            "penalized": deepcopy(self._r43_penalized),
            "qualification": evidence,
            "penalties": deepcopy(penalties),
        }
        return observations, rewards, terminated, truncated, info

    def public_view(self):
        result = super().public_view()
        result["shared_charger"] = {
            "last_penalty": deepcopy(getattr(self, "_r43_last_event", None)),
        }
        return result

    def snapshot(self):
        payload = super().snapshot()
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
            raise ValueError("r4.3 snapshot must be an object")
        value = deepcopy(payload.get(SNAPSHOT_KEY))
        if (not isinstance(value, dict) or value.get("version") != VERSION
                or set(value.get("streaks", {})) != set(self.agent_ids)
                or set(value.get("penalized", {})) != set(self.agent_ids)
                or any(type(v) is not int or v < 0
                       for v in value["streaks"].values())
                or any(type(v) is not bool
                       for v in value["penalized"].values())):
            raise ValueError("r4.3 shared-charger snapshot differs")
        base = deepcopy(dict(payload))
        del base[SNAPSHOT_KEY]
        super().restore(base, **kwargs)
        self._r43_streaks = deepcopy(value["streaks"])
        self._r43_penalized = deepcopy(value["penalized"])
        self._r43_last_event = deepcopy(value.get("last_event"))
