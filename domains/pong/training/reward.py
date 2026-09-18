"""Single-source sparse Pong training reward with an optional timing control."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from ..environment.engine import PongEnvironment
from ..environment.model import PongTransition


@dataclass
class RewardLedger:
    """Prevents a missed opportunity from becoming more than one reward event."""

    timing: str = "score_at_bottom"
    rewarded_ids: set[str] = field(default_factory=set)
    entries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.timing not in {"score_at_bottom", "failure_at_contact"}:
            raise ValueError("reward timing must be score_at_bottom or failure_at_contact")

    def reward(self, environment: PongEnvironment, transition: PongTransition) -> tuple[float, dict[str, float]]:
        components = {"small_miss": 0.0, "large_miss": 0.0, "total": 0.0}
        if self.timing == "score_at_bottom":
            candidates = [item for item in transition.events if item.get("event") == "miss_scored"]
        else:
            candidates = [item for item in transition.events
                          if item.get("event") == "encounter" and item.get("outcome") == "missed"
                          and self._will_score_before_terminal(environment, str(item.get("encounter_id")))]
        for event in candidates:
            event_id = str(event.get("encounter_id"))
            if not event_id or event_id in self.rewarded_ids:
                continue
            kind = str(event.get("ball_kind", event.get("kind", "small")))
            penalty = 3.0 if kind == "large" else 1.0
            self.rewarded_ids.add(event_id)
            components[f"{kind}_miss"] -= penalty
            components["total"] -= penalty
            self.entries.append({"encounter_id": event_id, "kind": kind, "penalty": penalty,
                                 "reward_frame": transition.after.frame,
                                 "failure_frame": event.get("frame") if event.get("event") == "encounter" else None,
                                 "score_frame": transition.after.frame if event.get("event") == "miss_scored" else None,
                                 "timing": self.timing})
        # In the early-feedback control, scoring stays authoritative at the
        # lower boundary but must not create a second training penalty. Link
        # that later event back to the original reward for auditability.
        for event in transition.events:
            if event.get("event") != "miss_scored":
                continue
            event_id = str(event.get("encounter_id"))
            for entry in self.entries:
                if str(entry.get("encounter_id")) == event_id:
                    entry["score_frame"] = int(event.get("frame", transition.after.frame))
                    entry["score_time_seconds"] = event.get("time_seconds")
                    break
        return components["total"], components

    @staticmethod
    def _will_score_before_terminal(environment: PongEnvironment, encounter_id: str) -> bool:
        ball = next((item for item in environment.balls if item.pending_miss_id == encounter_id), None)
        if ball is None or ball.vy <= 0:
            return False
        distance = max(0.0, float(environment.config.height - ball.height_cells) - ball.y)
        # A score can happen only after the first whole physics update that
        # reaches the lower edge. Rounding could incorrectly pay a terminal
        # encounter one frame early.
        frames = int(math.ceil(distance / max(1e-8, ball.vy * environment.config.fixed_dt)))
        return environment.frame_index + frames <= environment.config.max_frames

    def state_dict(self) -> dict[str, Any]:
        return {"timing": self.timing, "rewarded_ids": sorted(self.rewarded_ids), "entries": list(self.entries)}

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        if str(payload["timing"]) != self.timing:
            raise ValueError("reward timing mismatch")
        self.rewarded_ids = set(map(str, payload.get("rewarded_ids", ())))
        self.entries = [dict(item) for item in payload.get("entries", ())]
