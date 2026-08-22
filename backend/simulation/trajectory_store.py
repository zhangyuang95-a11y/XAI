"""Persistent episode store for exact simulator checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import gzip
import pickle
from pathlib import Path
from typing import Any

from backend.adapters.base import EnvironmentSnapshot, RolloutFrame


@dataclass
class TrajectoryStore:
    episodes: dict[str, list[RolloutFrame]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, episode_id: str, frame: RolloutFrame) -> None:
        errors = self.validate_frame(frame)
        if errors:
            raise ValueError(
                "Incomplete trajectory frame: " + "; ".join(errors)
            )
        if any(
            existing.frame == frame.frame
            for existing in self.episodes.get(episode_id, ())
        ):
            raise ValueError(
                f"Episode {episode_id!r} already contains frame {frame.frame}."
            )
        self.episodes.setdefault(episode_id, []).append(frame)

    @staticmethod
    def validate_frame(frame: RolloutFrame) -> tuple[str, ...]:
        """Check the evidence-bearing decision/transition contract."""

        errors: list[str] = []
        if frame.frame != frame.snapshot.frame:
            errors.append("frame ID does not match the decision snapshot")
        if frame.next_snapshot is None:
            errors.append("next_snapshot is missing")
        elif frame.next_snapshot.frame <= frame.snapshot.frame:
            errors.append("next_snapshot does not advance simulator time")
        agents = set(frame.snapshot.observations)
        required_mappings = {
            "observations": frame.observations,
            "distributions": frame.distributions,
            "action_masks": frame.action_masks,
            "proposed_actions": frame.proposed_actions,
            "executed_actions": frame.executed_actions,
        }
        for name, value in required_mappings.items():
            if set(value) != agents:
                errors.append(
                    f"{name} does not cover every policy-controlled agent"
                )
        for agent_id, distribution in frame.distributions.items():
            if not distribution.logits:
                errors.append(f"{agent_id} has no raw Actor logits")
            if not distribution.action_mask:
                errors.append(f"{agent_id} has no action mask")
            if not distribution.probabilities:
                errors.append(
                    f"{agent_id} has no post-mask action probabilities"
                )
        if not frame.reward_breakdown:
            errors.append("reward_breakdown is missing")
        if not frame.task_state:
            errors.append("task_state is missing")
        if not frame.charging_state:
            errors.append("charging_state is missing")
        if not frame.rng_state:
            errors.append("RNG state is missing")
        if not frame.checkpoint_id:
            errors.append("checkpoint_id is missing")
        return tuple(errors)

    def frames(self, episode_id: str) -> tuple[RolloutFrame, ...]:
        return tuple(self.episodes.get(episode_id, ()))

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.episodes))

    def frame(
        self,
        episode_id: str,
        frame: int,
        *,
        by_index: bool = False,
    ) -> RolloutFrame:
        frames = self.frames(episode_id)
        if by_index:
            try:
                return frames[frame]
            except IndexError as exc:
                raise KeyError(
                    f"Episode {episode_id!r} has no frame index {frame}."
                ) from exc
        for item in frames:
            if item.frame == frame:
                return item
        raise KeyError(f"Episode {episode_id!r} has no simulator frame {frame}.")

    def snapshot(self, episode_id: str, frame: int) -> EnvironmentSnapshot:
        return self.frame(episode_id, frame).snapshot

    def decision_snapshot(
        self,
        episode_id: str,
        frame: int,
    ) -> EnvironmentSnapshot:
        """Return a frame state aligned with that frame's recorded decision.

        A raw decision snapshot is captured before ``environment.step`` and
        therefore cannot yet contain the transition's executed actions or
        rewards. The enclosing ``RolloutFrame`` does. This method merges those
        records back into the same pre-transition state so an explanation
        never combines the current observation with a previous action.
        """

        item = self.frame(episode_id, frame)
        return replace(
            item.snapshot,
            actions=dict(item.actions),
            action_distributions=dict(item.distributions),
            action_masks={
                key: tuple(value)
                for key, value in item.action_masks.items()
            },
            proposed_actions=dict(item.proposed_actions),
            executed_actions=dict(item.executed_actions),
            rewards=dict(item.reward),
            metadata={
                **dict(item.snapshot.metadata),
                "action_resolution": dict(
                    item.info.get("action_resolution", {})
                ),
                "decision_outcome_frame": (
                    item.next_snapshot.frame
                    if item.next_snapshot is not None
                    else None
                ),
                "decision_evidence_aligned": True,
                "environment_events": tuple(
                    item.environment_events
                ),
            },
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wb") as handle:
            pickle.dump(
                {
                    "format": "warehouse_collaborative_trajectory_v1",
                    "episodes": self.episodes,
                    "metadata": self.metadata,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryStore":
        with gzip.open(Path(path), "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Trajectory file does not contain an episode mapping.")
        if payload.get("format") == "warehouse_collaborative_trajectory_v1":
            episodes = payload.get("episodes")
            metadata = payload.get("metadata", {})
        else:
            raise ValueError(
                "Incompatible trajectory format; regenerate it for the "
                "two-robot collaborative experiment."
            )
        if not isinstance(episodes, dict) or not isinstance(metadata, dict):
            raise ValueError("Trajectory payload has an invalid schema.")
        return cls(episodes=episodes, metadata=metadata)
