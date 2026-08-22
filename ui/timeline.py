"""Exact-frame timeline state independent of Tk widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from backend.adapters.base import ActionDistribution, EnvironmentSnapshot


@dataclass(frozen=True)
class TimelineFrame:
    snapshot: EnvironmentSnapshot
    decision_snapshot: EnvironmentSnapshot | None = None
    actions: Mapping[str, str] = field(default_factory=dict)
    distributions: Mapping[str, ActionDistribution] = field(default_factory=dict)
    rewards: Mapping[str, float] = field(default_factory=dict)
    info: Mapping[str, object] = field(default_factory=dict)


class Timeline:
    def __init__(self) -> None:
        self._frames: list[TimelineFrame] = []
        self._index = 0

    def reset(self, initial: TimelineFrame) -> None:
        self._frames = [initial]
        self._index = 0

    def append(self, frame: TimelineFrame) -> int:
        # Continuing from a restored historical frame creates a new branch.
        # Discard the no-longer-reachable future so frame indices stay exact.
        if self._index < len(self._frames) - 1:
            self._frames = self._frames[: self._index + 1]
        self._frames.append(frame)
        self._index = len(self._frames) - 1
        return self._index

    def replace_current(self, frame: TimelineFrame) -> None:
        if not self._frames:
            raise IndexError("Timeline is empty.")
        self._frames[self._index] = frame

    def select(self, index: int) -> TimelineFrame:
        if not self._frames:
            raise IndexError("Timeline is empty.")
        self._index = max(0, min(int(index), len(self._frames) - 1))
        return self._frames[self._index]

    def checkpoint(self) -> tuple[tuple[TimelineFrame, ...], int]:
        """Return an immutable snapshot suitable for transaction rollback."""

        return tuple(self._frames), self._index

    def restore_checkpoint(
        self,
        checkpoint: tuple[tuple[TimelineFrame, ...], int],
    ) -> None:
        frames, index = checkpoint
        if not frames:
            raise ValueError("A timeline checkpoint must contain a frame.")
        self._frames = list(frames)
        self._index = max(0, min(int(index), len(self._frames) - 1))

    def simulator_frame(self, frame_id: int) -> TimelineFrame:
        for item in self._frames:
            if item.snapshot.frame == int(frame_id):
                return item
        raise KeyError(
            f"Timeline has no simulator frame {frame_id}."
        )

    @property
    def current(self) -> TimelineFrame:
        return self._frames[self._index]

    @property
    def index(self) -> int:
        return self._index

    @property
    def count(self) -> int:
        return len(self._frames)

    @property
    def max_index(self) -> int:
        return max(0, len(self._frames) - 1)

    @property
    def frames(self) -> tuple[TimelineFrame, ...]:
        """Immutable view used for client-side reference playback."""

        return tuple(self._frames)
