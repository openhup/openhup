"""Adaptive sampling and motion gating. Pure numpy, no I/O.

This module is the reason a four-camera install idles at single-digit watts instead of pinning a
CPU. It answers one question per frame: is running a detector on this worth the electricity?

Two mechanisms, in order of importance:

1. **Motion gate.** Frame-difference inside the ROI. On an empty kitchen this suppresses the great
   majority of detector invocations. Crucially it compares against the last *evaluated* frame rather
   than the previous frame, so a slow change - a pile growing over ten minutes - still accumulates
   past the threshold instead of sliding under a per-frame delta.

2. **Adaptive cadence.** Fast while a scene is active, slow when it has been still for a while.
   A kitchen in use gets looked at every few seconds; a kitchen nobody has entered since Tuesday
   gets looked at every two minutes.

A heartbeat overrides both. Without one, a genuinely static scene would emit no observations, the
skill engine would see stale signals, and every skill on the anchor would go STALE - reporting a
dead camera when the camera is fine and the room is simply tidy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .roi import Frame, Region


@dataclass(frozen=True, slots=True)
class Cadence:
    """Sampling policy for one anchor/detector pair."""

    #: Interval while the scene is active.
    active: timedelta = timedelta(seconds=5)
    #: Interval once the scene has been quiet for `settle`.
    idle: timedelta = timedelta(seconds=30)
    #: Interval after a long period of nothing happening.
    dormant: timedelta = timedelta(minutes=2)
    #: How long without motion before dropping from active to idle.
    settle: timedelta = timedelta(minutes=2)
    #: How long without motion before dropping to dormant.
    hibernate: timedelta = timedelta(minutes=10)
    #: Emit at least this often regardless of motion, so signals never look stale.
    heartbeat: timedelta = timedelta(minutes=5)

    def interval(self, quiet_for: timedelta) -> timedelta:
        if quiet_for >= self.hibernate:
            return self.dormant
        if quiet_for >= self.settle:
            return self.idle
        return self.active


@dataclass(frozen=True, slots=True)
class SampleDecision:
    """Why a frame was or was not processed. Exported as metrics and shown in the debug UI."""

    run: bool
    reason: str
    motion: float = 0.0
    quiet_for: timedelta = timedelta(0)

    def __str__(self) -> str:  # pragma: no cover - display helper
        verb = "run" if self.run else "skip"
        return f"{verb} ({self.reason}, motion={self.motion:.3f})"


def motion_score(current: Frame, reference: Frame, region: Region | None = None) -> float:
    """Fraction of ROI pixels that changed materially since the reference frame.

    Greyscale mean-absolute-difference with a per-pixel threshold, rather than a mean over the whole
    patch: a small object appearing in a large ROI barely moves the mean, but it does change a
    definite number of pixels. Counting pixels makes the gate sensitive to the thing we actually
    care about - a mug arriving on a counter.
    """
    if current.shape != reference.shape:
        return 1.0  # resolution changed; treat as fully novel

    grey_now = current.astype(np.int16).mean(axis=2)
    grey_ref = reference.astype(np.int16).mean(axis=2)
    delta = np.abs(grey_now - grey_ref)

    if region is not None and not region.is_full_frame:
        height, width = delta.shape
        mask = region.mask(height, width)
        if mask.any():
            delta = delta[mask]

    if delta.size == 0:
        return 0.0
    # 18 grey levels: above sensor noise and JPEG artefacts, below a real object appearing.
    return float((delta > 18).mean())


@dataclass
class AnchorSampler:
    """Tracks sampling state for one anchor. Not thread-safe; one per anchor per source loop."""

    anchor_id: str
    cadence: Cadence = field(default_factory=Cadence)
    #: Fraction of changed pixels that counts as motion. Raise it for a camera with noisy gain.
    motion_threshold: float = 0.012
    region: Region | None = None

    _reference: Frame | None = field(default=None, repr=False)
    _last_run: datetime | None = None
    _last_motion_at: datetime | None = None
    _last_emit: datetime | None = None

    #: Counters, exported to /metrics.
    frames_seen: int = 0
    frames_processed: int = 0
    frames_skipped_interval: int = 0
    frames_skipped_still: int = 0
    heartbeats: int = 0

    def consider(self, frame: Frame, now: datetime) -> SampleDecision:
        """Decide whether to run detectors on this frame, and update state.

        Call for every decoded frame. Cheap: the expensive path is only taken once the interval has
        elapsed, so a 5fps stream costs a timestamp comparison 24 times out of 25.
        """
        self.frames_seen += 1

        if self._reference is None or self._last_run is None:
            # First frame: always process, to establish a reference and get signals flowing.
            self._accept(frame, now, motion=1.0)
            return SampleDecision(True, "first frame", 1.0)

        quiet_for = now - (self._last_motion_at or self._last_run)
        interval = self.cadence.interval(quiet_for)

        if now - self._last_run < interval:
            self.frames_skipped_interval += 1
            return SampleDecision(False, f"within {interval} interval", quiet_for=quiet_for)

        motion = motion_score(frame, self._reference, self.region)
        if motion >= self.motion_threshold:
            self._last_motion_at = now
            self._accept(frame, now, motion)
            return SampleDecision(True, "motion", motion, timedelta(0))

        # Nothing moved. Still emit periodically, or the engine cannot tell a tidy room from a dead
        # camera - and it is designed to complain loudly about the latter.
        since_emit = now - (self._last_emit or self._last_run)
        if since_emit >= self.cadence.heartbeat:
            self.heartbeats += 1
            self._accept(frame, now, motion)
            return SampleDecision(True, "heartbeat", motion, quiet_for)

        self.frames_skipped_still += 1
        self._last_run = now  # interval consumed; do not re-diff every frame
        return SampleDecision(False, "no motion", motion, quiet_for)

    def _accept(self, frame: Frame, now: datetime, motion: float) -> None:
        self.frames_processed += 1
        self._last_run = now
        self._last_emit = now
        # Reference is the last *evaluated* frame, so gradual accumulation is not invisible.
        self._reference = frame.copy()
        if self._last_motion_at is None:
            self._last_motion_at = now

    def force_next(self) -> None:
        """Drop the interval guard, e.g. after a plan change or an explicit snapshot request."""
        self._last_run = None
        self._reference = None

    @property
    def efficiency(self) -> float:
        """Share of frames that did *not* need a detector. The number that justifies this module."""
        if not self.frames_seen:
            return 0.0
        return 1.0 - (self.frames_processed / self.frames_seen)

    def stats(self) -> dict[str, float | int]:
        return {
            "frames_seen": self.frames_seen,
            "frames_processed": self.frames_processed,
            "skipped_interval": self.frames_skipped_interval,
            "skipped_still": self.frames_skipped_still,
            "heartbeats": self.heartbeats,
            "efficiency": round(self.efficiency, 4),
        }


__all__ = ["AnchorSampler", "Cadence", "SampleDecision", "motion_score"]
