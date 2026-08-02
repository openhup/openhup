"""Bounded, time-evicting history for signals.

The engine keeps a short window of samples per `SignalKey` - just long enough for the deepest
temporal operator any enabled skill uses, plus a margin. Two properties matter:

* **Bounded.** A camera that starts emitting at 30 fps must not grow the heap. Every window has a
  hard sample cap as well as a time-based retention.
* **Out-of-order tolerant.** Observations arrive from several vision hosts over an at-least-once
  bus, so a sample can land slightly late. Windows insert in timestamp order rather than assuming
  monotonicity, because a late sample silently landing at the end of the buffer would corrupt
  every `for:` calculation that follows.

Windows are in-memory and rebuilt from Postgres on engine start (see `openhup.engine`); losing them
costs a warm-up period, never correctness.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from openhup_schemas import Observation, SignalKey

#: Safety margin added to every window's retention, so a predicate needing exactly 15m of history
#: is not defeated by eviction racing evaluation.
RETENTION_MARGIN = timedelta(minutes=1)

#: Absolute cap per window. At a realistic 1 sample/second this is ~34 minutes of history for a
#: signal nothing asks much of; deep-history predicates raise the effective floor via retention.
DEFAULT_MAX_SAMPLES = 2048


@dataclass(frozen=True, slots=True)
class Sample:
    """One measurement of one signal at one instant."""

    ts: datetime
    value: Any
    confidence: float | None = None

    def __lt__(self, other: Sample) -> bool:  # for bisect
        return self.ts < other.ts


class SignalWindow:
    """Trailing history for a single signal."""

    __slots__ = ("_appends", "_evicted", "_samples", "key", "max_samples", "retention")

    def __init__(
        self,
        key: SignalKey,
        retention: timedelta,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ) -> None:
        self.key = key
        self.retention = retention
        self.max_samples = max_samples
        self._samples: deque[Sample] = deque()
        self._appends = 0
        self._evicted = 0

    # -- writing ----------------------------------------------------------------------

    def append(self, sample: Sample) -> None:
        """Add a sample, keeping the buffer sorted by timestamp."""
        if sample.ts.tzinfo is None:
            raise ValueError(f"{self.key}: sample timestamps must be timezone-aware")
        self._appends += 1

        if not self._samples or sample.ts >= self._samples[-1].ts:
            self._samples.append(sample)
        else:
            # Late arrival. Rare, so paying O(n) here to keep reads simple is the right trade.
            ordered = list(self._samples)
            ordered.insert(bisect.bisect_right(ordered, sample), sample)
            self._samples = deque(ordered)

        while len(self._samples) > self.max_samples:
            self._samples.popleft()
            self._evicted += 1

    def extend(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self.append(sample)

    def evict_before(self, cutoff: datetime) -> int:
        """Drop samples older than `cutoff`. Returns how many went."""
        dropped = 0
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()
            dropped += 1
        self._evicted += dropped
        return dropped

    def evict(self, now: datetime) -> int:
        return self.evict_before(now - self.retention - RETENTION_MARGIN)

    def widen(self, retention: timedelta) -> None:
        """Grow retention when a new skill needs deeper history for this signal."""
        self.retention = max(self.retention, retention)

    # -- reading ----------------------------------------------------------------------

    def latest(self) -> Sample | None:
        return self._samples[-1] if self._samples else None

    def previous(self) -> Sample | None:
        return self._samples[-2] if len(self._samples) > 1 else None

    def all(self) -> list[Sample]:
        return list(self._samples)

    def since(self, cutoff: datetime) -> list[Sample]:
        """Samples with ts >= cutoff."""
        samples = self._samples
        if not samples:
            return []
        index = bisect.bisect_left(
            [s.ts for s in samples], cutoff
        )  # small buffers; clarity over micro-optimisation
        return list(samples)[index:]

    def between(self, start: datetime, end: datetime) -> list[Sample]:
        return [s for s in self._samples if start <= s.ts <= end]

    def age(self, now: datetime) -> timedelta | None:
        """How stale is the freshest sample? None when the window is empty."""
        latest = self.latest()
        return None if latest is None else now - latest.ts

    def span(self) -> timedelta:
        if len(self._samples) < 2:
            return timedelta(0)
        return self._samples[-1].ts - self._samples[0].ts

    @property
    def stats(self) -> dict[str, int]:
        return {"size": len(self._samples), "appends": self._appends, "evicted": self._evicted}

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SignalWindow {self.key} n={len(self._samples)} retention={self.retention}>"


class BindingWindows(Mapping[str, SignalWindow]):
    """Windows addressed by *binding id*, which is the view the evaluator wants.

    The engine builds one of these per (skill, anchor) pair, mapping the skill's local names
    (`clutter`, `burner`, `people`) onto the concrete windows behind them. The evaluator therefore
    never learns what an anchor or a detector is - it only sees named histories, which is what
    keeps it a pure function (ADR-003).
    """

    __slots__ = ("_by_binding",)

    def __init__(self, by_binding: Mapping[str, SignalWindow]) -> None:
        self._by_binding = dict(by_binding)

    def __getitem__(self, binding_id: str) -> SignalWindow:
        return self._by_binding[binding_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_binding)

    def __len__(self) -> int:
        return len(self._by_binding)

    def samples(self, binding_id: str) -> list[Sample]:
        window = self._by_binding.get(binding_id)
        return window.all() if window else []

    def empty_bindings(self) -> list[str]:
        """Bindings with no data at all - the engine reads this to decide STALE vs ARMED."""
        return sorted(k for k, w in self._by_binding.items() if not len(w))

    def stale_bindings(self, now: datetime, timeout: timedelta) -> list[str]:
        stale = []
        for binding_id, window in self._by_binding.items():
            age = window.age(now)
            if age is None or age > timeout:
                stale.append(binding_id)
        return sorted(stale)


class WindowStore:
    """All signal windows for a deployment, plus the retention bookkeeping.

    Retention per key is the deepest horizon of any skill that reads it, so enabling a skill with
    `absent_for: 4h` automatically deepens the buffers it depends on, and disabling it does not
    shrink them until the next `reconcile()`.
    """

    def __init__(self, default_retention: timedelta = timedelta(minutes=30)) -> None:
        self.default_retention = default_retention
        self._windows: dict[SignalKey, SignalWindow] = {}

    def ensure(self, key: SignalKey, retention: timedelta | None = None) -> SignalWindow:
        window = self._windows.get(key)
        wanted = retention or self.default_retention
        if window is None:
            window = SignalWindow(key, wanted)
            self._windows[key] = window
        else:
            window.widen(wanted)
        return window

    def get(self, key: SignalKey) -> SignalWindow | None:
        return self._windows.get(key)

    def ingest(self, observation: Observation) -> list[SignalKey]:
        """Fan one observation out into its windows. Returns the keys that were touched.

        Unknown keys are dropped rather than buffered: if no enabled skill reads a signal, holding
        its history is pure memory cost. This is why disabling every skill on an anchor really does
        stop all downstream work.
        """
        touched: list[SignalKey] = []
        for signal in observation.signals:
            key = SignalKey(observation.source.anchor_id, observation.detector.name, signal.key)
            window = self._windows.get(key)
            if window is None:
                continue
            window.append(
                Sample(ts=observation.ts, value=signal.value, confidence=signal.confidence)
            )
            touched.append(key)
        return touched

    def view(self, bindings: Mapping[str, SignalKey]) -> BindingWindows:
        """Build the evaluator's view for one skill instance."""
        return BindingWindows(
            {
                binding_id: self._windows[key]
                for binding_id, key in bindings.items()
                if key in self._windows
            }
        )

    def evict(self, now: datetime) -> int:
        return sum(window.evict(now) for window in self._windows.values())

    def drop(self, keys: Iterable[SignalKey]) -> None:
        for key in keys:
            self._windows.pop(key, None)

    def tracked_keys(self) -> list[SignalKey]:
        """Every signal being buffered. Named to avoid reading like a Mapping method."""
        return list(self._windows)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "windows": len(self._windows),
            "samples": sum(len(w) for w in self._windows.values()),
        }


__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "RETENTION_MARGIN",
    "BindingWindows",
    "Sample",
    "SignalWindow",
    "WindowStore",
]
