"""Test helpers for the skill engine.

Everything is built around a fixed clock (`T0`) and relative offsets, because the whole point of
keeping the engine pure is that time can be a parameter instead of a race condition.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from openhup_schemas import Anchor, SignalKey

from openhup.skills.window import BindingWindows, Sample, SignalWindow

UTC = UTC

#: Fixed reference instant for every test. A Monday, mid-afternoon local-ish, so weekday and
#: quiet-hour logic has something unambiguous to bite on.
T0 = datetime(2026, 8, 17, 14, 0, 0, tzinfo=UTC)


def at(**offset: float) -> datetime:
    """A time relative to T0: `at(minutes=-15)` is fifteen minutes of history ago."""
    return T0 + timedelta(**offset)


def samples(pairs: Sequence[tuple[float, Any]], *, confidence: float | None = None) -> list[Sample]:
    """Build samples from (minutes_before_now, value) pairs.

    Written back-to-front on purpose: `[(20, 0.8), (10, 0.8), (0, 0.8)]` reads as "twenty minutes
    ago, ten minutes ago, now", which is how the conditions under test are described.
    """
    return [
        Sample(ts=T0 - timedelta(minutes=minutes), value=value, confidence=confidence)
        for minutes, value in pairs
    ]


def window(
    pairs: Sequence[tuple[float, Any]],
    *,
    key: SignalKey | None = None,
    retention: timedelta = timedelta(hours=6),
    confidence: float | None = None,
) -> SignalWindow:
    signal_key = key or SignalKey("test.anchor", "test_detector", "test_signal")
    win = SignalWindow(signal_key, retention)
    win.extend(samples(pairs, confidence=confidence))
    return win


def windows(**by_binding: SignalWindow) -> BindingWindows:
    """`windows(clutter=window([...]), people=window([...]))`."""
    return BindingWindows(by_binding)


def steady(value: Any, *, minutes: float, every: float = 1.0) -> list[tuple[float, Any]]:
    """A constant value sampled every `every` minutes over the last `minutes` minutes."""
    count = int(minutes / every) + 1
    return [(minutes - i * every, value) for i in range(count)]


def ramp(values: Iterable[Any], *, every: float = 1.0) -> list[tuple[float, Any]]:
    """Values oldest-first, `every` minutes apart, ending at now."""
    listed = list(values)
    span = (len(listed) - 1) * every
    return [(span - i * every, value) for i, value in enumerate(listed)]


@pytest.fixture
def anchors() -> dict[str, Anchor]:
    """A small but realistic anchor set: kitchen counter with a baseline, stove, shelf, TV."""
    return {
        a.id: a
        for a in [
            Anchor(
                id="kitchen.counter",
                camera_id="kitchen",
                label="Kitchen counter",
                polygon=[[0.05, 0.30], [0.95, 0.30], [0.95, 0.80], [0.05, 0.80]],
                baseline_ref="snap://baseline/kitchen.counter.jpg",
                baseline_captured_at=T0 - timedelta(days=3),
                subregions=[
                    {
                        "id": "left",
                        "label": "Left third",
                        "order": 0,
                        "polygon": [[0.05, 0.30], [0.35, 0.30], [0.35, 0.80], [0.05, 0.80]],
                    },
                    {
                        "id": "middle",
                        "label": "Middle third",
                        "order": 1,
                        "polygon": [[0.35, 0.30], [0.65, 0.30], [0.65, 0.80], [0.35, 0.80]],
                    },
                    {
                        "id": "right",
                        "label": "Right third",
                        "order": 2,
                        "polygon": [[0.65, 0.30], [0.95, 0.30], [0.95, 0.80], [0.65, 0.80]],
                    },
                ],
            ),
            Anchor(id="kitchen.stove", camera_id="kitchen", label="Stove top"),
            Anchor(
                id="office.shelf",
                camera_id="office",
                label="Office shelf",
                baseline_ref="snap://baseline/office.shelf.jpg",
            ),
            Anchor(id="living.tv", camera_id="living", label="TV"),
            Anchor(id="living.walkway", camera_id="living", label="Walkway", enabled=True),
        ]
    }
