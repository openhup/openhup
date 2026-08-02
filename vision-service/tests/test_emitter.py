"""Snapshot policy, redaction ordering, retention, and dead-banding.

These tests are the enforcement mechanism for the privacy claims in the README, so they assert
behaviour rather than implementation: nothing on disk when nothing was promised, no unredacted
pixels, and expired files actually gone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from openhup_schemas import Signal, SignalKind

from openhup_vision.emitter import (
    ObservationEmitter,
    SnapshotPolicy,
    SnapshotStore,
)

UTC = UTC
T0 = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)

# Pillow rather than OpenCV: it is a small dev dependency, and these tests must actually run.
# They are the enforcement mechanism for the privacy claims in the README, so skipping them
# silently would be the worst outcome.
Image = pytest.importorskip("PIL.Image", reason="needs an image encoder (Pillow or OpenCV)")


def read_back(path) -> np.ndarray:
    """Load a written snapshot as BGR, matching what the emitter wrote."""
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]


def frame(value: int = 90, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def store(tmp_path) -> SnapshotStore:
    return SnapshotStore(tmp_path / "snapshots")


# ------------------------------------------------------------------ policy resolution


def test_strictest_policy_wins() -> None:
    """A privacy setting that can be widened by adding a skill is not a privacy setting."""
    resolved = SnapshotPolicy.strictest(
        [
            SnapshotPolicy(attach=True, mode="archive", retention=timedelta(days=30)),
            SnapshotPolicy(attach=True, mode="thumbnail", retention=timedelta(days=2)),
        ]
    )
    assert resolved.mode == "thumbnail"
    assert resolved.retention == timedelta(days=2)


def test_one_ephemeral_skill_makes_the_anchor_ephemeral() -> None:
    resolved = SnapshotPolicy.strictest(
        [
            SnapshotPolicy(attach=True, mode="full"),
            SnapshotPolicy(attach=False, mode="ephemeral"),
        ]
    )
    assert not resolved.writes_anything


def test_redaction_targets_are_unioned() -> None:
    resolved = SnapshotPolicy.strictest(
        [
            SnapshotPolicy(redact=("faces",)),
            SnapshotPolicy(redact=("screens",)),
        ]
    )
    assert set(resolved.redact) == {"faces", "screens"}


def test_no_skills_means_no_snapshots() -> None:
    assert not SnapshotPolicy.strictest([]).writes_anything


# ------------------------------------------------------------------ writing


def test_ephemeral_writes_absolutely_nothing(tmp_path) -> None:
    snapshots = store(tmp_path)
    result = snapshots.write(
        frame(),
        anchor_id="kitchen.counter",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTA",
        policy=SnapshotPolicy(attach=False, mode="ephemeral"),
        now=T0,
    )
    assert result is None
    assert (
        not list((tmp_path / "snapshots").rglob("*")) if (tmp_path / "snapshots").exists() else True
    )


def test_full_snapshot_is_written_with_a_sidecar(tmp_path) -> None:
    snapshots = store(tmp_path)
    ref = snapshots.write(
        frame(),
        anchor_id="kitchen.counter",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTA",
        policy=SnapshotPolicy(retention=timedelta(days=7)),
        now=T0,
    )
    assert ref is not None
    assert ref.snapshot_ref.startswith("snap://2026/08/17/kitchen.counter/")
    path = snapshots.path_for(ref.snapshot_ref)
    assert path.is_file()
    assert path.with_suffix(".json").is_file()
    assert ref.ttl_s == 7 * 86400


def test_thumbnail_mode_shrinks(tmp_path) -> None:
    """Enough to recognise the place, not enough to read a document left on the counter."""
    snapshots = store(tmp_path)
    ref = snapshots.write(
        np.full((720, 1280, 3), 100, dtype=np.uint8),
        anchor_id="a.b",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTB",
        policy=SnapshotPolicy(mode="thumbnail"),
        now=T0,
    )
    assert max(ref.width, ref.height) <= 160


def test_redaction_happens_before_encoding(tmp_path) -> None:
    """Unredacted pixels must never reach the filesystem, so we check what was actually written."""
    snapshots = store(tmp_path)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    ref = snapshots.write(
        image,
        anchor_id="a.b",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTC",
        policy=SnapshotPolicy(redact=("faces",)),
        redact_boxes=[(0.0, 0.0, 0.5, 0.5)],
        now=T0,
    )
    written = read_back(snapshots.path_for(ref.snapshot_ref))
    assert written[:32, :32].std() < image[:32, :32].std()
    assert ref.redacted == ["faces"]


def test_redaction_requested_but_impossible_writes_nothing(tmp_path) -> None:
    """If we cannot find the people we promised to blur, we do not write the frame.

    This is the case where the object detector is not running but a skill still asks for face
    redaction. Writing an unredacted frame "just this once" is how privacy guarantees die.
    """
    snapshots = store(tmp_path)
    result = snapshots.write(
        frame(),
        anchor_id="a.b",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTD",
        policy=SnapshotPolicy(redact=("faces",)),
        redact_boxes=None,
        now=T0,
    )
    assert result is None


# ------------------------------------------------------------------ retention


def test_reaper_deletes_expired_snapshots(tmp_path) -> None:
    snapshots = store(tmp_path)
    ref = snapshots.write(
        frame(),
        anchor_id="a.b",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTE",
        policy=SnapshotPolicy(retention=timedelta(days=1)),
        now=T0,
    )
    path = snapshots.path_for(ref.snapshot_ref)
    assert path.exists()

    assert snapshots.reap(now=T0 + timedelta(hours=12)) == 0
    assert path.exists()

    assert snapshots.reap(now=T0 + timedelta(days=2)) == 1
    assert not path.exists()
    assert not path.with_suffix(".json").exists()


def test_reaper_needs_no_database(tmp_path) -> None:
    """Retention is driven by the sidecar, so it still works if Postgres is down."""
    snapshots = store(tmp_path)
    for index in range(3):
        snapshots.write(
            frame(),
            anchor_id="a.b",
            observation_id=f"01K3XQ8V4W7YB2M9C6NZ0PRS{index:02d}",
            policy=SnapshotPolicy(retention=timedelta(hours=1)),
            now=T0,
        )
    assert snapshots.reap(now=T0 + timedelta(days=1)) == 3
    assert snapshots.usage_bytes() == 0


def test_usage_bytes_reports_disk_footprint(tmp_path) -> None:
    snapshots = store(tmp_path)
    snapshots.write(
        frame(),
        anchor_id="a.b",
        observation_id="01K3XQ8V4W7YB2M9C6NZ0PRSTF",
        policy=SnapshotPolicy(),
        now=T0,
    )
    assert snapshots.usage_bytes() > 0


# ------------------------------------------------------------------ dead-banding


def emitter(tmp_path) -> ObservationEmitter:
    return ObservationEmitter(store=store(tmp_path), redis=None, deadband=0.05)


def scalar_observation(emit: ObservationEmitter, value: float, *, at: datetime):
    return emit.build(
        camera_id="kitchen",
        anchor_id="kitchen.counter",
        detector="clutter_score",
        detector_version="test@1",
        backend="test",
        signals=[Signal(key="clutter_level", kind=SignalKind.SCALAR, value=value)],
        now=at,
    )


def test_first_observation_always_publishes(tmp_path) -> None:
    emit = emitter(tmp_path)
    assert emit.should_publish(scalar_observation(emit, 0.5, at=T0))


async def test_unchanged_scalar_is_suppressed(tmp_path) -> None:
    emit = emitter(tmp_path)
    await emit.publish(scalar_observation(emit, 0.50, at=T0))
    assert not emit.should_publish(scalar_observation(emit, 0.51, at=T0 + timedelta(seconds=30)))
    assert emit.should_publish(scalar_observation(emit, 0.70, at=T0 + timedelta(seconds=30)))


async def test_heartbeat_beats_the_deadband(tmp_path) -> None:
    """A static scene must still emit, or every skill on the anchor goes STALE."""
    emit = emitter(tmp_path)
    emit.force_every = timedelta(minutes=5)
    await emit.publish(scalar_observation(emit, 0.5, at=T0))
    assert emit.should_publish(scalar_observation(emit, 0.5, at=T0 + timedelta(minutes=6)))


async def test_state_changes_are_never_suppressed(tmp_path) -> None:
    """Enum and boolean transitions are exactly what the temporal operators watch for."""
    emit = emitter(tmp_path)
    build = lambda value, at: emit.build(  # noqa: E731
        camera_id="kitchen",
        anchor_id="kitchen.stove",
        detector="zero_shot_state",
        detector_version="test@1",
        backend="test",
        signals=[Signal(key="burner_state", kind=SignalKind.ENUM, value=value)],
        now=at,
    )
    await emit.publish(build("off", T0))
    assert emit.should_publish(build("on", T0 + timedelta(seconds=10)))


async def test_stats_report_suppression(tmp_path) -> None:
    emit = emitter(tmp_path)
    await emit.publish(scalar_observation(emit, 0.5, at=T0))
    await emit.publish(scalar_observation(emit, 0.5, at=T0 + timedelta(seconds=10)))
    assert emit.stats()["suppressed_by_deadband"] == 1
