"""Observation assembly, snapshot writing, and publishing to the bus.

This is where the privacy promises in the README become code:

* **Redaction happens before encoding.** `write_snapshot` blurs person boxes in the numpy array and
  only then hands it to the JPEG encoder. Unredacted pixels never touch the filesystem, so there is
  no window in which a stray copy exists and no reliance on a later cleanup pass.
* **Retention is written down at write time.** Each snapshot gets a sidecar recording its expiry, so
  the reaper needs no database to do its job correctly - which means retention still works if
  Postgres is down or the row was deleted.
* **Ephemeral means ephemeral.** With `attach: false` nothing is written at all. Not a thumbnail,
  not a temp file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openhup_schemas import (
    DetectorInfo,
    MediaRef,
    Observation,
    ObservationSource,
    Signal,
    Topic,
)

from .roi import Frame, blur_boxes

log = logging.getLogger(__name__)
UTC = UTC

#: Long edge for thumbnail mode: enough to recognise the place, not enough to read a document left
#: on the counter. That distinction is the whole point of the mode existing.
THUMBNAIL_EDGE = 160


@dataclass(frozen=True, slots=True)
class SnapshotPolicy:
    """Resolved from the strictest requirement among the skills watching an anchor.

    Strictest wins, deliberately: if one skill wants ephemeral snapshots and another wants a 30-day
    archive, the anchor gets ephemeral. A privacy setting that can be widened by adding a skill is
    not a privacy setting.
    """

    attach: bool = True
    mode: str = "full"  # ephemeral | thumbnail | full | archive
    retention: timedelta = timedelta(days=7)
    redact: tuple[str, ...] = ()

    @classmethod
    def strictest(cls, policies: list[SnapshotPolicy]) -> SnapshotPolicy:
        if not policies:
            return cls(attach=False, mode="ephemeral")
        order = {"ephemeral": 0, "thumbnail": 1, "full": 2, "archive": 3}
        return cls(
            attach=all(p.attach for p in policies),
            mode=min((p.mode for p in policies), key=lambda m: order.get(m, 2)),
            retention=min(p.retention for p in policies),
            redact=tuple(sorted({target for p in policies for target in p.redact})),
        )

    @property
    def writes_anything(self) -> bool:
        return self.attach and self.mode != "ephemeral"


class SnapshotStore:
    """Filesystem-backed snapshot store, addressed by `snap://` references.

    Layout: `<root>/YYYY/MM/DD/<anchor>/<ulid>.jpg` with a `.json` sidecar. Date-partitioned so the
    reaper can skip whole directories, and so an operator can delete a day by hand without
    consulting a database.
    """

    def __init__(self, root: str | Path, *, quality: int = 80) -> None:
        self.root = Path(root)
        self.quality = quality

    def path_for(self, reference: str) -> Path:
        if not reference.startswith("snap://"):
            raise ValueError(f"not a snapshot reference: {reference!r}")
        return self.root / reference[len("snap://") :]

    def write(
        self,
        frame: Frame,
        *,
        anchor_id: str,
        observation_id: str,
        policy: SnapshotPolicy,
        redact_boxes: list[tuple[float, float, float, float]] | None = None,
        now: datetime | None = None,
    ) -> MediaRef | None:
        """Redact, encode, and write. Returns None when policy says write nothing."""
        if not policy.writes_anything:
            return None

        now = now or datetime.now(tz=UTC)
        image = frame

        # Redaction first, always. Everything after this point operates on redacted pixels.
        wants_person_redaction = bool({"faces", "people"} & set(policy.redact))
        if wants_person_redaction and redact_boxes:
            image = blur_boxes(image, redact_boxes)
        elif wants_person_redaction and redact_boxes is None:
            # Asked to redact people but given no boxes: the detector that finds them is not
            # running. Refuse to write rather than write an unredacted frame.
            log.warning(
                "anchor %s: redaction requested but no person boxes available; skipping snapshot",
                anchor_id,
            )
            return None

        if policy.mode == "thumbnail":
            image = _shrink(image, THUMBNAIL_EDGE)

        relative = f"{now:%Y/%m/%d}/{anchor_id}/{observation_id}.jpg"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not _encode_jpeg(destination, image, self.quality):
            return None

        expires_at = now + policy.retention
        destination.with_suffix(".json").write_text(
            json.dumps(
                {
                    "anchor_id": anchor_id,
                    "observation_id": observation_id,
                    "created_at": now.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "mode": policy.mode,
                    "redacted": list(policy.redact),
                },
                indent=2,
            )
        )
        return MediaRef(
            snapshot_ref=f"snap://{relative}",
            ttl_s=int(policy.retention.total_seconds()),
            redacted=list(policy.redact),
            height=int(image.shape[0]),
            width=int(image.shape[1]),
        )

    def reap(self, *, now: datetime | None = None) -> int:
        """Delete expired snapshots. Driven by the sidecars, so it needs no database.

        Run from a timer, not from the request path. A snapshot whose sidecar is missing is deleted
        after a week based on mtime: an orphan with no recorded expiry is exactly the kind of file
        that would otherwise live forever.
        """
        now = now or datetime.now(tz=UTC)
        removed = 0
        if not self.root.exists():
            return 0

        for sidecar in self.root.rglob("*.json"):
            try:
                meta = json.loads(sidecar.read_text())
                expires = datetime.fromisoformat(meta["expires_at"])
            except (OSError, ValueError, KeyError):
                continue
            if expires <= now:
                sidecar.with_suffix(".jpg").unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                removed += 1

        for image in self.root.rglob("*.jpg"):
            if image.with_suffix(".json").exists():
                continue
            age = now.timestamp() - image.stat().st_mtime
            if age > timedelta(days=7).total_seconds():
                image.unlink(missing_ok=True)
                removed += 1

        for directory in sorted(self.root.rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        return removed

    def usage_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(f.stat().st_size for f in self.root.rglob("*.jpg"))


def _shrink(frame: Frame, long_edge: int) -> Frame:
    import numpy as np

    height, width = frame.shape[:2]
    if max(height, width) <= long_edge:
        return frame
    scale = long_edge / max(height, width)
    rows = (np.arange(max(int(height * scale), 1)) / scale).astype(int).clip(0, height - 1)
    cols = (np.arange(max(int(width * scale), 1)) / scale).astype(int).clip(0, width - 1)
    return frame[rows][:, cols]


def _encode_jpeg(destination: Path, frame: Frame, quality: int) -> bool:
    """Encode with OpenCV, falling back to Pillow. False if neither is installed."""
    try:
        import cv2

        return bool(cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]))
    except ImportError:
        pass
    try:
        from PIL import Image

        Image.fromarray(frame[:, :, ::-1]).save(destination, quality=quality)
        return True
    except ImportError:
        log.error("no JPEG encoder available (install opencv-python-headless or Pillow)")
        return False


@dataclass
class ObservationEmitter:
    """Builds observations and publishes them to the bus.

    Dead-banding is applied here rather than in the detectors: a scalar signal that has not moved by
    more than `deadband` since the last emission is dropped, unless `force_every` has elapsed. This
    cuts observation volume on a static scene by an order of magnitude while guaranteeing the engine
    still sees fresh data often enough not to declare the camera dead.
    """

    store: SnapshotStore
    redis: object | None = None
    deadband: float = 0.02
    force_every: timedelta = timedelta(minutes=5)
    published: int = 0
    suppressed: int = 0

    _last_values: dict[tuple[str, str, str], float] = None  # type: ignore[assignment]
    _last_emit: dict[tuple[str, str], datetime] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._last_values = {}
        self._last_emit = {}

    def build(
        self,
        *,
        camera_id: str,
        anchor_id: str,
        detector: str,
        detector_version: str,
        backend: str | None,
        signals: list[Signal],
        frame: Frame | None = None,
        policy: SnapshotPolicy | None = None,
        redact_boxes: list[tuple[float, float, float, float]] | None = None,
        frame_seq: int | None = None,
        cost_ms: float | None = None,
        now: datetime | None = None,
    ) -> Observation:
        observation = Observation(
            ts=now or datetime.now(tz=UTC),
            source=ObservationSource(camera_id=camera_id, anchor_id=anchor_id, frame_seq=frame_seq),
            detector=DetectorInfo(name=detector, version=detector_version, backend=backend),
            signals=signals,
            cost_ms=cost_ms,
        )
        if frame is not None and policy is not None and policy.writes_anything:
            observation.media = self.store.write(
                frame,
                anchor_id=anchor_id,
                observation_id=observation.id,
                policy=policy,
                redact_boxes=redact_boxes,
                now=observation.ts,
            )
        return observation

    def should_publish(self, observation: Observation) -> bool:
        """Dead-band filter. True when this observation carries new information."""
        key = (observation.source.anchor_id, observation.detector.name)
        last = self._last_emit.get(key)
        if last is None or observation.ts - last >= self.force_every:
            return True

        for signal in observation.signals:
            signal_key = (observation.source.anchor_id, observation.detector.name, signal.key)
            previous = self._last_values.get(signal_key)
            if not isinstance(signal.value, (int, float)) or isinstance(signal.value, bool):
                # Non-numeric signals (enums, sets, booleans) always count as news: a state change
                # is exactly what the temporal operators are watching for.
                return True
            if previous is None or abs(float(signal.value) - previous) >= self.deadband:
                return True
        return False

    async def publish(self, observation: Observation) -> bool:
        """Publish to Redis if it carries news. Returns whether it was published."""
        if not self.should_publish(observation):
            self.suppressed += 1
            return False

        key = (observation.source.anchor_id, observation.detector.name)
        self._last_emit[key] = observation.ts
        for signal in observation.signals:
            if isinstance(signal.value, (int, float)) and not isinstance(signal.value, bool):
                self._last_values[(*key, signal.key)] = float(signal.value)

        if self.redis is None:
            log.debug("no bus configured; dropping observation %s", observation.id)
            return False

        await self.redis.xadd(  # type: ignore[attr-defined]
            Topic.OBSERVATIONS.value,
            {"payload": observation.model_dump_json(by_alias=True)},
            maxlen=100_000,
            approximate=True,
        )
        self.published += 1
        return True

    def stats(self) -> dict[str, int]:
        return {
            "published": self.published,
            "suppressed_by_deadband": self.suppressed,
            "snapshot_bytes": self.store.usage_bytes(),
        }


def policy_from_skill(snapshot: object) -> SnapshotPolicy:
    """Translate a skill's `SnapshotSpec` into a policy."""
    return SnapshotPolicy(
        attach=bool(getattr(snapshot, "attach", True)),
        mode=str(
            getattr(getattr(snapshot, "mode", "full"), "value", getattr(snapshot, "mode", "full"))
        ),
        retention=getattr(snapshot, "retention", timedelta(days=7)),
        redact=tuple(str(getattr(t, "value", t)) for t in getattr(snapshot, "redact", ()) or ()),
    )


def default_store() -> SnapshotStore:
    return SnapshotStore(os.environ.get("OPENHUP_SNAPSHOT_DIR", "/var/lib/openhup/snapshots"))


__all__ = [
    "THUMBNAIL_EDGE",
    "ObservationEmitter",
    "SnapshotPolicy",
    "SnapshotStore",
    "default_store",
    "policy_from_skill",
]
