"""Cameras and Anchors.

An **Anchor** is the unit skills actually watch: a named region of interest with its own stable
identity, its own "clean" baseline, and its own history. Cameras come and go; anchors persist.
Re-aim the camera, redraw the polygon, and every streak and metric survives (ADR-010).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import Duration, Ident, RedactionTarget, Slug, StrEnum


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceKind(StrEnum):
    RTSP = "rtsp"
    #: USB or CSI camera, reached through a camera-agent on the host that owns the device.
    USB = "usb"
    #: Agent pushes JPEGs to us; used for hosts behind NAT or on wifi-only devices.
    AGENT_PUSH = "agent_push"
    #: Consume an existing Frigate install's detections over MQTT instead of decoding twice.
    FRIGATE = "frigate"
    #: Periodic still image (ESP32-CAM, HTTP snapshot endpoint).
    SNAPSHOT_URL = "snapshot_url"


class Transport(StrEnum):
    TCP = "tcp"
    UDP = "udp"


class Point(_Base):
    """Normalised frame coordinate, 0..1, so polygons survive a resolution change."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_pair(cls, data: Any) -> Any:
        """Allow the compact ``[0.1, 0.2]`` form, which is what a polygon editor emits."""
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return {"x": data[0], "y": data[1]}
        return data


class Camera(_Base):
    """A video source.

    Credentials are *referenced*, never inlined: `password_env: KITCHEN_CAM_PASSWORD`. This keeps
    camera configs safe to commit and to paste into a bug report, and keeps secrets in one place
    (see docs/SECURITY_PRIVACY.md).
    """

    id: Slug
    name: str
    enabled: bool = True
    kind: SourceKind = SourceKind.RTSP

    #: Main stream. Used for snapshots that a human will look at.
    url: str | None = None
    #: Low-resolution substream. Used for detection - decoding 4K at 5fps to find a coffee cup
    #: is the single most common way to melt a home server.
    substream_url: str | None = None
    username: str | None = None
    password_env: str | None = Field(
        default=None, description="Name of the env var holding the password. Never the password."
    )
    transport: Transport = Transport.TCP

    #: Device path for USB sources, e.g. /dev/video0.
    device: str | None = None
    #: Agent identity for agent_push sources.
    agent_id: Slug | None = None
    #: Frigate camera name for frigate sources.
    frigate_camera: str | None = None

    #: Hard ceiling on decode rate. The sampler may go slower; it will never go faster.
    max_fps: float = Field(default=5.0, gt=0, le=30)
    read_timeout: Duration = Field(default_factory=lambda: timedelta(seconds=10))
    #: "none" | "vaapi" | "qsv" | "cuda" - passed to the decoder.
    hwaccel: str = "none"
    #: Blur these on every snapshot from this camera regardless of what skills ask for. A
    #: camera-level privacy floor for shared spaces.
    always_redact: list[RedactionTarget] = Field(default_factory=list)
    timezone: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _require_locator(self) -> Self:
        required = {
            SourceKind.RTSP: "url",
            SourceKind.SNAPSHOT_URL: "url",
            SourceKind.USB: "device",
            SourceKind.AGENT_PUSH: "agent_id",
            SourceKind.FRIGATE: "frigate_camera",
        }[self.kind]
        if not getattr(self, required):
            raise ValueError(f"camera {self.id!r} of kind {self.kind} requires '{required}'")
        return self

    @property
    def detect_url(self) -> str | None:
        """Prefer the substream for detection; fall back to the main stream."""
        return self.substream_url or self.url


class SubRegion(_Base):
    """A named slice of an anchor, used for spatial micro-tasking ("the left third")."""

    id: Ident
    label: str
    polygon: list[Point] = Field(min_length=3)
    order: int = 0


class ClutterWeights(_Base):
    """Per-anchor fusion weights for the three clutter components (ADR-005).

    Weights are relative and get normalised to sum to 1, so `2 / 1 / 1` and `0.5 / 0.25 / 0.25`
    mean the same thing. Defaults are a reasonable starting point; a wide living-room shot usually
    wants more `baseline_diff`, a small countertop more `object_density`.
    """

    baseline_diff: float = Field(default=0.4, ge=0.0)
    object_density: float = Field(default=0.3, ge=0.0)
    semantic: float = Field(default=0.3, ge=0.0)

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        total = self.baseline_diff + self.object_density + self.semantic
        if total <= 0:
            raise ValueError("clutter weights must not all be zero")
        self.baseline_diff /= total
        self.object_density /= total
        self.semantic /= total
        return self


class DetectorOverride(_Base):
    """Per-anchor detector tuning, applied on top of whatever a skill's binding params say."""

    detector: Ident
    params: dict[str, Any] = Field(default_factory=dict)
    #: Minimum interval between runs of this detector on this anchor.
    min_interval: Duration | None = None


class Anchor(_Base):
    """A named region of interest that skills watch."""

    id: Slug
    camera_id: Slug
    label: str
    enabled: bool = True

    #: Empty polygon means the whole frame.
    polygon: list[Point] = Field(default_factory=list)
    subregions: list[SubRegion] = Field(default_factory=list)

    #: Snapshot reference for the "this is what clean looks like" image. Captured via
    #: POST /anchors/{id}/baseline while the space is actually tidy.
    baseline_ref: str | None = None
    baseline_captured_at: datetime | None = None

    clutter_weights: ClutterWeights = Field(default_factory=ClutterWeights)
    #: Remaps a detector's raw output curve. 0.5 is neutral, higher is touchier.
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    detector_overrides: list[DetectorOverride] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_polygon(self) -> Self:
        if self.polygon and len(self.polygon) < 3:
            raise ValueError(
                f"anchor {self.id!r}: polygon needs 3+ points (or none for full frame)"
            )
        ids = [s.id for s in self.subregions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"anchor {self.id!r}: duplicate subregion ids")
        return self

    @property
    def is_full_frame(self) -> bool:
        return not self.polygon

    def ordered_subregions(self) -> list[SubRegion]:
        return sorted(self.subregions, key=lambda s: (s.order, s.id))


__all__ = [
    "Anchor",
    "Camera",
    "ClutterWeights",
    "DetectorOverride",
    "Point",
    "SourceKind",
    "SubRegion",
    "Transport",
]
