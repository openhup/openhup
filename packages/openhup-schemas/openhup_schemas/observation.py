"""The vision → bus contract.

An Observation is a statement of fact about one anchor at one instant, carrying no policy
whatsoever: no thresholds, no "this is bad", no task. Detectors report, the skill engine decides
(ADR-003). This is what lets five skills share one detector pass, and what makes
`POST /skills/{id}/simulate` possible - stored observations can be replayed against a draft skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import ULID, Ident, SignalKind, Slug, new_ulid, utcnow

OBSERVATION_SCHEMA = "openhup.observation/v1"


@dataclass(frozen=True, slots=True)
class SignalKey:
    """Fully qualified address of a signal: which anchor, which detector, which key.

    Frozen and hashable so it can key the engine's ring-buffer registry directly.
    """

    anchor_id: str
    detector: str
    key: str

    def __str__(self) -> str:
        return f"{self.anchor_id}/{self.detector}.{self.key}"

    @classmethod
    def parse(cls, text: str) -> SignalKey:
        anchor, _, rest = text.partition("/")
        detector, _, key = rest.partition(".")
        if not (anchor and detector and key):
            raise ValueError(f"malformed signal key {text!r} (want anchor/detector.key)")
        return cls(anchor, detector, key)


class BBox(BaseModel):
    """Normalised 0..1 box relative to the *anchor crop*, not the full frame."""

    model_config = ConfigDict(extra="forbid")

    label: str
    score: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class Signal(BaseModel):
    """One measurement. `kind` tells the engine which operators are legal against it."""

    model_config = ConfigDict(extra="forbid")

    key: Ident
    kind: SignalKind
    value: float | int | bool | str | list[str] | list[BBox]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Detector-specific breakdown. ClutterScore puts its three fused components here so the UI
    #: can explain *why* a score is what it is, which is what makes calibration possible.
    components: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_value_matches_kind(self) -> Self:
        kind, value = self.kind, self.value
        if kind is SignalKind.SCALAR:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif kind is SignalKind.COUNT:
            ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        elif kind is SignalKind.BOOLEAN:
            ok = isinstance(value, bool)
        elif kind is SignalKind.ENUM:
            ok = isinstance(value, str)
        elif kind is SignalKind.SET:
            ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        elif kind is SignalKind.BBOX_LIST:
            ok = isinstance(value, list) and all(isinstance(v, BBox) for v in value)
        else:  # pragma: no cover - exhaustive above
            ok = False
        if not ok:
            raise ValueError(f"signal {self.key!r}: value {value!r} is not valid for kind {kind}")
        return self


class ObservationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: Slug
    anchor_id: Slug
    frame_seq: int | None = None
    #: Set when the observation came from a replay/simulation rather than live capture.
    replay: bool = False


class DetectorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Ident
    #: Model identity including quantisation, e.g. "clip-vit-b32-int8@1.2". Recorded on every
    #: observation so a metric series can be interpreted after a model upgrade shifts the scale.
    version: str
    backend: str | None = Field(
        default=None, description="onnxruntime-cpu | onnxruntime-openvino | onnxruntime-cuda | ..."
    )


class MediaRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Opaque reference resolved by the snapshot store, e.g. "snap://2026/08/17/kitchen/01K3.jpg".
    snapshot_ref: str
    ttl_s: int | None = Field(default=None, ge=0)
    redacted: list[str] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None


class Observation(BaseModel):
    """What one detector saw on one anchor at one instant."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["openhup.observation/v1"] = Field(
        default=OBSERVATION_SCHEMA, alias="schema", serialization_alias="schema"
    )
    id: ULID = Field(default_factory=new_ulid)
    ts: datetime = Field(default_factory=utcnow)
    source: ObservationSource
    detector: DetectorInfo
    signals: list[Signal] = Field(min_length=1)
    media: MediaRef | None = None
    cost_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_aware_ts(self) -> Self:
        if self.ts.tzinfo is None:
            raise ValueError("Observation.ts must be timezone-aware")
        if len({s.key for s in self.signals}) != len(self.signals):
            raise ValueError("duplicate signal keys in one observation")
        return self

    def signal(self, key: str) -> Signal | None:
        return next((s for s in self.signals if s.key == key), None)

    def keys(self) -> list[SignalKey]:
        """Every SignalKey this observation carries, for engine fan-out."""
        return [SignalKey(self.source.anchor_id, self.detector.name, s.key) for s in self.signals]


class SensorReading(BaseModel):
    """Non-camera input (MQTT, Zigbee2MQTT, Home Assistant) normalised into the same pipeline.

    Sensors get first-class treatment because "trash full for 4h" is much cheaper to answer with
    a lid contact switch than with a camera, and skills should not care which one answered.
    """

    model_config = ConfigDict(extra="forbid")

    anchor_id: Slug
    key: Ident
    kind: SignalKind
    value: float | int | bool | str | list[str]
    ts: datetime = Field(default_factory=utcnow)
    origin: str = Field(default="mqtt", description="mqtt | homeassistant | http | agent")

    def to_observation(self) -> Observation:
        return Observation(
            source=ObservationSource(camera_id="sensor", anchor_id=self.anchor_id),
            detector=DetectorInfo(name="sensor", version=self.origin),
            signals=[Signal(key=self.key, kind=self.kind, value=self.value)],
            ts=self.ts,
        )


AnySignalValue = Annotated[
    Any, Field(description="float | int | bool | str | list[str] | list[BBox]")
]

__all__ = [
    "OBSERVATION_SCHEMA",
    "BBox",
    "DetectorInfo",
    "MediaRef",
    "Observation",
    "ObservationSource",
    "SensorReading",
    "Signal",
    "SignalKey",
]
