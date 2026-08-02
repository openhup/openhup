"""Vision service configuration.

Two sources, deliberately separate:

* `vision.yaml` - properties of *this host*: which cameras it owns, where the bus is,
  which execution provider to use, where snapshots go. Written once per machine.
* `GET /api/v1/vision/plan` - what to actually look at, derived from the currently enabled
  skills. Pulled at runtime and refreshed on a bus command.

Keeping them apart is what makes "disable every skill on an anchor and its detectors stop consuming
CPU" true, and it means adding a skill never requires touching a config file or restarting anything.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml
from openhup_schemas import Anchor, Camera, Duration
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BusSettings(_Base):
    url: str = "redis://127.0.0.1:6379/0"
    #: Cap on the observation stream. At ~1 obs/sec this is over a day of replay, which is what
    #: `/skills/{id}/simulate` reads when it dry-runs a skill against recent history.
    stream_maxlen: int = 100_000


class InferenceSettings(_Base):
    #: Leave empty to auto-select the best available provider.
    providers: list[str] = Field(default_factory=list)
    threads: int = 2
    model_dir: str | None = None
    registry: str | None = None
    #: Refuse to load a model whose checksum is not pinned. Keep this true outside development.
    require_verified_models: bool = True
    #: Per-detector model overrides, e.g. {object_inventory: dfine-s}.
    models: dict[str, str] = Field(default_factory=dict)


class SnapshotSettings(_Base):
    directory: str = "/var/lib/openhup/snapshots"
    jpeg_quality: int = Field(default=80, ge=30, le=100)
    #: Refuse to write more than this. A full disk on a home server breaks everything else too.
    max_bytes: int = 5 * 1024**3
    reap_interval: Duration = Field(default_factory=lambda: timedelta(minutes=15))


class AgentSettings(_Base):
    """HTTP listener for camera-agents pushing JPEGs in.

    For hosts that own a camera but cannot be reached from the vision service (a Pi Zero on wifi,
    anything behind NAT). Agents POST frames; the vision service never connects out to them.
    """

    #: Disabled unless a camera is configured with `kind: agent_push`.
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8090, ge=1, le=65535)
    #: Name of the env var holding the shared bearer token. Empty means accept unauthenticated
    #: uploads, which is only sensible on a loopback or fully trusted network.
    token_env: str = "OPENHUP_AGENT_TOKEN"


class MqttSettings(_Base):
    """MQTT broker, used to consume a Frigate install's detections and external sensors."""

    host: str = "127.0.0.1"
    port: int = Field(default=1883, ge=1, le=65535)
    #: Frigate publishes detection events here (`frigate/events` by default).
    frigate_topic: str = "frigate/events"
    username: str | None = None
    #: Name of the env var holding the broker password, so config files stay secret-free.
    password_env: str | None = None


class SamplingSettings(_Base):
    """Defaults for every anchor; the plan can override per anchor."""

    active_interval: Duration = Field(default_factory=lambda: timedelta(seconds=5))
    idle_interval: Duration = Field(default_factory=lambda: timedelta(seconds=30))
    dormant_interval: Duration = Field(default_factory=lambda: timedelta(minutes=2))
    settle_after: Duration = Field(default_factory=lambda: timedelta(minutes=2))
    hibernate_after: Duration = Field(default_factory=lambda: timedelta(minutes=10))
    heartbeat: Duration = Field(default_factory=lambda: timedelta(minutes=5))
    motion_threshold: float = Field(default=0.012, ge=0.0, le=1.0)


class VisionSettings(_Base):
    """Top-level vision.yaml."""

    #: Identifies this host in logs and in the backend's health view. Matters once capture is split
    #: across machines.
    node_id: str = "vision-1"
    backend_url: str = "http://127.0.0.1:8080"
    #: Bearer token for the backend. Agents and services authenticate; browsers use sessions.
    api_token_env: str = "OPENHUP_VISION_TOKEN"

    bus: BusSettings = Field(default_factory=BusSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    snapshots: SnapshotSettings = Field(default_factory=SnapshotSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    mqtt: MqttSettings = Field(default_factory=MqttSettings)

    #: Cameras this host owns. A camera listed on two hosts would be decoded twice; the backend
    #: warns about that in /system/health.
    cameras: list[Camera] = Field(default_factory=list)
    anchors: list[Anchor] = Field(default_factory=list)

    plan_refresh: Duration = Field(default_factory=lambda: timedelta(minutes=5))
    log_level: str = "INFO"
    #: Run detectors but publish nothing. For calibrating thresholds against a live scene without
    #: creating any tasks.
    dry_run: bool = False

    @classmethod
    def load(cls, *paths: str | Path) -> VisionSettings:
        """Load and merge YAML files. Later files win, shallowly, per top-level key.

        Camera and anchor definitions are usually kept in their own file (cameras.yaml) so they can
        be shared with the backend; the merge is what lets that work.
        """
        merged: dict[str, Any] = {}
        for path in paths:
            candidate = Path(path)
            if not candidate.is_file():
                continue
            data = yaml.safe_load(candidate.read_text()) or {}
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        return cls.model_validate(merged)

    def camera(self, camera_id: str) -> Camera | None:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def anchors_for(self, camera_id: str) -> list[Anchor]:
        return [a for a in self.anchors if a.camera_id == camera_id and a.enabled]

    def enabled_cameras(self) -> list[Camera]:
        return [c for c in self.cameras if c.enabled]


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------


class DetectorPlan(_Base):
    """Run this detector on this anchor at this cadence, with these params."""

    detector: str
    #: Merged from every skill binding using this detector on this anchor. Conflicting values
    #: are resolved by the backend, which knows which skills are involved and can warn.
    params: dict[str, Any] = Field(default_factory=dict)
    min_interval: Duration = Field(default_factory=lambda: timedelta(seconds=30))
    #: Signal keys any enabled skill actually reads. Signals nobody reads are not published.
    wanted_signals: list[str] = Field(default_factory=list)


class AnchorPlan(_Base):
    anchor_id: str
    camera_id: str
    label: str
    detectors: list[DetectorPlan] = Field(default_factory=list)
    #: Strictest snapshot policy among the skills watching this anchor.
    snapshot_attach: bool = False
    snapshot_mode: str = "full"
    snapshot_retention: Duration = Field(default_factory=lambda: timedelta(days=7))
    snapshot_redact: list[str] = Field(default_factory=list)
    #: Score subregions too, for spatial micro-tasking.
    score_subregions: bool = False

    @property
    def idle(self) -> bool:
        return not self.detectors


class VisionPlan(_Base):
    """The full response from GET /api/v1/vision/plan."""

    generated_at: str
    #: Changes when any enabled skill changes. The service compares it to decide whether to
    #: reconfigure, so an unchanged plan costs one string comparison.
    revision: str
    anchors: list[AnchorPlan] = Field(default_factory=list)
    #: Enrolled member gallery: (id, embedding) pairs for the consent-gated face_id detector
    #: (ADR-016). Names never travel - only ids and vectors. Empty when identity is disabled.
    members: list[dict[str, Any]] = Field(default_factory=list)

    def for_camera(self, camera_id: str) -> list[AnchorPlan]:
        return [a for a in self.anchors if a.camera_id == camera_id and not a.idle]

    def detector_names(self) -> set[str]:
        return {d.detector for anchor in self.anchors for d in anchor.detectors}

    @property
    def active_anchor_count(self) -> int:
        return sum(1 for a in self.anchors if not a.idle)


DEFAULT_CONFIG_PATHS = (
    "/etc/openhup/vision.yaml",
    "/etc/openhup/cameras.yaml",
    "./vision.yaml",
)


__all__ = [
    "DEFAULT_CONFIG_PATHS",
    "AgentSettings",
    "AnchorPlan",
    "BusSettings",
    "DetectorPlan",
    "InferenceSettings",
    "MqttSettings",
    "SamplingSettings",
    "SnapshotSettings",
    "VisionPlan",
    "VisionSettings",
]
