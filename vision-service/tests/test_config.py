"""Vision configuration and detector-plan tests.

These cover the configuration surface that is loaded before any camera or model work begins. A bad
host config should fail at startup, while an empty or disabled plan should be represented clearly.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml
from openhup_schemas import Anchor, Camera
from pydantic import ValidationError

from openhup_vision.config import (
    AnchorPlan,
    DetectorPlan,
    VisionPlan,
    VisionSettings,
)

CAMERA = {
    "id": "kitchen",
    "name": "Kitchen",
    "kind": "rtsp",
    "url": "rtsp://camera.invalid/main",
}
ANCHOR = {
    "id": "kitchen.counter",
    "camera_id": "kitchen",
    "label": "Kitchen counter",
}


def test_load_merges_vision_and_camera_files(tmp_path: Path) -> None:
    vision = tmp_path / "vision.yaml"
    cameras = tmp_path / "cameras.yaml"
    vision.write_text(yaml.safe_dump({"node_id": "node-a", "sampling": {"motion_threshold": 0.2}}))
    cameras.write_text(yaml.safe_dump({"cameras": [CAMERA], "anchors": [ANCHOR]}))

    settings = VisionSettings.load(vision, cameras)

    assert settings.node_id == "node-a"
    assert settings.sampling.motion_threshold == 0.2
    assert settings.camera("kitchen") is not None
    assert settings.anchors_for("kitchen")[0].id == "kitchen.counter"


def test_missing_files_use_defaults() -> None:
    settings = VisionSettings.load("missing-vision.yaml", "missing-cameras.yaml")

    assert settings.node_id == "vision-1"
    assert settings.cameras == []
    assert settings.anchors == []
    assert settings.enabled_cameras() == []


def test_later_nested_settings_override_earlier_values(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump({"sampling": {"active_interval": "2s", "heartbeat": "7m"}}))
    second.write_text(yaml.safe_dump({"sampling": {"active_interval": "9s"}}))

    settings = VisionSettings.load(first, second)

    assert settings.sampling.active_interval == timedelta(seconds=9)
    assert settings.sampling.heartbeat == timedelta(minutes=7)


def test_unknown_vision_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vision.yaml"
    path.write_text("not_a_setting: true\n")

    with pytest.raises(ValidationError):
        VisionSettings.load(path)


@pytest.mark.parametrize("field, value", [("jpeg_quality", 29), ("jpeg_quality", 101)])
def test_snapshot_quality_is_bounded(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        VisionSettings.model_validate({"snapshots": {field: value}})


def test_camera_filters_and_anchor_filters_respect_enabled_flags() -> None:
    settings = VisionSettings(
        cameras=[
            Camera.model_validate(CAMERA),
            Camera.model_validate({**CAMERA, "id": "off", "enabled": False}),
        ],
        anchors=[
            Anchor.model_validate(ANCHOR),
            Anchor.model_validate({**ANCHOR, "id": "off.anchor", "enabled": False}),
        ],
    )

    assert [camera.id for camera in settings.enabled_cameras()] == ["kitchen"]
    assert [anchor.id for anchor in settings.anchors_for("kitchen")] == ["kitchen.counter"]
    assert settings.camera("missing") is None


def test_detector_plan_defaults_and_anchor_idle_state() -> None:
    plan = AnchorPlan(anchor_id="a", camera_id="cam", label="A")
    detector = DetectorPlan(detector="clutter_score", wanted_signals=["clutter_level"])

    assert plan.idle is True
    plan.detectors.append(detector)
    assert plan.idle is False
    assert detector.min_interval == timedelta(seconds=30)


def test_vision_plan_filters_idle_anchors_and_lists_detector_names() -> None:
    active = AnchorPlan(
        anchor_id="a",
        camera_id="cam",
        label="A",
        detectors=[DetectorPlan(detector="clutter_score")],
    )
    idle = AnchorPlan(anchor_id="b", camera_id="cam", label="B")
    other = AnchorPlan(
        anchor_id="c",
        camera_id="other",
        label="C",
        detectors=[DetectorPlan(detector="screen_on")],
    )
    plan = VisionPlan(generated_at="now", revision="r1", anchors=[active, idle, other])

    assert plan.for_camera("cam") == [active]
    assert plan.detector_names() == {"clutter_score", "screen_on"}
    assert plan.active_anchor_count == 2


def test_plan_rejects_unknown_detector_plan_keys() -> None:
    with pytest.raises(ValidationError):
        DetectorPlan.model_validate({"detector": "clutter_score", "typo": True})
