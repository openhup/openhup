"""Frame-source tests that do not require cameras or video codecs.

The source contract is small and important: keep only the latest frame, count drops, and construct
credentials from environment names rather than config secrets. A tiny in-memory source exercises
that contract deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from openhup_schemas import Camera

from openhup_vision.sources import (
    FrameSource,
    PushSource,
    RTSPSource,
    SnapshotURLSource,
    SourceStats,
    USBSource,
    _with_credentials,
    build_source,
)


class MemorySource(FrameSource):
    def _read_loop(self) -> None:
        raise NotImplementedError


def frame(value: int = 10) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_latest_frame_buffer_keeps_only_the_newest() -> None:
    source = MemorySource("cam")

    source._publish(frame(1))
    first = source.latest()
    source._publish(frame(2))
    second = source.latest()

    assert first is not None and first.frame[0, 0, 0] == 1
    assert second is not None and second.frame[0, 0, 0] == 2
    assert second.sequence == 2
    assert source.stats.frames_decoded == 2
    assert source.stats.frames_dropped == 1


def test_latest_is_none_before_any_frame() -> None:
    assert MemorySource("cam").latest() is None


def test_source_stats_health_requires_a_recent_frame() -> None:
    stats = SourceStats()
    assert stats.healthy is False

    stats.last_frame_at = datetime.now(tz=UTC) - timedelta(seconds=30)
    assert stats.healthy is True

    stats.last_frame_at = datetime.now(tz=UTC) - timedelta(seconds=61)
    assert stats.healthy is False


def test_push_source_has_no_reader_thread() -> None:
    source = PushSource("cam")

    source.start()
    assert source._thread is None
    source.stop()


def test_build_source_selects_each_native_camera_kind() -> None:
    cases = [
        ({"kind": "rtsp", "url": "rtsp://camera.invalid/stream"}, RTSPSource),
        ({"kind": "usb", "device": "/dev/video0"}, USBSource),
        ({"kind": "snapshot_url", "url": "http://camera.invalid/capture"}, SnapshotURLSource),
        ({"kind": "agent_push", "agent_id": "agent-1"}, PushSource),
    ]

    for extra, expected in cases:
        camera = Camera.model_validate({"id": "cam", "name": "Cam", **extra})
        assert isinstance(build_source(camera), expected)


def test_frigate_is_explicitly_a_bridge_not_a_native_source() -> None:
    camera = Camera.model_validate(
        {"id": "cam", "name": "Cam", "kind": "frigate", "frigate_camera": "cam"}
    )

    with pytest.raises(ValueError, match="no native source"):
        build_source(camera)


def test_camera_credentials_are_added_only_when_env_secret_exists(monkeypatch) -> None:
    camera = Camera.model_validate(
        {
            "id": "cam",
            "name": "Cam",
            "kind": "rtsp",
            "url": "rtsp://camera.invalid:554/stream",
            "username": "openhup",
            "password_env": "CAM_PASSWORD",
        }
    )
    url = "rtsp://camera.invalid:554/stream"

    monkeypatch.delenv("CAM_PASSWORD", raising=False)
    assert _with_credentials(url, camera) == url

    monkeypatch.setenv("CAM_PASSWORD", "secret")
    assert _with_credentials(url, camera) == "rtsp://openhup:secret@camera.invalid:554/stream"


def test_camera_without_password_configuration_is_unchanged() -> None:
    camera = Camera.model_validate(
        {"id": "cam", "name": "Cam", "kind": "rtsp", "url": "rtsp://camera.invalid/stream"}
    )

    assert _with_credentials(camera.url, camera) == camera.url
