"""Tests for the Frigate MQTT bridge: translating detection events into observations.

The network part (paho) is exercised only through its pure pieces - `FrigateBridge.translate` and
`FrigateSource.drain` - so these run with no broker, no MQTT client, and no camera.
"""

from __future__ import annotations

from openhup_schemas import SignalKind

from openhup_vision.sources import FrigateBridge, FrigateSource


def bridge() -> FrigateBridge:
    return FrigateBridge(camera_id="door.cam", frigate_camera="doorcam", anchor_id="door")


def test_frigate_bridge_ignores_events_for_other_cameras() -> None:
    assert bridge().translate({"after": {"camera": "backyard", "label": "person"}}) is None


def test_frigate_bridge_ignores_events_without_a_label() -> None:
    """Frigate's `before` events carry no label yet; they must not become observations."""
    assert bridge().translate({"after": {"camera": "doorcam"}}) is None


def test_frigate_bridge_translates_a_person() -> None:
    translated = bridge().translate(
        {"after": {"camera": "doorcam", "label": "person", "top_score": 0.93}}
    )
    assert translated is not None
    assert translated["camera_id"] == "door.cam"
    assert translated["anchor_id"] == "door"
    assert translated["detector"] == "object_inventory"
    assert translated["score"] == 0.93

    signals = {s.key: s for s in translated["signals"]}
    assert signals["objects"].kind is SignalKind.SET
    assert signals["objects"].value == ["person"]
    assert signals["object_count"].value == 1
    assert signals["person_count"].value == 1


def test_frigate_bridge_translates_a_non_person() -> None:
    translated = bridge().translate({"after": {"camera": "doorcam", "label": "cat"}})
    assert translated is not None
    signals = {s.key: s for s in translated["signals"]}
    assert signals["person_count"].value == 0
    assert signals["objects"].value == ["cat"]


def test_frigate_source_drains_translated_events() -> None:
    source = FrigateSource([bridge()], host="127.0.0.1")
    translated = bridge().translate({"after": {"camera": "doorcam", "label": "person"}})
    assert translated is not None
    source.queue.put(translated)

    drained = source.drain()
    assert len(drained) == 1
    assert drained[0]["anchor_id"] == "door"
    # A drain empties the queue, so the next one is clean.
    assert source.drain() == []
