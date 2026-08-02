"""Tests for the sensor feed: value parsing and the change-tracking store.

These run with no MQTT broker - `SensorMqtt`'s network thread is left out, and the pure parts it
depends on (`coerce_sensor_value`, `SensorFeed`) are exercised directly.
"""

from __future__ import annotations

import pytest
from openhup_schemas import SignalKind

from openhup_vision.sensor_feed import SensorBinding, SensorFeed, coerce_sensor_value

# ------------------------------------------------------------------ value parsing


def test_coerce_scalar() -> None:
    assert coerce_sensor_value("22.4", SignalKind.SCALAR) == 22.4
    assert coerce_sensor_value(b"0.75", SignalKind.SCALAR) == 0.75


def test_coerce_count() -> None:
    assert coerce_sensor_value("3", SignalKind.COUNT) == 3


def test_coerce_boolean() -> None:
    for truthy in ("true", "ON", "yes", "1", "open", "detected"):
        assert coerce_sensor_value(truthy, SignalKind.BOOLEAN) is True
    for falsy in ("false", "off", "no", "0", "closed", "clear"):
        assert coerce_sensor_value(falsy, SignalKind.BOOLEAN) is False


def test_coerce_boolean_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        coerce_sensor_value("maybe", SignalKind.BOOLEAN)


def test_coerce_enum_is_identity() -> None:
    assert coerce_sensor_value("half_full", SignalKind.ENUM) == "half_full"


def test_coerce_set_splits_commas_and_parses_json() -> None:
    assert coerce_sensor_value("a,b,c", SignalKind.SET) == ["a", "b", "c"]
    assert coerce_sensor_value('["x", "y"]', SignalKind.SET) == ["x", "y"]


# ------------------------------------------------------------------ the feed


def test_set_value_reports_only_actual_changes() -> None:
    feed = SensorFeed()
    assert feed.set_value("trash", "lid_open", SignalKind.BOOLEAN, True) is True
    assert feed.set_value("trash", "lid_open", SignalKind.BOOLEAN, True) is False
    assert feed.set_value("trash", "lid_open", SignalKind.BOOLEAN, False) is True


def test_take_returns_none_until_changed_and_clears_the_flag() -> None:
    feed = SensorFeed()
    assert feed.take("trash", "lid_open") is None
    feed.set_value("trash", "lid_open", SignalKind.BOOLEAN, True)
    signal = feed.take("trash", "lid_open")
    assert signal is not None
    assert signal.value is True
    # Consumed once: the next read is quiet until the value changes again.
    assert feed.take("trash", "lid_open") is None


def test_drain_returns_changed_values_with_their_anchor() -> None:
    feed = SensorFeed()
    feed.set_value("trash", "lid_open", SignalKind.BOOLEAN, True)
    feed.set_value("bowl", "fill", SignalKind.SCALAR, 0.8)

    drained = feed.drain()
    by_key = {(anchor, signal.key): signal for anchor, signal in drained}
    assert by_key[("trash", "lid_open")].value is True
    assert by_key[("bowl", "fill")].value == 0.8
    # Draining clears everything; a second drain is empty.
    assert feed.drain() == []


def test_latest_reads_without_consuming() -> None:
    feed = SensorFeed()
    feed.set_value("bowl", "fill", SignalKind.SCALAR, 0.5)
    assert feed.latest("bowl", "fill").value == 0.5
    assert feed.latest("bowl", "fill").value == 0.5  # still there


def test_binding_is_immutable_and_typed() -> None:
    binding = SensorBinding(
        topic="zigbee/trash", anchor_id="trash", key="lid_open", kind=SignalKind.BOOLEAN
    )
    assert binding.anchor_id == "trash"
    assert binding.kind is SignalKind.BOOLEAN
