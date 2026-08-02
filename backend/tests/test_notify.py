"""Notification policy.

Every test here is about restraint - except the ones about safety alerts, which are the opposite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from openhup_schemas import NotificationRequest, TimeWindow, Urgency

from openhup.notify.channels import Channel, Dispatcher, build_channels

UTC = UTC
T0 = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
NIGHT = datetime(2026, 8, 17, 23, 30, tzinfo=UTC)


class Recorder(Channel):
    """Test channel that records instead of sending."""

    supports_images = True

    def __init__(self, channel_id: str = "rec", config: dict | None = None) -> None:
        super().__init__(channel_id, config or {})
        self.sent: list[NotificationRequest] = []
        self.fail = False

    async def send(self, request, *, image=None) -> None:
        if self.fail:
            raise RuntimeError("channel is down")
        self.sent.append(request)


def request(**kwargs) -> NotificationRequest:
    kwargs.setdefault("channels", [])
    kwargs.setdefault("title", "Kitchen counter")
    kwargs.setdefault("body", "It has become a situation.")
    return NotificationRequest(**kwargs)


def dispatcher(channel: Channel | None = None, **kwargs) -> tuple[Dispatcher, Recorder]:
    recorder = channel or Recorder()
    return Dispatcher(channels={recorder.id: recorder}, **kwargs), recorder


# ------------------------------------------------------------------ construction


def test_unknown_channel_type_is_skipped_not_fatal() -> None:
    channels = build_channels({"broken": {"type": "carrier_pigeon"}, "ok": {"type": "log"}})
    assert "broken" not in channels
    assert "ok" in channels


def test_misconfigured_channel_is_skipped() -> None:
    channels = build_channels({"nt": {"type": "ntfy"}})  # missing required topic
    assert "log" in channels  # falls back so notifications are never silently swallowed


def test_no_channels_falls_back_to_the_log() -> None:
    assert set(build_channels({})) == {"log"}


# ------------------------------------------------------------------ quiet hours


async def test_quiet_hours_hold_rather_than_drop() -> None:
    """Discarding a notification would be worse than waking someone up. It waits instead."""
    quiet = TimeWindow.model_validate({"between": ["22:00", "07:00"], "tz": "UTC"})
    dispatch, recorder = dispatcher(quiet_hours=quiet)

    results = await dispatch.dispatch(request(urgency=Urgency.LOW), now=NIGHT)
    assert results[0].status == "held"
    assert recorder.sent == []
    assert len(dispatch.held) == 1

    released = await dispatch.release_held(now=T0)
    assert recorder.sent
    assert any(r.status == "sent" for r in released)


async def test_safety_alerts_ignore_quiet_hours() -> None:
    """A burner left on at 3am is precisely when you need to be told."""
    quiet = TimeWindow.model_validate({"between": ["22:00", "07:00"], "tz": "UTC"})
    dispatch, recorder = dispatcher(quiet_hours=quiet)
    results = await dispatch.dispatch(request(urgency=Urgency.HIGH), now=NIGHT)
    assert results[0].status == "sent"
    assert len(recorder.sent) == 1


async def test_held_notifications_stay_held_inside_the_window() -> None:
    quiet = TimeWindow.model_validate({"between": ["22:00", "07:00"], "tz": "UTC"})
    dispatch, _ = dispatcher(quiet_hours=quiet)
    await dispatch.dispatch(request(urgency=Urgency.LOW), now=NIGHT)
    assert await dispatch.release_held(now=NIGHT + timedelta(minutes=5)) == []
    assert len(dispatch.held) == 1


# ------------------------------------------------------------------ rate limiting


async def test_rate_limit_protects_the_phone_battery() -> None:
    dispatch, recorder = dispatcher(max_per_hour=3)
    for _ in range(5):
        await dispatch.dispatch(request(urgency=Urgency.NORMAL), now=T0)
    assert len(recorder.sent) == 3


async def test_rate_limit_never_applies_to_high_urgency() -> None:
    dispatch, recorder = dispatcher(max_per_hour=1)
    for _ in range(4):
        await dispatch.dispatch(request(urgency=Urgency.CRITICAL), now=T0)
    assert len(recorder.sent) == 4


async def test_rate_limit_window_rolls_forward() -> None:
    dispatch, recorder = dispatcher(max_per_hour=1)
    await dispatch.dispatch(request(urgency=Urgency.NORMAL), now=T0)
    await dispatch.dispatch(request(urgency=Urgency.NORMAL), now=T0 + timedelta(hours=2))
    assert len(recorder.sent) == 2


# ------------------------------------------------------------------ dedupe


async def test_duplicates_are_suppressed() -> None:
    dispatch, recorder = dispatcher()
    key = "kitchen-clutter-buster:kitchen.counter"
    await dispatch.dispatch(request(urgency=Urgency.LOW, dedupe_key=key), now=T0)
    results = await dispatch.dispatch(
        request(urgency=Urgency.LOW, dedupe_key=key), now=T0 + timedelta(minutes=2)
    )
    assert results[0].status == "suppressed"
    assert len(recorder.sent) == 1


async def test_dedupe_expires() -> None:
    dispatch, recorder = dispatcher(dedupe_window=timedelta(minutes=5))
    key = "k"
    await dispatch.dispatch(request(urgency=Urgency.LOW, dedupe_key=key), now=T0)
    await dispatch.dispatch(
        request(urgency=Urgency.LOW, dedupe_key=key), now=T0 + timedelta(minutes=10)
    )
    assert len(recorder.sent) == 2


async def test_dedupe_does_not_apply_to_urgent_alerts() -> None:
    dispatch, recorder = dispatcher()
    for _ in range(3):
        await dispatch.dispatch(request(urgency=Urgency.HIGH, dedupe_key="stove"), now=T0)
    assert len(recorder.sent) == 3


# ------------------------------------------------------------------ routing and failure


async def test_min_urgency_routes_chores_and_safety_differently() -> None:
    chores = Recorder("chores", {"min_urgency": "info"})
    sms = Recorder("sms", {"min_urgency": "high"})
    dispatch = Dispatcher(channels={"chores": chores, "sms": sms})

    await dispatch.dispatch(request(urgency=Urgency.LOW), now=T0)
    assert len(chores.sent) == 1
    assert sms.sent == []

    await dispatch.dispatch(request(urgency=Urgency.CRITICAL), now=T0)
    assert len(sms.sent) == 1


async def test_one_dead_channel_does_not_block_the_others() -> None:
    broken = Recorder("broken")
    broken.fail = True
    working = Recorder("working")
    dispatch = Dispatcher(channels={"broken": broken, "working": working})

    results = await dispatch.dispatch(request(urgency=Urgency.HIGH), now=T0)
    statuses = {r.channel: r.status for r in results}
    assert statuses["broken"] == "failed"
    assert statuses["working"] == "sent"
    assert len(working.sent) == 1


async def test_disabled_channel_is_skipped() -> None:
    off = Recorder("off", {"enabled": False})
    dispatch = Dispatcher(channels={"off": off})
    results = await dispatch.dispatch(request(urgency=Urgency.HIGH), now=T0)
    assert results[0].status == "suppressed"


async def test_explicit_channel_list_is_respected() -> None:
    a, b = Recorder("a"), Recorder("b")
    dispatch = Dispatcher(channels={"a": a, "b": b})
    await dispatch.dispatch(request(channels=["b"], urgency=Urgency.NORMAL), now=T0)
    assert a.sent == []
    assert len(b.sent) == 1


async def test_channel_test_reports_failure_clearly() -> None:
    broken = Recorder("broken")
    broken.fail = True
    result = await broken.test()
    assert not result.ok
    assert "down" in result.detail
