"""Engine identity tracking (ADR-016): face_id observations become presence and consent asks."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from openhup_schemas import DetectorInfo, Observation, ObservationSource, Signal, SignalKind
from sqlalchemy import select

from openhup.db import (
    ConsentAskRow,
    MemberRow,
    PresenceWindowRow,
    create_all,
    dispose,
    init_engine,
    session_scope,
)


def _face_observation(*, known: list[str] | None = None, unknown: bool = False) -> Observation:
    signals = [
        Signal(key="known_members", kind=SignalKind.SET, value=known or []),
        Signal(key="unknown_face", kind=SignalKind.BOOLEAN, value=unknown),
        Signal(
            key="face_count",
            kind=SignalKind.COUNT,
            value=len(known or []) + (1 if unknown else 0),
        ),
    ]
    return Observation(
        source=ObservationSource(camera_id="kitchen", anchor_id="kitchen.counter"),
        detector=DetectorInfo(name="face_id", version="yunet@1"),
        signals=signals,
    )


@pytest.fixture
async def engine(tmp_path):
    from openhup.bus import Bus
    from openhup.core.config import Settings
    from openhup.engine import Engine
    from openhup.llm import PersonalityRenderer
    from openhup.notify import Dispatcher, build_channels

    settings = Settings(
        state_dir=str(tmp_path),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "test.db")},
        bus={"url": "redis://127.0.0.1:6399/0"},
        llm={"provider": "echo"},
        snapshots={"directory": str(tmp_path / "snapshots")},
        notify={"channels": {}},
        personality={"default_personality": "kind_coach"},
    )
    init_engine(settings.database)
    await create_all()

    bus = Bus(url=settings.bus.url)
    await bus.connect()
    engine = Engine(
        settings=settings,
        bus=bus,
        renderer=PersonalityRenderer(None, personalities={}),
        dispatcher=Dispatcher(channels=build_channels({})),
    )
    try:
        yield engine
    finally:
        await bus.close()
        await dispose()


async def test_known_member_opens_a_presence_window(engine) -> None:
    async with session_scope() as session:
        session.add(
            MemberRow(
                id="member-sam",
                name="Sam",
                embedding=[0.1] * 128,
                enrolled_at=datetime.now(tz=UTC),
            )
        )
        await session.flush()

    async with session_scope() as session:
        await engine._track_identity(
            [_face_observation(known=["member-sam"])], session, now=datetime.now(tz=UTC)
        )

    async with session_scope() as session:
        windows = (await session.execute(select(PresenceWindowRow))).scalars().all()
        assert len(windows) == 1
        assert windows[0].member_id == "member-sam"
        assert windows[0].anchor_id == "kitchen.counter"
        assert windows[0].ended_at is None

    # The member's last_seen_at was bumped.
    async with session_scope() as session:
        member = await session.get(MemberRow, "member-sam")
        assert member is not None
        assert member.last_seen_at is not None


async def test_member_leaving_closes_the_window(engine) -> None:
    async with session_scope() as session:
        session.add(
            PresenceWindowRow(
                id="window-1",
                member_id="member-sam",
                anchor_id="kitchen.counter",
                started_at=datetime.now(tz=UTC),
                ended_at=None,
            )
        )
        await session.flush()

    async with session_scope() as session:
        await engine._track_identity(
            [_face_observation(known=[], unknown=False)], session, now=datetime.now(tz=UTC)
        )

    async with session_scope() as session:
        windows = (await session.execute(select(PresenceWindowRow))).scalars().all()
        assert len(windows) == 1
        assert windows[0].ended_at is not None


async def test_unknown_face_creates_one_consent_marker_per_anchor_per_day(engine) -> None:
    now = datetime.now(tz=UTC)
    async with session_scope() as session:
        # Two unknown faces in the same anchor on the same day: one ask.
        await engine._track_identity(
            [_face_observation(unknown=True), _face_observation(unknown=True)],
            session,
            now=now,
        )

    async with session_scope() as session:
        markers = (await session.execute(select(ConsentAskRow))).scalars().all()
        assert len(markers) == 1
        assert markers[0].anchor_id == "kitchen.counter"
        assert markers[0].answer == "no"  # asked; not yet answered


async def test_known_member_never_creates_a_consent_ask(engine) -> None:
    """Consent markers exist only for unknown faces - an enrolled member is never asked."""
    async with session_scope() as session:
        await engine._track_identity(
            [_face_observation(known=["member-sam"], unknown=False)],
            session,
            now=datetime.now(tz=UTC),
        )

    async with session_scope() as session:
        markers = (await session.execute(select(ConsentAskRow))).scalars().all()
        assert markers == []
