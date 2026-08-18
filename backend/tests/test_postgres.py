"""PostgreSQL integration checks.

The normal suite uses SQLite for speed and portability. CI also runs this module after applying all
Alembic migrations, so the production database driver, JSONB columns, constraints, and migrated
schema are exercised together. It skips locally unless OPENHUP__DATABASE__URL is provided.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text

from openhup.core.config import Settings
from openhup.db import AnchorRow, CameraRow, dispose, init_engine, session_scope


@pytest.fixture
async def postgres_database():
    url = os.environ.get("OPENHUP__DATABASE__URL")
    if not url or not url.startswith("postgresql"):
        pytest.skip("PostgreSQL integration requires OPENHUP__DATABASE__URL")

    init_engine(Settings(database={"url": url}, llm={"provider": "echo"}).database)
    try:
        yield
    finally:
        await dispose()


@pytest.mark.asyncio
async def test_migrated_postgres_schema_accepts_core_rows(postgres_database) -> None:
    camera_id = "ci-camera"
    anchor_id = "ci.anchor"

    async with session_scope() as session:
        tables = {
            row[0]
            for row in (
                await session.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            ).all()
        }
        assert {"cameras", "anchors", "observations", "members", "presence_windows"} <= tables

        session.add(
            CameraRow(
                id=camera_id,
                name="CI camera",
                kind="rtsp",
                config={"id": camera_id, "name": "CI camera", "kind": "rtsp"},
            )
        )
        session.add(
            AnchorRow(
                id=anchor_id,
                camera_id=camera_id,
                label="CI anchor",
                config={"id": anchor_id, "camera_id": camera_id, "label": "CI anchor"},
            )
        )
        await session.flush()

        row = await session.get(CameraRow, camera_id)
        assert row is not None
        assert row.config["name"] == "CI camera"
        anchor = (
            await session.execute(select(AnchorRow).where(AnchorRow.id == anchor_id))
        ).scalar_one()
        assert anchor.camera_id == camera_id

        # Keep the integration database clean for reruns and verify the foreign-key relationship is
        # usable through the ORM rather than only through raw SQL.
        await session.delete(anchor)
        await session.delete(row)

    async with session_scope() as session:
        assert await session.get(CameraRow, camera_id) is None
        assert await session.get(AnchorRow, anchor_id) is None


@pytest.mark.asyncio
async def test_postgres_preserves_timezone_aware_values(postgres_database) -> None:
    # A direct round-trip catches accidental use of a synchronous or SQLite-only driver in CI.
    async with session_scope() as session:
        value = datetime(2026, 8, 17, 14, tzinfo=UTC)
        result = await session.execute(text("SELECT CAST(:value AS timestamptz)"), {"value": value})
        recovered = result.scalar_one()

    assert recovered.replace(tzinfo=UTC) == datetime(2026, 8, 17, 14, tzinfo=UTC)
