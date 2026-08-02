"""Async engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..core.config import DatabaseSettings
from .models import Base

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: DatabaseSettings) -> AsyncEngine:
    global _engine, _sessionmaker
    kwargs: dict[str, object] = {"echo": settings.echo, "pool_pre_ping": True}
    if not settings.is_sqlite:
        kwargs["pool_size"] = settings.pool_size
        kwargs["max_overflow"] = settings.max_overflow
    _engine = create_async_engine(settings.url, **kwargs)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("database engine not initialised; call init_engine() first")
    return _engine


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transaction per block: commit on success, roll back on any exception."""
    if _sessionmaker is None:
        raise RuntimeError("database not initialised")
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_scope() as session:
        yield session


async def create_all() -> None:
    """Create tables directly, bypassing Alembic.

    For tests and for the SQLite single-camera profile. Production uses `alembic upgrade head`,
    because the migration adds the BRIN index and monthly partitioning that this cannot express.
    """
    async with engine().begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


__all__ = [
    "create_all",
    "dispose",
    "engine",
    "get_session",
    "init_engine",
    "session_scope",
]
