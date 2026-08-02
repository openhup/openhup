"""The FastAPI application.

I/O only: no CPU work, no inference, no LLM calls on a request path that a browser is waiting on
(skill parsing is the exception, and it has a timeout and a fallback). The skill engine runs in a
separate process precisely so this one stays responsive while a 7B model thinks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..bus import Bus
from ..core.config import Settings, load_settings
from ..db import init_engine, session_scope
from ..llm import PersonalityRenderer, UsageLog, build_provider
from ..notify import Dispatcher, build_channels
from ..skills.compile import SkillCompileError
from ..voice import VoiceProvider
from .state import AppState
from .v1 import router as v1_router

log = logging.getLogger("openhup.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    for warning in settings.warnings():
        log.warning(warning)

    init_engine(settings.database)

    bus = Bus(
        url=settings.bus.url,
        observation_maxlen=settings.bus.observation_maxlen,
        block_ms=settings.bus.consumer_block_ms,
        claim_after=settings.bus.claim_after,
        consumer_name="openhup-api",
    )
    await bus.connect()

    usage = UsageLog()
    provider = None
    try:
        provider = build_provider(settings.llm)
    except Exception as exc:
        log.error("LLM provider unavailable (%s); running with templates only", exc)

    state = AppState(
        settings=settings,
        bus=bus,
        usage=usage,
        provider=provider,
        renderer=PersonalityRenderer(
            provider,
            settings=settings.personality,
            personalities={},
            usage=usage,
            timeout_s=settings.llm.timeout.total_seconds(),
        ),
        dispatcher=Dispatcher(
            channels=build_channels(settings.notify.channels),
            max_per_hour=settings.notify.max_per_hour,
        ),
        voice=VoiceProvider(settings.voice, usage=usage),
    )
    await state.load_registry()
    app.state.openhup = state

    # Forward events from the engine (over Redis) and from this process to connected browsers.
    state.wire_bus_to_websockets()
    relay_task = asyncio.create_task(state.relay_bus_events(), name="bus-event-relay")

    log.info(
        "OpenHup API %s ready on %s:%s (%d personalities, %d skills)",
        __version__,
        settings.security.bind_host,
        settings.security.bind_port,
        len(state.personalities),
        len(state.compiled),
    )
    try:
        yield
    finally:
        relay_task.cancel()
        await asyncio.gather(relay_task, return_exceptions=True)
        await bus.close()
        from ..db import dispose

        await dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    app = FastAPI(
        title="OpenHup",
        version=__version__,
        summary="Self-hosted, local-first home assistant that turns what cameras see into tasks.",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings

    if settings.security.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(v1_router, prefix="/api/v1")

    @app.exception_handler(SkillCompileError)
    async def _compile_error(request: Request, exc: SkillCompileError) -> JSONResponse:
        """Compile failures are user errors with actionable messages, not server faults.

        422 with every finding, so the UI can list them all at once rather than making someone fix
        one problem per save.
        """
        return JSONResponse(
            status_code=422,
            content={
                "error": "skill_compile_failed",
                "skill_id": exc.skill_id,
                "findings": [
                    {
                        "code": f.code,
                        "message": f.message,
                        "binding": f.binding,
                        "error": f.error,
                    }
                    for f in exc.findings
                ],
            },
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Ready means the database answers. The bus and the LLM are optional by design."""
        from sqlalchemy import text

        try:
            async with session_scope() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:
            return JSONResponse(status_code=503, content={"status": "degraded", "error": str(exc)})
        return JSONResponse(content={"status": "ready"})

    # The built frontend, when present. Mounted last so it cannot shadow the API.
    frontend = Path(__file__).resolve().parents[2] / "static"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()


def run() -> None:
    """Entrypoint for `openhup-api`."""
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "openhup.api.main:application",
        host=settings.security.bind_host,
        port=settings.security.bind_port,
        log_level=settings.log_level.lower(),
        proxy_headers=settings.security.trust_proxy_headers,
        forwarded_allow_ips="*" if settings.security.trust_proxy_headers else None,
    )


def application() -> FastAPI:
    """ASGI factory, used by `uvicorn openhup.api.main:application --factory`."""
    return create_app()


__all__ = ["application", "create_app", "run"]
