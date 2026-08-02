"""WebSocket endpoint for live events.

Topics are filtered server-side because the alternative - sending everything and letting the browser
discard it - means a phone on mobile data receives every observation from every camera.
"""

from __future__ import annotations

import contextlib
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from openhup_schemas import Envelope, EventType

router = APIRouter()
log = logging.getLogger(__name__)

#: Event families a client may subscribe to. `observation` is opt-in and high volume: it is what the
#: calibration view uses and nothing else should ask for it.
TOPICS = frozenset({"task", "alert", "skill", "metric", "goal", "system", "observation", "all"})


@router.websocket("/ws/events")
async def events(
    socket: WebSocket,
    topics: str = Query(default="task,alert,system"),
    anchor: str | None = Query(default=None),
) -> None:
    """Stream events.

    Frames are `Envelope` objects, the same shape the internal bus carries, so the frontend and the
    engine cannot disagree about structure.
    """
    state = socket.app.state.openhup
    wanted = {t.strip() for t in topics.split(",") if t.strip()} & TOPICS or {"task", "alert"}

    await socket.accept()
    await state.hub.connect(socket, wanted)
    await socket.send_json(
        Envelope(
            type=EventType.SKILL_ARMED,
            payload={
                "hello": True,
                "topics": sorted(wanted),
                "plan_revision": state.plan_revision,
                "anchor_filter": anchor,
            },
            source="api",
        ).model_dump(mode="json")
    )

    try:
        while True:
            # The client sends nothing meaningful; this is how disconnects are noticed promptly
            # rather than on the next failed broadcast.
            message = await socket.receive_text()
            if message == "ping":
                await socket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("websocket closed: %s", exc)
    finally:
        state.hub.disconnect(socket)
        with contextlib.suppress(Exception):
            await socket.close()


__all__ = ["TOPICS", "router"]
