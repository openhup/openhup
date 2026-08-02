"""Voice API.

The browser handles STT/TTS locally by default, so three of these endpoints only matter when a
remote provider is configured. `/voice/command` is the one the client always uses: it turns a
transcript into an action and a spoken reply.

The audio endpoints take raw bytes in the request body (a MediaRecorder blob with its own
Content-Type) rather than multipart, so the PWA can upload without assembling a form.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import get_session
from ...voice import VoiceUnavailable, route_command
from ..state import AppState

router = APIRouter(tags=["voice"])

Session = Annotated[AsyncSession, Depends(get_session)]


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    voice: str | None = None


class CommandRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    #: The member id this device already knows (per-device "who am I"). Declared, never inferred:
    #: the person told the device who they are, and the device passes it along.
    speaker: str | None = None


@router.get("/voice/config")
async def voice_config(request: Request) -> dict[str, Any]:
    """What the client needs to decide between browser and server speech paths."""
    state = state_of(request)
    if not state.settings.voice.enabled:
        return {**state.voice.config(), "enabled": False}
    return state.voice.config()


@router.post("/voice/transcribe")
async def transcribe(request: Request) -> dict[str, str]:
    """Audio in (raw body), text out. Only reachable when `stt_provider` is remote."""
    state = state_of(request)
    audio = await request.body()
    try:
        text = await state.voice.transcribe(
            audio, content_type=request.headers.get("content-type", "audio/webm")
        )
    except VoiceUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"text": text}


@router.post("/voice/synthesize")
async def synthesize(request: Request, payload: SynthesizeRequest) -> Response:
    """Text in, audio out. Only reachable when `tts_provider` is remote."""
    state = state_of(request)
    try:
        audio, media_type = await state.voice.synthesize(payload.text, voice=payload.voice)
    except VoiceUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(content=audio, media_type=media_type)


@router.post("/voice/command")
async def command(
    request: Request,
    session: Session,
    payload: CommandRequest,
) -> dict[str, Any]:
    """Turn a transcript into an action and a spoken reply.

    Stateless on the wire: the client sends recognised text, gets back an intent, any side effect
    already applied (task completed/snoozed), and the text to speak. Navigation is returned as a
    target route for the client to follow.
    """
    state = state_of(request)
    result = await route_command(
        payload.text, state=state, session=session, speaker=payload.speaker
    )
    return asdict(result)


__all__ = ["router"]
