"""Remote speech providers.

OpenHup's default voice path is entirely client-side: the PWA uses the browser's Web Speech API for
both recognition and synthesis, and this module is never touched. It exists for the explicit opt-in
where the operator puts a key in the environment and chooses a remote STT/TTS provider.

The contract is deliberately tiny and OpenAI-compatible, so it works against the public API and
against a local gateway (whisper.cpp server, a Piper wrapper) by pointing ``base_url`` at it. Remote
use is gated at config load by ``voice.allow_remote_voice``, and every call is recorded in the same
usage audit log as LLM egress - the claim "you can see exactly what left the house" covers voice
as well.
"""

from __future__ import annotations

import logging

import httpx

from ..core.config import VoiceSettings
from ..llm.base import UsageLog

log = logging.getLogger(__name__)

#: What the public OpenAI API serves for each provider, when no base_url is given.
_DEFAULT_BASE = "https://api.openai.com/v1"

#: Filename sent for transcription, guessed from the MIME type the recorder produced.
_EXTENSIONS = {
    "audio/webm": "audio.webm",
    "audio/ogg": "audio.ogg",
    "audio/mp4": "audio.m4a",
    "audio/mpeg": "audio.mp3",
    "audio/wav": "audio.wav",
    "audio/x-wav": "audio.wav",
    "audio/flac": "audio.flac",
}


class VoiceUnavailable(RuntimeError):
    """The configured provider cannot serve this request (browser-only, or not configured)."""


class VoiceProvider:
    """STT and TTS through an OpenAI-compatible endpoint.

    Constructed once at startup; holds no per-request state.
    """

    def __init__(self, settings: VoiceSettings, usage: UsageLog | None = None) -> None:
        self.settings = settings
        self.usage = usage or UsageLog()

    # -- the capabilities the frontend asks about -------------------------------------

    @property
    def stt_on_server(self) -> bool:
        return self.settings.stt_provider != "browser"

    @property
    def tts_on_server(self) -> bool:
        return self.settings.tts_provider != "browser"

    def config(self) -> dict[str, object]:
        """What GET /voice/config exposes, so the client can pick browser vs server paths."""
        return {
            "enabled": self.settings.enabled,
            "stt_provider": self.settings.stt_provider,
            "tts_provider": self.settings.tts_provider,
            "stt_remote": self.settings.stt_remote,
            "tts_remote": self.settings.tts_remote,
            "stt_on_server": self.stt_on_server,
            "tts_on_server": self.tts_on_server,
            "wake_word": self.settings.wake_word,
            "language": self.settings.language,
            "tts_voice": self.settings.tts_voice,
        }

    # -- speech-to-text ---------------------------------------------------------------

    async def transcribe(self, audio: bytes, *, content_type: str = "audio/webm") -> str:
        """Turn a recorded clip into text. Raises `VoiceUnavailable` when STT is browser-only."""
        if not self.stt_on_server:
            raise VoiceUnavailable("stt_provider is 'browser'; transcribe in the client")
        base = self._base(self.settings.stt_provider)
        filename = _EXTENSIONS.get(content_type.split(";")[0].strip(), "audio.webm")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{base}/audio/transcriptions",
                headers=self._auth(),
                data={
                    "model": self.settings.stt_model,
                    "language": self.settings.language,
                },
                files={"file": (filename, audio, content_type)},
            )
            self._record("voice_stt", len(audio), response)
            response.raise_for_status()
            data = response.json()
        text = data.get("text", "")
        if not isinstance(text, str):
            raise VoiceUnavailable("provider returned no transcription")
        return text.strip()

    # -- text-to-speech ---------------------------------------------------------------

    async def synthesize(self, text: str, *, voice: str | None = None) -> tuple[bytes, str]:
        """Turn text into audio. Returns ``(audio_bytes, media_type)``.

        Raises `VoiceUnavailable` when TTS is browser-only.
        """
        if not self.tts_on_server:
            raise VoiceUnavailable("tts_provider is 'browser'; synthesize in the client")
        base = self._base(self.settings.tts_provider)
        payload = {
            "model": self.settings.tts_model,
            "voice": voice or self.settings.tts_voice,
            "input": text,
            "response_format": "mp3",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{base}/audio/speech", headers=self._auth(), json=payload)
            self._record("voice_tts", len(text.encode()), response)
            response.raise_for_status()
        media_type = response.headers.get("content-type", "audio/mpeg").split(";")[0].strip()
        return response.content, media_type

    # -- internals ---------------------------------------------------------------------

    def _base(self, provider: str) -> str:
        if self.settings.base_url:
            return self.settings.base_url.rstrip("/")
        if provider == "openai":
            return _DEFAULT_BASE
        raise VoiceUnavailable(
            f"{provider!r} needs voice.base_url pointing at the gateway (or use 'openai')"
        )

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.api_key}"} if self.settings.api_key else {}

    def _record(
        self,
        purpose: str,
        payload_bytes: int,
        response: httpx.Response,
    ) -> None:
        # Logged once per call, whether or not it succeeded, so a failed upload is still visible.
        remote = self.settings.stt_remote if purpose == "voice_stt" else self.settings.tts_remote
        model = self.settings.stt_model if purpose == "voice_stt" else self.settings.tts_model
        self.usage.record(
            provider=self.settings.stt_provider
            if purpose == "voice_stt"
            else self.settings.tts_provider,
            model=model if response.is_success else "?",
            purpose=purpose,
            local=not remote,
            prompt_bytes=payload_bytes,
            response_bytes=len(response.content),
            ok=response.is_success,
        )


__all__ = ["VoiceProvider", "VoiceUnavailable"]
