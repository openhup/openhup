"""Voice interface: speech-to-text and text-to-speech.

``provider.py`` turns audio into text and text into audio through a remote provider when one is
configured. ``commands.py`` turns a transcript into an action - task commands, queries, skill
dictation, or navigation - without ever depending on the LLM.

By default neither module does anything on the server: ``stt_provider: browser`` and
``tts_provider: browser`` mean the Web Speech API in the PWA handles both, and nothing is uploaded
to OpenHup at all. The remote providers are opt-in behind ``voice.allow_remote_voice``.
"""

from __future__ import annotations

from .commands import (
    CommandResult,
    command_action,
    is_query,
    navigation_target,
    route_command,
    snooze_minutes,
)
from .provider import VoiceProvider, VoiceUnavailable

__all__ = [
    "CommandResult",
    "VoiceProvider",
    "VoiceUnavailable",
    "command_action",
    "is_query",
    "navigation_target",
    "route_command",
    "snooze_minutes",
]
