"""LLM provider abstraction.

The contract is deliberately small - completion, optional JSON-schema constraint, optional vision -
because OpenHup asks an LLM to do only two things: turn a sentence into a draft skill, and phrase
something in a voice. Neither is load-bearing (ADR-008): every call site has a deterministic
template fallback, and the internal fallback (never a shipped personality) makes no model calls.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


def system(content: str) -> Message:
    return Message(Role.SYSTEM, content)


def user(content: str) -> Message:
    return Message(Role.USER, content)


@dataclass(frozen=True, slots=True)
class ProviderCaps:
    """What a backend can actually do, so the gateway can degrade instead of failing."""

    name: str
    #: Runs on hardware the operator controls. False means data leaves the LAN.
    local: bool
    #: Native structured output. When False, the gateway uses a validate-and-repair loop.
    json_mode: bool = False
    vision: bool = False
    context_tokens: int = 8192
    #: Rough guide for the gateway's timeout budget. Local 7B models on CPU are slow.
    typical_latency_s: float = 3.0


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: True when this came from a cache rather than the model.
    cached: bool = False

    def json(self) -> Any:
        """Parse the completion as JSON, tolerating the fences models like to add."""
        return json.loads(strip_code_fences(self.text))


class LLMError(RuntimeError):
    """Base for provider failures. Always caught at the gateway; never reaches a user."""


class LLMUnavailable(LLMError):
    """The backend could not be reached, or timed out. Expected, and handled by falling back."""


class LLMRefused(LLMError):
    """The backend returned something structurally unusable after repair attempts."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal provider interface."""

    caps: ProviderCaps

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_s: float | None = None,
    ) -> Completion:
        """Generate a completion. Must raise `LLMUnavailable` rather than hanging forever."""
        ...

    async def describe_image(
        self,
        image: bytes,
        prompt: str,
        *,
        timeout_s: float | None = None,
    ) -> Completion:
        """Describe an image. Providers without vision should raise `LLMUnavailable`."""
        ...

    async def health(self) -> bool:
        """Cheap liveness probe, used by /system/health and by the gateway's circuit breaker."""
        ...


def strip_code_fences(text: str) -> str:
    """Remove ```json fences and surrounding chatter.

    Every instruct model does this occasionally regardless of instructions, and a stray fence must
    not be the reason a skill fails to parse.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    # Some models prepend prose before the object. Salvage the outermost JSON value.
    if not cleaned.startswith(("{", "[")):
        start = min(
            (index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0),
            default=-1,
        )
        if start >= 0:
            cleaned = cleaned[start:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


@dataclass
class UsageLog:
    """Per-call audit record.

    Every outbound LLM call is logged with destination and payload size, because the privacy promise
    in the README ("you can see exactly what left the house") has to be inspectable to mean
    anything. Surfaced at GET /api/v1/system/llm-usage.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)
    max_entries: int = 500

    def record(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        local: bool,
        prompt_bytes: int,
        response_bytes: int,
        included_image: bool = False,
        ok: bool = True,
    ) -> None:
        self.entries.append(
            {
                "provider": provider,
                "model": model,
                "purpose": purpose,
                "local": local,
                "prompt_bytes": prompt_bytes,
                "response_bytes": response_bytes,
                "included_image": included_image,
                "ok": ok,
            }
        )
        if len(self.entries) > self.max_entries:
            del self.entries[: len(self.entries) - self.max_entries]

    @property
    def remote_calls(self) -> int:
        return sum(1 for e in self.entries if not e["local"])

    @property
    def remote_bytes_sent(self) -> int:
        return sum(e["prompt_bytes"] for e in self.entries if not e["local"])


__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "LLMRefused",
    "LLMUnavailable",
    "Message",
    "ProviderCaps",
    "Role",
    "UsageLog",
    "strip_code_fences",
    "system",
    "user",
]
