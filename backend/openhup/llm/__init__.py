"""LLM integration and the personality layer.

    base.py        provider protocol, messages, usage audit log
    providers.py   ollama (default), openai_compatible, anthropic, echo
    prompts.py     every prompt the system can send, in one reviewable file
    safety.py      the output filter - deny-list per boundary, applied after generation
    render.py      PersonalityRenderer: facts in, one line of voice out, template fallback

Nothing here is load-bearing. Every call site has a deterministic template fallback, the internal
fallback (not a shipped personality) never calls a model, and the `echo` provider makes the whole
layer testable offline (ADR-008).
"""

from .base import (
    Completion,
    LLMError,
    LLMProvider,
    LLMRefused,
    LLMUnavailable,
    Message,
    ProviderCaps,
    Role,
    UsageLog,
    system,
    user,
)
from .providers import (
    AnthropicProvider,
    EchoProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    build_provider,
)
from .render import PLAIN, PersonalityRenderer, Rendered
from .safety import FilterResult, audit_personality, check

__all__ = [
    "PLAIN",
    "AnthropicProvider",
    "Completion",
    "EchoProvider",
    "FilterResult",
    "LLMError",
    "LLMProvider",
    "LLMRefused",
    "LLMUnavailable",
    "Message",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "PersonalityRenderer",
    "ProviderCaps",
    "Rendered",
    "Role",
    "UsageLog",
    "audit_personality",
    "build_provider",
    "check",
    "system",
    "user",
]
