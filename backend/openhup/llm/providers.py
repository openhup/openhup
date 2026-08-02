"""Concrete LLM providers.

`ollama` is the default because it is the shortest path to a local model on a self-hosted box.
`openai_compatible` covers llama.cpp's server, vLLM, LM Studio, LocalAI, and OpenRouter with one
implementation, which is most of the ecosystem for free. `anthropic` is there because some people
will want the better model and should be able to make that choice knowingly.

`echo` is not a toy: it is what makes the test suite deterministic and what `--offline` uses.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Sequence
from typing import Any

import httpx

from .base import (
    Completion,
    LLMRefused,
    LLMUnavailable,
    Message,
    ProviderCaps,
    Role,
)

DEFAULT_TIMEOUT_S = 60.0


class EchoProvider:
    """Deterministic provider for tests and offline mode.

    Returns canned responses keyed by a marker in the prompt. It never fails, never varies, and
    never talks to anything - so a test asserting that a task got phrased a particular way is
    asserting about OpenHup's code rather than about a model's mood.
    """

    def __init__(self, responses: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.caps = ProviderCaps(
            name="echo", local=True, json_mode=True, vision=True, typical_latency_s=0.0
        )
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_s: float | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        if self.fail:
            raise LLMUnavailable("echo provider configured to fail")

        prompt = "\n".join(m.content for m in messages)
        for marker, response in self.responses.items():
            if marker in prompt:
                return Completion(text=response, provider="echo", model="echo")

        if json_schema is not None:
            # Without a canned answer, produce something schema-shaped enough to exercise the
            # repair path deliberately rather than by accident.
            return Completion(text="{}", provider="echo", model="echo")
        last_user = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        return Completion(text=last_user[:200], provider="echo", model="echo")

    async def describe_image(
        self, image: bytes, prompt: str, *, timeout_s: float | None = None
    ) -> Completion:
        if self.fail:
            raise LLMUnavailable("echo provider configured to fail")
        return Completion(
            text=self.responses.get("__image__", f"an image of {len(image)} bytes"),
            provider="echo",
            model="echo",
        )

    async def health(self) -> bool:
        return not self.fail


class OllamaProvider:
    """Local models via Ollama's native API.

    Uses `format: json` for structured output, which Ollama implements by constraining generation -
    considerably more reliable than asking politely.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:7b-instruct",
        vision_model: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        keep_alive: str = "10m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.keep_alive = keep_alive
        self._client = client
        self.caps = ProviderCaps(
            name="ollama",
            local=True,
            json_mode=True,
            vision=vision_model is not None,
            context_tokens=8192,
            # A 7B model on CPU is genuinely slow. The gateway's budget accounts for it.
            typical_latency_s=8.0,
        )

    async def _post(self, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        client = self._client or httpx.AsyncClient(timeout=timeout_s)
        try:
            response = await client.post(f"{self.base_url}{path}", json=payload, timeout=timeout_s)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"ollama at {self.base_url}: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_s: float | None = None,
    ) -> Completion:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_schema is not None:
            # Ollama accepts a JSON Schema here on recent versions and "json" on older ones.
            payload["format"] = json_schema
        data = await self._post("/api/chat", payload, timeout_s or DEFAULT_TIMEOUT_S)
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMRefused("ollama returned an empty completion")
        return Completion(
            text=text,
            provider="ollama",
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def describe_image(
        self, image: bytes, prompt: str, *, timeout_s: float | None = None
    ) -> Completion:
        if not self.vision_model:
            raise LLMUnavailable("no vision model configured for ollama")
        data = await self._post(
            "/api/generate",
            {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [base64.b64encode(image).decode()],
                "stream": False,
            },
            timeout_s or DEFAULT_TIMEOUT_S * 2,
        )
        return Completion(text=data.get("response", ""), provider="ollama", model=self.vision_model)

    async def health(self) -> bool:
        client = self._client or httpx.AsyncClient(timeout=5)
        try:
            response = await client.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
        finally:
            if self._client is None:
                await client.aclose()


class OpenAICompatibleProvider:
    """Anything speaking the OpenAI chat-completions API.

    Covers llama.cpp's server, vLLM, LM Studio, LocalAI, OpenRouter, and OpenAI itself. `local` is
    a constructor argument rather than an inference, because whether the endpoint is on your LAN is
    a fact about your deployment that the gateway needs in order to enforce the egress policy.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        local: bool = False,
        supports_json: bool = True,
        vision: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = client
        self.caps = ProviderCaps(
            name="openai_compatible",
            local=local,
            json_mode=supports_json,
            vision=vision,
            typical_latency_s=2.0 if not local else 6.0,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_s: float | None = None,
    ) -> Completion:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None and self.caps.json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "openhup", "schema": json_schema, "strict": True},
            }

        client = self._client or httpx.AsyncClient(timeout=timeout_s or DEFAULT_TIMEOUT_S)
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout_s or DEFAULT_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"openai-compatible endpoint {self.base_url}: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

        choices = data.get("choices") or []
        if not choices:
            raise LLMRefused("no choices in response")
        usage = data.get("usage") or {}
        return Completion(
            text=choices[0]["message"]["content"] or "",
            provider="openai_compatible",
            model=data.get("model", self.model),
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def describe_image(
        self, image: bytes, prompt: str, *, timeout_s: float | None = None
    ) -> Completion:
        if not self.caps.vision:
            raise LLMUnavailable("vision not enabled for this endpoint")
        encoded = base64.b64encode(image).decode()
        message = Message(
            Role.USER,
            json.dumps(
                [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ]
            ),
        )
        return await self.complete([message], timeout_s=timeout_s)

    async def health(self) -> bool:
        client = self._client or httpx.AsyncClient(timeout=5)
        try:
            response = await client.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False
        finally:
            if self._client is None:
                await client.aclose()


class AnthropicProvider:
    """Anthropic's Messages API.

    Remote by definition, so the gateway will refuse to use it unless `allow_remote_llm` is set and
    a redaction profile is chosen. That refusal is a feature.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        *,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.caps = ProviderCaps(
            name="anthropic",
            local=False,
            json_mode=False,  # tool-use gives structure; the repair loop covers the gap
            vision=True,
            context_tokens=200_000,
            typical_latency_s=2.5,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_s: float | None = None,
    ) -> Completion:
        started = time.perf_counter()
        system_prompt = "\n\n".join(m.content for m in messages if m.role is Role.SYSTEM)
        turns = [m.as_dict() for m in messages if m.role is not Role.SYSTEM]
        if json_schema is not None:
            system_prompt += (
                "\n\nRespond with a single JSON object and no other text. It must validate "
                f"against this schema:\n{json.dumps(json_schema)}"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system_prompt:
            payload["system"] = system_prompt

        client = self._client or httpx.AsyncClient(timeout=timeout_s or DEFAULT_TIMEOUT_S)
        try:
            response = await client.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=self._headers(),
                timeout=timeout_s or DEFAULT_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"anthropic: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

        blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        usage = data.get("usage") or {}
        return Completion(
            text="".join(blocks),
            provider="anthropic",
            model=data.get("model", self.model),
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )

    async def describe_image(
        self, image: bytes, prompt: str, *, timeout_s: float | None = None
    ) -> Completion:
        encoded = base64.b64encode(image).decode()
        payload = {
            "model": self.model,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        client = self._client or httpx.AsyncClient(timeout=timeout_s or DEFAULT_TIMEOUT_S)
        try:
            response = await client.post(
                f"{self.base_url}/v1/messages", json=payload, headers=self._headers()
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"anthropic vision: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return Completion(text="".join(blocks), provider="anthropic", model=self.model)

    async def health(self) -> bool:
        # No cheap unauthenticated probe; assume reachable and let the first real call fail loudly.
        return bool(self.api_key)


def build_provider(config: Any) -> Any:
    """Construct a provider from `LLMSettings`. Kept here so `core.config` stays declarative."""
    kind = getattr(config, "provider", "ollama")
    if kind == "echo":
        return EchoProvider()
    if kind == "ollama":
        return OllamaProvider(
            base_url=config.base_url or "http://127.0.0.1:11434",
            model=config.model,
            vision_model=config.vision_model,
        )
    if kind == "openai_compatible":
        return OpenAICompatibleProvider(
            base_url=config.base_url or "http://127.0.0.1:8000/v1",
            model=config.model,
            api_key=config.api_key,
            local=config.treat_as_local,
            vision=bool(config.vision_model),
        )
    if kind == "anthropic":
        if not config.api_key:
            raise ValueError("anthropic provider requires llm.api_key")
        # `base_url` defaults to the Ollama URL; treat that as "unset" for Anthropic so an
        # existing config without a base_url keeps working against the public API.
        base_url = config.base_url
        if not base_url or base_url.rstrip("/") == "http://127.0.0.1:11434":
            base_url = "https://api.anthropic.com"
        return AnthropicProvider(api_key=config.api_key, model=config.model, base_url=base_url)
    raise ValueError(f"unknown LLM provider {kind!r}")


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "AnthropicProvider",
    "EchoProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "build_provider",
]
