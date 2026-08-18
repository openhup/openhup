"""Deployment configuration.

Layered, in increasing precedence: defaults in this file → `/etc/openhup/config.yaml` →
environment variables (`OPENHUP__SECTION__KEY`) → command-line flags. YAML is the primary surface
because most of this is written once and then read by a human six months later; the environment is
for secrets and for container overrides.

The privacy-relevant settings are grouped and commented, because they are the ones people need to
find and understand before trusting this with a camera in their kitchen.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from openhup_schemas import Duration, PersonalitySettings
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatabaseSettings(_Section):
    url: str = "postgresql+asyncpg://openhup:openhup@127.0.0.1:5432/openhup"
    pool_size: int = 5
    max_overflow: int = 5
    echo: bool = False

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


class BusSettings(_Section):
    url: str = "redis://127.0.0.1:6379/0"
    #: Cap on the observation stream: roughly a day of history at 1/s, which is what
    #: `/skills/{id}/simulate` replays. Raise it if you want deeper dry-runs.
    observation_maxlen: int = 100_000
    consumer_block_ms: int = 2_000
    #: How long a claimed-but-unacknowledged message waits before another consumer may take it.
    claim_after: Duration = Field(default_factory=lambda: timedelta(minutes=2))


class EngineSettings(_Section):
    """The skill engine worker."""

    #: The tick that lets `for:` and `absent_for:` become true without new data arriving. One second
    #: is cheap - evaluation is pure in-memory work over short ring buffers.
    tick: Duration = Field(default_factory=lambda: timedelta(seconds=1))
    #: Only one engine may run per deployment or every task would be created twice. Held as a Redis
    #: lock and renewed; a second instance waits as a warm standby.
    leader_lock_key: str = "openhup:engine:leader"
    leader_ttl: Duration = Field(default_factory=lambda: timedelta(seconds=30))
    #: Rebuild signal windows from stored observations on startup, so a restart does not reset every
    #: `for: 4h` condition to zero.
    warm_start: bool = True
    warm_start_window: Duration = Field(default_factory=lambda: timedelta(hours=6))
    #: Pause every skill. For holidays, guests, or a house full of builders.
    paused: bool = False


class LLMSettings(_Section):
    provider: Literal["ollama", "openai_compatible", "anthropic", "echo"] = "ollama"
    base_url: str | None = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    vision_model: str | None = None
    api_key: str | None = None
    timeout: Duration = Field(default_factory=lambda: timedelta(seconds=30))

    # -- egress policy ----------------------------------------------------------------
    #: Must be set explicitly before any non-local provider is allowed to run. Without it a remote
    #: provider is refused at startup rather than quietly used.
    allow_remote_llm: bool = False
    #: What may be sent to a remote provider.
    #:   text_only      - facts and labels only. No images, no raw object inventories.
    #:   redacted_image - snapshots with people blurred.
    #:   full           - anything. You are choosing this knowingly.
    redaction_profile: Literal["text_only", "redacted_image", "full"] = "text_only"
    #: Treat an openai_compatible endpoint as local (llama.cpp or vLLM on your own LAN).
    treat_as_local: bool = False

    @model_validator(mode="after")
    def _enforce_egress_policy(self) -> Self:
        remote = self.provider == "anthropic" or (
            self.provider == "openai_compatible" and not self.treat_as_local
        )
        if remote and not self.allow_remote_llm:
            raise ValueError(
                f"llm.provider is {self.provider!r}, which sends data off your network. "
                "Set llm.allow_remote_llm: true to confirm, and review llm.redaction_profile. "
                "See docs/SECURITY_PRIVACY.md."
            )
        return self


class SnapshotSettings(_Section):
    directory: str = "/var/lib/openhup/snapshots"
    #: Serve snapshots through the API rather than exposing the directory to the reverse proxy, so
    #: authentication actually applies to imagery of your home.
    serve_via_api: bool = True
    max_bytes: int = 5 * 1024**3
    #: Global retention ceiling. A skill asking for longer is clamped to this.
    max_retention: Duration = Field(default_factory=lambda: timedelta(days=90))


class SecuritySettings(_Section):
    #: Loopback by default. Change to 0.0.0.0 only behind a reverse proxy or on a trusted LAN.
    bind_host: str = "127.0.0.1"
    bind_port: int = 8080
    #: Signs session cookies. Generated on first run into the state directory if unset.
    secret_key: str | None = None
    session_lifetime: Duration = Field(default_factory=lambda: timedelta(days=30))
    #: Long-lived bearer tokens for vision services and camera agents.
    service_tokens: dict[str, str] = Field(default_factory=dict)
    #: Trust X-Forwarded-* headers. Only enable when genuinely behind a proxy, or client IPs can be
    #: spoofed in the audit log.
    trust_proxy_headers: bool = False
    cors_origins: list[str] = Field(default_factory=list)
    #: Refuse to start when bound to a non-loopback address with no authentication configured.
    require_auth: bool = True

    @model_validator(mode="after")
    def _refuse_open_exposure(self) -> Self:
        exposed = self.bind_host not in {"127.0.0.1", "localhost", "::1"}
        if exposed and not self.require_auth:
            raise ValueError(
                f"security.bind_host is {self.bind_host!r} with require_auth disabled. That "
                "publishes live camera imagery of your home to the network with no login. "
                "Either bind to 127.0.0.1 behind a reverse proxy, or leave require_auth on."
            )
        return self


class VoiceSettings(_Section):
    """Voice interface: speech-to-text and text-to-speech.

    This is a *voice interface*, not ambient surveillance. The microphone is gated behind a wake
    word that is matched in the client, nothing is recorded, and by default nothing leaves the
    device at all - `browser` uses the Web Speech API inside the PWA. The server only ever sees
    audio when a remote provider is explicitly configured below. See docs/VOICE.md.
    """

    #: Master switch. False hides the microphone in the UI entirely.
    enabled: bool = True
    #: Where speech recognition runs. `browser` = the Web Speech API inside the PWA (nothing is
    #: uploaded to OpenHup). `openai` / `openai_compatible` = server-side transcription.
    stt_provider: Literal["browser", "openai", "openai_compatible"] = "browser"
    #: Where synthesis runs. `browser` = the browser's built-in speech synthesis.
    tts_provider: Literal["browser", "openai", "openai_compatible"] = "browser"
    #: Endpoint for the remote providers. `openai` defaults to the public API; `openai_compatible`
    #: requires a base_url (a local whisper.cpp/Piper gateway counts as local with treat_as_local).
    base_url: str | None = None
    api_key: str | None = None
    stt_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    #: Wake phrase, matched in the client before anything is processed.
    wake_word: str = "hey openhup"
    language: str = "en"
    #: Treat an openai_compatible endpoint as running on hardware you control.
    treat_as_local: bool = False
    #: Must be set explicitly before audio or its transcript leaves your network. Mirrors
    #: llm.allow_remote_llm - the privacy promise is the same, so the gate is the same shape.
    allow_remote_voice: bool = False

    @model_validator(mode="after")
    def _enforce_egress_policy(self) -> Self:
        remote = any(
            p == "openai" or (p == "openai_compatible" and not self.treat_as_local)
            for p in (self.stt_provider, self.tts_provider)
        )
        if remote and not self.allow_remote_voice:
            raise ValueError(
                "voice.stt_provider or voice.tts_provider sends audio or its transcript off your "
                "network. Set voice.allow_remote_voice: true to confirm. See docs/VOICE.md."
            )
        return self

    @property
    def stt_remote(self) -> bool:
        return self.stt_provider == "openai" or (
            self.stt_provider == "openai_compatible" and not self.treat_as_local
        )

    @property
    def tts_remote(self) -> bool:
        return self.tts_provider == "openai" or (
            self.tts_provider == "openai_compatible" and not self.treat_as_local
        )


class IdentitySettings(_Section):
    """Consent-gated household identity: who is in a room, only ever for people who said yes.

    This is the ADR-016 reversal, recorded with its guardrails rather than hidden: the project's
    original position was "no face recognition at all", and that position was about *identifying
    people without consent*. This setting keeps the consent flow as the gate - an embedding is
    only ever stored for a member who answered yes, an unknown face is never persisted, and the
    24-hour no-reask marker stores no biometric data.
    """

    #: Master switch. On by default (the household opted in at first-run consent, ADR-016): the
    #: face_id detector runs and an unknown face earns the consent question. The consent flow is
    #: the gate - nobody is *known* until they say yes, and the models must be fetched separately
    #: like every other opt-in weight. Set false to remove identity entirely.
    enabled: bool = True
    #: Minimum cosine similarity for a gallery match. Below this, a face is unknown and earns the
    #: consent question rather than a confident (and wrong) name.
    match_threshold: float = 0.55
    #: How often the face_id detector runs on an anchor that uses it. Slower than object detection
    #: on purpose: identity is presence, and presence changes slowly.
    min_interval: Duration = Field(default_factory=lambda: timedelta(seconds=15))


class NotifySettings(_Section):
    #: Channel definitions, keyed by id. Shape depends on `type`; see examples/notifications/.
    channels: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: Hold non-urgent notifications during these hours; they surface in the UI immediately and are
    #: delivered when the window ends. High-urgency alerts ignore this.
    quiet_hours: dict[str, Any] | None = None
    #: Per-channel ceiling, so a miscalibrated skill cannot empty your phone battery.
    max_per_hour: int = 12


class UXSettings(_Section):
    """Defaults for the interface. Every one of these is per-user overridable."""

    #: Show one task at a time across the whole app, not just per skill.
    global_single_task_focus: bool = False
    #: Attach a snapshot to every task view. Visual anchoring is the core accessibility affordance.
    show_snapshots: bool = True
    #: Never show a count of outstanding tasks anywhere. Backlog counts are the main reason people
    #: abandon tools in this category.
    hide_task_counts: bool = True
    #: Surface before/after pairs on completion.
    celebrate_completions: bool = True
    locale: str = "en"
    timezone: str = "UTC"


class Settings(BaseSettings):
    """Root configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENHUP__",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Keep environment overrides stronger than YAML/constructor section values.

        `Settings.load()` supplies nested dictionaries from YAML as init values. Pydantic Settings
        otherwise treats that whole nested section as explicit and never applies a more specific
        `OPENHUP__SECTION__KEY` variable, contrary to the documented precedence.
        """
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )

    instance_name: str = "OpenHup"
    state_dir: str = "/var/lib/openhup"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    bus: BusSettings = Field(default_factory=BusSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    snapshots: SnapshotSettings = Field(default_factory=SnapshotSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)
    ux: UXSettings = Field(default_factory=UXSettings)

    #: Where cameras, anchors, skills, and personalities are loaded from on startup.
    config_dir: str = "/etc/openhup"

    @classmethod
    def load(cls, *paths: str | Path) -> Settings:
        """Read YAML files then let the environment override.

        Explicit rather than magic: the file is authoritative for structure, the environment for
        secrets. That keeps `config.yaml` safe to commit in a personal dotfiles repo.
        """
        merged: dict[str, Any] = {}
        candidates = paths or (
            os.environ.get("OPENHUP_CONFIG", ""),
            "/etc/openhup/config.yaml",
            "./config.yaml",
        )
        for path in candidates:
            if not path:
                continue
            candidate = Path(path)
            if not candidate.is_file():
                continue
            data = yaml.safe_load(candidate.read_text()) or {}
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        return cls(**merged)

    @property
    def snapshot_dir(self) -> Path:
        return Path(self.snapshots.directory)

    def resolved_secret_key(self) -> str:
        """Return the signing key, generating and persisting one on first run."""
        if self.security.secret_key:
            return self.security.secret_key
        key_file = Path(self.state_dir) / "secret_key"
        if key_file.is_file():
            return key_file.read_text().strip()
        import secrets

        key = secrets.token_urlsafe(48)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(key)
        key_file.chmod(0o600)
        return key

    def warnings(self) -> list[str]:
        """Configuration smells worth telling the operator about at startup.

        Deliberately not errors: someone running LAN-only on a trusted network is making a
        legitimate choice, and being lectured by a startup crash is not helpful. Being told once,
        clearly, is.
        """
        notes: list[str] = []
        if self.security.bind_host not in {"127.0.0.1", "localhost", "::1"}:
            notes.append(
                f"listening on {self.security.bind_host}: make sure a reverse proxy with TLS is in "
                f"front, or restrict this to your LAN. See docs/SECURITY_PRIVACY.md."
            )
        if self.llm.allow_remote_llm and self.llm.redaction_profile == "full":
            notes.append(
                "remote LLM with redaction_profile 'full': unredacted snapshots of your home may "
                "be sent to a third party. This is logged per call at /api/v1/system/llm-usage."
            )
        if self.personality.roast_consent and self.personality.humor_ceiling >= 4:
            notes.append(
                "roast mode is enabled at high intensity. Safety alerts are still phrased plainly, "
                "and the boundary filter still applies."
            )
        if not self.notify.channels:
            notes.append("no notification channels configured: alerts appear in the UI only.")
        if self.voice.allow_remote_voice and (self.voice.stt_remote or self.voice.tts_remote):
            notes.append(
                "remote voice provider enabled: spoken commands and/or replies are sent to a "
                "third party. This is logged per call at /api/v1/system/llm-usage."
            )
        return notes


def load_settings(*paths: str | Path) -> Settings:
    return Settings.load(*paths)


__all__ = [
    "BusSettings",
    "DatabaseSettings",
    "EngineSettings",
    "IdentitySettings",
    "LLMSettings",
    "NotifySettings",
    "SecuritySettings",
    "Settings",
    "SnapshotSettings",
    "UXSettings",
    "VoiceSettings",
    "load_settings",
]
