"""Configuration contract tests.

Configuration is part of the product boundary: a typo must fail loudly, a remote provider must
never become active accidentally, and generated secrets must be stable and private. These tests use
only temporary files and environment variables, so they exercise the real settings loader offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openhup.core.config import LLMSettings, SecuritySettings, Settings, VoiceSettings


def test_load_merges_top_level_yaml_sections(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "instance_name": "Base House",
                "database": {"url": "sqlite+aiosqlite:///base.db", "pool_size": 3},
                "security": {"bind_port": 9000},
            }
        )
    )
    override.write_text(
        yaml.safe_dump(
            {
                "instance_name": "Override House",
                "database": {"pool_size": 9},
            }
        )
    )

    settings = Settings.load(base, override)

    assert settings.instance_name == "Override House"
    assert settings.database.url == "sqlite+aiosqlite:///base.db"
    assert settings.database.pool_size == 9
    assert settings.security.bind_port == 9000


def test_environment_overrides_yaml(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("database:\n  pool_size: 2\n")
    monkeypatch.setenv("OPENHUP__DATABASE__POOL_SIZE", "17")

    settings = Settings.load(path)

    assert settings.database.pool_size == 17


def test_missing_config_files_use_safe_defaults() -> None:
    settings = Settings.load("does-not-exist.yaml")

    assert settings.database.is_sqlite is False
    assert settings.security.bind_host == "127.0.0.1"
    assert settings.security.require_auth is True
    assert settings.ux.hide_task_counts is True


def test_secret_key_is_created_once_and_reused(tmp_path: Path) -> None:
    settings = Settings(state_dir=str(tmp_path), llm={"provider": "echo"})

    first = settings.resolved_secret_key()
    key_path = tmp_path / "secret_key"
    second = settings.resolved_secret_key()

    assert first
    assert first == second
    assert key_path.read_text() == first
    assert key_path.stat().st_mode & 0o777 == 0o600


def test_configured_secret_key_does_not_touch_disk(tmp_path: Path) -> None:
    settings = Settings(
        state_dir=str(tmp_path),
        security={"secret_key": "already-configured"},
        llm={"provider": "echo"},
    )

    assert settings.resolved_secret_key() == "already-configured"
    assert not (tmp_path / "secret_key").exists()


@pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
def test_remote_llm_requires_explicit_consent(provider: str) -> None:
    with pytest.raises(ValidationError, match="allow_remote_llm"):
        LLMSettings(provider=provider, base_url="https://provider.example")


def test_local_openai_compatible_llm_is_allowed() -> None:
    settings = LLMSettings(
        provider="openai_compatible",
        base_url="http://llm:8000/v1",
        treat_as_local=True,
    )

    assert settings.allow_remote_llm is False


def test_remote_voice_requires_explicit_consent() -> None:
    with pytest.raises(ValidationError, match="allow_remote_voice"):
        VoiceSettings(stt_provider="openai")


def test_local_voice_gateway_is_not_remote() -> None:
    settings = VoiceSettings(
        stt_provider="openai_compatible",
        tts_provider="openai_compatible",
        base_url="http://speech:8080/v1",
        treat_as_local=True,
    )

    assert settings.stt_remote is False
    assert settings.tts_remote is False


def test_non_loopback_without_auth_is_rejected() -> None:
    with pytest.raises(ValidationError, match="require_auth"):
        SecuritySettings(bind_host="0.0.0.0", require_auth=False)


def test_warnings_surface_privacy_relevant_choices() -> None:
    settings = Settings(
        llm={"provider": "anthropic", "allow_remote_llm": True, "redaction_profile": "full"},
        voice={"stt_provider": "openai", "allow_remote_voice": True},
        security={"bind_host": "0.0.0.0"},
    )

    warnings = settings.warnings()

    assert any("reverse proxy" in warning for warning in warnings)
    assert any("unredacted snapshots" in warning for warning in warnings)
    assert any("remote voice provider" in warning for warning in warnings)
    assert any("no notification channels" in warning for warning in warnings)


def test_sqlite_database_settings_are_detected() -> None:
    settings = Settings(
        database={"url": "sqlite+aiosqlite:///tmp/test.db"},
        llm={"provider": "echo"},
    )

    assert settings.database.is_sqlite is True
