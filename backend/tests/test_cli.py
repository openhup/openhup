"""Regression tests for `openhup-cli check-config`.

The vision file is validated *structurally* (the backend deliberately does not depend on the vision
package). These tests pin the behaviour that a bad vision file fails loudly rather than being
silently swallowed, which was a real bug: a duplicate `problems = []` reset discarded vision errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from openhup.cli import main
from openhup.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]

VISION = """\
node_id: vision-1
backend_url: http://127.0.0.1:8080
bus: {url: redis://127.0.0.1:6379/0}
inference: {}
snapshots: {}
sampling: {}
"""


def _check_config(tmp_path: Path, vision_text: str) -> int:
    vision = tmp_path / "vision.yaml"
    vision.write_text(vision_text)
    return main(
        [
            "check-config",
            "--config",
            str(ROOT / "config" / "config.yaml.example"),
            "--cameras",
            str(ROOT / "examples" / "cameras" / "cameras.yaml"),
            "--personalities",
            str(ROOT / "examples" / "personalities" / "personalities.yaml"),
            "--vision",
            str(vision),
        ]
    )


def test_check_config_accepts_a_valid_vision_file(tmp_path: Path) -> None:
    assert _check_config(tmp_path, VISION) == 0


def test_check_config_rejects_an_unknown_vision_key(tmp_path: Path, capsys) -> None:
    code = _check_config(tmp_path, VISION + "typo_key: true\n")
    assert code == 1
    assert "vision:" in capsys.readouterr().err


def test_check_config_requires_the_core_vision_keys(tmp_path: Path, capsys) -> None:
    code = _check_config(tmp_path, "node_id: vision-1\n")
    assert code == 1
    assert "vision:" in capsys.readouterr().err


def test_provider_none_is_refused() -> None:
    """The AI layer is core (ADR-008): 'none' is not a provider, and a config that says so must
    fail validation rather than silently run brainless."""
    with pytest.raises(ValidationError):
        Settings(llm={"provider": "none"})


def test_provider_echo_is_accepted() -> None:
    """The deterministic offline provider stays a valid choice (tests, --offline, template
    work)."""
    settings = Settings(llm={"provider": "echo"})
    assert settings.llm.provider == "echo"


def test_anthropic_provider_honours_a_custom_base_url() -> None:
    """Cloud does not mean OpenAI's URL only: an Anthropic gateway or proxy endpoint is wired
    through, and an existing config without base_url still uses the public API."""
    from openhup.llm.providers import AnthropicProvider, build_provider

    with_custom = Settings(
        llm={
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "base_url": "https://llm.myserver.example",
            "allow_remote_llm": True,
        }
    )
    provider = build_provider(with_custom.llm)
    assert isinstance(provider, AnthropicProvider)
    assert provider.base_url == "https://llm.myserver.example"

    # No base_url in an existing config: the default is the Ollama URL, which must not leak in.
    no_base = Settings(
        llm={"provider": "anthropic", "api_key": "sk-ant-test", "allow_remote_llm": True}
    )
    provider = build_provider(no_base.llm)
    assert isinstance(provider, AnthropicProvider)
    assert provider.base_url == "https://api.anthropic.com"
