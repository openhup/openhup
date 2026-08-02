"""The first-run setup wizard (`openhup setup`).

The behaviour under test: the personality step offers exactly the five gamble voices, a pick and a
gamble are both possible, a gamble is written to config as `gamble: true` (the draw happens at
first launch), and - the whole point - **nothing the wizard prints or writes ever names a gambled
voice**. The mystery is the feature.

Since the wizard is also the whole first run, it is tested for the bootstrap (config/ from the
shipped examples), the environment file (real random secrets), the inference-profile question, and
the guided handoff (exact commands, one at a time, waiting for confirmation).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openhup.personality import GAMBLE_POOL
from openhup.setup import (
    bootstrap_files,
    decide_profile,
    decide_voice,
    decide_voice_runtime,
    generate_env,
    next_steps,
    personality_descriptions,
    run_setup,
    voice_section,
    write_config,
)


def test_pool_has_exactly_five_voices() -> None:
    assert len(GAMBLE_POOL) == 5


def test_pick_by_number() -> None:
    for index, personality_id in enumerate(GAMBLE_POOL, start=1):
        choice = decide_voice(str(index))
        assert choice.default_personality == personality_id
        assert not choice.gamble


def test_pick_by_id() -> None:
    choice = decide_voice("sarcastic")
    assert choice.default_personality == "sarcastic"
    assert not choice.gamble


def test_gamble_answers() -> None:
    for answer in ("gamble", "random", "mystery", "surprise", "g", "?"):
        choice = decide_voice(answer)
        assert choice.gamble
        assert choice.default_personality is None


def test_default_keeps_the_stock_voice() -> None:
    for answer in ("", "default", "d"):
        choice = decide_voice(answer)
        assert not choice.gamble
        assert choice.default_personality is None


def test_brief_is_an_explicit_pick() -> None:
    choice = decide_voice("brief")
    assert choice.default_personality == "brief"
    assert not choice.gamble


def test_out_of_range_and_nonsense_are_refused() -> None:
    for answer in ("0", "6", "999", "purple", "yes"):
        with pytest.raises(ValueError):
            decide_voice(answer)


def test_personality_descriptions_come_from_the_shipped_presets() -> None:
    presets = Path(__file__).resolve().parents[2] / "examples/personalities/personalities.yaml"
    descriptions = personality_descriptions(presets)
    assert set(descriptions) == set(GAMBLE_POOL)
    for personality_id in GAMBLE_POOL:
        assert personality_id in descriptions[personality_id].lower()


def test_write_config_pick(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    merged = write_config(
        path,
        instance_name="My House",
        voice=decide_voice("3"),
        voice_enabled=True,
    )
    assert merged["personality"]["default_personality"] == GAMBLE_POOL[2]
    assert "gamble" not in merged["personality"]
    assert merged["voice"]["enabled"] is True
    assert merged["instance_name"] == "My House"
    # And the file on disk round-trips.
    assert yaml.safe_load(path.read_text())["personality"]["default_personality"] == GAMBLE_POOL[2]


def test_write_config_gamble_never_names_a_voice(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    merged = write_config(
        path,
        instance_name="OpenHup",
        voice=decide_voice("gamble"),
        voice_enabled=True,
    )
    assert merged["personality"]["gamble"] is True
    assert "default_personality" not in merged["personality"]
    # Nothing in the written file could name the future draw.
    assert not any(pid in path.read_text() for pid in GAMBLE_POOL)


def test_write_config_merges_over_an_existing_file(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"llm": {"provider": "ollama"}, "personality": {"roast_consent": True}})
    )
    merged = write_config(
        path,
        instance_name="OpenHup",
        voice=decide_voice("1"),
        voice_enabled=False,
    )
    # Unrelated keys survive; the personality and voice keys are updated, not duplicated.
    assert merged["llm"]["provider"] == "ollama"
    assert merged["personality"]["roast_consent"] is True
    assert merged["personality"]["default_personality"] == GAMBLE_POOL[0]
    assert merged["voice"]["enabled"] is False


def test_run_setup_pick_flow(tmp_path) -> None:
    # name, voice #2, voice on, runtime Enter (browser), Enter=ollama, model Enter
    answers = iter(["My House", "2", "", "", "", ""])
    echoes: list[str] = []
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=echoes.append,
    )
    assert config["personality"]["default_personality"] == GAMBLE_POOL[1]
    assert config["voice"]["enabled"] is True
    # The AI step is required and defaults to local Ollama.
    assert config["llm"]["provider"] == "ollama"
    assert "allow_remote_llm" not in config["llm"]
    assert any(f"Voice set to {GAMBLE_POOL[1]}" in line for line in echoes)


def test_run_setup_gamble_flow_never_names_the_voice(tmp_path) -> None:
    answers = iter(
        ["OpenHup", "gamble", "n", "", ""]
    )  # name, gamble, voice off, Enter=ollama, model Enter
    echoes: list[str] = []
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=echoes.append,
    )
    assert config["personality"]["gamble"] is True
    assert config["voice"]["enabled"] is False
    assert config["llm"]["provider"] == "ollama"
    joined = " ".join(echoes)
    # The wizard shows the five voices as options (by display name), but never says which one
    # will be drawn.
    for personality_id in GAMBLE_POOL:
        assert personality_id.title() in joined  # shown as the menu
    assert "mystery voice will be drawn" in joined
    assert "will not be announced" in joined


def test_run_setup_reasks_on_a_bad_answer(tmp_path) -> None:
    # bad voice, real pick, voice on, runtime Enter, Enter=ollama, model Enter
    answers = iter(["OpenHup", "purple", "1", "", "", "", ""])
    run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    # The bad answer was rejected and the flow continued to a real pick.
    assert (
        yaml.safe_load((tmp_path / "config.yaml").read_text())["personality"]["default_personality"]
        == GAMBLE_POOL[0]
    )


def test_run_setup_cloud_provider_requires_egress_confirmation(tmp_path) -> None:
    # name, voice #1, voice on, runtime Enter (browser), provider 4 (anthropic), model Enter,
    # base URL Enter (default), api key, confirm yes, redaction Enter (text_only)
    answers = iter(["OpenHup", "1", "y", "", "4", "", "", "sk-ant-123", "yes", ""])
    env_path = tmp_path / "openhup.env"
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        env_path=env_path,
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    # The cloud pick is written with the egress gate open and a redaction profile...
    assert config["llm"]["provider"] == "anthropic"
    assert config["llm"]["allow_remote_llm"] is True
    assert config["llm"]["redaction_profile"] == "text_only"
    assert config["llm"]["base_url"] == "https://api.anthropic.com"
    # ...and the key goes to the environment file, never into config.yaml.
    assert "sk-ant-123" not in (tmp_path / "config.yaml").read_text()
    assert "OPENHUP__LLM__API_KEY=sk-ant-123" in env_path.read_text()


def test_run_setup_cloud_refusal_goes_back_to_a_local_provider(tmp_path) -> None:
    # name, voice #1, voice on, runtime Enter (browser), provider 3 (openai), model Enter,
    # base URL Enter, key blank, confirm no, then Enter again = local ollama, model Enter
    answers = iter(["OpenHup", "1", "y", "", "3", "", "", "", "no", "", ""])
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    assert config["llm"]["provider"] == "ollama"
    assert "allow_remote_llm" not in config["llm"]


def test_run_setup_openai_compatible_local_gateway(tmp_path) -> None:
    # name, voice #1, voice on, runtime Enter (browser), provider 2 (openai_compatible),
    # base URL Enter, model llama3.1:8b
    answers = iter(["OpenHup", "1", "y", "", "2", "", "llama3.1:8b"])
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    assert config["llm"]["provider"] == "openai_compatible"
    assert config["llm"]["treat_as_local"] is True
    assert config["llm"]["model"] == "llama3.1:8b"


# --------------------------------------------------------------------------- bootstrap


def _fake_repo(tmp_path, *, with_compose: bool = True) -> Path:
    """A minimal checkout: the shipped examples and a compose file."""
    root = tmp_path / "openhup"
    (root / "config").mkdir(parents=True)
    (root / "examples/cameras").mkdir(parents=True)
    (root / "examples/personalities").mkdir(parents=True)
    (root / "deploy/env").mkdir(parents=True)
    if with_compose:
        (root / "deploy/compose").mkdir(parents=True)
        (root / "deploy/compose/docker-compose.yml").write_text("services: {}")
    (root / "config/config.yaml.example").write_text("instance_name: OpenHup\n")
    (root / "config/vision.yaml.example").write_text("node_id: node1\n")
    (root / "examples/cameras/cameras.yaml").write_text("cameras: []\n")
    (root / "examples/personalities/personalities.yaml").write_text("- id: kind_coach\n")
    env = root / "deploy/env/openhup.env.example"
    env.write_text(
        "POSTGRES_PASSWORD=CHANGE_ME_openssl_rand_base64_32\n"
        "OPENHUP__DATABASE__URL=postgresql+asyncpg://openhup:CHANGE_ME_openssl_rand_base64_32@postgres:5432/openhup\n"
        "OPENHUP_VISION_TOKEN=CHANGE_ME_openssl_rand_hex_32\n"
        "NTFY_TOPIC=openhup-CHANGE_ME_something_unguessable\n"
        "RENDER_GID=109\n"
    )
    return root


def test_bootstrap_files_creates_the_config_directory(tmp_path) -> None:
    root = _fake_repo(tmp_path)
    created = bootstrap_files(root)
    assert len(created) == 4
    assert (root / "config/config.yaml").is_file()
    assert (root / "config/vision.yaml").is_file()
    assert (root / "config/cameras.yaml").is_file()
    assert (root / "config/personalities.yaml").is_file()


def test_bootstrap_files_never_overwrites_an_existing_config(tmp_path) -> None:
    root = _fake_repo(tmp_path)
    (root / "config").mkdir(parents=True, exist_ok=True)
    mine = root / "config/config.yaml"
    mine.write_text("instance_name: My House\n")
    bootstrap_files(root)
    # The user's file is untouched; the missing ones are still created.
    assert mine.read_text() == "instance_name: My House\n"
    assert (root / "config/cameras.yaml").is_file()


def test_generate_env_replaces_every_secret(tmp_path) -> None:
    root = _fake_repo(tmp_path)
    generated = generate_env(root)
    assert generated is not None
    text = generated.read_text()
    assert "CHANGE_ME" not in text
    assert (generated.stat().st_mode & 0o777) == 0o600
    # The database password and the URL's password segment agree (after URL-decoding).
    from urllib.parse import unquote

    def value(prefix: str) -> str:
        return next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith(prefix))

    password = value("POSTGRES_PASSWORD=")
    url = value("OPENHUP__DATABASE__URL=")
    assert unquote(url.split("@")[0].rsplit(":", 1)[1]) == password


def test_generate_env_never_overwrites_an_existing_env_file(tmp_path) -> None:
    root = _fake_repo(tmp_path)
    existing = root / "deploy/env/openhup.env"
    existing.write_text("POSTGRES_PASSWORD=my-real-password\n")
    assert generate_env(root) is None
    assert existing.read_text() == "POSTGRES_PASSWORD=my-real-password\n"


def test_decide_profile_accepts_numbers_and_names() -> None:
    assert decide_profile("") == "cpu"
    assert decide_profile("1") == "cpu"
    assert decide_profile("2") == "openvino"
    assert decide_profile("openvino") == "openvino"
    assert decide_profile("3") == "cuda"
    assert decide_profile("nvidia") == "cuda"
    with pytest.raises(ValueError):
        decide_profile("purple")


def test_next_steps_match_the_config(tmp_path) -> None:
    root = _fake_repo(tmp_path)
    config = {"llm": {"provider": "ollama", "model": "qwen2.5:7b-instruct"}}
    steps = next_steps(root, profile="cpu", config=config)
    titles = [step["title"] for step in steps]
    assert titles == [
        "Start the stack",
        "Pull the qwen2.5:7b-instruct model into Ollama (one time)",
        "Fetch the vision model weights (one time)",
        "Open the app",
    ]
    start = next(s for s in steps if s["title"] == "Start the stack")
    # The compose command carries both the inference and the ollama profile, and says where sudo
    # may be needed.
    assert "--profile cpu" in start["command"]
    assert "--profile ollama" in start["command"]
    assert start["sudo"]


def test_next_steps_skip_ollama_pull_for_a_cloud_provider(tmp_path) -> None:
    root = _fake_repo(tmp_path)
    config = {"llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}
    steps = next_steps(root, profile="openvino", config=config)
    assert all("ollama" not in step["title"] for step in steps)
    assert not any("--profile ollama" in step["command"] for step in steps)
    assert steps[0]["command"].startswith("cd ")


# --------------------------------------------------------------------------- voice runtime


def test_decide_voice_runtime_answers() -> None:
    assert decide_voice_runtime("") == "browser"
    assert decide_voice_runtime("1") == "browser"
    assert decide_voice_runtime("browser") == "browser"
    assert decide_voice_runtime("2") == "openai"
    assert decide_voice_runtime("cloud") == "openai"
    assert decide_voice_runtime("3") == "gateway"
    assert decide_voice_runtime("local") == "gateway"
    with pytest.raises(ValueError):
        decide_voice_runtime("purple")


def test_voice_section_browser_is_explicit() -> None:
    section = voice_section(True, "browser")
    assert section["enabled"] is True
    assert section["stt_provider"] == "browser"
    assert section["tts_provider"] == "browser"
    assert section["allow_remote_voice"] is False


def test_voice_section_cloud_opens_the_egress_gate() -> None:
    section = voice_section(True, "openai")
    assert section["stt_provider"] == "openai"
    assert section["tts_provider"] == "openai"
    assert section["allow_remote_voice"] is True


def test_voice_section_gateway_points_at_the_local_server() -> None:
    section = voice_section(True, "gateway", base_url="http://whisper:8080/v1")
    assert section["stt_provider"] == "openai_compatible"
    assert section["tts_provider"] == "openai_compatible"
    assert section["base_url"] == "http://whisper:8080/v1"
    assert section["treat_as_local"] is True
    assert section["allow_remote_voice"] is False


def test_run_setup_cloud_voice_requires_egress_confirmation(tmp_path) -> None:
    # name, voice #1, voice on, runtime 2 (cloud), base URL Enter (default), api key,
    # confirm yes, provider Enter (ollama), model Enter
    answers = iter(["OpenHup", "1", "y", "2", "", "sk-voice-123", "yes", "", ""])
    env_path = tmp_path / "openhup.env"
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        env_path=env_path,
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    voice = config["voice"]
    assert voice["stt_provider"] == "openai"
    assert voice["tts_provider"] == "openai"
    assert voice["allow_remote_voice"] is True
    assert voice["base_url"] == "https://api.openai.com/v1"
    # The key goes to the env file under the voice variable, never into config.yaml.
    assert "sk-voice-123" not in (tmp_path / "config.yaml").read_text()
    assert "OPENHUP__VOICE__API_KEY=sk-voice-123" in env_path.read_text()


def test_run_setup_cloud_voice_refusal_falls_back_to_the_browser(tmp_path) -> None:
    # name, voice #1, voice on, runtime 2 (cloud), base URL Enter, api key blank, confirm no,
    # then 1 = browser, provider Enter (ollama), model Enter
    answers = iter(["OpenHup", "1", "y", "2", "", "", "no", "1", "", ""])
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    assert config["voice"]["stt_provider"] == "browser"
    assert config["voice"]["allow_remote_voice"] is False


def test_run_setup_local_speech_server_writes_the_gateway(tmp_path) -> None:
    # name, voice #1, voice on, runtime 3 (gateway), base URL Enter, provider Enter (ollama),
    # model Enter
    answers = iter(["OpenHup", "1", "y", "3", "", "", ""])
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    assert config["voice"]["stt_provider"] == "openai_compatible"
    assert config["voice"]["tts_provider"] == "openai_compatible"
    assert config["voice"]["treat_as_local"] is True
    assert config["voice"]["base_url"]


def test_run_setup_cloud_provider_custom_base_url(tmp_path) -> None:
    # name, voice #1, voice on, runtime Enter (browser), provider 3 (openai cloud),
    # model Enter, custom base URL, api key blank, confirm yes, redaction Enter
    answers = iter(
        ["OpenHup", "1", "y", "", "3", "", "https://llm.myserver.example/v1", "", "yes", ""]
    )
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    # Cloud is OpenAI-compatible at the custom endpoint, egress confirmed.
    assert config["llm"]["provider"] == "openai_compatible"
    assert config["llm"]["base_url"] == "https://llm.myserver.example/v1"
    assert config["llm"]["allow_remote_llm"] is True


def test_run_setup_cloud_voice_custom_base_url(tmp_path) -> None:
    # name, voice #1, voice on, runtime 2 (cloud), custom base URL, api key, confirm yes,
    # provider Enter (ollama), model Enter
    answers = iter(
        [
            "OpenHup",
            "1",
            "y",
            "2",
            "https://speech.myserver.example/v1",
            "sk-voice-1",
            "yes",
            "",
            "",
        ]
    )
    env_path = tmp_path / "openhup.env"
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        env_path=env_path,
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    assert config["voice"]["stt_provider"] == "openai"
    assert config["voice"]["base_url"] == "https://speech.myserver.example/v1"
    assert config["voice"]["allow_remote_voice"] is True
    assert "OPENHUP__VOICE__API_KEY=sk-voice-1" in env_path.read_text()


def test_run_setup_local_voice_with_cloud_llm(tmp_path) -> None:
    """The mix the user asked for: local STT/TTS, cloud generation."""
    # name, voice #1, voice on, runtime 3 (gateway), base URL Enter, provider 4 (anthropic),
    # model Enter, base URL Enter, api key, confirm yes, redaction Enter
    answers = iter(["OpenHup", "1", "y", "3", "", "4", "", "", "sk-ant-123", "yes", ""])
    config = run_setup(
        tmp_path / "config.yaml",
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        echo=lambda _line: None,
    )
    # Voice stays local (whisper.cpp/Piper), the brain is cloud Anthropic.
    assert config["voice"]["stt_provider"] == "openai_compatible"
    assert config["voice"]["tts_provider"] == "openai_compatible"
    assert config["voice"]["treat_as_local"] is True
    assert config["voice"]["allow_remote_voice"] is False
    assert config["llm"]["provider"] == "anthropic"
    assert config["llm"]["allow_remote_llm"] is True


def test_run_setup_full_first_run_bootstraps_and_guides(tmp_path) -> None:
    """The whole first run with a fake checkout: bootstrap, secrets, profile, handoff."""
    root = _fake_repo(tmp_path)
    # name, profile Enter (cpu), voice #1, voice on, runtime Enter (browser), provider Enter
    # (ollama), model Enter, then one Enter per handoff step (4: stack, pull, fetch, browser).
    answers = iter(["My House", "", "1", "y", "", "", "", "", "", "", ""])
    confirms: list[str] = []
    echoes: list[str] = []
    config = run_setup(
        root / "config/config.yaml",
        root=root,
        presets_path=Path(__file__).resolve().parents[2]
        / "examples/personalities/personalities.yaml",
        ask=lambda _prompt: next(answers),
        confirm=lambda _prompt: next(confirms) if confirms else "",
        echo=echoes.append,
    )
    # Bootstrap happened: every config file exists, and the env file has no placeholders.
    assert (root / "config/vision.yaml").is_file()
    env = root / "deploy/env/openhup.env"
    assert env.is_file()
    assert "CHANGE_ME" not in env.read_text()
    # The answers were applied.
    assert config["instance_name"] == "My House"
    assert config["llm"]["provider"] == "ollama"
    # The handoff ran: it named the commands and asked for confirmation after each step.
    joined = " ".join(echoes)
    assert "Start the stack" in joined
    assert "docker compose --profile cpu" in joined
    assert "Pull the qwen2.5:7b-instruct model" in joined
    assert "sudo" in joined
    assert "second terminal" in joined
    assert "All set" in joined
