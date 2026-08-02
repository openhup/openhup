"""First-run setup wizard (`openhup setup`) - the whole onboarding, in one command.

It is designed to be the first thing you run after cloning. It:

1. **Bootstraps the config directory** from the shipped examples, if it is empty.
2. **Generates the environment file**: copies the example, then replaces every `CHANGE_ME`
   placeholder with a real random secret (Postgres password, vision token, ntfy topic), so
   `docker compose up` works the first time without editing a file.
3. **Asks the questions that are genuinely painful afterwards** - the voice (ADR-014) and the
   AI provider (the brain is core; there is no "no AI" answer).
4. **Guides the handoff**: prints the exact commands for the steps that need another terminal
   (docker compose, model fetches, browser) and waits for you to confirm each one is done.
   Where a command may need `sudo` (the docker group), it says so in plain text.

The decision logic, config writing, and secret generation are pure functions, so the wizard is
testable without a terminal; `run_setup` is a thin stdin/stdout shell over them.
"""

from __future__ import annotations

import secrets
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml
from openhup_schemas import Personality

from .personality import GAMBLE_POOL

#: One-line descriptions for the gamble voices, used when the shipped presets file is not
#: available (the setup runs from a source checkout, so it usually is). Kept in sync with
#: examples/personalities/personalities.yaml.
_FALLBACK_DESCRIPTIONS: dict[str, str] = {
    "friendly": "warm, casual, genuinely pleased when things get done",
    "shy": "quiet, hesitant, brief - speaks softly and only when it matters",
    "sassy": "playful and cheeky; the cheek is aimed at the mess, never at you",
    "sarcastic": "dry, ironic, deadpan with an edge; the human is never the punchline",
    "angry": "gruff and impatient - at the mess and the clock, never at you",
}

#: Answers that mean "draw it for me, don't tell me".
_GAMBLE_ANSWERS = {"gamble", "random", "mystery", "surprise", "g", "?"}

#: Answers that mean "the stock default, thanks".
_DEFAULT_ANSWERS = {"", "default", "d", "stock"}

#: Answers for the hardware menu: number or name, plus Enter = CPU.
_PROFILE_ANSWERS: dict[str, str] = {
    "": "cpu",
    "1": "cpu",
    "cpu": "cpu",
    "2": "openvino",
    "openvino": "openvino",
    "intel": "openvino",
    "igpu": "openvino",
    "3": "cuda",
    "cuda": "cuda",
    "nvidia": "cuda",
}

#: Example files to bootstrap into config/ when it is empty. First tuple element is the example
#: path relative to the root; second is the destination relative to the root.
_BOOTSTRAP_FILES = (
    ("config/config.yaml.example", "config/config.yaml"),
    ("config/vision.yaml.example", "config/vision.yaml"),
    ("examples/cameras/cameras.yaml", "config/cameras.yaml"),
    ("examples/personalities/personalities.yaml", "config/personalities.yaml"),
)

#: Every secret placeholder in the env example, with a generator that produces a replacement.
#: The database password is deliberately reused between POSTGRES_PASSWORD and the asyncpg URL,
#: because compose reads both and they must agree.
_ENV_PLACEHOLDERS = {
    "CHANGE_ME_openssl_rand_base64_32": lambda: secrets.token_urlsafe(32),
    "CHANGE_ME_openssl_rand_hex_32": lambda: secrets.token_hex(32),
    "CHANGE_ME_something_unguessable": lambda: "openhup-" + secrets.token_urlsafe(8),
}

#: The env example's placeholder for the database URL's password segment.
_URL_PASSWORD_PLACEHOLDER = "CHANGE_ME_openssl_rand_base64_32"


#: Compose profile per inference choice, for the guided handoff.
_PROFILE_TO_COMPOSE = {"cpu": "cpu", "openvino": "openvino", "cuda": "cuda"}

#: Where the vision models fetch runs inside the container (compose mounts ../../models).
_VISION_FETCH_CMD = (
    "docker compose exec vision-cpu python -m openhup_vision.backends --fetch --trust-first-use"
)


@dataclass(frozen=True, slots=True)
class VoiceChoice:
    """What the personality step decided, expressed as config changes."""

    #: Explicit pick, written to personality.default_personality. None when gambling.
    default_personality: str | None = None
    #: Draw a mystery voice at first launch instead of announcing a choice.
    gamble: bool = False


@dataclass(frozen=True, slots=True)
class ProviderChoice:
    """What the AI step decided: where the assistant's brain lives."""

    #: ollama | openai_compatible (local gateway) | openai | anthropic
    provider: str
    model: str
    base_url: str | None = None
    #: True for the two cloud choices: the egress gate the user confirmed by typing "yes".
    allow_remote_llm: bool = False
    redaction_profile: str = "text_only"

    @property
    def local(self) -> bool:
        return not self.allow_remote_llm


#: Model defaults per provider, so the wizard is one Enter away from a working config.
_PROVIDER_DEFAULTS: dict[str, ProviderChoice] = {
    "ollama": ProviderChoice("ollama", "qwen2.5:7b-instruct"),
    "openai_compatible": ProviderChoice(
        "openai_compatible", "", base_url="http://127.0.0.1:8000/v1"
    ),
    "openai": ProviderChoice("openai", "gpt-4o-mini", base_url="https://api.openai.com/v1"),
    "anthropic": ProviderChoice(
        "anthropic", "claude-sonnet-4-5", base_url="https://api.anthropic.com"
    ),
}

#: Answers for the provider menu: number or name, plus Enter = local Ollama.
_PROVIDER_ANSWERS: dict[str, str] = {
    "": "ollama",
    "1": "ollama",
    "ollama": "ollama",
    "2": "openai_compatible",
    "local": "openai_compatible",
    "compatible": "openai_compatible",
    "3": "openai",
    "openai": "openai",
    "4": "anthropic",
    "anthropic": "anthropic",
}

_REDACTION_ANSWERS = {"1": "text_only", "2": "redacted_image", "3": "full"}

#: Answers for the voice-runtime menu: where speech recognition and synthesis run. Enter = the
#: browser, which needs no account and sends nothing anywhere. `cloud` is a first-class choice -
#: a weak box should not be a reason to skip voice - and `gateway` is whisper.cpp/Piper on the
#: user's own network.
_VOICE_RUNTIME_ANSWERS: dict[str, str] = {
    "": "browser",
    "1": "browser",
    "browser": "browser",
    "2": "openai",
    "cloud": "openai",
    "openai": "openai",
    "3": "gateway",
    "gateway": "gateway",
    "local": "gateway",
    "whisper": "gateway",
    "piper": "gateway",
}

#: Default endpoint for the local speech-server choice, mirroring the LLM gateway default.
_VOICE_GATEWAY_DEFAULT_URL = "http://127.0.0.1:8000/v1"

#: Default endpoint for the cloud speech choice. A custom URL is a first-class option: any
#: OpenAI-compatible speech endpoint works, not just OpenAI's.
_VOICE_CLOUD_DEFAULT_URL = "https://api.openai.com/v1"


def personality_descriptions(presets_path: Path) -> dict[str, str]:
    """Display name and description for each gamble voice, from the shipped presets file."""
    try:
        entries = yaml.safe_load(presets_path.read_text()) or []
    except OSError:
        entries = []
    out: dict[str, str] = {}
    for entry in entries:
        personality = Personality.model_validate(entry)
        if personality.id in GAMBLE_POOL:
            description = (personality.description or "").split(".")[0].strip()
            out[personality.id] = f"{personality.display_name} - {description}"
    return out or {pid: f"{pid.title()} - {_FALLBACK_DESCRIPTIONS[pid]}" for pid in GAMBLE_POOL}


def decide_voice(answer: str, *, pool: Sequence[str] = GAMBLE_POOL) -> VoiceChoice:
    """Map the setup answer to config changes.

    * ``""`` or ``"default"`` - keep the stock default (kind_coach).
    * ``"1"``..``"5"`` (or a voice id) - an explicit pick from the pool.
    * ``"gamble"`` / ``"random"`` / ``"mystery"`` - a mystery draw at first launch.
    * ``"brief"`` - the quiet voice: says what needs saying, nothing else. Still the model,
      because no shipped personality switches the AI layer off.

    Raises ValueError for anything else, so the wizard can re-ask rather than guess.
    """
    answer = answer.strip().lower()
    if answer in _DEFAULT_ANSWERS:
        return VoiceChoice(gamble=False)
    if answer in _GAMBLE_ANSWERS:
        return VoiceChoice(gamble=True)
    if answer == "brief":
        return VoiceChoice(default_personality="brief", gamble=False)
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(pool):
            return VoiceChoice(default_personality=pool[index - 1], gamble=False)
    if answer in pool:
        return VoiceChoice(default_personality=answer, gamble=False)
    raise ValueError(
        f"{answer!r} is not one of the choices. Pick a number 1-{len(pool)}, "
        "'gamble', 'brief', or press Enter for the default."
    )


def decide_voice_runtime(answer: str) -> str:
    """Map the voice-runtime answer: where speech runs. Enter means the browser.

    * ``""`` / ``"1"`` / ``"browser"`` - the Web Speech API inside the PWA. No account, and
      nothing (audio or transcript) leaves the device.
    * ``"2"`` / ``"cloud"`` / ``"openai"`` - OpenAI's servers do STT and TTS. First-class on
      purpose: a box without the hardware for local speech is not a reason to skip voice.
    * ``"3"`` / ``"gateway"`` / ``"local"`` - whisper.cpp or Piper on your own network.

    Raises ValueError for anything else, so the wizard can re-ask rather than guess.
    """
    runtime = _VOICE_RUNTIME_ANSWERS.get(answer.strip().lower())
    if runtime is None:
        raise ValueError(
            "that is not one of the choices. Pick 1-3 (or 'browser', 'cloud', 'local'), "
            "or press Enter for the browser."
        )
    return runtime


def decide_profile(answer: str) -> str:
    """Map the hardware answer to an inference profile. Enter means CPU.

    Raises ValueError for anything else, so the wizard can re-ask rather than guess.
    """
    profile = _PROFILE_ANSWERS.get(answer.strip().lower())
    if profile is None:
        raise ValueError(
            "that is not one of the choices. Pick 1-3 (or 'cpu', 'openvino', 'cuda'), "
            "or press Enter for CPU."
        )
    return profile


def bootstrap_files(root: Path) -> list[Path]:
    """Copy the shipped examples into config/ when they are missing.

    Never overwrites: an existing config.yaml is your configuration, and it wins. Returns the
    paths that were actually created, so the wizard can say what it did.
    """
    created: list[Path] = []
    for example_rel, target_rel in _BOOTSTRAP_FILES:
        target = root / target_rel
        if target.exists():
            continue
        example = root / example_rel
        if not example.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example, target)
        created.append(target)
    return created


def generate_env(root: Path, *, profile: str = "cpu") -> Path | None:
    """Create deploy/env/openhup.env from the example, with real random secrets.

    Every CHANGE_ME placeholder is replaced (database password, vision token, ntfy topic), so
    `docker compose up` works on the first try without editing a file. Never overwrites an
    existing env file - that holds your secrets. For the openvino profile the host's render
    group id is appended, so the iGPU is actually used. Returns the path written, or None when
    the example is missing or the file already exists.
    """
    example = root / "deploy/env/openhup.env.example"
    target = root / "deploy/env/openhup.env"
    if not example.is_file() or target.exists():
        return None

    text = example.read_text()
    for placeholder, generate in _ENV_PLACEHOLDERS.items():
        value = generate()
        if placeholder == _URL_PASSWORD_PLACEHOLDER:
            # The URL needs the percent-encoded password; POSTGRES_PASSWORD gets the raw one.
            text = text.replace(placeholder, quote(value, safe=""), 1)
            text = text.replace(placeholder, value, 1)
        else:
            text = text.replace(placeholder, value)

    if profile == "openvino":
        # The example ships the placeholder value; overwrite it with this machine's group id.
        text = text.replace("RENDER_GID=109", f"RENDER_GID={_render_gid()}", 1)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    target.chmod(0o600)
    return target


def _render_gid() -> str:
    """The host's render group id, for the openvino profile. Best-effort: falls back to 109."""
    try:
        out = subprocess.run(
            ["stat", "-c", "%g", "/dev/dri/renderD128"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            gid = out.stdout.strip()
            if gid.isdigit():
                return gid
    except (OSError, subprocess.SubprocessError):
        pass
    return "109"


def next_steps(
    root: Path,
    *,
    profile: str,
    config: dict,
    env_path: Path | None = None,
) -> list[dict[str, str]]:
    """The commands that need a second terminal, in order, with a check for each.

    Each step is ``{"title", "command", "check", "sudo"}``. `sudo` names what may need
    elevating, when anything does; the command itself is never prefixed with sudo, because the
    wizard cannot know whether the user's account is in the docker group.
    """
    compose_profile = _PROFILE_TO_COMPOSE[profile]
    llm: dict = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    model = str(llm.get("model") or "").strip()
    provider = str(llm.get("provider") or "").strip()

    steps: list[dict[str, str]] = []
    steps.append(
        {
            "title": "Start the stack",
            "command": (
                f"cd {root / 'deploy/compose'} && "
                f"docker compose --profile {compose_profile} "
                + ("--profile ollama " if provider == "ollama" else "")
                + "up -d"
            ),
            "check": "docker compose ps",
            "sudo": "if docker needs sudo on this machine",
        }
    )
    if provider == "ollama" and model:
        steps.append(
            {
                "title": f"Pull the {model} model into Ollama (one time)",
                "command": f"docker compose exec ollama ollama pull {model}",
                "check": "docker compose exec ollama ollama list",
                "sudo": "",
            }
        )
    steps.append(
        {
            "title": "Fetch the vision model weights (one time)",
            "command": _VISION_FETCH_CMD,
            "check": "docker compose exec vision-cpu python -m openhup_vision.backends --info",
            "sudo": "",
        }
    )
    steps.append(
        {
            "title": "Open the app",
            "command": "xdg-open http://127.0.0.1:8080",
            "check": "curl -fsS http://127.0.0.1:8080/healthz",
            "sudo": "",
        }
    )
    return steps


def decide_provider(answer: str) -> ProviderChoice:
    """Map the AI step answer to a provider. Enter means local Ollama.

    Raises ValueError for anything else, so the wizard re-asks rather than guessing. There is
    deliberately no "no provider" answer: the AI layer is core.
    """
    kind = _PROVIDER_ANSWERS.get(answer.strip().lower())
    if kind is None:
        raise ValueError(
            "that is not one of the choices. Pick 1-4 (or 'ollama', 'local', 'openai', "
            "'anthropic'), or press Enter for local Ollama."
        )
    return _PROVIDER_DEFAULTS[kind]


def provider_config(choice: ProviderChoice) -> dict:
    """The `llm:` section for a provider choice. Never contains the API key."""
    if choice.provider == "ollama":
        return {"provider": "ollama", "model": choice.model}
    if choice.provider == "openai_compatible":
        return {
            "provider": "openai_compatible",
            "base_url": choice.base_url,
            "model": choice.model,
            "treat_as_local": True,
        }
    if choice.provider == "openai":
        return {
            "provider": "openai_compatible",
            "base_url": choice.base_url,
            "model": choice.model,
            "treat_as_local": False,
            "allow_remote_llm": True,
            "redaction_profile": choice.redaction_profile,
        }
    return {
        "provider": "anthropic",
        "model": choice.model,
        "base_url": choice.base_url,
        "allow_remote_llm": True,
        "redaction_profile": choice.redaction_profile,
    }


def write_env_key(
    env_path: Path | None,
    api_key: str,
    marker: str = "OPENHUP__LLM__API_KEY=",
) -> bool:
    """Write an API key into the environment file, replacing any existing value.

    `marker` selects the variable (default the LLM key; the voice key passes
    ``OPENHUP__VOICE__API_KEY=``). Keys belong in the env file (gitignored, chmod 600), never in
    config.yaml, which stays safe to commit. Returns False when there is nothing to write.
    """
    if not api_key or env_path is None:
        return False
    lines = env_path.read_text().splitlines() if env_path.is_file() else []
    kept = [line for line in lines if not line.startswith(marker)]
    kept.append(f"{marker}{api_key}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(kept) + "\n")
    env_path.chmod(0o600)
    return True


def write_config(
    path: Path,
    *,
    instance_name: str,
    voice: VoiceChoice,
    voice_enabled: bool,
    provider: ProviderChoice | None = None,
    voice_runtime: str = "browser",
    voice_base_url: str | None = None,
) -> dict:
    """Merge the setup answers into config.yaml, keeping whatever was already there.

    Returns the merged config. The file is written even if it existed, so the setup can be
    re-run; comments in a hand-written config are not preserved by this round-trip. The API key
    is deliberately never written here - see `write_env_key`.
    """
    existing: dict = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text()) or {}
        if isinstance(raw, dict):
            existing = raw

    changes: dict = {"instance_name": instance_name}
    personality: dict = {}
    if voice.gamble:
        personality["gamble"] = True
    elif voice.default_personality is not None:
        personality["default_personality"] = voice.default_personality
    if personality:
        changes["personality"] = personality
    changes["voice"] = voice_section(voice_enabled, voice_runtime, voice_base_url)
    if provider is not None:
        changes["llm"] = provider_config(provider)

    merged = _deep_merge(existing, changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(merged, sort_keys=False, default_flow_style=False))
    return merged


def run_setup(
    config_path: Path,
    *,
    presets_path: Path | None = None,
    env_path: Path | None = None,
    ask: Callable[[str], str] = input,
    echo: Callable[[str], None] = print,
    root: Path | None = None,
    confirm: Callable[[str], str] | None = None,
    profile: str = "cpu",
) -> dict:
    """The interactive wizard. Returns the config that was written.

    When `root` (the checkout root) is given, the wizard owns the whole first run: it bootstraps
    config/ from the shipped examples, generates deploy/env/openhup.env with real secrets, asks
    which inference profile the box has, and closes by guiding the commands that need a second
    terminal - waiting for confirmation after each one. `ask`/`echo`/`confirm` are injectable so
    tests can drive it without a terminal.
    """
    presets_path = presets_path or Path(__file__).resolve().parents[2] / (
        "examples/personalities/personalities.yaml"
    )
    descriptions = personality_descriptions(presets_path)
    # The full first-run flow (bootstrap, env generation, profile question, guided handoff) runs
    # when the caller passes the checkout root - which is what the CLI always does. Callers that
    # pass only a config path (tests, embedded use) get the plain questions.
    full_flow = root is not None
    root = root or Path(__file__).resolve().parents[2]

    echo("")
    echo("  OpenHup setup")
    echo("  =============")
    echo("")
    if full_flow:
        echo("  The whole first run, in one command. This will:")
        echo("    - create config/ and deploy/env/openhup.env from the shipped examples")
        echo("      (existing files are never touched - your config wins)")
        echo("    - ask about the voice and the AI provider")
        echo("    - hand you the exact commands for the rest, one at a time")
        echo("")

        created = bootstrap_files(root)
        if created:
            echo("  Bootstrapped:")
            for path in created:
                echo(f"    - {path.relative_to(root)}")
            echo("")

        env_file = generate_env(root, profile=profile)
        if env_file:
            echo(f"  Generated {env_file.relative_to(root)} with random secrets")
            echo("    (edit it if you already have a Postgres password or ntfy topic)")
            echo("")
        # The env file the wizard writes secrets into: the caller's, or the generated one, or an
        # existing one at the conventional path.
        default_env = root / "deploy/env/openhup.env"
        env_path = env_path or (env_file or default_env if default_env.is_file() else None)
    else:
        echo("  A few questions, then you are done. Everything else is configured")
        echo("  later, from the Settings screen or config.yaml.")
        echo("")

    instance_name = ask("Instance name [OpenHup]: ").strip() or "OpenHup"

    if full_flow and (root / "deploy/compose/docker-compose.yml").is_file():
        echo("")
        echo("  This machine's inference hardware")
        echo("  --------------------------------")
        echo("    1. CPU only (default)   - works anywhere, slowest")
        echo("    2. Intel iGPU (openvino) - N100, NUC; the sweet spot for a home box")
        echo("    3. NVIDIA GPU (cuda)")
        echo("")
        while True:
            answer = ask("Inference profile (1-3) [1]: ")
            try:
                profile = decide_profile(answer)
                break
            except ValueError as exc:
                echo(f"  {exc}")

    # --- the voice -------------------------------------------------------------
    echo("")
    echo("  The assistant's voice")
    echo("  ---------------------")
    echo("  Five voices ship with the gamble. Pick one, or gamble and let it")
    echo("  choose for you - in which case it will never be announced. You will")
    echo("  discover it by living with it. (It is always one of the five below;")
    echo("  the docs and config.yaml are the only places the answer is written.)")
    echo("")
    for index, personality_id in enumerate(GAMBLE_POOL, start=1):
        echo(f"    {index}. {descriptions.get(personality_id, personality_id)}")
    echo("")
    echo("    g        gamble - a mystery voice, drawn at first launch")
    echo("    brief    says what needs saying, nothing else - still the model")
    echo("    (Enter)  keep the stock default")
    echo("")

    while True:
        answer = ask("Choose a voice: ")
        try:
            voice = decide_voice(answer)
            break
        except ValueError as exc:
            echo(f"  {exc}")

    voice_answer = ask("Enable voice? [Y/n]: ").strip().lower()
    voice_enabled = voice_answer not in ("n", "no")
    voice_runtime, voice_base_url, voice_api_key = "browser", None, ""
    if voice_enabled:
        voice_runtime, voice_base_url, voice_api_key = _ask_voice_runtime(ask, echo)

    # --- the brain -------------------------------------------------------------
    provider, api_key = _ask_provider(ask, echo)

    config = write_config(
        config_path,
        instance_name=instance_name,
        voice=voice,
        voice_enabled=voice_enabled,
        provider=provider,
        voice_runtime=voice_runtime,
        voice_base_url=voice_base_url,
    )
    llm_key_written = write_env_key(env_path, api_key)
    voice_key_written = write_env_key(env_path, voice_api_key, marker="OPENHUP__VOICE__API_KEY=")

    echo("")
    echo(f"  Wrote {config_path}")
    if llm_key_written:
        echo(f"  LLM API key written to {env_path}")
    elif api_key:
        echo("  LLM API key: set OPENHUP__LLM__API_KEY in your environment before first run.")
    if voice_key_written:
        echo(f"  Voice API key written to {env_path}")
    elif voice_api_key:
        echo("  Voice API key: set OPENHUP__VOICE__API_KEY in your environment before first run.")
    if voice_enabled:
        echo("")
        if voice_runtime == "openai":
            echo(f"  Voice runs via the cloud at {voice_base_url} - logged per call.")
        elif voice_runtime == "gateway":
            echo(f"  Voice runs on your local speech server at {voice_base_url}.")
        else:
            echo("  Voice runs in the browser - on-device, nothing leaves it.")
    if voice.gamble:
        echo("")
        echo("  A mystery voice will be drawn at first launch.")
        echo("  It will not be announced. Happy figuring-out.")
    elif voice.default_personality:
        echo("")
        echo(f"  Voice set to {voice.default_personality}.")
        echo("  From here on it speaks for itself.")
    else:
        echo("")
        echo("  Keeping the stock default voice.")

    if confirm is not None and full_flow:
        echo("")
        echo("  The rest needs a second terminal. Run each command there;")
        echo("  come back and press Enter when it is done. Where sudo is needed,")
        echo("  run the same command with `sudo` in front.")
        for step in next_steps(root, profile=profile, config=config, env_path=env_path):
            echo("")
            echo(f"  {step['title']}")
            echo(f"    {step['command']}")
            if step["sudo"]:
                echo(f"    (sudo may be needed: {step['sudo']})")
            while True:
                reply = confirm("    done? (Enter to continue, 'skip' to skip): ").strip().lower()
                if reply in ("", "done", "y", "yes", "next"):
                    break
                if reply in ("skip", "s"):
                    break
                echo("    say 'skip' to move on, or press Enter when the command finished.")
    echo("")
    echo("  All set. Edit config/cameras.yaml for your cameras, then open")
    echo("  http://127.0.0.1:8080 and follow the first-run checklist.")
    echo("")
    return config


def _ask_voice_runtime(
    ask: Callable[[str], str],
    echo: Callable[[str], None],
) -> tuple[str, str | None, str]:
    """Where speech recognition and synthesis run: browser, cloud, or a local speech server.

    Returns ``(runtime, base_url, api_key)``. Cloud is a first-class choice - a machine without
    the hardware for local speech is not a reason to skip voice - and it carries the same egress
    gate as the LLM step: the user types "yes" to send audio and transcripts to OpenAI, and the
    key goes to the environment file, never into config.yaml.
    """
    echo("")
    echo("  Where voice runs")
    echo("  ----------------")
    echo("  Recognition and speech can run in the browser, on OpenAI's servers, or")
    echo("  on a local speech server (whisper.cpp / Piper). You do not need a")
    echo("  powerful machine for voice - the cloud option is first-class.")
    echo("")
    echo("    1. In the browser (default)  - Web Speech API, no account, on-device")
    echo("    2. Cloud, OpenAI-compatible  - OpenAI by default, or any endpoint (API key)")
    echo("    3. Local speech server       - whisper.cpp / Piper on your network")
    echo("")
    echo("  This is independent of the brain step: local voice with a cloud LLM")
    echo("  (or the reverse) is fine.")

    while True:
        answer = ask("Where should voice run (1-3) [1]: ")
        try:
            runtime = decide_voice_runtime(answer)
            break
        except ValueError as exc:
            echo(f"  {exc}")

    if runtime == "browser":
        return runtime, None, ""
    if runtime == "gateway":
        base_url = ask(f"Base URL [{_VOICE_GATEWAY_DEFAULT_URL}]: ").strip()
        base_url = base_url or _VOICE_GATEWAY_DEFAULT_URL
        return runtime, base_url, ""

    # Cloud: the egress gate is the feature, exactly as in the LLM step. A custom base URL is
    # first-class, so any OpenAI-compatible speech endpoint works, not just OpenAI's.
    base_url = ask(f"Base URL [{_VOICE_CLOUD_DEFAULT_URL}]: ").strip()
    base_url = base_url or _VOICE_CLOUD_DEFAULT_URL
    api_key = ask(
        "API key (blank = set OPENHUP__VOICE__API_KEY in your environment later): "
    ).strip()
    echo("")
    echo(f"  This sends your spoken commands and the assistant's spoken replies to {base_url}.")
    echo("  Every call is logged at /api/v1/system/llm-usage. Type 'yes'")
    echo("  to confirm, or 'no' to pick a different voice setup.")
    while True:
        confirm = ask("Confirm egress [yes/no]: ").strip().lower()
        if confirm == "yes":
            return runtime, base_url, api_key
        if confirm in ("no", "n"):
            return _ask_voice_runtime(ask, echo)
        echo("  Say 'yes' or 'no'.")


def _ask_provider(
    ask: Callable[[str], str],
    echo: Callable[[str], None],
) -> tuple[ProviderChoice, str]:
    """The required AI step: pick where the assistant's brain lives.

    Local is the default and needs nothing else. The two cloud choices demand an explicit
    egress confirmation (typing "yes") and a redaction profile, and the API key goes to the
    environment file, never into config.yaml.
    """
    echo("")
    echo("  The assistant's brain")
    echo("  ---------------------")
    echo("  The AI layer is the core of the assistant - the voice, the memory,")
    echo("  the noticing. You need a provider. Local is the default; a cloud")
    echo("  provider you trust works too, and the egress gate is yours to confirm.")
    echo("")
    echo("    1. Ollama, local (default)   - nothing leaves your network")
    echo("    2. OpenAI-compatible, local  - llama.cpp, vLLM, LM Studio on your box")
    echo("    3. Cloud, OpenAI-compatible  - OpenAI by default, or any endpoint you trust")
    echo("    4. Anthropic, cloud          - api.anthropic.com, or a custom endpoint")
    echo("")
    echo("  The cloud choices accept a custom base URL, so you are not limited to")
    echo("  OpenAI's own endpoints.")

    while True:
        answer = ask("Choose a provider (1-4) [1]: ")
        try:
            choice = decide_provider(answer)
            break
        except ValueError as exc:
            echo(f"  {exc}")

    if choice.provider == "ollama":
        model = ask(f"Model [{choice.model}]: ").strip() or choice.model
        return ProviderChoice("ollama", model), ""

    if choice.provider == "openai_compatible":
        base_url = ask(f"Base URL [{choice.base_url}]: ").strip() or choice.base_url
        model = ask("Model (e.g. llama3.1:8b): ").strip()
        return ProviderChoice("openai_compatible", model, base_url=base_url), ""

    # Cloud: the egress gate is the feature. Nothing about this is skippable.
    model = ask(f"Model [{choice.model}]: ").strip() or choice.model
    base_url = ask(f"Base URL [{choice.base_url}]: ").strip() or choice.base_url
    api_key = ask("API key (blank = set OPENHUP__LLM__API_KEY in your environment later): ").strip()
    echo("")
    echo(f"  This sends prompts - and, at higher redaction profiles, imagery - to {base_url}.")
    echo("  Every call is logged at /api/v1/system/llm-usage.")
    echo("  Type 'yes' to confirm, or 'no' to go back and pick a local provider.")
    while True:
        confirm = ask("Confirm egress [yes/no]: ").strip().lower()
        if confirm == "yes":
            break
        if confirm in ("no", "n"):
            return _ask_provider(ask, echo)
        echo("  Say 'yes' or 'no'.")
    redaction = ask("Redaction profile (1=text_only, 2=redacted_image, 3=full) [1]: ").strip()
    profile = _REDACTION_ANSWERS.get(redaction, "text_only")
    return (
        ProviderChoice(
            choice.provider,
            model,
            base_url=base_url,
            allow_remote_llm=True,
            redaction_profile=profile,
        ),
        api_key,
    )


def voice_section(enabled: bool, runtime: str, base_url: str | None = None) -> dict:
    """The `voice:` block for the setup answers.

    Explicit on every field the wizard manages, so re-running converges: a later 'browser'
    choice clears a previous cloud choice - including its base_url - instead of merging into it.
    """
    section: dict = {
        "enabled": enabled,
        "stt_provider": "browser",
        "tts_provider": "browser",
        "allow_remote_voice": False,
        "base_url": None,
        "treat_as_local": False,
    }
    if runtime == "openai":
        section.update(
            {"stt_provider": "openai", "tts_provider": "openai", "allow_remote_voice": True}
        )
        if base_url:
            section["base_url"] = base_url
    elif runtime == "gateway":
        section.update(
            {
                "stt_provider": "openai_compatible",
                "tts_provider": "openai_compatible",
                "base_url": base_url,
                "treat_as_local": True,
            }
        )
    return section


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge `overlay` into `base`, recursing into dicts. Returns a new mapping."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


__all__ = [
    "ProviderChoice",
    "VoiceChoice",
    "bootstrap_files",
    "decide_profile",
    "decide_provider",
    "decide_voice",
    "decide_voice_runtime",
    "generate_env",
    "next_steps",
    "personality_descriptions",
    "provider_config",
    "run_setup",
    "voice_section",
    "write_config",
    "write_env_key",
]
