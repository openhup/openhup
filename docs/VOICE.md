# Voice

OpenHup can talk and listen. This is a **voice interface**, not audio surveillance: the microphone
is gated behind a wake word that is matched locally, nothing is recorded, and by default nothing
leaves the device at all.

The design is recorded in [ADR-011](adr/README.md).

## The short version

- **Speech runs in the browser by default.** Recognition and synthesis both use the browser's Web
  Speech API, so no audio and no transcript reach OpenHup, let alone the internet.
- **The wake word is matched locally.** Continuous recognition listens for `hey openhup` (the
  default) in JavaScript, and only *after* it is heard is a single command processed.
- **Commands are deterministic.** There is no LLM deciding what you said. A keyword router turns
  the transcript into one of a few actions, the same way skill parsing degrades to a heuristic.
- **Remote speech is opt-in.** If you put a key in the environment and choose a cloud STT/TTS
  provider, OpenHup refuses to start unless `voice.allow_remote_voice: true`, and every outbound
  call is recorded with its byte count at `/api/v1/system/llm-usage`.

## Enabling it

Voice is on by default. On the device you want it to work on, open **Settings → Voice** and flip
**Enable voice on this device**. Your browser then asks for microphone permission the first time you
press the talk button or arm the wake word.

The setup wizard (`openhup setup`) asks where speech should run and writes the answer here: the
browser (default), any OpenAI-compatible cloud (OpenAI by default, or a custom base URL), or a
local whisper.cpp/Piper gateway. Cloud is a first-class choice — a machine without the hardware
for local speech is not a reason to skip voice — and it carries the same typed-"yes" egress
confirmation as a remote LLM, with the key written to the environment file.

Browser speech needs a browser that ships the Web Speech API (Chrome, Edge, Safari). Firefox does
not, so voice falls back to the server provider if one is configured, or shows as unavailable.

```yaml
# config/config.yaml
voice:
  enabled: true
  stt_provider: browser     # browser | openai | openai_compatible
  tts_provider: browser     # browser | openai | openai_compatible
  wake_word: hey openhup
  language: en
```

## Using it

Two ways to speak to it:

- **Wake word.** Say the wake word, then your command. There is a short chime to say "go ahead".
- **The talk button.** The microphone button in the header listens for one command, no wake word
  needed. Press it once to grant permission; it then stays armed for the wake word if you have that
  setting on.

It speaks back: replies to your commands, safety alerts as they arrive, new tasks as a gentle
nudge, learned-pattern predictions ("the trash usually fills around now"), and **wins** — when a
place stays clear for whole days, it says so in its own voice ("the counter has stayed clear 3
days"). Each of those is a toggle in Settings, and wins are on by default: progress noticed is the
caring half of the voice. See ADR-015.

## What you can say

| Example | What it does |
|---|---|
| "what should I do" / "read my task" | speaks the one thing to do now |
| "done" / "finished" | completes the current task |
| "start" | marks the current task in progress |
| "snooze for an hour" / "later" | snoozes the current task (default one hour) |
| "not a real task" | marks it a false positive — this is the feedback that tunes thresholds |
| "dismiss" | dismisses the current task |
| "show tasks" / "open cameras" / "go to habits" | navigates |
| "remind me when the trash is full" | drafts a skill from the sentence (never armed automatically) |
| "remember that bin day is Tuesday" | teaches a household fact (stored locally) |
| "what do you remember" / "what do you know about the trash" | recalls what it has been told |
| "forget that bin day is Tuesday" / "forget everything" | forgets one fact, or the whole store |
| "what have you noticed" / "what have you noticed about the trash" | recalls what it has *learned* from your data |
| "it's Sam" / "Sam's done" / "that's Sam's task" | declares who is speaking (see below) |
| "yes, remember me" / "no thanks" | answers the identity consent question |

A draft from speech behaves exactly like one typed into the skill builder: it is created disabled,
shown for review, and only runs after you enable it. Speech never arms a skill by itself.

### Who is speaking

In a household with more than one person, a bare "done" or "what should I do" gets "I don't know
who's asking — say it's Sam" instead of acting on the wrong person's task. Identity is **declared,
never inferred** (ADR-016): you tell the assistant who you are — "it's Sam" — or mark which member
this device belongs to in **Settings → Who lives here**, and it targets the right person's tasks and
nudges from then on. The camera never guesses who is speaking, and the assistant never names anyone
who has not consented to be remembered.

## Teaching it things

The assistant remembers only what you tell it, in your words — "I call the spare room the junk
room", "bin day is Tuesday". Facts live in local Postgres and are used as context when a task line
is phrased, so a task about the spare room can say "the junk room". Retrieval is keyword matching,
not a model, and the memory store never leaves your network by itself: the only way a fact reaches an
LLM provider is as a snippet inside a phrasing prompt, which is gated by `allow_remote_llm` and
logged like every other call.

Everything it knows is reviewable and deletable in **Settings → What the assistant remembers**.
Say "forget everything" to wipe the store by voice. See ADR-012.

## What it notices on its own

The assistant also *learns* — not from listening, but from the episodes the engine already records.After a couple of weeks of
history it can notice that "the trash usually fills about every 3 days",
and it will say so when asked ("what have you noticed") or speak
up when a pattern says something is about due. Every learned claim is forward-facing ("usually about
every 3 days", never "you've left it for 6 days"), shown in Settings with the numbers behind it, and
dismissable — a dismissed pattern is never learned again. A pattern nudge obeys the subject skill's
own `limits` — its `quiet_hours`, `cooldown`, and `max_per_day` apply exactly as they do to the
skill's triggers — and never fires when a task already covers that spot. See ADR-013.

And it notices when a place **stays clean**: once a task is resolved, the gap since the previous
clean stretch is measured, and a whole-day milestone (1/3/7/14/30 days) or a new 90-day record is
celebrated through the current personality — at most once per milestone, forward-facing only ("has
stayed clear for 3 days", never "you left it"), and never inside the skill's quiet hours. The
ledger is reviewable in **Settings → Wins the assistant has noticed**. See ADR-015.

## Remote providers

Two shapes, mirroring the LLM's egress model:

| Provider | Meaning | Gate |
|---|---|---|
| `openai` | the public OpenAI Whisper (STT) and TTS endpoints | requires `allow_remote_voice: true` |
| `openai_compatible` | any OpenAI-shaped gateway, e.g. a local whisper.cpp server or a Piper wrapper | gated unless `treat_as_local: true` |

```yaml
voice:
  stt_provider: openai_compatible
  tts_provider: openai_compatible
  base_url: http://whisper:8080/v1   # your gateway
  treat_as_local: true               # it runs on your hardware
```

or, for the public API:

```bash
# deploy/env/openhup.env
OPENHUP__VOICE__STT_PROVIDER=openai
OPENHUP__VOICE__TTS_PROVIDER=openai
OPENHUP__VOICE__API_KEY=sk-...
OPENHUP__VOICE__ALLOW_REMOTE_VOICE=true
```

When a remote provider is used, the browser records a short clip (MediaRecorder) and uploads it for
transcription, and TTS audio is fetched and played back. The clip is held in memory only and never
written to disk.

## What is deliberately not done

- **No continuous recording.** Audio exists only for the duration of one command; it is never
  stored, retained, or replayed.
- **No server-side wake word.** The wake word is matched in the client before anything is uploaded.
- **No voice identification.** There is no per-person voice model and no "who said that".
- **No ambient room listening.** Cameras still have no microphones; this feature lives in the
  browser tab, not on the capture pipeline.

## Known limits

- Chrome's built-in recognition uses Google's speech service even in "browser" mode, so "local by
  default" means *local to your OpenHup deployment*, not necessarily on-device. Browsers differ;
  check your browser's speech settings.
- The wake word stops working when the tab is closed. It is a dashboard feature, not a smart
  speaker.
- The honest path to fully on-device voice is a WASM keyword spotter plus local Whisper/Piper. The
  provider seam (backend `openhup/voice`, frontend `$lib/voice`) is written so that slots in without
  touching the command router.
