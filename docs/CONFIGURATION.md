# Configuring OpenHup

Four files, in increasing order of how often you will touch them:

| File | What lives there |
|---|---|
| `deploy/env/openhup.env` | secrets only: database password, camera passwords, API keys |
| `config/config.yaml` | deployment settings: database, bus, LLM, notifications, UX defaults |
| `config/vision.yaml` | vision host settings: bus, execution provider, snapshots, sampling |
| `config/cameras.yaml` | cameras and anchors |
| skills | authored in the UI, or as YAML files you import |

The split is deliberate: `config.yaml` and `cameras.yaml` contain no secrets, so they stay safe to
commit to your own dotfiles repo and safe to paste into a bug report. Camera passwords are
*referenced* by environment-variable name, never inlined.

Environment variables override YAML: `OPENHUP__SECURITY__BIND_HOST` beats `security.bind_host`.

---

## Cameras

```yaml
cameras:
  - id: kitchen
    name: Kitchen
    kind: rtsp
    url: rtsp://192.168.20.11:554/cam/realmonitor?channel=1&subtype=0        # main
    substream_url: rtsp://192.168.20.11:554/cam/realmonitor?channel=1&subtype=1  # detect
    username: openhup
    password_env: KITCHEN_CAM_PASSWORD
    transport: tcp
    max_fps: 5
    hwaccel: vaapi
    always_redact: [faces]
```

**Use the substream.** Every commodity IP camera serves a second, low-resolution stream. Decoding 4K
at 5 fps to find a coffee mug is the single most common way people melt a home server; 640×360 is
plenty for clutter and object detection. Point `substream_url` at it and OpenHup uses it for
detection automatically, keeping `url` for the snapshots a human will look at.

**`transport: tcp`.** UDP over wifi produces torn frames that look like motion and waste inference.

**`always_redact`** is a camera-level privacy floor: everything from this camera is redacted before it
touches disk, regardless of what any individual skill asks for. Sensible for a room people relax in.

Source kinds: `rtsp`, `usb` (via a camera-agent on the host that owns the device), `snapshot_url`
(ESP32-CAM and similar — fine for binary states, useless for clutter scoring), `agent_push` (for hosts
behind NAT), `frigate` (consume an existing Frigate install's detections).

---

## Anchors

An **anchor** is a named region of interest, and it is the thing skills actually watch. Anchors have
their own identity and history, so replacing a camera does not orphan your streaks (ADR-010).

```yaml
anchors:
  - id: kitchen.counter
    camera_id: kitchen
    label: Kitchen counter
    polygon: [[0.05, 0.34], [0.94, 0.30], [0.96, 0.78], [0.04, 0.82]]
    subregions:
      - {id: left,   label: Left third,   order: 0, polygon: [...]}
      - {id: middle, label: Middle third, order: 1, polygon: [...]}
      - {id: right,  label: Right third,  order: 2, polygon: [...]}
    clutter_weights: {baseline_diff: 0.35, object_density: 0.45, semantic: 0.20}
    sensitivity: 0.5
```

Coordinates are normalised 0–1, so a polygon survives a resolution change. Draw them in the UI rather
than by hand.

**Draw tight regions.** A counter ROI that clips the doorway will light up every time somebody walks
past, and you will conclude clutter detection does not work. A tight polygon means less compute, fewer
false positives, and less imagery retained.

**Subregions enable spatial micro-tasking.** With three of them, "tidy the shelf" becomes "just clear
the left third" — a real, finishable task whose completion the camera can verify on its own, with no
LLM involved. This is the single highest-value thing you can configure for the ADHD use case.

**Capture a baseline while the space is genuinely tidy:**

```sh
curl -X POST http://localhost:8080/api/v1/anchors/kitchen.counter/baseline
```

Every clutter score is measured against that image. Captured mid-mess, the anchor becomes permanently
and confusingly relaxed. `GET /api/v1/system/health` tells you which anchors are missing one.

---

## Skills

A skill answers five questions: what to **watch**, which **signals** to read, the **conditions** that
fire it, the **effect**, and how it **resolves**.

```yaml
id: kitchen-clutter-buster
watch:  [{anchor: kitchen.counter}]
signals:
  - {id: clutter, detector: clutter_score, signal: clutter_level, params: {reference: baseline}}
conditions:
  all:
    - {signal: clutter, op: gte, value: 0.6, for: 15m}
    - {time_window: {between: ["07:00", "22:00"], tz: local}}
effect:
  type: task
  mode: single_task_focus
  title_hint: clear the kitchen counter
  micro_steps: auto:3
  urgency: low
resolve:
  conditions: {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m
limits: {cooldown: 45m, max_per_day: 4}
snapshot: {attach: true, retention: 7d, redact: [faces]}
```

### Read trigger and resolve together

This is the one concept worth understanding properly. Trigger at `>= 0.60`, resolve at `<= 0.25`. The
gap between them is why this produces one task instead of forty: a counter hovering at 0.5 satisfies
neither condition, so nothing happens — which is correct, because a counter at 0.5 is a counter in
use.

OpenHup **refuses to save** a skill whose trigger and resolve ranges overlap, and warns when they
merely touch. That check is the most valuable validation in the system.

### Operators

| Operator | Means |
|---|---|
| `gte / lte / gt / lt / eq / neq` | compare a number, enum, or boolean |
| `contains / not_contains` | membership in a set signal (`objects`) |
| `changed_to` | a transition — a door *already* open when the skill was enabled does not fire |
| `for: 15m` | held continuously for this long |
| `within: 1h` | happened at least once in the trailing window |
| `absent_for: 5m` | did *not* happen in the window — "nobody has been in the kitchen for 5 minutes" |
| `count_over: {window: 1d, n: 3}` | three separate occurrences, counting edges not samples |
| `max_gap: 90s` | longest data gap tolerated inside a `for:` run |

`max_gap` matters on safety skills. Without it, a camera that dropped out for twenty minutes can
satisfy "burner on for 10m" from two samples either side of the outage — an outage that looks exactly
like a hazard. False alarms destroy trust in precisely the skill that most needs it.

### Effects

- **task** — something to do, which the camera can see is done. Gets micro-steps, cooldowns, snoozing.
- **alert** — something unsafe. Notifies immediately; at `urgency: high` or above the personality
  layer is bypassed entirely and the wording is factual.
- **metric** — measure it, never mention it. No task, no notification. This is how "help me watch
  less TV" works without the system having opinions about your evening.

### Detectors

`GET /api/v1/detectors` is the authoritative list. Currently:

| Detector | Answers | Cost |
|---|---|---|
| `clutter_score` | how messy is this surface, 0–1 | medium |
| `object_inventory` | what objects are here, how many, how much area | medium |
| `zero_shot_state` | which of several states is this in (burner on/off) | medium |
| `screen_on` | is a screen displaying something | trivial |
| `walkway_clear` | is the floor path unobstructed | medium |
| `presence_absence` | is a specific named thing here (opt-in, slow) | high |
| `fill_level` | how full is a container, 0–1 | medium |
| `door_state` | is a door open, closed, or ajar | medium |
| `pose_fall` | is a person down and not moving (opt-in, best-effort) | high |
| `sensor` | values pushed in from MQTT or Home Assistant | trivial |

`zero_shot_state` is the flexible one. It takes text probes and needs no training:

```yaml
- id: burner
  detector: zero_shot_state
  signal: burner_state          # you choose this name
  params:
    probes:
      on:  a lit stove burner with a visible flame or a glowing red element
      off: an unlit stove burner with no flame and no glow
    min_margin: 0.10            # below this gap, report `unknown` rather than guess
```

Describe states visually and concretely. "A lit gas burner with a blue flame" beats "the stove is on".
Always leave `unknown` unmatched by both branches so ambiguity produces silence.

### Natural language

`POST /api/v1/skills/parse` turns a sentence into a draft. The draft is **never enabled** — you see
the compiled meaning, can simulate it, and then decide. With no LLM configured, a keyword fallback
produces a template-based draft and says plainly that it guessed.

---

## Tuning it so it does not nag

This deserves its own section because it is the difference between a tool you keep and a tool you
uninstall.

| Setting | Do this |
|---|---|
| `limits.cooldown` | 45m for surfaces, hours for spaces you rarely touch. Under 5m feels like nagging. |
| `limits.max_per_day` | Always set it. Cheapest protection against a miscalibrated threshold. |
| `limits.quiet_hours` | Non-negotiable for chore skills. Nobody wants a to-do list at 2am. |
| `for:` | Longer than you think. 15m for a counter, 1h for a shelf. |
| `mode: single_task_focus` | The default. One task at a time, house-wide, per skill. |
| `resolve.grace` | 5m. Leave the completed task visible; it is the only moment of reward the loop has. |
| `effect.expires_after` | Let unaddressed tasks disappear rather than age into monuments. |

Then watch **`nag_index`** in the weekly report: notifications sent per completed task. It is an
anti-metric. If it climbs, the thresholds are wrong and OpenHup is becoming the thing it was built to
avoid.

And use the simulator. Seriously: `POST /api/v1/skills/{id}/simulate` replays your actual last week
and tells you "would have fired 14 times". It costs nothing and it is the only honest way to know.

### Calibrating clutter sensitivity

If a single mug sets it off, or a genuine disaster is ignored, adjust in this order:

1. **`anchor.sensitivity`** (0–1, 0.5 neutral). This is a property of the *place* and the right knob
   for "it keeps nagging me about one mug". Skills may be shared; anchors are yours.
2. **`clutter_weights`.** A small close countertop benefits from more `object_density`; a wide room
   shot from more `baseline_diff`. The three components ride along on every observation, so the UI can
   show you which one is driving the score.
3. **Recapture the baseline.** Often the real problem.
4. **The threshold in the skill.** Last resort, because it affects every anchor that skill watches.

---

## LLM configuration

```yaml
llm:
  provider: ollama              # ollama | openai_compatible | anthropic | echo
  base_url: http://127.0.0.1:11434
  model: qwen2.5:7b-instruct
  allow_remote_llm: false
  redaction_profile: text_only
```

The AI layer is core, not a bolt-on: the first-run wizard (`openhup setup`) always asks for a
provider, and the server treats a missing one as a degraded deployment (flagged by
`/system/health`, with a startup warning saying every surface is running on templates). Every
surface still has a deterministic fallback so a slow or unavailable model degrades gracefully
instead of wedging the house — that is resilience, not optionality.

Local model recommendations: `qwen2.5:7b-instruct` or `llama3.1:8b`. These tasks need
instruction-following, not brilliance. ~8 GB VRAM, or 16 GB RAM for CPU inference.

A remote provider is **refused at startup** unless `allow_remote_llm: true`. Every outbound call is
logged with destination and byte count at `/api/v1/system/llm-usage`. See
[SECURITY_PRIVACY.md](SECURITY_PRIVACY.md).

---

## Personalities

Ten ship: `kind_coach` (default), `deadpan_butler`, `chaos_goblin`, `drill_sergeant_lite`,
`brief` (the quiet voice: says what needs saying, nothing else - still the model, because no
personality switches the AI layer off), and the gamble pool (ADR-014): `friendly`, `shy`,
`sassy`, `sarcastic`, `angry`. Copy `examples/personalities/personalities.yaml` and edit.

```yaml
- id: chaos_goblin
  intensity: 4                  # 1 gentle .. 5 unhinged
  tone: [gleeful, absurd, conspiratorial]
  flavor_words: [forbidden, artifact, lair, hoard]
  avoid_words: [lazy, disgusting, again]
  boundaries:
    never: [shame_language, backlog_counts, coercion, body_or_appearance_comments]
    max_words: 28
    emoji: allowed
  templates:
    task: "The {anchor} has begun forming a civilisation. Evict it. Two minutes, tops."
```

Things enforced in code, not by asking a model politely:

- `urgency >= high` **bypasses personality entirely**. A burner alert reads the same on every install.
- Output tripping a `boundaries.never` rule falls back to the template. It is not retried.
- `backlog_counts` is always filtered, even in roast mode. "You've left this for six days" is the
  fastest way to make someone stop opening an app about their house.
- Intensity 4+ requires `personality.roast_consent: true`, and is capped by
  `personality.humor_ceiling` regardless of what any individual personality asks for.

`POST /api/v1/personalities/{id}/preview` shows sample output, including the alert sample, so you can
see the bypass for yourself.

Templates matter more than they look. When the model is unavailable they are the entire voice of
the product, so write them as complete, usable sentences.

---

## Notifications

```yaml
notify:
  quiet_hours: {between: ["22:00", "07:00"], tz: local}
  max_per_hour: 12
  channels:
    ntfy:
      type: ntfy
      url: https://ntfy.sh
      topic: ${NTFY_TOPIC}
      min_urgency: low
    sms_webhook:
      type: webhook
      url: https://your-gateway.example/send
      min_urgency: high
```

Channels: `ntfy` (recommended — self-hostable, no account, carries images), `webhook`, `discord`,
`matrix`, `mqtt`, `smtp`, `log`.

`min_urgency` lets you route chores and safety differently, which is the setup most people end up
wanting: clutter to a quiet channel, burners to a loud one.

Quiet hours **hold** rather than drop: the notification appears in the UI immediately and is delivered
when the window ends. High-urgency alerts ignore quiet hours, rate limits, and dedupe windows
entirely.

---

## Configuring for different needs

**Executive-function difficulty / ADHD.** `mode: single_task_focus`, `micro_steps: auto:3` with
subregions defined, `ux.hide_task_counts: true`, generous `cooldown`, `max_per_day: 1` on the hard
spaces, `expires_after: 3d`, and `verify_on_manual_complete: false` so the system never argues with
you. See [UX_NEURODIVERGENT.md](UX_NEURODIVERGENT.md).

**Autistic users who want predictability.** `personality: brief`, `micro_steps: none`,
`mode: backlog` so the full list is visible and nothing is hidden, no `expires_after` so tasks do not
silently disappear, and fixed `time_window`s so the system's behaviour is the same every day.

**Sensory sensitivity.** `emoji: none`, longer `quiet_hours`, `max_per_hour: 3`, and route everything
to one quiet channel.

**Shared households.** Camera-level `always_redact: [faces]`, `hide_task_counts: true` so nobody sees
a scoreboard of someone else's undone work, `third_party_remarks` in every personality's `never` list
(it is there by default), and a low `humor_ceiling` — the roast setting should be the choice of the
person being roasted.

**Care and safety monitoring.** `personality: brief` throughout, `urgency: critical`, no quiet hours,
`repeat_every` set, at least two notification channels, and please read the honest limitations on
`pose_fall` in `examples/skills/more-examples.yaml` before relying on any of it.
