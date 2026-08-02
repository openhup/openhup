# <img src="../frontend/static/logo-mark.svg" alt="" width="26" style="vertical-align: -4px" /> OpenHup documentation

A self-hosted, local-first home assistant that watches spaces with cameras and turns what it sees into
tasks, safety alerts, and habit metrics — in a voice you choose.

## Start here

| If you want to… | Read |
|---|---|
| get it running | [INSTALL.md](INSTALL.md) |
| understand how it works | [ARCHITECTURE.md](ARCHITECTURE.md) |
| know why it was built this way | [adr/README.md](adr/README.md) |
| set up cameras, anchors, and skills | [CONFIGURATION.md](CONFIGURATION.md) |
| know exactly what is stored and what leaves your network | [SECURITY_PRIVACY.md](SECURITY_PRIVACY.md) |
| buy the right hardware | [HARDWARE.md](HARDWARE.md) |
| configure it for ADHD or autism | [UX_NEURODIVERGENT.md](UX_NEURODIVERGENT.md) |
| call the API | [API.md](API.md) |
| use voice (speak & listen) | [VOICE.md](VOICE.md) |
| contribute code | [DEVELOPERS.md](DEVELOPERS.md) |

## The idea in one page

A camera watches a named region — an **anchor** — like your kitchen counter. The vision service reports
plain facts about it: `clutter_level=0.72`, `objects=[cup, plate]`, `burner_state=on`. It makes no
judgements at all.

A **skill** decides what those facts mean:

```yaml
conditions: {signal: clutter, op: gte, value: 0.6, for: 15m}
effect:     {type: task, title_hint: clear the kitchen counter}
resolve:    {conditions: {signal: clutter, op: lte, value: 0.25, for: 2m}}
```

Clutter above 0.6 for fifteen minutes becomes a task, with a photograph attached. Clutter below 0.25
for two minutes closes it — on its own, with an "after" photograph. Nobody ticked anything off.

The gap between 0.6 and 0.25 is the whole trick. A counter hovering at 0.5 satisfies neither condition,
so nothing happens, which is correct: a counter at 0.5 is a counter in use. OpenHup refuses to save a
skill whose thresholds overlap.

Three kinds of thing a skill can produce:

- **task** — something to do, which the camera can see is done
- **alert** — something unsafe, notified immediately and phrased plainly
- **metric** — something to measure and never mention, which is how "help me watch less TV" works

## Four worked examples

Fully annotated, and they run as written:

- [`kitchen-clutter-buster.yaml`](../examples/skills/kitchen-clutter-buster.yaml) — the canonical skill
- [`stove-burner-safety.yaml`](../examples/skills/stove-burner-safety.yaml) — safety, presence gating,
  and why personality is bypassed
- [`adhd-micro-task-shelf.yaml`](../examples/skills/adhd-micro-task-shelf.yaml) — one small step at a
  time, verified by the camera
- [`tv-time-tracking.yaml`](../examples/skills/tv-time-tracking.yaml) — measurement with no nagging

Plus nine more in [`more-examples.yaml`](../examples/skills/more-examples.yaml): trash, bread, pet
bowls, walkways, doors, cooking sessions, dish racks, fall detection.

## Three promises, and where they are enforced

**Nothing leaves your network unless you say so.** The default LLM is local; a remote one is refused at
startup unless you set a flag and choose a redaction profile. Every outbound call is logged with its
byte count. The systemd units deny egress at the init-system level.
→ [SECURITY_PRIVACY.md](SECURITY_PRIVACY.md)

**It will not nag you.** Trigger and resolve thresholds are separate so tasks cannot flap. Cooldowns,
daily caps, and quiet hours are schema fields. Task counts are hidden by default and the API does not
compute them. `nag_index` — notifications per completed task — is tracked as an anti-metric.
→ [UX_NEURODIVERGENT.md](UX_NEURODIVERGENT.md)

**Safety outranks comedy.** At `urgency >= high` the personality layer is bypassed in code, quiet hours
are ignored, and rate limits do not apply. A burner alert reads the same on every install.
→ [adr/README.md](adr/README.md) ADR-009

## What it is not

Not an NVR — use Frigate, and OpenHup can run on top of its detections. Not a home automation hub — it
publishes to MQTT and Home Assistant. Not face recognition without consent — identity exists only for
people who said yes to being remembered, and it is presence, never attribution (ADR-016). Not a cloud
service: no account, no telemetry, nowhere for data to go.
