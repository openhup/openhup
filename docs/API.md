# API

Base path `/api/v1`. Interactive docs at `/api/docs`, OpenAPI at `/api/openapi.json`.

Auth: session cookie for browsers, `Authorization: Bearer <token>` for services and scripts (tokens in
`security.service_tokens`). LAN-only by default — see [SECURITY_PRIVACY.md](SECURITY_PRIVACY.md).

## Endpoints

### Skills

| Method | Path | Notes |
|---|---|---|
| GET | `/skills` | summaries including a plain-language `explanation` and compile `warnings` |
| POST | `/skills` | compiled before storing; `422` with every finding if it would misbehave |
| GET | `/skills/{id}` | definition, summary, and resolved signal keys |
| PATCH | `/skills/{id}` | partial update, recompiled |
| DELETE | `/skills/{id}` | removes FSM instance state too |
| POST | `/skills/parse` | natural language → draft. Never enabled, never saved. |
| POST | `/skills/{id}/simulate` | **replay against real history.** Use this before enabling anything. |
| POST | `/skills/import` | multi-document YAML; disabled unless `?enable=true` |
| GET | `/skills/{id}/state` | FSM phase per anchor, and when it could next fire |

### Cameras, anchors, vision

| Method | Path | Notes |
|---|---|---|
| GET/POST | `/cameras` | `password_env` is returned; a password never is |
| PATCH/DELETE | `/cameras/{id}` | deleting keeps the anchors (ADR-010) |
| GET/POST | `/anchors` | includes `needs_baseline`, the commonest cause of "it does nothing" |
| PATCH | `/anchors/{id}` | polygon, subregions, sensitivity, weights |
| POST | `/anchors/{id}/baseline` | capture the "clean" reference. Do it while tidy. |
| GET | `/snapshots/{path}` | proxied through the API so auth applies to imagery |
| GET | `/vision/plan` | detector schedules derived from enabled skills. Consumed by the vision service. |
| GET | `/detectors` | the registry the skill builder renders from |
| GET | `/observations` | raw signal feed, for debugging and calibration |

### Tasks and alerts

| Method | Path | Notes |
|---|---|---|
| GET | `/tasks` | `?state=open\|done\|all`. **No total count, by design.** |
| GET | `/tasks/next` | the one thing to do now. Single-task-focus mode calls only this. |
| GET | `/tasks/{id}` | |
| PATCH | `/tasks/{id}` | `{action: complete\|start\|dismiss\|snooze\|reopen\|false_positive}` |
| GET | `/alerts` | |
| POST | `/alerts/{id}/ack` | stops repetition; leaves the alert open until it truly resolves |
| GET | `/episodes` | trigger→resolve cycles: the raw material for every metric |

`false_positive` is the most valuable call in this API. It feeds threshold suggestions and the
`false_positive_rate` metric, which is how you find out which skill needs attention.

### Voice

| Method | Path | Notes |
|---|---|---|
| GET | `/voice/config` | providers, wake word, whether audio leaves the device |
| POST | `/voice/transcribe` | raw audio body → `{text}`. Only used with a remote STT provider. |
| POST | `/voice/synthesize` | `{text}` → audio bytes. Only used with a remote TTS provider. |
| POST | `/voice/command` | `{text}` → intent + spoken reply + any side effect already applied |

`/voice/command` is deterministic keyword routing, not an LLM. Intents: `task_command` (complete,
start, snooze, dismiss, false-positive on the next task), `query` ("what should I do"), `navigate`
(a route for the client to follow), `memory` (teach/recall/forget a household fact, see below),
`skill_dictation` (the existing `/skills/parse` pipeline, never armed), and `unknown`. The client
speaks the `reply` field. The request may carry `speaker` (a member id the device already knows —
declared per-device in Settings, never inferred) so task commands and queries can target the right
person in a shared house. See [VOICE.md](VOICE.md).

### Members (consent-gated identity)

| Method | Path | Notes |
|---|---|---|
| GET | `/members` | everyone who said yes to the consent question, plus whether identity is enabled |
| POST | `/members` | `{name, embedding}` → enroll. The only path that stores a face embedding. |
| POST | `/members/consent` | `{anchor_id, answer, name?}` → record a yes/no. A "no" stops the re-ask. |
| DELETE | `/members/{id}` | forget a member: embedding and presence history go with them |

Identity is consent-gated end to end (ADR-016): the camera computes an embedding only to ask
whether it may remember a person, nothing is stored until they say yes, a "no" writes a date
marker and nothing else, and identity is presence — it names who was *in* a room, never who
*did* anything. The consent question is asked at most once per anchor per day, spoken in the
active personality when voice is on.

### Memory

| Method | Path | Notes |
|---|---|---|
| GET | `/memory` | everything the assistant has been told, newest first |
| POST | `/memory` | `{fact, topic?}` → teach it something |
| DELETE | `/memory/{id}` | forget one fact. A forgotten fact is gone. |
| GET | `/memory/patterns` | learned patterns, freshly recomputed, with evidence |
| DELETE | `/memory/patterns/{id}` | dismiss a pattern: never surfaced or nudged again |

### The personality gamble and wins

The **gamble** (ADR-014) is a setup twist: `openhup setup` shows the five voices — `friendly`,
`shy`, `sassy`, `sarcastic`, `angry` — and the user either picks one or gambles. A gamble means
one is drawn at random on first launch and becomes the effective default *without being announced*;
from then on the voice is never shown anywhere, and the user discovers it by living with it.
`GET /personality/draw` returns the state (the id is kept back from the UI — the settings screen
will re-draw or turn the gamble off, but will not name the voice). Every re-draw is an explicit
POST and increments `reroll_count`. Deleting the draw restores the configured `default_personality`
exactly; `/system/info` reports the effective default for operators — the "deep configuration" that
is the only place a gambled voice is written down. All five pool voices sit at intensity 3 or
below, so a draw is never silently clamped by the default `humor_ceiling`.

**Wins** (ADR-015) are the assistant noticing progress: when a task resolves, the engine computes
how long that anchor stayed clear and celebrates, at most once per milestone, either a whole-day
band (1/3/7/14/30 days) or a 90-day record. `GET /personality/wins` returns the reviewable ledger
with tone-free summaries. The spoken note rides the WebSocket as `system.win_note` (`{text, plain,
days, record}`) and is forward-facing only — "has stayed clear for N days", never anything about
what was left undone.

Memory has two halves. **Facts** are plain, human-readable claims you taught it, in your words.
**Patterns** are derived on-device from the episodes the engine already records — "the kitchen
counter usually needs attention about every 3 days" — only when there is enough evidence, and
always with the numbers behind them in `evidence`. Both live in local Postgres, and neither leaves
your network except as a relevant snippet inside a phrasing prompt, which is gated by
`llm.allow_remote_llm` and logged like every other call. Everything is listable and dismissable — a
memory the user cannot inspect and delete is not a memory. A dismissed pattern is never learned
again. See [SECURITY_PRIVACY.md](SECURITY_PRIVACY.md), ADR-012, and ADR-013.

### Metrics, goals, personalities, notifications, system

| Method | Path | Notes |
|---|---|---|
| GET | `/metrics/catalog` | built-in metrics and their meanings |
| GET | `/metrics/series` | `?metric=&days=&anchor=` |
| GET/POST | `/metrics/goals` | progress is never reported as "failed" |
| DELETE | `/metrics/goals/{id}` | |
| GET | `/metrics/report/weekly` | coaching summary. No "missed" or "overdue" counts anywhere. |
| GET | `/personalities` | includes `effective_intensity` after the humor ceiling is applied |
| PUT | `/personalities/{id}` | shipped presets are read-only; copy under a new id |
| POST | `/personalities/{id}/preview` | sample output, including an alert to show the bypass |
| GET | `/personality/draw` | the gamble state: `drawn` (the mystery voice), `reroll_count`, `pool` |
| POST | `/personality/draw` | draw the mystery voice, or re-draw it. Each re-draw is counted. |
| DELETE | `/personality/draw` | stop the gamble: the configured default speaks again |
| GET | `/personality/wins` | the ledger of wins the assistant has noticed (see below) |
| DELETE | `/personality/wins/{id}` | forget a win. A deleted win is gone; the same milestone can be celebrated again |
| GET | `/notify/channels` | |
| POST | `/notify/channels/{id}/test` | |
| GET | `/notify/held` | notifications waiting for quiet hours to end |
| GET | `/system/info` | versions, LLM posture, UX defaults, config warnings |
| GET | `/system/health` | **reports stale cameras as problems**, not as silence |
| GET | `/system/llm-usage` | what was sent, where, how big, whether an image went too |
| GET | `/healthz`, `/readyz` | liveness and readiness |

### WebSocket

```
ws://host/api/v1/ws/events?topics=task,alert,system
```

Topics: `task`, `alert`, `skill`, `metric`, `goal`, `system`, `observation`, `all`. Filtered
server-side — `observation` is high volume and only the calibration view should ask for it. Frames are
`Envelope` objects, the same shape the internal bus carries.

---

## Worked examples

### Set up one anchor and one skill

```sh
API=http://localhost:8080/api/v1

curl -X POST $API/cameras -H 'content-type: application/json' -d '{
  "id": "kitchen", "name": "Kitchen", "kind": "rtsp",
  "url": "rtsp://192.168.20.11:554/stream1",
  "substream_url": "rtsp://192.168.20.11:554/stream2",
  "username": "openhup", "password_env": "KITCHEN_CAM_PASSWORD"
}'

curl -X POST $API/anchors -H 'content-type: application/json' -d '{
  "id": "kitchen.counter", "camera_id": "kitchen", "label": "Kitchen counter",
  "polygon": [[0.05,0.34],[0.94,0.30],[0.96,0.78],[0.04,0.82]]
}'

# While the counter is actually tidy:
curl -X POST $API/anchors/kitchen.counter/baseline

curl -X POST $API/skills/import --data-binary @examples/skills/kitchen-clutter-buster.yaml \
     -H 'content-type: text/plain'
```

### Dry-run before arming

```sh
curl -s -X POST "$API/skills/kitchen-clutter-buster/simulate?days=7" | jq '{verdict, advice, per_day}'
```

```json
{
  "verdict": "Would have fired 3x (0.43/day); typical episode 1h12m",
  "advice": [],
  "per_day": 0.43
}
```

If it says `Would have fired 47x (6.7/day)` you have just saved yourself a bad week. Raise the
threshold, lengthen `for:`, then simulate again.

### What a rejected skill looks like

```sh
curl -s -X POST $API/skills -H 'content-type: application/json' -d @flapping-skill.json | jq
```

```json
{
  "error": "skill_compile_failed",
  "skill_id": "flapper",
  "findings": [{
    "code": "no_hysteresis",
    "binding": "clutter",
    "error": true,
    "message": "'clutter': trigger (gte 0.6) and resolve (lte 0.7) ranges overlap, so a task would open and close repeatedly. resolve at a value below the trigger, e.g. trigger >= 0.6, resolve <= 0.24"
  }]
}
```

Every finding comes back at once, with a concrete suggested fix.

### Single-task focus

```sh
curl -s $API/tasks/next | jq '{current_text, before_snapshot, progress}'
```

```json
{
  "current_text": "Just clear left third",
  "before_snapshot": "snap://2026/08/17/kitchen.counter/01K3XQ8V.jpg",
  "progress": 0.0
}
```

`current_text` is the current micro-step, not the whole task. In this mode the UI shows nothing else,
and the backlog is never sent to the client.

### Natural language

```sh
curl -s -X POST $API/skills/parse -H 'content-type: application/json' \
  -d '{"text":"alert me if the stove is left on with nobody in the kitchen"}' | jq '{ok, explanation, heuristic}'
```

```json
{
  "ok": true,
  "explanation": "Watching kitchen.stove: when burner eq 'on' for 10m and not (people gte 1) for 5m, raise a high alert. It clears itself when burner eq 'off' for 30s or people gte 1.",
  "heuristic": false
}
```

The draft comes back with `enabled: false` and `needs_confirmation: true` always. Nothing arms itself
because a model said so.

### Feed a sensor in instead of a camera

A lid switch is cheaper and more reliable than a camera for a binary question. Publish to MQTT and the
`sensor` detector turns it into the same kind of observation:

```sh
mosquitto_pub -t 'openhup/sensor/kitchen.trash/lid_open' -m 'true'
```

Skills bind to it exactly as they would a visual signal, and do not care which answered.

---

## Errors

| Status | Meaning |
|---|---|
| 400 | malformed request (including snapshot path traversal) |
| 404 | not found, or a snapshot that has already expired |
| 409 | id already exists, or an attempt to edit a shipped preset |
| 422 | schema validation failed, or a skill would misbehave — `findings` lists every problem |
| 503 | `/readyz` only: the database is unreachable |

`422` from `/skills` is the interesting one. It carries a `findings` array rather than a single message,
because a skill can have several problems and fixing them one round trip at a time is miserable.
