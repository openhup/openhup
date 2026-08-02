# Developing OpenHup

## Get it running in five minutes

```sh
git clone https://github.com/openhup/openhup && cd openhup

# The pure engine tests need no Postgres, no Redis, no cameras, and no models.
cd packages/openhup-schemas && uv sync && uv run pytest -q && cd ../..
cd backend                   && uv sync && uv run pytest -q && cd ..
cd vision-service            && uv sync && uv run pytest -q && cd ..
```

All three suites pass on a laptop with nothing installed but Python. That is a design property, not a
testing trick: the bus degrades to in-process queues, the LLM degrades to templates, and the vision
service's maths depends only on numpy. It means you can work on the interesting parts before touching
any hardware.

```sh
make test          # all three suites
make lint          # ruff check + format --check
make typecheck     # mypy strict on the engine core
```

## Where the interesting code is

Read in this order:

1. **`backend/tests/test_operators.py`** — the best available documentation of intended semantics.
   Every test there is a bug that would otherwise show up as a 3am task or a burner alert that never
   fires.
2. **`backend/openhup/skills/operators.py`** — the temporal operators. Three decisions carry the
   design: `for:` measures a contiguous run rather than a window average; a data gap breaks a run
   (`max_gap`), so a camera outage cannot masquerade as a sustained condition; and `count_over` counts
   rising edges rather than samples.
3. **`backend/openhup/skills/evaluate.py`** — pure condition evaluation returning a `Verdict` with
   per-node reasons and data health, never a bare bool.
4. **`backend/openhup/skills/fsm.py`** — the per-instance state machine. Absence of data never
   resolves a task.
5. **`backend/openhup/skills/compile.py`** — the lints. The anti-flap hysteresis check is the most
   valuable validation in the system.
6. **`vision-service/openhup_vision/fusion.py`** — why clutter scoring needs three signals.

## Architecture in one paragraph

The vision service reports facts (`clutter_level=0.72` on `kitchen.counter`) and holds no policy. The
backend's skill engine holds all the policy: thresholds, timing, hysteresis, whether something is a
task or an alert, how it is phrased. They communicate through Redis Streams using types defined once
in `packages/openhup-schemas`. Full detail in [ARCHITECTURE.md](ARCHITECTURE.md); the reasoning behind
each choice is in [adr/README.md](adr/README.md).

## Design rules to preserve

These are not style preferences. Breaking them breaks something specific.

**`now` is always a parameter.** `evaluate()`, `advance()`, `simulate()`, and `Engine._evaluate()` all
take the current time as an argument and read no clock. This is what makes grace periods, daily caps,
cooldowns, and clock edges testable without sleeping, and what makes replay identical to live
operation.

**The engine core is pure.** `skills/` does no I/O. Effects are described as dataclasses (`CreateTask`,
`ResolveTask`) and executed elsewhere. If you find yourself needing a database inside `skills/`, the
information should be passed in via `EngineContext` instead.

**Absence of data never resolves anything.** A `Verdict` carries `missing` and `stale`, and the FSM
refuses to act on an unevaluable resolve verdict. A dead camera must not tidy someone's house.

**Effects are idempotent per episode.** The bus is at-least-once. Safety comes from the unique index
on `tasks.episode_id`, not from hoping a message is never redelivered.

**No feature may depend on the LLM.** Every surface has a deterministic template fallback, so a
slow or unavailable model degrades gracefully instead of wedging the house. The AI layer is core:
no shipped personality switches it off - the quietest voice (`brief`) still calls the model, and
the template fallback is resilience, not a choice. The `echo` provider makes every test deterministic.

**Safety outranks tone.** `urgency >= high` bypasses the personality layer in code
(`openhup/llm/render.py`), not by asking a model nicely.

**No counts of unfinished work.** Anywhere. See [UX_NEURODIVERGENT.md](UX_NEURODIVERGENT.md).

## Adding things

### A detector

1. Declare it in `packages/openhup-schemas/openhup_schemas/detectors.py`: signals, kinds, params,
   cost. The UI's skill builder renders from this, so it becomes selectable with no frontend change.
2. Implement it in `vision-service/openhup_vision/detectors.py` — a class with `name`, `models`, and
   `detect(frame, context) -> DetectorResult`.
3. Register it in `build_registry()`.
4. Add a test. `test_detectors.py::test_the_gap_between_schema_and_implementation_is_explicit` will
   fail until the declaration and implementation agree, which is the point.
5. If it needs weights, add them to `models/registry.yaml` **with the licence recorded**.

Emit facts, not judgements. A detector that returns `is_messy: true` has taken a decision that belongs
to the skill engine.

### A notification channel

Subclass `Channel` in `backend/openhup/notify/channels.py`, set `required_config` so a missing setting
fails at startup rather than at 3am, add it to `CHANNEL_TYPES`. Test with a `Recorder`-style fake
rather than a live service.

### An LLM provider

Implement the `LLMProvider` protocol in `backend/openhup/llm/base.py` and add a branch to
`build_provider`. Set `caps.local` honestly — the egress policy depends on it.

### A skill operator

`operators.py` for the implementation, a branch in `evaluate._eval_predicate`, a field on
`SignalPredicate`, and tests covering the empty-history, single-sample, gap, and clock-edge cases.
Think hard about what your operator does when there is no data: that answer is the whole safety story.

## Testing conventions

| Layer | Approach |
|---|---|
| Operators, evaluate, FSM | pure functions over synthetic histories at a fixed `T0` |
| Compile | assert on lint *codes*, not message text |
| Detectors | fixture frames, loose numeric bounds, no flaky exact floats |
| API | `httpx.ASGITransport` against SQLite; no Postgres needed |
| End-to-end | drive the real `Engine` with synthetic observations (`test_api.py`) |
| LLM | the `echo` provider; the safety filter against a corpus of deliberately bad output |
| Examples | every shipped example is loaded and compiled (`test_examples.py`) |

Write the test that describes the *behaviour you want to guarantee*, and say why in the docstring. Half
the tests in this repo document a decision as much as they check a result.

## Repo layout

```
packages/openhup-schemas/   shared wire types. Change here first; both services import it.
backend/openhup/
  skills/     window → operators → evaluate → compile → fsm → parse → simulate   (pure)
  tasks/      FSM actions → rows, wording, snapshots, notifications
  llm/        providers, prompts, safety filter, personality renderer
  voice/      STT/TTS provider + deterministic command router (see VOICE.md, ADR-011)
  memory/     taught facts + learned patterns, keyword retrieval (see ADR-012/013)
  personality/  the gamble: draw/reveal/reroll the mystery voice (see ADR-014)
  wins.py     pure win detection from episodes - "stayed clear N days" (see ADR-015)
  identity.py consent-gated member matching + presence windows (see ADR-016)
  notify/     channels + dispatch policy
  bus/        Redis Streams with an in-process fallback
  db/         SQLAlchemy models, session, migrations
  api/        FastAPI routers, WebSocket hub, shared state
  engine.py   the worker
vision-service/openhup_vision/
  roi.py sampler.py fusion.py     pure numpy, fully tested
  detectors.py backends.py sources.py emitter.py main.py
frontend/                    SvelteKit PWA
deploy/                      compose, systemd, Caddy, env
docs/  examples/  models/  scripts/
```

## Conventions

- Python 3.12+, line length 100, ruff for lint and format, mypy strict on `openhup/skills`.
- Comments explain *why*, never *what*. If a comment restates the code, delete it.
- Type hints everywhere. `from __future__ import annotations` at the top of every module.
- Pydantic models use `extra="forbid"`: a typo in a config file must fail loudly on save.
- Prefer a clear function over a clever one. This is a home automation project read by hobbyists at
  midnight.

## Things that are deliberately absent

Do not add these; they will be declined:

- Face recognition without consent. Identity (ADR-016) is consent-gated: an embedding is only
  stored when a person answers yes, and identity annotates presence — it never triggers skills,
  never attributes actions, and never tracks behaviour per person.
- Screen-content classification.
- Ambient audio capture or voice identification. The voice *interface* is supported — see
  [VOICE.md](VOICE.md) and ADR-011 — but room listening and voice-ID are not.
- Telemetry or crash reporting, including opt-in.
- Cloud sync or a hosted control plane.
- Task counts, streak-broken notifications, or "overdue" states.
- Gamification with a leaderboard between household members.

Each is a design position rather than an oversight; the reasoning is in
[SECURITY_PRIVACY.md](SECURITY_PRIVACY.md) and [UX_NEURODIVERGENT.md](UX_NEURODIVERGENT.md).

## Before opening a pull request

```sh
make lint typecheck test
```

Then say in the description what behaviour changed and why. If you changed a threshold, a default, or
anything in `skills/`, explain what it does when data is missing — that is the question a reviewer will
ask first.
