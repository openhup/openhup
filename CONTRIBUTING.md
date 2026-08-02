# Contributing to OpenHup

Contributions are welcome. This document is short and specific: what to expect, what the constraints
are, and where the edges that will poke you live.

## Getting set up

```bash
git clone https://github.com/openhup/openhup && cd openhup
make install
make test
```

Every suite passes with nothing installed BUT Python. No Postgres, no Redis, no cameras, no model
weights, zero, zilch, nada. That is a deliberate design property (the bus degrades to in-process queues, the LLM degrades to templates, the vision maths depends only on numpy), and it means you can work on the interesting parts immediately without dealing with Satan.

```sh
make check      # lint + typecheck + tests + docs + examples: what CI runs
```

## Where to start

Good first contributions, roughly in order of usefulness (remove as needed):

- **A detector.** `fill_level`, `door_state`, and `pose_fall` are declared in the schema but not yet
  implemented in the vision service — the gap is tracked explicitly in
  `detectors.NOT_YET_IMPLEMENTED` and there is a test asserting it stays honest. Start with
  `fill_level`: it unlocks the trash, dish-rack, and pet-bowl examples.
- **A notification channel.** Signal, Telegram, Gotify, Pushover. The pattern is 30 lines.
- **Frontend.** The SvelteKit app is a scaffold. Read
  [docs/UX_NEURODIVERGENT.md](docs/UX_NEURODIVERGENT.md) first — it lists the things that will be
  pushed back on in review, and they are not obvious.
- **Hardware notes.** Ran this on a Pi 5 with a Hailo, or an Orin, or a Coral? Real numbers in
  [docs/HARDWARE.md](docs/HARDWARE.md) are worth more than any amount of code. This has tested on real hardware, but it's hyper-specific hardware and having a more generalized vision is better.
- **Better clutter scoring.** The three-way fusion in `vision-service/openhup_vision/fusion.py` is a
  reasonable first attempt, not a finished answer.

[docs/DEVELOPERS.md](docs/DEVELOPERS.md) has the architecture, the design rules, and the
step-by-step for each extension point.

## Design constraints that will not be relaxed

Not style preferences. Each one prevents something specific.

**`now` is always a parameter.** `evaluate()`, `advance()`, `simulate()`, and `Engine._evaluate()` take
the current time as an argument. This is what makes grace periods, cooldowns, and clock edges testable
without sleeping, and what makes replay behave identically to live operation.

**The engine core is pure.** `backend/openhup/skills/` does no I/O. Effects are dataclasses executed
elsewhere. If you need a database inside `skills/`, pass the fact in through `EngineContext`.

**Absence of data never resolves anything.** A dead camera must not tidy someone's house. `Verdict`
carries `missing` and `stale`; the FSM refuses to act on an unevaluable resolve verdict.

**No feature may depend on the LLM.** Every surface has a deterministic template fallback, so a
slow or unavailable model degrades gracefully instead of wedging the house. The AI layer is core:
no shipped personality switches it off - the quietest voice (`brief`) still calls the model, and
the template fallback is resilience, not a choice.

**Safety outranks tone.** `urgency >= high` bypasses the personality layer in code, not by asking a
model nicely.

**No counts of unfinished work.** Anywhere, in any form, including badges. The reasoning is in
[docs/UX_NEURODIVERGENT.md](docs/UX_NEURODIVERGENT.md) and it is the difference between a tool people
keep and a tool people uninstall.

## Things that will be declined

Please do not spend time on these, you will be closed on the spot:

- Face recognition **without consent**. The consent-gated identity layer (ADR-016) is implemented:
  an embedding is stored only at the moment a person says yes, and identity is presence, never
  attribution. Anything that recognises, names, or tracks a person without that consent will be
  closed on the spot.
- Screen-content classification (what someone is watching).
- Ambient audio capture or voice identification. The opt-in voice _interface_ (speech-to-text and
  text-to-speech in the browser) is supported. See [docs/VOICE.md](docs/VOICE.md). But continuous
  room listening, storing/transcribing audio, and identifying people by voice are not.
- Telemetry, crash reporting, or usage analytics, including opt-in.
- Cloud sync, hosted accounts, or a remote control plane.
- Task counts, "overdue" states, or streak-broken notifications.
- Gamification that ranks household members against each other.
- Making Ultralytics YOLO the default detector. It is AGPL-3.0 and stays opt-in (ADR-004).

Each is a position rather than an oversight, argued in the docs. If you think one is wrong, open an
issue and make the case before writing code.

## Pull requests

1. One concern per PR. A detector _and_ a refactor is two PRs.
2. `make check` passes.
3. Tests for behaviour, not implementation. If you touched `skills/`, include a test for what happens
   when data is **missing**. That is the first question a reviewer will ask.
4. Comments explain _why_. If a comment restates the code, delete it.
5. In the description: what behaviour changed, and why. Not a diff summary.
6. Changed a default or a threshold? Say what you observed that motivated it. Defaults here are
   load-bearing UX decisions, not arbitrary numbers.

Do not commit: model weights (CI fails on this), `deploy/env/openhup.env`, snapshots, or anything
generated into `packages/openhup-schemas/jsonschema/`.

## Adding a model to the registry

`models/registry.yaml` must record the upstream URL, a sha256, and the **licence**. CI fails if a
licence is missing, and fails if any weights file is committed. If a model's licence would impose
obligations on users' deployments (AGPL, non-commercial, research-only), it must be `optional: true`
and, for the awkward cases, `requires_explicit_consent: true`.

This is not bureaucracy. An Apache-2.0 project that quietly ships AGPL weights has misled everyone who
deployed it.

## Reporting bugs

Include:

- what you expected, what happened
- `GET /api/v1/system/info` and `GET /api/v1/system/health` output
- the relevant skill YAML (safe to paste — it contains no secrets by design)
- your hardware and inference profile (`python -m openhup_vision.backends --info`)

For "it fires too often" or "it never fires", the single most useful thing you can attach is the output
of `POST /api/v1/skills/{id}/simulate`. It usually contains the answer.

**Security issues**: do not open a public issue. See
[docs/SECURITY_PRIVACY.md](docs/SECURITY_PRIVACY.md).

## Code of conduct

Be a decent human being. This project is partly aimed at people managing executive-function difficulty, chronic
illness, and depression, some of whom will be in the issue tracker. Assume good faith, do not
diagnose anyone, and remember that the person filing an untidy bug report may be having a hard life.

Unacceptable behaviour can be reported to the maintainers privately.

## Licence

Apache-2.0. By contributing you agree your work is licensed the same way. No CLA.
