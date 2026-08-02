# Architecture Decision Records

Short records of choices that were genuinely contested, with the alternative that was rejected and
the condition under which we'd revisit. Format: context → decision → tradeoff → revisit trigger.

---

## ADR-001 — Python everywhere in the backend, not Go

**Context.** The API, skill engine, vision pipeline, and LLM integration all need to share the same
wire types. The vision service must be Python (ONNX Runtime, OpenCV, model tooling all live there).

**Decision.** Python 3.12+ for backend and vision service. FastAPI + Pydantic v2 + SQLAlchemy 2.0
async + Alembic.

**Tradeoff.** Go would give a single static binary, better long-lived concurrency, and lower memory
for the API — genuinely attractive for a self-hosted appliance. But it would mean two languages,
two schema definitions (or codegen between them), and two dependency toolchains for a project whose
hardest code is unavoidably Python. One language beats a marginally better API runtime here.

**Mitigations for Python's weaknesses.** No CPU-bound work in the API process; async DB driver
(asyncpg); the engine runs as a separate process; `uvicorn --workers N` behind the proxy.

**Revisit if.** The API becomes latency-critical for many concurrent WS clients (unlikely for a
household), or someone wants a single-binary distribution badly enough to maintain a Go rewrite of
just the API layer.

---

## ADR-002 — Redis Streams as the event bus

**Context.** Observations flow from N vision instances to one engine. We need at-least-once
delivery, consumer groups, short replay, and one fewer moving part.

**Decision.** Redis Streams. Redis is already wanted for caching, leader election, rate limiting,
and window buffers.

**Alternatives.**
- *NATS / JetStream* — better semantics and lower latency, but an extra service to run and learn
  for a household-scale deployment. Reconsider at multi-host scale.
- *MQTT (Mosquitto)* — great for home automation interop, weak on consumer groups and replay.
  Chosen as an **integration surface** (bridge in/out) rather than the core bus.
- *Postgres LISTEN/NOTIFY* — zero extra infra, but no replay, 8 KB payload limit, and it couples
  bus health to DB health.
- *Kafka* — absurd at this scale.

**Tradeoff.** Redis persistence needs configuring (AOF everysec) or a crash can lose seconds of
observations. Acceptable: an observation is a sample of a continuous state, and the next one arrives
in seconds. Tasks and alerts are in Postgres, not the bus, so nothing user-visible is lost.

---

## ADR-003 — Separate "observation" from "decision"

**Context.** The tempting design puts thresholds in the vision service ("emit event when clutter >
0.6"). That couples policy to inference and makes skills unable to share detector output.

**Decision.** Vision emits raw typed signals with no policy. All thresholds, timing, hysteresis,
and effects live in the skill engine.

**Payoff.** Retune a threshold without restarting inference; run five skills off one detector pass;
replay stored observations to simulate a new skill against history (`/skills/{id}/simulate`);
swap YOLOX for RT-DETR without touching a single skill definition.

**Cost.** More observations on the bus and in the DB than strictly needed. Mitigated by
dead-banding (only emit when a signal moves more than ε, or every Nth sample regardless) and by
retention on the observations table.

---

## ADR-004 — ONNX Runtime as the inference layer, permissive models by default

**Context.** Users' hardware spans Raspberry Pi 5, Intel N100 iGPU, Coral, and used NVIDIA cards.
Ultralytics YOLO is the most popular option and is **AGPL-3.0**.

**Decision.** ONNX Runtime with a switchable execution provider (CPU / OpenVINO / CUDA / TensorRT).
Default models are Apache-2.0 or MIT (YOLOX, RT-DETR/D-FINE, CLIP). Ultralytics is an opt-in
backend; no AGPL weights or code are vendored. Weights are downloaded and hash-verified at setup
time via `models/registry.yaml`, never committed.

**Tradeoff.** Ultralytics has nicer ergonomics and slightly better accuracy per FLOP on some tasks.
Not worth exporting an Apache-2.0 project's users to AGPL obligations by default.

---

## ADR-005 — Fuse three clutter signals instead of picking one

**Context.** "Is this surface messy?" has no off-the-shelf model. Object detection alone misses
non-COCO clutter (mail, wrappers, cables). Pixel diff alone breaks on lighting changes and chairs
moving. CLIP alone is noisy and uncalibrated per-room.

**Decision.** Fuse baseline embedding+structural diff, ROI object density, and a CLIP zero-shot
tidy/cluttered probe, with per-anchor weights. Always publish the three components alongside the
fused score.

**Tradeoff.** More compute per observation and more calibration surface. Bought with per-anchor
baselines and a sensitivity curve, plus the explainability the UI needs to make calibration
possible for a non-expert.

**Revisit if.** A well-licensed, purpose-trained "surface tidiness" model appears, or enough users
opt into contributing labelled snapshots to train one (opt-in only, never default).

---

## ADR-006 — SvelteKit for the frontend

**Context.** The UI is dashboard-shaped: live snapshots, a task list, a skill builder, charts.
It runs on a low-power self-hosted box and is often opened on a phone.

**Decision.** SvelteKit + TypeScript + Tailwind, built as a PWA, served as static assets by the
backend (or Caddy). Types generated from the backend's JSON Schema.

**Alternatives.** React/Next.js has the deeper ecosystem and more contributors — a real argument
for an OSS project. Rejected because the app needs no SSR complexity, and Svelte's smaller bundle
and simpler reactive stores fit a WebSocket-driven live view with less code. HTMX + server
templates was considered and rejected: the ROI polygon editor and live charts want a real client.

**Revisit if.** Contributor traffic clearly stalls on Svelte unfamiliarity.

---

## ADR-007 — PostgreSQL, with TimescaleDB optional

**Context.** Relational data (skills, tasks, episodes) plus time series (observations, metrics).

**Decision.** Plain PostgreSQL 16 with BRIN indexes on time columns and monthly partitioning for
`observations`. TimescaleDB is an optional drop-in for users with heavy retention needs.

**Alternatives.** SQLite would simplify single-box deployment considerably and is tempting for the
"just run it" path — but concurrent writers (API + engine + rollup worker) and partial-index needs
make Postgres the safer default. A SQLite profile for single-camera installs is a reasonable future
addition. InfluxDB/Prometheus for metrics was rejected: a second datastore for data that must join
against tasks and episodes.

---

## ADR-008 — The AI layer is core; every surface keeps a deterministic fallback

**Context.** The assistant's voice, memory, and noticing are the product. A deployment without a
provider is a broken deployment, not a cheaper one. Local LLMs on a mini PC are slow and
occasionally unavailable, and remote LLMs are a privacy decision the operator must make
consciously - both real constraints, neither a reason to make the brain optional.

**Decision.** The AI layer is core: the setup wizard always asks for a provider (local Ollama by
default, or a trusted cloud provider behind an explicit egress gate), `/system/health` reports a
missing provider as a problem, and a startup warning says plainly that every surface is degrading
to templates. At the same time, every LLM surface keeps a deterministic template fallback so a
slow or momentarily unavailable model degrades gracefully rather than wedging the house. Skill
parsing failure falls back to a structured form; `brief` is the quietest user-facing voice and
still calls the model, and a misconfigured install is flagged by health rather than silently
degrading. Remote providers require an explicit config flag plus a redaction profile.

**Consequence.** No *single surface* may depend on an LLM to function correctly — the LLM improves
phrasing and lowers the authoring barrier — but the operator is never allowed to configure the
brain away. This is also what makes the system testable: the `echo` provider makes every test
deterministic.

---

## ADR-009 — Personality is a filtered rendering layer, not a system prompt

**Context.** "Roast me" features fail in two directions: bland, or genuinely hurtful. Prompt text
alone cannot be trusted to hold a line.

**Decision.** Personality is a config object (tone, vocabulary, boundaries, intensity, templates)
applied *after* the facts are decided, with a deny-list filter on the output and a hard bypass for
`urgency >= high`. Intensity is user-adjustable and capped by an operator-level `humor_ceiling`.

**Tradeoff.** Less creative range than an unconstrained persona prompt. Correct call: the failure
mode of an unfiltered roast bot aimed at someone's home and executive function is severe, and a
30-word cap plus a deny-list costs almost nothing.

---

## ADR-011 — Voice is a gated interface, not ambient audio capture

**Context.** The project originally declined "audio" outright. That position was really about
*ambient audio capture* - a microphone listening to a room all the time - which stays declined. It
was never intended to ban a *voice interface*: a person asking the assistant something and getting
a spoken answer. The distinction matters, so it is recorded here rather than left implicit.

**Decision.** Add speech-to-text and text-to-speech as an opt-in interface. By default both run in
the browser via the Web Speech API, so no audio and no transcript leave the device at all. The wake
word is matched locally in the client before anything is processed. A server-side path exists for
operators who configure one, gated by `voice.allow_remote_voice` (the same shape as
`llm.allow_remote_llm`) and recorded in the same usage audit log.

**What voice does.** STT feeds a *deterministic* command router - task control ("done", "snooze for
an hour"), queries ("what should I do"), navigation, and skill dictation through the existing
natural-language parser. TTS speaks the replies, plus safety alerts and new-task nudges. Nothing
depends on the LLM; the router is keyword matching, exactly like the skill parser's fallback.

**What stays declined.** Continuous ambient recording, storing or transcribing audio, wake-word
matching on the server, and any identification of people by voice. Audio is never written to disk
and never retained.

**Tradeoff.** Browser speech quality and locality depend on the browser (Chrome's recognition, for
example, uses its own speech service). A truly on-device stack - a WASM keyword spotter for the wake
word plus local Whisper/Piper - is the honest next step for people who want zero browser
involvement; the provider seam in `openhup/voice` and the client-side speech module were built for
that swap.

**Revisit if.** Users ask for wake-word listening that works with the tab closed, or for voice
without any browser dependency - both point at a local model path rather than the Web Speech API.

---

## ADR-010 — Anchors, not cameras, are the unit of watching

**Context.** Skills need to survive a camera being re-aimed, replaced, or renamed. A task about
"the kitchen counter" should not be tied to `rtsp://192.168.1.42/h264`.

**Decision.** An `Anchor` is a named ROI (polygon + optional clean baseline) that belongs to a
camera but has its own stable identity and history. Skills watch anchors; tasks, episodes, metrics,
and baselines all key off anchors.

**Payoff.** Swap the camera hardware, redraw the polygon, keep every metric and streak intact.
Also makes multi-anchor cameras (one wide shot covering counter, sink, and stove) natural, which is
how people actually mount cameras.

---

## ADR-012 — Memory is a teachable local store, not learned behaviour

**Context.** An assistant that phrases tasks has a wall it cannot cross: it cannot use the household's
own words or facts. It cannot say "the junk room" unless told, cannot know bin day, and every
conversation starts from zero. But "memory" has a dark shape in a home product: a system that
quietly accumulates facts about people it observed, with no way to inspect or delete them, is
surveillance wearing an assistant costume.

**Decision.** Add a deliberately small memory: plain, human-readable facts the household teaches in
its own words ("I call the spare room the junk room", "bin day is Tuesday"), stored in local
Postgres. Voice can teach, recall, and forget them; the Settings review screen can list and delete
every one. Nothing is ever learned by observation - the assistant has no opinion about your life
until you hand it one.

**Where memory touches the LLM.** Only as context: when a task line is phrased, the relevant facts
are retrieved by keyword matching (no embeddings, no model - the store is hundreds of rows, not
millions, and nothing here may depend on the LLM, ADR-008) and injected into the prompt as things
to keep in mind. That is the *only* way a fact leaves the house, and it is already gated by
`llm.allow_remote_llm`, subject to the redaction profile, and recorded in the usage audit. The
`plain` line never includes memory: the tone-free wording stays factual.

**What stays declined.** Learned or inferred facts, per-person memories, embeddings pipelines, and
any retention that is not user-visible. A fact that is no longer trusted is deleted, never edited
silently, and "forget everything" wipes the store. Retention is by user action, not a timer.

**Tradeoff.** Facts are only as good as what was taught - there is no proactive "I noticed you do
X" yet. That is a feature: pattern discovery from metrics is the next step, and it must be
reviewable in exactly the same way before it earns a place.

**Revisit if.** Users want the assistant to volunteer patterns it derived from the metrics it
already records - that is a separate feature with the same reviewability bar.

---

## ADR-013 — Learned patterns are derived, reviewed, and forward-facing

**Context.** ADR-012 deliberately stopped at *taught* memory: the assistant has no opinion about
your life until you hand it one. The obvious next step is the assistant noticing the trash fills
about every three days by itself. But learned memory has a sharper edge than taught memory - a
system that quietly accumulates inferences about you is how a helpful assistant becomes a
surveillance system - and the tempting framing ("the counter has been undone for 6 days") is
precisely the guilt the project exists to avoid.

**Decision.** Patterns are derived deterministically from the episodes the skill engine already
records, never from a model, and only when there is evidence: at least four episodes spanning ten
days, a cadence between one day and a month (everyday activity is not a pattern), and a confidence
score that grows with sample size and shrinks with spread. Claims are forward-facing by
construction - "usually needs attention about every 3 days" is the only shape that exists, so the
safety filter never has to catch a backwards one. They are keyed by (skill, anchor), never per
person.

**The reviewability bar is identical to taught facts.** Every pattern is listed in Settings with
its evidence (how many episodes, over what span), dismissable, and a dismissed pattern is never
learned again. A pattern is context for phrasing prompts exactly like a taught fact - gated,
redacted, and audited when it leaves the house - and never appears in `plain` text.

**Proactive nudging is a nudge about a skill, so it obeys that skill's own limits.** A cadence
pattern may speak only inside a predicted window around its median interval, at most once per
episode cycle, only when no task or alert already covers that spot - and it is governed by the
subject skill's `limits` exactly as the skill's own triggers are: `quiet_hours`, `cooldown`, and
`max_per_day` all apply. There is no hidden global cap to discover and fight; a household that
wants the assistant to stay on it about the trash sets `max_per_day` high and `cooldown` short,
the same way it tunes the skill itself. Every guard is enforced in code, not requested of a model.

**What stays declined.** Inferred facts about people, per-person patterns, embeddings pipelines,
and any proactive claim that is not derived, reviewable, and forward-facing.

**Revisit if.** Users want pattern-driven nudges on notification channels (not just spoken) - a
small extension to the same guarded path.

---

## ADR-014 — The personality gamble: a voice drawn, never announced

**Context.** ADR-009 made personality a filtered rendering layer, and the presets so far are a menu:
Kind Coach, Deadpan Butler, Chaos Goblin, Drill Sergeant (Lite), Brief. A menu is the safe default,
but it is also boring - and "personal" for a household assistant is largely about voice. The
project wanted a setup experience with a twist: the assistant gets a personality you did not
choose, and you discover it by living with it.

**Decision.** The voice is chosen **once, at setup** (`openhup setup`), where the five voices are
shown and the user either **picks one** or **gambles**. A gamble means: on first launch one of the
five - `friendly`, `shy`, `sassy`, `sarcastic`, `angry` - is drawn at random and becomes the
effective default *without being announced*. From that point the voice is never shown again: the
user discovers it by living with it, and the only place the answer is written down is the
configuration - `default_personality` (a pick) or the `personality_draw` row (a gamble). The
settings screen will happily re-draw (each re-draw is explicit and counted) or switch the gamble
off, but it will not name the voice. The docs and the preset file document the pool.

**The invariants that keep a mystery voice honest.**

* **A draw is never clamped in secret.** All five pool presets sit at intensity 3 or below, so the
default `humor_ceiling: 3` with no `roast_consent` can never quietly tone down what was drawn. A
wider operator pool is validated against the loaded personalities at draw time.
* **The draw is an override, not a replacement.** `default_personality` in config stays the
operator's choice; the draw shadows it in memory, and deleting the draw restores it exactly. The
engine picks up a re-draw within a minute without a restart.
* **The five voices are bounded by the same filter as every other personality.** "Angry" is
gruff at the mess and the clock, never at the person; "sarcastic" drops the irony for a genuine
win. Shame language, backlog counts, and coercion are force-added boundaries at intensity 4+ and
listed explicitly for these presets, and the safety filter runs on whatever a model produces.

**What stays declined.** Personality per person (voice-ID and per-person tracking remain off), any
personality that targets the person rather than the objects, and a draw that could silently resolve
to a clamped version of itself.

**Tradeoff.** A surprise is not for everyone, so a pick is always offered alongside the gamble at
setup, and a household that hates its voice re-draws or turns the gamble off without ever being
told what it was. The mystery is enforced socially (no surface announces the draw), not
cryptographically - an operator who looks at `/system/info` or the database can find the answer,
which is the point of "documented, not secret".

**Revisit if.** Users find the never-announced voice hostile rather than charming and want the
settings screen to name it, or want a re-draw to be time-boxed to setup rather than available
forever.

---

## ADR-015 — Wins: the assistant notices progress, not just problems

**Context.** Everything the system says, unprompted, is about something being wrong: a task, an
alert, a nudge, a due pattern. The data to notice *progress* is already there - episodes close,
streaks form, the counter stays clear - and the UX doc has long listed "before/after pairs from
weeks ago, surfaced unprompted" as worth building. A voice that only ever nags is exhausting; one
that notices when a place stays clean is what makes a personality feel like it cares.

**Decision.** *Wins*. When a task resolves, the executor computes the clear stretch for that
(anchor, skill) from the episode history - the gap between the previous episode closing and this
one opening - and celebrates at most one thing about it:

* **A band milestone.** The stretch crossed a whole-day band (1, 3, 7, 14, 30 days) without
  setting a record. The band floor is the dedupe key, so each band is celebrated once ever.
* **A 90-day record.** The stretch is the longest for that anchor in the trailing 90 days, beating
the previous best by at least half a day (float noise must not re-celebrate). A record outranks
the band claim: the line already names the length.

**The guards, all in code.** Only the most recent episode can produce a win (everything earlier was
celebrated when it closed). Wins fire only when there is evidence - the previous episode actually
closed, and the stretch is at least a day; sub-day clear cycles are life, not achievements. Each
win is claimed once, deduped by the `win_milestones` ledger (unique on anchor, kind, value), which
is also the review screen: *Wins the assistant has noticed*, same reviewability bar as facts and
patterns. The spoken note is rendered through the current personality (a win has a voice; the
`brief` personality gets the plainest line) and passes the same safety filter as everything else.

**Forward-facing only, by construction.** The claim shape is "has stayed clear for N days" - how
long things have been good. There is no code path that can produce "you left it for six days" or
"your longest stretch was last month"; the safety filter never has to catch one, and tests assert
this. Quiet hours suppress the *spoken* note but never the milestone itself: a 2 a.m. clear is
still a clear, it just is not announced twice.

**What stays declined.** Wins for alerts (a resolved burner is a relief, not a celebration),
comparisons between household members, any per-person framing, and any claim that implies a broken
streak or a fall from grace.

**Tradeoff.** The first stretch ever recorded is automatically a record, because there is no
benchmark - defensible, since it is genuinely the longest known, and the band milestones keep
steady progress celebrated afterwards.

**Revisit if.** Users want wins on notification channels (a small extension of the same
quiet-hours-gated path), or want the review screen to show the before/after snapshot pair behind
each win.

---

## ADR-016 — Identity is consented, never inferred

**Context.** "No face recognition" was a position, not a gap, and it was the project's hardest
line. It existed because face recognition without consent is surveillance, and because a wrong
name is the worst thing a nagging assistant can say. But it left a real hole: in a shared house
the assistant could not answer "who left the plates" - and the honest version of that question
is "who was in the kitchen when the plates appeared", which the detectors could already see
(`person_count`) without any identity at all. The missing layer was never the detector; it was
a consent flow that lets a person *choose* to be known.

**Decision.** *Identity is a person saying yes, and nothing else.* The camera computes an
embedding for every face it sees, but an embedding is only ever stored when the person answers
"yes" to the consent question - asked once per anchor per day, in the active personality, spoken
if voice is on. A "no" (or silence) writes nothing but a date marker: the system remembers that
it asked, never what the person looked like, and asks again tomorrow. A "yes" enrolls the
person into the household member store: name (what they gave, never guessed) + embedding, in
local Postgres, deletable like a fact, reviewable in Settings.

**What identity may do.** An enrolled member's name appears in presence windows ("kitchen
occupied 19:40–20:10, Sam"), in win notes ("the counter stayed clear 3 days"), and as a target
for task commands, nudges, and queries - all through the same deterministic router and renderer
as everything else. Identity is *ambient context*, never a skill trigger: no skill can ever
fire on "Sam is in the kitchen". Skills and their tests are untouched.

**What identity may not do.** It is presence, not attribution. A presence window says a person
was *in* the room; nothing anywhere says they *did* anything - the "who left the plates"
question is answered as "someone was present when this happened", and the household does the
naming among themselves. Identity is never inferred from behaviour, never inferred from voice,
and never derived from the face without consent. The default is *on* only in the sense that the
consent flow is armed: nothing is stored until a real person answers yes in front of the camera.

**Tradeoff.** This reverses the plainest sentence in CONTRIBUTING.md and SECURITY_PRIVACY.md,
and a reversal this load-bearing must be recorded, not smuggled. The consent flow is the price
of admission: it is the only thing that turns "face recognition" from surveillance into a
feature. The remaining risk is a wrong match - lighting, twins, a similar face - so identity is
always a hint, never a verdict, and the nudge wording never accuses. Weights are fetched, not
committed, and the licence of every identity model is recorded in `models/registry.yaml` like
all the others.

**Revisit if.** Users report that the consent question itself is intrusive (the 24-hour marker
is the first dial), or ask for identity to survive a camera being re-aimed (enrollment is per
member, matching is per anchor - a gallery re-capture is the answer).
