# Designing for executive function

This document exists because the difference between a tool that helps and a tool that becomes another
source of guilt is almost entirely in the details, and those details are easy to get wrong while
believing you are being helpful.

It is written for two audiences: someone configuring OpenHup for themselves, and someone contributing
to the UI who needs to know which constraints are deliberate.

---

## The problem this is actually solving

"Tidy the shelf" is not a task. It is a project with no edges, no defined end state, and no obvious
first move. For a lot of people that is fine — they start somewhere and it resolves itself. For
someone with ADHD, or in a depressive episode, or exhausted, the absence of a defined first action is
exactly where the whole thing stalls. Then the shelf becomes a _feeling_, walking past it costs
something, and the cost compounds.

Standard to-do software makes this worse in a specific way: it converts a stalled project into a
permanent visible reminder of failure. The list grows. The count next to it grows. Every glance is a
small debit. Eventually the app gets closed and not reopened.

OpenHup has one structural advantage over a to-do list: **a camera can see when something is done.**
That is what makes the following possible.

---

## The design rules, and why each exists

### One task at a time

`mode: single_task_focus` is the default. The UI in this mode calls only `GET /api/v1/tasks/next` — so
the backlog is not filtered out on the client, it is _never sent_. A list you cannot see cannot
overwhelm you.

`ux.global_single_task_focus` extends this across all skills, for when lists themselves are the
problem rather than the solution.

### Never show a count

`ux.hide_task_counts` defaults to true, and `GET /api/v1/tasks` does not compute a total at all.

A number next to unfinished work is the single most reliable way to make someone stop opening the app.
It converts "here is a thing you could do" into "here is the size of your failure". There is a test
asserting the API response contains no total, because this is the kind of helpful-seeming feature that
gets added back by accident.

### Micro-steps, verified by the camera

`micro_steps: auto:3` splits an anchor into a ladder. With subregions defined on the anchor, the split
is **spatial**: "just clear the left third". That step is real, finishable, and — crucially —
verifiable. When that region's clutter score drops, the step is ticked off **by the next observation**.

You do not report progress. You do not remember you were doing it. You do not argue with a checkbox.
You move three mugs and the thing notices.

The worst region goes first, so step one is also the most visibly satisfying.

### Partial credit is permanent

Clearing one of three steps is logged as a win. The episode does not need to complete for the work to
have counted. If you stop after step one, step one still happened.

### Every task carries a picture

Visual anchoring is not decoration. "Clear the kitchen counter" is ambiguous — which part, how clear,
did I already do this? A photograph removes the ambiguity, removes the working-memory load of
reconstructing what was meant, and makes the task concrete enough to act on.

The "after" snapshot matters as much as the "before". `mode: archive` keeps before/after pairs past
the normal TTL, and seeing the "after" from three weeks ago is the most motivating thing this system
produces.

### Tasks expire quietly

`effect.expires_after: 3d` retires an unaddressed task with no notification, no summary, and no
mention of it later. It comes back when the cooldown lapses, with no reference to having been here
before — `backlog_counts` is a filtered boundary, so no personality can bring it up either.

A task that ages out is not a failure to be reported back. It is a task that stopped being relevant.

### The system argues at most once

`verify_on_manual_complete: true` means that when you press done, a fresh observation is taken. If the
camera disagrees, the task reopens **once**, with gentler wording. After that, you are right —
regardless of what the camera thinks.

Arguing with a human twice about the state of their own home is a bug. For the hardest spaces, set
`verify_on_manual_complete: false` and skip the argument entirely.

### Forward-facing language

Streak language points forward ("next win"), never backward ("you broke a 9-day streak"). Best-ever
streaks are recorded; broken ones are never mentioned. The weekly report leads with something that
went well, and is allowed exactly **one** suggestion — more than one reads as a lecture.

### Timing is a feature

`quiet_hours` on chore skills is non-negotiable. The ADHD shelf example fires only between 14:00 and
20:00: not first thing, when the day already has demands, and not late, when starting a project is a
bad idea. Waking up to a to-do list is how people uninstall things.

---

## Two modes, both legitimate

**Single-task focus** — one thing, verified, with a picture. For when the backlog is the problem.

**Continuous monitoring** (`mode: backlog`) — the full list, nothing hidden, nothing expiring. For
people who find hidden state more distressing than a long list, which is common among autistic users
and anyone who needs to trust that the system is not quietly deciding things.

Neither is the "advanced" mode. They are answers to different problems, and the same household may
want one for the kitchen and the other for the office.

---

## Choosing a personality

The personality system is not a gimmick; tone is an accessibility setting. What lands as warm for one
person lands as patronising for another, and what one person finds motivating another finds like being
shouted at.

| Personality           | Good for                                                                                                                                                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind_coach`          | the default. Warm, brief, never disappointed in you.                                                                                                                                                                               |
| `deadpan_butler`      | people who find encouragement patronising.                                                                                                                                                                                         |
| `chaos_goblin`        | people for whom absurdity beats sincerity. Requires `roast_consent`.                                                                                                                                                               |
| `drill_sergeant_lite` | people who respond to a timer and an imperative.                                                                                                                                                                                   |
| `brief`               | says what needs saying, nothing else. Still the model — no personality switches the AI layer off — but the fewest words, no warmth, no jokes. For shared displays, screen readers, and anyone who finds a chatty house unbearable. |

Five more ship as the **personality gamble** pool (ADR-014) — `friendly`, `shy`, `sassy`,
`sarcastic`, `angry`. With `personality.gamble: true`, one is drawn at random on a fresh install
and becomes the default without being announced; you discover it by living with it, and reveal or
re-draw it in Settings. All five sit at intensity 3 or below, so a draw is never quietly toned
down, and the same filter applies to all of them — "angry" is gruff at the mess and the clock,
never at you. If predictability matters more than novelty, leave the gamble off and choose
plainly.

`brief` is not a degraded mode. It is the quietest _voice_: the words are still chosen by the model,
just kept to the minimum. The deterministic templates underneath every personality are what keep the
product working when the model is slow or unavailable — resilience, not a personality choice.

### What no personality can do

Enforced in code (`openhup/llm/safety.py`), applied after generation, not requested in a prompt:

- No shaming language, ever.
- No comments on your body, appearance, or hygiene.
- No naming or implying a diagnosis. OpenHup is not qualified and it is not welcome.
- **No backlog counts.** "You've left this for six days" is filtered even in roast mode.
- No comparisons to other people. No household benchmark exists.
- No threats, ultimatums, or invented consequences.
- No remarks about other members of the household.

Output that trips a rule falls back to the deterministic template. It is not retried with a sterner
prompt — comedy is not worth a second round trip, and a model that just produced something cruel is
not the thing to ask again.

Roasting requires explicit opt-in (`roast_consent`), is capped by an operator-level `humor_ceiling`,
and is aimed at the mess rather than the person. "The counter has grown a civilisation" is the shape;
"you are a slob" is filtered.

### The safety exception

`urgency >= high` bypasses personality entirely. A burner alert reads:

> Front-left burner has been on for 12 minutes. No one has been in the kitchen for 9 minutes.

on every install, with every personality, at every intensity. Safety outranks comedy, and that is
enforced in code rather than left to a prompt.

---

## Configuration recipes

**ADHD / executive-function difficulty**

```yaml
effect: { mode: single_task_focus, micro_steps: auto:3, urgency: low, personality: kind_coach }
resolve: { grace: 10m, verify_on_manual_complete: false, auto_expire_after: 3d }
limits: { cooldown: 6h, max_per_day: 1, quiet_hours: { between: ["20:00", "14:00"] } }
snapshot: { attach: true, mode: archive, retention: 30d }
```

Plus subregions on the anchor, so the ladder is spatial and self-verifying.

**Predictability preferred (common for autistic users)**

```yaml
effect: { mode: backlog, micro_steps: none, personality: brief }
resolve: { grace: 0s, verify_on_manual_complete: true } # no auto_expire_after: nothing vanishes
limits: { cooldown: 1h } # no max_per_day: nothing is hidden
```

Fixed `time_window`s so behaviour is identical every day, and `ux.hide_task_counts: false` if seeing
the whole picture is the reassuring thing rather than the stressful one.

**Low-energy periods** — raise `cooldown`, drop `max_per_day` to 1, shorten `expires_after`, and
switch to `kind_coach` or `brief`. It is entirely reasonable to set `engine.paused: true` for a week.
Nothing is lost and nothing accumulates.

**Sensory sensitivity** — `emoji: none`, `max_per_hour: 3`, one quiet channel, long quiet hours.

---

## For contributors working on the UI

Things that will be pushed back on in review:

- Any badge, count, or number representing outstanding work.
- Red as the colour for "not done". Red means unsafe, and only that.
- Progress bars that show how far from complete something is, rather than how much has been done.
- "Overdue", "late", "missed", "you forgot", or a streak-broken notification.
- A task list in single-task-focus mode, even collapsed. Do not fetch it.
- Confirmation dialogs on completing a task. Trust the press.
- Animations longer than ~200 ms in a flow someone uses many times a day.

Things worth building:

- The "after" snapshot, shown large, on completion.
- Before/after pairs from weeks ago, surfaced unprompted.
- A single, obvious primary action per screen.
- Snooze that is as easy to reach as complete. Sometimes the right answer is "not now", and making
  that awkward just teaches people to dismiss things instead.
