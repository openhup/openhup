# Security and privacy

OpenHup watches the inside of your home. That deserves a document that is specific about what is
guaranteed, what is merely default, and where the sharp edges are.

## The short version

- Frames are decoded in RAM and **discarded** unless a skill explicitly attaches a snapshot.
- Snapshots get a per-skill TTL (7 days default) and can blur people **before** the JPEG is written.
- The default LLM is local. A remote one is refused at startup unless you set a flag and pick a
  redaction profile, and every outbound call is logged with its byte count.
- Nothing binds to a public interface by default. There is no cloud account, no phone-home, and no
  telemetry of any kind.
- Identity is **consented, never inferred** (ADR-016). An embedding is computed for a face only to
  ask a question, and is stored only when the person answers yes. A no writes a date marker, never a
  face. Nothing is recognised until someone chooses to be known.
- Voice runs in the browser by default (Web Speech API): no audio and no transcript reach OpenHup.
  A remote speech provider is refused at startup unless `voice.allow_remote_voice: true`, and is
  audited per call. See [VOICE.md](VOICE.md).
- Memory has two halves: facts you explicitly taught, in your words, and patterns the assistant
  *learned* from the episodes the engine already records ("the trash usually fills about every 3
  days"). Both live in local Postgres, are reviewable and dismissable in Settings, and the only way
  either leaves your network is as a snippet inside a phrasing prompt sent to your LLM provider —
  gated, redacted, and audited like every other call. Learned claims are forward-facing only, keyed
  by place and skill (never per person), and dismissed patterns are never learned again. See
  ADR-012 and ADR-013.
- Wins are the assistant noticing progress — "the counter stayed clear 3 days". They are derived
  from the episodes the engine already records, forward-facing only, celebrated at most once per
  milestone, and reviewable and forgettable in Settings like everything else the assistant thinks
  it knows. See ADR-015.

---

## What is stored, where, and for how long

| Data | Where | Default retention | Turn it off with |
|---|---|---|---|
| Video frames | RAM only | never written | n/a — no code path writes raw frames |
| Snapshots (task/alert imagery) | `/var/lib/openhup/snapshots` | per-skill, 7d default, 90d ceiling | `snapshot.attach: false` |
| Observations (numeric signals) | Postgres | 14 days | `RETENTION` in `db/models.py` |
| Tasks, alerts, episodes | Postgres | episodes 400d, rest indefinite | delete via API |
| Metric points | Postgres | indefinite (tiny) | delete via API |
| Notification log | Postgres | 30 days | config |
| LLM call audit | Postgres | 30 days | config |
| Camera credentials | environment only | n/a | never in the database or config files |
| Voice (commands & replies) | RAM only | never written | n/a — nothing is recorded or retained |
| Memory facts (what you taught it) | Postgres | until you delete them | `DELETE /memory/{id}`, Settings, or "forget everything" by voice |
| Memory patterns (what it learned) | Postgres | until you dismiss them | `DELETE /memory/patterns/{id}` or Settings |
| Wins (progress it noticed) | Postgres | until you delete them | `DELETE /personality/wins/{id}` or Settings |
| Member identity (name + face embedding) | Postgres | until you forget them | Settings → *Who lives here*, or consent "no" at first sight |
| Unknown-face consent markers | Postgres | 24 hours | n/a — a marker is a date, never a face |

Retention is enforced by a reaper driven by **sidecar files**, not by the database. Each snapshot is
written with a `.json` recording its expiry, so retention still works correctly if Postgres is down,
if the row was deleted, or if you restore an old backup. An image whose sidecar is missing is deleted
after a week on mtime — an orphan with no recorded expiry is exactly the file that would otherwise
live forever.

### Snapshot modes

| Mode | Behaviour |
|---|---|
| `ephemeral` | used for detection, never written. `attach: true` with this mode is a config error. |
| `thumbnail` | 160 px long edge — enough to recognise the place, not to read a document on the counter |
| `full` | full-resolution JPEG at the configured quality |
| `archive` | before/after pairs kept past the normal TTL, for progress history |

Where several skills watch one anchor, **the strictest policy wins**. One skill wanting ephemeral
snapshots makes the whole anchor ephemeral. A privacy setting that can be widened by adding a skill is
not a privacy setting.

### Redaction happens before encoding

`redact: [faces]` blurs person boxes in the numpy array and only then hands it to the JPEG encoder.
Unredacted pixels never reach the filesystem, so there is no window in which a stray copy exists and
no reliance on a later cleanup pass.

The blur is irreversible (downsample-and-repeat, not a reversible filter), and it is sized as a target
block count rather than a divisor, so it stays destructive on a 30-pixel distant face as well as a
300-pixel close one.

**If redaction is requested but no person boxes are available** — because the object detector is not
running — OpenHup **refuses to write the snapshot at all** rather than writing an unredacted frame.
Writing one "just this once" is how privacy guarantees die.

---

## LLM egress

This is the only place where data can leave your network by design, so it is gated three ways.

1. **Local by default.** `provider: ollama` against `127.0.0.1`. Nothing leaves.
2. **Explicit opt-in.** A remote provider raises a startup error unless `allow_remote_llm: true`. The
   error tells you what you are agreeing to.
3. **Redaction profile**, required with any remote provider:

| Profile | What goes over the wire |
|---|---|
| `text_only` (default) | facts and labels only: "clutter_level 0.72 on Kitchen counter". No images. |
| `redacted_image` | snapshots with people blurred |
| `full` | anything, including unredacted imagery. You are choosing this knowingly. |

Every call is recorded — provider, model, purpose, whether it was local, prompt bytes, response bytes,
whether an image was attached — and served at `GET /api/v1/system/llm-usage`. The claim "you can see
exactly what left the house" has to be inspectable to mean anything.

Household memory (facts and learned patterns, see ADR-012 and ADR-013) follows the same rule. It
is stored locally, never sent on its own, and only reaches a provider as fragments inside a
phrasing prompt — which is a call like any other, with the same gate, redaction profile, and audit
row. The retrieval that selects facts, and the discovery that derives patterns, both run on your
hardware; neither is a model.

At the systemd level, the API and engine units set `IPAddressDeny=any` with only loopback and RFC1918
allowed. With a remote LLM you must open that explicitly, which makes the policy enforced by the init
system rather than merely intended.

---

## Threat model

Threats taken seriously, and what is actually done about each.

### A compromised camera

The most likely one. Cheap camera firmware is the least trustworthy software on a home network, ships
with known CVEs, and often phones home by default.

- Put cameras on a **VLAN or subnet with no route to the internet.** This single step matters more
  than everything else in this document.
- OpenHup connects *to* cameras and never accepts connections from them (except `agent_push`, which is
  authenticated with a bearer token).
- Credentials are per-camera, so one compromised camera does not yield the others.
- Camera credentials live in the environment, never in the database, so a database dump is not a
  credential leak.

### A stolen disk

Snapshots and the database are imagery of, and metadata about, the inside of your home.

- Put `/var/lib/openhup` and the Postgres data directory on **LUKS**. Full-disk encryption at rest is
  the correct answer here, and application-level encryption is not a substitute for it.
- Keep retention short. The best protection for imagery is not having it: the default seven days
  reflects that.
- `thumbnail` mode is genuinely useful — a 160 px image identifies the place without being a usable
  photograph of anything on the surface.

### Accidental public exposure

The failure that hurts most, and the easiest to make.

- `bind_host` defaults to `127.0.0.1`.
- Binding to a non-loopback address with `require_auth: false` is a **startup error**, not a warning.
- No UPnP, no automatic port mapping, no cloud relay. If OpenHup is reachable from the internet, it is
  because you configured that.
- Prefer Tailscale or WireGuard over a public hostname. Nothing published means nothing to attack.
- Snapshots are served **through the API**, not as a static directory, so authentication applies to
  imagery. A misconfigured reverse proxy cannot accidentally expose the snapshot folder.

### A hostile device on your LAN

- Snapshot paths are checked against traversal.
- Services authenticate with bearer tokens; browsers use signed session cookies.
- The API performs no CPU-heavy work on unauthenticated paths, so it is a poor DoS target.
- Redis and Postgres should not be published. The compose file does not publish them.

### An untrustworthy household member

Worth stating plainly, because it is a real dynamic in shared homes.

- `hide_task_counts` is on by default: no scoreboard of anyone's undone work.
- `third_party_remarks` is in every shipped personality's `never` list — the system does not comment on
  other people in the house.
- `humor_ceiling` is operator-level and caps everyone, but roast intensity should be the choice of the
  person being roasted. If you are configuring this for someone else, ask them.
- Identity is per *person consenting*, never per person generally: only enrolled members are ever
  named, and names come from the person, not the system. There is still no per-person behavioural
  tracking — metrics, patterns, and wins are keyed by place and skill, and a presence window says a
  person was *in* a room, never that they *did* anything.

---

## Things OpenHup deliberately cannot do

Refused as a matter of design, not configuration:

- **Face recognition without consent.** The identity layer (ADR-016) exists, and its entire point is
  that it cannot start without a person saying yes: an embedding is only stored at the moment of
  consent, a "no" stores nothing, and identity is presence, never attribution or behaviour tracking.
- **Content classification of screens.** `screen_on` measures brightness and temporal variance. It is
  architecturally incapable of telling what you are watching, which is a much stronger promise than a
  policy saying we will not look.
- **Ambient audio capture and voice identification.** Cameras have no microphones and there is no
  per-person voice model. The voice *interface* is browser-only, wake-word-gated, and discards audio
  after each command — see [VOICE.md](VOICE.md).
- **Per-person behavioural tracking.** Metrics are per anchor, never per person.
- **Cloud sync or remote management.** There is nowhere for data to sync to.
- **Telemetry, crash reporting, or usage analytics.** None, including opt-in. Nothing to leak.

---

## Hardening checklist

Roughly in order of value per minute spent.

- [ ] Cameras on a VLAN with no WAN route
- [ ] `/var/lib/openhup` and Postgres data on LUKS
- [ ] Unique, generated camera passwords (they are per-camera for a reason)
- [ ] `bind_host: 127.0.0.1`, reverse proxy in front
- [ ] Tailscale or WireGuard for remote access, rather than a public hostname
- [ ] `chmod 600 deploy/env/openhup.env`
- [ ] Shortest snapshot retention you can live with
- [ ] `redact: [faces]` on any anchor covering a space people occupy
- [ ] `allow_remote_llm: false` unless you have decided otherwise deliberately
- [ ] Review `GET /api/v1/system/llm-usage` after your first week
- [ ] `systemd-analyze security openhup-api.service` — should score under 3.0
- [ ] Automatic OS updates, and OpenHup updates when you have time to check them

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting (Security →
Advisories → Report a vulnerability), as described in `SECURITY.md`. Include what you did, what
happened, and what you expected. We will acknowledge within a week.

This is a hobbyist-scale project with no security team and no bug bounty. What we can promise is that
reports are taken seriously, fixed in the open, and credited if you want them to be.
