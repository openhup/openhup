<div align="center">

<img src="frontend/static/logo.svg" alt="OpenHup" width="240" />

# OpenHup

**A self-hosted, local-first home assistant that sees your space and helps you keep it.**

Cameras in → tasks, safety alerts, and habit metrics out, delivered by an assistant whose
personality you choose. No cloud account. No telemetry. Your frames stay on your hardware.

[Architecture](docs/ARCHITECTURE.md) · [Decisions](docs/adr/README.md) ·
[Install](docs/INSTALL.md) · [Configuration](docs/CONFIGURATION.md) ·
[Privacy](docs/SECURITY_PRIVACY.md) · [Hardware](docs/HARDWARE.md)

Apache-2.0 · Python · SvelteKit · PostgreSQL · ONNX Runtime · Ollama

</div>

> I'm homicidal, and I got a taste, I want to wipe out the monster race!

---

## What it does

- **Sees state, not just motion.** Is the counter cluttered? Is the trash full? Is the dish rack
  empty? Is the burner still on? Is the walkway blocked?
- **Turns that into tasks** that create themselves when the mess appears and **complete themselves**
  when it's gone. Every task carries a snapshot, so you know exactly what it means.
- **Raises safety alerts** for the things that matter: burner left on with nobody home, a blocked
  exit, a door left open, a person on the floor.
- **Micro-tasks overwhelming spaces.** One small step at a time, "just the left third of the
  shelf". widthh progress verified by the camera, not by your willpower.
- **Tracks habits over time.** Clean streaks, trash cycles per week, TV minutes per day, cooking
  sessions. Set a goal like "cook more" and get a short weekly report.
- **Has a personality you pick. Or gamble with it.** Kind Coach, Deadpan Butler, Chaos Goblin, Drill
  Sergeant, or the five-voice gamble (Friendly, Shy, Sassy, Sarcastic, Angry, drawn at first
  launch and never announced; you discover it by living with it). Roasting is opt-in,
  intensity-capped, and switched off entirely for when there's safety alerts.
- **Runs on the AI layer you choose.** A local model (Ollama, llama.cpp, vLLM) by default, or a
  cloud provider you trust behind an explicit egress gate. The setup wizard always asks; there is
  no "no brain" configuration. That's not how it works.

## What it is not

Not an NVR (use Frigate. OpenHup can even run _on top of_ Frigate's detections). Not a home
automation hub (it publishes to MQTT and Home Assistant). Not face recognition — except for
people who explicitly consent to being remembered, which is the only way the system ever names
anyone (ADR-016). Not a cloud service, either.

## How it works

```
cameras ──▶ vision-service ──▶ event bus ──▶ skill engine ──▶ tasks / alerts / metrics
            (detect state)     (Redis)       (rules + timing)      │
                                                                   ├─▶ notifications
                                                                   ├─▶ web UI (live)
                                                                   └─▶ LLM personality layer
```

The vision service reports only facts (`clutter_level=0.72` on `kitchen.counter`). All policy —
thresholds, how long a condition must hold, whether it's a task or an alert, how it's phrased —
lives in **Skills**. Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## A Skill, in full

```yaml
id: stove-burner-safety
enabled: true
watch:
  - anchor: kitchen.stove
signals:
  - { id: burner, detector: zero_shot_state, signal: burner_state }
  - { id: people, detector: object_inventory, signal: person_count }
conditions:
  all:
    - { signal: burner, op: eq, value: "on", for: 10m }
    - { signal: people, op: eq, value: 0, for: 5m }
effect:
  type: alert
  urgency: high # → personality bypassed, phrasing stays factual
  channels: [ntfy, mqtt]
resolve:
  conditions:
    any:
      - { signal: burner, op: eq, value: "off", for: 30s }
      - { signal: people, op: gte, value: 1 }
limits: { cooldown: 5m }
snapshot: { attach: true, retention: 30d, redact: [faces] }
```

More in [`examples/skills/`](examples/skills/): kitchen clutter buster, ADHD micro-task shelf,
TV time tracking, low on bread, pet bowl, walkway safety.

## Status

**The core works end to end.** Schemas, the skill engine, the backend API, and the vision pipeline
are implemented and tested (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/adr/README.md](docs/adr/README.md) for the design). What remains is deliberately listed below
so the README never oversells the repo.

| Component                                                  | State                                                                                                                                                                                               |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared wire schemas                                        | done, tested                                                                                                                                                                                        |
| Skill engine (window, operators, FSM, simulate, compile)   | done, tested                                                                                                                                                                                        |
| Backend API + persistence + notifications                  | done, tested                                                                                                                                                                                        |
| Vision pipeline (sources, ROI, fusion, detectors, emitter) | done, tested                                                                                                                                                                                        |
| Deploy (Compose, systemd, Caddy)                           | authored, not yet built end to end                                                                                                                                                                  |
| Frontend (SvelteKit)                                       | functional scaffold — all routes render and build; Today is the reference for tone                                                                                                                  |
| Detectors                                                  | all 10 declared detectors implemented (`object_inventory`, `clutter_score`, `zero_shot_state`, `fill_level`, `door_state`, `presence_absence`, `screen_on`, `walkway_clear`, `pose_fall`, `sensor`) |
| Docs set (INSTALL/CONFIG/DEV/SECURITY/HARDWARE/UX)         | done                                                                                                                                                                                                |

Known gaps that are tracked in code, not just here:

- Model checksums: `yolox-s` and `clip-vit-b32` are pinned and verified; `dfine-s`,
  `clip-vit-b32-int8`, `yolo-world-s`, `owlv2-base`, and `rtmpose-s` are `PENDING` because their
  upstream URLs are broken or gated (see `models/registry.yaml`). Until a working mirror is
  sourced, the opt-in `pose_fall` detector and the open-vocabulary models cannot be fetched.
- `pose_fall` is opt-in and best-effort, not a medical device, read its caveats in
  `examples/skills/more-examples.yaml` before enabling it.
- The frontend is a functional scaffold, not a finished product (see the table above).
- Deploy artefacts (Compose, systemd, Caddy) are authored but not yet built end to end.

## Requirements

- Linux host (Debian tested), Docker + Compose _or_ Python 3.12+ and systemd
- PostgreSQL 16, Redis 7
- One or more RTSP/ONVIF cameras, or a USB webcam via `camera-agents/python-agent`
- Optional: Ollama for local LLM (~8 GB VRAM or 16 GB RAM); Intel iGPU / NVIDIA GPU / Hailo for
  faster inference. CPU-only works — see [docs/HARDWARE.md](docs/HARDWARE.md) for realistic
  camera-count expectations per tier.

Model weights are not committed due to obvious reasons; `scripts/fetch_models.py` downloads and sha256-verifies them from
`models/registry.yaml`.

## Privacy in one paragraph

Frames are decoded in RAM and discarded unless a Skill explicitly attaches a snapshot. Snapshots get
a per-Skill TTL (7 days default) and can blur people before ever hitting disk. The default LLM is
local; using a remote API requires setting `allow_remote_llm: true` and choosing a redaction
profile, and every outbound call is logged with its destination and byte count. Nothing binds to a
public interface by default, and there is no telemetry of any kind.
[docs/SECURITY_PRIVACY.md](docs/SECURITY_PRIVACY.md) has the threat model.

## License

Apache-2.0. See [LICENSE](LICENSE). Contributions welcome, just look at [CONTRIBUTING.md](CONTRIBUTING.md).
