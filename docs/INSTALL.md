# Installing OpenHup

Two supported paths. Docker Compose is the fast one; systemd is leaner and gives you `journalctl`
and systemd's sandboxing. Both are first-class.

Before anything else: **put your cameras on a VLAN with no route to the internet.** Cheap camera
firmware is the least trustworthy software on a home network, and OpenHup only needs to reach the
cameras, never the other way round.

---

## Quickest possible start (15 minutes, one camera)

```sh
git clone https://github.com/openhup/openhup && cd openhup
cd backend && uv sync && cd ..
make setup
```

That's it — `openhup setup` is the whole first run in one command. It:

1. **Bootstraps `config/`** from the shipped examples (`config.yaml`, `vision.yaml`, `cameras.yaml`,
   `personalities.yaml`). Existing files are never touched — your config wins.
2. **Generates `deploy/env/openhup.env`** with real random secrets (Postgres password, vision
   token, ntfy topic), so `docker compose up` works on the first try.
3. **Asks what only you know**: this machine's inference hardware, the instance name, the
   assistant's voice (pick one of the five, or gamble — a gambled voice is drawn at first launch
   and never announced), **where speech runs** (in the browser by default; any OpenAI-compatible
   cloud for STT/TTS — a first-class choice, so a modest machine is not a reason to skip voice; or
   a local whisper.cpp/Piper server), and the AI provider (local Ollama by default, or a trusted
   cloud provider — OpenAI-compatible or Anthropic — behind an explicit egress confirmation).
   Both cloud choices accept a **custom base URL**, so you are not limited to OpenAI's own
   endpoints, and both demand the same typed "yes" egress confirmation, with keys going into the
   env file, never into `config.yaml`.
4. **Hands you the exact commands for the rest, one at a time**, waiting for you to run each in a
   second terminal and press Enter: `docker compose up`, pulling the model into Ollama, fetching
   the vision model weights, and opening the app. Where a command may need `sudo` (the docker
   group), it says so in plain text.

Then edit `config/cameras.yaml` with your camera's RTSP URLs and open <http://127.0.0.1:8080>.

`openhup setup` writes `personality.default_personality` if you picked a voice, or
`personality.gamble: true` if you gambled — the draw happens on first launch, and from then on the
voice is never shown anywhere. Re-run it any time; it merges over the existing config and never
overwrites existing files. See [ADR-014](adr/README.md).

Open <http://127.0.0.1:8080>. Then, in order:

1. **Check the camera connects.** Cameras → your camera → the snapshot should appear. If it does not,
   see the RTSP troubleshooting below before going further.
2. **Draw an anchor.** Cameras → Add region. Draw a tight polygon around one surface — a countertop,
   not the whole kitchen. Tight regions mean less CPU, fewer false positives, and much less imagery
   on disk.
3. **Capture a baseline while it is tidy.** Anchors → Capture baseline. Every clutter score is
   measured against this photograph, so capturing it mid-mess makes the anchor permanently and
   confusingly relaxed.
4. **Add one skill.** Copy `examples/skills/kitchen-clutter-buster.yaml` and change the anchor id.
5. **Simulate before enabling it.** Skills → Simulate. Let observations accumulate for a few hours
   first, then dry-run it against that history. You will see "would have fired 3x (0.4/day)" or
   "would have fired 47x", and the second answer saves you a bad week.
6. **Then enable it.**

Steps 3 and 5 are the two people skip, and they are the two that decide whether this feels useful or
irritating.

---

## Hardware, briefly

Full detail in [HARDWARE.md](HARDWARE.md). The short version:

| You have | Expect |
|---|---|
| Raspberry Pi 5, CPU only | 1–2 anchors, long intervals. Works, but tight. |
| Pi 5 + Hailo-8L | 4–6 cameras comfortably. |
| **Intel N100 mini PC** | **4–6 cameras. The sweet spot: ~35 W, ~£140 used.** |
| Any x86 box + used RTX 3060 | More cameras than you own. Also runs a 7B LLM well. |

With 30-second intervals and motion gating, a four-camera install averages well under one inference
per second. The models are small; it is the decoding that costs, which is why OpenHup uses the
camera's substream.

---

## Docker Compose

Pick exactly one inference profile — they install conflicting `onnxruntime` builds.

```sh
docker compose --profile cpu up -d          # anywhere
docker compose --profile openvino up -d     # Intel iGPU: N100, NUC
docker compose --profile cuda up -d         # NVIDIA
```

The AI layer is core, so the `ollama` profile is the normal setup, not an optional extra — and
`openhup setup` wires your provider of choice and prints the exact `up` command for the profile
you picked. Optional profile: `proxy` (Caddy with automatic TLS).

The wizard also writes `RENDER_GID` for you when you pick the OpenVINO profile. If you set up by
hand, add it so the container can reach the iGPU:

```sh
echo "RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)" >> deploy/env/openhup.env
```

Without it, OpenVINO silently falls back to CPU and you spend an afternoon wondering why the iGPU is
idle.

### Coexisting with Frigate

Already running Frigate? Do not decode every stream twice. Set the camera's `kind: frigate` and
OpenHup consumes Frigate's MQTT detection events instead of running its own object detection:

```yaml
cameras:
  - id: garage
    name: Garage
    kind: frigate
    frigate_camera: garage
```

OpenHup then contributes the skill engine, tasks, metrics, and personality on top of detections
Frigate already computes, at a fraction of the CPU. Note that Frigate answers "what objects are
there", not "is this surface cluttered" — clutter skills still need a native source for that anchor.

---

## Bare metal with systemd

See [deploy/systemd/README.md](../deploy/systemd/README.md) for the full walkthrough. Summary:

```sh
sudo useradd --system --home /var/lib/openhup --shell /usr/sbin/nologin openhup
sudo git clone https://github.com/openhup/openhup /opt/openhup
cd /opt/openhup
sudo -u openhup sh -c 'cd backend && uv sync --frozen'
sudo -u openhup sh -c 'cd vision-service && uv sync --frozen --extra openvino'
sudo cp deploy/systemd/openhup-*.service /etc/systemd/system/
sudo systemctl enable --now openhup-api openhup-engine openhup-vision
```

The units deny outbound network access by default and allow only loopback and RFC1918 ranges. That is
what makes "nothing leaves your network" enforced by the init system rather than merely intended. If
you use a remote LLM or a cloud notification service, you will need to open that deliberately — see
the note in the unit files.

---

## Remote access

Ranked by how much you should trust them:

1. **Tailscale or WireGuard.** Nothing is published to the internet at all. `tailscale serve --bg
   8080` and you are done. This is the recommended option and it is not close.
2. **Caddy on your LAN, no public DNS.** TLS inside the house, nothing reachable from outside.
3. **Public hostname with an identity-aware proxy** (Authelia, Authentik, tinyauth) in front. Only if
   you have a specific reason.
4. **Public hostname with basic auth.** One shared password guarding live imagery of your home. Try
   the other three first.

The `proxy` profile ships a Caddyfile with all four sketched out and the LAN-only option active by
default.

---

## Troubleshooting the things that actually go wrong

**The camera snapshot is black or times out.** Test the URL outside OpenHup first:

```sh
ffprobe -rtsp_transport tcp 'rtsp://user:pass@192.168.20.11:554/stream1'
```

Common causes: the wrong substream path (check your camera's web UI), UDP transport on wifi (use
TCP), or a password with characters that need URL-encoding. OpenHup takes credentials from the
environment rather than the URL specifically to avoid the last one.

**Everything runs but no tasks ever appear.** In order of likelihood:

- The skill is not enabled. `GET /api/v1/skills` shows `enabled`.
- No baseline captured. `GET /api/v1/system/health` says so explicitly.
- The threshold is wrong for your space. Simulate it — that is what the endpoint is for.
- The `for:` duration has not elapsed yet. `for: 15m` means fifteen minutes of *continuous*
  clutter.

**Tasks appear and vanish repeatedly.** Trigger and resolve thresholds are too close. OpenHup refuses
to save a skill whose ranges actually overlap, but 0.60/0.55 is legal and will still flap on a jittery
signal. Widen the gap: trigger 0.6, resolve 0.25.

**`ModelUnavailable: no sha256 recorded`.** The registry ships with placeholder checksums, and
OpenHup will not load unverified model weights. Run the fetch with `--trust-first-use` once; it pins
what it downloaded into `models/registry.lock.yaml` and verifies from then on.

**High CPU with nothing happening.** Check the motion gate is working:
`GET /api/v1/system/info` → the vision node reports `efficiency`, which should be above 0.9 on an
idle scene. If it is low, raise `sampling.motion_threshold` — a camera with noisy gain sees motion in
its own sensor noise.

**It works but feels like nagging.** That is a configuration problem with a specific fix, not
something to endure: see the anti-nag section of [CONFIGURATION.md](CONFIGURATION.md), and watch
`nag_index` in the weekly report.

---

## Upgrading

```sh
cd openhup && git pull
cd deploy/compose && docker compose build && docker compose up -d
```

Migrations run automatically (the `migrate` service, or the API unit's `ExecStartPre`). Your skills,
anchors, and history are in Postgres and are not touched. Shipped personality presets are replaced on
upgrade; personalities you created are never modified.

## Backing up

Three things matter, in this order:

```sh
# 1. The database: skills, anchors, tasks, episodes, metrics. This is the irreplaceable part.
docker compose exec postgres pg_dump -U openhup openhup | zstd > openhup-$(date +%F).sql.zst

# 2. Config, including your camera layout and anchor polygons.
tar czf openhup-config-$(date +%F).tgz config/ deploy/env/openhup.env

# 3. Snapshots, if you want the before/after history. Large, and entirely optional.
```

Snapshots are deliberately expendable — they expire on a schedule anyway. Losing them costs you
progress photos, not function.
