# Hardware

Honest numbers, and the buying advice nobody gives you until after you have bought the wrong thing.

## The one-line answer

**A used Intel N100 mini PC (~£140) with two or three PoE cameras on a VLAN.** It handles 4–6 cameras
at ~35 W with OpenVINO on the iGPU, runs Postgres and Redis alongside, and has enough left over to run
a 7B LLM slowly if you want one. Nothing else in this document beats it on value.

---

## Compute tiers

Measured as total detector invocations per second across all anchors, `clutter_score` at 640×360.

| Tier | ~inf/s | Comfortable cameras | Power | Notes |
|---|---|---|---|---|
| Raspberry Pi 5, CPU only | 1–2 | 1–2 anchors, long intervals | 6 W | Works. Tight. Skip the LLM. |
| Pi 5 + Hailo-8L (26 TOPS) | 15–25 | 4–6 | 9 W | Best efficiency available. Needs ONNX→HEF conversion. |
| **Intel N100 mini PC** | **10–20** | **4–6** | **~35 W** | **The recommendation.** OpenVINO on the iGPU. |
| Ryzen 5600 / i5-10400, CPU | 6–10 | 3–4 | 65 W | Fine if you already own it. |
| Used RTX 3060 12 GB | 60+ | more than you own | 170 W | Also runs a 7B LLM comfortably. |
| Jetson Orin Nano 8 GB | 30–40 | 6–8 | 15 W | Elegant, expensive, awkward software stack. |

**These numbers are less important than they look.** With 30-second default intervals and motion
gating suppressing most frames on an idle scene, a four-camera install averages **well under one
inference per second**. The models are small. Decoding is the real cost, which is why OpenHup pulls the
camera's substream and why hardware decode matters more than raw inference throughput.

### Accelerators, honestly

| Device | Verdict |
|---|---|
| **Intel iGPU (UHD 730/770)** | Free with the CPU, works via OpenVINO, does hardware decode too. Start here. |
| **Hailo-8L** on Pi 5 | Genuinely excellent perf/watt. Costs a model-conversion step per model. |
| **Coral USB TPU** | Skip it. int8-only, 8-bit quantisation hurts CLIP badly, and the software stack has stagnated. Fine for Frigate's small detectors, poor for embedding models. |
| **Used NVIDIA (3060 12 GB)** | The pragmatic choice if you also want a local LLM. Idle power is the cost. |
| **Apple Silicon** | Works well for development. macOS is not a supported deployment target. |

---

## Cameras

### What to look for

1. **RTSP with a low-resolution substream.** Non-negotiable. Without a substream you decode the main
   stream to find a coffee mug and your CPU graph becomes a flat line at the top.
2. **ONVIF** for discovery and consistent snapshot URLs.
3. **PoE.** One cable. No batteries, no wifi congestion, no dead camera in February.
4. **Local recording and local RTSP with no cloud dependency.** If it needs an app and an account to
   view a stream, it will eventually need them to keep working.
5. **A wide lens and a mount that sees surfaces at an angle.** More on this below — it matters more
   than the sensor.

### What to avoid

- **Cloud-only cameras** (most Ring, Nest, Arlo). No local RTSP, no OpenHup.
- **Battery cameras.** They sleep to save power, which is precisely wrong for continuous state
  monitoring.
- **Tapo and Kasa.** RTSP works but has been removed and restored across firmware versions. Workable,
  not dependable.
- **4K as a goal.** You are detecting "is there a pile here", not reading a licence plate. 1080p main
  and 640×360 substream is ideal; higher resolution costs decode and buys nothing.

### Known-good, roughly by price

| Camera | ~Price | Notes |
|---|---|---|
| Reolink RLC-520A / 510A | £45 | Reliable RTSP + substream, PoE. The default suggestion. |
| Amcrest IP4M / Dahua OEM | £60 | Excellent RTSP, good low light, fiddly web UI. |
| Hikvision OEM (Annke, Sv3C) | £50 | Good hardware, check firmware locking before you buy. |
| Reolink Duo 2 PoE | £90 | 180° dual lens. One camera covering a whole kitchen. |
| USB webcam + Pi Zero 2 W | £35 | Via `camera-agents/python-agent`. Great for one shelf or desk. |
| ESP32-CAM | £8 | Only for binary states ("is the door open"). Useless for clutter scoring. |

### Placement — the part that actually determines whether this works

More installs fail on mounting than on hardware.

- **Look down at surfaces, at roughly 30–45°.** A camera level with a counter sees the front edge of a
  pile and nothing behind it. From above at an angle, the whole surface is visible.
- **Avoid windows in frame.** Backlighting wrecks exposure, and moving sunlight is the single largest
  source of false clutter readings. If unavoidable, keep the window outside the anchor polygon.
- **Mount high and out of the way.** Reduces occlusion by people and stops the camera being knocked.
- **One camera can cover several anchors.** A wide kitchen shot serving counter, sink, stove, and bin as
  four separate anchors is the normal arrangement and the reason anchors exist as their own concept.
- **Mind the privacy geometry.** Point at the surfaces you care about, not at where people sit. This
  costs nothing and makes the system much easier to live with — for you and for anyone else in the
  house.
- **Lighting beats sensor.** A £20 under-cabinet LED strip improves detection more than a £100 camera
  upgrade. Consistent light also makes the baseline comparison far more stable.

---

## Sensors instead of cameras

Often the better answer. A lid contact switch answers "is the bin open" more reliably, more cheaply,
and with far fewer privacy questions than any camera. OpenHup's `sensor` detector accepts values from
MQTT, Zigbee2MQTT, or Home Assistant and feeds them into the same pipeline, so skills do not care which
answered.

| Instead of a camera watching | Consider |
|---|---|
| a bin lid | Aqara door/window contact (~£10) |
| a stove | Zigbee power monitor, or a temperature probe |
| a door | contact sensor |
| room occupancy | mmWave presence sensor (LD2410, ~£8) — far better than PIR |
| a washing machine | vibration sensor or power monitor |

Cameras earn their place where the question is genuinely visual: *how much stuff* is on a surface,
*which* objects, *is the path clear*. For binary state, buy the switch.

---

## Storage

| Install | Disk |
|---|---|
| 2 cameras, 7-day snapshots | ~5 GB |
| 6 cameras, 30-day archive mode | ~50 GB |
| Observations (Postgres), 6 cameras, 14 days | ~2 GB |

Modest, because OpenHup is not an NVR — it stores decisions and the occasional still, not video. Put
`/var/lib/openhup` on an SSD (snapshot writes are small and frequent) and on **LUKS** (it holds imagery
of your home).

---

## Two complete example builds

**Minimal (~£190)** — Raspberry Pi 5 8 GB (£70), NVMe hat + 256 GB (£45), Reolink 510A (£45), PoE
injector (£15). One camera, two or three anchors on the kitchen, and Ollama running a 7B model
on the Pi's CPU (slow but usable — the AI layer is core, so the setup wizard always wires a
provider). Works well; it is a real install, not a demo.

**Recommended (~£340)** — Used N100 mini PC 16 GB / 512 GB (£140), 2× Reolink 520A (£90), TP-Link
5-port PoE switch (£35), lighting and cabling (£30), spare drive for backups (£45). Four to six
anchors, OpenVINO on the iGPU, and a 7B LLM running slowly but usably for skill parsing.

Add a used RTX 3060 to the second build if you want fast local LLM responses and to stop thinking about
inference budgets. It is not necessary.
