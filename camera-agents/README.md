# Camera agents

For cameras the vision service cannot reach directly.

The normal arrangement is the vision service pulling RTSP from IP cameras. An agent inverts that: it
runs on the host that owns the camera and pushes frames out. Use one when the camera is a USB or CSI
device, or when the host is on wifi behind NAT.

## python-agent

Two dependencies (OpenCV headless and `requests`), no OpenHup package needed, installs on a Pi Zero 2 W
in a couple of minutes.

```sh
pip install opencv-python-headless requests

export OPENHUP_AGENT_TOKEN=...        # from security.service_tokens in config.yaml
./python-agent/agent.py \
    --camera-id office \
    --device /dev/video0 \
    --vision-url http://192.168.20.5:8090
```

The matching camera entry:

```yaml
cameras:
  - id: office
    name: Office
    kind: agent_push
    agent_id: office-pi
    max_fps: 2
```

### What it does locally

**Motion gating.** Only uploads when something changed — comparing against the last *uploaded* frame,
not the previous frame, so a pile growing over ten minutes still accumulates past the threshold rather
than sliding under a per-frame delta forever. On a wifi Pi this is the difference between a trickle and
a constant stream.

**A heartbeat.** It uploads every `--heartbeat` seconds (120 by default) regardless of motion. Without
that, a genuinely tidy static scene would produce no observations, every skill on the anchor would go
STALE, and OpenHup would report a dead camera. A tidy room and a broken camera must not look the same.

**No buffering.** If an upload is slow, frames are dropped. A late frame is worthless for deciding
whether a shelf is cluttered *now*.

### Run it as a service

```ini
# /etc/systemd/system/openhup-agent.service
[Unit]
Description=OpenHup camera agent
After=network-online.target

[Service]
Type=exec
User=openhup
SupplementaryGroups=video
Environment=OPENHUP_AGENT_TOKEN=...
ExecStart=/usr/bin/python3 /opt/openhup-agent/agent.py \
    --camera-id office --device /dev/video0 --vision-url http://192.168.20.5:8090
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Tuning

| Flag | Try this |
|---|---|
| `--min-interval` | 2s for a desk, 10s for a shelf nobody touches |
| `--heartbeat` | 120s. Must be shorter than the skill's `staleness_timeout`. |
| `--motion-threshold` | raise it if the camera has noisy gain and sees motion in its own sensor noise |
| `--width/--height` | 1280×720 is plenty. Higher costs bandwidth and buys nothing. |
| `--quality` | 80. Below 60 starts to affect the clutter baseline comparison. |

## ESP32-CAM and other still-image devices

No agent needed — point OpenHup at the HTTP snapshot endpoint:

```yaml
cameras:
  - id: hall
    name: Hallway
    kind: snapshot_url
    url: http://192.168.20.31/capture
    max_fps: 0.5
```

Be realistic about what an £8 camera can do. Frame quality and rate are low enough that clutter
scoring is unreliable, but for a binary question — is the door open, is the bin lid up — it works
fine. For those questions a £10 Zigbee contact sensor is usually better still: see the sensor section
of [docs/HARDWARE.md](../docs/HARDWARE.md).

## Choosing an approach

| Situation | Use |
|---|---|
| IP camera with RTSP | nothing — the vision service pulls it directly |
| USB or CSI camera on the vision host | `kind: usb`, no agent |
| USB camera on a different host | `python-agent` |
| Host on wifi behind NAT | `python-agent` |
| Very cheap still-image camera | `kind: snapshot_url` |
| A genuinely binary question | a Zigbee sensor and the `sensor` detector |
| Frigate already running | `kind: frigate` — do not decode the same stream twice |
