# openhup-vision

Decodes frames, runs detectors, emits observations. It holds **no policy**: no thresholds, no
"this is bad", no tasks. It reports `clutter_level=0.72` and the skill engine decides what that
means (ADR-003).

```
openhup_vision/
├── config.py      vision.yaml + camera/anchor config, execution-provider selection
├── roi.py         polygon → mask, crops, subregion scoring       (numpy only, pure)
├── sampler.py     adaptive cadence + motion gate                 (numpy only, pure)
├── fusion.py      the three-way clutter fusion of ADR-005        (numpy only, pure)
├── detectors.py   detector implementations + registry
├── backends.py    ONNX Runtime session management, EP selection, model registry
├── sources.py     RTSP (PyAV), USB, agent-push, snapshot-URL, Frigate-MQTT
├── emitter.py     observation assembly, redaction, snapshot writing, bus publish
└── main.py        the loop: plan pull → sample → detect → emit
```

The pure modules (`roi`, `sampler`, `fusion`) depend on numpy and nothing else, so the interesting
maths is unit-tested without ONNX Runtime, a GPU, or a camera. `tests/test_vision.py` runs on any
machine.

## Install

Choose one execution-provider extra; they install conflicting `onnxruntime` distributions.

```sh
uv sync --extra cpu        # works anywhere
uv sync --extra openvino   # Intel iGPU (N100, NUC) - the sweet spot for a home box
uv sync --extra cuda       # NVIDIA
uv run python -m openhup_vision.backends --fetch   # download + verify model weights
uv run openhup-vision --config /etc/openhup/vision.yaml
```

Weights are never committed. `models/registry.yaml` at the repo root records each model's URL,
sha256, and licence, and the fetch step verifies the hash before use.

## Why it is built this way

**Substream for detection, main stream for snapshots.** Decoding 4K to find a coffee mug is the
usual reason a home server catches fire. The sampler pulls 640×360.

**Frames are dropped, never queued.** A late frame is worthless and a growing queue is an outage.
Queue depth and drop rate are exported.

**The plan comes from the backend.** `GET /api/v1/vision/plan` returns per-anchor detector
schedules derived from *currently enabled skills*. Disable every skill on an anchor and its
detectors stop consuming CPU. Nothing here decides what is worth looking at.

**Motion gating first.** On an idle kitchen, frame-differencing inside the ROI suppresses ~95% of
detector invocations. This is the difference between 8 W and 40 W of continuous CPU.

## Realistic throughput

Measured as total detector invocations per second across all anchors, `clutter_score` at 640×360:

| Hardware | ~inferences/s | Comfortable camera count |
|---|---|---|
| Raspberry Pi 5 (CPU) | 1–2 | 1–2 anchors, long intervals |
| Pi 5 + Hailo-8L | 15–25 | 4–6 cameras |
| Intel N100 (OpenVINO iGPU) | 10–20 | 4–6 cameras |
| Ryzen 5600 (CPU) | 6–10 | 3–4 cameras |
| RTX 3060 (CUDA) | 60+ | more than you have |

With 30-second default intervals and motion gating, a 4-camera install needs well under 1
inference/second on average. See `docs/HARDWARE.md`.
