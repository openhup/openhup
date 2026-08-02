#!/usr/bin/env python3
"""OpenHup camera agent.

For hosts that own a camera but cannot be reached from the vision service: a Pi Zero 2 W taped behind
a shelf, a laptop with a webcam, anything on wifi behind NAT. The agent captures frames locally and
pushes JPEGs to the vision service, inverting the connection direction.

Deliberately dependency-light — OpenCV and `requests` — so it installs on a Pi Zero in a couple of
minutes and needs no OpenHup package at all.

    pip install opencv-python-headless requests
    ./agent.py --camera-id office --device /dev/video0 \\
        --vision-url http://192.168.20.5:8090 --token "$OPENHUP_AGENT_TOKEN"

Two things it does that matter:

* **Local motion gating.** It only uploads a frame when something changed, or every `--heartbeat`
  seconds regardless. On a wifi-connected Pi this is the difference between a trickle and a constant
  stream, and the heartbeat is what stops a genuinely static scene from looking like a dead camera.
* **It never buffers.** If an upload is slow, frames are dropped rather than queued. A late frame is
  worthless for deciding whether a shelf is cluttered right now.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass

try:
    import cv2
    import numpy as np
    import requests
except ImportError:  # pragma: no cover
    sys.exit("install dependencies first: pip install opencv-python-headless requests")

log = logging.getLogger("openhup.agent")


@dataclass
class Config:
    camera_id: str
    device: str
    vision_url: str
    token: str
    width: int = 1280
    height: int = 720
    jpeg_quality: int = 80
    #: Minimum seconds between uploads, even with constant motion.
    min_interval: float = 2.0
    #: Upload at least this often regardless of motion, so the anchor never looks stale.
    heartbeat: float = 120.0
    #: Fraction of changed pixels that counts as motion.
    motion_threshold: float = 0.012
    verbose: bool = False


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {config.token}"
        self.capture: cv2.VideoCapture | None = None
        self.reference: np.ndarray | None = None
        self.last_upload = 0.0
        self.uploaded = 0
        self.skipped = 0
        self.running = True

    # -- capture ------------------------------------------------------------------------

    def open(self) -> None:
        device = int(self.config.device) if self.config.device.isdigit() else self.config.device
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open {self.config.device}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        # Depth 1: we want the newest frame, not the oldest one still sitting in a buffer.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info("capturing %s at %dx%d", self.config.device, self.config.width, self.config.height)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # -- gating -------------------------------------------------------------------------

    def motion_score(self, frame: np.ndarray) -> float:
        """Fraction of pixels that changed materially since the last *uploaded* frame.

        Comparing against the last uploaded frame rather than the previous frame means gradual
        change - a pile growing over ten minutes - accumulates past the threshold instead of sliding
        under a per-frame delta forever.
        """
        small = cv2.resize(frame, (160, 120))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if self.reference is None:
            return 1.0
        delta = np.abs(grey - self.reference)
        # 18 grey levels: above sensor noise and JPEG artefacts, below a real object appearing.
        return float((delta > 18).mean())

    def remember(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, (160, 120))
        self.reference = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)

    # -- upload -------------------------------------------------------------------------

    def upload(self, frame: np.ndarray) -> bool:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            log.warning("JPEG encode failed")
            return False
        try:
            response = self.session.post(
                f"{self.config.vision_url.rstrip('/')}/agent/frame",
                params={"camera_id": self.config.camera_id},
                data=buffer.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            log.warning("upload failed: %s", exc)
            return False
        self.uploaded += 1
        return True

    # -- loop ---------------------------------------------------------------------------

    def run(self) -> int:
        backoff = 1.0
        while self.running:
            try:
                self.open()
                backoff = 1.0
                self._capture_loop()
            except Exception as exc:  # noqa: BLE001 - an agent must survive a flaky USB port
                log.warning("%s; retrying in %.0fs", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                self.close()
        log.info("stopped: %d uploaded, %d skipped", self.uploaded, self.skipped)
        return 0

    def _capture_loop(self) -> None:
        assert self.capture is not None
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("frame read failed")

            now = time.monotonic()
            since_upload = now - self.last_upload
            if since_upload < self.config.min_interval:
                time.sleep(0.05)
                continue

            score = self.motion_score(frame)
            heartbeat_due = since_upload >= self.config.heartbeat

            if score < self.config.motion_threshold and not heartbeat_due:
                self.skipped += 1
                time.sleep(0.2)
                continue

            reason = "heartbeat" if score < self.config.motion_threshold else f"motion {score:.3f}"
            if self.upload(frame):
                self.remember(frame)
                self.last_upload = now
                log.debug("uploaded (%s)", reason)

    def stop(self, *_: object) -> None:
        self.running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenHup camera agent")
    parser.add_argument("--camera-id", required=True, help="must match a camera in cameras.yaml")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--vision-url", required=True, help="e.g. http://192.168.20.5:8090")
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENHUP_AGENT_TOKEN", ""),
        help="bearer token; prefer the OPENHUP_AGENT_TOKEN environment variable",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--min-interval", type=float, default=2.0)
    parser.add_argument("--heartbeat", type=float, default=120.0)
    parser.add_argument("--motion-threshold", type=float, default=0.012)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    if not args.token:
        log.warning("no token given; the vision service will reject uploads if it requires one")

    agent = Agent(
        Config(
            camera_id=args.camera_id,
            device=args.device,
            vision_url=args.vision_url,
            token=args.token,
            width=args.width,
            height=args.height,
            jpeg_quality=args.quality,
            min_interval=args.min_interval,
            heartbeat=args.heartbeat,
            motion_threshold=args.motion_threshold,
            verbose=args.verbose,
        )
    )
    signal.signal(signal.SIGINT, agent.stop)
    signal.signal(signal.SIGTERM, agent.stop)
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
