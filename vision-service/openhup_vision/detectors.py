"""Detector implementations.

Each detector answers one question about one region and returns typed signals with no policy
attached. The registry at the bottom maps the names in `openhup_schemas.BUILTIN_DETECTORS` to these
implementations, so the backend's declared contract and the code that satisfies it cannot drift
without a test failing.

Note `ScreenOn` needs no model at all - it is brightness and temporal variance in the screen region.
That is deliberate: TV-time tracking is the one habit skill people are most nervous about, and a
detector that is *architecturally incapable* of knowing what is on screen is a much stronger promise
than a policy saying we won't look.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol

import numpy as np
from openhup_schemas import BBox, Signal, SignalKind

from . import fusion
from .backends import ModelUnavailable, SessionCache
from .roi import Frame, Region, crop, to_region_coords
from .sensor_feed import SensorFeed

log = logging.getLogger(__name__)

COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]


@dataclass(frozen=True, slots=True)
class DetectorContext:
    """Everything a detector needs beyond the pixels themselves."""

    anchor_id: str
    anchor_label: str
    region: Region
    params: dict[str, Any] = field(default_factory=dict)
    #: The anchor's stored clean reference, already cropped to the region. None until captured.
    baseline: Frame | None = None
    sensitivity: float = 0.5
    clutter_weights: fusion.Weights = field(default_factory=fusion.Weights)
    subregions: tuple[Region, ...] = ()
    #: Enrolled member gallery: (id, embedding) pairs, for the consent-gated face_id detector.
    #: Names never reach the vision service - only ids and vectors (ADR-016).
    gallery: tuple[tuple[str, list[float]], ...] = ()

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)


@dataclass(frozen=True, slots=True)
class DetectorResult:
    signals: list[Signal]
    #: Extra data not part of the signal contract: subregion scores, an explanation string.
    extra: dict[str, Any] = field(default_factory=dict)
    #: Boxes to redact before any snapshot is written.
    redact_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


class Detector(Protocol):
    name: str
    #: Model ids this detector needs. Empty means it needs none.
    models: tuple[str, ...]

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult: ...


# --------------------------------------------------------------------------------------


class ScreenOn:
    """Is a screen displaying something? Brightness plus temporal variance. No model.

    Two signals because they answer different questions: `screen_on` is "is it lit", and
    `screen_activity` is "is something moving on it", which separates a playing film from a paused
    menu left up for three hours. Neither can tell you what is playing, and no detector here can.
    """

    name = "screen_on"
    models: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        patch = crop(frame, context.region)
        grey = patch.astype(np.float32).mean(axis=2)
        # Only consider the lit part of the region: a dark bezel drags a whole-region mean down and
        # makes a bright screen in a big ROI look off.
        bright = grey[grey > np.percentile(grey, 60)]
        luminance = float(bright.mean() / 255.0) if bright.size else 0.0
        spatial_variance = float(grey.std() / 128.0)

        history = self._history.setdefault(context.anchor_id, [])
        history.append(luminance)
        del history[:-10]
        temporal = float(np.std(history)) if len(history) > 2 else 0.0

        threshold = float(context.param("luminance_threshold", 0.32))
        # A lit screen is bright *and* has structure. A sunlit blank wall is bright and flat.
        is_on = luminance > threshold and spatial_variance > 0.08

        return DetectorResult(
            signals=[
                Signal(key="screen_on", kind=SignalKind.BOOLEAN, value=is_on, confidence=0.8),
                Signal(
                    key="screen_activity",
                    kind=SignalKind.SCALAR,
                    value=round(min(temporal * 6.0, 1.0), 4),
                    confidence=0.6,
                ),
            ],
            extra={
                "luminance": round(luminance, 4),
                "spatial_variance": round(spatial_variance, 4),
            },
        )


class ObjectInventory:
    """Closed-set object detection inside the region.

    Emits an inventory rather than events: the set of labels present, how many, how much area they
    cover, and the boxes. `person_count` is separated out because presence gating is what makes
    safety skills liveable, and because person boxes are what the redactor needs.
    """

    name = "object_inventory"
    models = ("yolox-s",)

    def __init__(self, sessions: SessionCache, model_id: str = "yolox-s") -> None:
        self.sessions = sessions
        self.model_id = model_id

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        patch = crop(frame, context.region)
        session = self.sessions.try_get(self.model_id)
        if session is None:
            # Degrade rather than kill the loop: an anchor with no detector reports nothing, the
            # engine sees a stale signal, and the user gets told the vision stack needs attention.
            raise ModelUnavailable(f"{self.name}: {self.model_id} not loaded")

        outputs = session.infer(patch)
        boxes = self._decode(outputs, session.spec.input_size, patch.shape[:2], context)

        min_score = float(context.param("min_score", 0.35))
        wanted = {c.lower() for c in context.param("classes", []) or []}
        kept = [
            b for b in boxes if b.score >= min_score and (not wanted or b.label.lower() in wanted)
        ]

        labels = sorted({b.label for b in kept})
        people = [b for b in kept if b.label == "person"]
        movable = [b for b in kept if b.label.replace("_", " ") not in fusion.STRUCTURAL_COCO]

        return DetectorResult(
            signals=[
                Signal(key="objects", kind=SignalKind.SET, value=labels),
                Signal(key="object_count", kind=SignalKind.COUNT, value=len(kept)),
                Signal(key="person_count", kind=SignalKind.COUNT, value=len(people)),
                Signal(
                    key="object_area_fraction",
                    kind=SignalKind.SCALAR,
                    value=round(min(sum(b.area for b in movable), 1.0), 4),
                ),
                Signal(key="boxes", kind=SignalKind.BBOX_LIST, value=kept),
            ],
            redact_boxes=[(b.x1, b.y1, b.x2, b.y2) for b in people],
        )

    def _decode(
        self,
        outputs: list[np.ndarray],
        input_size: int,
        patch_shape: tuple[int, int],
        context: DetectorContext,
    ) -> list[BBox]:
        """Decode raw model output into normalised boxes.

        YOLOX emits [1, N, 85] as (cx, cy, w, h, obj, ...classes) in input-space pixels. Other
        exports differ; `backends.ModelSpec` records the recipe and this is where a new family gets
        its branch. Deliberately tolerant: a shape we do not recognise yields no detections rather
        than an exception in the capture loop.
        """
        if not outputs:
            return []
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2 or raw.shape[1] < 6:
            log.warning("%s: unexpected output shape %s", self.name, raw.shape)
            return []

        height, width = patch_shape
        scale = min(input_size / max(height, 1), input_size / max(width, 1))
        pad_x = (input_size - round(width * scale)) // 2
        pad_y = (input_size - round(height * scale)) // 2

        objectness = raw[:, 4]
        class_scores = raw[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        confidences = objectness * class_scores[np.arange(len(class_ids)), class_ids]

        results: list[BBox] = []
        for index in np.where(confidences > 0.05)[0]:
            cx, cy, box_w, box_h = raw[index, :4]
            box = to_region_coords(
                (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2),
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                region_w=width,
                region_h=height,
            )
            label_index = int(class_ids[index])
            results.append(
                BBox(
                    label=COCO80[label_index] if label_index < len(COCO80) else str(label_index),
                    score=float(min(confidences[index], 1.0)),
                    x1=box[0],
                    y1=box[1],
                    x2=box[2],
                    y2=box[3],
                )
            )
        return _nms(results, iou_threshold=float(context.param("nms_iou", 0.45)))


class ClutterScore:
    """The fused clutter score of ADR-005.

    Runs the three components, publishes all of them, and scores each subregion separately so the
    micro-task ladder has somewhere to start. Every component is optional: no baseline means two
    components instead of three (renormalised, not zeroed), and no CLIP model means the semantic
    term drops out. A partially-equipped install still produces a usable number.
    """

    name = "clutter_score"
    models = ("clip-vit-b32", "yolox-s")

    def __init__(
        self,
        sessions: SessionCache,
        inventory: ObjectInventory | None = None,
        embed_model: str = "clip-vit-b32",
    ) -> None:
        self.sessions = sessions
        self.inventory = inventory
        self.embed_model = embed_model
        self._probe_cache: dict[str, np.ndarray] = {}

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        patch = crop(frame, context.region)

        use_baseline = context.param("reference", "baseline") == "baseline"
        baseline_diff = (
            fusion.structural_difference(patch, context.baseline)
            if use_baseline and context.baseline is not None
            else None
        )

        density = 0.0
        redact: list[tuple[float, float, float, float]] = []
        if self.inventory is not None:
            try:
                inv = self.inventory.detect(frame, context)
            except ModelUnavailable:
                log.debug("%s: object density unavailable", self.name)
            else:
                boxes = next(s.value for s in inv.signals if s.key == "boxes")
                density = fusion.object_density(
                    [b.label.replace("_", " ") for b in boxes], [b.area for b in boxes]
                )
                redact = inv.redact_boxes

        semantic = self._semantic(patch)

        components = fusion.fuse(
            baseline_diff=baseline_diff,
            object_density_score=density,
            semantic=semantic,
            weights=context.clutter_weights,
            sensitivity=float(context.param("sensitivity", context.sensitivity)),
        )

        # Subregion scores drive spatial micro-tasking: "just clear the left third".
        subregion_scores: dict[str, float] = {}
        for sub in context.subregions:
            sub_patch = crop(frame, sub)
            sub_baseline = crop(context.baseline, sub) if context.baseline is not None else None
            sub_diff = (
                fusion.structural_difference(sub_patch, sub_baseline)
                if sub_baseline is not None and sub_baseline.size
                else None
            )
            sub_components = fusion.fuse(
                baseline_diff=sub_diff,
                object_density_score=density,
                semantic=self._semantic(sub_patch),
                weights=context.clutter_weights,
                sensitivity=float(context.param("sensitivity", context.sensitivity)),
            )
            subregion_scores[sub.id] = round(sub_components.fused, 4)

        return DetectorResult(
            signals=[
                Signal(
                    key="clutter_level",
                    kind=SignalKind.SCALAR,
                    value=round(components.fused, 4),
                    confidence=0.75 if baseline_diff is not None else 0.6,
                    components=components.as_dict(),
                ),
                Signal(
                    key="baseline_diff",
                    kind=SignalKind.SCALAR,
                    value=round(components.baseline_diff, 4),
                ),
                Signal(
                    key="object_density",
                    kind=SignalKind.SCALAR,
                    value=round(components.object_density, 4),
                ),
                Signal(
                    key="semantic_clutter",
                    kind=SignalKind.SCALAR,
                    value=round(components.semantic, 4),
                ),
            ],
            extra={
                "explanation": fusion.explain(components, anchor_label=context.anchor_label),
                "subregions": fusion.subregion_scores(subregion_scores),
                "used_baseline": baseline_diff is not None,
            },
            redact_boxes=redact,
        )

    def _semantic(self, patch: Frame) -> float | None:
        """CLIP tidy-vs-cluttered probe. None when no embedding model is available."""
        session = self.sessions.try_get(self.embed_model)
        if session is None or not patch.size:
            return None
        try:
            embedding = _l2(session.infer(patch)[0].reshape(-1))
            tidy, cluttered = self._probes()
            return fusion.semantic_clutter(float(embedding @ tidy), float(embedding @ cluttered))
        except (ModelUnavailable, ValueError, IndexError) as exc:
            log.debug("semantic component unavailable: %s", exc)
            return None

    def _probes(self) -> tuple[np.ndarray, np.ndarray]:
        """Text embeddings for the two probes.

        Computed once and cached: the text encoder must never be in the per-frame path. In a build
        without the text encoder present, the cache is seeded from a shipped .npy of the two default
        probe embeddings so the semantic component still works.
        """
        if "tidy" not in self._probe_cache:
            raise ModelUnavailable("probe embeddings not initialised")
        return self._probe_cache["tidy"], self._probe_cache["cluttered"]

    def prime_probes(self, tidy: np.ndarray, cluttered: np.ndarray) -> None:
        self._probe_cache["tidy"] = _l2(tidy)
        self._probe_cache["cluttered"] = _l2(cluttered)


class ZeroShotState:
    """Classify a region into one of several user-described states via CLIP text probes.

    This is how "burner on vs off" and "door open vs closed" work without training anything. The
    signal key is chosen by the user in their skill binding, which is why the schema marks this
    detector `dynamic`.

    `min_margin` is the important parameter: when the top two states are too close, report `unknown`
    rather than guessing. `unknown` matches neither branch of a well-written skill, so ambiguity
    produces silence instead of a false burner alert.
    """

    name = "zero_shot_state"
    models = ("clip-vit-b32",)

    def __init__(self, sessions: SessionCache, embed_model: str = "clip-vit-b32") -> None:
        self.sessions = sessions
        self.embed_model = embed_model
        self._text_cache: dict[str, np.ndarray] = {}

    def cache_probe(self, text: str, embedding: np.ndarray) -> None:
        self._text_cache[text] = _l2(embedding)

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        probes: dict[str, str] = context.param("probes") or {}
        signal_key = context.param("emit_as") or context.param("signal") or "state"
        if not probes:
            raise ValueError(f"{self.name} on {context.anchor_id}: params.probes is required")

        session = self.sessions.try_get(self.embed_model)
        missing = [text for text in probes.values() if text not in self._text_cache]
        if session is None or missing:
            return DetectorResult(
                signals=[
                    Signal(key=signal_key, kind=SignalKind.ENUM, value="unknown", confidence=0.0)
                ],
                extra={"reason": "embedding model or probe cache unavailable"},
            )

        embedding = _l2(session.infer(crop(frame, context.region))[0].reshape(-1))
        scored = {
            state: float(embedding @ self._text_cache[text]) for state, text in probes.items()
        }
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)

        logits = np.array([score for _, score in ranked]) * 100.0
        logits -= logits.max()
        probabilities = np.exp(logits) / np.exp(logits).sum()
        margin = float(probabilities[0] - (probabilities[1] if len(probabilities) > 1 else 0.0))

        min_margin = float(context.param("min_margin", 0.08))
        state = ranked[0][0] if margin >= min_margin else "unknown"

        return DetectorResult(
            signals=[
                Signal(
                    key=signal_key,
                    kind=SignalKind.ENUM,
                    value=state,
                    confidence=round(float(probabilities[0]), 4),
                )
            ],
            extra={
                "scores": {k: round(v, 4) for k, v in scored.items()},
                "margin": round(margin, 4),
            },
        )


class WalkwayClear:
    """Is the traversable floor path unobstructed?

    Distinct from clutter on purpose: this cares only about objects sitting *on the floor region*,
    because a trip hazard is a different thing from an untidy surface and deserves different
    thresholds.
    """

    name = "walkway_clear"
    models = ("yolox-s",)

    def __init__(self, inventory: ObjectInventory) -> None:
        self.inventory = inventory

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        inv = self.inventory.detect(frame, context)
        boxes = next(s.value for s in inv.signals if s.key == "boxes")
        obstructions = [
            b
            for b in boxes
            if b.label != "person" and b.label.replace("_", " ") not in fusion.STRUCTURAL_COCO
        ]
        area = min(sum(b.area for b in obstructions), 1.0)
        threshold = float(context.param("max_obstruction", 0.08))
        return DetectorResult(
            signals=[
                Signal(key="walkway_clear", kind=SignalKind.BOOLEAN, value=area <= threshold),
                Signal(key="obstruction_area", kind=SignalKind.SCALAR, value=round(area, 4)),
            ],
            redact_boxes=inv.redact_boxes,
        )


class FillLevel:
    """How full is a container, 0..1, via CLIP text probes.

    Three probes (empty / half-full / overflowing) are compared against the frame embedding and
    interpolated to a scalar, so "trash full", "pet bowl low", and "dish rack full" all become one
    number the skill engine can threshold. `overflowing` is reported separately because "full" and
    "breaching the rim" genuinely differ for a bin you are about to tie up.

    No model means no guess: `ModelUnavailable` propagates so the anchor goes stale rather than
    reading as empty, which is the failure that would otherwise refill a bowl nobody asked about.
    """

    name = "fill_level"
    models = ("clip-vit-b32",)

    def __init__(self, sessions: SessionCache, embed_model: str = "clip-vit-b32") -> None:
        self.sessions = sessions
        self.embed_model = embed_model
        self._text_cache: dict[str, np.ndarray] = {}

    def cache_probe(self, text: str, embedding: np.ndarray) -> None:
        self._text_cache[text] = _l2(embedding)

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        container = context.param("container")
        if not container:
            raise ValueError(f"{self.name} on {context.anchor_id}: params.container is required")

        levels = ("empty", "half full", "overflowing")
        probes = {level: f"a {container} that is {level}" for level in levels}

        session = self.sessions.try_get(self.embed_model)
        missing = [text for text in probes.values() if text not in self._text_cache]
        if session is None or missing:
            raise ModelUnavailable(
                f"{self.name}: embedding model or probe cache unavailable for {container!r}"
            )

        embedding = _l2(session.infer(crop(frame, context.region))[0].reshape(-1))
        logits = np.array([float(embedding @ self._text_cache[text]) for text in probes.values()])
        logits = (logits - logits.max()) * 100.0
        probabilities = np.exp(logits) / np.exp(logits).sum()

        anchors = np.array([0.0, 0.5, 1.0])
        fill = float(np.dot(probabilities, anchors))
        overflowing = bool(probabilities[-1] > 0.5)

        return DetectorResult(
            signals=[
                Signal(
                    key="fill_level",
                    kind=SignalKind.SCALAR,
                    value=round(min(max(fill, 0.0), 1.0), 4),
                    confidence=round(float(probabilities.max()), 4),
                ),
                Signal(
                    key="overflowing",
                    kind=SignalKind.BOOLEAN,
                    value=overflowing,
                    confidence=round(float(probabilities[-1]), 4),
                ),
            ],
            extra={
                "scores": {
                    level: round(float(p), 4)
                    for level, p in zip(levels, probabilities, strict=True)
                },
                "container": container,
            },
        )


class DoorState:
    """Open, closed, or ajar, via CLIP probes against the door region.

    `unknown` is a real value, not an afterthought: when the probes are ambiguous the detector says
    so, and a well-written skill matches neither branch, producing silence instead of a false
    "door left open" alert.
    """

    name = "door_state"
    models = ("clip-vit-b32",)

    _LEVELS = ("open", "ajar", "closed")
    _PROBES: ClassVar[dict[str, str]] = {
        "open": "a door standing wide open",
        "ajar": "a door slightly ajar",
        "closed": "a door closed flush in its frame",
    }

    def __init__(self, sessions: SessionCache, embed_model: str = "clip-vit-b32") -> None:
        self.sessions = sessions
        self.embed_model = embed_model
        self._text_cache: dict[str, np.ndarray] = {}

    def cache_probe(self, text: str, embedding: np.ndarray) -> None:
        self._text_cache[text] = _l2(embedding)

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        session = self.sessions.try_get(self.embed_model)
        missing = [text for text in self._PROBES.values() if text not in self._text_cache]
        if session is None or missing:
            return DetectorResult(
                signals=[
                    Signal(key="door_state", kind=SignalKind.ENUM, value="unknown", confidence=0.0)
                ],
                extra={"reason": "embedding model or probe cache unavailable"},
            )

        embedding = _l2(session.infer(crop(frame, context.region))[0].reshape(-1))
        scored = {
            level: float(embedding @ self._text_cache[text]) for level, text in self._PROBES.items()
        }
        logits = np.array([scored[level] for level in self._LEVELS]) * 100.0
        logits -= logits.max()
        probabilities = np.exp(logits) / np.exp(logits).sum()

        margin = float(probabilities[0] - (probabilities[1] if len(probabilities) > 1 else 0.0))
        min_margin = float(context.param("min_margin", 0.08))
        state = self._LEVELS[int(np.argmax(probabilities))] if margin >= min_margin else "unknown"

        return DetectorResult(
            signals=[
                Signal(
                    key="door_state",
                    kind=SignalKind.ENUM,
                    value=state,
                    confidence=round(float(probabilities.max()), 4),
                )
            ],
            extra={
                "scores": {k: round(v, 4) for k, v in scored.items()},
                "margin": round(margin, 4),
            },
        )


class PresenceAbsence:
    """Is a named thing in the anchor? A binary CLIP probe pair.

    Open-vocabulary in the sense that the thing can be anything you can describe - "dish drying
    rack", "pet bowl", "bread on the shelf". The signal name is chosen by the user, so the backend
    tells this detector which key to emit via `params.emit_as`.
    """

    name = "presence_absence"
    models = ("clip-vit-b32",)

    def __init__(self, sessions: SessionCache, embed_model: str = "clip-vit-b32") -> None:
        self.sessions = sessions
        self.embed_model = embed_model
        self._text_cache: dict[str, np.ndarray] = {}

    def cache_probe(self, text: str, embedding: np.ndarray) -> None:
        self._text_cache[text] = _l2(embedding)

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        query = context.param("query")
        signal_key = context.param("emit_as") or context.param("signal") or "present"
        if not query:
            raise ValueError(f"{self.name} on {context.anchor_id}: params.query is required")

        present_text = f"{query}"
        absent_text = f"no {query}"

        session = self.sessions.try_get(self.embed_model)
        missing = [t for t in (present_text, absent_text) if t not in self._text_cache]
        if session is None or missing:
            raise ModelUnavailable(f"{self.name}: embedding model or probe cache unavailable")

        embedding = _l2(session.infer(crop(frame, context.region))[0].reshape(-1))
        present_score = float(embedding @ self._text_cache[present_text])
        absent_score = float(embedding @ self._text_cache[absent_text])

        logits = np.array([present_score, absent_score]) * 100.0
        logits -= logits.max()
        probabilities = np.exp(logits) / np.exp(logits).sum()
        prob_present = float(probabilities[0])

        min_score = float(context.param("min_score", 0.25))
        present = bool(prob_present >= max(0.5, min_score))

        return DetectorResult(
            signals=[
                Signal(
                    key=signal_key,
                    kind=SignalKind.BOOLEAN,
                    value=present,
                    confidence=round(prob_present, 4),
                )
            ],
            extra={
                "query": query,
                "present_score": round(present_score, 4),
                "absent_score": round(absent_score, 4),
            },
        )


class Sensor:
    """External sensor values, fed by MQTT and surfaced through the normal observation path.

    Deliberately not a frame detector: the value arrives over MQTT and lands in the shared
    `SensorFeed` (see `sensor_feed.py`), and `detect` merely returns the latest *changed* value as
    a signal. That keeps the skill engine's view of "lid switch" and "camera" identical - neither
    the engine nor the skill has to know which one answered.
    """

    name = "sensor"
    models: tuple[str, ...] = ()

    def __init__(self, feed: SensorFeed) -> None:
        self.feed = feed

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        # Sensor values never depend on a camera frame: they arrive over MQTT and are published
        # directly by the service's sensor loop as a `SensorReading`. Returning nothing keeps the
        # frame loop from attaching a snapshot policy to a signal that has no pixels.
        return DetectorResult(signals=[])


class PoseFall:
    """Pose-based fall detection: is a person down and not moving?

    Opt-in, and explicitly not a medical device (see the caveats in docs/CONFIGURATION.md). The
    judgement lives in pure geometry - torso angle says "down", keypoint displacement says "not
    moving" - so it is testable without a pose model; only the keypoint extraction needs one.
    """

    name = "pose_fall"
    models = ("rtmpose-s", "movenet-thunder")

    def __init__(self, sessions: SessionCache, model_id: str = "rtmpose-s") -> None:
        self.sessions = sessions
        self.model_id = model_id
        self._down_since: dict[str, datetime] = {}
        self._last_keypoints: dict[str, np.ndarray] = {}

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        session = self.sessions.try_get(self.model_id)
        if session is None:
            raise ModelUnavailable(f"{self.name}: {self.model_id} not loaded")

        patch = crop(frame, context.region)
        keypoints = _decode_keypoints(session.infer(patch))
        if keypoints is None:
            raise ModelUnavailable(f"{self.name}: unrecognised pose output shape")

        down = _is_down(keypoints)
        previous = self._last_keypoints.get(context.anchor_id)
        motion = _motion_level(keypoints, previous)
        self._last_keypoints[context.anchor_id] = keypoints
        down_seconds = self._update_down_seconds(context.anchor_id, down)

        return DetectorResult(
            signals=[
                Signal(
                    key="person_down",
                    kind=SignalKind.BOOLEAN,
                    value=down,
                    confidence=0.8 if down else 0.6,
                ),
                Signal(
                    key="person_down_seconds",
                    kind=SignalKind.SCALAR,
                    value=round(down_seconds, 1),
                ),
                Signal(
                    key="motion_level",
                    kind=SignalKind.SCALAR,
                    value=round(motion, 4),
                ),
            ],
            redact_boxes=[(0.0, 0.0, 1.0, 1.0)],  # a person is in frame; redact before any snapshot
        )

    def _update_down_seconds(self, anchor_id: str, down: bool) -> float:
        now = datetime.now(tz=UTC)
        if down:
            self._down_since.setdefault(anchor_id, now)
            return (now - self._down_since[anchor_id]).total_seconds()
        self._down_since.pop(anchor_id, None)
        return 0.0


class FaceIdentity:
    """Consent-gated identity: who is present, only ever for people who said yes (ADR-016).

    Two models: YuNet finds faces, MobileFaceNet embeds each crop, and cosine similarity against
    the enrolled gallery (from the plan) decides known vs unknown. The rules that make this the
    project's consent flow rather than its surveillance system:

    * **Names never arrive.** The gallery is (id, embedding) pairs - the vision service emits ids,
      and the backend is the only place an id becomes a name.
    * **An unknown face is reported, never stored.** `unknown_face` triggers the backend's consent
      question; the embedding that produced it is discarded at the end of this call.
    * **Identity annotates, it never accuses.** The signals here say someone is *present*; there
      is no path from them to "X left the plates".

    Model absence degrades the same way every other model-backed detector does: raise
    `ModelUnavailable`, the anchor goes stale, and /system/health tells the operator why.
    """

    name = "face_id"
    models = ("yunet-face", "mobilefacenet")

    def __init__(
        self,
        sessions: SessionCache,
        detect_model: str = "yunet-face",
        embed_model: str = "mobilefacenet",
    ) -> None:
        self.sessions = sessions
        self.detect_model = detect_model
        self.embed_model = embed_model

    def detect(self, frame: Frame, context: DetectorContext) -> DetectorResult:
        detector = self.sessions.try_get(self.detect_model)
        embedder = self.sessions.try_get(self.embed_model)
        if detector is None or embedder is None:
            raise ModelUnavailable(
                f"{self.name}: {self.detect_model} or {self.embed_model} not loaded"
            )

        patch = crop(frame, context.region)
        boxes = _decode_faces(detector.infer(patch), patch.shape[:2])
        if not boxes:
            return DetectorResult(
                signals=[
                    Signal(key="known_members", kind=SignalKind.SET, value=[]),
                    Signal(key="unknown_face", kind=SignalKind.BOOLEAN, value=False),
                    Signal(key="face_count", kind=SignalKind.COUNT, value=0),
                ]
            )

        min_score = float(context.param("min_score", 0.5))
        threshold = float(context.param("match_threshold", 0.55))
        gallery = [(mid, emb) for mid, emb in context.gallery if emb]

        known: list[str] = []
        unknown_faces = 0
        redact: list[tuple[float, float, float, float]] = []
        for x1, y1, x2, y2, score in boxes:
            if score < min_score:
                continue
            redact.append((x1, y1, x2, y2))
            crop_box = patch[
                max(int(y1 * patch.shape[0]), 0) : max(int(y2 * patch.shape[0]), 1),
                max(int(x1 * patch.shape[1]), 0) : max(int(x2 * patch.shape[1]), 1),
            ]
            if crop_box.size == 0:
                continue
            embedding = embedder.infer(crop_box)[0].reshape(-1).tolist()
            best_id = _best_match(embedding, gallery, threshold=threshold)
            if best_id is not None:
                if best_id not in known:
                    known.append(best_id)
            else:
                unknown_faces += 1

        return DetectorResult(
            signals=[
                Signal(key="known_members", kind=SignalKind.SET, value=sorted(known)),
                Signal(key="unknown_face", kind=SignalKind.BOOLEAN, value=unknown_faces > 0),
                Signal(key="face_count", kind=SignalKind.COUNT, value=len(redact)),
            ],
            redact_boxes=redact,
        )


def _decode_faces(
    outputs: list[np.ndarray], patch_shape: tuple[int, int]
) -> list[tuple[float, float, float, float, float]]:
    """YuNet-style output: [1, N, 15] of (x1, y1, x2, y2, score, landmarks...).

    Returns normalised (x1, y1, x2, y2, score) tuples. A shape we do not recognise yields no
    faces rather than an exception in the capture loop - the same tolerance as `_decode_keypoints`.
    """
    if not outputs:
        return []
    raw = outputs[0]
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 5:
        log.warning("face_id: unexpected YuNet output shape %s", raw.shape)
        return []
    height, width = patch_shape
    boxes = []
    for row in raw:
        x1, y1, x2, y2, score = (float(row[i]) for i in range(5))
        boxes.append(
            (
                min(max(x1 / width, 0.0), 1.0),
                min(max(y1 / height, 0.0), 1.0),
                min(max(x2 / width, 0.0), 1.0),
                min(max(y2 / height, 0.0), 1.0),
                score,
            )
        )
    return boxes


def _best_match(
    embedding: list[float],
    gallery: list[tuple[str, list[float]]],
    *,
    threshold: float,
) -> str | None:
    """Best gallery id above the cosine threshold, or None for an unknown face.

    Deliberately the same rule as `openhup.identity.match`: a gallery of one is not a licence to
    guess - below the threshold it is an unknown person, not a confident match.
    """
    best_id: str | None = None
    best_score = -1.0
    for member_id, enrolled in gallery:
        score = _cosine(embedding, enrolled)
        if score > best_score:
            best_score = score
            best_id = member_id
    return best_id if best_score >= threshold else None


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


#: COCO 17-keypoint layout. `_is_down` uses only the shoulders and hips, so any pose model whose
#: export follows this layout (or is remapped to it) works.
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _decode_keypoints(outputs: list[np.ndarray]) -> np.ndarray | None:
    """Extract normalised (x, y) keypoints from a pose model's raw output.

    Two families are handled, matched on shape rather than name:

    * MoveNet-style ``[1, 1, K, 3]`` - keypoints as (y, x, confidence) in 0..1.
    * Heatmap-style ``[1, K, H, W]`` - argmax per keypoint channel.

    Anything else returns None so the caller degrades to `ModelUnavailable` rather than guessing.
    RTMPose's SimCC output is intentionally *not* decoded here - it needs a dedicated recipe that
    will be added with the model once its upstream URL is restored (see models/registry.yaml).
    """
    if not outputs:
        return None
    raw = outputs[0]

    if raw.ndim == 4 and raw.shape[1] == 1 and raw.shape[3] == 3:
        keypoints = raw[0, 0]  # Kx3 (y, x, confidence)
        return np.stack([keypoints[:, 1], keypoints[:, 0]], axis=1)  # → Kx2 (x, y)

    if raw.ndim == 4 and raw.shape[1] > 3:
        heatmaps = raw[0]  # KxHxW
        coords = []
        for channel in heatmaps:
            y, x = np.unravel_index(int(np.argmax(channel)), channel.shape)
            coords.append((x / channel.shape[1], y / channel.shape[0]))
        return np.asarray(coords, dtype=np.float32)

    return None


def _is_down(keypoints: np.ndarray) -> bool:
    """Is the torso more horizontal than vertical? A standing person's hip-shoulder axis is
    vertical; a prone or supine person's is horizontal.

    ``keypoints`` is a Kx2 array of normalised (x, y) coordinates in COCO layout.
    """
    if keypoints.ndim != 2 or keypoints.shape[0] <= 12:
        return False
    shoulder = (keypoints[5] + keypoints[6]) / 2
    hip = (keypoints[11] + keypoints[12]) / 2
    return float(abs(hip[0] - shoulder[0])) > float(abs(hip[1] - shoulder[1]))


def _motion_level(current: np.ndarray, previous: np.ndarray | None) -> float:
    """Mean keypoint displacement since the last frame, clipped to 0..1.

    The concerning case for a fallen person is *near zero* movement, so this is what a skill
    thresholds on the low side.
    """
    if previous is None:
        return 0.0
    if previous.shape != current.shape:
        return 1.0
    return float(min(float(np.mean(np.abs(current - previous))), 1.0))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _nms(boxes: list[BBox], *, iou_threshold: float = 0.45) -> list[BBox]:
    """Greedy per-class non-maximum suppression."""
    kept: list[BBox] = []
    for candidate in sorted(boxes, key=lambda b: b.score, reverse=True):
        if all(
            existing.label != candidate.label or _iou(existing, candidate) < iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def _iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - overlap
    return overlap / union if union > 0 else 0.0


def build_registry(
    sessions: SessionCache, *, sensor_feed: SensorFeed | None = None
) -> dict[str, Detector]:
    """Wire up the detectors this build can actually run.

    Every detector in `openhup_schemas.BUILTIN_DETECTORS` is present, so `NOT_YET_IMPLEMENTED` is
    empty. `sensor` and `pose_fall` degrade gracefully when their inputs are unavailable - sensor
    values simply have not arrived yet, and a missing pose model raises `ModelUnavailable` rather
    than emitting confident nonsense.
    """
    inventory = ObjectInventory(sessions)
    clutter = ClutterScore(sessions, inventory=inventory)
    feed = sensor_feed or SensorFeed()
    return {
        "object_inventory": inventory,
        "clutter_score": clutter,
        "zero_shot_state": ZeroShotState(sessions),
        "fill_level": FillLevel(sessions),
        "door_state": DoorState(sessions),
        "presence_absence": PresenceAbsence(sessions),
        "screen_on": ScreenOn(),
        "walkway_clear": WalkwayClear(inventory),
        "sensor": Sensor(feed),
        "pose_fall": PoseFall(sessions),
        "face_id": FaceIdentity(sessions),
    }


#: Declared in the shared schema but not yet implemented here. Empty: every built-in detector now
#: has an implementation, even when its inputs (sensor values, pose weights) are not present.
NOT_YET_IMPLEMENTED: tuple[str, ...] = ()


__all__ = [
    "COCO80",
    "COCO_KEYPOINTS",
    "NOT_YET_IMPLEMENTED",
    "ClutterScore",
    "Detector",
    "DetectorContext",
    "DetectorResult",
    "DoorState",
    "FaceIdentity",
    "FillLevel",
    "ObjectInventory",
    "PoseFall",
    "PresenceAbsence",
    "ScreenOn",
    "Sensor",
    "WalkwayClear",
    "ZeroShotState",
    "_best_match",
    "_cosine",
    "_decode_faces",
    "_decode_keypoints",
    "_is_down",
    "_motion_level",
    "build_registry",
]
