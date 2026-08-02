"""Clutter scoring: fusing three weak signals into one usable number. Pure numpy.

There is no off-the-shelf "is this surface messy" model, and each obvious approach fails on its own
(ADR-005):

* **Object detection alone** misses everything COCO never heard of - post, wrappers, cables, craft
  supplies, the actual contents of a real kitchen counter.
* **Pixel differencing alone** breaks the first time the sun moves, someone turns a lamp on, or a
  chair shifts three centimetres.
* **CLIP alone** is uncalibrated per room. "A cluttered counter" scores differently in a bright
  minimalist kitchen than in a warm cluttered one, and neither user wants to think about that.

So all three run, each is normalised into 0..1, and they are combined with per-anchor weights. Every
component is published alongside the fused score, which is what makes the number explainable in the
UI and therefore calibratable by someone who does not want to read this file.

`sensitivity` then remaps the curve. It exists because "raise the threshold" is the wrong advice to
give a user: thresholds live in skills that may be shared, while sensitivity is a property of the
place. A gain knob on the anchor is the right shape for "it keeps nagging me about one mug".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roi import Frame

#: Movable things. Structural objects (sofa, refrigerator, sink, oven, tv) are excluded because they
#: are supposed to be there, and counting them would make every kitchen permanently cluttered.
MOVABLE_COCO = frozenset(
    {
        "bottle",
        "wine glass",
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
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "book",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "sports ball",
        "bottle opener",
        "remote",
        "cell phone",
        "laptop",
        "mouse",
        "keyboard",
        "vase",
        "clock",
        "potted plant",
    }
)

STRUCTURAL_COCO = frozenset(
    {
        "sofa",
        "couch",
        "chair",
        "dining table",
        "bed",
        "toilet",
        "tv",
        "refrigerator",
        "oven",
        "microwave",
        "sink",
        "bench",
        "door",
    }
)

#: Text probes for the zero-shot component. Pairs, not single prompts: CLIP similarity is only
#: meaningful relative to an alternative.
CLUTTER_PROBES: tuple[str, str] = (
    "a tidy, clear, empty surface in a home",
    "a cluttered surface covered in scattered objects, dishes and packaging",
)


@dataclass(frozen=True, slots=True)
class ClutterComponents:
    """The three inputs and the result. Rides along on every observation.

    The UI shows these as a small bar chart under the score. When someone says "why does it think my
    counter is messy", this is the answer, and it is usually enough for them to fix the weights
    themselves.
    """

    baseline_diff: float
    object_density: float
    semantic: float
    fused: float
    #: How much of the score came from each component after weighting. Sums to `fused`.
    contributions: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "baseline_diff": round(self.baseline_diff, 4),
            "object_density": round(self.object_density, 4),
            "semantic_clutter": round(self.semantic, 4),
        }

    def dominant(self) -> str:
        """Which component drove the score. Used in the explanation text."""
        names = ("baseline_diff", "object_density", "semantic_clutter")
        return names[int(np.argmax(self.contributions))]


@dataclass(frozen=True, slots=True)
class Weights:
    baseline_diff: float = 0.4
    object_density: float = 0.3
    semantic: float = 0.3

    def normalised(self) -> Weights:
        total = self.baseline_diff + self.object_density + self.semantic
        if total <= 0:
            return Weights(1 / 3, 1 / 3, 1 / 3)
        return Weights(
            self.baseline_diff / total, self.object_density / total, self.semantic / total
        )


# --------------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------------


def structural_difference(current: Frame, baseline: Frame, *, blocks: int = 8) -> float:
    """Block-mean absolute difference between a frame and the anchor's clean reference.

    Block means rather than per-pixel: this is meant to notice "there is now a pile here", not
    "the shadows moved four pixels left". Blocks also make it cheap and resolution-independent.

    Per-block contrast normalisation absorbs uniform lighting changes - turning on a lamp raises
    every block equally, and subtracting the median shift discards that. It does not absorb a
    directional change like late-afternoon sun across half a counter, which is why this is one
    component of three rather than the whole score.
    """
    if current.shape != baseline.shape:
        current = _fit_to(current, baseline.shape)

    now = _block_means(current, blocks)
    ref = _block_means(baseline, blocks)
    delta = np.abs(now - ref)

    # Remove the global brightness shift: whatever happened to every block equally is lighting.
    delta = np.clip(delta - np.median(delta), 0, None)
    # 60 grey levels of block-mean change is a substantial visual difference.
    return float(np.clip(delta.mean() / 60.0, 0.0, 1.0))


def _block_means(frame: Frame, blocks: int) -> np.ndarray:
    grey = frame.astype(np.float32).mean(axis=2)
    height, width = grey.shape
    rows = np.array_split(np.arange(height), min(blocks, height))
    cols = np.array_split(np.arange(width), min(blocks, width))
    return np.array([[grey[np.ix_(r, c)].mean() for c in cols] for r in rows], dtype=np.float32)


def _fit_to(frame: Frame, shape: tuple[int, ...]) -> Frame:
    target_h, target_w = shape[0], shape[1]
    height, width = frame.shape[:2]
    rows = (np.linspace(0, height - 1, target_h)).astype(np.int32)
    cols = (np.linspace(0, width - 1, target_w)).astype(np.int32)
    return frame[rows][:, cols]


def object_density(
    labels: list[str],
    areas: list[float],
    *,
    saturation_count: int = 10,
    saturation_area: float = 0.35,
) -> float:
    """Density of *movable* objects in the region: how many, and how much space they take.

    Count and area are combined with a max rather than an average, because both failure modes are
    real clutter: twelve small items is a mess, and one enormous pile is also a mess.
    """
    movable = [
        (label, area)
        for label, area in zip(labels, areas, strict=False)
        if label.lower() not in STRUCTURAL_COCO
    ]
    if not movable:
        return 0.0

    count_score = min(len(movable) / saturation_count, 1.0)
    area_score = min(sum(area for _, area in movable) / saturation_area, 1.0)
    return float(max(count_score, area_score))


def semantic_clutter(tidy_similarity: float, cluttered_similarity: float) -> float:
    """Softmax over the two CLIP probe similarities, returning P(cluttered).

    Temperature 100 matches CLIP's own logit scale. The output is a probability, not a distance,
    which keeps it comparable with the other two components.
    """
    logits = np.array([tidy_similarity, cluttered_similarity], dtype=np.float64) * 100.0
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return float(probabilities[1])


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


def apply_sensitivity(score: float, sensitivity: float) -> float:
    """Remap a raw score through a gain curve. 0.5 is identity.

    Implemented as a gamma curve: sensitivity 0.8 pushes middling scores up (touchier), 0.2 pushes
    them down (more forgiving), and 0 and 1 are never allowed to saturate the output completely so
    a badly set slider cannot make an anchor permanently clean or permanently filthy.
    """
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    # sensitivity 0 -> gamma 3 (needs a lot to register); 1 -> gamma 1/3 (registers easily)
    gamma = 3.0 ** (1.0 - 2.0 * sensitivity)
    return float(np.clip(np.power(np.clip(score, 0.0, 1.0), gamma), 0.0, 1.0))


def fuse(
    *,
    baseline_diff: float | None,
    object_density_score: float,
    semantic: float | None,
    weights: Weights | None = None,
    sensitivity: float = 0.5,
) -> ClutterComponents:
    """Combine the components into one score.

    Missing components are dropped and the remaining weights renormalised, rather than being treated
    as zero. An anchor with no baseline captured yet should not read as permanently tidy - it should
    read as "scored from the other two signals", which is exactly what a user who has not got round
    to capturing a baseline expects.
    """
    weights = (weights or Weights()).normalised()

    present: list[tuple[float, float]] = []
    if baseline_diff is not None:
        present.append((baseline_diff, weights.baseline_diff))
    present.append((object_density_score, weights.object_density))
    if semantic is not None:
        present.append((semantic, weights.semantic))

    total_weight = sum(weight for _, weight in present) or 1.0
    raw = sum(value * weight for value, weight in present) / total_weight
    fused = apply_sensitivity(raw, sensitivity)

    contributions = (
        (baseline_diff or 0.0) * weights.baseline_diff / total_weight,
        object_density_score * weights.object_density / total_weight,
        (semantic or 0.0) * weights.semantic / total_weight,
    )
    return ClutterComponents(
        baseline_diff=baseline_diff if baseline_diff is not None else 0.0,
        object_density=object_density_score,
        semantic=semantic if semantic is not None else 0.0,
        fused=fused,
        contributions=contributions,
    )


def subregion_scores(
    region_scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Order subregions worst-first, for spatial micro-tasking.

    This is what turns "tidy the shelf" into "just clear the left third": the ladder starts with the
    worst slice, so the first step is both the most useful and the most visibly satisfying.
    """
    return sorted(region_scores.items(), key=lambda item: item[1], reverse=True)


def explain(components: ClutterComponents, *, anchor_label: str) -> str:
    """One sentence explaining the score, shown under the snapshot in the UI."""
    reasons = {
        "baseline_diff": f"{anchor_label} looks different from its clean reference",
        "object_density": f"several movable items are on {anchor_label}",
        "semantic_clutter": f"{anchor_label} reads as cluttered overall",
    }
    return f"{reasons[components.dominant()]} (score {components.fused:.2f})"


__all__ = [
    "CLUTTER_PROBES",
    "MOVABLE_COCO",
    "STRUCTURAL_COCO",
    "ClutterComponents",
    "Weights",
    "apply_sensitivity",
    "explain",
    "fuse",
    "object_density",
    "semantic_clutter",
    "structural_difference",
    "subregion_scores",
]
