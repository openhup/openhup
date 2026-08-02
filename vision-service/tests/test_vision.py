"""Tests for the pure vision maths: ROI geometry, motion gating, clutter fusion.

Deliberately runs with numpy alone - no ONNX Runtime, no camera, no GPU - so the parts of the vision
service that encode judgement rather than plumbing are testable on any machine, including CI.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from openhup_vision import fusion, roi
from openhup_vision.sampler import AnchorSampler, Cadence, motion_score

UTC = UTC
T0 = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)

FULL = roi.Region(id="full", label="Full frame", points=())
LEFT_HALF = roi.Region(
    id="left", label="Left half", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
)
CENTRE = roi.Region(
    id="centre", label="Centre", points=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))
)


def frame(value: int = 40, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


# ------------------------------------------------------------------ ROI geometry


def test_full_frame_region_covers_everything() -> None:
    assert FULL.is_full_frame
    assert FULL.mask(32, 32).all()
    assert FULL.bbox(32, 32) == (0, 0, 32, 32)


def test_polygon_mask_covers_roughly_the_right_area() -> None:
    assert LEFT_HALF.area_fraction(64, 64) == pytest.approx(0.5, abs=0.03)
    assert CENTRE.area_fraction(64, 64) == pytest.approx(0.25, abs=0.03)


def test_polygon_mask_is_in_the_right_place() -> None:
    mask = LEFT_HALF.mask(64, 64)
    assert mask[:, 5].all()
    assert not mask[:, 60].any()


def test_crop_masks_pixels_outside_the_polygon() -> None:
    """A counter ROI clipping the doorway must not light up when someone walks past."""
    image = frame(200)
    cropped = roi.crop(image, CENTRE)
    assert cropped.shape[0] < image.shape[0]
    assert cropped.max() == 200


def test_crop_blacks_out_corners_of_a_diagonal_region() -> None:
    triangle = roi.Region(id="t", label="T", points=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cropped = roi.crop(frame(255), triangle)
    # Bottom-right corner is outside the triangle and must be zeroed.
    assert cropped[-1, -1].sum() == 0
    assert cropped[0, 0].sum() > 0


def test_degenerate_polygon_does_not_produce_an_empty_crop() -> None:
    sliver = roi.Region(id="s", label="S", points=((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)))
    cropped = roi.crop(frame(), sliver)
    assert cropped.size > 0


def test_letterbox_preserves_aspect_and_reports_padding() -> None:
    wide = np.full((40, 80, 3), 90, dtype=np.uint8)
    out, scale, pad_x, pad_y = roi.resize_letterbox(wide, 64)
    assert out.shape == (64, 64, 3)
    assert scale == pytest.approx(0.8)
    assert pad_x == 0
    assert pad_y > 0


def test_box_maps_back_to_region_coordinates() -> None:
    box = roi.to_region_coords(
        (10, 20, 30, 40), scale=0.5, pad_x=0, pad_y=8, region_w=64, region_h=64
    )
    assert all(0.0 <= v <= 1.0 for v in box)
    assert box[0] < box[2]


def test_blur_boxes_destroys_detail_only_inside_the_box() -> None:
    """Redaction must be irreversible and must not touch the rest of the frame."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    blurred = roi.blur_boxes(image, [(0.0, 0.0, 0.5, 0.5)])
    left = blurred[:32, :32]
    right = blurred[:32, 32:]
    assert left.std() < image[:32, :32].std()
    assert np.array_equal(right, image[:32, 32:])


def test_region_from_anchor_accepts_pairs_and_objects() -> None:
    region = roi.region_from_anchor("a.b", "Label", [[0.1, 0.2], {"x": 0.8, "y": 0.2}, [0.5, 0.9]])
    assert len(region.points) == 3
    assert region.points[1] == (0.8, 0.2)


# ------------------------------------------------------------------ motion gating


def test_identical_frames_have_no_motion() -> None:
    assert motion_score(frame(), frame()) == 0.0


def test_a_changed_patch_registers_motion() -> None:
    after = frame()
    after[10:30, 10:30] = 220
    assert motion_score(after, frame()) > 0.05


def test_motion_outside_the_region_is_ignored() -> None:
    """The whole point of ROI-scoped gating."""
    after = frame()
    after[:, 40:60] = 220  # right side only
    assert motion_score(after, frame(), LEFT_HALF) == pytest.approx(0.0, abs=0.01)
    assert motion_score(after, frame(), FULL) > 0.05


def test_uniform_noise_below_threshold_is_not_motion() -> None:
    """Sensor noise and JPEG artefacts must not wake the detectors."""
    noisy = frame().astype(np.int16) + np.random.default_rng(1).integers(-8, 8, (64, 64, 3))
    assert motion_score(noisy.clip(0, 255).astype(np.uint8), frame()) < 0.01


def test_resolution_change_counts_as_fully_novel() -> None:
    assert motion_score(frame(size=32), frame(size=64)) == 1.0


# ------------------------------------------------------------------ sampling policy


def test_first_frame_always_runs() -> None:
    sampler = AnchorSampler(anchor_id="kitchen.counter")
    decision = sampler.consider(frame(), T0)
    assert decision.run
    assert decision.reason == "first frame"


def test_frames_inside_the_interval_are_skipped() -> None:
    sampler = AnchorSampler(anchor_id="a", cadence=Cadence(active=timedelta(seconds=5)))
    sampler.consider(frame(), T0)
    decision = sampler.consider(frame(), T0 + timedelta(seconds=1))
    assert not decision.run
    assert "interval" in decision.reason


def test_a_still_scene_is_skipped_after_the_interval() -> None:
    sampler = AnchorSampler(anchor_id="a", cadence=Cadence(active=timedelta(seconds=5)))
    sampler.consider(frame(), T0)
    decision = sampler.consider(frame(), T0 + timedelta(seconds=6))
    assert not decision.run
    assert decision.reason == "no motion"


def test_motion_after_the_interval_runs() -> None:
    sampler = AnchorSampler(anchor_id="a", cadence=Cadence(active=timedelta(seconds=5)))
    sampler.consider(frame(), T0)
    moved = frame()
    moved[10:40, 10:40] = 230
    decision = sampler.consider(moved, T0 + timedelta(seconds=6))
    assert decision.run
    assert decision.reason == "motion"


def test_heartbeat_fires_on_a_permanently_still_scene() -> None:
    """A tidy room must keep producing observations, or every skill on it goes STALE."""
    cadence = Cadence(active=timedelta(seconds=5), heartbeat=timedelta(minutes=5))
    sampler = AnchorSampler(anchor_id="a", cadence=cadence)
    sampler.consider(frame(), T0)

    ran = []
    for step in range(1, 80):
        decision = sampler.consider(frame(), T0 + timedelta(seconds=step * 10))
        if decision.run:
            ran.append(decision.reason)
    assert "heartbeat" in ran
    assert sampler.heartbeats >= 1


def test_cadence_slows_down_when_nothing_happens() -> None:
    cadence = Cadence(
        active=timedelta(seconds=5),
        idle=timedelta(seconds=30),
        dormant=timedelta(minutes=2),
        settle=timedelta(minutes=2),
        hibernate=timedelta(minutes=10),
    )
    assert cadence.interval(timedelta(seconds=10)) == timedelta(seconds=5)
    assert cadence.interval(timedelta(minutes=3)) == timedelta(seconds=30)
    assert cadence.interval(timedelta(minutes=20)) == timedelta(minutes=2)


def test_gating_is_worth_the_code() -> None:
    """The claim in the README: an idle scene skips the overwhelming majority of frames."""
    sampler = AnchorSampler(anchor_id="a", cadence=Cadence(active=timedelta(seconds=5)))
    for step in range(300):  # 5fps for a minute of an empty room
        sampler.consider(frame(), T0 + timedelta(seconds=step * 0.2))
    assert sampler.efficiency > 0.9
    assert sampler.stats()["frames_seen"] == 300


def test_slow_accumulation_is_still_detected() -> None:
    """A pile growing over ten minutes must not slide under a per-frame delta.

    The reference is the last *evaluated* frame, not the previous frame, so gradual change
    accumulates rather than being invisible one crumb at a time.
    """
    sampler = AnchorSampler(anchor_id="a", cadence=Cadence(active=timedelta(seconds=5)))
    sampler.consider(frame(), T0)

    ran_on_growth = False
    image = frame()
    for step in range(1, 40):
        # Add one small object every step - each change alone is tiny.
        image = image.copy()
        image[step, 0:6] = 230
        decision = sampler.consider(image, T0 + timedelta(seconds=step * 6))
        if decision.run and decision.reason == "motion":
            ran_on_growth = True
    assert ran_on_growth


def test_force_next_overrides_the_interval() -> None:
    sampler = AnchorSampler(anchor_id="a")
    sampler.consider(frame(), T0)
    sampler.force_next()
    assert sampler.consider(frame(), T0 + timedelta(milliseconds=1)).run


# ------------------------------------------------------------------ clutter fusion


def test_identical_frame_has_no_structural_difference() -> None:
    assert fusion.structural_difference(frame(), frame()) == pytest.approx(0.0, abs=0.01)


def test_added_objects_raise_structural_difference() -> None:
    messy = frame(40)
    messy[10:50, 10:50] = 220
    assert fusion.structural_difference(messy, frame(40)) > 0.1


def test_uniform_lighting_change_is_mostly_absorbed() -> None:
    """Turning a lamp on must not read as a mess. This is why pixel diff alone is not enough."""
    brighter = frame(40).astype(np.int16) + 45
    score = fusion.structural_difference(brighter.clip(0, 255).astype(np.uint8), frame(40))
    assert score < 0.1


def test_object_density_ignores_structural_furniture() -> None:
    """A counter with a sink and an oven in shot is not thereby cluttered."""
    assert fusion.object_density(["sink", "oven", "refrigerator"], [0.2, 0.3, 0.2]) == 0.0


def test_object_density_counts_movable_items() -> None:
    labels = ["cup", "bowl", "book", "cell phone", "vase"]
    assert fusion.object_density(labels, [0.02] * 5) == pytest.approx(0.5, abs=0.01)


def test_object_density_saturates_on_one_huge_pile() -> None:
    """Twelve small items is a mess; one enormous pile is also a mess."""
    assert fusion.object_density(["backpack"], [0.4]) == 1.0


def test_semantic_clutter_is_a_probability() -> None:
    assert fusion.semantic_clutter(0.30, 0.30) == pytest.approx(0.5)
    assert fusion.semantic_clutter(0.20, 0.32) > 0.7
    assert fusion.semantic_clutter(0.32, 0.20) < 0.3


def test_sensitivity_is_identity_at_half() -> None:
    for value in (0.1, 0.4, 0.6, 0.9):
        assert fusion.apply_sensitivity(value, 0.5) == pytest.approx(value, abs=1e-9)


def test_sensitivity_raises_and_lowers_middling_scores() -> None:
    assert fusion.apply_sensitivity(0.5, 0.9) > 0.5  # touchier
    assert fusion.apply_sensitivity(0.5, 0.1) < 0.5  # more forgiving


def test_sensitivity_cannot_saturate_completely() -> None:
    """A badly set slider must not make an anchor permanently filthy or permanently spotless."""
    assert fusion.apply_sensitivity(0.5, 1.0) < 1.0
    assert fusion.apply_sensitivity(0.5, 0.0) > 0.0


def test_fusion_combines_and_reports_components() -> None:
    result = fusion.fuse(baseline_diff=0.8, object_density_score=0.6, semantic=0.7)
    assert 0.6 < result.fused < 0.8
    assert result.as_dict()["baseline_diff"] == 0.8
    assert sum(result.contributions) == pytest.approx(result.fused, abs=0.02)


def test_missing_baseline_renormalises_instead_of_scoring_zero() -> None:
    """An anchor with no baseline yet must not read as permanently tidy."""
    without = fusion.fuse(baseline_diff=None, object_density_score=0.8, semantic=0.8)
    treated_as_zero = fusion.fuse(baseline_diff=0.0, object_density_score=0.8, semantic=0.8)
    assert without.fused > treated_as_zero.fused
    assert without.fused == pytest.approx(0.8, abs=0.05)


def test_weights_shift_the_result() -> None:
    density_heavy = fusion.Weights(baseline_diff=0.1, object_density=0.8, semantic=0.1)
    result = fusion.fuse(
        baseline_diff=0.0, object_density_score=1.0, semantic=0.0, weights=density_heavy
    )
    assert result.fused > 0.7
    assert result.dominant() == "object_density"


def test_weights_are_relative_not_absolute() -> None:
    a = fusion.fuse(
        baseline_diff=0.9,
        object_density_score=0.1,
        semantic=0.1,
        weights=fusion.Weights(2, 1, 1),
    )
    b = fusion.fuse(
        baseline_diff=0.9,
        object_density_score=0.1,
        semantic=0.1,
        weights=fusion.Weights(0.5, 0.25, 0.25),
    )
    assert a.fused == pytest.approx(b.fused)


def test_subregions_are_ordered_worst_first() -> None:
    """The micro-task ladder starts with the worst slice, so step one is the most satisfying."""
    ordered = fusion.subregion_scores({"left": 0.2, "middle": 0.9, "right": 0.5})
    assert [name for name, _ in ordered] == ["middle", "right", "left"]


def test_explanation_names_the_dominant_component() -> None:
    components = fusion.fuse(baseline_diff=0.1, object_density_score=0.9, semantic=0.1)
    text = fusion.explain(components, anchor_label="the kitchen counter")
    assert "movable items" in text
    assert "score 0." in text
