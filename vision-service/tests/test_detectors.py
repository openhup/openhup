"""Detector tests that need no models.

`ScreenOn` is fully exercised because it needs no weights at all. The model-backed detectors are
tested for their *degradation* behaviour, which is the part that decides whether a half-configured
install is merely limited or actively misleading.
"""

from __future__ import annotations

import numpy as np
import pytest
from openhup_schemas import BUILTIN_DETECTORS, SignalKind

from openhup_vision import detectors, roi
from openhup_vision.backends import ModelRegistry, ModelUnavailable, SessionCache

FULL = roi.Region(id="full", label="Full frame", points=())


def context(**kwargs) -> detectors.DetectorContext:
    kwargs.setdefault("anchor_id", "living.tv")
    kwargs.setdefault("anchor_label", "TV")
    kwargs.setdefault("region", FULL)
    return detectors.DetectorContext(**kwargs)


def dark(size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), 12, dtype=np.uint8)


def lit_screen(size: int = 64, *, seed: int = 0) -> np.ndarray:
    """Bright with structure, like a screen showing something."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 190, dtype=np.uint8)
    image[::4] = rng.integers(40, 255, (len(image[::4]), size, 3), dtype=np.uint8)
    return image


def flat_bright(size: int = 64) -> np.ndarray:
    """Bright and featureless, like a sunlit wall. Must NOT read as a screen."""
    return np.full((size, size, 3), 210, dtype=np.uint8)


# ------------------------------------------------------------------ ScreenOn


def test_dark_room_screen_is_off() -> None:
    result = detectors.ScreenOn().detect(dark(), context())
    assert result.signals[0].key == "screen_on"
    assert result.signals[0].value is False


def test_lit_screen_is_on() -> None:
    result = detectors.ScreenOn().detect(lit_screen(), context())
    assert result.signals[0].value is True


def test_flat_bright_surface_is_not_a_screen() -> None:
    """A sunlit wall is bright and flat; a screen is bright and structured."""
    result = detectors.ScreenOn().detect(flat_bright(), context())
    assert result.signals[0].value is False


def test_screen_activity_rises_with_changing_content() -> None:
    """Distinguishes a playing film from a paused menu left up for hours."""
    detector = detectors.ScreenOn()
    ctx = context()
    for seed in range(8):
        varied = detector.detect(lit_screen(seed=seed) // (1 + seed % 3), ctx)
    changing = varied.signals[1].value

    still_detector = detectors.ScreenOn()
    for _ in range(8):
        static = still_detector.detect(lit_screen(seed=1), ctx)
    paused = static.signals[1].value

    assert changing > paused


def test_screen_signals_match_the_declared_contract() -> None:
    result = detectors.ScreenOn().detect(lit_screen(), context())
    spec = BUILTIN_DETECTORS.get("screen_on")
    for signal in result.signals:
        declared = spec.signal(signal.key)
        assert declared is not None, signal.key
        assert declared.kind is signal.kind


def test_screen_threshold_is_tunable() -> None:
    bright = lit_screen()
    strict = detectors.ScreenOn().detect(bright, context(params={"luminance_threshold": 0.99}))
    assert strict.signals[0].value is False


# ------------------------------------------------------------------ degradation


def test_object_inventory_raises_clearly_with_no_model(tmp_path) -> None:
    """A missing model must produce a legible error, not a mysterious crash in the capture loop."""
    registry = ModelRegistry(models={}, directory=tmp_path)
    sessions = SessionCache(registry)
    detector = detectors.ObjectInventory(sessions, model_id="yolox-s")
    with pytest.raises((ModelUnavailable, RuntimeError)):
        detector.detect(dark(), context())


def test_clutter_score_survives_with_no_models_at_all(tmp_path) -> None:
    """Baseline diff alone still yields a usable score, renormalised rather than zeroed.

    This is the "I have not installed onnxruntime yet" path, and it must produce a number rather
    than an exception - otherwise first-run experience is a stack trace.
    """
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.ClutterScore(sessions, inventory=None)

    baseline = dark()
    messy = dark()
    messy[10:50, 10:50] = 220

    result = detector.detect(messy, context(anchor_id="kitchen.counter", baseline=baseline))
    clutter = next(s for s in result.signals if s.key == "clutter_level")
    assert clutter.kind is SignalKind.SCALAR
    assert clutter.value > 0.05
    assert result.extra["used_baseline"] is True


def test_clutter_score_without_a_baseline_still_scores(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.ClutterScore(sessions, inventory=None)
    result = detector.detect(dark(), context(baseline=None))
    assert result.extra["used_baseline"] is False
    assert next(s for s in result.signals if s.key == "clutter_level").value >= 0.0


def test_clutter_score_publishes_every_component(tmp_path) -> None:
    """The components are what make the number explainable, so they are not optional."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.ClutterScore(sessions, inventory=None)
    result = detector.detect(dark(), context(baseline=dark()))
    keys = {s.key for s in result.signals}
    assert keys == {"clutter_level", "baseline_diff", "object_density", "semantic_clutter"}
    assert "explanation" in result.extra


def test_clutter_scores_subregions_for_the_ladder(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.ClutterScore(sessions, inventory=None)
    left = roi.Region(
        id="left", label="Left", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
    )
    right = roi.Region(
        id="right", label="Right", points=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
    )

    baseline = dark()
    messy = dark()
    messy[:, :30] = 230  # only the left side is a mess

    result = detector.detect(messy, context(baseline=baseline, subregions=(left, right)))
    ordered = result.extra["subregions"]
    assert ordered[0][0] == "left"  # worst first, so step one is the most satisfying


def test_zero_shot_reports_unknown_rather_than_guessing(tmp_path) -> None:
    """No model, no probe cache: say 'unknown', which matches neither branch of a good skill."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.ZeroShotState(sessions)
    result = detector.detect(
        dark(),
        context(params={"probes": {"on": "a lit burner", "off": "an unlit burner"}}),
    )
    assert result.signals[0].value == "unknown"
    assert result.signals[0].confidence == 0.0


def test_zero_shot_requires_probes(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    with pytest.raises(ValueError, match=r"params\.probes is required"):
        detectors.ZeroShotState(sessions).detect(dark(), context())


def test_fill_level_requires_a_container(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    with pytest.raises(ValueError, match=r"params\.container is required"):
        detectors.FillLevel(sessions).detect(dark(), context())


def test_fill_level_raises_clearly_with_no_model(tmp_path) -> None:
    """No model, no probe cache: raise rather than reading a bowl as empty."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.FillLevel(sessions)
    with pytest.raises((ModelUnavailable, RuntimeError)):
        detector.detect(dark(), context(params={"container": "pet water bowl"}))


def test_door_state_reports_unknown_rather_than_guessing(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    result = detectors.DoorState(sessions).detect(dark(), context())
    assert result.signals[0].key == "door_state"
    assert result.signals[0].value == "unknown"
    assert result.signals[0].confidence == 0.0


def test_presence_absence_requires_a_query(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    with pytest.raises(ValueError, match=r"params\.query is required"):
        detectors.PresenceAbsence(sessions).detect(dark(), context())


def test_presence_absence_raises_clearly_with_no_model(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    with pytest.raises((ModelUnavailable, RuntimeError)):
        detectors.PresenceAbsence(sessions).detect(
            dark(), context(params={"query": "a dish drying rack"})
        )


def test_sensor_detector_emits_nothing_on_a_frame(tmp_path) -> None:
    """Sensor values arrive over MQTT, not camera frames, so the frame path must be a no-op."""
    from openhup_vision.sensor_feed import SensorFeed

    detector = detectors.Sensor(SensorFeed())
    assert detector.detect(dark(), context()).signals == []


def test_pose_fall_raises_clearly_with_no_model(tmp_path) -> None:
    """A missing pose model must say so, not read as 'nobody is down'."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    with pytest.raises((ModelUnavailable, RuntimeError)):
        detectors.PoseFall(sessions).detect(dark(), context())


# ------------------------------------------------------------------ pose geometry


def _standing_keypoints() -> np.ndarray:
    """A COCO-layout person standing upright: shoulders above hips, torso vertical."""
    points = np.zeros((17, 2), dtype=np.float32)
    points[5] = (0.45, 0.35)  # left shoulder
    points[6] = (0.55, 0.35)  # right shoulder
    points[11] = (0.45, 0.65)  # left hip
    points[12] = (0.55, 0.65)  # right hip
    return points


def _prone_keypoints() -> np.ndarray:
    """A person lying down: shoulders and hips at the same height, torso horizontal."""
    points = np.zeros((17, 2), dtype=np.float32)
    points[5] = (0.20, 0.60)  # left shoulder
    points[6] = (0.30, 0.60)  # right shoulder
    points[11] = (0.50, 0.60)  # left hip
    points[12] = (0.60, 0.60)  # right hip
    return points


def test_is_down_detects_a_horizontal_torso() -> None:
    assert detectors._is_down(_prone_keypoints()) is True


def test_is_down_rejects_a_vertical_torso() -> None:
    assert detectors._is_down(_standing_keypoints()) is False


def test_is_down_tolerates_a_short_keypoint_array() -> None:
    assert detectors._is_down(np.zeros((5, 2), dtype=np.float32)) is False


def test_motion_level_rises_with_displacement() -> None:
    assert detectors._motion_level(_prone_keypoints(), None) == 0.0
    moved = _prone_keypoints() + 0.05
    assert detectors._motion_level(moved, _prone_keypoints()) > 0.0


def test_decode_keypoints_handles_movenet_shape() -> None:
    """MoveNet exports [1, 1, K, 3] as (y, x, confidence); we remap to (x, y)."""
    raw = np.zeros((1, 1, 17, 3), dtype=np.float32)
    raw[0, 0, 5] = (0.35, 0.45, 0.9)  # shoulder: y=0.35, x=0.45
    keypoints = detectors._decode_keypoints([raw])
    assert keypoints is not None
    assert keypoints[5][0] == 0.45
    assert keypoints[5][1] == 0.35


def test_decode_keypoints_returns_none_for_unknown_shape() -> None:
    assert detectors._decode_keypoints([np.zeros((1, 3, 640, 640), dtype=np.float32)]) is None


# ------------------------------------------------------------------ face identity (ADR-016)


def test_best_match_names_a_member_above_the_threshold() -> None:
    """An enrolled face scores as known; the id comes back, never a name."""
    gallery = [("m-sam", [1.0, 0.0, 0.0, 0.0])]
    assert detectors._best_match([1.0, 0.0, 0.0, 0.0], gallery, threshold=0.55) == "m-sam"
    assert detectors._best_match([0.9, 0.1, 0.0, 0.0], gallery, threshold=0.55) == "m-sam"


def test_best_match_returns_none_below_the_threshold() -> None:
    """A gallery of one is not a licence to guess: a poor match is an unknown person."""
    gallery = [("m-sam", [1.0, 0.0, 0.0, 0.0])]
    assert detectors._best_match([0.0, 1.0, 0.0, 0.0], gallery, threshold=0.55) is None


def test_best_match_picks_the_closest_of_several_members() -> None:
    gallery = [("m-lee", [0.0, 1.0, 0.0, 0.0]), ("m-sam", [1.0, 0.0, 0.0, 0.0])]
    assert detectors._best_match([0.95, 0.05, 0.0, 0.0], gallery, threshold=0.55) == "m-sam"


def test_face_identity_raises_clearly_with_no_models(tmp_path) -> None:
    """Missing identity weights must say so, not read as 'nobody is here'."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.FaceIdentity(sessions)
    with pytest.raises((ModelUnavailable, RuntimeError)):
        detector.detect(dark(), context())


def test_decode_faces_handles_yunet_shape() -> None:
    """YuNet exports [1, N, 15]; we take the box and score from the first five columns."""
    raw = np.zeros((1, 1, 15), dtype=np.float32)
    raw[0, 0] = [10, 20, 40, 50, 0.9] + [0.0] * 10
    faces = detectors._decode_faces([raw], patch_shape=(100, 100))
    assert len(faces) == 1
    x1, y1, x2, y2, score = faces[0]
    assert (x1, y1, x2, y2) == pytest.approx((0.1, 0.2, 0.4, 0.5))
    assert score == pytest.approx(0.9)


def test_decode_faces_returns_empty_for_unknown_shape() -> None:
    assert detectors._decode_faces([np.zeros((1, 3, 640, 640), dtype=np.float32)], (100, 100)) == []


# ------------------------------------------------------------------ registry parity


def test_implemented_detectors_are_all_declared_in_the_schema(tmp_path) -> None:
    """The vision service must not invent a detector the backend has never heard of."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    for name in detectors.build_registry(sessions):
        assert BUILTIN_DETECTORS.get(name) is not None, name


def test_the_gap_between_schema_and_implementation_is_explicit(tmp_path) -> None:
    """Anything declared but unimplemented must be listed, so it is a known gap not a surprise."""
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    implemented = set(detectors.build_registry(sessions))
    declared = set(BUILTIN_DETECTORS.names())
    missing = declared - implemented
    assert missing == set(detectors.NOT_YET_IMPLEMENTED)


def test_declared_signal_keys_match_implementations(tmp_path) -> None:
    sessions = SessionCache(ModelRegistry(models={}, directory=tmp_path))
    detector = detectors.ClutterScore(sessions, inventory=None)
    result = detector.detect(dark(), context(baseline=dark()))
    spec = BUILTIN_DETECTORS.get("clutter_score")
    for signal in result.signals:
        declared = spec.signal(signal.key)
        assert declared is not None, f"clutter_score emits undeclared signal {signal.key}"
        assert declared.kind is signal.kind


# ------------------------------------------------------------------ NMS


def test_nms_removes_duplicate_boxes() -> None:
    from openhup_schemas import BBox

    boxes = [
        BBox(label="cup", score=0.9, x1=0.1, y1=0.1, x2=0.3, y2=0.3),
        BBox(label="cup", score=0.7, x1=0.11, y1=0.11, x2=0.31, y2=0.31),
        BBox(label="cup", score=0.8, x1=0.6, y1=0.6, x2=0.8, y2=0.8),
    ]
    kept = detectors._nms(boxes)
    assert len(kept) == 2


def test_nms_keeps_overlapping_boxes_of_different_classes() -> None:
    from openhup_schemas import BBox

    boxes = [
        BBox(label="cup", score=0.9, x1=0.1, y1=0.1, x2=0.3, y2=0.3),
        BBox(label="bowl", score=0.8, x1=0.1, y1=0.1, x2=0.3, y2=0.3),
    ]
    assert len(detectors._nms(boxes)) == 2
