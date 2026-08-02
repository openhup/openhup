"""Every shipped example must parse and compile.

Examples are documentation that runs. A broken example is worse than a missing one: someone copies
it, it silently does the wrong thing, and they conclude the tool is unreliable. This test loads the
real files from `examples/` and compiles them against the real anchors from `examples/cameras/`, so
the two cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openhup_schemas import (
    Anchor,
    Camera,
    Personality,
    Skill,
    Urgency,
)

from openhup.llm import audit_personality
from openhup.skills.compile import compile_skill

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SKILL_DIR = EXAMPLES / "skills"


def load_example_skills() -> list[Skill]:
    skills: list[Skill] = []
    for path in sorted(SKILL_DIR.glob("*.yaml")):
        text = path.read_text()
        # A skills directory legitimately holds other things - goals.yaml, notes. Identify skill
        # documents structurally (a mapping with `watch` and `conditions`) rather than by filename,
        # so adding a non-skill file here never breaks this loader.
        documents = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
        if not any("watch" in d and "conditions" in d for d in documents):
            continue
        skills.extend(Skill.model_validate(document) for document in documents)
    return skills


@pytest.fixture(scope="module")
def example_anchors() -> dict[str, Anchor]:
    raw = yaml.safe_load((EXAMPLES / "cameras" / "cameras.yaml").read_text())
    anchors = {}
    for entry in raw["anchors"]:
        anchor = Anchor.model_validate(entry)
        # Every example that uses clutter_score against a baseline needs one to exist; in a real
        # deployment this is captured via the API while the space is tidy.
        anchor.baseline_ref = f"snap://baseline/{anchor.id}.jpg"
        anchors[anchor.id] = anchor
    return anchors


@pytest.fixture(scope="module")
def example_skills() -> list[Skill]:
    return load_example_skills()


def test_examples_exist(example_skills) -> None:
    ids = {s.id for s in example_skills}
    # The four named in the design brief must be present and must keep these ids.
    assert {
        "kitchen-clutter-buster",
        "stove-burner-safety",
        "adhd-micro-task-shelf",
        "tv-time-tracking",
    } <= ids
    assert len(example_skills) >= 12


def test_example_cameras_parse() -> None:
    raw = yaml.safe_load((EXAMPLES / "cameras" / "cameras.yaml").read_text())
    cameras = [Camera.model_validate(entry) for entry in raw["cameras"]]
    assert {c.id for c in cameras} >= {"kitchen", "living", "office", "hall"}
    # No plaintext secret may ever appear in a committed example.
    for camera in cameras:
        assert not hasattr(camera, "password")
        if camera.username:
            assert camera.password_env, f"{camera.id} has a username but no password_env"


def test_every_anchor_referenced_by_an_example_exists(example_skills, example_anchors) -> None:
    for skill in example_skills:
        for watch in skill.watch:
            if watch.anchor:
                assert watch.anchor in example_anchors, f"{skill.id} -> {watch.anchor}"


def test_every_example_compiles(example_skills, example_anchors) -> None:
    for skill in example_skills:
        compiled = compile_skill(skill, anchors=example_anchors)
        assert compiled.anchor_ids


def test_no_example_has_a_blocking_lint(example_skills, example_anchors) -> None:
    """Warnings are fine and often instructive. Errors would mean a broken example."""
    for skill in example_skills:
        compiled = compile_skill(skill, anchors=example_anchors, strict=False)
        errors = [w for w in compiled.warnings if w.error]
        assert not errors, f"{skill.id}: {errors}"


def test_task_examples_are_calm(example_skills, example_anchors) -> None:
    """Every shipped task example must have anti-nag limits set.

    Examples get copied. If the examples are noisy, every derived skill is noisy.
    """
    for skill in example_skills:
        if skill.effect.type != "task":
            continue
        assert skill.limits.cooldown.total_seconds() >= 300, skill.id
        assert skill.limits.max_per_day is not None, skill.id


def test_safety_examples_do_not_wear_a_costume(example_skills) -> None:
    """High-urgency examples must not set a jokey personality, even though it would be ignored."""
    for skill in example_skills:
        if skill.urgency.rank >= Urgency.HIGH.rank:
            assert skill.effective_personality in (None, "brief"), skill.id


def test_metric_examples_store_no_imagery(example_skills) -> None:
    for skill in example_skills:
        if skill.effect.type == "metric":
            assert not skill.snapshot.attach, skill.id


def test_examples_have_hysteresis_gaps(example_skills) -> None:
    """Spot-check the headline example's thresholds, since it is the one people copy."""
    kitchen = next(s for s in example_skills if s.id == "kitchen-clutter-buster")
    trigger = kitchen.conditions.all_[0]
    resolve = kitchen.resolve.conditions.all_[0]
    assert trigger.value > resolve.value * 2


def test_example_personalities_parse_and_are_clean() -> None:
    raw = yaml.safe_load((EXAMPLES / "personalities" / "personalities.yaml").read_text())
    personalities = [Personality.model_validate(entry) for entry in raw]
    # The five originals plus the five personality-gamble presets (ADR-014).
    assert len(personalities) == 10
    for personality in personalities:
        # A shipped personality's own style_prompt must not itself contain something the filter
        # would reject if a model echoed it back.
        assert not audit_personality(personality.style_prompt), personality.id
