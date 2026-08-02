"""Skill compilation: turn a validated `Skill` into something the engine can run, and reject the
skills that would misbehave.

Pydantic validation catches malformed skills. Compilation catches *wrong* ones - the ones that parse
fine and then behave badly at three in the morning:

* an operator that cannot apply to its signal's kind (`contains` against a scalar);
* a detector that needs a clean baseline on an anchor that has never had one captured;
* and the big one: **a resolve threshold that does not sit inside the trigger threshold**, which
  produces a task that closes and reopens forever. Refusing to save that skill is the single most
  valuable check in this file.

Everything here is synchronous and pure; anchors and the detector registry are passed in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from openhup_schemas import (
    BUILTIN_DETECTORS,
    Anchor,
    Condition,
    DetectorRegistry,
    MetricEffect,
    Op,
    SignalKey,
    SignalKind,
    SignalPredicate,
    SignalSpec,
    Skill,
    TaskEffect,
    iter_predicates,
)

#: Below this, a task-producing skill re-triggers fast enough to feel like nagging.
MIN_SANE_COOLDOWN = timedelta(minutes=5)

#: Numeric operators meaning "rose into range" and "fell into range".
_UPWARD = frozenset({Op.GTE, Op.GT})
_DOWNWARD = frozenset({Op.LTE, Op.LT})
_NUMERIC_KINDS = frozenset({SignalKind.SCALAR, SignalKind.COUNT})


@dataclass(frozen=True, slots=True)
class Lint:
    """A compile finding. `error=True` blocks the save."""

    code: str
    message: str
    error: bool = False
    binding: str | None = None

    def __str__(self) -> str:
        return f"{'error' if self.error else 'warning'}[{self.code}] {self.message}"


class SkillCompileError(ValueError):
    """Raised when a skill cannot be compiled. Carries every problem, not just the first."""

    def __init__(self, skill_id: str, findings: Sequence[Lint]) -> None:
        self.skill_id = skill_id
        self.findings = list(findings)
        detail = "\n".join(f"  - {f}" for f in self.findings)
        super().__init__(f"skill {skill_id!r} failed to compile:\n{detail}")


@dataclass(frozen=True, slots=True)
class CompiledBinding:
    """A signal binding resolved against the detector registry."""

    id: str
    detector: str
    signal: str
    spec: SignalSpec
    params: Mapping[str, object] = field(default_factory=dict)

    def key(self, anchor_id: str) -> SignalKey:
        return SignalKey(anchor_id, self.detector, self.signal)


@dataclass(frozen=True, slots=True)
class CompiledSkill:
    """Everything the engine needs to run one skill, with anchors already expanded."""

    skill: Skill
    anchor_ids: tuple[str, ...]
    bindings: tuple[CompiledBinding, ...]
    horizon: timedelta
    warnings: tuple[Lint, ...] = ()

    def binding(self, binding_id: str) -> CompiledBinding:
        for candidate in self.bindings:
            if candidate.id == binding_id:
                return candidate
        raise KeyError(f"skill {self.skill.id!r} has no binding {binding_id!r}")

    def signal_keys(self, anchor_id: str) -> dict[str, SignalKey]:
        """binding id → SignalKey for one anchor. This is what builds a `BindingWindows` view."""
        return {b.id: b.key(anchor_id) for b in self.bindings}

    def all_signal_keys(self) -> list[SignalKey]:
        return [b.key(anchor) for anchor in self.anchor_ids for b in self.bindings]

    @property
    def staleness_timeout(self) -> timedelta:
        return self.skill.limits.staleness_timeout

    @property
    def instances(self) -> list[tuple[str, str]]:
        """(skill_id, anchor_id) pairs - one FSM instance each."""
        return [(self.skill.id, anchor) for anchor in self.anchor_ids]


def compile_skill(
    skill: Skill,
    *,
    registry: DetectorRegistry | None = None,
    anchors: Mapping[str, Anchor] | None = None,
    strict: bool = True,
) -> CompiledSkill:
    """Compile one skill.

    Args:
        skill: a schema-valid skill.
        registry: detectors available in this deployment. Defaults to the built-ins.
        anchors: known anchors by id. When provided, anchor existence and baseline availability are
            checked; when omitted (unit tests, static linting) those checks are skipped.
        strict: when False, errors are downgraded to warnings - for the "explain why this imported
            skill will not run" UI path. The engine never uses it.

    Raises:
        SkillCompileError: with every finding, so the UI can show a complete list.
    """
    registry = registry or BUILTIN_DETECTORS
    findings: list[Lint] = []

    anchor_ids = _resolve_anchors(skill, anchors, findings)
    bindings = _resolve_bindings(skill, registry, findings)
    by_id = {b.id: b for b in bindings}

    _check_tree(skill.conditions, by_id, findings, where="conditions")
    if skill.resolve is not None and skill.resolve.conditions is not None:
        _check_tree(skill.resolve.conditions, by_id, findings, where="resolve.conditions")

    _check_baselines(bindings, registry, anchors, anchor_ids, findings)
    _check_required_params(bindings, registry, findings)
    _check_hysteresis(skill, by_id, findings)
    _advise(skill, findings)

    errors = [f for f in findings if f.error]
    if errors and strict:
        raise SkillCompileError(skill.id, errors)

    return CompiledSkill(
        skill=skill,
        anchor_ids=tuple(anchor_ids),
        bindings=tuple(bindings),
        horizon=skill.horizon,
        warnings=tuple(f for f in findings if not f.error),
    )


def compile_all(
    skills: Sequence[Skill],
    *,
    registry: DetectorRegistry | None = None,
    anchors: Mapping[str, Anchor] | None = None,
) -> tuple[list[CompiledSkill], dict[str, SkillCompileError]]:
    """Compile many skills, isolating failures.

    One broken skill must not stop the engine running the other nineteen. Failures come back for
    surfacing in the UI and the startup log.
    """
    compiled: list[CompiledSkill] = []
    failures: dict[str, SkillCompileError] = {}
    for skill in skills:
        try:
            compiled.append(compile_skill(skill, registry=registry, anchors=anchors))
        except SkillCompileError as exc:
            failures[skill.id] = exc
    return compiled, failures


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def _resolve_anchors(
    skill: Skill,
    anchors: Mapping[str, Anchor] | None,
    findings: list[Lint],
) -> list[str]:
    """Expand `watch` into concrete anchor ids, including `camera:` wildcards."""
    resolved: list[str] = []
    for entry in skill.watch:
        if entry.anchor is not None:
            if anchors is not None and entry.anchor not in anchors:
                findings.append(
                    Lint(
                        "unknown_anchor",
                        f"watch references anchor {entry.anchor!r}, which does not exist",
                        error=True,
                    )
                )
                continue
            resolved.append(entry.anchor)
            continue

        if anchors is None:
            findings.append(
                Lint(
                    "camera_wildcard_unresolved",
                    f"watch uses camera {entry.camera!r}; anchors are required to expand it",
                )
            )
            continue

        matching = [a.id for a in anchors.values() if a.camera_id == entry.camera and a.enabled]
        if not matching:
            findings.append(
                Lint(
                    "camera_has_no_anchors",
                    f"camera {entry.camera!r} has no enabled anchors, so this skill would "
                    f"watch nothing",
                    error=True,
                )
            )
        resolved.extend(matching)

    deduped = list(dict.fromkeys(resolved))
    if not deduped and anchors is not None:
        findings.append(Lint("no_anchors", "skill resolves to zero anchors", error=True))
    return deduped


def _resolve_bindings(
    skill: Skill,
    registry: DetectorRegistry,
    findings: list[Lint],
) -> list[CompiledBinding]:
    bindings: list[CompiledBinding] = []
    for binding in skill.signals:
        detector = registry.get(binding.detector)
        if detector is None:
            findings.append(
                Lint(
                    "unknown_detector",
                    f"binding {binding.id!r} uses unknown detector {binding.detector!r}; "
                    f"available: {', '.join(registry.names())}",
                    error=True,
                    binding=binding.id,
                )
            )
            continue

        spec = detector.signal(binding.signal)
        if spec is None:
            available = ", ".join(s.key for s in detector.signals) or "(dynamic only)"
            findings.append(
                Lint(
                    "unknown_signal",
                    f"binding {binding.id!r}: detector {binding.detector!r} emits no signal "
                    f"{binding.signal!r}; available: {available}",
                    error=True,
                    binding=binding.id,
                )
            )
            continue

        if detector.optional:
            findings.append(
                Lint(
                    "optional_detector",
                    f"detector {detector.name!r} is opt-in: download its models and enable it in "
                    f"the vision config before this skill can run",
                    binding=binding.id,
                )
            )

        bindings.append(
            CompiledBinding(
                id=binding.id,
                detector=binding.detector,
                signal=binding.signal,
                spec=spec,
                params=dict(binding.params),
            )
        )
    return bindings


def _check_tree(
    tree: Condition,
    bindings: Mapping[str, CompiledBinding],
    findings: list[Lint],
    *,
    where: str,
) -> None:
    """Validate every predicate in one tree against its signal's kind."""
    for predicate in iter_predicates(tree):
        binding = bindings.get(predicate.signal)
        if binding is None:
            # Already reported by _resolve_bindings (unknown detector/signal) or by the schema's
            # own reference check. Nothing useful to add.
            continue

        spec = binding.spec
        if not spec.allows(predicate.op):
            findings.append(
                Lint(
                    "operator_kind_mismatch",
                    f"{where}: operator {predicate.op.value!r} cannot apply to "
                    f"{predicate.signal!r}, which is a {spec.kind.value} signal",
                    error=True,
                    binding=predicate.signal,
                )
            )
            continue

        _check_value(predicate, spec, findings, where=where)


def _check_value(
    predicate: SignalPredicate,
    spec: SignalSpec,
    findings: list[Lint],
    *,
    where: str,
) -> None:
    value = predicate.value

    if spec.kind in _NUMERIC_KINDS and predicate.op in (_UPWARD | _DOWNWARD):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            findings.append(
                Lint(
                    "non_numeric_threshold",
                    f"{where}: {predicate.signal!r} is numeric but the threshold is {value!r}",
                    error=True,
                    binding=predicate.signal,
                )
            )
            return
        if spec.range is not None:
            low, high = spec.range
            if not low <= float(value) <= high:
                findings.append(
                    Lint(
                        "threshold_out_of_range",
                        f"{where}: threshold {value} for {predicate.signal!r} is outside the "
                        f"signal's range {low}..{high}; this condition can never be true",
                        error=True,
                        binding=predicate.signal,
                    )
                )

    if (
        spec.kind is SignalKind.ENUM
        and spec.enum_values
        and isinstance(value, str)
        and value not in spec.enum_values
    ):
        findings.append(
            Lint(
                "unknown_enum_value",
                f"{where}: {predicate.signal!r} never takes the value {value!r}; "
                f"valid values are {spec.enum_values}",
                error=True,
                binding=predicate.signal,
            )
        )


def _check_baselines(
    bindings: Sequence[CompiledBinding],
    registry: DetectorRegistry,
    anchors: Mapping[str, Anchor] | None,
    anchor_ids: Sequence[str],
    findings: list[Lint],
) -> None:
    """A baseline-dependent detector on an anchor with no reference image is a silent no-op."""
    if anchors is None:
        return
    for binding in bindings:
        detector = registry.get(binding.detector)
        needs_baseline = bool(detector and detector.requires_baseline)
        # clutter_score only needs one when explicitly configured to compare against it.
        if binding.detector == "clutter_score" and binding.params.get("reference", "baseline") == (
            "baseline"
        ):
            needs_baseline = True
        if not needs_baseline:
            continue

        for anchor_id in anchor_ids:
            anchor = anchors.get(anchor_id)
            if anchor is not None and not anchor.baseline_ref:
                findings.append(
                    Lint(
                        "missing_baseline",
                        f"binding {binding.id!r} compares against a clean baseline, but anchor "
                        f"{anchor_id!r} has none. Capture one while the space is tidy "
                        f"(POST /api/v1/anchors/{anchor_id}/baseline), or set "
                        f"params.reference: none",
                        error=True,
                        binding=binding.id,
                    )
                )


def _check_required_params(
    bindings: Sequence[CompiledBinding],
    registry: DetectorRegistry,
    findings: list[Lint],
) -> None:
    for binding in bindings:
        detector = registry.get(binding.detector)
        if detector is None:
            continue
        for param in detector.params:
            if param.required and param.name not in binding.params:
                findings.append(
                    Lint(
                        "missing_required_param",
                        f"binding {binding.id!r}: detector {detector.name!r} requires "
                        f"params.{param.name} ({param.description})",
                        error=True,
                        binding=binding.id,
                    )
                )


def _numeric_bounds(tree: Condition | None, binding_id: str) -> list[tuple[Op, float]]:
    """Every numeric comparison against one binding in a tree."""
    if tree is None:
        return []
    bounds: list[tuple[Op, float]] = []
    for predicate in iter_predicates(tree):
        if predicate.signal != binding_id:
            continue
        if predicate.op in (_UPWARD | _DOWNWARD) and isinstance(predicate.value, (int, float)):
            bounds.append((predicate.op, float(predicate.value)))
    return bounds


def _check_hysteresis(
    skill: Skill,
    bindings: Mapping[str, CompiledBinding],
    findings: list[Lint],
) -> None:
    """The anti-flap check.

    Trigger at `clutter >= 0.6` and resolve at `clutter <= 0.25` and a surface hovering at 0.5
    produces exactly one task. Resolve at `<= 0.6` instead, and it produces a task every time the
    score wobbles across a single value - the classic way this category of product becomes
    unusable. So: overlapping thresholds are an error, touching thresholds a warning.
    """
    if skill.resolve is None or skill.resolve.conditions is None:
        return

    for binding_id, binding in bindings.items():
        if binding.spec.kind not in _NUMERIC_KINDS:
            continue

        triggers = _numeric_bounds(skill.conditions, binding_id)
        resolves = _numeric_bounds(skill.resolve.conditions, binding_id)
        if not triggers or not resolves:
            continue

        for trigger_op, trigger_value in triggers:
            for resolve_op, resolve_value in resolves:
                rising = trigger_op in _UPWARD and resolve_op in _DOWNWARD
                falling = trigger_op in _DOWNWARD and resolve_op in _UPWARD
                if not (rising or falling):
                    continue

                if rising:
                    overlapping = resolve_value > trigger_value
                    touching = resolve_value == trigger_value
                    advice = (
                        f"resolve at a value below the trigger, e.g. "
                        f"trigger >= {trigger_value}, resolve <= {round(trigger_value * 0.4, 3)}"
                    )
                else:
                    overlapping = resolve_value < trigger_value
                    touching = resolve_value == trigger_value
                    advice = (
                        f"resolve at a value above the trigger, e.g. "
                        f"trigger <= {trigger_value}, resolve >= {round(trigger_value * 1.6, 3)}"
                    )

                if overlapping:
                    findings.append(
                        Lint(
                            "no_hysteresis",
                            f"{binding_id!r}: trigger ({trigger_op.value} {trigger_value}) and "
                            f"resolve ({resolve_op.value} {resolve_value}) ranges overlap, so a "
                            f"task would open and close repeatedly. {advice}",
                            error=True,
                            binding=binding_id,
                        )
                    )
                elif touching:
                    findings.append(
                        Lint(
                            "tight_hysteresis",
                            f"{binding_id!r}: trigger and resolve both use {trigger_value}. Any "
                            f"jitter around that value will flap. {advice}",
                            binding=binding_id,
                        )
                    )


def _advise(skill: Skill, findings: list[Lint]) -> None:
    """Non-blocking advice. These are the mistakes that make a working skill annoying."""
    trigger_predicates = list(iter_predicates(skill.conditions))

    if trigger_predicates and not any(
        p.for_ or p.within or p.absent_for or p.count_over or p.op is Op.CHANGED_TO
        for p in trigger_predicates
    ):
        findings.append(
            Lint(
                "instantaneous_trigger",
                "no predicate uses a temporal qualifier, so a single noisy frame can fire this "
                "skill. Adding `for: 2m` to the main condition removes almost all false positives",
            )
        )

    if isinstance(skill.effect, TaskEffect):
        if skill.limits.cooldown < MIN_SANE_COOLDOWN:
            findings.append(
                Lint(
                    "short_cooldown",
                    f"cooldown is {skill.limits.cooldown}; under "
                    f"{MIN_SANE_COOLDOWN} this skill will feel like nagging",
                )
            )
        if skill.limits.max_per_day is None:
            findings.append(
                Lint(
                    "no_daily_cap",
                    "no max_per_day set. A cap is the cheapest protection against a "
                    "miscalibrated threshold flooding the task list",
                )
            )
        if skill.effect.urgency.bypasses_personality:
            findings.append(
                Lint(
                    "task_at_alert_urgency",
                    f"urgency {skill.effect.urgency.value!r} on a task bypasses the personality "
                    f"layer and reads as a safety message. Consider an alert effect instead",
                )
            )

    if isinstance(skill.effect, MetricEffect) and skill.snapshot.attach:
        findings.append(
            Lint(
                "metric_with_snapshots",
                "metric skills produce no task or alert, so attached snapshots are stored and "
                "never shown. Set snapshot.attach: false to save disk and reduce retained imagery",
            )
        )

    if skill.effective_personality and skill.urgency.bypasses_personality:
        findings.append(
            Lint(
                "personality_will_be_bypassed",
                f"urgency is {skill.urgency.value!r}, so personality "
                f"{skill.effective_personality!r} is bypassed and the wording stays plain. "
                f"This is intentional (ADR-009), noted here so it is not a surprise",
            )
        )

    if skill.resolve is not None and skill.resolve.manual_only and skill.snapshot.attach:
        findings.append(
            Lint(
                "manual_only_with_snapshots",
                "manual_only skills cannot verify completion visually; snapshots are still "
                "attached for context, which may not be what you want",
            )
        )


__all__ = [
    "MIN_SANE_COOLDOWN",
    "CompiledBinding",
    "CompiledSkill",
    "Lint",
    "SkillCompileError",
    "compile_all",
    "compile_skill",
]
