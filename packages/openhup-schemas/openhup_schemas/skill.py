"""The Skill schema: user intent, made executable.

A Skill answers five questions:

    watch       which anchors
    signals     which detector outputs, bound to short local names
    conditions  when does this fire (a boolean tree with temporal operators)
    effect      what happens - task, alert, or metric
    resolve     when is it over (deliberately a *separate* condition, so trigger and resolve
                thresholds can differ; that asymmetry is what stops tasks from flapping)

Natural language is a front door, not the format: `POST /skills/parse` produces one of these for
the user to confirm. Nothing auto-arms a skill the user has not seen (ADR-008).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from .common import (
    Duration,
    EffectType,
    Ident,
    MicroStepStrategy,
    Op,
    RedactionTarget,
    Slug,
    SnapshotMode,
    StrEnum,
    TaskMode,
    TimeWindow,
    Urgency,
)


class _Base(BaseModel):
    """Strict by default: an unknown key in a skill file is a typo, and typos must fail loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------------------
# Watching and signal binding
# --------------------------------------------------------------------------------------


class WatchSpec(_Base):
    """What a skill watches. Exactly one of `anchor` or `camera`.

    `camera` is a convenience that expands to every anchor on that camera at compile time.
    """

    anchor: Slug | None = None
    camera: Slug | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if bool(self.anchor) == bool(self.camera):
            raise ValueError("watch entry needs exactly one of 'anchor' or 'camera'")
        return self

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"anchor:{self.anchor}" if self.anchor else f"camera:{self.camera}/*"


class SignalBinding(_Base):
    """Bind a detector output to a short name that conditions can reference.

    The indirection is what lets you swap `object_inventory` from YOLOX to RT-DETR, or move a
    signal from a camera detector to a physical sensor, without editing any condition.
    """

    id: Ident
    detector: Ident
    signal: Ident
    #: Passed through to the detector (sensitivity, reference mode, class filters, text probes).
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------


class CountOver(_Base):
    """N rising edges within a window: "the bowl went empty 3 times today".

    Counts *edges*, not samples - otherwise a 5-second sampling interval would satisfy
    "3 times" in fifteen seconds.
    """

    window: Duration
    n: int = Field(ge=1)


class SignalPredicate(_Base):
    """A comparison against one bound signal, optionally qualified by a temporal operator.

    At most one temporal qualifier may be set. With none, the predicate tests the latest sample.
    """

    signal: Ident
    op: Op
    value: float | int | bool | str | None = None

    #: Must hold continuously for this long. Needs the engine's timer tick, since it can become
    #: true while no new observations arrive.
    for_: Duration | None = Field(default=None, alias="for", serialization_alias="for")
    #: Must have held at least once in this trailing window.
    within: Duration | None = None
    #: Must NOT have held at any point in this trailing window. Note that a total absence of
    #: observations satisfies this - that is intentional, and how "no one has been in the
    #: kitchen for 9 minutes" is expressed.
    absent_for: Duration | None = None
    count_over: CountOver | None = None

    #: Longest tolerated gap between samples inside a `for` run. A run interrupted by a longer
    #: silence is not "continuous", so a camera that dropped out cannot satisfy `for: 10m`.
    max_gap: Duration | None = None
    #: Ignore samples the detector was unsure about.
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _one_temporal_qualifier(self) -> Self:
        set_qualifiers = [
            name
            for name, value in (
                ("for", self.for_),
                ("within", self.within),
                ("absent_for", self.absent_for),
                ("count_over", self.count_over),
            )
            if value is not None
        ]
        if len(set_qualifiers) > 1:
            raise ValueError(
                f"predicate on {self.signal!r} sets multiple temporal qualifiers "
                f"({', '.join(set_qualifiers)}); use one, or nest predicates under all/any"
            )
        if self.op is not Op.CHANGED_TO and self.value is None:
            raise ValueError(f"predicate on {self.signal!r} with op {self.op} requires a value")
        if self.max_gap is not None and self.for_ is None:
            raise ValueError("max_gap only applies together with 'for'")
        return self

    @property
    def window(self) -> timedelta:
        """How far back this predicate needs history. Drives ring-buffer retention."""
        if self.for_ is not None:
            return self.for_
        if self.within is not None:
            return self.within
        if self.absent_for is not None:
            return self.absent_for
        if self.count_over is not None:
            return self.count_over.window
        return timedelta(0)

    def describe(self) -> str:
        """Human-readable form, used verbatim in task/alert explanations and the UI."""
        from .common import format_duration

        base = f"{self.signal} {self.op.value} {self.value!r}"
        if self.for_ is not None:
            return f"{base} for {format_duration(self.for_)}"
        if self.within is not None:
            return f"{base} within {format_duration(self.within)}"
        if self.absent_for is not None:
            return f"not ({base}) for {format_duration(self.absent_for)}"
        if self.count_over is not None:
            return (
                f"{base} at least {self.count_over.n}x in {format_duration(self.count_over.window)}"
            )
        return base


class TimeWindowCondition(_Base):
    """Gate a skill to certain hours or weekdays."""

    time_window: TimeWindow

    def describe(self) -> str:
        return f"time in {self.time_window}"


class AllOf(_Base):
    all_: list[Condition] = Field(alias="all", serialization_alias="all", min_length=1)


class AnyOf(_Base):
    any_: list[Condition] = Field(alias="any", serialization_alias="any", min_length=1)


class NotOf(_Base):
    not_: Condition = Field(alias="not", serialization_alias="not")


def _coerce_condition(value: Any) -> Any:
    """Accept a bare YAML list as an implicit `all`.

    Writing

        conditions:
          - {signal: clutter, op: gte, value: 0.6, for: 15m}
          - {time_window: {between: ["07:00", "22:00"]}}

    is what people naturally do, and it means exactly the same thing as wrapping it in `all:`.
    Rejecting it would be pedantry.
    """
    if isinstance(value, (list, tuple)):
        return {"all": list(value)}
    return value


#: Left-to-right union: the container shapes are tried first, and `extra="forbid"` on every
#: member means a node can only match one variant. Avoids needing a discriminator tag in YAML,
#: which would make skill files noisier for no benefit.
Condition = Annotated[
    AllOf | AnyOf | NotOf | TimeWindowCondition | SignalPredicate,
    Field(union_mode="left_to_right"),
    BeforeValidator(_coerce_condition),
]


def iter_conditions(node: Condition) -> Iterator[Condition]:
    """Depth-first walk over a condition tree, including the root."""
    yield node
    if isinstance(node, AllOf):
        for child in node.all_:
            yield from iter_conditions(child)
    elif isinstance(node, AnyOf):
        for child in node.any_:
            yield from iter_conditions(child)
    elif isinstance(node, NotOf):
        yield from iter_conditions(node.not_)


def iter_predicates(node: Condition) -> Iterator[SignalPredicate]:
    """Every signal predicate in a tree. Used to validate bindings and size ring buffers."""
    for child in iter_conditions(node):
        if isinstance(child, SignalPredicate):
            yield child


def condition_horizon(node: Condition) -> timedelta:
    """Longest history any predicate in the tree needs."""
    return max((p.window for p in iter_predicates(node)), default=timedelta(0))


# --------------------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------------------


class MicroStepSpec(_Base):
    """How to break one overwhelming anchor into a ladder of small steps.

    Accepts three shorthands in YAML, because this field is written by hand constantly:

        micro_steps: auto:3               → strategy auto, 3 steps
        micro_steps: none                 → no laddering
        micro_steps: ["cups", "mail"]     → explicit steps
    """

    strategy: MicroStepStrategy = MicroStepStrategy.AUTO
    count: int = Field(default=3, ge=1, le=10)
    steps: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            text = data.strip().lower()
            if text in {"none", "false", "off"}:
                return {"strategy": MicroStepStrategy.NONE, "count": 1}
            if text.startswith("auto"):
                _, _, count = text.partition(":")
                return {"strategy": MicroStepStrategy.AUTO, "count": int(count or 3)}
            raise ValueError(f"unrecognised micro_steps shorthand {data!r}")
        if isinstance(data, bool):
            return {"strategy": MicroStepStrategy.AUTO if data else MicroStepStrategy.NONE}
        if isinstance(data, list):
            return {
                "strategy": MicroStepStrategy.EXPLICIT,
                "count": max(1, len(data)),
                "steps": list(data),
            }
        return data

    @model_validator(mode="after")
    def _explicit_needs_steps(self) -> Self:
        if self.strategy is MicroStepStrategy.EXPLICIT and not self.steps:
            raise ValueError("micro_steps strategy 'explicit' requires a non-empty steps list")
        return self

    @property
    def enabled(self) -> bool:
        return self.strategy is not MicroStepStrategy.NONE


class TaskEffect(_Base):
    type: Literal[EffectType.TASK] = EffectType.TASK
    mode: TaskMode = TaskMode.SINGLE_TASK_FOCUS
    #: Plain-language seed for the wording. The personality layer rewrites it; if the LLM is
    #: unavailable this is what the user sees, so write it as a usable sentence.
    title_hint: str = Field(min_length=1, max_length=200)
    micro_steps: MicroStepSpec = Field(default_factory=MicroStepSpec)
    urgency: Urgency = Urgency.LOW
    personality: Slug | None = None
    channels: list[Slug] = Field(default_factory=list)
    #: Ask before adding to the list, for skills the user is still learning to trust.
    require_confirmation: bool = False
    #: Quietly expire instead of accumulating guilt. None means never expire.
    expires_after: Duration | None = None


class AlertEffect(_Base):
    type: Literal[EffectType.ALERT] = EffectType.ALERT
    #: Alerts default to HIGH, which means the personality layer is bypassed and the wording
    #: stays factual (ADR-009).
    urgency: Urgency = Urgency.HIGH
    title_hint: str | None = Field(default=None, max_length=200)
    personality: Slug | None = None
    channels: list[Slug] = Field(default_factory=list)
    #: Re-notify while unresolved and unacknowledged. None = notify once.
    repeat_every: Duration | None = None
    requires_ack: bool = True


class MetricAggregation(StrEnum):
    #: Number of trigger→resolve episodes in the bucket.
    EPISODES = "episodes"
    #: Total time the condition held, in minutes. TV time, stove time.
    DURATION_MINUTES = "duration_minutes"
    #: Time since the last episode, in hours. Clean streaks.
    TIME_SINCE_LAST_HOURS = "time_since_last_hours"
    #: Mean of the underlying signal over the bucket.
    MEAN = "mean"
    MAX = "max"


class MetricEffect(_Base):
    """Track something instead of nagging about it. No task, no notification."""

    type: Literal[EffectType.METRIC] = EffectType.METRIC
    metric: Ident
    aggregation: MetricAggregation = MetricAggregation.EPISODES
    unit: str | None = None
    bucket: Duration = Field(default_factory=lambda: timedelta(days=1))


Effect = Annotated[TaskEffect | AlertEffect | MetricEffect, Field(discriminator="type")]


# --------------------------------------------------------------------------------------
# Resolution, limits, snapshots
# --------------------------------------------------------------------------------------


class ResolveSpec(_Base):
    """When does the episode end?

    Keeping this separate from `conditions` is the whole anti-flap mechanism: trigger at
    clutter >= 0.6, resolve at <= 0.25, and a surface hovering around 0.5 produces exactly one
    task instead of forty.
    """

    conditions: Condition | None = None
    #: Keep the completed task visible briefly so the win registers. Small but load-bearing UX.
    grace: Duration = Field(default_factory=lambda: timedelta(0))
    #: On manual completion, request a fresh observation and reopen once if it disagrees.
    #: Only ever once - arguing with the user twice is a bug, not a feature.
    verify_on_manual_complete: bool = True
    #: Episodes that never resolve visually (someone unplugged the camera) shouldn't hang forever.
    auto_expire_after: Duration | None = None
    #: True for skills whose completion cannot be seen ("call the dentist").
    manual_only: bool = False

    @model_validator(mode="after")
    def _need_a_way_out(self) -> Self:
        if self.conditions is None and not self.manual_only and self.auto_expire_after is None:
            raise ValueError(
                "resolve needs conditions, manual_only: true, or auto_expire_after - "
                "otherwise the episode can never end"
            )
        return self


class Limits(_Base):
    """Anti-nag controls. These are not optional extras; see ARCHITECTURE.md section 5."""

    #: After an episode resolves, refuse to re-trigger for this long.
    cooldown: Duration = Field(default_factory=lambda: timedelta(minutes=30))
    max_per_day: int | None = Field(default=None, ge=1)
    quiet_hours: TimeWindow | None = None
    #: No observations for this long → phase STALE and a system notice. A dead camera must not
    #: be indistinguishable from a tidy house.
    staleness_timeout: Duration = Field(default_factory=lambda: timedelta(minutes=15))
    #: Cap concurrent open tasks for this skill (ignored in single_task_focus, which caps at 1).
    max_open_tasks: int | None = Field(default=None, ge=1)


class SnapshotSpec(_Base):
    attach: bool = True
    mode: SnapshotMode = SnapshotMode.FULL
    retention: Duration = Field(default_factory=lambda: timedelta(days=7))
    #: Applied *before* the JPEG is written, so unredacted pixels never touch the disk.
    redact: list[RedactionTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ephemeral_is_not_attachable(self) -> Self:
        if self.mode is SnapshotMode.EPHEMERAL and self.attach:
            raise ValueError(
                "snapshot mode 'ephemeral' cannot be attached (nothing is persisted); "
                "set attach: false or choose thumbnail/full"
            )
        return self


class SkillOrigin(StrEnum):
    USER = "user"
    LLM = "llm"
    PRESET = "preset"
    IMPORT = "import"


# --------------------------------------------------------------------------------------
# Skill
# --------------------------------------------------------------------------------------


class Skill(_Base):
    """A complete, executable skill definition."""

    id: Slug
    version: int = Field(default=1, ge=1)
    enabled: bool = True
    description: str = ""

    watch: list[WatchSpec] = Field(min_length=1)
    signals: list[SignalBinding] = Field(min_length=1)
    conditions: Condition
    effect: Effect
    resolve: ResolveSpec | None = None
    limits: Limits = Field(default_factory=Limits)
    snapshot: SnapshotSpec = Field(default_factory=SnapshotSpec)

    #: Skill-level default; `effect.personality` wins if both are set.
    personality: Slug | None = None
    tags: list[str] = Field(default_factory=list)
    origin: SkillOrigin = SkillOrigin.USER
    #: The sentence the user originally typed, kept for display and re-parsing.
    source_text: str | None = None

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        binding_ids = [b.id for b in self.signals]
        duplicates = {b for b in binding_ids if binding_ids.count(b) > 1}
        if duplicates:
            raise ValueError(f"duplicate signal binding id(s): {sorted(duplicates)}")

        known = set(binding_ids)
        trees: list[Condition] = [self.conditions]
        if self.resolve is not None and self.resolve.conditions is not None:
            trees.append(self.resolve.conditions)

        for tree in trees:
            for predicate in iter_predicates(tree):
                if predicate.signal not in known:
                    raise ValueError(
                        f"condition references undeclared signal {predicate.signal!r}; "
                        f"declared: {sorted(known)}"
                    )
        return self

    @model_validator(mode="after")
    def _default_resolve(self) -> Self:
        """Metric skills need no resolution; task and alert skills must have one."""
        if self.resolve is None:
            if isinstance(self.effect, MetricEffect):
                return self
            raise ValueError(
                f"skill {self.id!r} produces a {self.effect.type} and therefore needs a "
                f"'resolve' block (or resolve.manual_only: true)"
            )
        return self

    # -- derived properties -------------------------------------------------------------

    @property
    def effective_personality(self) -> str | None:
        return getattr(self.effect, "personality", None) or self.personality

    @property
    def urgency(self) -> Urgency:
        return getattr(self.effect, "urgency", Urgency.INFO)

    @property
    def horizon(self) -> timedelta:
        """History depth this skill needs across both condition trees."""
        windows = [condition_horizon(self.conditions)]
        if self.resolve is not None and self.resolve.conditions is not None:
            windows.append(condition_horizon(self.resolve.conditions))
        return max(windows)

    def binding(self, binding_id: str) -> SignalBinding:
        for candidate in self.signals:
            if candidate.id == binding_id:
                return candidate
        raise KeyError(f"skill {self.id!r} has no signal binding {binding_id!r}")

    def to_yaml_dict(self) -> dict[str, Any]:
        """Serialisable form matching the on-disk YAML (aliases, no nulls)."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


AllOf.model_rebuild()
AnyOf.model_rebuild()
NotOf.model_rebuild()
Skill.model_rebuild()


def load_skill_yaml(text: str) -> Skill:
    """Parse one skill from YAML. Requires the optional `yaml` extra."""
    import yaml  # imported lazily: the vision service does not need a YAML parser

    return Skill.model_validate(yaml.safe_load(text))


def load_skills_yaml(text: str) -> list[Skill]:
    """Parse a multi-document YAML file (``---`` separated) into skills."""
    import yaml

    return [Skill.model_validate(doc) for doc in yaml.safe_load_all(text) if doc]


__all__ = [
    "AlertEffect",
    "AllOf",
    "AnyOf",
    "Condition",
    "CountOver",
    "Effect",
    "Limits",
    "MetricAggregation",
    "MetricEffect",
    "MicroStepSpec",
    "NotOf",
    "ResolveSpec",
    "SignalBinding",
    "SignalPredicate",
    "Skill",
    "SkillOrigin",
    "SnapshotSpec",
    "TaskEffect",
    "TimeWindowCondition",
    "WatchSpec",
    "condition_horizon",
    "iter_conditions",
    "iter_predicates",
    "load_skill_yaml",
    "load_skills_yaml",
]
