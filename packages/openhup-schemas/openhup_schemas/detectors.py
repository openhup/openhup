"""The detector registry: what the vision service can measure.

This is a *contract*, not an implementation. The vision service publishes it via
`GET /api/v1/detectors`; the backend uses it to reject nonsensical skills at save time (a `contains`
operator against a scalar signal, a baseline-dependent detector on an anchor with no baseline); and
the frontend's skill builder renders its inputs from it, so adding a detector makes it selectable in
the UI without any frontend change.

`BUILTIN_DETECTORS` below is the canonical list that `vision-service` implements. Keeping it here
rather than in the vision package means the backend can validate skills with no CV dependencies
installed - which is also what makes the whole test suite runnable without ONNX Runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import OPS_BY_KIND, Duration, Ident, Op, SignalKind, StrEnum


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DetectorCost(StrEnum):
    """Rough compute cost, used by the scheduler to stagger expensive detectors."""

    #: Pixel math only, no model. Runs at frame rate on a Pi.
    TRIVIAL = "trivial"
    #: One small CNN pass.
    LOW = "low"
    #: A detection transformer or CLIP pass.
    MEDIUM = "medium"
    #: Open-vocabulary detection or a VLM. Seconds per frame on CPU.
    HIGH = "high"


class ParamSpec(_Base):
    """One tunable knob, described well enough for the UI to render a control for it."""

    name: Ident
    type: str = Field(description="float | int | bool | str | list[str] | dict[str,str]")
    description: str = ""
    default: object | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] | None = None
    required: bool = False


class SignalSpec(_Base):
    key: Ident
    kind: SignalKind
    description: str = ""
    unit: str | None = None
    #: For enum signals: the complete value set, so the skill builder can offer a dropdown.
    enum_values: list[str] | None = None
    range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _enum_values_only_for_enums(self) -> Self:
        if self.enum_values and self.kind is not SignalKind.ENUM:
            raise ValueError(f"signal {self.key!r}: enum_values only apply to enum signals")
        return self

    def allows(self, op: Op) -> bool:
        return op in OPS_BY_KIND[self.kind]


class DetectorSpec(_Base):
    """One vision capability."""

    name: Ident
    title: str
    description: str = ""
    signals: list[SignalSpec] = Field(default_factory=list)
    params: list[ParamSpec] = Field(default_factory=list)
    cost: DetectorCost = DetectorCost.LOW

    #: Some detectors emit a signal whose *name* the user chooses, because the thing being watched
    #: is arbitrary: `zero_shot_state` with probes for on/off becomes `burner_state` on one anchor
    #: and `curtains_state` on another. For these, any valid identifier is an acceptable signal
    #: key and `dynamic_kind` gives its kind.
    dynamic: bool = False
    dynamic_kind: SignalKind | None = None

    #: Needs the anchor's "clean" reference image. Compile fails with a clear message if missing.
    requires_baseline: bool = False
    #: Off by default: extra weights to download, or a heavier licence/compute commitment.
    optional: bool = False
    #: Suggested minimum interval between runs. The scheduler may go slower, never faster.
    default_interval: Duration = Field(default_factory=lambda: timedelta(seconds=30))
    #: Model ids from models/registry.yaml.
    models: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _dynamic_needs_kind(self) -> Self:
        if self.dynamic and self.dynamic_kind is None:
            raise ValueError(f"detector {self.name!r} is dynamic and must declare dynamic_kind")
        if not self.dynamic and not self.signals:
            raise ValueError(f"detector {self.name!r} declares no signals and is not dynamic")
        return self

    def signal(self, key: str) -> SignalSpec | None:
        """Resolve a signal key, synthesising a spec for dynamic detectors."""
        for spec in self.signals:
            if spec.key == key:
                return spec
        if self.dynamic and self.dynamic_kind is not None:
            return SignalSpec(
                key=key,
                kind=self.dynamic_kind,
                description=f"user-defined {self.dynamic_kind} signal from {self.name}",
            )
        return None


class DetectorRegistry(_Base):
    """All detectors known to a deployment."""

    detectors: list[DetectorSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [d.name for d in self.detectors]
        if len(set(names)) != len(names):
            raise ValueError("duplicate detector names in registry")
        return self

    def get(self, name: str) -> DetectorSpec | None:
        return next((d for d in self.detectors if d.name == name), None)

    def resolve(self, detector: str, signal: str) -> SignalSpec | None:
        spec = self.get(detector)
        return spec.signal(signal) if spec else None

    def names(self) -> list[str]:
        return [d.name for d in self.detectors]

    def __iter__(self) -> Iterator[DetectorSpec]:  # type: ignore[override]
        return iter(self.detectors)


# --------------------------------------------------------------------------------------
# Built-ins
# --------------------------------------------------------------------------------------

_S = SignalSpec

BUILTIN_DETECTORS = DetectorRegistry(
    detectors=[
        DetectorSpec(
            name="object_inventory",
            title="Object inventory",
            description=(
                "Closed-set object detection inside the anchor. The workhorse: it answers "
                "'what is on this surface and how much of it is there'."
            ),
            cost=DetectorCost.MEDIUM,
            models=["yolox-s", "dfine-s"],
            default_interval=timedelta(seconds=20),
            signals=[
                _S(key="objects", kind=SignalKind.SET, description="Distinct labels present."),
                _S(key="object_count", kind=SignalKind.COUNT, description="Total detections."),
                _S(
                    key="person_count",
                    kind=SignalKind.COUNT,
                    description="People in the anchor. Used for presence gating on safety skills.",
                ),
                _S(
                    key="object_area_fraction",
                    kind=SignalKind.SCALAR,
                    description="Fraction of the anchor covered by movable objects.",
                    range=(0.0, 1.0),
                ),
                _S(key="boxes", kind=SignalKind.BBOX_LIST, description="Raw boxes, for the UI."),
            ],
            params=[
                ParamSpec(
                    name="classes",
                    type="list[str]",
                    description="Restrict to these COCO labels. Empty means all.",
                    default=[],
                ),
                ParamSpec(
                    name="min_score",
                    type="float",
                    description="Detection confidence floor.",
                    default=0.35,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
        ),
        DetectorSpec(
            name="clutter_score",
            title="Clutter score",
            description=(
                "How messy is this surface, 0..1. Fuses three independent signals - baseline "
                "difference, object density, and a CLIP tidy/cluttered probe - because no single "
                "method survives real rooms (ADR-005). The three components ride along on every "
                "observation so the UI can explain the number."
            ),
            cost=DetectorCost.MEDIUM,
            models=["clip-vit-b32", "yolox-s"],
            requires_baseline=False,
            default_interval=timedelta(seconds=30),
            signals=[
                _S(
                    key="clutter_level",
                    kind=SignalKind.SCALAR,
                    description="Fused clutter score. 0 is the baseline's tidiness, 1 is chaos.",
                    range=(0.0, 1.0),
                ),
                _S(
                    key="baseline_diff",
                    kind=SignalKind.SCALAR,
                    description="Component: distance from the anchor's clean reference.",
                    range=(0.0, 1.0),
                ),
                _S(
                    key="object_density",
                    kind=SignalKind.SCALAR,
                    description="Component: movable-object coverage.",
                    range=(0.0, 1.0),
                ),
                _S(
                    key="semantic_clutter",
                    kind=SignalKind.SCALAR,
                    description="Component: zero-shot tidy-vs-cluttered probe.",
                    range=(0.0, 1.0),
                ),
            ],
            params=[
                ParamSpec(
                    name="reference",
                    type="str",
                    description=(
                        "'baseline' compares against the anchor's clean reference; 'none' uses "
                        "only density and semantics (for anchors with no stable clean state)."
                    ),
                    default="baseline",
                    choices=["baseline", "none"],
                ),
                ParamSpec(
                    name="sensitivity",
                    type="float",
                    description="Remaps the output curve. 0.5 neutral, higher is touchier.",
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
        ),
        DetectorSpec(
            name="zero_shot_state",
            title="Zero-shot state",
            description=(
                "Classify the anchor into one of several user-described states using CLIP text "
                "probes. This is how 'burner on vs off', 'curtains open vs closed', or 'bowl full "
                "vs empty' work without training anything."
            ),
            cost=DetectorCost.MEDIUM,
            models=["clip-vit-b32"],
            dynamic=True,
            dynamic_kind=SignalKind.ENUM,
            default_interval=timedelta(seconds=15),
            params=[
                ParamSpec(
                    name="probes",
                    type="dict[str,str]",
                    description=(
                        "Map of state name → text description, e.g. "
                        "{on: 'a lit gas stove burner', off: 'an unlit stove burner'}."
                    ),
                    required=True,
                ),
                ParamSpec(
                    name="min_margin",
                    type="float",
                    description=(
                        "Minimum probability gap between the top two states; below it the signal "
                        "reports 'unknown' rather than guessing."
                    ),
                    default=0.08,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            notes="Always include an 'unknown' path in skills: CLIP will be unsure sometimes.",
        ),
        DetectorSpec(
            name="presence_absence",
            title="Presence / absence",
            description=(
                "Is a specific named thing in the anchor? Open-vocabulary, so it covers nouns "
                "COCO never heard of: dish rack, pet bowl, recycling bin."
            ),
            cost=DetectorCost.MEDIUM,
            models=["clip-vit-b32"],
            dynamic=True,
            dynamic_kind=SignalKind.BOOLEAN,
            default_interval=timedelta(minutes=2),
            params=[
                ParamSpec(
                    name="query",
                    type="str",
                    description="The thing to look for, in plain words: 'dish drying rack'.",
                    required=True,
                ),
                ParamSpec(
                    name="min_score",
                    type="float",
                    default=0.25,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            notes="Expensive. Give it a long interval; presence rarely changes second to second.",
        ),
        DetectorSpec(
            name="fill_level",
            title="Fill level",
            description=(
                "Estimate how full a container is, 0..1. Trash bins, laundry baskets, pet bowls, "
                "the sink."
            ),
            cost=DetectorCost.MEDIUM,
            models=["clip-vit-b32"],
            default_interval=timedelta(minutes=1),
            signals=[
                _S(
                    key="fill_level",
                    kind=SignalKind.SCALAR,
                    description="0 empty, 1 overflowing.",
                    range=(0.0, 1.0),
                ),
                _S(
                    key="overflowing",
                    kind=SignalKind.BOOLEAN,
                    description="True when contents breach the container's rim.",
                ),
            ],
            params=[
                ParamSpec(
                    name="container",
                    type="str",
                    description="What kind of container, in words: 'kitchen trash can'.",
                    required=True,
                )
            ],
        ),
        DetectorSpec(
            name="screen_on",
            title="Screen on",
            description=(
                "Is a TV or monitor displaying something? Brightness and temporal variance in the "
                "screen region - cheap, and it does not need to see what is playing."
            ),
            cost=DetectorCost.TRIVIAL,
            default_interval=timedelta(seconds=30),
            signals=[
                _S(key="screen_on", kind=SignalKind.BOOLEAN, description="Screen is active."),
                _S(
                    key="screen_activity",
                    kind=SignalKind.SCALAR,
                    description=(
                        "Temporal variance; distinguishes playing video from a paused menu."
                    ),
                    range=(0.0, 1.0),
                ),
            ],
            notes=(
                "Deliberately cannot tell what is on screen. Content classification was left out "
                "on purpose - see docs/SECURITY_PRIVACY.md."
            ),
        ),
        DetectorSpec(
            name="door_state",
            title="Door / drawer state",
            description=(
                "Open, closed, or ajar, from CLIP probes against the door region. Reports "
                "`unknown` when the probes are ambiguous."
            ),
            cost=DetectorCost.MEDIUM,
            models=["clip-vit-b32"],
            default_interval=timedelta(seconds=20),
            signals=[
                _S(
                    key="door_state",
                    kind=SignalKind.ENUM,
                    enum_values=["open", "ajar", "closed", "unknown"],
                    description="Door position.",
                )
            ],
        ),
        DetectorSpec(
            name="walkway_clear",
            title="Walkway clear",
            description=(
                "Is the floor path through this anchor unobstructed? Distinct from clutter: this "
                "cares only about the traversable region, which is what matters for trip hazards."
            ),
            cost=DetectorCost.MEDIUM,
            models=["yolox-s"],
            default_interval=timedelta(minutes=1),
            signals=[
                _S(key="walkway_clear", kind=SignalKind.BOOLEAN, description="Path is clear."),
                _S(
                    key="obstruction_area",
                    kind=SignalKind.SCALAR,
                    description="Fraction of the path covered by objects.",
                    range=(0.0, 1.0),
                ),
            ],
        ),
        DetectorSpec(
            name="pose_fall",
            title="Fall detection",
            description=(
                "Pose-based detection of a person who is down and has not moved. Opt-in, and "
                "explicitly not a medical device - see the caveats in docs/CONFIGURATION.md."
            ),
            cost=DetectorCost.HIGH,
            models=["rtmpose-s", "movenet-thunder"],
            optional=True,
            default_interval=timedelta(seconds=5),
            signals=[
                _S(key="person_down", kind=SignalKind.BOOLEAN, description="A person is prone."),
                _S(
                    key="person_down_seconds",
                    kind=SignalKind.SCALAR,
                    description="How long the prone pose has persisted.",
                    unit="s",
                ),
                _S(
                    key="motion_level",
                    kind=SignalKind.SCALAR,
                    description="Movement of the prone person; near zero is the concerning case.",
                    range=(0.0, 1.0),
                ),
            ],
            notes=(
                "Pair with `for: 30s` and a presence check. Never wire this to a personality "
                "other than plain."
            ),
        ),
        DetectorSpec(
            name="face_id",
            title="Face identity (consent-gated)",
            description=(
                "Who is present, only ever for people who said yes (ADR-016). An unknown face is "
                "reported without a name and triggers the consent flow; an enrolled face is "
                "reported as a member id. Names never leave the backend - the vision service only "
                "ever sees ids and embeddings."
            ),
            cost=DetectorCost.HIGH,
            models=["yunet-face", "mobilefacenet"],
            optional=True,
            default_interval=timedelta(seconds=15),
            signals=[
                _S(
                    key="known_members",
                    kind=SignalKind.SET,
                    description=(
                        "Ids of enrolled members whose face matched. Names live only in the "
                        "backend."
                    ),
                ),
                _S(
                    key="unknown_face",
                    kind=SignalKind.BOOLEAN,
                    description="A face is present that no enrolled member matches.",
                ),
                _S(key="face_count", kind=SignalKind.COUNT, description="Faces detected."),
            ],
            params=[
                ParamSpec(
                    name="min_score",
                    type="float",
                    description="Face detection confidence floor.",
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ParamSpec(
                    name="match_threshold",
                    type="float",
                    description="Cosine similarity floor for a gallery match.",
                    default=0.55,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            notes=(
                "Consent-gated by design: the gallery holds only enrolled members, an unknown "
                "face never persists, and the 24-hour no-reask marker stores no biometric data. "
                "Never used to trigger a task or an alert - identity annotates presence, it does "
                "not accuse (ADR-016)."
            ),
        ),
        DetectorSpec(
            name="sensor",
            title="External sensor",
            description=(
                "Values pushed in from MQTT, Zigbee2MQTT, or Home Assistant, normalised into the "
                "same pipeline. A lid contact switch answers 'is the trash open' far more cheaply "
                "than a camera, and skills should not have to care which one answered."
            ),
            cost=DetectorCost.TRIVIAL,
            dynamic=True,
            dynamic_kind=SignalKind.SCALAR,
            default_interval=timedelta(seconds=1),
            params=[
                ParamSpec(
                    name="topic",
                    type="str",
                    description="MQTT topic or HA entity id to bind.",
                    required=True,
                ),
                ParamSpec(
                    name="kind",
                    type="str",
                    description="Override the signal kind: scalar | count | boolean | enum | set.",
                    default="scalar",
                    choices=["scalar", "count", "boolean", "enum", "set"],
                ),
            ],
        ),
    ]
)

__all__ = [
    "BUILTIN_DETECTORS",
    "DetectorCost",
    "DetectorRegistry",
    "DetectorSpec",
    "ParamSpec",
    "SignalSpec",
]
