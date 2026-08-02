"""Primitives shared by every OpenHup service: durations, IDs, time windows, enums.

Nothing in this module imports from the rest of the package, so it can be used from the
vision service without pulling in skill or task models.
"""

from __future__ import annotations

import re
import secrets
import time as _time
from datetime import UTC, datetime, time, timedelta
from enum import Enum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    model_validator,
)

UTC = UTC

# --------------------------------------------------------------------------------------
# Durations
# --------------------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r"^(?:(?P<d>\d+)d)?(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+)s)?(?:(?P<ms>\d+)ms)?$"
)


def parse_duration(value: Any) -> timedelta:
    """Parse a human duration into a timedelta.

    Accepts ``"15m"``, ``"1h30m"``, ``"2d"``, ``"500ms"``, a bare number of seconds, or an
    existing timedelta. Rejects empty and unit-less strings so a typo like ``for: 15`` in YAML
    is caught at load time rather than silently meaning 15 seconds.
    """
    if isinstance(value, timedelta):
        return value
    if isinstance(value, bool):  # bool is an int subclass; almost certainly a mistake
        raise ValueError("duration cannot be a boolean")
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    if not isinstance(value, str):
        raise TypeError(f"cannot parse duration from {type(value).__name__}")

    text = value.strip().replace(" ", "").lower()
    if not text:
        raise ValueError("duration is empty")
    # "ms" must be tried before "m" would swallow it; the regex handles ordering already,
    # but a lone number with no unit is ambiguous and therefore an error.
    if text.isdigit():
        raise ValueError(f"duration {value!r} has no unit (use 30s, 15m, 2h, 1d)")

    match = _DURATION_RE.fullmatch(text)
    if not match or not any(match.groupdict().values()):
        raise ValueError(f"invalid duration {value!r} (expected forms like 30s, 15m, 1h30m, 2d)")

    parts = {k: int(v) for k, v in match.groupdict().items() if v is not None}
    return timedelta(
        days=parts.get("d", 0),
        hours=parts.get("h", 0),
        minutes=parts.get("m", 0),
        seconds=parts.get("s", 0),
        milliseconds=parts.get("ms", 0),
    )


def format_duration(value: timedelta) -> str:
    """Render a timedelta back into the compact form, so YAML round-trips unchanged."""
    total_ms = round(value.total_seconds() * 1000)
    if total_ms == 0:
        return "0s"
    sign = "-" if total_ms < 0 else ""
    total_ms = abs(total_ms)

    days, rem = divmod(total_ms, 86_400_000)
    hours, rem = divmod(rem, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)

    out = ""
    for amount, unit in ((days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s")):
        if amount:
            out += f"{amount}{unit}"
    if millis:
        out += f"{millis}ms"
    return sign + out


Duration = Annotated[
    timedelta,
    BeforeValidator(parse_duration),
    PlainSerializer(format_duration, return_type=str, when_used="unless-none"),
]

# --------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

ULID = Annotated[str, StringConstraints(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")]

#: User-authored identifiers: lowercase, dot/dash/underscore separated. Used for skill ids,
#: anchor ids ("kitchen.counter"), camera ids, personality ids.
Slug = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$", strip_whitespace=True),
]

#: Signal keys and detector names as emitted by the vision service.
Ident = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$", strip_whitespace=True)]


def new_ulid(ts_ms: int | None = None, randomness: int | None = None) -> str:
    """Generate a ULID: 48-bit millisecond timestamp + 80 bits of randomness, Crockford base32.

    Lexicographic order matches creation order, which is why these are used for episode ids and
    database keys. Arguments exist purely so tests can pin the output.
    """
    stamp = int(_time.time() * 1000) if ts_ms is None else ts_ms
    if not 0 <= stamp < (1 << 48):
        raise ValueError("ULID timestamp out of range")
    rand = secrets.randbits(80) if randomness is None else randomness
    n = (stamp << 80) | (rand & ((1 << 80) - 1))

    chars = [""] * 26
    for i in range(25, -1, -1):
        chars[i] = _CROCKFORD[n & 0x1F]
        n >>= 5
    return "".join(chars)


def ulid_timestamp(value: str) -> datetime:
    """Recover the creation time encoded in a ULID."""
    n = 0
    for char in value:
        index = _CROCKFORD.find(char)
        if index < 0:
            raise ValueError(f"invalid ULID character {char!r}")
        n = (n << 5) | index
    return datetime.fromtimestamp((n >> 80) / 1000, tz=UTC)


def utcnow() -> datetime:
    """Timezone-aware UTC now. The engine never calls this directly - `now` is always injected."""
    return datetime.now(tz=UTC)


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class StrEnum(str, Enum):
    """str-valued enum that serialises as its value in JSON and YAML."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class SignalKind(StrEnum):
    """The shape of a signal value, which determines the operators that may be applied."""

    SCALAR = "scalar"  # float, usually normalised 0..1 (clutter_level)
    COUNT = "count"  # non-negative int (person_count, object_count)
    BOOLEAN = "boolean"  # true/false (screen_on)
    ENUM = "enum"  # one of a fixed set of strings (burner_state: on/off/unknown)
    SET = "set"  # set of strings present in the frame (objects)
    BBOX_LIST = "bbox_list"  # list of boxes with labels and scores


class Op(StrEnum):
    """Predicate operators. Which ones are legal depends on the signal's kind."""

    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"
    EQ = "eq"
    NEQ = "neq"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    CHANGED_TO = "changed_to"


#: Which operators are valid for which signal kinds. Enforced at skill compile time so a
#: nonsensical skill fails on save, not at 3am when the condition first evaluates.
OPS_BY_KIND: dict[SignalKind, frozenset[Op]] = {
    SignalKind.SCALAR: frozenset({Op.GTE, Op.LTE, Op.GT, Op.LT, Op.EQ, Op.NEQ}),
    SignalKind.COUNT: frozenset({Op.GTE, Op.LTE, Op.GT, Op.LT, Op.EQ, Op.NEQ}),
    SignalKind.BOOLEAN: frozenset({Op.EQ, Op.NEQ, Op.CHANGED_TO}),
    SignalKind.ENUM: frozenset({Op.EQ, Op.NEQ, Op.CHANGED_TO}),
    SignalKind.SET: frozenset({Op.CONTAINS, Op.NOT_CONTAINS}),
    SignalKind.BBOX_LIST: frozenset({Op.CONTAINS, Op.NOT_CONTAINS}),
}


class Urgency(StrEnum):
    INFO = "info"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _URGENCY_RANK[self]

    @property
    def bypasses_personality(self) -> bool:
        """Safety outranks comedy: high and above are always phrased plainly.

        See ADR-009. This is enforced in code rather than requested in a prompt.
        """
        return self.rank >= Urgency.HIGH.rank


_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.INFO: 0,
    Urgency.LOW: 1,
    Urgency.NORMAL: 2,
    Urgency.HIGH: 3,
    Urgency.CRITICAL: 4,
}


class EffectType(StrEnum):
    TASK = "task"
    ALERT = "alert"
    METRIC = "metric"


class TaskMode(StrEnum):
    #: Only ever one open task per (skill, anchor). The ADHD-friendly default.
    SINGLE_TASK_FOCUS = "single_task_focus"
    #: Every trigger produces a task; the user sees the backlog.
    BACKLOG = "backlog"


class TaskState(StrEnum):
    PROPOSED = "proposed"
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    SNOOZED = "snoozed"
    RESOLVED_AUTO = "resolved_auto"
    RESOLVED_MANUAL = "resolved_manual"
    DISMISSED = "dismissed"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_TASK_STATES

    @property
    def is_resolved(self) -> bool:
        return self in {TaskState.RESOLVED_AUTO, TaskState.RESOLVED_MANUAL}


_TERMINAL_TASK_STATES = frozenset(
    {
        TaskState.RESOLVED_AUTO,
        TaskState.RESOLVED_MANUAL,
        TaskState.DISMISSED,
        TaskState.EXPIRED,
    }
)


class SkillPhase(StrEnum):
    """Skill-instance FSM phase. One row per (skill, anchor). See ARCHITECTURE.md section 5."""

    DISABLED = "disabled"
    IDLE = "idle"
    ARMED = "armed"
    TRIGGERED = "triggered"
    ACTING = "acting"
    RESOLVING = "resolving"
    COOLDOWN = "cooldown"
    #: No fresh observations. Surfaced as a system notice - a dead camera must not look like
    #: a tidy house.
    STALE = "stale"


class TextSource(StrEnum):
    """Where a task or alert's wording came from. Shown in the UI; useful when debugging tone."""

    LLM = "llm"
    TEMPLATE = "template"
    USER = "user"


class RedactionTarget(StrEnum):
    FACES = "faces"
    PEOPLE = "people"
    SCREENS = "screens"
    TEXT = "text"


class SnapshotMode(StrEnum):
    #: Used for detection, never written to disk.
    EPHEMERAL = "ephemeral"
    #: 160px long edge only - enough to recognise the place, not to read a document.
    THUMBNAIL = "thumbnail"
    FULL = "full"
    #: Keep before/after pairs past the normal TTL for progress history.
    ARCHIVE = "archive"


class MicroStepStrategy(StrEnum):
    #: Let the engine pick: spatial when the anchor defines sub-regions, semantic otherwise.
    AUTO = "auto"
    #: Subdivide the ROI polygon and order sub-regions by clutter density. No LLM needed.
    SPATIAL = "spatial"
    #: Ask the LLM to propose steps from the detected object inventory.
    SEMANTIC = "semantic"
    #: Explicit user-authored steps.
    EXPLICIT = "explicit"
    NONE = "none"


# --------------------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------------------

_DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class TimeWindow(BaseModel):
    """A recurring daily window, optionally restricted to certain weekdays.

    Windows may wrap midnight: ``between: ["22:00", "07:00"]`` is the obvious way to write
    quiet hours and is handled correctly.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    start: time
    end: time
    tz: str = Field(
        default="local",
        description="IANA zone name, or 'local' for the host's zone.",
    )
    days: list[str] | None = Field(
        default=None,
        description="Restrict to weekdays, e.g. ['mon','tue','wed','thu','fri'].",
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_between(cls, data: Any) -> Any:
        """Allow the compact ``{between: ["07:00", "22:00"]}`` form used throughout the docs."""
        if isinstance(data, dict) and "between" in data:
            data = dict(data)
            between = data.pop("between")
            if not isinstance(between, (list, tuple)) or len(between) != 2:
                raise ValueError("time_window.between must be a two-element list [start, end]")
            data.setdefault("start", between[0])
            data.setdefault("end", between[1])
        return data

    @model_validator(mode="after")
    def _check_days(self) -> Self:
        if self.days is not None:
            normalised = [d.strip().lower()[:3] for d in self.days]
            bad = [d for d in normalised if d not in _DAY_NAMES]
            if bad:
                raise ValueError(f"unknown weekday(s): {bad}; use {list(_DAY_NAMES)}")
            self.days = normalised
        return self

    @property
    def wraps_midnight(self) -> bool:
        return self.start > self.end

    def tzinfo(self) -> Any:
        if self.tz == "local":
            return datetime.now().astimezone().tzinfo
        from zoneinfo import ZoneInfo

        return ZoneInfo(self.tz)

    def contains(self, moment: datetime) -> bool:
        """Is `moment` inside the window? `moment` must be timezone-aware."""
        if moment.tzinfo is None:
            raise ValueError("TimeWindow.contains requires a timezone-aware datetime")
        local = moment.astimezone(self.tzinfo())

        if self.days is not None and _DAY_NAMES[local.weekday()] not in self.days:
            return False

        current = local.time()
        if self.start == self.end:  # degenerate: treat as always-on
            return True
        if not self.wraps_midnight:
            return self.start <= current < self.end
        # Wrapped window: inside if after the start tonight or before the end this morning.
        return current >= self.start or current < self.end

    def __str__(self) -> str:  # pragma: no cover - display helper
        days = f" on {','.join(self.days)}" if self.days else ""
        return f"{self.start:%H:%M}-{self.end:%H:%M} {self.tz}{days}"


__all__ = [
    "OPS_BY_KIND",
    "ULID",
    "UTC",
    "Duration",
    "EffectType",
    "Ident",
    "MicroStepStrategy",
    "Op",
    "RedactionTarget",
    "SignalKind",
    "SkillPhase",
    "Slug",
    "SnapshotMode",
    "StrEnum",
    "TaskMode",
    "TaskState",
    "TextSource",
    "TimeWindow",
    "Urgency",
    "format_duration",
    "new_ulid",
    "parse_duration",
    "ulid_timestamp",
    "utcnow",
]
