"""The skill-instance state machine.

Each test here corresponds to a way this product could become unpleasant to live with: duplicate
tasks, tasks that close themselves when a camera dies, nagging through the night, or a burner alert
that gets swallowed by a daily cap.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from openhup_schemas import SkillPhase, load_skill_yaml

from openhup.skills.compile import compile_skill
from openhup.skills.evaluate import Verdict
from openhup.skills.fsm import (
    CloseEpisode,
    CreateAlert,
    CreateTask,
    EngineContext,
    ExpireTask,
    InstanceState,
    Notice,
    OpenEpisode,
    RecordMetricEpisode,
    ResolveTask,
    advance,
)

from .conftest import T0, at

CLUTTER_YAML = """
id: kitchen-clutter-buster
watch: [{anchor: kitchen.counter}]
signals:
  - {id: clutter, detector: clutter_score, signal: clutter_level}
conditions: {signal: clutter, op: gte, value: 0.6, for: 15m}
effect: {type: task, mode: single_task_focus, title_hint: clear the counter, urgency: low}
resolve:
  conditions: {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m
limits: {cooldown: 45m, max_per_day: 4}
"""

STOVE_YAML = """
id: stove-burner-safety
watch: [{anchor: kitchen.stove}]
signals:
  - id: burner
    detector: zero_shot_state
    signal: burner_state
    params: {probes: {on: a lit burner, off: an unlit burner}}
conditions: {signal: burner, op: eq, value: "on", for: 10m}
effect: {type: alert, urgency: high, channels: [ntfy]}
resolve: {conditions: {signal: burner, op: eq, value: "off", for: 30s}}
limits: {cooldown: 5m, quiet_hours: {between: ["13:00", "15:00"], tz: UTC}, max_per_day: 1}
"""

TV_YAML = """
id: tv-time
watch: [{anchor: living.tv}]
signals:
  - {id: screen, detector: screen_on, signal: screen_on}
conditions: {signal: screen, op: eq, value: true, for: 2m}
effect: {type: metric, metric: tv_on_minutes_per_day, aggregation: duration_minutes}
resolve: {conditions: {signal: screen, op: eq, value: false, for: 5m}}
snapshot: {attach: false}
"""

YES = Verdict(matched=True)
NO = Verdict(matched=False)
BLIND = Verdict(matched=False, missing=("clutter",))
#: Resolve condition "true" but with no usable data behind it - the dangerous case.
BLIND_YES = Verdict(matched=True, missing=("clutter",))


@pytest.fixture
def clutter(anchors):
    return compile_skill(load_skill_yaml(CLUTTER_YAML), anchors=anchors)


@pytest.fixture
def stove(anchors):
    return compile_skill(load_skill_yaml(STOVE_YAML), anchors=anchors)


@pytest.fixture
def tv(anchors):
    return compile_skill(load_skill_yaml(TV_YAML), anchors=anchors)


def state_for(compiled, phase=SkillPhase.IDLE, **kwargs) -> InstanceState:
    return InstanceState(
        skill_id=compiled.skill.id,
        anchor_id=compiled.anchor_ids[0],
        phase=phase,
        **kwargs,
    )


def ctx(**kwargs) -> EngineContext:
    kwargs.setdefault("now", T0)
    return EngineContext(**kwargs)


def actions_of(decision, kind):
    return [a for a in decision.actions if isinstance(a, kind)]


# ------------------------------------------------------------------ arming and firing


def test_idle_arms_when_data_is_available(clutter) -> None:
    decision = advance(state_for(clutter), clutter, NO, NO, ctx())
    assert decision.state.phase is SkillPhase.ARMED
    assert not decision.actions


def test_armed_trigger_creates_one_task_and_opens_an_episode(clutter) -> None:
    decision = advance(state_for(clutter, SkillPhase.ARMED), clutter, YES, NO, ctx())
    assert decision.state.phase is SkillPhase.ACTING
    assert len(actions_of(decision, CreateTask)) == 1
    assert len(actions_of(decision, OpenEpisode)) == 1
    assert decision.state.episode_id
    task = actions_of(decision, CreateTask)[0]
    expected = f"kitchen-clutter-buster:kitchen.counter:{decision.state.episode_id}"
    assert task.idempotency_key == expected


def test_acting_does_not_create_a_second_task(clutter) -> None:
    """The mess is still there. That is not news, and it is not a second task."""
    state = state_for(clutter, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, clutter, YES, NO, ctx(now=at(minutes=20)))
    assert not actions_of(decision, CreateTask)
    assert decision.state.phase is SkillPhase.ACTING


def test_alert_effect_raises_an_alert_not_a_task(stove) -> None:
    decision = advance(state_for(stove, SkillPhase.ARMED), stove, YES, NO, ctx(now=at(hours=4)))
    assert len(actions_of(decision, CreateAlert)) == 1
    assert not actions_of(decision, CreateTask)


def test_metric_effect_creates_nothing_on_trigger(tv) -> None:
    decision = advance(state_for(tv, SkillPhase.ARMED), tv, YES, NO, ctx())
    assert not actions_of(decision, CreateTask)
    assert not actions_of(decision, CreateAlert)
    assert decision.state.phase is SkillPhase.ACTING


def test_metric_episode_is_recorded_on_resolve(tv) -> None:
    state = state_for(tv, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    decision = advance(state, tv, NO, YES, ctx(now=at(minutes=90)))
    recorded = actions_of(decision, RecordMetricEpisode)
    assert len(recorded) == 1
    assert recorded[0].duration == timedelta(minutes=90)
    assert recorded[0].metric == "tv_on_minutes_per_day"


# ------------------------------------------------------------------ resolution and grace


def test_resolve_enters_grace_then_completes(clutter) -> None:
    state = state_for(clutter, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")

    entering = advance(state, clutter, NO, YES, ctx(now=at(minutes=30)))
    assert entering.state.phase is SkillPhase.RESOLVING
    assert not actions_of(entering, ResolveTask)  # the win is held briefly, deliberately

    still_waiting = advance(entering.state, clutter, NO, YES, ctx(now=at(minutes=32)))
    assert still_waiting.state.phase is SkillPhase.RESOLVING

    done = advance(entering.state, clutter, NO, YES, ctx(now=at(minutes=36)))
    assert done.state.phase is SkillPhase.COOLDOWN
    assert actions_of(done, ResolveTask)[0].auto is True
    assert actions_of(done, CloseEpisode)
    assert done.state.open_task_id is None


def test_mess_returning_during_grace_cancels_completion(clutter) -> None:
    state = replace(
        state_for(clutter, SkillPhase.RESOLVING, episode_id="E1", episode_opened_at=T0),
        resolve_pending_since=at(minutes=30),
        open_task_id="task-1",
    )
    decision = advance(state, clutter, YES, NO, ctx(now=at(minutes=32)))
    assert decision.state.phase is SkillPhase.ACTING
    assert not actions_of(decision, ResolveTask)
    assert decision.state.open_task_id == "task-1"  # same task, not a new one
    assert decision.state.resolve_pending_since is None


def test_zero_grace_resolves_immediately(anchors) -> None:
    skill = load_skill_yaml(CLUTTER_YAML.replace("grace: 5m", "grace: 0s"))
    compiled = compile_skill(skill, anchors=anchors)
    state = state_for(compiled, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, compiled, NO, YES, ctx(now=at(minutes=30)))
    assert decision.state.phase is SkillPhase.COOLDOWN
    assert actions_of(decision, ResolveTask)


def test_auto_expire_closes_an_endless_episode(anchors) -> None:
    skill = load_skill_yaml(
        CLUTTER_YAML.replace("  grace: 5m", "  grace: 5m\n  auto_expire_after: 12h")
    )
    compiled = compile_skill(skill, anchors=anchors)
    state = state_for(compiled, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, compiled, YES, NO, ctx(now=at(hours=13)))
    assert actions_of(decision, ExpireTask)
    assert decision.state.phase is SkillPhase.COOLDOWN


def test_manual_only_skill_never_auto_resolves(anchors) -> None:
    skill = load_skill_yaml(
        CLUTTER_YAML.replace(
            """resolve:
  conditions: {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m""",
            """resolve:
  manual_only: true""",
        )
    )
    compiled = compile_skill(skill, anchors=anchors)
    state = state_for(compiled, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, compiled, NO, YES, ctx(now=at(hours=2)))
    assert decision.state.phase is SkillPhase.ACTING
    assert not actions_of(decision, ResolveTask)


# ------------------------------------------------------------------ the camera-died case


def test_stale_data_never_resolves_a_task(clutter) -> None:
    """The single most important guard in the FSM.

    A resolve verdict can be `matched=True` purely because absence of data satisfies an
    `absent_for`-shaped condition. Acting on that would mean an unplugged camera tidies your house.
    """
    state = state_for(clutter, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, clutter, NO, BLIND_YES, ctx(now=at(minutes=30)))
    assert decision.state.phase is SkillPhase.ACTING
    assert not actions_of(decision, ResolveTask)


def test_armed_with_no_data_goes_stale_and_says_so(clutter) -> None:
    decision = advance(state_for(clutter, SkillPhase.ARMED), clutter, BLIND, BLIND, ctx())
    assert decision.state.phase is SkillPhase.STALE
    notice = actions_of(decision, Notice)[0]
    assert notice.severity == "warning"
    assert "not watching anything" in notice.message


def test_stale_notice_is_not_repeated_every_tick(clutter) -> None:
    first = advance(state_for(clutter, SkillPhase.ARMED), clutter, BLIND, BLIND, ctx())
    second = advance(first.state, clutter, BLIND, BLIND, ctx(now=at(minutes=1)))
    assert not actions_of(second, Notice)


def test_recovering_from_stale_rearms(clutter) -> None:
    stale = advance(state_for(clutter, SkillPhase.ARMED), clutter, BLIND, BLIND, ctx())
    back = advance(stale.state, clutter, NO, NO, ctx(now=at(minutes=5)))
    assert back.state.phase is SkillPhase.ARMED
    assert back.state.stale_notified is False


def test_stale_recovery_can_fire_in_the_same_step(clutter) -> None:
    """Data comes back and the condition is already satisfied: fire, do not wait a tick."""
    stale = advance(state_for(clutter, SkillPhase.ARMED), clutter, BLIND, BLIND, ctx())
    back = advance(stale.state, clutter, YES, NO, ctx(now=at(minutes=5)))
    assert actions_of(back, CreateTask)


# ------------------------------------------------------------------ suppression


def test_cooldown_blocks_retrigger_then_expires(clutter) -> None:
    state = state_for(clutter, SkillPhase.COOLDOWN, last_resolved_at=T0)

    blocked = advance(state, clutter, YES, NO, ctx(now=at(minutes=10)))
    assert blocked.state.phase is SkillPhase.COOLDOWN
    assert not actions_of(blocked, CreateTask)

    released = advance(state, clutter, YES, NO, ctx(now=at(minutes=50)))
    assert released.state.phase is SkillPhase.ARMED

    fires = advance(released.state, clutter, YES, NO, ctx(now=at(minutes=51)))
    assert actions_of(fires, CreateTask)


def test_daily_cap_suppresses_with_an_explanation(clutter) -> None:
    state = state_for(clutter, SkillPhase.ARMED, triggers_today=4, counter_day=T0.date())
    decision = advance(state, clutter, YES, NO, ctx())
    assert not actions_of(decision, CreateTask)
    assert "daily cap of 4 reached" in actions_of(decision, Notice)[0].message


def test_daily_cap_resets_on_a_new_day(clutter) -> None:
    state = state_for(clutter, SkillPhase.ARMED, triggers_today=4, counter_day=T0.date())
    decision = advance(state, clutter, YES, NO, ctx(now=at(days=1)))
    assert actions_of(decision, CreateTask)
    assert decision.state.triggers_today == 1


def test_suppression_notice_fires_once_per_reason(clutter) -> None:
    state = state_for(clutter, SkillPhase.ARMED, triggers_today=4, counter_day=T0.date())
    first = advance(state, clutter, YES, NO, ctx())
    second = advance(first.state, clutter, YES, NO, ctx(now=at(minutes=1)))
    assert actions_of(first, Notice)
    assert not actions_of(second, Notice)


def test_quiet_hours_hold_a_task(anchors) -> None:
    quiet = load_skill_yaml(
        CLUTTER_YAML.replace(
            "limits: {cooldown: 45m, max_per_day: 4}",
            "limits: {cooldown: 45m, max_per_day: 4, "
            'quiet_hours: {between: ["13:00", "15:00"], tz: UTC}}',
        )
    )
    compiled = compile_skill(quiet, anchors=anchors)
    decision = advance(state_for(compiled, SkillPhase.ARMED), compiled, YES, NO, ctx())
    assert not actions_of(decision, CreateTask)
    assert "quiet hours" in actions_of(decision, Notice)[0].message


def test_safety_alerts_ignore_quiet_hours_and_caps(stove) -> None:
    """A burner left on at 2am is exactly when you need to be told.

    The stove skill sets quiet_hours covering T0 *and* max_per_day: 1 with the cap already used.
    Both are overridden because urgency is high.
    """
    state = state_for(stove, SkillPhase.ARMED, triggers_today=1, counter_day=T0.date())
    decision = advance(state, stove, YES, NO, ctx())
    assert len(actions_of(decision, CreateAlert)) == 1


def test_single_task_focus_holds_the_second_task(clutter) -> None:
    """Two messy anchors, one task. The whole point of the ADHD mode."""
    state = state_for(clutter, SkillPhase.ARMED)
    decision = advance(state, clutter, YES, NO, ctx(open_tasks_for_skill=1))
    assert not actions_of(decision, CreateTask)
    assert "single-task focus" in actions_of(decision, Notice)[0].message


def test_backlog_mode_allows_concurrent_tasks(anchors) -> None:
    backlog = load_skill_yaml(CLUTTER_YAML.replace("single_task_focus", "backlog"))
    compiled = compile_skill(backlog, anchors=anchors)
    decision = advance(
        state_for(compiled, SkillPhase.ARMED), compiled, YES, NO, ctx(open_tasks_for_skill=3)
    )
    assert actions_of(decision, CreateTask)


def test_max_open_tasks_caps_backlog_mode(anchors) -> None:
    backlog = load_skill_yaml(
        CLUTTER_YAML.replace("single_task_focus", "backlog").replace(
            "limits: {cooldown: 45m, max_per_day: 4}",
            "limits: {cooldown: 45m, max_per_day: 4, max_open_tasks: 3}",
        )
    )
    compiled = compile_skill(backlog, anchors=anchors)
    decision = advance(
        state_for(compiled, SkillPhase.ARMED), compiled, YES, NO, ctx(open_tasks_for_skill=3)
    )
    assert not actions_of(decision, CreateTask)
    assert "max_open_tasks" in actions_of(decision, Notice)[0].message


# ------------------------------------------------------------------ disabling and pausing


def test_disabling_a_skill_expires_its_open_task(anchors) -> None:
    disabled = load_skill_yaml(CLUTTER_YAML.replace("watch:", "enabled: false\nwatch:"))
    compiled = compile_skill(disabled, anchors=anchors)
    state = state_for(compiled, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, compiled, YES, NO, ctx())
    assert decision.state.phase is SkillPhase.DISABLED
    assert actions_of(decision, ExpireTask)[0].reason == "skill disabled"


def test_global_pause_stops_everything(clutter) -> None:
    state = state_for(clutter, SkillPhase.ACTING, episode_id="E1", episode_opened_at=T0)
    state = state.attach_task("task-1")
    decision = advance(state, clutter, YES, NO, ctx(globally_paused=True))
    assert decision.state.phase is SkillPhase.DISABLED
    assert actions_of(decision, ExpireTask)[0].reason == "monitoring paused"


def test_disabled_skill_stays_quiet(anchors) -> None:
    disabled = load_skill_yaml(CLUTTER_YAML.replace("watch:", "enabled: false\nwatch:"))
    compiled = compile_skill(disabled, anchors=anchors)
    state = state_for(compiled, SkillPhase.DISABLED)
    decision = advance(state, compiled, YES, NO, ctx())
    assert not decision.actions
    assert not decision.changed


def test_reenabling_returns_to_idle(clutter) -> None:
    decision = advance(state_for(clutter, SkillPhase.DISABLED), clutter, NO, NO, ctx())
    assert decision.state.phase is SkillPhase.ARMED
