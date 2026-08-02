"""The skill engine.

Layered so the interesting logic is pure and the I/O is thin:

    window.py      bounded, time-evicting signal history
    operators.py   temporal operators (for / within / absent_for / count_over / changed_to)
    evaluate.py    pure condition evaluation → Verdict (matched + reasons + data health)
    compile.py     Skill → CompiledSkill, plus the lints that block bad skills
    fsm.py         pure per-instance state machine → Decision (next state + actions)
    parse.py       natural language → draft Skill, via the LLM, schema-validated
    simulate.py    replay stored observations against a draft skill

`evaluate.advance` and `fsm.advance` take `now` as a parameter and perform no I/O, which is why the
awkward cases (grace periods, daily caps, dropped cameras, clock edges) are unit-testable.
"""

from .compile import CompiledSkill, Lint, SkillCompileError, compile_all, compile_skill
from .evaluate import Reason, Verdict, evaluate, evaluate_both
from .fsm import Decision, EngineContext, InstanceState, advance
from .window import BindingWindows, Sample, SignalWindow, WindowStore

__all__ = [
    "BindingWindows",
    "CompiledSkill",
    "Decision",
    "EngineContext",
    "InstanceState",
    "Lint",
    "Reason",
    "Sample",
    "SignalWindow",
    "SkillCompileError",
    "Verdict",
    "WindowStore",
    "advance",
    "compile_all",
    "compile_skill",
    "evaluate",
    "evaluate_both",
]
