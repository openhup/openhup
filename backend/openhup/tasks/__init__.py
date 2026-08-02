"""Task and alert execution: FSM actions → rows, wording, snapshots, notifications."""

from .engine import MAX_REOPENS, Executor, is_single_focus

__all__ = ["MAX_REOPENS", "Executor", "is_single_focus"]
