"""API v1.

/skills       CRUD, natural-language parse, simulate, FSM state
/cameras      cameras, anchors, baselines, snapshots
/vision/plan  detector schedules derived from enabled skills
/detectors    the detector registry the skill builder renders from
/tasks        list, next (single-task focus), complete/snooze/dismiss
/alerts       history and acknowledgement
/metrics      series, goals, weekly report
/personalities  tune tone, preview output
/notify       channels and held notifications
/memory       what the assistant has been told: list, add, delete
/members      who consented to be remembered (ADR-016): list, enroll, forget
/system       info, health, LLM audit trail
/ws/events    live event stream
"""

from fastapi import APIRouter

from . import cameras, insights, members, memory, skills, tasks, voice, ws

router = APIRouter()
router.include_router(skills.router)
router.include_router(cameras.router)
router.include_router(tasks.router)
router.include_router(insights.router)
router.include_router(voice.router)
router.include_router(memory.router)
router.include_router(members.router)
router.include_router(ws.router)

__all__ = ["router"]
