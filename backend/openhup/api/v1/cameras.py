"""Cameras, anchors, snapshots, and the vision plan.

The plan endpoint is the interesting one: it derives, from the currently enabled skills, which
detectors should run on which anchors and how often. That inversion - the backend telling the
vision service what to look at, rather than the vision service deciding - is what makes "disable a
skill and its CPU cost disappears" literally true (ADR-003).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from openhup_schemas import (
    BUILTIN_DETECTORS,
    Anchor,
    Camera,
    Topic,
    VisionCommand,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import AnchorRow, CameraRow, MemberRow, ObservationRow, get_session
from ..state import AppState

router = APIRouter(tags=["cameras"])
UTC = UTC

Session = Annotated[AsyncSession, Depends(get_session)]


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


async def _identity_gallery(session: AsyncSession) -> list[dict[str, Any]]:
    """Enrolled members as (id, embedding) for the vision service to match against.

    Names never leave the backend - the vision service matches faces to ids and emits ids, and
    the backend is the only place an id becomes a name. The gallery is the entire biometric
    surface that ever crosses a service boundary, and it is exactly the consenting members, never
    an unknown face (ADR-016).
    """
    rows = (await session.execute(select(MemberRow).where(MemberRow.active))).scalars().all()
    return [{"id": row.id, "embedding": row.embedding} for row in rows]


class CameraOut(BaseModel):
    id: str
    name: str
    enabled: bool
    kind: str
    #: Never includes credentials - only the name of the env var holding the password.
    password_env: str | None = None
    max_fps: float
    anchors: list[str] = Field(default_factory=list)
    last_frame_at: datetime | None = None
    online: bool = False
    last_error: str | None = None


@router.get("/cameras", response_model=list[CameraOut])
async def list_cameras(request: Request, session: Session) -> list[CameraOut]:
    state = state_of(request)
    rows = (await session.execute(select(CameraRow))).scalars().all()
    out = []
    for row in rows:
        camera = Camera.model_validate(row.config)
        out.append(
            CameraOut(
                id=camera.id,
                name=camera.name,
                enabled=camera.enabled,
                kind=camera.kind.value,
                password_env=camera.password_env,
                max_fps=camera.max_fps,
                anchors=[a.id for a in state.anchors_for_camera(camera.id)],
                last_frame_at=row.last_frame_at,
                online=bool(
                    row.last_frame_at
                    and datetime.now(tz=UTC) - row.last_frame_at < timedelta(minutes=2)
                ),
                last_error=row.last_error,
            )
        )
    return out


@router.post("/cameras", status_code=201)
async def create_camera(
    request: Request, session: Session, payload: dict[str, Any] = Body(...)
) -> dict[str, str]:
    state = state_of(request)
    camera = Camera.model_validate(payload)
    if await session.get(CameraRow, camera.id):
        raise HTTPException(409, f"camera {camera.id!r} already exists")
    session.add(
        CameraRow(
            id=camera.id,
            name=camera.name,
            enabled=camera.enabled,
            kind=camera.kind.value,
            config=camera.model_dump(mode="json"),
        )
    )
    await session.flush()
    state.cameras[camera.id] = camera
    state.recompile()
    return {"created": camera.id}


@router.patch("/cameras/{camera_id}")
async def update_camera(
    camera_id: str, request: Request, session: Session, payload: dict[str, Any] = Body(...)
) -> dict[str, str]:
    state = state_of(request)
    row = await session.get(CameraRow, camera_id)
    if row is None:
        raise HTTPException(404, "no such camera")
    camera = Camera.model_validate({**row.config, **payload, "id": camera_id})
    row.config = camera.model_dump(mode="json")
    row.name = camera.name
    row.enabled = camera.enabled
    row.kind = camera.kind.value
    await session.flush()
    state.cameras[camera_id] = camera
    state.recompile()
    await _command(state, VisionCommand(action="reload_plan", camera_id=camera_id))
    return {"updated": camera_id}


@router.delete("/cameras/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, request: Request, session: Session) -> None:
    """Delete a camera. Its anchors survive with a null camera_id.

    This is ADR-010 in practice: replacing hardware must not destroy the history of the places it
    watched. Re-point the anchors at the new camera and every streak and metric carries on.
    """
    state = state_of(request)
    row = await session.get(CameraRow, camera_id)
    if row is None:
        raise HTTPException(404, "no such camera")
    await session.delete(row)
    await session.flush()
    state.cameras.pop(camera_id, None)
    state.recompile()


class AnchorOut(BaseModel):
    id: str
    camera_id: str | None
    label: str
    enabled: bool
    has_baseline: bool
    baseline_captured_at: datetime | None = None
    subregions: list[str] = Field(default_factory=list)
    sensitivity: float = 0.5
    watching_skills: list[str] = Field(default_factory=list)
    #: True when a skill needs a baseline here and none has been captured. Surfaced prominently in
    #: the UI, because it is the most common reason a new install appears to do nothing.
    needs_baseline: bool = False


@router.get("/anchors", response_model=list[AnchorOut])
async def list_anchors(request: Request, camera: str | None = None) -> list[AnchorOut]:
    state = state_of(request)
    anchors = state.anchors.values()
    if camera:
        anchors = [a for a in anchors if a.camera_id == camera]

    out = []
    for anchor in sorted(anchors, key=lambda a: a.id):
        watching = state.skills_watching(anchor.id)
        needs_baseline = not anchor.baseline_ref and any(
            binding.detector == "clutter_score"
            and binding.params.get("reference", "baseline") == "baseline"
            for compiled in watching
            for binding in compiled.bindings
        )
        out.append(
            AnchorOut(
                id=anchor.id,
                camera_id=anchor.camera_id,
                label=anchor.label,
                enabled=anchor.enabled,
                has_baseline=bool(anchor.baseline_ref),
                baseline_captured_at=anchor.baseline_captured_at,
                subregions=[s.id for s in anchor.ordered_subregions()],
                sensitivity=anchor.sensitivity,
                watching_skills=[c.skill.id for c in watching],
                needs_baseline=needs_baseline,
            )
        )
    return out


@router.post("/anchors", status_code=201)
async def create_anchor(
    request: Request, session: Session, payload: dict[str, Any] = Body(...)
) -> dict[str, str]:
    state = state_of(request)
    anchor = Anchor.model_validate(payload)
    if await session.get(AnchorRow, anchor.id):
        raise HTTPException(409, f"anchor {anchor.id!r} already exists")
    session.add(
        AnchorRow(
            id=anchor.id,
            camera_id=anchor.camera_id,
            label=anchor.label,
            enabled=anchor.enabled,
            config=anchor.model_dump(mode="json"),
            baseline_ref=anchor.baseline_ref,
        )
    )
    await session.flush()
    state.anchors[anchor.id] = anchor
    state.recompile()
    return {"created": anchor.id}


@router.patch("/anchors/{anchor_id}")
async def update_anchor(
    anchor_id: str, request: Request, session: Session, payload: dict[str, Any] = Body(...)
) -> dict[str, str]:
    state = state_of(request)
    row = await session.get(AnchorRow, anchor_id)
    if row is None:
        raise HTTPException(404, "no such anchor")
    anchor = Anchor.model_validate({**row.config, **payload, "id": anchor_id})
    row.config = anchor.model_dump(mode="json")
    row.label = anchor.label
    row.enabled = anchor.enabled
    row.camera_id = anchor.camera_id
    await session.flush()
    state.anchors[anchor_id] = anchor
    state.recompile()
    await _command(state, VisionCommand(action="reload_plan", anchor_id=anchor_id))
    return {"updated": anchor_id}


@router.post("/anchors/{anchor_id}/baseline")
async def capture_baseline(anchor_id: str, request: Request, session: Session) -> dict[str, str]:
    """Capture the "this is what clean looks like" reference.

    Call it while the space is genuinely tidy: every clutter score is measured against this image,
    so a baseline captured mid-mess makes the anchor permanently and confusingly relaxed.
    """
    state = state_of(request)
    row = await session.get(AnchorRow, anchor_id)
    if row is None:
        raise HTTPException(404, "no such anchor")

    await _command(state, VisionCommand(action="capture_baseline", anchor_id=anchor_id))
    reference = f"snap://baseline/{anchor_id}.jpg"
    row.baseline_ref = reference
    row.baseline_captured_at = datetime.now(tz=UTC)
    row.config = {
        **row.config,
        "baseline_ref": reference,
        "baseline_captured_at": row.baseline_captured_at.isoformat(),
    }
    await session.flush()
    state.anchors[anchor_id] = Anchor.model_validate(row.config)
    return {"anchor_id": anchor_id, "baseline_ref": reference, "status": "requested"}


@router.get("/snapshots/{path:path}")
async def get_snapshot(path: str, request: Request) -> Response:
    """Serve a snapshot.

    Deliberately proxied through the API rather than exposed as a static directory, so that
    authentication actually applies to imagery of the inside of someone's home. Path traversal is
    checked explicitly.
    """
    state = state_of(request)
    root = state.settings.snapshot_dir.resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):
        raise HTTPException(400, "invalid snapshot path")
    if not target.is_file():
        raise HTTPException(404, "snapshot not found or already expired")
    return FileResponse(target, media_type="image/jpeg")


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------


@router.get("/vision/plan")
async def vision_plan(
    request: Request, session: Session, node: str | None = None
) -> dict[str, Any]:
    """Per-anchor detector schedules, derived from currently enabled skills.

    Merging rules, all of which exist because several skills may share one anchor:

    * detector params are merged across skills; the *touchiest* value wins for sensitivity, so one
      skill cannot make another blind.
    * the interval is the shortest any skill needs.
    * `wanted_signals` is the union, so signals nobody reads are never published.
    * the snapshot policy is the strictest, never the most permissive.
    """
    state = state_of(request)
    anchors: list[dict[str, Any]] = []

    for anchor_id, anchor in sorted(state.anchors.items()):
        if not anchor.enabled:
            continue
        watching = state.skills_watching(anchor_id)
        by_detector: dict[str, dict[str, Any]] = {}

        for compiled in watching:
            for binding in compiled.bindings:
                spec = BUILTIN_DETECTORS.get(binding.detector)
                entry = by_detector.setdefault(
                    binding.detector,
                    {
                        "detector": binding.detector,
                        "params": {},
                        "min_interval": (spec.default_interval if spec else timedelta(seconds=30)),
                        "wanted_signals": set(),
                    },
                )
                for key, value in binding.params.items():
                    if key == "sensitivity" and key in entry["params"]:
                        entry["params"][key] = max(entry["params"][key], value)
                    else:
                        entry["params"].setdefault(key, value)
                if spec is not None and spec.dynamic:
                    # Dynamic detectors emit a signal whose name the user chose in the binding.
                    # Tell the detector which key to publish so its output matches wanted_signals.
                    entry["params"]["emit_as"] = binding.signal
                entry["wanted_signals"].add(binding.signal)

        # Per-anchor overrides beat the detector default, but only to slow things down or speed
        # them up deliberately - never to disable a detector that a skill needs.
        for override in anchor.detector_overrides:
            if override.detector in by_detector and override.min_interval:
                by_detector[override.detector]["min_interval"] = override.min_interval
            if override.detector in by_detector:
                by_detector[override.detector]["params"].update(override.params)

        # Identity is ambient context, not a skill: when the household opts in (ADR-016), the
        # face_id detector runs on every enabled anchor and the gallery rides the plan. The
        # gallery holds ids and embeddings only - names never leave the backend.
        if state.settings.identity.enabled:
            entry = by_detector.setdefault(
                "face_id",
                {
                    "detector": "face_id",
                    "params": {"match_threshold": state.settings.identity.match_threshold},
                    "min_interval": state.settings.identity.min_interval,
                    "wanted_signals": {"known_members", "unknown_face", "face_count"},
                },
            )

        policies = [c.skill.snapshot for c in watching]
        attach = bool(policies) and all(p.attach for p in policies)
        order = {"ephemeral": 0, "thumbnail": 1, "full": 2, "archive": 3}
        mode = min((p.mode.value for p in policies), key=lambda m: order[m]) if policies else "full"
        retention = min((p.retention for p in policies), default=timedelta(days=7))
        redact = sorted({t.value for p in policies for t in p.redact})

        anchors.append(
            {
                "anchor_id": anchor_id,
                "camera_id": anchor.camera_id or "",
                "label": anchor.label,
                "detectors": [
                    {
                        "detector": entry["detector"],
                        "params": entry["params"],
                        "min_interval": _duration(entry["min_interval"]),
                        "wanted_signals": sorted(entry["wanted_signals"]),
                    }
                    for entry in by_detector.values()
                ],
                "snapshot_attach": attach,
                "snapshot_mode": mode,
                "snapshot_retention": _duration(
                    min(retention, state.settings.snapshots.max_retention)
                ),
                "snapshot_redact": redact,
                # Only score subregions when a skill actually ladders on this anchor - it is three
                # extra inferences per pass and pointless otherwise.
                "score_subregions": bool(anchor.subregions)
                and any(
                    getattr(c.skill.effect, "micro_steps", None)
                    and c.skill.effect.micro_steps.enabled
                    for c in watching
                ),
            }
        )

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "revision": state.plan_revision,
        "anchors": anchors,
        "members": await _identity_gallery(session),
    }


@router.get("/detectors")
async def list_detectors() -> dict[str, Any]:
    """The detector registry.

    The skill builder renders its inputs from this, so adding a detector makes it selectable in the
    UI with no frontend change.
    """
    return {
        "detectors": [
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "cost": spec.cost.value,
                "optional": spec.optional,
                "requires_baseline": spec.requires_baseline,
                "dynamic": spec.dynamic,
                "dynamic_kind": spec.dynamic_kind.value if spec.dynamic_kind else None,
                "default_interval_s": spec.default_interval.total_seconds(),
                "models": spec.models,
                "notes": spec.notes,
                "signals": [
                    {
                        "key": signal.key,
                        "kind": signal.kind.value,
                        "description": signal.description,
                        "unit": signal.unit,
                        "enum_values": signal.enum_values,
                        "range": signal.range,
                    }
                    for signal in spec.signals
                ],
                "params": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "description": param.description,
                        "default": param.default,
                        "required": param.required,
                        "choices": param.choices,
                        "minimum": param.minimum,
                        "maximum": param.maximum,
                    }
                    for param in spec.params
                ],
            }
            for spec in BUILTIN_DETECTORS.detectors
        ]
    }


@router.get("/observations")
async def list_observations(
    session: Session,
    anchor: str | None = None,
    detector: str | None = None,
    minutes: int = Query(default=60, ge=1, le=60 * 24 * 14),
    limit: int = Query(default=200, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Raw observation feed, for debugging and for the calibration view."""
    since = datetime.now(tz=UTC) - timedelta(minutes=minutes)
    query = (
        select(ObservationRow)
        .where(ObservationRow.ts >= since)
        .order_by(ObservationRow.ts.desc())
        .limit(limit)
    )
    if anchor:
        query = query.where(ObservationRow.anchor_id == anchor)
    if detector:
        query = query.where(ObservationRow.detector == detector)
    rows = (await session.execute(query)).scalars().all()
    return [
        {
            "id": row.id,
            "ts": row.ts.isoformat(),
            "anchor_id": row.anchor_id,
            "detector": row.detector,
            "signals": row.signals,
            "snapshot_ref": row.snapshot_ref,
            "cost_ms": row.cost_ms,
        }
        for row in rows
    ]


async def _command(state: AppState, command: VisionCommand) -> None:
    if not state.bus.connected:
        return
    await state.bus._redis.xadd(
        Topic.VISION_COMMANDS.value,
        {"payload": command.model_dump_json()},
        maxlen=1000,
        approximate=True,
    )


def _duration(value: timedelta) -> str:
    from openhup_schemas import format_duration

    return format_duration(value)


def _snapshot_root(state: AppState) -> Path:
    return state.settings.snapshot_dir


__all__ = ["router"]
