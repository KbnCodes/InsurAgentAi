"""Day 5: synthetic incident-intake state machine.

Demo only: no real customer data, claims submission, Saudi regulatory guidance,
or live human transfer. State is ephemeral and scoped only by a shared demo token.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from enum import Enum
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter(prefix="/api/agent/v3", tags=["Day 5: Intake Workflow"])
SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 1000
_lock = threading.Lock()
_sessions: dict[str, dict] = {}


class WorkflowState(str, Enum):
    COLLECTING = "collecting"
    ESCALATED = "escalated"
    DRAFT_READY = "draft_ready"


class IntakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    injury_status: Literal["yes", "no", "unknown"] | None = None
    incident_type: Literal["collision", "parking_damage", "other"] | None = None
    city: str | None = Field(default=None, min_length=2, max_length=60)
    request_human: bool = False
    vehicle_count: int | None = Field(default=None, ge=1, le=10)
    
    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean = value.strip()
        if len(clean) < 2:
            raise ValueError("City must contain at least 2 non-space characters")
        return clean


class IntakeResponse(BaseModel):
    session_id: str
    state: WorkflowState
    message: str
    next_field: str | None
    collected: dict[str, str]
    requires_human: bool
    draft_id: str | None = None
    source: str = "synthetic_intake"


def require_demo_token(token: str | None) -> None:
    expected = os.getenv("DEMO_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Demo token is not configured")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid demo token")


def _new_session() -> tuple[str, dict]:
    session_id = str(uuid.uuid4())
    state = {
        "state": WorkflowState.COLLECTING,
        "collected": {},
        "draft_id": None,
        "expires_at": time.monotonic() + SESSION_TTL_SECONDS,
    }
    _sessions[session_id] = state
    return session_id, state


def _get_session(session_id: str | None) -> tuple[str, dict]:
    now = time.monotonic()
    for key in list(_sessions):
        if _sessions[key]["expires_at"] <= now:
            del _sessions[key]

    if session_id is None:
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(status_code=503, detail="Demo session capacity reached")
        return _new_session()

    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    state["expires_at"] = now + SESSION_TTL_SECONDS
    return session_id, state


def _render(session_id: str, state: dict) -> IntakeResponse:
    status = state["state"]
    collected = dict(state["collected"])

    if status == WorkflowState.ESCALATED:
        return IntakeResponse(
            session_id=session_id,
            state=status,
            message=(
                "Human assistance is required. This demo does not contact a representative. "
                "If anyone is in immediate danger, contact local emergency services. "
                "Do not rely on this demo for accident procedures or coverage decisions."
            ),
            next_field=None,
            collected=collected,
            requires_human=True,
        )

    if status == WorkflowState.DRAFT_READY:
        return IntakeResponse(
            session_id=session_id,
            state=status,
            message=(
                "Synthetic intake draft prepared for human review. "
                "Nothing has been submitted to an insurer or authority."
            ),
            next_field=None,
            collected=collected,
            requires_human=True,
            draft_id=state["draft_id"],
        )

    questions = {
        "injury_status": "For this demo, has anyone been injured? (yes / no / unknown)",
        "incident_type": "Which demo incident type applies? (collision / parking_damage / other)",
        "city": "Which city should be recorded in this synthetic scenario?",
    }
    next_field = next(key for key in questions if key not in collected)
    return IntakeResponse(
        session_id=session_id,
        state=status,
        message=questions[next_field],
        next_field=next_field,
        collected=collected,
        requires_human=False,
    )


@router.post("/intake", response_model=IntakeResponse)
def intake(
    request: IntakeRequest,
    x_demo_token: str | None = Header(default=None, alias="X-Demo-Token"),
):
    require_demo_token(x_demo_token)
    # No network/LLM calls inside this critical section.
    with _lock:
        session_id, state = _get_session(request.session_id)

        if state["state"] != WorkflowState.COLLECTING:
            return _render(session_id, state)

        if request.request_human or request.injury_status in ("yes", "unknown"):
            state["state"] = WorkflowState.ESCALATED
            return _render(session_id, state)

        if request.incident_type == "other":
            state["state"] = WorkflowState.ESCALATED
            return _render(session_id, state)

        # Whitelisted fields only; never persist free-form customer messages.
        for field in ("injury_status", "incident_type", "city"):
            value = getattr(request, field)
            if value is not None:
                state["collected"][field] = value

        if all(key in state["collected"] for key in ("injury_status", "incident_type", "city")):
            state["state"] = WorkflowState.DRAFT_READY
            state["draft_id"] = f"DRAFT-{uuid.uuid4().hex[:10].upper()}"

        return _render(session_id, state)


@router.get("/intake/{session_id}", response_model=IntakeResponse)
def get_intake(
    session_id: str,
    x_demo_token: str | None = Header(default=None, alias="X-Demo-Token"),
):
    require_demo_token(x_demo_token)
    with _lock:
        actual_id, state = _get_session(session_id)
        return _render(actual_id, state)
