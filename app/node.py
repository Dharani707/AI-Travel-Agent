"""API routes wrapping the LangGraph orchestrator.

Single APIRouter hosting all endpoints plus the root health check.
"""
from fastapi import APIRouter, HTTPException

from core import orchestrator

from app.schemas import (
    HealthResponse,
    PlanCreatedResponse,
    PlanStatusResponse,
    ReviewDecisionSchema,
    ReviewResponse,
    FinalPlanResponse,
    TravelRequestSchema,
)

node = APIRouter()


@node.get("/", response_model=HealthResponse)
def health_check():
    """Liveness probe. 200 with API name/status if the service is up."""
    return HealthResponse(name="AI Travel Planner", status="ok")


@node.post("/plan", response_model=PlanCreatedResponse)
def create_plan(body: TravelRequestSchema):
    """Start a planning session. 200 with plan_id/status on success, 422 on invalid input, 500 on internal failure."""
    try:
        result = orchestrator.start_planning_session(body.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to start planning session (internal/agent error).",
        ) from exc
    return PlanCreatedResponse(**result)


@node.get("/plan/{plan_id}", response_model=PlanStatusResponse)
def get_plan(plan_id: str):
    """Fetch current status/stage of a planning session. 200 if found, 404 if plan_id doesn't exist, 500 on internal failure."""
    try:
        state = orchestrator.get_session_state(plan_id)
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Plan '{plan_id}' not found."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve planning session state."
        ) from exc
    return PlanStatusResponse(**state)


@node.post("/plan/{plan_id}/review", response_model=ReviewResponse)
def review_plan(plan_id: str, body: ReviewDecisionSchema):
    """Submit a human review decision. 200 on success, 404 if plan_id doesn't exist, 409 if not reviewable, 422 on invalid input, 500 on internal failure."""
    try:
        state = orchestrator.get_session_state(plan_id)
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Plan '{plan_id}' not found."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve planning session state."
        ) from exc

    if state["status"] == "approved" or not state["pending_review"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Plan '{plan_id}' is not in a reviewable state "
                f"(current status: {state['status']})."
            ),
        )

    try:
        result = orchestrator.submit_review(plan_id, body.decision, body.feedback)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to process review decision."
        ) from exc
    return ReviewResponse(**result)


@node.get("/plan/{plan_id}/final", response_model=FinalPlanResponse)
def get_final_plan(plan_id: str):
    """Return the final approved plan. 200 if approved, 404 if plan_id doesn't exist, 409 if not yet approved, 500 on internal failure."""
    try:
        state = orchestrator.get_session_state(plan_id)
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Plan '{plan_id}' not found."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve planning session state."
        ) from exc

    if state["status"] != "approved":
        raise HTTPException(
            status_code=409,
            detail=f"Plan not yet approved. Current status: {state['status']}",
        )
    return FinalPlanResponse(final_output=state["final_output"])