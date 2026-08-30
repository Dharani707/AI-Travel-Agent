"""Pydantic request/response models for the AI Travel Planner API."""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TravelRequestSchema(BaseModel):
    """Client input for creating a new planning session."""

    destination: str = Field(min_length=1, description="Destination city or region.")
    start_date: str = Field(description="Trip start date (ISO format YYYY-MM-DD).")
    end_date: str = Field(description="Trip end date (ISO format YYYY-MM-DD).")
    budget: str = Field(min_length=1, description="Budget description or range.")
    num_travelers: int = Field(gt=0, description="Number of travelers (must be > 0).")
    interests: list[str] = Field(description="Traveler interests used for research.")

    @field_validator("destination", "budget")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_date_range(self):
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except ValueError:
            raise ValueError(
                "start_date and end_date must be valid ISO dates (YYYY-MM-DD)."
            )
        if end < start:
            raise ValueError("end_date must be on or after start_date.")
        return self


class ReviewDecisionSchema(BaseModel):
    """Human-in-the-loop review decision submitted against a draft itinerary."""

    decision: Literal["approve", "reject", "modify"]
    feedback: Optional[str] = None

    @model_validator(mode="after")
    def _feedback_required_when_not_approving(self):
        if self.decision in ("reject", "modify") and not (self.feedback or "").strip():
            raise ValueError(
                "feedback is required when decision is 'reject' or 'modify'."
            )
        return self


class PlanCreatedResponse(BaseModel):
    """Response returned by POST /plan (mirrors start_planning_session)."""

    plan_id: str
    status: str
    draft_itinerary: Optional[dict] = None
    final_output: Optional[dict] = None


class PlanStatusResponse(BaseModel):
    """Response returned by GET /plan/{id} (mirrors get_session_state)."""

    plan_id: str
    status: str
    draft_itinerary: Optional[dict] = None
    final_output: Optional[dict] = None
    pending_review: bool


class ReviewResponse(BaseModel):
    """Response returned by POST /plan/{id}/review (mirrors submit_review)."""

    plan_id: str
    status: str
    draft_itinerary: Optional[dict] = None
    final_output: Optional[dict] = None


class FinalPlanResponse(BaseModel):
    """Response returned by GET /plan/{id}/final (final output only)."""

    final_output: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response returned by GET /."""

    name: str
    status: str