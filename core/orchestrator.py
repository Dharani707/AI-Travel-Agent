from typing import Any, TypedDict, Optional
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt

from core.agents.research_agent import run_research_agent
from core.agents.planner_agent import run_planner_agent


class TravelState(TypedDict, total=False):
    travel_request: dict
    research_data: dict
    draft_itinerary: dict
    review_status: str
    feedback: Optional[str]
    final_output: Optional[dict]


DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
DECISION_MODIFY = "modify"

VALID_DECISIONS = {DECISION_APPROVE, DECISION_REJECT, DECISION_MODIFY}

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_MODIFIED = "modified"


def research_node(state: TravelState) -> dict:
    req = state["travel_request"]
    destination = req["destination"]
    start_date = req["start_date"]
    end_date = req["end_date"]
    interests = req.get("interests", [])
    research_data = run_research_agent(destination, start_date, end_date, interests)
    return {"research_data": research_data}


def planner_node(state: TravelState) -> dict:
    research_data = state["research_data"]
    travel_request = state["travel_request"]
    feedback = state.get("feedback")

    revision_request = dict(travel_request)
    if state.get("draft_itinerary"):
        revision_request["previous_itinerary"] = state["draft_itinerary"]

    draft_itinerary = run_planner_agent(
        research_data=research_data,
        travel_request=revision_request,
        feedback=feedback,
    )
    return {"draft_itinerary": draft_itinerary, "final_output": None}


def hitl_node(state: TravelState) -> dict:
    decision = interrupt({
        "draft_itinerary": state["draft_itinerary"],
        "prompt": "Review the draft itinerary. Reply with decision 'approve', 'reject', or 'modify', and optional free-text feedback.",
    })
    decision_str = str(decision.get("decision", "")).strip().lower()
    feedback = decision.get("feedback")

    if decision_str == DECISION_APPROVE:
        status = STATUS_APPROVED
    elif decision_str == DECISION_REJECT:
        status = STATUS_REJECTED
    elif decision_str == DECISION_MODIFY:
        status = STATUS_MODIFIED
    else:
        raise ValueError(
            f"Invalid review decision '{decision_str}'. Must be one of {sorted(VALID_DECISIONS)}."
        )

    return {"review_status": status, "feedback": feedback or state.get("feedback")}


def finalize_node(state: TravelState) -> dict:
    return {"final_output": state["draft_itinerary"]}


def route_after_review(state: TravelState) -> str:
    status = state.get("review_status", STATUS_PENDING)
    if status == STATUS_APPROVED:
        return "finalize_node"
    return "planner_node"


def _build_graph():
    builder = StateGraph(TravelState)

    builder.add_node("research_node", research_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("hitl_node", hitl_node)
    builder.add_node("finalize_node", finalize_node)

    builder.add_edge(START, "research_node")
    builder.add_edge("research_node", "planner_node")
    builder.add_edge("planner_node", "hitl_node")
    builder.add_conditional_edges(
        "hitl_node",
        route_after_review,
        {
            "planner_node": "planner_node",
            "finalize_node": "finalize_node",
        },
    )
    builder.add_edge("finalize_node", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = _build_graph()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _validated_request(travel_request: dict) -> dict:
    missing = [k for k in ("destination", "start_date", "end_date") if k not in travel_request]
    if missing:
        raise ValueError(f"travel_request missing required fields: {', '.join(missing)}")
    return travel_request


def _current_state(thread_id: str) -> dict:
    snapshot = graph.get_state(_config(thread_id))
    return dict(snapshot.values)


def start_planning_session(travel_request: dict) -> dict:
    _validated_request(travel_request)
    thread_id = str(uuid4())
    try:
        graph.invoke(
            {"travel_request": travel_request, "review_status": STATUS_PENDING},
            _config(thread_id),
        )
    except GraphInterrupt:
        pass

    state = _current_state(thread_id)
    return {
        "plan_id": thread_id,
        "status": state.get("review_status", STATUS_PENDING),
        "draft_itinerary": state.get("draft_itinerary"),
        "final_output": state.get("final_output"),
    }


def get_session_state(thread_id: str) -> dict:
    snapshot = graph.get_state(_config(thread_id))
    state = dict(snapshot.values)
    if not state:
        raise ValueError(f"No planning session found for plan_id '{thread_id}'.")

    pending_review = bool(snapshot.interrupts)

    status = state.get("review_status", STATUS_PENDING)
    if not pending_review and status in (
        STATUS_PENDING,
        STATUS_REJECTED,
        STATUS_MODIFIED,
    ):
        pending_review = True

    return {
        "plan_id": thread_id,
        "status": status,
        "draft_itinerary": state.get("draft_itinerary"),
        "final_output": state.get("final_output"),
        "pending_review": pending_review,
    }


def submit_review(thread_id: str, decision: str, feedback: Optional[str] = None) -> dict:
    decision = str(decision).strip().lower()
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision '{decision}'. Must be one of {sorted(VALID_DECISIONS)}."
        )

    get_session_state(thread_id)

    try:
        graph.invoke(
            Command(resume={"decision": decision, "feedback": feedback}),
            _config(thread_id),
        )
    except GraphInterrupt:
        pass

    state = _current_state(thread_id)
    return {
        "plan_id": thread_id,
        "status": state.get("review_status", STATUS_PENDING),
        "draft_itinerary": state.get("draft_itinerary"),
        "final_output": state.get("final_output"),
    }
