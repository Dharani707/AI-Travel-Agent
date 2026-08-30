import json
from core.orchestrator import (
    start_planning_session,
    get_session_state,
    submit_review,
)

def main():
    print("==================================================")
    print("Testing Orchestrator (LangGraph HITL flow)...")
    print("==================================================")

    travel_request = {
        "destination": "Rome, Italy",
        "start_date": "2026-09-10",
        "end_date": "2026-09-12",
        "budget": "Mid-range",
        "num_travelers": 2,
        "interests": ["ancient history", "pizza", "art museums"],
    }

    print("\n[1/4] Submitting new travel request...")
    result = start_planning_session(travel_request)
    plan_id = result.get("plan_id")
    print("\n--------------------------------------------------")
    print("Initial session (paused at HITL node):")
    print("--------------------------------------------------")
    print(f"  plan_id:  {plan_id}")
    print(f"  status:   {result.get('status')}")
    print(f"  final_output: {result.get('final_output')}")
    print("  draft_itinerary present:", bool(result.get("draft_itinerary")))
    state = get_session_state(plan_id)
    print(f"  pending_review: {state.get('pending_review')}")
    if result.get("draft_itinerary"):
        print("\n  Draft itinerary:")
        print(json.dumps(result["draft_itinerary"], indent=2))

    assert plan_id, "start_planning_session did not return a plan_id"
    assert state.get("pending_review") is True, "Session should pause at HITL"
    assert result.get("draft_itinerary"), "Draft itinerary should be present after planning"

    print("\n[2/4] Submitting REJECT with feedback...")
    feedback = "swap day 2's museum for a park walk"
    rejected = submit_review(plan_id, "reject", feedback=feedback)
    print("\n--------------------------------------------------")
    print("After REJECT (planner re-ran, paused at HITL again):")
    print("--------------------------------------------------")
    print(f"  status:   {rejected.get('status')}")
    print(f"  feedback: {feedback}")
    print(f"  final_output: {rejected.get('final_output')}")
    print("  draft_itinerary present:", bool(rejected.get("draft_itinerary")))
    state = get_session_state(plan_id)
    print(f"  pending_review: {state.get('pending_review')}")
    if rejected.get("draft_itinerary"):
        print("\n  Updated draft itinerary:")
        print(json.dumps(rejected["draft_itinerary"], indent=2))

    assert rejected.get("status") == "rejected", "Status should be 'rejected' after reject"
    assert rejected.get("final_output") is None, "No final output until approval"
    assert state.get("pending_review") is True, "Should pause at HITL again after reject"

    print("\n[3/4] Submitting APPROVE...")
    approved = submit_review(plan_id, "approve")
    print("\n--------------------------------------------------")
    print("After APPROVE (workflow completed):")
    print("--------------------------------------------------")
    print(f"  status:   {approved.get('status')}")
    print(f"  final_output present: {bool(approved.get('final_output'))}")
    state = get_session_state(plan_id)
    print(f"  pending_review: {state.get('pending_review')}")
    if approved.get("final_output"):
        print("\n  Final plan:")
        print(json.dumps(approved["final_output"], indent=2))

    assert approved.get("status") == "approved", "Status should be 'approved' after approve"
    assert approved.get("final_output"), "Final output should be produced after approval"
    assert state.get("pending_review") is False, "Workflow done, no longer awaiting review"

    print("\n[4/4] Final session state:")
    print("--------------------------------------------------")
    print(json.dumps(state, indent=2))

    print("\nSUCCESS! Orchestrator flow completed (reject -> revise -> approve -> final).")

if __name__ == "__main__":
    main()
