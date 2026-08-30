import json
from core.agents.planner_agent import run_planner_agent

def main():
    print("==================================================")
    print("Testing Planner Agent...")
    print("==================================================")
    
    # Realistic mock research data (similar to what research_agent returns)
    mock_research_data = {
        "destination_summary": "Rome, the Eternal City, is famous for its nearly 3,000 years of globally influential art, architecture, and culture. It is renowned for its historic ruins like the Colosseum and Forum, and delicious cuisine including pizza and pasta.",
        "top_attractions": [
            "Colosseum: Ancient Roman amphitheater located in the center of the city.",
            "Vatican Museums & Sistine Chapel: World-renowned art collections and Michelangelo's famous frescoes.",
            "Pantheon: Best-preserved ancient Roman monument, now a church.",
            "Piazza Navona: Beautiful square featuring Baroque fountains and cafes.",
            "Trastevere neighborhood: Charming medieval area known for traditional pizzerias and trattorias."
        ],
        "safety_notes": "Watch out for pickpockets, especially around the Colosseum, Termini Station, and public transit. Avoid unregistered taxis.",
        "weather_summary": "Expected weather in mid-September is warm and pleasant, with average highs of 26°C and lows of 15°C. Low chance of rain.",
        "seasonal_considerations": "September is a peak travel month in Rome. Attractions will be crowded, and pre-booking tickets for the Colosseum and Vatican Museums is highly recommended."
    }
    
    travel_request = {
        "destination": "Rome, Italy",
        "start_date": "2026-09-10",
        "end_date": "2026-09-12", # 3-day trip
        "budget": "Mid-range",
        "num_travelers": 2,
        "interests": ["ancient history", "pizza", "art museums"]
    }
    
    print("Running initial itinerary generation...")
    try:
        initial_result = run_planner_agent(
            research_data=mock_research_data,
            travel_request=travel_request
        )
        print("\nSUCCESS! Structured Initial Itinerary:")
        print(json.dumps(initial_result, indent=2))
        
        # Now test the HITL feedback iteration
        feedback = "On Day 2, replace the museum visit with a relaxed park walk in Villa Borghese and a picnic."
        
        # Inject the previous itinerary into the travel request so the agent has context to revise it
        travel_request["previous_itinerary"] = initial_result
        
        print("\n--------------------------------------------------")
        print("Testing Human-in-the-Loop Feedback Revision...")
        print(f"Feedback: '{feedback}'")
        print("--------------------------------------------------")
        
        revised_result = run_planner_agent(
            research_data=mock_research_data,
            travel_request=travel_request,
            feedback=feedback
        )
        
        print("\nSUCCESS! Structured Revised Itinerary:")
        print(json.dumps(revised_result, indent=2))
        
    except Exception as e:
        print(f"\nFAILED to run Planner Agent: {e}")

if __name__ == "__main__":
    main()
