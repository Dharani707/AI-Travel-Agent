import json
from core.agents.research_agent import run_research_agent

def main():
    print("==================================================")
    print("Testing Research Agent...")
    print("==================================================")
    
    destination = "Rome, Italy"
    start_date = "2026-09-10"
    end_date = "2026-09-15"
    interests = ["ancient history", "pizza", "art museums"]
    
    print(f"Destination: {destination}")
    print(f"Dates: {start_date} to {end_date}")
    print(f"Interests: {interests}")
    print("\nRunning research agent (this may take a moment to query Tavily and Open-Meteo)...")
    
    try:
        result = run_research_agent(
            destination=destination,
            start_date=start_date,
            end_date=end_date,
            interests=interests
        )
        print("\nSUCCESS! Structured Research Agent Output:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nFAILED to run Research Agent: {e}")

if __name__ == "__main__":
    main()
