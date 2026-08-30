import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_route(origin: str, destination: str, mode: str = "driving") -> dict:
    """
    Uses Google Directions API to get distance and travel time between two points.
    Used for itinerary sequencing/optimization.
    
    Args:
        origin (str): Starting location address or coordinates.
        destination (str): Ending location address or coordinates.
        mode (str): Mode of transport (e.g., 'driving', 'walking', 'transit', 'bicycling').
        
    Returns:
        dict: A dictionary containing distance, duration, and routing instructions, or an error dictionary.
    """
    api_key = os.getenv("GOOGLE_ROUTES_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_ROUTES_API_KEY is not set."}

    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK":
            return {"error": f"Google Routes API Error: {data.get('status')}", "details": data.get("error_message")}
            
        # Extract the first route and its primary leg
        route = data.get("routes", [])[0]
        leg = route.get("legs", [])[0]
        
        return {
            "origin": leg.get("start_address"),
            "destination": leg.get("end_address"),
            "distance": leg.get("distance", {}).get("text"),
            "duration": leg.get("duration", {}).get("text"),
            "distance_meters": leg.get("distance", {}).get("value"),
            "duration_seconds": leg.get("duration", {}).get("value"),
            "mode": mode
        }
    except Exception as e:
        return {"error": f"Exception during Google Routes search: {str(e)}"}

if __name__ == "__main__":
    print("Testing get_route...")
    # Using coordinates or clear landmarks
    result = get_route("Eiffel Tower, Paris", "Louvre Museum, Paris", mode="walking")
    print(result)
