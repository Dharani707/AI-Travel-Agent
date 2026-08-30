import os
import requests
from dotenv import load_dotenv

load_dotenv()

def search_places(location: str, place_type: str = "tourist_attraction") -> list[dict] | dict:
    """
    Uses Google Places API (Legacy Text Search) to find attractions, activities, and restaurants.
    
    Args:
        location (str): The general area/city to search in (e.g., "Paris, France").
        place_type (str): The type of place (e.g., "tourist_attraction", "restaurant").
        
    Returns:
        list[dict] | dict: A list of relevant places containing name, address, and rating, or an error dictionary.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_PLACES_API_KEY is not set."}

    try:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        query = f"{place_type} in {location}"
        
        params = {
            "query": query,
            "key": api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK":
            return {"error": f"Google Places API Error: {data.get('status')}", "details": data.get("error_message")}
            
        results = []
        for place in data.get("results", [])[:5]: # Return top 5 matches
            results.append({
                "name": place.get("name"),
                "address": place.get("formatted_address"),
                "rating": place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total"),
                "types": place.get("types", [])
            })
            
        return results if results else {"error": "No places found for the given query."}
    except Exception as e:
        return {"error": f"Exception during Google Places search: {str(e)}"}

if __name__ == "__main__":
    print("Testing search_places...")
    result = search_places("Paris, France", "tourist_attraction")
    print(result)
