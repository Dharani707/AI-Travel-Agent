import requests
from datetime import datetime, timedelta

def get_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """
    Uses Open-Meteo's free API to get weather forecast or historical climate averages for travel dates.
    Falls back to historical data (same dates, previous year) if dates are more than 14 days in the future.
    
    Args:
        latitude (float): Location latitude.
        longitude (float): Location longitude.
        start_date (str): Travel start date in YYYY-MM-DD format.
        end_date (str): Travel end date in YYYY-MM-DD format.
        
    Returns:
        dict: Weather forecast data or historical weather averages.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if start_date is within the 14-day forecast window
        now = datetime.now()
        is_future_beyond_14_days = (start_dt - now).days > 14
        
        if is_future_beyond_14_days:
            # Fallback to historical archive (same dates, exactly 1 year ago as an approximation for climate)
            historical_start = (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")
            historical_end = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": historical_start,
                "end_date": historical_end,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "auto"
            }
            is_historical = True
        else:
            # Use forecast API
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "auto"
            }
            is_historical = False
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        
        return {
            "type": "historical_average" if is_historical else "forecast",
            "dates": daily.get("time", []),
            "max_temps_celsius": daily.get("temperature_2m_max", []),
            "min_temps_celsius": daily.get("temperature_2m_min", []),
            "precipitation_mm": daily.get("precipitation_sum", [])
        }
    except Exception as e:
        return {"error": f"Error fetching weather data: {str(e)}"}

if __name__ == "__main__":
    print("Testing get_weather (forecast)...")
    # Using dates within 14 days for forecast (Update dates dynamically to avoid failure next week)
    today = datetime.now()
    d1 = today.strftime("%Y-%m-%d")
    d2 = (today + timedelta(days=3)).strftime("%Y-%m-%d")
    print(get_weather(48.8566, 2.3522, d1, d2)) # Paris coordinates
    
    print("\nTesting get_weather (historical fallback)...")
    # Using dates 6 months in the future
    future1 = (today + timedelta(days=180)).strftime("%Y-%m-%d")
    future2 = (today + timedelta(days=185)).strftime("%Y-%m-%d")
    print(get_weather(48.8566, 2.3522, future1, future2))
