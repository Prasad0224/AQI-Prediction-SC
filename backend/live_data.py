import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("WAQI_API_KEY", "demo")


def get_live_data(city="mumbai"):
    url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"
    try:
        res = requests.get(url, timeout=6).json()

        if res.get("status") != "ok":
            return {"error": str(res.get("data", "API error")),
                    "PM2.5": 0, "PM10": 0, "NO2": 0, "CO": 0}

        iaqi      = res["data"]["iaqi"]
        live_aqi  = res["data"].get("aqi", 0)
        city_name = res["data"].get("city", {}).get("name", city)

        return {
            "PM2.5":    iaqi.get("pm25", {}).get("v", 0),
            "PM10":     iaqi.get("pm10", {}).get("v", 0),
            "NO2":      iaqi.get("no2",  {}).get("v", 0),
            "CO":       iaqi.get("co",   {}).get("v", 0),
            "live_aqi": live_aqi,
            "city":     city_name,
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out", "PM2.5": 0, "PM10": 0, "NO2": 0, "CO": 0}
    except Exception as e:
        return {"error": str(e), "PM2.5": 0, "PM10": 0, "NO2": 0, "CO": 0}