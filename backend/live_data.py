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

        # WAQI returns US EPA AQI values (0-500), but models expect raw concentrations.
        # We perform a rough approximate reverse conversion so predictions don't explode.
        raw_pm25 = iaqi.get("pm25", {}).get("v", 0) * 0.5
        raw_pm10 = iaqi.get("pm10", {}).get("v", 0) * 1.0
        raw_no2  = iaqi.get("no2",  {}).get("v", 0) * 0.8
        raw_co   = iaqi.get("co",   {}).get("v", 0) * 0.1

        return {
            "PM2.5":    round(raw_pm25, 2),
            "PM10":     round(raw_pm10, 2),
            "NO2":      round(raw_no2, 2),
            "CO":       round(raw_co, 2),
            "live_aqi": live_aqi,
            "city":     city_name,
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out", "PM2.5": 0, "PM10": 0, "NO2": 0, "CO": 0}
    except Exception as e:
        return {"error": str(e), "PM2.5": 0, "PM10": 0, "NO2": 0, "CO": 0}