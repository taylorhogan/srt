from datetime import datetime
import pytz
import requests
import sys
import os
from astral import LocationInfo
from astral.sun import sun


if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from configs import config

cfg = config.data()




def get_sunrise_sunset() -> tuple[datetime, datetime]:
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    name = cfg["location"]["city"]
    timezone = cfg["location"]["timezone"]
    city = LocationInfo(name, "USA", timezone, latitude, longitude)
    s = sun(city.observer, date=datetime.now())

    sunrise = s["sunrise"]  # datetime with timezone
    sunset = s["sunset"]  # datetime with timezone
    print(sunrise, sunset)
    return sunrise, sunset


def get_weather_by_hour(lat: float, lon: float, hours: int) -> tuple[list, list, list, list, list]:
    # Open-Meteo Forecast API (no key needed)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_days = max(1, (hours + 23) // 24)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["cloud_cover", "precipitation_probability", "wind_speed_80m", "relative_humidity_2m"],
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    local_cloud_times: list = []
    local_cloud_covers: list = []
    local_precipitation_probability: list = []
    local_wind_speed: list = []
    local_humidity: list = []

    try:
        response = requests.get(forecast_url, params=params)
        response.raise_for_status()
        data = response.json()

        cloud_times = data["hourly"]["time"]
        cloud_covers = data["hourly"]["cloud_cover"]
        precipitation_probability = data["hourly"]["precipitation_probability"]
        wind_speed = data["hourly"]["wind_speed_80m"]
        humidity = data["hourly"]["relative_humidity_2m"]

        local_tz = pytz.timezone('America/New_York')

        now = datetime.now(local_tz)


        for i in range(len(cloud_times)):
            forecast_time = datetime.fromisoformat(cloud_times[i])
            forcast_time_local = forecast_time.astimezone(local_tz)
            if forcast_time_local < now:
                continue

            time_str = forcast_time_local.strftime("%Y-%m-%d %H:%M")
            hour = forcast_time_local.hour
            print(f"{hour}: {cloud_covers[i]}% cloud cover")
            local_cloud_times.append(hour)
            local_cloud_covers.append(cloud_covers[i])
            local_precipitation_probability.append(precipitation_probability[i])
            local_wind_speed.append(wind_speed[i])
            local_humidity.append(humidity[i])


    except requests.RequestException as e:
        print(f"Error fetching forecast: {e}")

    return local_cloud_times, local_cloud_covers, local_precipitation_probability, local_wind_speed, local_humidity


if __name__ == '__main__':
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    get_weather_by_hour(latitude, longitude, 24)
    print(get_sunrise_sunset())
