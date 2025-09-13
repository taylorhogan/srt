import asyncio
from datetime import datetime
import python_weather
import pytz
import requests

import config

cfg = config.data()


async def get_sunrise_sunset_internal() -> [datetime, datetime]:
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        # fetch a weather forecast from a city

        city = cfg["location"]["city"]
        forecast = await client.get(city)
        today = forecast.daily_forecasts[0]
        tomorrow = forecast.daily_forecasts[1]

        sunset = today.sunset
        sunrise = today.sunrise

        return sunrise, sunset


def get_sunrise_sunset() -> [datetime, datetime]:
    return asyncio.run(get_sunrise_sunset_internal())


def get_weather_by_hour(lat, lon, hours):
    # Open-Meteo Forecast API (no key needed)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["cloud_cover", "precipitation_probability", "wind_speed_80m"],
        "forecast_days": 2,  # Ensure enough days
        "timezone": "auto"
    }

    try:
        response = requests.get(forecast_url, params=params)
        response.raise_for_status()
        data = response.json()



        cloud_times = data["hourly"]["time"]
        cloud_covers = data["hourly"]["cloud_cover"]
        precipitation_probability = data["hourly"]["precipitation_probability"]
        wind_speed = data["hourly"]["wind_speed_80m"]

        local_tz = pytz.timezone('America/New_York')
        utc_timezone = pytz.utc
        local_cloud_times = []
        local_cloud_covers = []
        local_precipitation_probability = []
        local_wind_speed = []

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


    except requests.RequestException as e:
        print(f"Error fetching forecast: {e}")

    return local_cloud_times, local_cloud_covers, local_precipitation_probability, local_wind_speed


if __name__ == '__main__':
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    get_weather_by_hour(latitude, longitude, 24)
    print(get_sunrise_sunset())
