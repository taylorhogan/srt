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


def get_air_quality_by_hour(lat: float, lon: float, hours: int) -> tuple[list, list, list, list]:
    """Hourly air-quality forecast from Open-Meteo (no key needed).

    Mirrors get_weather_by_hour exactly — same past-hour filtering and hour-of-day
    alignment — so the returned hours line up with the weather hours for a given
    forecast. Returns (hours, aod, pm2_5, us_aqi_pm2_5) as parallel lists, where
    ``aod`` is the column aerosol optical depth (the light-extinction measure that
    matters for starlight; smoke drives it up). Returns empty lists on any error,
    which callers treat as "smoke unknown".

    The AQI returned is the PM2.5 *sub-index*, not the composite ``us_aqi``. The
    composite is the max across all pollutants, so a hot summer afternoon of
    surface ozone pushes it to 100+ with perfectly clean air — ozone has no effect
    on visible-band transparency and must never gate imaging. Only particulates do.
    """
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    forecast_days = max(1, (hours + 23) // 24)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["aerosol_optical_depth", "pm2_5", "us_aqi_pm2_5"],
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    local_times: list = []
    local_aod: list = []
    local_pm25: list = []
    local_pm25_aqi: list = []

    try:
        # Non-critical data — fail fast rather than stall the nightly weather check.
        response = requests.get(air_quality_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        times = data["hourly"]["time"]
        aod = data["hourly"]["aerosol_optical_depth"]
        pm25 = data["hourly"]["pm2_5"]
        pm25_aqi = data["hourly"]["us_aqi_pm2_5"]

        local_tz = pytz.timezone('America/New_York')
        now = datetime.now(local_tz)

        for i in range(len(times)):
            forecast_time = datetime.fromisoformat(times[i])
            forecast_time_local = forecast_time.astimezone(local_tz)
            if forecast_time_local < now:
                continue

            hour = forecast_time_local.hour
            local_times.append(hour)
            local_aod.append(aod[i])
            local_pm25.append(pm25[i])
            local_pm25_aqi.append(pm25_aqi[i])

    except requests.RequestException as e:
        print(f"Error fetching air quality: {e}")

    return local_times, local_aod, local_pm25, local_pm25_aqi


# Jet-stream wind level (hPa) used as the astronomical-seeing proxy. Upper-air
# wind — not ground wind — drives the high-altitude turbulence that bloats FWHM;
# 250 hPa (~10 km) is the canonical jet-stream level amateurs correlate with
# seeing. Bump this constant to 300/200 hPa to sample a different layer.
JET_LEVEL_HPA = 250


def get_jetstream_by_hour(lat: float, lon: float, hours: int) -> tuple[list, list]:
    """Hourly jet-stream wind speed (km/h at JET_LEVEL_HPA) from Open-Meteo.

    Mirrors get_air_quality_by_hour — same past-hour filtering and hour-of-day
    alignment — so the returned hours line up with the weather hours for a given
    forecast. This upper-air wind is the best cheap proxy for astronomical seeing
    (fast wind aloft = turbulent air = bloated FWHM), unlike the surface wind in
    get_weather_by_hour. Returns (hours, wind_kmh) as parallel lists; empty lists
    on any error, which callers treat as "seeing unknown".
    """
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_days = max(1, (hours + 23) // 24)
    field = f"wind_speed_{JET_LEVEL_HPA}hPa"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [field],
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    local_times: list = []
    local_wind: list = []

    try:
        # Non-critical data — fail fast rather than stall the nightly weather check.
        response = requests.get(forecast_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        times = data["hourly"]["time"]
        wind = data["hourly"][field]

        local_tz = pytz.timezone('America/New_York')
        now = datetime.now(local_tz)

        for i in range(len(times)):
            forecast_time = datetime.fromisoformat(times[i])
            forecast_time_local = forecast_time.astimezone(local_tz)
            if forecast_time_local < now:
                continue
            local_times.append(forecast_time_local.hour)
            local_wind.append(wind[i])

    except requests.RequestException as e:
        print(f"Error fetching jet-stream wind: {e}")

    return local_times, local_wind


def seeing_from_jetstream(wind_kmh: float | None) -> str:
    """Qualitative seeing label from jet-stream (JET_LEVEL_HPA) wind in km/h.

    Thresholds follow common amateur guidance: calm aloft = steady stars.
    """
    if wind_kmh is None:
        return "unknown"
    if wind_kmh < 30:
        return "good"
    if wind_kmh < 55:
        return "fair"
    if wind_kmh < 80:
        return "poor"
    return "bad"


if __name__ == '__main__':
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    get_weather_by_hour(latitude, longitude, 24)
    print(get_sunrise_sunset())
    aq_hours, aod, pm25, pm25_aqi = get_air_quality_by_hour(latitude, longitude, 24)
    for i in range(len(aq_hours)):
        print(f"{aq_hours[i]:>2}h: AOD {aod[i]}  PM2.5 {pm25[i]}  PM2.5 AQI {pm25_aqi[i]}")
    jet_hours, jet_wind = get_jetstream_by_hour(latitude, longitude, 24)
    for i in range(len(jet_hours)):
        print(f"{jet_hours[i]:>2}h: jet {jet_wind[i]:>3.0f} km/h  seeing {seeing_from_jetstream(jet_wind[i])}")
