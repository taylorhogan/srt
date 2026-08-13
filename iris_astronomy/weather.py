from datetime import datetime
from zoneinfo import ZoneInfo
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
    # Both arguments matter. Without tzinfo, astral resolves the event for the
    # UTC date, and at this longitude the sunset whose UTC timestamp falls on
    # today is *yesterday evening* local — e.g. on 2026-08-03 it returned
    # 00:07 UTC = 2026-08-02 20:07 EDT, a time already in the past. The
    # scheduler's pre-sunset check is sunset - 10 min, so it fell straight
    # through and generated the night's sequence around noon instead of dusk.
    # Passing the local date with tzinfo gives the local evening: 20:06 EDT.
    local_date = datetime.now(ZoneInfo(timezone)).date()
    s = sun(city.observer, date=local_date, tzinfo=city.timezone)

    sunrise = s["sunrise"]  # local-date event, tz-aware
    sunset = s["sunset"]    # local-date event, tz-aware
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
        # Timeout is not optional here. Without one this call inherits the
        # Windows dead-TCP ceiling, ~240 s, and it is on the critical path of
        # the 5-minute live skymap push: on the night of 2026-08-12 five runs
        # stalled at exactly 240.3 s against a 57.5 s mean and pushed nothing,
        # leaving 10-minute holes in the live chart. The other two Open-Meteo
        # calls in this file already pass timeout=10.
        response = requests.get(forecast_url, params=params, timeout=10)
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


# Wind level (hPa) used as the astronomical-seeing proxy.
#
# This was 250 hPa — the canonical jet-stream level amateurs correlate with
# seeing — and for THIS site that was the wrong layer. Measured 2026-08-04 on 9
# nights / 330 frames of sh2-92, median FWHM against wind speed by altitude:
#
#     surface  +0.73    500 hPa  +0.65
#     850 hPa  +0.87    300 hPa  +0.37
#     700 hPa  +0.73    250 hPa  +0.33  <- the jet: no relationship
#                       200 hPa  -0.18
#
# The correlation decays monotonically with height, which is the part that makes
# it believable: noise does not sort itself by altitude. Jet-stream seeing
# forecasts are aimed at mountain observatories that sit ABOVE the boundary
# layer; a near-sea-level backyard site is inside it, so the turbulence that
# bloats FWHM here is low-level. 850 hPa (~1.5 km) is the best single predictor.
SEEING_LEVEL_HPA = 850

# Band edges for seeing_from_wind, in km/h at SEEING_LEVEL_HPA. Named because
# the tonight chart draws them as threshold lines, and a chart whose lines sat
# at different numbers than the words in the report would be worse than no lines.
SEEING_FAIR_KMH = 20
SEEING_POOR_KMH = 30
SEEING_BAD_KMH = 45


def get_seeing_wind_by_hour(lat: float, lon: float, hours: int) -> tuple[list, list]:
    """Hourly wind speed (km/h at SEEING_LEVEL_HPA) from Open-Meteo.

    Mirrors get_air_quality_by_hour — same past-hour filtering and hour-of-day
    alignment — so the returned hours line up with the weather hours for a given
    forecast. Low-level wind is the best cheap proxy for seeing at this site (see
    SEEING_LEVEL_HPA), and is a different thing from the surface wind in
    get_weather_by_hour. Returns (hours, wind_kmh) as parallel lists; empty lists
    on any error, which callers treat as "seeing unknown".
    """
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_days = max(1, (hours + 23) // 24)
    field = f"wind_speed_{SEEING_LEVEL_HPA}hPa"
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
        print(f"Error fetching seeing-level wind: {e}")

    return local_times, local_wind


def seeing_from_wind(wind_kmh: float | None) -> str:
    """Qualitative seeing label from SEEING_LEVEL_HPA wind in km/h.

    Thresholds are measured on this observatory's own frames rather than taken
    from general guidance. Over 9 nights of sh2-92 the 850 hPa wind split the
    nights with NO overlap at ~22 km/h (12 kn):

        under 22 km/h   6 nights, median FWHM 1.73-2.46"
        over  22 km/h   3 nights, median FWHM 2.74-2.98"

    So "good" ends at 20 and "poor" starts at 30, with the band between them
    reported as "fair" — that gap is where this site has no data yet, and saying
    "fair" there is honest about it rather than guessing which side it falls on.
    Nine nights is a thin calibration: treat the labels as a steer, not a promise,
    and re-check them once there are more nights (scripts/seeing_vs_weather.py).
    """
    if wind_kmh is None:
        return "unknown"
    if wind_kmh < SEEING_FAIR_KMH:
        return "good"
    if wind_kmh < SEEING_POOR_KMH:
        return "fair"
    if wind_kmh < SEEING_BAD_KMH:
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
    wind_hours, seeing_wind = get_seeing_wind_by_hour(latitude, longitude, 24)
    for i in range(len(wind_hours)):
        print(f"{wind_hours[i]:>2}h: {SEEING_LEVEL_HPA} hPa wind {seeing_wind[i]:>3.0f} km/h  "
              f"seeing {seeing_from_wind(seeing_wind[i])}")
