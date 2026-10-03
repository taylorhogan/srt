import logging
import time
from datetime import datetime, timedelta, timezone as dt_timezone
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
_logger = logging.getLogger(__name__)




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


# Where cloud cover comes from. Scored against the sky camera on 2026-09-30
# (scripts/forecast_score.py, ~30 moon-free nights 08-14..09-25): Open-Meteo's
# default best_match was identical to GFS/HRRR in every logged hour and the
# worst source at the noon check's 9-18 h lead (rank correlation 0.49, Heidke
# 0.28, half its "clear" calls cloudy), while the NWS grid was the steadiest at
# every lead (0.66-0.69, Heidke 0.47-0.50). Only cloud cover moves: rain,
# wind and humidity stay on Open-Meteo, and so does cloud for any hour NWS does
# not answer. "open_meteo" in cfg["weather"]["cloud_source"] reverts.
CLOUD_SOURCE = (cfg.get("weather") or {}).get("cloud_source", "nws")

NWS_POINTS = "https://api.weather.gov/points/%.4f,%.4f"
# api.weather.gov rejects requests without a contact in the User-Agent.
NWS_HEADERS = {"User-Agent": "iris-observatory (taylor.hogan@gmail.com)"}
# The grid is re-issued roughly hourly; the live skymap asks every 5 minutes.
NWS_CACHE_S = 30 * 60
_nws_cache: dict = {}


def _floor_hour_utc(dt: datetime) -> datetime:
    return dt.astimezone(dt_timezone.utc).replace(minute=0, second=0, microsecond=0)


def expand_nws(values: list) -> dict:
    """NWS gives value + ISO-8601 duration intervals; flatten to {UTC hour: value}.

    A single entry can cover many hours (a settled forecast is published as one
    long interval), so reading one value per entry would both misalign the
    series and drop most of the horizon.
    """
    grid = {}
    for v in values:
        head, _, dur = v["validTime"].partition("/")
        t = datetime.fromisoformat(head).astimezone(dt_timezone.utc)

        # PnDTnH -- days and hours are the only units NWS uses here.
        days = hrs = 0
        num = ""
        for ch in dur.lstrip("P"):
            if ch.isdigit():
                num += ch
            elif ch == "D":
                days = int(num or 0)
                num = ""
            elif ch == "H":
                hrs = int(num or 0)
                num = ""
            else:
                num = ""
        total = days * 24 + hrs or 1
        for k in range(total):
            grid[_floor_hour_utc(t + timedelta(hours=k))] = v["value"]
    return grid


def get_nws_sky_cover(lat: float, lon: float, timeout: float = 10, use_cache: bool = True) -> tuple[str, dict]:
    """(office, {UTC hour: sky cover %}) from the NWS forecast grid.

    Raises on any failure; callers decide what falling back means.
    """
    key = (round(lat, 4), round(lon, 4))
    hit = _nws_cache.get(key)
    if use_cache and hit and time.monotonic() - hit[0] < NWS_CACHE_S:
        return hit[1], hit[2]
    p = requests.get(NWS_POINTS % (lat, lon), headers=NWS_HEADERS, timeout=timeout)
    p.raise_for_status()
    props = p.json()["properties"]
    g = requests.get(props["forecastGridData"], headers=NWS_HEADERS, timeout=timeout)
    g.raise_for_status()
    office = "%s %d,%d" % (props["gridId"], props["gridX"], props["gridY"])
    grid = expand_nws(g.json()["properties"]["skyCover"]["values"])
    _nws_cache[key] = (time.monotonic(), office, grid)
    return office, grid


def get_weather_by_hour(lat: float, lon: float, hours: int) -> tuple[list, list, list, list, list]:
    """(hours, cloud, precip prob, wind, humidity) lists keyed by CLOCK HOUR only.
    See get_weather_by_datetime for why that is ambiguous late in the evening."""
    rows = _hourly_rows(lat, lon, hours)
    return ([r[0].hour for r in rows], [r[1] for r in rows], [r[2] for r in rows],
            [r[3] for r in rows], [r[4] for r in rows])


def get_weather_by_datetime(lat: float, lon: float, hours: int) -> dict:
    """{(year, month, day, hour) local: (cloud %, precip %, wind, humidity)}.

    get_weather_by_hour keys by clock hour alone, so once tonight's early hours
    have passed (they are dropped as past), a lookup for 20:00 finds TOMORROW's
    20:00. On 2026-10-02 at 22:00 that painted tonight's cloudy 20:00-22:00 with
    tomorrow evening's 0-2%. Callers that place values on a specific night key
    by date as well.
    """
    return {(d.year, d.month, d.day, d.hour): (c, p, w, h)
            for d, c, p, w, h in _hourly_rows(lat, lon, hours)}


def _hourly_rows(lat: float, lon: float, hours: int) -> list:
    """[(local datetime, cloud, precip prob, wind 80 m, humidity)] from this hour on."""
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_days = max(1, (hours + 23) // 24)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["cloud_cover", "precipitation_probability", "wind_speed_80m", "relative_humidity_2m"],
        "forecast_days": forecast_days,
        "timezone": "auto"
    }
    rows: list = []

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

        nws_sky: dict = {}
        if CLOUD_SOURCE == "nws":
            try:
                _office, nws_sky = get_nws_sky_cover(lat, lon)
            except (requests.RequestException, KeyError, ValueError) as e:
                # Logged, not just printed: a print goes nowhere from the chat
                # server, and on 2026-10-02 22:00 a failed fetch left no trace.
                _logger.warning("NWS sky cover unavailable, using Open-Meteo cloud: %s", e)
        n_nws = 0


        for i in range(len(cloud_times)):
            forecast_time = datetime.fromisoformat(cloud_times[i])
            forcast_time_local = forecast_time.astimezone(local_tz)
            # Keep the hour that is under way: it has not ended, and dropping it
            # left the current hour blank in every report.
            if forcast_time_local + timedelta(hours=1) <= now:
                continue

            time_str = forcast_time_local.strftime("%Y-%m-%d %H:%M")
            hour = forcast_time_local.hour
            cover = nws_sky.get(_floor_hour_utc(forcast_time_local))
            if cover is None:
                cover = cloud_covers[i]
            else:
                n_nws += 1
            print(f"{hour}: {cover}% cloud cover")
            rows.append((forcast_time_local, cover, precipitation_probability[i],
                         wind_speed[i], humidity[i]))
        if CLOUD_SOURCE == "nws":
            print(f"cloud cover: NWS for {n_nws} of {len(rows)} hours, Open-Meteo for the rest")

    except requests.RequestException as e:
        # Logged: an empty forecast crashed `tonight` at 22:00 on 2026-10-02 and
        # the print that said why went to a console nobody reads.
        _logger.warning("Open-Meteo forecast fetch failed: %s", e)

    return rows


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
