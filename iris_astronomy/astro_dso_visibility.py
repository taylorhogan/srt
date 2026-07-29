# for all dso objects
#   remove those that have been imaged, done
#   how many hours will it be visible > some altitude
# https://astroplan.readthedocs.io/en/stable/
import os as _os
import pathlib as _pathlib
# Redirect astropy cache to a project-local directory before any astropy import.
# This prevents PermissionErrors when the default cache dir (~/.astropy) is not
# writable for the process context (e.g. launched via bat file vs Task Manager).
_cache_dir = str(_pathlib.Path(__file__).resolve().parent.parent / "local" / "astropy_cache")
_os.makedirs(_cache_dir, exist_ok=True)
_os.environ.setdefault("ASTROPY_CACHE_DIR", _cache_dir)

import datetime
import json
import math

import operator
import os
import sys
from collections.abc import Generator
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-GUI backend — required when running in a subprocess
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytz
import astroplan.plots
from astroplan import FixedTarget
from astroplan import Observer
from astroplan import moon_illumination
from astroquery.simbad import Simbad
from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time
from matplotlib import dates
from pathlib import Path



if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from astropy.utils.iers import conf
conf.auto_max_age = None
from astropy.utils import iers
iers.conf.auto_download = False

from configs import config
from iris_astronomy import weather
from utils import utils


CFG = config.data()
LOGGER = utils.set_logger()
CFG["logger"]["logging"] = LOGGER


_GALAXY_TYPES = {
    'G', 'GiC', 'GiG', 'GiP', 'SyG', 'Sy1', 'Sy2', 'rG', 'HzG',
    'AGN', 'EmG', 'BiC', 'BCD', 'dSph', 'LSB', 'cD', 'LIN',
}
_CLUSTER_TYPES = {'ClG', 'GrG', 'PCG', 'SCG'}
_NEBULA_TYPES  = {'PN', 'HII', 'SNR', 'RNe', 'MoC', 'Neb', 'HH', 'DNeb', 'GNeb', 'RfN'}

_simbad = Simbad()
_simbad.add_votable_fields('otype')


def get_dso_type(name: str) -> str:
    try:
        result = _simbad.query_object(name)
        if result is None:
            return "?"
        col = 'OTYPE' if 'OTYPE' in result.colnames else 'otype'
        otype = str(result[col][0])
        if otype in _GALAXY_TYPES:
            return "G"
        if otype in _CLUSTER_TYPES:
            return "C"
        if otype in _NEBULA_TYPES:
            return "N"
        return "?"
    except Exception:
        return "?"


def is_a_dso_object(name: str) -> Optional[FixedTarget]:
    try:
        from astropy.coordinates import SkyCoord
        coord = SkyCoord.from_name(name, cache=False)
        return FixedTarget(coord=coord, name=name)
    except Exception as e:
        print(f"DEBUG is_a_dso_object({name}): {type(e).__name__}: {e}")
        return None


def resolve_target(instruction: dict) -> Optional[FixedTarget]:
    """Resolve an instruction record to a FixedTarget.

    Stored RA/Dec (``ra_deg``/``dec_deg``) is authoritative when present, so an
    explicit position is never silently overridden by a name lookup. Falls back
    to resolving the ``dso`` name via Simbad when no coordinates are stored.
    """
    ra = instruction.get("ra_deg")
    dec = instruction.get("dec_deg")
    if ra is not None and dec is not None:
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        return FixedTarget(coord=coord, name=instruction.get("dso", ""))
    return is_a_dso_object(instruction["dso"])


def get_horizon_from_azimuth(this_az: float, az: list[float], al: list[float]) -> float:
    for idx in range(len(az) - 1):
        if this_az >= az[idx] and this_az <= az[idx + 1]:
            return al[idx]
    return al[-1]


def get_dark_times(my_observatory: Observer, observe_time: Time) -> tuple[datetime.datetime, datetime.datetime]:

    start = observe_time[0]
    local_tz = pytz.timezone('America/New_York')
    utc_timezone = pytz.utc

    end_of_dark_naive = my_observatory.twilight_morning_astronomical(Time(start), which='next').datetime
    end_of_dark_utc = utc_timezone.localize(end_of_dark_naive)
    end_of_dark_local = end_of_dark_utc.astimezone(local_tz)

    start_of_dark_naive = my_observatory.twilight_evening_astronomical(Time(start), which='next').datetime
    start_of_dark_utc = utc_timezone.localize(start_of_dark_naive)
    start_of_dark_local = start_of_dark_utc.astimezone(local_tz)

    return start_of_dark_local, end_of_dark_local

def find_alt_az_horizon_times(
    dso: FixedTarget,
    my_observatory: Observer,
    observe_time: Time,
) -> tuple[Any, Any, list[float], Optional[datetime.datetime], Optional[datetime.datetime], Optional[datetime.timedelta], float]:
    altitude = (my_observatory.altaz(observe_time, dso).alt) * (1 / u.deg)

    # Azimuth MUST be given to plot() in radians.
    azimuth = my_observatory.altaz(observe_time, dso).az * (1 / u.deg)
    # print out in local time

    local_datetime = my_observatory.astropy_time_to_datetime(observe_time)

    horizon = []
    az, al = map_az_to_horizon()

    start_time = None
    finish_time = None
    start = observe_time[0]
    local_tz = pytz.timezone('America/New_York')
    utc_timezone = pytz.utc
    start_of_dark_local, end_of_dark_local = get_dark_times(my_observatory, observe_time)
    max_altitude = 0



    for idx in range(len(local_datetime)):
        h = get_horizon_from_azimuth(azimuth[idx], az, al)
        horizon.append(h)

        if local_datetime[idx] > end_of_dark_local:
            if finish_time is None:
                finish_time = local_datetime[idx]
        else:
            if local_datetime[idx] > start_of_dark_local:
                if altitude[idx] >= h:
                    finish_time = None
                    if start_time is None:
                        start_time = local_datetime[idx]
                if altitude[idx] < h:
                    if finish_time is None:
                        finish_time = local_datetime[idx]

                if altitude[idx] > max_altitude:
                    max_altitude = altitude[idx]

    elapsed_time = None
    if start_time is not None:
        elapsed_time = finish_time - start_time

    return altitude, azimuth, horizon, start_time, finish_time, elapsed_time, max_altitude


def format_local_time(dt: datetime.datetime) -> str:
    """Format a local datetime like '10:04pm'."""
    return dt.strftime("%I:%M%p").lstrip("0").lower()


def plot_my_dso_and_horizon(dso: FixedTarget, my_observatory: Observer, observe_time: Time) -> tuple[bool, str]:
    altitude, azimuth, horizon, start_time, finish_time, elapsed_time, max_altitude = find_alt_az_horizon_times(dso, my_observatory,
                                                                                                  observe_time)

    masked_altitude = np.ma.array(altitude, mask=altitude < 0)
    local_datetime = my_observatory.astropy_time_to_datetime(observe_time)

    plt.figure(figsize=(10, 12))
    ax = plt.gca()



    longitude = CFG["location"]["longitude"]
    latitude = CFG["location"]["latitude"]
    elevation = CFG["location"]["elevation"]
    observatory_name = CFG["location"]["observatory_name"]
    city = CFG["location"]["city"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")


    local_tz = pytz.timezone('America/New_York')
    #plot the altitude of the DSO
    ax.plot(local_datetime, masked_altitude, color='purple', label = dso.name,linewidth=6)

    #plot the horizon
    ax.plot(local_datetime, horizon, color='green', linestyle='solid', label = 'Horizon' ,linewidth=6  )

    #plot the moon
    moon = get_body("moon", observe_time, location=location)
    altaz = moon.transform_to(AltAz(obstime=observe_time, location=location))
    moon_alt = altaz.alt.deg
    moon_alt = np.ma.array(moon_alt, mask=moon_alt < 0)
    year = local_datetime[0].year
    month = local_datetime[0].month
    day = local_datetime[0].day
    specific_date = Time(str(year) + '-' + str(month) + '-' + str(day) + 'T00:00:00', format='isot', scale='utc')
    illumination = int(float(moon_illumination(specific_date) * 100))

    ax.plot(local_datetime, moon_alt, color='blue', label = 'Moon(' + str(illumination) + "%)",linewidth=2)

    #plot the cloud cover
    cloud_times,cloud_covers,pp,wsp,hum = weather.get_weather_by_hour(latitude, longitude, 48)
    # Air quality (smoke) — first occurrence of each hour wins (today over tomorrow).
    aq_hours, aq_aod, aq_pm25, aq_pm25_aqi = weather.get_air_quality_by_hour(latitude, longitude, 48)
    pm25_aqi_by_hour: dict = {}
    pm25_by_hour: dict = {}
    for k in range(len(aq_hours)):
        if aq_hours[k] not in pm25_aqi_by_hour:
            pm25_aqi_by_hour[aq_hours[k]] = aq_pm25_aqi[k]
            pm25_by_hour[aq_hours[k]] = aq_pm25[k]
    peak_pm25_aqi = None   # worst (highest PM2.5 AQI) hour over the imaging window
    peak_pm25 = None
    # Jet-stream wind (250 hPa) as a seeing proxy — first occurrence of each hour wins.
    jet_hours, jet_wind = weather.get_jetstream_by_hour(latitude, longitude, 48)
    jet_by_hour: dict = {}
    for k in range(len(jet_hours)):
        if jet_hours[k] not in jet_by_hour:
            jet_by_hour[jet_hours[k]] = jet_wind[k]
    peak_jet = None      # worst (highest) jet-stream wind over the imaging window
    time_format = "%Y-%m-%d %H:%M"
    clipped_cloud = []
    clipped_pp = []
    clipped_wsp = []
    clipped_hum = []
    clipped_smoke = []   # PM2.5 AQI scaled onto the 0-90 axis (AQI/150*90)
    clipped_jet = []     # jet-stream km/h scaled onto the 0-90 axis (kmh/150*90)
    weather_ok = True
    issues: set[str] = set()
    # Judge weather only during the imaging window (dark + above horizon).
    # If the object never rises tonight, fall back to the whole plotted span.
    window_finish = finish_time if finish_time is not None else local_datetime[-1]
    for i in range(len(local_datetime)):
        hour = local_datetime[i].hour
        found_hour = False
        for j in range(len(cloud_times)):
            cloud_time_hour = cloud_times[j]
            if hour == cloud_time_hour:
                found_hour = True
                clipped_cloud.append(cloud_covers[j]/100*90)
                clipped_pp.append(pp[j]/100*90)
                clipped_wsp.append(wsp[j]/40*90)
                clipped_hum.append(hum[j]/100*90)
                smoke_aqi = pm25_aqi_by_hour.get(hour)
                clipped_smoke.append(smoke_aqi/150*90 if smoke_aqi is not None else float('nan'))
                jet_kmh = jet_by_hour.get(hour)
                clipped_jet.append(jet_kmh/150*90 if jet_kmh is not None else float('nan'))
                in_window = (start_time is None
                             or start_time <= local_datetime[i] <= window_finish)
                if in_window:
                    hour_pm25_aqi = pm25_aqi_by_hour.get(hour)
                    hour_issues = weather_issues(cloud_covers[j], pp[j], wsp[j])
                    if hour_issues:
                        weather_ok = False
                        issues.update(hour_issues)
                    if hour_pm25_aqi is not None and (peak_pm25_aqi is None
                                                      or hour_pm25_aqi > peak_pm25_aqi):
                        peak_pm25_aqi = hour_pm25_aqi
                        peak_pm25 = pm25_by_hour.get(hour)
                    hour_jet = jet_by_hour.get(hour)
                    if hour_jet is not None and (peak_jet is None or hour_jet > peak_jet):
                        peak_jet = hour_jet

            if found_hour:
                break

    if start_time is not None:
        window_txt = f"between {format_local_time(start_time)} and {format_local_time(window_finish)}"
        if weather_ok:
            weather_msg = f"Weather is good for imaging {dso.name} {window_txt}"
        else:
            weather_msg = f"Weather is not good for imaging {dso.name}: {' and '.join(sorted(issues))} {window_txt}"
    else:
        if weather_ok:
            weather_msg = "Weather ok tonight"
        else:
            weather_msg = f"Weather not ok tonight: {' and '.join(sorted(issues))}"

        if not found_hour:
            clipped_cloud.append(float('nan'))
            clipped_pp.append(float('nan'))
            clipped_wsp.append(float('nan'))
            clipped_hum.append(float('nan'))
            clipped_smoke.append(float('nan'))
            clipped_jet.append(float('nan'))

    # Always report the smoke level for the imaging window, even when it's clear.
    if peak_pm25_aqi is not None:
        weather_msg += (f"\nSmoke: {smoke_descriptor(peak_pm25_aqi)} "
                        f"(PM2.5 AQI {peak_pm25_aqi:.0f}, PM2.5 {peak_pm25:.0f} µg/m³)")
        advisory = smoke_advisory(peak_pm25_aqi)
        if advisory:
            weather_msg += f"\n{advisory}"
    else:
        weather_msg += "\nSmoke: unknown"

    # Report the jet-stream seeing proxy for the imaging window (worst hour).
    if peak_jet is not None:
        weather_msg += (f"\nSeeing (jet stream): {weather.seeing_from_jetstream(peak_jet)} "
                        f"({peak_jet:.0f} km/h at {weather.JET_LEVEL_HPA} hPa)")
    else:
        weather_msg += "\nSeeing (jet stream): unknown"

    ax.plot(local_datetime, clipped_cloud, color='red', label = 'Cloud Cover',linewidth=2)
    ax.plot(local_datetime, clipped_pp, color='pink', label='Prob. Precip.',linewidth=2)
    ax.plot(local_datetime, clipped_wsp, color='black', label='Wind Speed % of 25 mph',linewidth=2)
    ax.plot(local_datetime, clipped_hum, color='green', label='Humidity %',linewidth=2)
    ax.plot(local_datetime, clipped_smoke, color='saddlebrown', linestyle='--',
            label='Smoke (PM2.5 AQI, /150)', linewidth=2)
    ax.axhline(_SMOKE_AQI_ADVISORY / 150 * 90, color='saddlebrown', linestyle=':',
               linewidth=1, alpha=0.4,
               label=f'Smoke advisory (PM2.5 AQI {_SMOKE_AQI_ADVISORY})')
    ax.plot(local_datetime, clipped_jet, color='darkorange', linestyle='-.',
            label='Jet Stream (250 hPa km/h, /150)', linewidth=2)
    ax.axhline(55 / 150 * 90, color='darkorange', linestyle=':',
               linewidth=1, alpha=0.4, label='Seeing gate (55 km/h)')

    ax.set_xlim([local_datetime[0], local_datetime[-1]])
    date_formatter = dates.DateFormatter('%H', tz=local_tz)
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.legend(loc="center", bbox_to_anchor = (0.5, -0.1), title="Legend", fontsize=10, ncol=4, fancybox=True, shadow=True)

    # Shade background during night time

    start = local_datetime[0]

    twilights = [
        (my_observatory.sun_set_time(Time(start), which='next').datetime, 0.0),
        (my_observatory.twilight_evening_civil(Time(start), which='next').datetime, 0.1),
        (my_observatory.twilight_evening_nautical(Time(start), which='next').datetime, 0.2),
        (my_observatory.twilight_evening_astronomical(Time(start), which='next').datetime, 0.3),
        (my_observatory.twilight_morning_astronomical(Time(start), which='next').datetime, 0.4),
        (my_observatory.twilight_morning_nautical(Time(start), which='next').datetime, 0.3),
        (my_observatory.twilight_morning_civil(Time(start), which='next').datetime, 0.2),
        (my_observatory.sun_rise_time(Time(start), which='next').datetime, 0.1),
    ]

    twilights.sort(key=operator.itemgetter(0))
    for i, twi in enumerate(twilights[1:], 1):
        ax.axvspan(twilights[i - 1][0], twilights[i][0],
                   ymin=0, ymax=1, color='grey', alpha=twi[1])

    ax.set_ylim([0, 90])

    # Set labels.
    ax.set_ylabel("Altitude")
    ax.set_xlabel("Time")
    if elapsed_time is not None:
        title = dso.name + "\n" + "Start: " + start_time.strftime("%H:%M") + " Finish: " + finish_time.strftime(
            "%H:%M") + " Elapsed Time: " + str(elapsed_time) + " Air mass: " + "{:.2f}".format(air_mass(max_altitude))
    else:
        title = dso.name + " Not Visible"
    plt.title(title)

    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()

    # Output.

    return weather_ok, weather_msg


def show_plots(dso: FixedTarget) -> tuple[Optional[str], Optional[str], Optional[str], bool, str]:


    longitude = CFG["location"]["longitude"]
    latitude = CFG["location"]["latitude"]
    elevation = CFG["location"]["elevation"]
    observatory_name = CFG["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")
    time = Time.now()

    sunset_tonight = my_observatory.sun_set_time(time, which='nearest')
    sunrise_tonight = my_observatory.sun_rise_time(time, which='nearest')

    #object_is_up = my_observatory.target_is_up(time, dso)

    observe_time = sunset_tonight
    observe_time = observe_time + np.linspace(-1, 14, 55) * u.hour

    dir_name = os.path.dirname(__file__)
    scratch_dir = os.path.join(dir_name + "/scratch")
    image_path = sky_path = altitude_path = None

    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)
    # try:
    #     #ax, hdu = plot_finder_image(dso, fov_radius=42*u.arcmin)
    #     ax, hdu = plot_finder_image(dso)
    #
    #     image_path = os.path.join(scratch_dir, "image.png")
    #     plt.savefig(image_path)
    #     plt.clf()
    # except:
    #     logger = logging.getLogger(__name__)
    #     logger.info('Problem')
    #     logger.exception("Exception")
    weather_ok = False
    weather_msg = "Weather check failed"
    try:

        astroplan.plots.plot_sky(dso, my_observatory, observe_time)
        # Overlay custom horizon from my.hrz on the polar sky chart.
        hz_az, hz_alt = map_az_to_horizon()
        plt.clf()  # clear the side-effect plot from map_az_to_horizon
        astroplan.plots.plot_sky(dso, my_observatory, observe_time)
        ax_sky = plt.gca()
        # Close the horizon loop back to the first point.
        hz_az_closed = hz_az + [hz_az[0] + 360]
        hz_alt_closed = hz_alt + [hz_alt[0]]
        # Polar plot: azimuth in radians, radius = 90 - altitude.
        hz_theta = [np.radians(a) for a in hz_az_closed]
        hz_r = [90 - a for a in hz_alt_closed]
        ax_sky.plot(hz_theta, hz_r, color='red', linewidth=2, linestyle='-', label='My Horizon', zorder=5)
        ax_sky.fill_between(hz_theta, hz_r, 90, color='red', alpha=0.15, zorder=4)
        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        sky_path = os.path.join(scratch_dir, "sky.png")
        plt.savefig(sky_path)
        plt.close()
    except:

        LOGGER.info('Problem')
        LOGGER.exception("Exception")

    try:
        print("a")
        weather_ok, weather_msg = plot_my_dso_and_horizon(dso, my_observatory, observe_time)
        altitude_path = os.path.join(scratch_dir, "horizon.png")
        print ("b")
        plt.savefig(altitude_path)
        plt.close()
    except:

        LOGGER.info('Problem')
        LOGGER.exception("Exception")


    return altitude_path, image_path, sky_path, weather_ok, weather_msg


def get_above_horizon_time(dso: FixedTarget, time: Time) -> tuple[Optional[datetime.timedelta], float]:
    longitude = CFG["location"]["longitude"]
    latitude = CFG["location"]["latitude"]
    elevation = CFG["location"]["elevation"]
    observatory_name = CFG["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")

    sunset_tonight = my_observatory.sun_set_time(time, which='nearest')
    sunrise_tonight = my_observatory.sun_rise_time(time, which='nearest')

    observe_time = sunset_tonight
    observe_time = observe_time + np.linspace(-1, 14, 55) * u.hour
    altitude, azimuth, horizon, start_time, finish_time, elapsed_time, max_altitude = find_alt_az_horizon_times(dso, my_observatory,
                                                                                                  observe_time)
    return elapsed_time, max_altitude


def get_finish_time(dso: FixedTarget, time: Time) -> Optional[datetime.datetime]:
    """Return when dso drops below the custom horizon tonight (local tz), or None."""
    longitude = CFG["location"]["longitude"]
    latitude  = CFG["location"]["latitude"]
    elevation = CFG["location"]["elevation"]
    location  = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=CFG["location"]["observatory_name"], timezone="US/Eastern")

    sunset_tonight = my_observatory.sun_set_time(time, which='nearest')
    observe_time   = sunset_tonight + np.linspace(-1, 14, 55) * u.hour
    _, _, _, _, finish_time, _, _ = find_alt_az_horizon_times(dso, my_observatory, observe_time)
    return finish_time


def air_mass(altitude: float) -> float:
    h = math.sin(math.radians(altitude))
    if h == 0:
        return 100
    else:
        return 1.0 / math.sin(math.radians(altitude))

# Smoke is REPORTED, NOT GATED. It is measured from the US AQI PM2.5 sub-index —
# never the composite us_aqi (the max over all pollutant sub-indices), because
# ground-level ozone does not scatter or absorb in the visible band. On 2026-07-27
# the composite hit 102 on pure surface ozone and blocked a night whose PM2.5
# sub-index was 61.
#
# The gate was removed 2026-07-27 because the threshold was never validated. It was
# fit to four sh2-92 nights (Jul 11-14 2026) treating the Jul 13/14 star-count
# collapse as smoke, but AirNow ground stations measured 10.8 and 13.3 µg/m³ at
# those imaging hours — clean air. That collapse was seeing (dew spread 5.2/4.5 °C,
# FWHM 2.98"/2.68"), not aerosol. The one real smoke event on record (Jul 15-19,
# measured peak 131 µg/m³) fell in a gap in the imaging record, so whether smoke
# degrades images here is UNTESTED, not disproven — which is no basis for throwing
# away clear nights.
#
# These are CAMS *model* values, which validate poorly here: vs AirNow over Jul
# 11-25 the model/measured ratio swings 0.24-1.70, running ~1.5x high on clean
# nights and badly under-calling the real smoke event. Measured PM2.5 is the column
# to trust — see scripts/night_quality_vs_conditions.py (pm25_epa).
#
# Restoring a gate needs a labelled night at genuinely high *measured* PM2.5.
_SMOKE_AQI_ADVISORY = 60   # PM2.5 AQI at/above which the report carries a warning


def smoke_descriptor(pm25_aqi: float | None) -> str:
    """Plain-language smoke level from the US AQI PM2.5 sub-index, calibrated to
    imaging transparency at this site (finer than the health-AQI categories: local
    smoke events that wreck transparency here only reach the AQI-60s)."""
    if pm25_aqi is None:
        return "unknown"
    if pm25_aqi < 55:
        return "clear"
    if pm25_aqi < 70:
        return "smoky/hazy"
    if pm25_aqi < 100:
        return "heavy smoke"
    return "very heavy smoke"


def weather_issues(cloud_cover: float, precipitation_probability: float,
                   wind_speed: float) -> list[str]:
    """Return human-readable weather problems for one forecast hour, empty if OK.

    Smoke is deliberately absent: it is reported as an advisory alongside the
    verdict, never allowed to veto a night. See _SMOKE_AQI_ADVISORY above.
    """
    issues = []
    if cloud_cover > 80:
        issues.append("heavy cloud cover")
    if precipitation_probability > 20:
        issues.append("a chance of rain")
    return issues


def is_weather_ok(cloud_cover: float, precipitation_probability: float,
                  wind_speed: float) -> bool:
    return not weather_issues(cloud_cover, precipitation_probability, wind_speed)


def smoke_advisory(pm25_aqi: float | None) -> str | None:
    """Warning line when smoke is high enough to be worth flagging, else None."""
    if pm25_aqi is None or pm25_aqi < _SMOKE_AQI_ADVISORY:
        return None
    return (f"WARNING: {smoke_descriptor(pm25_aqi)} — transparency may be reduced. "
            f"Not blocking imaging (modelled value, threshold unvalidated).")


def map_az_to_horizon() -> tuple[list[float], list[float]]:
    ax = plt.gca()
    data = []
    az = []
    al = []
    dir_name = os.path.dirname(__file__)
    path = Path ( dir_name)
    root = path.parent
    file_path = root / "configs" / "my.hrz"
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split the line into two columns
            columns = line.strip().split()
            if len(columns) == 2:
                # Assuming the columns are numeric, convert to float
                col1 = float(columns[0])
                col2 = float(columns[1])
                az.append(col1)
                al.append(col2)

    plt.plot(az, al)

    sky_path = root / "scratch" / "h.png"

    plt.savefig(sky_path)
    plt.close()
    return az, al


def enumerate_days_of_year() -> Generator[datetime.date, None, None]:

    now = datetime.datetime.now()

    first_day = datetime.date(now.year, now.month, now.day)


    for day_number in range(1, 365):
        current_day = first_day + datetime.timedelta(days=day_number - 1)
        yield current_day


def best_day_for_dso(dso: FixedTarget) -> tuple[Optional[datetime.datetime], Optional[datetime.timedelta], float]:
    longitude = CFG["location"]["longitude"]
    latitude = CFG["location"]["latitude"]
    elevation = CFG["location"]["elevation"]
    observatory_name = CFG["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")

    days = list(enumerate_days_of_year())
    n_days = len(days)
    n_pts = 55

    # Batch all 365 sunset lookups in a single call
    day_times = Time([datetime.datetime(d.year, d.month, d.day, 14, 0, 0) for d in days])
    sunsets = my_observatory.sun_set_time(day_times, which='nearest')

    # Build a flat (n_days * n_pts,) Time array via JD arithmetic — no loop needed
    offsets_hours = np.linspace(-1, 14, n_pts)
    all_jd = sunsets.jd[:, np.newaxis] + offsets_hours[np.newaxis, :] / 24.0  # (n_days, n_pts)
    all_times_flat = Time(all_jd.ravel(), format='jd')

    # Single altaz call for all days and time points combined
    altaz_all = my_observatory.altaz(all_times_flat, dso)
    altitude_all = altaz_all.alt.deg.reshape(n_days, n_pts)
    azimuth_all = altaz_all.az.deg.reshape(n_days, n_pts)

    # Load the custom horizon profile once
    az_horizon, al_horizon = map_az_to_horizon()

    best_time = None
    best_date = None
    best_max_altitude = 0.0

    for i, day in enumerate(days):
        try:
            observe_time_day = Time(all_jd[i], format='jd')
            altitude = altitude_all[i]
            azimuth = azimuth_all[i]

            local_datetime = my_observatory.astropy_time_to_datetime(observe_time_day)
            start_of_dark_local, end_of_dark_local = get_dark_times(my_observatory, observe_time_day)

            start_time_day = None
            finish_time_day = None
            max_altitude_day = 0.0

            for idx in range(n_pts):
                h = get_horizon_from_azimuth(azimuth[idx], az_horizon, al_horizon)

                if local_datetime[idx] > end_of_dark_local:
                    if finish_time_day is None:
                        finish_time_day = local_datetime[idx]
                else:
                    if local_datetime[idx] > start_of_dark_local:
                        if altitude[idx] >= h:
                            finish_time_day = None
                            if start_time_day is None:
                                start_time_day = local_datetime[idx]
                        if altitude[idx] < h:
                            if finish_time_day is None:
                                finish_time_day = local_datetime[idx]
                        if altitude[idx] > max_altitude_day:
                            max_altitude_day = altitude[idx]

            if start_time_day is not None:
                elapsed_time = finish_time_day - start_time_day
                print(day.year, day.month, day.day,
                      elapsed_time.total_seconds() / 3600,
                      "{:.2f}".format(air_mass(max_altitude_day)))
                if best_time is None or elapsed_time > best_time:
                    best_date = datetime.datetime(day.year, day.month, day.day, 14, 0, 0)
                    best_time = elapsed_time
                    best_max_altitude = max_altitude_day  # only update alongside best_date
        except Exception:
            continue  # skip bad days, don't abort the entire search

    if best_date is None:
        return None, None, None
    return best_date, best_time, best_max_altitude


def build_grid_html(
    object_lines: list,
    hour_header: list,
    weather_syms: list,
) -> str:
    """Return a dark-themed HTML table for tonight's imaging grid."""
    BG_MAIN     = "#0d1117"
    BG_HDR      = "#161b22"
    BG_ABOVE_OK = "#1a4d1a"   # above horizon + good weather
    BG_ABOVE_WX = "#1a2e40"   # above horizon + bad weather
    BG_BELOW    = "#131820"   # below horizon
    BG_WX_GOOD  = "#1a4d1a"
    BG_WX_BAD   = "#4d1a1a"
    BG_PRIORITY = "#3d2e00"
    CLR_TEXT    = "#c9d1d9"
    CLR_DIM     = "#8b949e"
    CLR_ACCENT  = "#58a6ff"
    CLR_BORDER  = "#21262d"

    def td(content, bg, color=CLR_DIM, extra=""):
        return (
            f'<td style="padding:4px 8px;border-bottom:1px solid {CLR_BORDER};'
            f'border-right:1px solid {CLR_BORDER};background:{bg};'
            f'color:{color};font-size:11px;text-align:center;'
            f'white-space:nowrap;{extra}">{content}</td>'
        )

    def cell(bg):
        return (
            f'<td style="padding:0;border-bottom:1px solid {CLR_BORDER};'
            f'border-right:1px solid {CLR_BORDER};background:{bg};'
            f'width:22px;min-width:22px;"></td>'
        )

    th_base = (
        f'padding:5px 8px;border-bottom:2px solid {CLR_BORDER};'
        f'border-right:1px solid {CLR_BORDER};background:{BG_HDR};'
        f'color:{CLR_DIM};font-size:10px;text-transform:uppercase;'
        f'letter-spacing:0.06em;font-weight:600;text-align:center;white-space:nowrap;'
    )
    name_th = f'style="{th_base}text-align:left;padding:5px 10px;"'
    hour_th = f'style="{th_base}width:22px;min-width:22px;padding:5px 3px;"'
    col_th  = f'style="{th_base}"'

    rows = []
    rows.append(
        f'<div style="font-size:13px;font-weight:600;color:{CLR_ACCENT};margin-bottom:8px;">'
        f"Tonight's Imaging Grid</div>"
    )
    rows.append(
        f'<div style="overflow-x:auto;border-radius:6px;border:1px solid {CLR_BORDER};">'
        f'<table style="border-collapse:collapse;font-family:inherit;background:{BG_MAIN};">'
    )

    # Header
    rows.append('<thead><tr>')
    rows.append(f'<th {name_th}>Object</th>')
    for h in hour_header:
        rows.append(f'<th {hour_th}>{h}</th>')
    rows.append(f'<th {col_th}>Alt</th><th {col_th}>Hrs</th><th {col_th}>Type</th>')
    rows.append('</tr></thead><tbody>')

    # Weather row
    name_style = (
        f'style="padding:4px 10px;border-bottom:1px solid {CLR_BORDER};'
        f'border-right:1px solid {CLR_BORDER};background:{BG_HDR};'
        f'color:{CLR_DIM};font-size:11px;white-space:nowrap;"'
    )
    rows.append('<tr>')
    rows.append(f'<td {name_style}>weather</td>')
    for sym in weather_syms:
        good = sym == "$"
        icon_color = "#3fb950" if good else "#f85149"
        rows.append(
            f'<td style="padding:0;border-bottom:1px solid {CLR_BORDER};'
            f'border-right:1px solid {CLR_BORDER};background:{BG_WX_GOOD if good else BG_WX_BAD};'
            f'width:22px;min-width:22px;text-align:center;font-size:9px;color:{icon_color};">{"✓" if good else "✕"}</td>'
        )
    for _ in range(3):
        rows.append(f'<td style="padding:4px 8px;border-bottom:1px solid {CLR_BORDER};border-right:1px solid {CLR_BORDER};background:{BG_HDR};"></td>')
    rows.append('</tr>')

    # Object rows
    for name, symbols, alt_str, hrs_str, dso_type, priority, need_str in object_lines:
        is_pri = priority > 5
        name_bg = BG_PRIORITY if is_pri else BG_HDR
        display = f"★ {name}" if is_pri else name
        rows.append('<tr>')
        rows.append(
            f'<td style="padding:4px 10px;border-bottom:1px solid {CLR_BORDER};'
            f'border-right:1px solid {CLR_BORDER};background:{name_bg};'
            f'color:{CLR_TEXT};font-size:12px;white-space:nowrap;">{display}</td>'
        )
        for i, sym in enumerate(symbols):
            above = sym == "+"
            wx_ok = weather_syms[i] == "$" if i < len(weather_syms) else False
            bg = (BG_ABOVE_OK if wx_ok else BG_ABOVE_WX) if above else BG_BELOW
            rows.append(cell(bg))
        rows.append(td(alt_str.strip(), BG_HDR))
        rows.append(td(hrs_str, BG_HDR))
        rows.append(td(dso_type, BG_HDR, extra="text-align:left;"))
        rows.append('</tr>')

    rows.append('</tbody></table></div>')
    return "".join(rows)


def rank_targets_tonight(instructions_path: Path | str, verbose: bool = False):
    """Rank tonight's ``waiting`` targets, best first.

    SINGLE source of truth for "tonight's target". Both the imaging grid and
    the status ticker go through this, so the two cannot disagree about the
    selection. They did until 2026-07-29: the grid read the queue raw while the
    ticker went via ``instructions.get_sorted_instructions``, which applies the
    convergence flip — so a converged high-priority target (sh2-92, priority
    100, is_dso_done True) was picked by the grid and skipped by the ticker,
    which fell through to sh2-129.

    Returns ``(rows, dark_hours, weather_by_hour)``, each row being
    ``(dso_name, good_count, max_alt, dso_type, start_time, symbols, priority)``
    sorted by priority, then good hours, then max altitude. Returns
    ``([], [], {})`` when nothing is selectable.

    Does no printing (unless *verbose*), no plotting and no file writing, so a
    polling caller such as the ticker can use it cheaply behind a short cache.
    """
    if verbose:
        print(f"DEBUG: loading instructions from: {instructions_path}")
    with open(instructions_path, "r") as f:
        objects = json.load(f)

    if verbose:
        print(f"DEBUG: total objects in file: {len(objects)}")
        for obj in objects:
            print(f"DEBUG:   {obj.get('dso', '?')} — status={obj.get('status', '?')}")

    waiting = [obj for obj in objects if obj.get("status") == "waiting"]
    if verbose:
        print(f"DEBUG: waiting objects: {len(waiting)}")
    if not waiting:
        if verbose:
            print("No waiting objects found.")
        return [], [], {}

    longitude = CFG["location"]["longitude"]
    latitude = CFG["location"]["latitude"]
    elevation = CFG["location"]["elevation"]
    observatory_name = CFG["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")

    sunset_tonight = my_observatory.sun_set_time(Time.now(), which="nearest")
    observe_time = sunset_tonight + np.linspace(-1, 14, 55) * u.hour

    start_of_dark, end_of_dark = get_dark_times(my_observatory, observe_time)
    az_horizon, al_horizon = map_az_to_horizon()

    cloud_times, cloud_covers, pp, wsp, hum = weather.get_weather_by_hour(latitude, longitude, 48)
    aq_hours, aq_aod, aq_pm25, aq_pm25_aqi = weather.get_air_quality_by_hour(latitude, longitude, 48)
    pm25_aqi_by_hour: dict[int, float] = {}
    for k in range(len(aq_hours)):
        if aq_hours[k] not in pm25_aqi_by_hour:
            pm25_aqi_by_hour[aq_hours[k]] = aq_pm25_aqi[k]
    # Use the first occurrence of each hour (today's data) — the API returns hours
    # in chronological order so duplicate hours (same hour tomorrow) must not overwrite.
    weather_by_hour: dict[int, bool] = {}
    for j in range(len(cloud_times)):
        hour = cloud_times[j]
        if hour not in weather_by_hour:
            weather_by_hour[hour] = is_weather_ok(cloud_covers[j], pp[j], wsp[j])

    # Build the list of full hours within the dark window
    t = start_of_dark.replace(minute=0, second=0, microsecond=0)
    if t < start_of_dark:
        t += datetime.timedelta(hours=1)
    dark_hours: list[datetime.datetime] = []
    while t <= end_of_dark:
        dark_hours.append(t)
        t += datetime.timedelta(hours=1)

    if not dark_hours:
        if verbose:
            print("No dark hours found tonight.")
        return [], [], {}

    # Convert to astropy Time (UTC) for altaz calculations
    hour_times = Time(
        [h.astimezone(pytz.utc).replace(tzinfo=None) for h in dark_hours],
        scale="utc"
    )

    # Build one row per object: (name, good_hour_count, max_altitude, dso_type, start_time, horizon_symbols)
    rows: list[tuple[str, int, float, str, Optional[datetime.datetime], list[str]]] = []
    for obj in waiting:
        dso_name = obj["dso"]
        # Stored RA/Dec wins over a name lookup so positional targets are not
        # silently dropped here (they have no resolvable name).
        dso = resolve_target(obj)
        if dso is None:
            if verbose:
                print(f"DEBUG: could not resolve DSO: {dso_name}")
            continue
        try:
            altaz = my_observatory.altaz(hour_times, dso)
            altitude = altaz.alt.deg
            azimuth = altaz.az.deg
        except Exception as e:
            if verbose:
                print(f"DEBUG: altaz failed for {dso_name}: {e}")
            continue

        symbols: list[str] = []
        good_count = 0
        max_alt = 0.0
        start_time: Optional[datetime.datetime] = None
        for i, dt in enumerate(dark_hours):
            h_limit = get_horizon_from_azimuth(azimuth[i], az_horizon, al_horizon)
            above = altitude[i] >= h_limit
            symbols.append("+" if above else "-")
            if above and weather_by_hour.get(dt.hour, False):
                good_count += 1
                if start_time is None:
                    start_time = dt
            if altitude[i] > max_alt:
                max_alt = float(altitude[i])
        priority = obj.get("priority", 5)
        rows.append((dso_name, good_count, max_alt, get_dso_type(dso_name), start_time, symbols, priority))

    # Sort by priority first (higher = better), then good hours, then max altitude.
    rows.sort(key=lambda x: (x[6], x[1], x[2]), reverse=True)

    return rows, dark_hours, weather_by_hour


def best_object_tonight(instructions_path: Path | str) -> tuple[str, Optional[datetime.datetime], int, str]:
    """
    Read a list of DSO objects from a JSON file, compute how many hours of
    good-weather imaging time each 'waiting' object has tonight, and print
    the results sorted best-first.

    Selection is delegated to rank_targets_tonight; this adds the console
    dump, the grid PNG and the HTML table. Callers that only need the target
    name should use rank_targets_tonight directly — it has no side effects.
    """
    rows, dark_hours, weather_by_hour = rank_targets_tonight(instructions_path, verbose=True)
    if not rows or not dark_hours:
        return "", None, 0, ""

    # Build table data shared by both the console print and the PNG
    col = 3
    label_width = max((len(r[0]) for r in rows), default=7)
    label_width = max(label_width, len("weather")) + 2

    hour_header   = [f"{dt.hour:02d}" for dt in dark_hours]
    weather_syms  = ["$" if weather_by_hour.get(dt.hour, False) else "*" for dt in dark_hours]
    try:
        from fits_processing.convergence import load_convergence, is_dso_done, frames_needed_estimate
        _conv_data = load_convergence()
    except Exception:
        _conv_data = None

    def _need_str(name: str) -> str:
        if _conv_data is None:
            return ""
        key = name.lower().replace(" ", "")
        if key not in _conv_data:
            return ""
        if is_dso_done(name):
            return "Done"
        est = frames_needed_estimate(name)
        return str(est) if est is not None else ""

    object_lines  = [
        (name, symbols, f"{max_alt:5.1f}°", f"{good_count}h", dso_type, priority, _need_str(name))
        for name, good_count, max_alt, dso_type, _start, symbols, priority in rows
    ]

    # --- Console print ---
    print("\n--- Tonight's Imaging Grid ---")
    print(" " * label_width + "".join(f"{h:>{col}s}" for h in hour_header))
    print(f"{'weather':<{label_width}}" + "".join(f"{s:>{col}s}" for s in weather_syms))
    for name, symbols, alt_str, hrs_str, dso_type, priority, need_str in object_lines:
        marker = "* " if priority > 5 else ""
        print(f"{marker}{name:<{label_width}}" + "".join(f"{s:>{col}s}" for s in symbols) + f"  {alt_str}  {hrs_str}  {dso_type}")

    # --- PNG ---
    n_obj  = len(object_lines)
    n_cols = len(dark_hours)

    # Colour maps for symbols
    ABOVE_COLOR   = "#1e5c1e"   # dark green   — above horizon
    BELOW_COLOR   = "#5c1e1e"   # dark red     — below horizon
    GOOD_WEATHER  = "#1e4a1e"   # dark green   — good weather
    BAD_WEATHER   = "#4a1e1e"   # dark red     — bad weather
    HEADER_COLOR  = "#1e2a3a"   # dark blue-grey — header row

    fig_w = max(14, 2 + n_cols * 0.55)
    fig_h = max(4,  1 + (n_obj + 2) * 0.50)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")

    # Build cell text and colours row by row
    col_labels = ["object"] + hour_header + ["alt", "hrs", "type"]
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    # Weather row
    cell_text.append(["weather"] + weather_syms + ["", "", ""])
    cell_colors.append(
        [HEADER_COLOR]
        + [GOOD_WEATHER if s == "$" else BAD_WEATHER for s in weather_syms]
        + [HEADER_COLOR, HEADER_COLOR, HEADER_COLOR]
    )

    # Object rows
    PRIORITY_COLOR = "#4a3a00"   # dark amber — prioritized target
    for name, symbols, alt_str, hrs_str, dso_type, priority, need_str in object_lines:
        display_name = f"★ {name}" if priority > 5 else name
        name_color = PRIORITY_COLOR if priority > 5 else HEADER_COLOR
        cell_text.append([display_name] + symbols + [alt_str.strip(), hrs_str, dso_type])
        cell_colors.append(
            [name_color]
            + [ABOVE_COLOR if s == "+" else BELOW_COLOR for s in symbols]
            + [HEADER_COLOR, HEADER_COLOR, HEADER_COLOR]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(list(range(len(col_labels))))

    ax.set_title("Tonight's Imaging Grid", fontsize=12, fontweight="bold", pad=10, color="white")

    # White text for all table cells
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a3a4a")

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    png_path = os.path.join(_project_root, CFG["location"]["image_grid"])
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, bbox_inches="tight", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Grid saved to {png_path}")

    grid_html = build_grid_html(object_lines, hour_header, weather_syms)

    if not rows:
        return "", None, 0, ""
    best_name, best_good_count, _, _, best_start, _, _ = rows[0]
    _publish_tonight_target(best_name, best_good_count)
    return best_name, best_start, best_good_count, grid_html


def _publish_tonight_target(dso: str, good_hours: int) -> None:
    """Record the grid's pick so every other consumer reads the same DSO.

    This function is the ONLY writer. See control.tonight_target for why the
    ticker and the end-of-night SNR report read the published value instead of
    re-running the selection.
    """
    from control import tonight_target
    tonight_target.publish(dso, good_hours)


def test_me() -> None:
    obj = is_a_dso_object("m74")
    #d, t, max_altitude = best_day_for_dso(obj)
    #hours = t.seconds/3600
    #print (hours)
    #print (d, t)
    p1, p2, p3, weather_ok, weather_msg = show_plots(obj)
    #print ("weather ok", weather_ok)



if __name__ == '__main__':

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    best_object_tonight(os.path.join(_project_root, config.data()["location"]["instructions"]))