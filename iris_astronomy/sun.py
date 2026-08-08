"""Where the sun is, and whether it is dark enough to be observing.

The single definition of "night" for the project. Anything that needs to know
should call is_night() rather than reimplement the test, so the observatory and
the things reporting on it cannot disagree about whether it is working.

Built on astropy rather than pysolar. pysolar is listed in requirements.txt but
was never actually installed in the venv, and this module had no callers, so
the pysolar version had never run. astropy is already a hard dependency that
the live skymap uses for exactly this transform, which means this module works
on a box that deploys by git pull without anyone installing anything.
"""
import datetime
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

from configs import config

# Degrees below the horizon before the sky is dark enough to be worth
# measuring. -10 is dimmer than civil twilight (-6) and brighter than
# astronomical dark (-18): the sky still holds twilight here, so a star count
# taken near this boundary is legitimately low and should not be read as cloud.
NIGHT_SUN_ALT_DEG = -10.0


def get_sun_angle():
    """The sun's altitude in degrees right now. Negative is below the horizon."""
    cfg = config.data()
    loc_cfg = cfg["location"]
    loc = EarthLocation.from_geodetic(
        float(loc_cfg["longitude"]) * u.deg,
        float(loc_cfg["latitude"]) * u.deg,
        float(loc_cfg.get("elevation", 0)) * u.m)
    now = Time(datetime.datetime.now(datetime.timezone.utc))
    return float(get_sun(now).transform_to(
        AltAz(obstime=now, location=loc)).alt.deg)


def is_night():
    """(is_it_night, sun_altitude_deg)."""
    sun_angle = get_sun_angle()
    return sun_angle < NIGHT_SUN_ALT_DEG, sun_angle


if __name__ == "__main__":
    night, alt = is_night()
    print("sun altitude %.2f deg -> %s" % (alt, "night" if night else "day"))
