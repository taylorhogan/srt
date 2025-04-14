import datetime

from pysolar.solar import *

import config


def get_sun_angle():
    cfg = config.data()

    latitude = cfg["location"]["latitude"]
    longitude = cfg["location"]["longitude"]
    date = datetime.datetime.now(datetime.timezone.utc)
    return get_altitude(float(latitude), float(longitude), date)


def is_night():
    sun_angle = get_sun_angle()
    return sun_angle < -10, sun_angle
