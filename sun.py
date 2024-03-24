import datetime
from pysolar.solar import *
import baseconfig as cfg


def get_sun_angle():
    config = cfg.FlowConfig().config

    latitude = config["location"]["latitude"]
    longitude = config["location"]["longitude"]
    date = datetime.datetime.now(datetime.timezone.utc)
    return get_altitude(float(latitude), float(longitude), date)


def is_night():
    sun_angle = get_sun_angle ()
    return sun_angle < -10, sun_angle
