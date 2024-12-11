from __future__ import print_function, division
import datetime
from PyAstronomy import pyasl
import numpy as np

def get_moon_phase ():

    # Convert calendar date to JD
    # using the datetime package
    now = datetime.datetime.now()
    jd = datetime.datetime(now.year, now.month, now.day)
    jd = pyasl.jdcnv(jd)

    mp = pyasl.moonphase(jd)
    return mp[0]*100


def is_moon_ok ():
    moon_phase = get_moon_phase()
    if moon_phase > 90:
        return False, moon_phase
    else:
        return True, moon_phase