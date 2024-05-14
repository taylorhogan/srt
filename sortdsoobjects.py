# for all dso objects
#   remove those that have been imaged, done
#   how many hours will it be visible > some altitude
# https://astroplan.readthedocs.io/en/stable/
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import astroplan
from astroplan import FixedTarget
from astroplan import Observer
from astroplan.plots import plot_finder_image
from astropy.coordinates import EarthLocation
from astropy.time import Time

import baseconfig as cfg

config = cfg.FlowConfig().config


def is_a_dso_object(name):
    dso = FixedTarget.from_name(name)
    return dso


def show_plots(dso):
    longitude = config["location"]["longitude"]
    latitude = config["location"]["latitude"]
    elevation = config["location"]["elevation"]
    observatory_name = config["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")
    time = Time.now()
    sunset_tonight = my_observatory.sun_set_time(time, which='nearest')
    sunrise_tonight = my_observatory.sun_rise_time(time, which='nearest')
    print(my_observatory.is_night(time))
    print(sunset_tonight.iso)
    print(sunrise_tonight.iso)

    object_is_up = my_observatory.target_is_up(time, dso)
    print(object_is_up)
    observe_time = sunset_tonight
    observe_time = observe_time + np.linspace(-5, 5, 55) * u.hour
    astroplan.plots.plot_altitude(dso, my_observatory, observe_time)

    plt.savefig("altitude.png")
    plt.show()

    ax, hdu = plot_finder_image(dso)

    plt.savefig("image.png")
    plt.show()

    astroplan.plots.plot_sky(dso, my_observatory, observe_time)
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.savefig("sky.png")
    plt.show()

    return "altitude.png", "image.png", "sky.png"

