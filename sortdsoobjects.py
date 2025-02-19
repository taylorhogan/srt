# for all dso objects
#   remove those that have been imaged, done
#   how many hours will it be visible > some altitude
# https://astroplan.readthedocs.io/en/stable/
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import astroplan
import datetime
from astroplan import FixedTarget
from astroplan import Observer
from astroplan.plots import plot_finder_image
from astropy.coordinates import EarthLocation
from astropy.time import Time
import os
import config
from matplotlib import dates
import operator
import pytz


cfg = config.data()


def is_a_dso_object(name):
    try:
        dso = FixedTarget.from_name(name)
        return dso
    except NameError:
        return None

def get_horizon_from_azimuth (this_az, az, al):
    for idx in range (len(az) - 1):
        if this_az >= az[idx] and this_az <= az[idx + 1]:
            return al[idx]
    return al[-1]


def plot_my_dso_and_horizon (dso, my_observatory, observe_time):
    altitude = (91 * u.deg - my_observatory.altaz(observe_time, dso).alt) * (1 / u.deg)
    altitude = (my_observatory.altaz(observe_time, dso).alt) * (1 / u.deg)

    # Azimuth MUST be given to plot() in radians.
    azimuth = my_observatory.altaz(observe_time, dso).az * (1 / u.deg)
    # print out in local time

    local_datetime = my_observatory.astropy_time_to_datetime(observe_time)

    horizon = []
    az, al = map_az_to_horizon()
    for idx in range(len(local_datetime)):
        h = get_horizon_from_azimuth (azimuth[idx], az, al)
        horizon.append(h)
        print(local_datetime[idx], altitude[idx], azimuth[idx], h)



    masked_altitude = np.ma.array(altitude, mask=altitude < 0)


    ax = plt.gca()
    style_kwargs = None


    local_tz = pytz.timezone('America/New_York')
    ax.plot(local_datetime, masked_altitude)
    ax.plot(local_datetime, horizon)

    ax.set_xlim([local_datetime[0], local_datetime[-1]])
    date_formatter = dates.DateFormatter('%H',tz=local_tz)
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

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
    plt.title(dso.name)



    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()

    # Output.
    return ax

def show_plots(dso):
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    elevation = cfg["location"]["elevation"]
    observatory_name = cfg["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")
    time = Time.now()
    now = datetime.datetime.now()
    sunset_tonight = my_observatory.sun_set_time(time, which='nearest')
    sunrise_tonight = my_observatory.sun_rise_time(time, which='nearest')
    print(my_observatory.is_night(time))
    print(sunset_tonight.iso)
    print(sunrise_tonight.iso)

    object_is_up = my_observatory.target_is_up(time, dso)
    print(object_is_up)
    observe_time = sunset_tonight
    observe_time = observe_time + np.linspace(-1, 14, 55) * u.hour


    dir_name = os.path.dirname(__file__)
    scratch_dir = os.path.join(dir_name + "/scratch")
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)




    ax, hdu = plot_finder_image(dso, fov_radius=42*u.arcmin, reticle=True)

    image_path = os.path.join(scratch_dir, "image.png")
    plt.savefig(image_path)
    plt.clf()

    astroplan.plots.plot_sky(dso, my_observatory, observe_time)
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    sky_path = os.path.join(scratch_dir, "sky.png")
    plt.savefig(sky_path)
    plt.clf()

    plot_my_dso_and_horizon(dso, my_observatory, observe_time)
    altitude_path = os.path.join(scratch_dir, "horizon.png")

    plt.savefig(altitude_path)
    plt.clf()





    return altitude_path, image_path, sky_path


def map_az_to_horizon ():


    ax = plt.gca()
    data = []
    az =[]
    al =[]
    with open("/Users/taylorhogan/Documents/tmh/tmh.hrz", 'r') as file:
        for line in file:
            # Strip whitespace and split the line into two columns
            columns = line.strip().split()
            if len(columns) == 2:
                # Assuming the columns are numeric, convert to float
                    col1 = float(columns[0])
                    col2 = float(columns[1])
                    az.append(col1)
                    al.append(col2)


    plt.plot (az, al)
    ir_name = os.path.dirname(__file__)
    scratch_dir = os.path.join("/Users/taylorhogan/Documents/tmh/" + "/scratch")
    sky_path = os.path.join(scratch_dir, "h.png")
    plt.savefig(sky_path)
    plt.clf()
    return az, al

if __name__ == '__main__':

    obj = is_a_dso_object("ngc2903")
    show_plots (obj)