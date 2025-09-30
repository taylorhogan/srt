# for all dso objects
#   remove those that have been imaged, done
#   how many hours will it be visible > some altitude
# https://astroplan.readthedocs.io/en/stable/
import datetime
import math

import weather
import operator
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytz
import astroplan.plots
from astroplan import FixedTarget
from astroplan import Observer
from astroplan import moon_illumination
from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time
from matplotlib import dates
import logging


import config

cfg = config.data()


def is_a_dso_object(name):
    try:
        dso = FixedTarget.from_name(name)
        return dso
    except :
        return None


def get_horizon_from_azimuth(this_az, az, al):
    for idx in range(len(az) - 1):
        if this_az >= az[idx] and this_az <= az[idx + 1]:
            return al[idx]
    return al[-1]


def get_dark_times (my_observatory, observe_time):

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

def find_alt_az_horizon_times(dso, my_observatory, observe_time):
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


def plot_my_dso_and_horizon(dso, my_observatory, observe_time):
    altitude, azimuth, horizon, start_time, finish_time, elapsed_time, max_altitude = find_alt_az_horizon_times(dso, my_observatory,
                                                                                                  observe_time)

    masked_altitude = np.ma.array(altitude, mask=altitude < 0)
    local_datetime = my_observatory.astropy_time_to_datetime(observe_time)

    plt.figure(figsize=(10, 12))
    ax = plt.gca()



    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    elevation = cfg["location"]["elevation"]
    observatory_name = cfg["location"]["observatory_name"]
    city = cfg["location"]["city"]

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
    print (illumination)
    ax.plot(local_datetime, moon_alt, color='blue', label = 'Moon(' + str(illumination) + "%)",linewidth=2)

    #plot the cloud cover
    cloud_times,cloud_covers,pp,wsp = weather.get_weather_by_hour(latitude, longitude, 24)
    time_format = "%Y-%m-%d %H:%M"
    clipped_cloud = []
    clipped_pp = []
    clipped_wsp = []
    weather_ok = True
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
                if cloud_covers[j] > 80:
                    weather_ok = False
                    print (cloud_time_hour, cloud_covers[j])

                if pp[j] > 20:
                    weather_ok = False
                    print (cloud_time_hour, pp[j])




            if found_hour:
                break


    ax.plot(local_datetime, clipped_cloud, color='red', label = 'Cloud Cover',linewidth=2)
    ax.plot(local_datetime, clipped_pp, color='pink', label='Prob. Precip.',linewidth=2)
    ax.plot(local_datetime, clipped_wsp, color='black', label='Wind Speed % of 25 mph',linewidth=2)


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

    return weather_ok


def show_plots(dso):
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    elevation = cfg["location"]["elevation"]
    observatory_name = cfg["location"]["observatory_name"]

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
    try:

        astroplan.plots.plot_sky(dso, my_observatory, observe_time)
        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        sky_path = os.path.join(scratch_dir, "sky.png")
        plt.savefig(sky_path)
        plt.clf()
    except:
        logger = logging.getLogger(__name__)
        logger.info('Problem')
        logger.exception("Exception")

    try:
        print("a")
        weather_ok = plot_my_dso_and_horizon(dso, my_observatory, observe_time)
        altitude_path = os.path.join(scratch_dir, "horizon.png")
        print ("b")
        plt.savefig(altitude_path)
        plt.clf()
    except:
        logger = logging.getLogger(__name__)
        logger.info('Problem')
        logger.exception("Exception")


    return altitude_path, image_path, sky_path, weather_ok


def get_above_horizon_time(dso, time):
    longitude = cfg["location"]["longitude"]
    latitude = cfg["location"]["latitude"]
    elevation = cfg["location"]["elevation"]
    observatory_name = cfg["location"]["observatory_name"]

    location = EarthLocation.from_geodetic(longitude * u.deg, latitude * u.deg, elevation * u.m)
    my_observatory = Observer(location=location, name=observatory_name, timezone="US/Eastern")

    sunset_tonight = my_observatory.sun_set_time(time, which='nearest')
    sunrise_tonight = my_observatory.sun_rise_time(time, which='nearest')

    observe_time = sunset_tonight
    observe_time = observe_time + np.linspace(-1, 14, 55) * u.hour
    altitude, azimuth, horizon, start_time, finish_time, elapsed_time, max_altitude = find_alt_az_horizon_times(dso, my_observatory,
                                                                                                  observe_time)
    return elapsed_time, max_altitude


def air_mass (altitude):
    h = math.sin(math.radians(altitude))
    if h == 0:
        return 100
    else:
        return 1.0 / math.sin(math.radians(altitude))

def map_az_to_horizon():
    ax = plt.gca()
    data = []
    az = []
    al = []
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name + "/my.hrz")
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
    ir_name = os.path.dirname(__file__)
    scratch_dir = os.path.join(dir_name + "/scratch")
    sky_path = os.path.join(scratch_dir, "h.png")
    plt.savefig(sky_path)
    plt.clf()
    return az, al


def enumerate_days_of_year():

    now = datetime.datetime.now()

    first_day = datetime.date(now.year, now.month, now.day)


    for day_number in range(1, 365):
        current_day = first_day + datetime.timedelta(days=day_number - 1)
        yield current_day


def best_day_for_dso(dso):
    # Example usage for the year 2025
    best_time = None
    best_date = None
    max_altitude = 0
    for  day in enumerate_days_of_year():
        try:
            this_day = datetime.datetime(day.year, day.month, day.day, 14, 0, 0)
            this_time = Time(this_day)
            above_time, max_altitude = get_above_horizon_time(dso, this_time)

            if above_time is not None:
                print(day.year, day.month, day.day, above_time / 3600, "{:.2f}".format((air_mass(max_altitude))))
                if best_time is None or above_time > best_time:
                    best_date = this_day
                    best_time = above_time
        except:
            return None, None, None
    if best_date is None:
        return None, None, None
    else:
        return best_date, best_time, max_altitude


def test_me():
    obj = is_a_dso_object("m74")
    #d, t, max_altitude = best_day_for_dso(obj)
    #hours = t.seconds/3600
    #print (hours)
    #print (d, t)
    p1, p2, p3, weather_ok = show_plots(obj)
    #print ("weather ok", weather_ok)



if __name__ == '__main__':
    test_me()