from io import StringIO

import requests

import config

import python_weather

import asyncio
import os

cfg = config.data()


async def get_sunrise_sunset() -> [bool]:
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        # fetch a weather forecast from a city
        forecast = await client.get('West Hartford')



        today = forecast.daily_forecasts[0]
        tomorrow = forecast.daily_forecasts[1]


        sunset = today.sunset
        sunrise = today.sunrise

        sunset_hour = sunset.hour
        sunrise_hour = sunrise.hour

        re
async def get_weather() -> [str,bool]:
    # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        # fetch a weather forecast from a city
        forecast = await client.get('West Hartford')

        description = "Current Temperature: " + str(forecast.temperature) + "\n"
        description +=  "Current Humidity: " + str(forecast.humidity) + "\n"
        description += "Current Wind Speed: " + str(forecast.wind_speed) + "\n"
        description += "Current Visibility: " + str(forecast.visibility) + "\n"
        description +=  "Current description: " + str(forecast.description) + "\n"

        today = forecast.daily_forecasts[0]
        tomorrow = forecast.daily_forecasts[1]
        description += "Moon Illumination: " + str(today.moon_illumination) + "\n"
        description += "Moon Rise: " + str(today.moonrise) + "\n"
        description += "Moon Set: " + str(today.moonset) + "\n"
        description += "Night Hours: " + str (24 - today.sunlight) +"\n"

        sunset = today.sunset
        sunrise = today.sunrise
        description += "Sunset: " + str(sunset) + "\n"
        description += "Sunrise " + str(sunrise) + "\n"
        sunset_hour = sunset.hour
        sunrise_hour = sunrise.hour

        weather_ok = True
        description += ("\nState / Hour / % Dry/ % Cloud "
                        ""
                        ""
                        ""
                        "/Description\n")
        for hourly in today:
            if hourly.time.hour >= sunset_hour:
                this_hourly="GOOD "
                if hourly.chances_of_remaining_dry < 80 or hourly.cloud_cover > 50:
                    weather_ok = False
                    this_hourly = "BAD "
                description +=  this_hourly +  str(hourly.time) + " " + str(hourly.chances_of_remaining_dry) + " " + str(hourly.cloud_cover) + " " + hourly.description + "\n"

        for hourly in tomorrow:
            if hourly.time.hour <= sunrise_hour:
                this_hourly = "GOOD "
                if hourly.chances_of_remaining_dry < 80 or hourly.cloud_cover > 50:
                    weather_ok = False
                    this_hourly = "BAD "
                description += this_hourly + str(hourly.time) + " " + str(hourly.chances_of_remaining_dry) + " " + str(
                    hourly.cloud_cover) + " " + hourly.description + "\n"
        if weather_ok:
            description += "Will image tonight"
        else:
            description += "Will NOT image tonight"
    return description, weather_ok


def get_current_weather()-> [str,bool]:
    # see https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
    # for more details
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    description, weather_ok = asyncio.run(get_weather())
    return description, weather_ok

if __name__ == '__main__':
    description, weather_ok = get_current_weather()
    print(description)

