from io import StringIO

import requests

import config

import python_weather

import asyncio
import os

cfg = config.data()


async def get_weather() -> str:
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
        # get the weather forecast for a few days
        for daily in forecast:
            print(daily)

            # hourly forecasts
            for hourly in daily:
                print(f' --> {hourly!r}')

        return description


def get_current_weather():
    # see https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
    # for more details
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    description = asyncio.run(get_weather())
    return description
if __name__ == '__main__':

    print(get_current_weather())
