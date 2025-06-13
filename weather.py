import asyncio
import datetime
import os

import python_weather

import config

cfg = config.data()


async def get_sunrise_sunset_internal() -> [datetime, datetime]:
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        # fetch a weather forecast from a city

        city = cfg["location"]["city"]
        forecast = await client.get(city)
        today = forecast.daily_forecasts[0]
        tomorrow = forecast.daily_forecasts[1]

        sunset = today.sunset
        sunrise = today.sunrise

        return sunrise, sunset


async def get_sunrise_sunset() -> [datetime, datetime]:
    return await get_sunrise_sunset_internal()


async def get_weather(current) -> [str, bool]:
    # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        # fetch a weather forecast from a city
        city = cfg["location"]["city"]
        forecast = await client.get(city)

        description = "Current Temperature: " + str(forecast.temperature) + "\n"
        description += "Current Humidity: " + str(forecast.humidity) + "\n"
        description += "Current Wind Speed: " + str(forecast.wind_speed) + "\n"
        description += "Current Visibility: " + str(forecast.visibility) + "\n"
        description += "Current description: " + str(forecast.description) + "\n"

        today = forecast.daily_forecasts[0]
        tomorrow = forecast.daily_forecasts[1]
        description += "Moon Illumination: " + str(today.moon_illumination) + "\n"
        description += "Moon Rise: " + str(today.moonrise) + "\n"
        description += "Moon Set: " + str(today.moonset) + "\n"
        description += "Night Hours: " + str(24 - today.sunlight) + "\n"

        sunset = today.sunset
        sunrise = today.sunrise
        description += "Sunset: " + str(sunset) + "\n"
        description += "Sunrise " + str(sunrise) + "\n"
        sunset_hour = sunset.hour
        sunrise_hour = sunrise.hour
        if current:
            return description, True
        else:

            description = ""
            weather_ok = True
            description += ("\nState  Hour  %Dry %Cloud Gust Description\n")
            hours = []

            for hourly in today:
                if hourly.time.hour >= sunset_hour:
                    hours.append(hourly)
            for hourly in tomorrow:
                if hourly.time.hour <= sunrise_hour:
                    hours.append(hourly)

            for hourly in hours:
                this_hourly = "GOOD "
                if hourly.chances_of_remaining_dry < 80 or hourly.cloud_cover > 50 or hourly.wind_gust > 20:
                    weather_ok = False
                    this_hourly = "BAD  "
                description += this_hourly + str(hourly.time) + "  " + str(
                    hourly.chances_of_remaining_dry) + "    " + str(hourly.cloud_cover) + "     " + str(
                    hourly.wind_gust) + "     " + hourly.description + "\n"

            if weather_ok:
                description += "Will image tonight"
            else:
                description += "Will NOT image tonight"
        return description, weather_ok



def get_current_weather(current) -> [str, bool]:
    # see https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
    # for more details
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print ("before get weather")
    description, weather_ok =  asyncio.run (get_weather(current))
    print ("after get weather")
    return description, weather_ok


if __name__ == '__main__':
    description, weather_ok = get_current_weather(False)
    print(description)
