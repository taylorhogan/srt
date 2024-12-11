import requests

import config_public as cfg
import moon

config = cfg.PublicConfig().data


def get_weather_string(json):
    y = json["main"]
    z = json["weather"]
    weather_description = z[0]["description"]
    cloud_percentage = json["clouds"]["all"]
    wind_speed = json["wind"]["speed"]
    current_temperature = y["temp"]
    current_humidity = y["humidity"]
    moon_phase = moon.get_moon_phase()

    description = "Temp (F): " + str(int((current_temperature - 273.15) * 9 / 5 + 32)) + "\n"
    description += "Humidity: " + str(current_humidity) + "\n"
    description += "Description: " + str(weather_description) + "\n"
    description += "Cloud%: " + str(cloud_percentage) + "\n"
    description += "Wind Speed: " + str(wind_speed) + "\n"
    description += "Moon Phase: " + "{:10.2f}".format(moon_phase) + "\n"

    return description, cloud_percentage, wind_speed, moon_phase


def get_weather():
    api_key = config["weather"]["api_key"]
    city_name = config["location"]["city"]
    # base_url variable to store url
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    # Give city name
    # city_name = input("Enter city name : ")
    # complete_url variable to store
    # complete url address
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name

    # get method of requests module
    # return response object
    response = requests.get(complete_url)

    # json method of response object
    # convert json format data into
    # python format data
    x = response.json()

    # Now x contains list of nested dictionaries
    # Check the value of "cod" key is equal to
    # "404", means city is found otherwise,
    # city is not found
    if x["cod"] != "404":
        description, clouds, wind_speed, moon_phase = get_weather_string(x)
        message_list = list()
        message_list.append({
            'topic': 'flow/weather',
            'payload': str(description),
            'qos': 1,
            'retain': True,
        })
        # something wrong with certs
        # publish.multiple(
        #     message_list,
        #     transport='tcp',
        #     hostname='localhost',
        #     port=8883,
        #     client_id='',
        #     keepalive=60,
        #     auth={'username': 'indi-allsky', 'password': 'Foo14me!'},
        #     tls={'ca_certs': '/etc/ssl/certs/ca-certificates.crt', 'cert_reqs': ssl.CERT_NONE, 'insecure': True},
        #
        # )
        print(description)
        return description, clouds, wind_speed, moon_phase

    else:
        print(" City Not Found ")
        return "Unknown City"


def is_good_weather():
    weather_description, cloud_percentage, wind_speed, moon_phase = get_weather()
    if cloud_percentage > 40:
        return False, weather_description
    if wind_speed > 20:
        return False, weather_description
    if "rain" in weather_description:
        return False, weather_description
    if "snow" in weather_description:
        return False, weather_description
    if "sleet" in weather_description:
        return False, weather_description

    return True, weather_description
