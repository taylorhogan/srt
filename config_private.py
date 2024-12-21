from collections import OrderedDict


class PrivateConfig(object):
    _config = dict({


        "mqtt": {
            "user_name": "indi-allsky",
            "password": "Foo14me!",
            "broker_url": "raspberrypi.local",
            "port": 1883,
            "ota_picture": "iris/inside_jpg",
            "roof_state": "iris/roof_state",
            "ota_state": "iris/ota_state",
            "picture_date": "iris/picture_date",
            "observatory_state?": "iris/state"

        },
        "weather": {
            "api_key": "ed8baa5b5b9c64b42fafc6836e75a3a3",
        },
        "pushover": {
            "token": "a7fycu94si1ctfnubk3sfqhbsioct2",
            "user": "ggd66ig5wrpo8z9y7eyncfihor4b33",
        },
        "mastodon": {
            "access_token": 'M67r7ztzqc8yWiNcokctuz0T5dJsL_P7Bh-o1VTHm9U',
            "api_base_url": 'https://mastodon.social/',
            "client_secret": "9c_OTf7f_Rmy3jPGpawgulxbZNLG3qqr_d5sZzlONqo",
            "client_key": "bknnjpd3ATuzldY2PGzqGYE9tzynywijhZ90ZsDC1b8",
            "instance": None,
            "me": '@Thogan'
        },
        "location": {
            "city": "Hartford",
            "latitude": 41.8096,
            "longitude": -72.8305,
            "elevation": 100,
            "observatory_name": "Iris"
        },

        "Install":"C:/Users/iriso\Documents\development\iris",

        "Super Users": {
            'Thogan', 'tmhobservatory'

        }

    })

    def data(self):
        return self._config



if __name__ == "__main__":
    mt = PrivateConfig()
    print(mt.data["mqtt"])
