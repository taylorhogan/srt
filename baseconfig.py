from collections import OrderedDict


class BaseConfig(object):
    _base_config = OrderedDict({
        "version": {
            "date": "2024.10.25.25"
        },
        "html":{
            "path":"/var/www/html/cds"
        },
        "logger": {
            "topic": "flow/log",
            "file": "iris.log",
            "logging": ""
        },
        "mqtt": {
            "user_name": "indi-allsky",
            "password": "Foo14me!",
            "broker_url": "raspberrypi.local",
            "port": 1883,
            "ota_picture":"iris/inside_jpg",
            "roof_state":"iris/roof_state",
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
        "nina": {
            "image_dir": "C:/Users/Primalucelab/Documents/N.I.N.A",
            "image_dir2": "/Users/taylorhogan/Desktop"
        },
        "camera safety":{
            "parked azimuth deg": 56,
            "parked altitude deg": 5,
            "roof template": "./base_images/roof_marker.jpg",
            "parked template": "./base_images/parked_marker.png",
            "open pos":(1,1),
            "closed pos":(490, 422),
            "parked pos":(929, 454),
            "scope_view": "./base_images/scope_view.jpg",
            "processed_view": "./base_images/processed.jpg",
            "no_image":"./base_images/no_image.jpg",
            "valid_data": False,
            "received_count": 0



            },
        "Globals":
            {
                "Observatory State": "In Development",
                "Imaging DSO": "Unknown"
                ""
            }

    })

    @property
    def base_config(self):
        return self._base_config

    @base_config.setter
    def base_config(self, new_base_config):
        pass  # read only


class FlowConfig(BaseConfig):

    def __init__(self):
        self._config = self.base_config.copy()  # populate initial values

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_config):
        pass  # read only


class Test(object):
    def __init__(self):
        self._config_obj = FlowConfig()

    def main(self):
        pass


if __name__ == "__main__":
    mt = Test()
    print(mt._config_obj.config["mqtt"])
