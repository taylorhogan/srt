from collections import OrderedDict


class BaseConfig(object):
    _base_config = OrderedDict({
        "version": {
            "date": "2024.6.23.01"
        },
        "html":{
            "path":"/var/www/html/cds"
        },
        "logger": {
            "topic": "flow/log",
            "file": "log.txt",
        },
        "mqtt": {
            "user_name": "indi-allsky",
            "password": "Foo14me!",
            "broker_url": "raspberrypi.local",
            "port": 1883
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
            "observatory_name": "tmh"
        },
        "Super Users": {
            "Id": '@Thogan'
        },
        "camera safety":{
            "image_path": "./base_images/inside.jpg",
            "open_template": "./base_images/open_template.jpg",
            "close_template": "./base_images/close_template.jpg",
            "parked_template": "./base_images/parked_template.jpg",
            "open_pos":(1,1),
            "close_pos":(2,2),
            "parked_pos":(3,3)


            },
        "Globals":
            {
                "Observatory State": "Unknown",
                "Imaging DSO": "Unknown"
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
