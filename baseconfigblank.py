from collections import OrderedDict


class BaseConfig(object):
    _base_config = OrderedDict({

        "mqtt": {
            "user_name": "",
            "mqtt_password": "",
            "broker_url": "",
            "port": 8886,
        },
        "weather": {
            "api_key": "",
        },
        "pushover":
            {
                "token": "",
                "user": "",
            },
        "location":
            {
                "city": "Boston",
                "latitude": "",
                "longitude": ""
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
