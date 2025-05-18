from collections import OrderedDict
from re import fullmatch

import config_private
from config_private import PrivateConfig


class PublicConfig():
    _config = dict({
        "install_location": '/Users/taylorhogan/Documents/tmh',

        "version": {
            "date": "2025.5.18.1"
        },

        "logger": {
            "logging": ""
        },
        "location": {
            "city": "Hartford",
            "latitude": 41.8096,
            "longitude": -72.8305,
            "elevation": 100,
            "observatory_name": "Iris"
        },

        "nina": {
            "image_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets"
        },

        "camera safety": {
            "parked azimuth deg": 56,
            "parked altitude deg": 5,
            "roof template": "./base_images/roof_marker.jpg",
            "parked template": "./base_images/parked_marker.png",
            "open pos": (1, 1),
            "closed pos": (490, 422),
            "parked pos": (929, 454),
            "scope_view": "./base_images/scope_view.jpg",
            "processed_view": "./base_images/processed.jpg",
            "no_image": "./base_images/no_image.jpg",
            "valid_data": False,
            "received_count": 0

        },
        "Calendar":
            {
                "image": "lightblue",
                "weather": "pink",
                "service": "orange"
            },
        "Globals":
            {
                "Observatory State": "In Development",
                "Imaging Tonight": "Unknown"
            }

    })

    def data(self):
      return self._config


if __name__ == "__main__":
    mt = PublicConfig()
    print(mt.data["mqtt"])
