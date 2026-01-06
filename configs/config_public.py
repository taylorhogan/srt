class PublicConfig():
    _config = dict({

        "globals": {
            "mastodon instance": None,
            "mqtt_client": None

        },
        "version": {
            "date": "2026.1.06.1"
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
            "closed template": "/base_images/closed_marker.jpg",
            "parked template": "/base_images/parked_marker.png",
            "open template": "/base_images/open_marker.jpg",
            "open pos": (172, 142),
            "closed pos": (829, 152),
            "parked pos": (590, 290),

            "accuracy": 150,
            "scope_view": "/base_images/scope_view.jpg",
            "processed_view": "/base_images/processed.jpg",
            "no_image": "/base_images/no_image.jpg",
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
            },

    })

    def data(self):
        return self._config


if __name__ == "__main__":
    mt = PublicConfig()
    print(mt.data["mqtt"])
