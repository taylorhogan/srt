class PublicConfig():
    _config = dict({

        "globals": {
            "mastodon instance": None,
            "mqtt_client": None

        },
        "version": {
            "date": "2026.4.21.1"
        },

        "logger": {
            "logging": ""
        },
        "location": {
            "city": "Boston",
            "latitude": 41.8096,
            "longitude": -72.8305,
            "elevation": 100,
            "observatory_name": "Iris",
            "timezone": "America/New_York",
            "instructions":"local/my_instructions.json",
            "image_grid":"local/imaging_grid.png"
    },

        "scratch": {
            "directory": "iris_astronomy/scratch",
            "latest_jpg": "latest_annotated.jpg"
        },

        "nina": {
            "image_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets",
            "sequence_output": "C:/Users/iriso/Documents/N.I.N.A/Sequences/full_for_tonight.json",
            "sequence_input": "C:/Users/iriso/Documents/N.I.N.A/Sequences/cdk_full_sequence.json",
            "sequence_input1": "/home/taylor/Documents/srt/nina_gen/nina_sequence_gen.py",
            "arc_sec_per_pixel": 0.26
    },

        "camera safety": {
            "parked azimuth deg": 57.2,
            "parked altitude deg": 1.0,
            "closed template": "./base_images/closed_marker.jpg",
            "parked template": "./base_images/parked_marker.png",
            "open template": "./base_images/open_marker.jpg",
            "open pos": (172, 142),
            "closed pos": (829, 152),
            "parked pos": (590, 290),

            "accuracy": 150,
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
            },

        "web_chat": {
            "enabled": True,
            "port": 8095,
            "host": "0.0.0.0",
            "mastodon_mirror": False,
            "max_history": 500,
        },

        "sync": {
            "rsync_path": "C:/Users/iriso/Documents/cwrsync_6.4.7_x64_free/bin/rsync",
            "source": "iriso@100.95.7.19:/cygdrive/c/Users/iriso/Documents/N.I.N.A/Targets",
            "destination": "~/Desktop"
        },

        "pegasus": {
            "unity_url": "http://localhost:32000",
            "driver_key": "",  # DriverUniqueKey from Unity; blank = auto-discover
        },

    })

    def data(self):
        return self._config


if __name__ == "__main__":
    mt = PublicConfig()
    print(mt.data["mqtt"])
