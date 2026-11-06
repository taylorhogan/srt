from collections import OrderedDict


class PrivateConfig(object):
    _config = dict({

        "weather": {
            "api_key": "",
        },
        # Optional: the Anthropic key for the gallery prose "api" backend
        # (scripts/article_prose). The Spark writes with its local model
        # without one.
        "anthropic": {
            "api_key": "",
        },
        "pushover": {
            "token": "",
            "user": "",
        },
        "mastodon": {
            "access_token": '',
            "api_base_url": '',
            "client_secret": "",
            "client_key": "",
            "me": ''
        },

        "Install": "C:",
        "InstallL":"/home",
        "Super Users": {
            '', ''

        }

    })

    def data(self):
        return self._config



if __name__ == "__main__":
    mt = PrivateConfig()
    print(sorted(mt.data))
