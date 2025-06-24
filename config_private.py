from collections import OrderedDict


class PrivateConfig(object):
    _config = dict({

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
            "me": '@Thogan'
        },

        "Install": "C:/Users/iriso/Documents/development/iris",
        "InstallL":"/home/taylorhogan/Documents/tmh/",
        "Super Users": {
            'Thogan', 'tmhobservatory'

        }

    })

    def data(self):
        return self._config



if __name__ == "__main__":
    mt = PrivateConfig()
    print(mt.data["mqtt"])
