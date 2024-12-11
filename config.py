
from config_private import PrivateConfig
from config_public import PublicConfig


def data():
     a = PublicConfig().data()
     b = PrivateConfig().data()
     a.update(b)
     return a




if __name__ == "__main__":
    fc = data()
    print (fc)
