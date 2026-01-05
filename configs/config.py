import os, sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs.config_private import PrivateConfig
from configs.config_public import PublicConfig


def data():
     a = PublicConfig().data()
     b = PrivateConfig().data()
     a.update(b)
     return a




if __name__ == "__main__":
    fc = data()
    print (fc)
