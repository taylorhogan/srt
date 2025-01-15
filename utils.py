
import config
import os

cfg = config.data()


def set_install_dir ():

    if os.name == 'posix':
        path = cfg["InstallL"]
    else:
        path = cfg["Install"]

    if os.path.exists(path):
        os.chdir(path)

    return path
