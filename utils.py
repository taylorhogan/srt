
import os



def set_install_dir ():
    path = os.path.dirname(__file__)

    if os.path.exists(path):
        os.chdir(path)

    return path
