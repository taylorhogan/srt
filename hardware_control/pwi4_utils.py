from pwi4_client import PWI4
import config
import logging
from utils import utils
import time

def park_scope ():

    pwi4 = PWI4()

    s = pwi4.status()
    print("Mount connected:", s.mount.is_connected)
    print("trying to park!")


    if not s.mount.is_connected:
        print("Connecting to mount...")
        s = pwi4.mount_connect()
        print("Mount connected:", s.mount.is_connected)
        if not s.mount.is_connected:
            return False
    print("Mount is connected")
    print ("start park")
    pwi4.mount_park()
    print ("waiting for 30")
    time.sleep(30)
    print ("end park")

    return True



def get_is_parked ():

    try:
        cfg = config.data()


        pwi4 = PWI4()

        s = pwi4.status()
        print("Mount connected:", s.mount.is_connected)
        print (s)
        if not s.mount.is_connected:
            print("Connecting to mount...")
            s = pwi4.mount_connect()
            print("Mount connected:", s.mount.is_connected)
            if not s.mount.is_connected:
                return False

        print ("Mount is connected")
        alt = s.mount.altitude_degs
        az = s.mount.azimuth_degs
        print (az)
        print (alt)

        moving = s.mount.is_slewing or s.mount.is_tracking
        print (moving)

        if moving:
            return False
        park_altitude = cfg["camera safety"]["parked altitude deg"]
        parked_azimuth=cfg["camera safety"]["parked azimuth deg"]

        delta_altitude = abs(park_altitude-alt)
        delta_azimuth = abs(parked_azimuth-az)
        print ("Delta al", delta_altitude)
        print ("Delta az", delta_azimuth)
        if  delta_altitude < 1 and delta_azimuth < 1:
            return True
        else:
            return False


    except:
        return False



if __name__ == "__main__":
    utils.set_install_dir()

    logging.basicConfig(filename='../iris.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    parked = get_is_parked()
    print (parked)
