import time
import logging
import os,sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hardware_control.pwi4_client import PWI4
from utils import utils
from configs import config

logger = logging.getLogger(__name__)


def park_scope ():

    pwi4 = PWI4()

    s = pwi4.status()
    logger.info("Mount connected: %s", s.mount.is_connected)
    logger.info("Trying to park")


    if not s.mount.is_connected:
        logger.info("Connecting to mount...")
        s = pwi4.mount_connect()
        logger.info("Mount connected: %s", s.mount.is_connected)
        if not s.mount.is_connected:
            return False
    logger.info("Mount is connected")
    logger.info("Start park")
    pwi4.mount_park()
    logger.info("Waiting for 30")
    time.sleep(30)
    logger.info("End park")

    return True



def get_is_parked ():

    try:
        cfg = config.data()


        pwi4 = PWI4()

        s = pwi4.status()
        logger.info("Mount connected: %s", s.mount.is_connected)
        logger.debug("%s", s)
        if not s.mount.is_connected:
            logger.info("Connecting to mount...")
            s = pwi4.mount_connect()
            logger.info("Mount connected: %s", s.mount.is_connected)
            if not s.mount.is_connected:
                return False

        logger.info("Mount is connected")
        alt = s.mount.altitude_degs
        az = s.mount.azimuth_degs
        logger.info("Azimuth: %s", az)
        logger.info("Altitude: %s", alt)

        moving = s.mount.is_slewing or s.mount.is_tracking
        logger.info("Moving: %s", moving)

        if moving:
            return False
        park_altitude = cfg["camera safety"]["parked altitude deg"]
        parked_azimuth=cfg["camera safety"]["parked azimuth deg"]

        delta_altitude = abs(park_altitude-alt)
        delta_azimuth = abs(parked_azimuth-az)
        logger.info("Delta alt: %s", delta_altitude)
        logger.info("Delta az: %s", delta_azimuth)
        if  delta_altitude < 1 and delta_azimuth < 1:
            return True
        else:
            return False


    except Exception as e:
        logger.exception("get_is_parked failed — PWI4 unreachable or mount error: %s", e)
        return False



if __name__ == "__main__":
    utils.set_install_dir()

    logging.basicConfig(filename='../iris.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    parked = get_is_parked()
    print (parked)
