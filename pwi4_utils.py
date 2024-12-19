from pwi4_client import PWI4
import config
import logging


def get_is_parked (logger):

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
        park_altitude = config["camera safety"]["parked altitude deg"]
        parked_azimuth=config["camera safety"]["parked azimuth deg"]

        delta_altitude = abs(park_altitude-alt)
        delta_azimuth = abs(parked_azimuth-az)
        print ("Delta al", delta_altitude)
        print ("Delta az", delta_azimuth)
        if  delta_altitude < 1 and delta_azimuth < 1:
            return True
        else:
            return False


    except:
        logger.info('Problem')
        logger.exception("Exception")
        return False



if __name__ == "__main__":
    parked = get_is_parked()
