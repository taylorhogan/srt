import os
from hardware_control import kasa_utils as ku
import asyncio

import logging
from configs import config
from pathlib import Path
from datetime import datetime



__logger = None


def create_timestamped_filename(base_name, extension=""):
    """Create a filename with base name plus current date and time."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if extension and not extension.startswith("."):
        extension = "." + extension
    return f"{base_name}_{timestamp}{extension}"

class _SuppressNoisyWarnings(logging.Filter):
    _PATTERNS = ("fit may be unsuccessful",)

    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


def set_logger ():
    global __logger
    if __logger is None:
        cfg = config.data()
        path = Path (__file__).parent.parent.resolve() / 'iris.log'
        print("logging at " + str(path))
        logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.getLogger("py.warnings").addFilter(_SuppressNoisyWarnings())
        logger = logging.getLogger(__name__)
        cfg["logger"]["logging"] = logger
        __logger = logger

    return __logger

def set_install_dir():
    path = Path(__file__).parent.parent.resolve()


    if os.path.exists(path):
        os.chdir(path)

    return path

def get_device_map ():

    logger = set_logger()
    dev_map = {}
    try:
        dev_map = asyncio.run(ku.make_discovery_map())
    except:
        logger.info('Problem')
        logger.exception("Exception")
    return dev_map


if __name__ == '__main__':
    set_install_dir()
    dev_map = get_device_map()
    print  (dev_map)