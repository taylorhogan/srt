import asyncio
import logging

import os,sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from cmd_processing import super_user_commands
from hardware_control import kasa_utils as ku
from hardware_control import utl_shelly
from configs import config
from utils import utils

if __name__ == "__main__":

    cfg = config.data()
    logger = utils.set_logger()
    utils.set_install_dir()
    cfg["logger"]["logging"] = logger
    logger.info('Start Start Sequence')



    logger.info('Setting safety to safe and not imaging')

    dev_map = asyncio.run(ku.make_discovery_map())
    instructions = (dict
        (
        {
            "Telescope mount": 'on',
            "Iris door light": 'off',
            "Iris inside light": "off",
            "Driveway lights": "off",
            "Grill Lights":"off",
            "Iris landscape lights": "off",
            "Main landscape lights": "off"

        }
    ))

    # Phase 2b (2026-09-25): mount power asks the conductor (Invariant B).
    # The prelude runs this right after the roof was confirmed open, so the
    # conductor's evidence is fresh. A refusal leaves the mount off; the
    # prelude then fails at the connect, which is the intended outcome.
    from iris import client as conductor
    allowed, reason, _reply = conductor.request_mount(
        "MOUNT_POWER_REQUESTED", "start.py", None, {"why": "prelude"})
    if not allowed:
        logger.error('Start sequence: mount NOT powered -- conductor refused: %s', reason)
        instructions.pop("Telescope mount", None)
        try:
            from utils import pushover
            pushover.push_message("Prelude: mount not powered -- conductor refused: %s" % reason)
        except Exception:
            logger.exception('pushover failed')
    elif reason:
        logger.warning('Start sequence: %s (mount power proceeding)', reason)

    results = asyncio.run(ku.kasa_do(dev_map, instructions))
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        logger.error('Start sequence: %d of %d switches FAILED: %s',
                     len(failed), len(results), ', '.join(failed))
    else:
        logger.info('Turning off lights (all %d verified)', len(results))

    utl_shelly.set_dehumidifier(False)
    logger.info('Turning off dehumidifier')
    logger.info('End Start Sequence')
    print ("Done with startup")
