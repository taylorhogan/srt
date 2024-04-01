import time

import pushover
import sun as s
import watchdog
import weather as w

debug_roof_open_switch = False
debug_roof_closed_switch = True


def is_weather_good():
    return True


def enough_stars():
    return True


def is_clouds_ok():
    return True


def is_scope_safe():
    return True


def is_roof_open():
    return debug_roof_open_switch


def is_roof_closed():
    return debug_roof_closed_switch


def get_manual_stop():
    return False


def get_manual_start():
    return False


def debug_on_open_roof():
    global debug_roof_open_switch, debug_roof_closed_switch
    debug_roof_open_switch = True
    print("Roof is open")
    debug_roof_closed_switch = False


def debug_on_close_roof():
    global debug_roof_open_switch, debug_roof_closed_switch
    debug_roof_open_switch = False
    print ("Roof is closed")
    debug_roof_closed_switch = True


def open_roof():
    print ("Opening Roof")
    t_open = watchdog.Watchdog(5, debug_on_open_roof)
    t_open.start()
    return True


def close_roof():
    print ("Closing Roof")
    t_close = watchdog.Watchdog(5, debug_on_close_roof)
    t_close.start()
    return True


def timer_done():
    print("Timer done")
    timer_going = False


def search_for_actions():
    currently_imaging = False
    old_start_imaging = None
    old_stop_imaging = None
    timer = watchdog.Watchdog(15, timer_done)
    #pushover.push_message("Starting Observatory")
    while not get_manual_stop():
        good_weather = w.is_good_weather()
        many_stars = enough_stars()
        no_clouds = is_clouds_ok()
        is_night, angle = s.is_night()
        roof_open = is_roof_open()
        roof_closed = is_roof_closed()
        scope_safe = is_scope_safe()
        if timer.is_timer_going():
            pass

        start_imaging = roof_closed and good_weather and many_stars and is_night and scope_safe and not roof_open and not currently_imaging
        stop_imaging = currently_imaging and roof_open and (not good_weather or not many_stars or not is_night)
        if start_imaging:
            pushover.push_message("Starting Imaging")
            currently_imaging = True
            timer.start()
            open_roof()
        elif stop_imaging:
            pushover.push_message("Stopping Imaging")
            currently_imaging = False
            timer.start()
            close_roof()

        if not is_night or not good_weather:
            time.sleep(10)
        else:
            time.sleep(5)



search_for_actions()
