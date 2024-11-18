import logging
import time

from config import baseconfig as cfg
import moon
import pushover
import sun as s
import watchdog
import weather

debug_roof_open_switch = False
debug_roof_closed_switch = True

config = cfg.FlowConfig().config
config["Globals"]["Observatory State"] = "Booting Up"''


def general_message_with_image(message, image=None):
    pushover.push_message(message)
    # social_server.post_social_message(message, image)
    print(message)


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
    print("Roof is closed")
    debug_roof_closed_switch = True


def open_roof():
    print("Opening Roof")
    t_open = watchdog.Watchdog(5, debug_on_open_roof)
    t_open.start()
    return True


def close_roof():
    print("Closing Roof")
    t_close = watchdog.Watchdog(5, debug_on_close_roof)
    t_close.start()
    return True


def timer_done():
    print("Timer done")
    timer_going = False


def state_machine():
    old_is_night, angle = s.is_night()
    currently_imaging = False
    old_start_imaging = None
    old_stop_imaging = None
    old_weather = None
    timer = watchdog.Watchdog(15, timer_done)

    version = config["version"]["date"]
    general_message_with_image("Start Observatory with version " + version, image="./db/day.jpeg")

    while not get_manual_stop():
        try:
            is_night, angle = s.is_night()
            good_weather, weather_description = weather.is_good_weather()
            if good_weather:
                weather_description += "Imaging is Good"
            else:
                weather_description += "Imaging is Bad"

            if old_weather is not good_weather and is_night:
                general_message_with_image(weather_description, None)
            old_weather = good_weather

            many_stars = enough_stars()
            no_clouds = is_clouds_ok()

            roof_open = is_roof_open()
            roof_closed = is_roof_closed()
            scope_safe = is_scope_safe()
            moon_ok = moon.is_moon_ok()
            if timer.is_timer_going():
                pass
            current_time = time.time()

            # What is going on when it's dark?

            old_is_night = is_night
            start_imaging = roof_closed and good_weather and moon_ok and many_stars and is_night and scope_safe and not roof_open and not currently_imaging
            stop_imaging = currently_imaging and roof_open and (
                    not good_weather or not many_stars or not is_night or not moon_ok)
            if is_night and not old_is_night and not currently_imaging and not start_imaging:
                d, c, w, m = weather.get_weather()
                message = "Not Imaging " + "\n " + d
                config["Globals"]["Observatory State"] = "Sleep Mode, Roof Closed"
                config["Globals"]["Current Image"] = "./db/day.jpeg"
                general_message_with_image(message, image=config["Globals"]["Current Image"])
            if start_imaging:
                start_imaging_time = current_time
                currently_imaging = True
                timer.start()
                open_roof()
                config["Globals"]["Observatory State"] = "Imaging, Roof Open"
                config["Globals"]["Current Image"] = "./db/m31.jpg"
                general_message_with_image("Starting Imaging", image=config["Globals"]["Current Image"])

            elif stop_imaging:
                stop_imaging_time = current_time
                time_elapsed = stop_imaging_time - start_imaging_time

                currently_imaging = False
                timer.start()
                close_roof()
                message = "Stopping Imaging, Imaging time: " + time.strftime("%Hhours %Mminutes",
                                                                             time.gmtime(time_elapsed))
                config["Globals"]["Observatory State"] = "Imaging, Roof Open"
                config["Globals"]["Current Image"] = "./db/m31.jpg"
                general_message_with_image(message, image=config["Globals"]["Current Image"])

        except:
            log = logging.getLogger()
            log.exception("Message for you, sir!")

        if not is_night or not good_weather:
            time.sleep(20 * 60)
        else:
            time.sleep(5 * 60)


def main():
    state_machine()


if __name__ == '__main__':
    main()
