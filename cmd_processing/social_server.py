import asyncio
import multiprocessing
import subprocess
import threading
from pathlib import Path
import datetime
import logging
import os
from datetime import date
import json
import time
import sys
from typing import Any, Optional
from bs4 import BeautifulSoup
try:
    from mastodon import Mastodon, StreamListener
    from mastodon.streaming import CallbackStreamListener
    _MASTODON_AVAILABLE = True
except ImportError:
    _MASTODON_AVAILABLE = False
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import requests as _requests_lib
from iris_astronomy import astro_dso_visibility, obs_calendar, show_dso
from configs import config
from end_points import end
from fits_processing import fitstojpg, fitsfwhm, sky_brightness as sb, session_stats as ss
from control import instructions
from cmd_processing import super_user_commands as su
from cmd_processing import message_bus
from cmd_processing import jobs
from nina_gen import nina_sequence_gen
from utils.utils import topic_to_sched
from utils import utils

_PREVIEW_MP_CONTEXT = multiprocessing.get_context("spawn")

# Exit code used to signal start_srt.py to git pull and restart
RESTART_EXIT_CODE = 42




_json_payload = None


def get_dso_object_name(words: list[str], index: int) -> Optional[str]:
    if len(words) > index + 1:
        dso = words[index + 1]
    else:
        return None
    if len(words) > index + 2:
        dso = dso + " " + words[index + 2]
    return dso


def image_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
    example image m 13
    """
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = astro_dso_visibility.is_a_dso_object(dso_name)
        if object is not None:
            if instructions.add_dso_object_instruction(dso_name, "", account):
                post_social_message(dso_name + " Added to list of objects to image\n")
            else:
                post_social_message(dso_name + " Already in the queue\n")
        else:
            post_social_message(dso_name + " Not a known object\n")


def best_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
       example best m 13
    """
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = astro_dso_visibility.is_a_dso_object(dso_name)
        if object is not None:
            best_date, best_time, max_altitude = astro_dso_visibility.best_day_for_dso(object)
            if best_date is not None:
                formatted_date = best_date.strftime("%Y-%m-%d")
                formatted_air_mass = "{:.2f}".format((astro_dso_visibility.air_mass(max_altitude)))
                post_social_message(dso_name + " is above horizon for " + str(best_time) + " on " + formatted_date  + " with air mass of "+formatted_air_mass )
                hours = best_time.seconds / 3600
                if hours > 3:
                    image_cmd(words, index, m, account)

            else:
                post_social_message(dso_name + " is never above horizon")
        else:
            post_social_message(dso_name + " Not a known object\n")


def best_radec_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
       Like 'best' but for an explicit RA/Dec position.

       With a name, the position is queued for imaging (best night > 3h),
       carrying its RA/Dec so it never needs a Simbad name lookup. Without a
       name it is analysis-only (reports the best night, queues nothing).

       example: bestradec wr134 20:10:14 +36:10:35   (named ⇒ queueable)
                bestradec 12:30:49 +12:23:28          (RA hours, Dec deg)
                bestradec 187.7 12.39                 (decimal degrees)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroplan import FixedTarget
    import re

    args = words[index + 1:]
    if len(args) >= 3:
        name, ra_str, dec_str = args[0].strip(), args[1].strip(), args[2].strip()
    elif len(args) == 2:
        name, ra_str, dec_str = None, args[0].strip(), args[1].strip()
    else:
        post_social_message(
            "Usage: bestradec [<name>] <ra> <dec>  "
            "(e.g. wr134 20:10:14 +36:10:35, or 187.7 12.39; name enables queueing)")
        return

    # In colon-delimited sexagesimal the h/m/s/d letters are redundant; a stray
    # one (e.g. '10:1:20s') makes astropy silently misparse to a wild value
    # rather than erroring, so strip letters whenever a ':' is present.
    def _clean(tok: str) -> str:
        return re.sub(r"[a-zA-Z]", "", tok) if ":" in tok else tok
    ra_str, dec_str = _clean(ra_str), _clean(dec_str)

    # Sexagesimal (h:m:s / d:m:s) ⇒ RA in hours, Dec in degrees; otherwise treat
    # both as decimal degrees.
    sexagesimal = any(c in ra_str + dec_str for c in (":", "h", "d"))
    try:
        if sexagesimal:
            coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        else:
            coord = SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg)
    except Exception as e:
        post_social_message(f"Could not parse RA/Dec '{ra_str} {dec_str}': {e}")
        return

    pos = (f"RA {coord.ra.to_string(unit=u.hour, sep=':', precision=0)} "
           f"Dec {coord.dec.to_string(sep=':', precision=0, alwayssign=True)}")
    label = f"{name} ({pos})" if name else pos
    target = FixedTarget(coord=coord, name=name or pos)
    best_date, best_time, max_altitude = astro_dso_visibility.best_day_for_dso(target)
    if best_date is None:
        post_social_message(f"{label} is never above horizon")
        return

    formatted_date = best_date.strftime("%Y-%m-%d")
    formatted_air_mass = "{:.2f}".format(astro_dso_visibility.air_mass(max_altitude))
    post_social_message(
        f"{label} is above horizon for {best_time} on {formatted_date} "
        f"with air mass of {formatted_air_mass} (max altitude {max_altitude:.1f}°)")

    # Queue named positions worth imaging, storing RA/Dec so the scheduler and
    # queue resolve coordinates without a name lookup. Mirrors 'best' (> 3h).
    if name and best_time.seconds / 3600 > 3:
        if instructions.add_dso_object_instruction(
                name, "", account,
                ra_deg=float(coord.ra.deg), dec_deg=float(coord.dec.deg)):
            post_social_message(f"{name} added to the imaging queue (RA/Dec stored)\n")
        else:
            post_social_message(f"{name} already in the queue\n")


def tonight_cmd(words: list[str], index: int, m: Mastodon, account: str) -> bool:
    print("in tonight cmd", words, index)
    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    instructions_path = os.path.join(_project_root, cfg["location"]["instructions"])

    best_name, best_start, best_good_hours, grid_html = astro_dso_visibility.best_object_tonight(instructions_path)
    post_social_message(f"Tonight's best object: {best_name} ({best_good_hours}h good imaging)")
    if grid_html:
        post_html_message(grid_html)

    if best_name:
        obj = instructions.resolve_target_by_name(best_name)
        if obj is not None:
            horizon, image, sky, weather_ok = astro_dso_visibility.show_plots(obj)

            if horizon is not None:
                post_social_message(f"{best_name} — altitude & conditions", horizon)
            if sky is not None:
                post_social_message(f"{best_name} — sky chart", sky)
            if weather_ok:
                post_social_message("Weather ok tonight")
            else:
                post_social_message("Weather not ok tonight")
            post_dso_preview(best_name)
            return weather_ok
        else:
            post_social_message(f"{best_name} not a known object")
    return False



def version_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    # Observatory State
    cfg = config.data()

    reply = "Version: " + cfg["version"]["date"] + "\n"
    reply += "Observatory Status: " + cfg["Globals"]["Observatory State"]
    post_social_message(reply)

def wait_for_mqtt_message(client: Any, userdata: Any, msg: Any) -> None:
    global _json_payload
    print ("setting payload to ")
    _json_payload = json.loads(msg.payload.decode("utf-8"))
    _json_payload = json.dumps(_json_payload)
    logger = logging.getLogger(__name__)
    logger.info("Received message: " + _json_payload)
    print("to ", _json_payload)




def status_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    cfg = config.data()

    end.determine_roof_state_visually(account)

    mode = su.get_mode()
    safe = "Safe" if su.is_safe() else "Unsafe"
    imaging = su.get_imaging_state().value.replace("_", " ").title()

    sched = su.get_scheduler_state()

    post_social_message(
        f"━━ Observatory Status ━━\n"
        f"Scheduler : {sched['state']}\n"
        f"Target    : {sched['dso']}\n"
        f"Imaging   : {sched['will image tonight']}\n"
        f"━━ Control ━━\n"
        f"Mode      : {mode.title()}\n"
        f"Safety    : {safe}\n"
        f"State     : {imaging}"
    )

    # Attach a short live audio clip of the observatory to the status report.
    try:
        from sentry import audio_classify
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])
        wav_path = os.path.join(scratch_dir, "status_audio.wav")
        recorded = audio_classify.record_wav(15, wav_path)
        if recorded:
            post_social_message("Observatory audio (15s)", audio=recorded)
        else:
            post_social_message("Audio capture unavailable")
    except Exception:
        logging.getLogger(__name__).exception("status audio capture failed")
        post_social_message("Audio capture unavailable")




def db_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    instructions.create_instructions_table()




def calendar_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    # Observatory State
    cfg = config.data()
    today = date.today()

    obs_calendar.print_month(today.year, today.month, cfg)
    post_social_message("", "cal.png")


def history_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
    Show recent command history. example: history  or  history 10
    """
    from cmd_processing import cmd_history
    n = 20
    if len(words) > index + 1:
        try:
            n = int(words[index + 1])
        except ValueError:
            pass
    post_social_message(cmd_history.format_history(n))


def help_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """Show command list, or detailed help for one command. Usage: help [<cmd>] or ? [<cmd>]"""
    from cmd_processing import help_registry as hr

    # words = [<mention>, <cmd-name>, <maybe target>, ...]
    target = words[index + 1].strip() if len(words) > index + 1 else ""
    if target:
        post_social_message(hr.format_command(target))
        return
    post_social_message(
        hr.format_list(include_super=su.is_super_user(account))
        + "\n\nFull docs: https://github.com/taylorhogan/srt/blob/main/docs/commands.md"
    )


def _last_sunset(cfg: dict) -> datetime.datetime:
    """Return the datetime of the most recent sunset (naive, local time)."""
    from astral import LocationInfo
    from astral.sun import sun as _sun
    loc  = cfg["location"]
    city = LocationInfo(loc["city"], "USA", loc["timezone"],
                        loc["latitude"], loc["longitude"])
    now  = datetime.datetime.now()
    # Today's sunset — if it hasn't happened yet, use yesterday's
    today_sun    = _sun(city.observer, date=now.date())
    today_sunset = today_sun["sunset"].replace(tzinfo=None)
    if today_sunset > now:
        import datetime as _dt
        yesterday    = now.date() - _dt.timedelta(days=1)
        yest_sun     = _sun(city.observer, date=yesterday)
        return yest_sun["sunset"].replace(tzinfo=None)
    return today_sunset


def latest_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    image_dir = cfg["nina"]["image_dir"]
    latest_fits = fitstojpg.get_latest_file(image_dir, "fits")
    if latest_fits is None:
        post_social_message("No FITS frames found")
        return

    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])
    output_path = Path(os.path.join(scratch_dir, cfg["scratch"]["latest_jpg"]))
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]

    filter_name = None
    try:
        from astropy.io import fits as _fits
        with _fits.open(latest_fits) as hdul:
            filter_name = str(hdul[0].header.get("FILTER", "")).strip() or None
    except Exception:
        pass

    sky_data = sb.measure_sky(Path(str(latest_fits)), arcsec_per_pixel=arcsec_per_pixel)

    latest_jpg, _mean_px, _mean_ecc = fitsfwhm.save_fwhm(
        Path(str(latest_fits)), output_path,
        arcsec_per_pixel=arcsec_per_pixel,
        annotate=False,
        filter_name=filter_name,
        sky_data=sky_data,
    )
    post_social_message("Latest frame", str(latest_jpg))


def schedule_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
    Generate a NINA sequence for tonight's best object. example: schedule
    """
    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    instructions_path = os.path.join(_project_root, cfg["location"]["instructions"])

    best_name, best_start, best_good_hours, _ = astro_dso_visibility.best_object_tonight(instructions_path)
    if not best_name:
        post_social_message("No suitable object found for tonight")
        return

    dso = instructions.resolve_target_by_name(best_name)
    if dso is None:
        post_social_message(f"{best_name} is not a known object")
        return

    from astropy.time import Time
    above_horizon, _ = astro_dso_visibility.get_above_horizon_time(dso, Time.now())
    above_horizon_seconds = above_horizon.total_seconds() if above_horizon is not None else None

    template_path = Path(os.path.join(_project_root, cfg["nina"]["sequence_input"]))
    output_path = Path(cfg["nina"]["sequence_output"])

    try:
        filter_plan = nina_sequence_gen.generate_sequence(
            template_path=template_path,
            dso_name=best_name,
            ra_hours=dso.coord.ra.hour,
            dec_degrees=dso.coord.dec.deg,
            output_path=output_path,
            above_horizon_seconds=above_horizon_seconds,
        )
        plan_str = "  ".join(f"{f}×{n}" for f, n in filter_plan.items()) if filter_plan else "no filter plan"
        post_social_message(
            f"Schedule generated for {best_name} ({best_good_hours:.1f}h above horizon)\n"
            f"{plan_str}\n→ {output_path.name}"
        )
    except Exception as e:
        post_social_message(f"Failed to generate schedule for {best_name}: {e}")


def _preview_worker(dso_name: str, scratch_dir: str, web_chat_port: int) -> None:
    """Runs in a spawned child process — fetches DSO survey image and posts via web chat API."""
    import os, sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize
    from iris_astronomy import show_dso
    import requests

    def _post(msg, image_path=None):
        url = f"http://localhost:{web_chat_port}/api/post"
        data = {"message": msg}
        if image_path:
            data["image_path"] = image_path
        requests.post(url, data=data, timeout=30)

    try:
        data, _ = show_dso.get_dso_image(dso_name, show=False)
        fov_w, fov_h = show_dso.field_of_view(
            show_dso.FOCAL_LENGTH_MM, show_dso.SENSOR_WIDTH_MM, show_dso.SENSOR_HEIGHT_MM,
        )
        norm = ImageNormalize(data, interval=ZScaleInterval())
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
        ax.imshow(data, origin="lower", cmap="gray", norm=norm, aspect="equal")
        ax.set_title(
            f"{dso_name.upper()}  |  DSS2 Red\n"
            f"FOV {fov_w * 60:.1f}' × {fov_h * 60:.1f}'",
            color="white", pad=10,
        )
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(scratch_dir, f"show_{dso_name.replace(' ', '_')}.jpg")
        fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        _post(dso_name.upper(), out_path)
    except Exception as e:
        _post(f"Could not fetch preview for {dso_name}: {e}")


def post_dso_preview(dso_name: str) -> None:
    """Fetch a DSS2 survey image in a child process and post it to the chat.

    Uses multiprocessing (spawn) so a hung SkyView/SIMBAD call can be
    forcefully terminated after the timeout rather than leaking a thread.
    """
    _TIMEOUT_SEC = 120

    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])
    web_chat_port = cfg.get("web_chat", {}).get("port", 8095)

    def _watchdog():
        p = _PREVIEW_MP_CONTEXT.Process(
            target=_preview_worker,
            args=(dso_name, scratch_dir, web_chat_port),
            daemon=True,
        )
        p.start()
        p.join(timeout=_TIMEOUT_SEC)
        if p.is_alive():
            p.terminate()
            p.join()
            post_social_message(
                f"Preview for {dso_name} timed out after {_TIMEOUT_SEC}s — SkyView/SIMBAD did not respond"
            )

    jobs.spawn(_watchdog)



def show_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
    Fetch and post a survey image of a DSO. example: show ngc 891
    """
    dso_name = get_dso_object_name(words, index)
    if dso_name is None:
        post_social_message("Usage: show <dso name>  e.g. show ngc 891")
        return
    post_social_message(f"Fetching survey image for {dso_name.upper()} …")
    post_dso_preview(dso_name)



def speedtest_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """Run an internet speed test. example: speedtest"""
    post_social_message("Running speed test, this takes about 30-60 seconds…")

    def _run():
        from sentry.internet_classify import get_speed
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(get_speed)
            try:
                result = future.result(timeout=120)
            except FuturesTimeout:
                post_social_message("Speed test timed out after 2 minutes — check network")
                return

        if result:
            post_social_message(
                f"━━ Speed Test ━━\n"
                f"Download : {result['download_mbps']} Mbps\n"
                f"Upload   : {result['upload_mbps']} Mbps\n"
                f"Ping     : {result['ping_ms']} ms\n"
                f"Server   : {result['server']}"
            )
        else:
            post_social_message("Speed test failed — check internet connection")

    jobs.spawn(_run)


keywords = {
    "tonight": tonight_cmd,
    "best": best_cmd,
    "bestradec": best_radec_cmd,
    "image": image_cmd,
    "db": db_cmd,
    "version": version_cmd,
    "status": status_cmd,
    "latest": latest_cmd,
    "schedule": schedule_cmd,
    "calendar": calendar_cmd,
    "show": show_cmd,
    "speedtest": speedtest_cmd,
    "history": history_cmd,
    "help": help_cmd,
    "?": help_cmd
}


def do_command(sentence: str, m: Optional[Any] = None, account: str = "") -> None:

    if not su.is_super_user(account):
        return

    cmd = sentence.lower()
    words = cmd.split(" ")
    seen_base_command = False
    seen_super_user_commands = False
    #todo does not seem to remove leading blanks

    logger = logging.getLogger(__name__)
    logger.info("Got Command: " + sentence)

    if len(words) < 2:
        post_social_message("Command not recognized, ? for help")
        return

    action = keywords.get(words[1].strip(), "no_key")
    logger.info("Action: " + str(action))

    if action != "no_key":
        action(words, 1, m, account)
        seen_base_command = True

    if seen_base_command is False:
        seen_super_user_commands = su.do_super_user_command(words, account)
    if seen_base_command is False and seen_super_user_commands is False:
        post_social_message("Command not recognized, ? for help")


def do_notification(notification: Any, m: Mastodon) -> None:
    cmd = ""
    try:
        print(notification['type'])
        account = notification.account.acct
        html = notification.status.content.lower()
        note_type = notification['type']
        if note_type == 'mention' or note_type == 'reblog':
            cmd = BeautifulSoup(html, 'html.parser').get_text()
            do_command(cmd, m, account)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.info('Problem')
        logger.exception("Exception")
        try:
            post_social_message("Oops I had a problem processing a command")
        except Exception:
            logger.exception("Failed to post error message")


class TheStreamListener(StreamListener):

    def on_update(self, status: dict) -> None:
        print(f"Got update: {status['content']}")

    def on_notification(self, notification: Any) -> None:
        cfg = config.data()
        logger = logging.getLogger(__name__)
        mastodon = cfg["globals"]["mastodon instance"]
        do_notification(notification, mastodon)


def get_mastodon_instance() -> Mastodon:
    cfg = config.data()
    logger = logging.getLogger(__name__)
    access_token = cfg["mastodon"]["access_token"]
    api_base_url = cfg["mastodon"]["api_base_url"]
    mastodon = Mastodon(access_token=access_token, api_base_url=api_base_url)
    return mastodon


def post_html_message(html: str) -> None:
    """Post an HTML-rendered message to the web chat (no Mastodon mirror)."""
    logger = logging.getLogger(__name__)
    cfg = config.data()
    if message_bus.is_initialized():
        message_bus.post_message("", html=html)
    else:
        try:
            port = cfg.get("web_chat", {}).get("port", 8095)
            _requests_lib.post(
                f"http://localhost:{port}/api/post",
                data={"message": "", "html": html},
                timeout=30,
            )
        except Exception:
            logger.exception("Failed to post HTML message via web chat API")


def post_social_message(message: str, image: Optional[str] = None, vis: Optional[str] = None,
                        audio: Optional[str] = None) -> None:
    logger = logging.getLogger(__name__)
    cfg = config.data()

    # Route through in-process message bus if available (web server process),
    # otherwise fall back to HTTP POST (scheduler process, standalone scripts).
    if message_bus.is_initialized():
        message_bus.post_message(message, image, audio_path=audio)
    else:
        try:
            from cmd_processing import jobs
            port = cfg.get("web_chat", {}).get("port", 8095)
            data = {"message": message}
            if image:
                data["image_path"] = image
            if audio:
                data["audio_path"] = audio
            # In a process-isolated worker the job is bound on this thread;
            # forward it so the post lands on that worker's card, not the
            # system feed.
            job_id = jobs.get_current_job()
            if job_id:
                data["job_id"] = job_id
            _requests_lib.post(f"http://localhost:{port}/api/post", data=data, timeout=30)
        except Exception:
            logger.exception("Failed to post message via web chat API")


def handle_mention(notification: Any) -> None:
    cfg = config.data()
    if notification.type == "mention":
        print(notification.status.content)
        mastodon = cfg["globals"]["mastodon instance"]
        do_notification(notification, mastodon)


def _tailscale_ip() -> Optional[str]:
    """This host's Tailscale IPv4, or None. Best-effort, short timeout."""
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0:
            ip = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
            return ip or None
    except Exception:
        pass
    return None


def _primary_ip() -> Optional[str]:
    """Best-effort primary outbound IPv4 (the address other hosts would reach)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # selects the default route; sends nothing
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def _webchat_urls(host: str, port: int, hostname: str) -> list[str]:
    """Build a deduped list of clickable web-chat URLs.

    When bound to all interfaces (0.0.0.0/::), advertise the addresses an
    operator would actually use: the Tailscale IP first (Iris is reached over
    Tailscale), then the MagicDNS/LAN hostname, then the LAN IP, then localhost.
    When bound to a specific host, just use that.
    """
    if host not in ("0.0.0.0", "", "::"):
        return [f"http://{host}:{port}"]
    candidates = [
        _tailscale_ip(),
        hostname if hostname and hostname != "unknown" else None,
        _primary_ip(),
        "localhost",
    ]
    urls: list[str] = []
    for c in candidates:
        if c:
            u = f"http://{c}:{port}"
            if u not in urls:
                urls.append(u)
    return urls


def start_interface() -> None:
    """Start the web chat server (FastAPI/uvicorn)."""
    import uvicorn
    from cmd_processing import web_server

    cfg = config.data()
    web_cfg = cfg.get("web_chat", {})
    host = web_cfg.get("host", "0.0.0.0")
    port = web_cfg.get("port", 8095)

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(_project_root, web_cfg.get("upload_dir", "saved_dso"))
    web_server.init(images_dir)

    import socket
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    post_social_message(f"Starting Version {cfg['version']['date']} on {hostname}")

    urls = _webchat_urls(host, port, hostname)
    if urls:
        post_social_message("Web chat ready at " + "   ".join(urls))

    # Optionally also listen on Mastodon
    if web_cfg.get("mastodon_mirror", False):
        try:
            mastodon = get_mastodon_instance()
            listener = CallbackStreamListener(notification_handler=handle_mention)
            mastodon.stream_user(listener, run_async=True, reconnect_async=True, timeout=600)
        except Exception:
            logging.getLogger(__name__).exception("Failed to start Mastodon listener")

    uvicorn.run(web_server.app, host=host, port=port, log_level="info", loop="asyncio")


def _startup_orphan_check() -> None:
    """After a (re)launch, detect an imaging run that was interrupted with the
    roof still open and the scope unparked, and alert the operator.

    This is the safety net for the combined-failure case (e.g. a machine-wide
    OOM that kills both this server and NINA): a plain relaunch would otherwise
    leave the roof open with nothing to notice. Alert-only by default; set
    ``web_chat.auto_safe_on_orphan = True`` to also run the emergency safe
    shutdown automatically.
    """
    logger = logging.getLogger(__name__)
    try:
        time.sleep(20)  # let the process settle and the scheduler reset its state
        from sentry import vision_safety
        from hardware_control import pwi4_utils
        from utils import pushover
        cfg = config.data()

        if su.is_nina_running():
            return  # imaging genuinely in progress — not an orphan

        parked, closed, is_open, _ = vision_safety.visual_status()
        try:
            mount_parked = pwi4_utils.get_is_parked()
        except Exception:
            mount_parked = parked

        if is_open and not mount_parked:
            inside_view = cfg["camera safety"]["scope_view"]
            msg = (
                "Iris startup check: roof appears OPEN with no NINA running and the "
                "mount not parked — possible interrupted run. Check the observatory."
            )
            logger.warning(msg)
            try:
                pushover.push_message(msg, inside_view)
            except Exception:
                logger.exception("startup orphan check: pushover failed")
            if cfg.get("web_chat", {}).get("auto_safe_on_orphan", False):
                logger.warning("auto_safe_on_orphan enabled — running emergency safe shutdown")
                su.unsafe_cmd(["", "stop!"], "iris")
    except Exception:
        logger.exception("startup orphan check failed")


def main() -> None:
    utils.set_install_dir()
    cfg = config.data()
    logger = logging.getLogger(__name__)

    cfg["logger"]["logging"] = logger
    path = utils.set_install_dir()
    path = os.path.join(path, 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info('Started Social Server')

    # Initialize Mastodon instance if mirroring is enabled
    web_cfg = cfg.get("web_chat", {})
    if web_cfg.get("mastodon_mirror", False):
        cfg["globals"]["mastodon instance"] = get_mastodon_instance()

    # Initialize the message bus
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(_project_root, web_cfg.get("upload_dir", "saved_dso"))
    message_bus.init(images_dir, max_history=web_cfg.get("max_history", 500))
    jobs.init()

    try:
        mqtt_client = utils.connect_mqtt()
        cfg["globals"]["mqtt_client"] = mqtt_client
        mqtt_client.subscribe(utils.topic_from_sched)
        mqtt_client.on_message = wait_for_mqtt_message
        mqtt_client.loop_start()
    except ConnectionRefusedError:
        logger.warning("MQTT broker not available — continuing without it; "
                       "scheduler updates will not be received")
        cfg["globals"]["mqtt_client"] = None

    # Safety net: if this launch is recovering from a crash that interrupted an
    # imaging run, detect a roof-open/unparked orphan and alert (see function).
    threading.Thread(target=_startup_orphan_check, daemon=True,
                     name="startup-orphan-check").start()

    try:
        start_interface()
    except Exception:
        logger.info('Problem')
        logger.exception("Exception")
        try:
            start_interface()
        except Exception:
            logger.exception("Social server failed to restart")

    print("stop")


if __name__ == '__main__':
    main()
