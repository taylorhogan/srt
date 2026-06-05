from multiprocessing import Process
import os
import subprocess
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from cmd_processing import social_server
from cmd_processing.social_server import RESTART_EXIT_CODE
import scheduler_server


def _imaging_state_at_crash() -> str:
    """Best-effort read of imaging.txt so a crash alert can say whether a run
    was in progress. Returns 'unknown' if it can't be read."""
    try:
        with open(os.path.join(project_root, "imaging.txt")) as f:
            return f.read().strip() or "NONE"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "")

    import threading
    import time
    from sentry.internet_classify import run_speed_test
    threading.Thread(target=run_speed_test, daemon=True).start()

    # Crash-loop guard: relaunch the social server after an unexpected exit, but
    # give up if it dies too many times in a short window (a persistent crash).
    CRASH_WINDOW_SECS = 600     # 10 minutes
    MAX_CRASHES = 5             # within the window before giving up
    crash_times: list[float] = []

    while True:
        p1 = Process(target=social_server.main)
        p2 = Process(target=scheduler_server.main)

        p1.start()
        p2.start()

        p1.join()  # wait for social server to exit
        exit_code = p1.exitcode

        # Capture imaging state before the scheduler relaunch can clear it.
        imaging_state = _imaging_state_at_crash()

        # Always clean up the scheduler when the social server exits
        if p2.is_alive():
            p2.terminate()
            p2.join(timeout=10)

        if exit_code == RESTART_EXIT_CODE:
            # Deliberate restart (the `update` command): pull then relaunch.
            crash_times.clear()
            result = subprocess.run(
                ["git", "-C", project_root, "pull"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"git pull failed (exit {result.returncode}):\n{result.stderr}")
                break
            continue  # relaunch both processes
        elif exit_code == 0:
            break  # clean exit — stop
        else:
            # Unexpected exit (crash). Relaunch so web chat self-heals, but bail
            # out of a tight crash loop. Do NOT git-pull on a crash relaunch.
            now = time.time()
            crash_times.append(now)
            crash_times[:] = [t for t in crash_times if now - t <= CRASH_WINDOW_SECS]
            if len(crash_times) >= MAX_CRASHES:
                msg = (
                    f"Iris: social server crashed {len(crash_times)} times within "
                    f"{CRASH_WINDOW_SECS}s (exit {exit_code}) — giving up. "
                    f"Imaging state at crash: {imaging_state}. Manual check needed."
                )
                print(msg)
                try:
                    from utils import pushover
                    pushover.push_message(msg)
                except Exception as exc:
                    print(f"pushover alert failed: {exc}")
                break
            backoff = min(30, 5 * len(crash_times))
            msg = (
                f"Iris: social server exited unexpectedly (exit {exit_code}); "
                f"relaunching in {backoff}s (crash {len(crash_times)}/{MAX_CRASHES}). "
                f"Imaging state at crash: {imaging_state}."
            )
            print(msg)
            try:
                from utils import pushover
                pushover.push_message(msg)
            except Exception as exc:
                print(f"pushover alert failed: {exc}")
            time.sleep(backoff)
            continue  # relaunch both processes
