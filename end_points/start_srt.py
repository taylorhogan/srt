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


# The shadow conductor (architecture plan, Phase 1). Read-only towards the
# observatory -- it watches the legacy state files and builds the journal;
# it commands nothing. Behind a config flag so it can be turned off without
# a code change if it ever misbehaves during the shadow period.
#
# MODULE LEVEL ON PURPOSE. Windows multiprocessing spawns a fresh interpreter
# that re-imports this file as __mp_main__ with the __main__ block skipped, then
# unpickles the target by name. From 2026-08-28 to 2026-09-06 these two lived
# inside the __main__ block: the parent pickled them happily, the child could
# not find them and died with exit 1 before a line of conductor code ran. The
# only symptom was 8096 not listening, and nothing was looking.
def _conductor_enabled() -> bool:
    try:
        from configs import config
        return bool(config.data().get("conductor", {}).get("shadow_enabled", True))
    except Exception:
        return False


def _conductor_target():
    from iris.conductor import main as conductor_main
    conductor_main.main()


if __name__ == "__main__":
    # No Prefect server: run against the ephemeral one. UNSET, not "" -- Prefect
    # reads an empty string as "a running server is configured" and builds an
    # events client for it, which rejects the empty URL. Every heartbeat (180s)
    # and state change then logged "Service 'EventsWorker' failed" from April
    # to 2026-09-06. Unset, events go to the ephemeral server like everything else.
    os.environ.pop("PREFECT_API_URL", None)

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
        p3 = Process(target=_conductor_target) if _conductor_enabled() else None

        p1.start()
        p2.start()
        if p3:
            p3.start()

        p1.join()  # wait for social server to exit
        exit_code = p1.exitcode

        # Capture imaging state before the scheduler relaunch can clear it.
        imaging_state = _imaging_state_at_crash()

        # Always clean up the siblings when the social server exits
        if p2.is_alive():
            p2.terminate()
            p2.join(timeout=10)
        if p3 is not None and p3.is_alive():
            p3.terminate()
            p3.join(timeout=10)

        if exit_code == RESTART_EXIT_CODE:
            # Deliberate restart (the `update` command): deploy the last GREEN
            # commit, then relaunch. `release` is fast-forwarded by CI only
            # when the tests pass, so a red main never reaches a running
            # observatory through this path. ff-only rather than a plain pull
            # because this checkout is also the development machine: if local
            # main is ahead of release (work in progress, or CI still
            # running), the merge refuses and we relaunch on the local code
            # rather than tangling the working copy.
            crash_times.clear()
            fetch = subprocess.run(
                ["git", "-C", project_root, "fetch", "origin", "release"],
                capture_output=True, text=True,
            )
            if fetch.returncode != 0:
                print(f"git fetch failed (exit {fetch.returncode}):\n{fetch.stderr}")
                break
            result = subprocess.run(
                ["git", "-C", project_root, "merge", "--ff-only", "origin/release"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print("release not fast-forwardable (local work ahead, or CI "
                      f"pending) — relaunching on local code:\n{result.stderr}")
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
