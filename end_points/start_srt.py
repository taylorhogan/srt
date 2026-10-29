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


OUR_PORTS = (8095, 8096)


def _kill_tree(proc) -> None:
    """Terminate *proc* and everything under it.

    Process.terminate() takes the child alone. The scheduler's Prefect server
    is a grandchild, and it survived every restart: the 13:30 update on
    2026-09-06 left one on 8169, and the noon flow on 2026-09-07 left one on
    127.0.0.1:8095 shadowing the web chat. taskkill /T takes the tree.
    """
    if proc is None or not proc.is_alive():
        return
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        pass
    if proc.is_alive():
        proc.terminate()
    proc.join(timeout=10)


def _evict_loopback_squatters(ports=OUR_PORTS) -> None:
    """Kill any process listening on 127.0.0.1:<one of our ports>; see utils/ports.py."""
    try:
        from utils import ports as _ports
        _ports.evict_loopback_squatters(ports)
    except Exception:  # noqa: BLE001
        pass


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

    # Supervision (end_points/supervise, docs/AUTOMATION_PLAN.md B7). Until
    # 2026-10-29 this loop joined the social server alone: a scheduler or a
    # conductor that died by itself was never relaunched and nobody was told,
    # and with the conductor holding roof and mount authority that is a
    # refused night in silence. Now every child is watched; the social
    # server's exit keeps its meanings (update / stop / crash -> everything
    # relaunches), a dead sibling relaunches alone with a push, and each
    # child has its own crash-loop guard.
    from end_points import supervise

    def _push(msg, priority=0):
        print(msg)
        try:
            from utils import pushover
            pushover.push_message(msg, priority=priority)
        except Exception as exc:  # noqa: BLE001
            print(f"pushover alert failed: {exc}")

    def _launch(name):
        if name == supervise.SOCIAL:
            p = Process(target=social_server.main)
        elif name == "scheduler":
            p = Process(target=scheduler_server.main)
        else:
            p = Process(target=_conductor_target) if _conductor_enabled() else None
        if p is not None:
            p.start()
        return p

    crash_times = {name: [] for name in supervise.CHILDREN}
    given_up = set()
    relaunch_all = True
    procs = {}
    while True:
        if relaunch_all:
            _evict_loopback_squatters()
            procs = {name: _launch(name) for name in supervise.CHILDREN}
            relaunch_all = False

        time.sleep(5)
        alive = {name: (p is not None and p.is_alive()) for name, p in procs.items()
                 if p is not None and name not in given_up}
        dead = supervise.dead_children(alive)
        if not dead:
            continue

        name = dead[0]
        exit_code = procs[name].exitcode
        imaging_state = _imaging_state_at_crash()
        verdict = supervise.decide(name, exit_code, crash_times[name], time.time(),
                                   restart_exit_code=RESTART_EXIT_CODE)
        if verdict["push"]:
            _push(verdict["push"] + f" Imaging state: {imaging_state}.", verdict["priority"])

        if verdict["action"] in ("restart_all", "stop", "relaunch_all"):
            # The social server decides for everyone, as it always did: whole
            # trees, so a Prefect server under the scheduler goes with it.
            for other in supervise.CHILDREN:
                if other != name:
                    _kill_tree(procs.get(other))
        if verdict["action"] == "restart_all":
            # Deliberate restart (the `update` command): deploy the last GREEN
            # commit, then relaunch. `release` is fast-forwarded by CI only
            # when the tests pass, so a red main never reaches a running
            # observatory through this path. ff-only rather than a plain pull
            # because this checkout is also the development machine: if local
            # main is ahead of release (work in progress, or CI still
            # running), the merge refuses and we relaunch on the local code
            # rather than tangling the working copy.
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
            given_up.clear()
            relaunch_all = True
        elif verdict["action"] == "stop":
            break                       # clean exit — stop
        elif verdict["action"] == "relaunch_all":
            time.sleep(verdict["backoff_s"])
            relaunch_all = True         # do NOT git-pull on a crash relaunch
        elif verdict["action"] == "relaunch":
            time.sleep(verdict["backoff_s"])
            procs[name] = _launch(name)
        else:                           # give_up on this child alone
            given_up.add(name)
            if name == supervise.SOCIAL:
                break
