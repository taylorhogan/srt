"""supervise.py -- what start_srt does when one of its three children dies. Pure.

docs/AUTOMATION_PLAN.md B7 (2026-10-29). start_srt launches the web chat
(social server), the scheduler and the conductor, and until this module it
waited on the social server alone: a scheduler or conductor that died by
itself was never relaunched and nobody was told. With the conductor holding
roof and mount authority, a dead conductor is a refused night, in silence.

decide() is the policy; start_srt's loop applies it every few seconds:

  * the social server's exit keeps its old meaning -- RESTART_EXIT_CODE is
    the `update` command (restart everything on the release branch), 0 is a
    clean stop, anything else a crash that relaunches everything;
  * a dead scheduler or conductor relaunches THAT CHILD ONLY, with a push,
    and the others run on (the scheduler resumes from the conductor's plan,
    the conductor from its journal);
  * every child has its own crash-loop guard: MAX_CRASHES exits inside
    CRASH_WINDOW_S and the supervisor gives up on it with an emergency push.
"""
from typing import Optional

CRASH_WINDOW_S = 600.0
MAX_CRASHES = 5
RESTART_EXIT_CODE_DEFAULT = 42
SOCIAL = "social server"
CHILDREN = (SOCIAL, "scheduler", "conductor")


def decide(name: str, exit_code: Optional[int], crash_times: list, now: float,
           restart_exit_code: int = RESTART_EXIT_CODE_DEFAULT,
           window_s: float = CRASH_WINDOW_S, max_crashes: int = MAX_CRASHES) -> dict:
    """What to do about child *name* that just exited with *exit_code*.

    crash_times is this child's recent crash epochs (mutated: pruned to the
    window, this crash appended when it counts as one).
    -> {"action": "restart_all" | "stop" | "relaunch_all" | "relaunch" | "give_up",
        "backoff_s", "push", "priority"}
    """
    if name == SOCIAL and exit_code == restart_exit_code:
        crash_times.clear()
        return {"action": "restart_all", "backoff_s": 0, "push": None, "priority": 0}
    if name == SOCIAL and exit_code == 0:
        return {"action": "stop", "backoff_s": 0, "push": None, "priority": 0}
    crash_times.append(now)
    crash_times[:] = [t for t in crash_times if now - t <= window_s]
    n = len(crash_times)
    if n >= max_crashes:
        return {"action": "give_up", "backoff_s": 0, "priority": 2,
                "push": ("Iris: %s crashed %d times within %.0fs (exit %s) -- giving up on it. "
                         "Manual check needed." % (name, n, window_s, exit_code))}
    backoff = min(30, 5 * n)
    action = "relaunch_all" if name == SOCIAL else "relaunch"
    return {"action": action, "backoff_s": backoff, "priority": 1 if name != SOCIAL else 0,
            "push": ("Iris: %s exited unexpectedly (exit %s); relaunching%s in %ds (crash %d/%d)."
                     % (name, exit_code, " everything" if name == SOCIAL else " it alone",
                        backoff, n, max_crashes))}


def dead_children(alive: dict) -> list:
    """Names of children that are not alive, in CHILDREN order (the social
    server first, so its exit code decides before a sibling's)."""
    return [n for n in CHILDREN if n in alive and not alive[n]]
