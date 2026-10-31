"""auto_stop.py -- which queued targets are finished, and why. Pure.

Phase A of docs/AUTOMATION_PLAN.md (2026-10-29). Nothing in the system
retired a target by itself: fits_processing.convergence.is_dso_done and the
filter advisor's "every goal met" both existed, but one only coloured the
`tonight` grid and the other a journal note, and a measurably converged
target was still ranked and picked every night. `completed` was written only
by the chat `dbc`.

"Done" is EITHER verdict:

  * convergence: every imaged filter calibrated, >= min frames, a flat tail
    and a low RMSE (is_dso_done);
  * the advisor's frame goals all met (filter_advisor.advise -> "done":
    sharp L / usable colour at the per-type goals).

Both reasons are kept so the chat line and the journal say which one fired.
A light-curve monitor (queue entry "lightcurve": true) is judged on neither:
it is done when it can draw its curve (owner, 2026-10-31), every star with a
period on the entry having a nightly point in at least 8 of 10 phase bins
(photometry.lightcurve.coverage); a star with no period never completes,
and the SNR of its stack has nothing to do with it.

Modes (cfg["planner"]["auto_stop"]):
  "off"     -- nothing is evaluated;
  "shadow"  -- the candidates ride the NIGHT_PLAN record ("auto_stop") and
               the noon chat line; the queue is not touched;
  "live"    -- a candidate's queue status is set to "converged", which the
               ranking (status == "waiting") no longer sees. Phase A3.

The object type for the advisor is taken OFFLINE (the queue row's obj_type,
then the local/dso_types.json cache): the noon plan runs under a 600 s
limit and a SIMBAD lookup hanging on 36 completed rows cost three replans
on 2026-10-27. An unknown type reads as broadband, whose L/R/G/B goals a
narrowband target never meets, so the error is always "not yet done".
"""
import json
import logging
import os
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "live")
DEFAULT_MODE = "shadow"
CONVERGED = "converged"             # the queue status a live auto-stop writes
HOLD_KEY = "auto_stop_hold_until"   # `reopen <dso> [for N]`: the last night auto-stop leaves it alone
DEFAULT_REOPEN_NIGHTS = 7


def mode(cfg) -> str:
    m = str(((cfg or {}).get("planner") or {}).get("auto_stop", DEFAULT_MODE)).lower()
    return m if m in MODES else DEFAULT_MODE


def verdict(dso: str, is_done: bool, advice: Optional[dict]) -> dict:
    """{dso, done, why: [...]} from the two verdicts. Pure.

    *advice* is filter_advisor.advise()'s result (or None when it could not
    run). action "monitor" (a light-curve target) is never done.
    """
    why = []
    action = (advice or {}).get("action")
    if action == "monitor":
        return {"dso": dso, "done": False, "why": ["light-curve monitor: never retired"]}
    if is_done:
        why.append("convergence: every filter at its flat tail (is_dso_done)")
    if action == "done":
        have = (advice or {}).get("have") or {}
        counts = ", ".join("%s %d" % (f, (h or {}).get("usable", 0))
                           for f, h in sorted(have.items()))
        why.append("frame goals met (filter advisor)%s" % (": " + counts if counts else ""))
    return {"dso": dso, "done": bool(why), "why": why}


def monitor_verdict(dso: str, stars: list) -> dict:
    """{dso, done, why} for a light-curve monitor from its stars' coverage. Pure.

    *stars*: [{"name", "period_days", "coverage": lightcurve.coverage()}].
    Done when every star with a period has a complete curve and there is at
    least one such star; a star without a period is named and not judged.
    """
    judged, why = [], []
    for st in stars or []:
        cov = st.get("coverage") or {}
        if not st.get("period_days"):
            why.append("%s: no period, not judged" % st.get("name"))
            continue
        judged.append(bool(cov.get("complete")))
        why.append("%s: %d of %d phase bins over %d nights%s" % (
            st.get("name"), int(cov.get("hit") or 0), int(cov.get("bins") or 0), int(cov.get("nights") or 0),
            "" if cov.get("complete") else " (not yet a curve)"))
    if not judged:
        return {"dso": dso, "done": False, "why": ["light-curve monitor: no period on the entry, the curve has no end"]}
    done = all(judged)
    return {"dso": dso, "done": done, "why": (["light curve complete: " + "; ".join(why)] if done else why)}


def reopened(row: dict, night) -> bool:
    """Is the row under a `reopen` hold for *night* (control/filter_plan dates)?

    A reopened target's convergence record still says done, so without this
    the next live noon would retire it again the morning after the owner
    asked for more frames (A4). The hold is the row's HOLD_KEY, the last
    night it covers; a malformed date is no hold. *night* None = judge the
    hold as standing.
    """
    raw = (row or {}).get(HOLD_KEY)
    if not raw:
        return False
    if night is None:
        return True
    from datetime import date
    try:
        return date.fromisoformat(str(raw)[:10]) >= night
    except ValueError:
        return False


def candidates(rows: list, is_done_fn: Callable[[str], bool],
               advise_fn: Callable[[str, dict], Optional[dict]], night=None,
               curve_fn: Optional[Callable[[str, dict], list]] = None) -> list:
    """The waiting rows that are done, each with its reasons. Pure.

    *rows* is the queue; only "waiting" / "in process" rows are judged (a
    held row is the operator's, a completed or converged one is already
    out, a reopened one is left alone until its hold's last night, *night*).
    A light-curve monitor is judged by *curve_fn* (its stars' coverage ->
    monitor_verdict) and never by the other two; without a curve_fn it is
    left alone. A row that raises in a verdict is skipped, logged, never
    stopped: a crash must err toward imaging a target once more.
    """
    out = []
    for row in rows or []:
        dso = str(row.get("dso") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        if not dso or status not in ("waiting", "in process"):
            continue
        if reopened(row, night):
            _logger.info("auto-stop: %s reopened by the owner until %s; not judged",
                         dso, str(row.get(HOLD_KEY))[:10])
            continue
        if row.get("lightcurve"):
            if curve_fn is None:
                continue
            try:
                v = monitor_verdict(dso, curve_fn(dso, row))
            except Exception:                  # noqa: BLE001
                _logger.exception("auto-stop: light-curve coverage failed for %s", dso)
                continue
            if v["done"]:
                out.append(v)
            continue
        try:
            done = bool(is_done_fn(dso))
        except Exception:                      # noqa: BLE001
            _logger.exception("auto-stop: convergence verdict failed for %s", dso)
            done = False
        try:
            adv = advise_fn(dso, row)
        except Exception:                      # noqa: BLE001
            _logger.exception("auto-stop: advisor verdict failed for %s", dso)
            adv = None
        v = verdict(dso, done, adv)
        if v["done"]:
            out.append(v)
    return out


def describe(found: list, live: bool) -> str:
    """The noon chat line. '' when there is nothing to say."""
    if not found:
        return ""
    head = ("Auto-stop: %d target(s) marked converged and out of the queue:"
            if live else
            "Auto-stop (shadow, nothing changed): %d target(s) would be retired:") % len(found)
    lines = [head]
    for v in found:
        lines.append("  %s -- %s" % (v["dso"], "; ".join(v["why"])))
    if not live:
        lines.append("  (cfg planner.auto_stop = live makes it so; `reopen <dso>` undoes one)")
    return "\n".join(lines)


# ------------------------------------------------------------------ live inputs

TYPE_WORDS = {"G": "galaxy", "N": "nebula"}     # the grid's letter -> the queue row's obj_type


def type_word(letter) -> Optional[str]:
    """The queue's obj_type for the imaging grid's one-letter class
    (astro_dso_visibility.get_dso_type): G -> galaxy, N -> nebula, else None
    (a star cluster plans broadband as "unknown" does; "?" is unknown). Pure.
    C4: `image` stores this on the row so the advisor, auto-stop and the
    generator never read SIMBAD or the type cache for a queued target."""
    return TYPE_WORDS.get(str(letter or "").strip().upper()[:1])


def _type_offline(row: dict, root: str) -> str:
    """galaxy / nebula / unknown without the network."""
    t = str(row.get("obj_type") or "").lower()
    if t in ("galaxy", "nebula"):
        return t
    try:
        with open(os.path.join(root, "local", "dso_types.json"), encoding="utf-8") as fh:
            cache = json.load(fh)
        key = str(row.get("dso") or "").lower().replace(" ", "")
        return type_word(cache.get(key) or cache.get(row.get("dso"))) or "unknown"
    except Exception:                          # noqa: BLE001
        return "unknown"


def advise_offline(dso: str, row: dict, root: Optional[str] = None) -> Optional[dict]:
    """filter_advisor.advise on the target's own frame_stats, type offline."""
    from fits_processing import filter_advisor
    root = root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    frames = filter_advisor._frames_for(dso)
    if not frames:
        return None
    return filter_advisor.advise(frames, _type_offline(row, root), row, {}, dso=dso,
                                 emission=filter_advisor.emission_for(dso),
                                 **filter_advisor.advisor_settings())


def curve_coverage(dso: str, row: dict) -> list:
    """monitor_verdict's input from local/lightcurves: every star of the
    entry (lightcurve.stars_for) with its nightly records folded on its
    period. No coordinates on the row = no stars = not done."""
    from photometry import lightcurve as lc
    if row.get("ra_deg") is None or row.get("dec_deg") is None:
        return []
    out = []
    for st in lc.stars_for(dso, (float(row["ra_deg"]), float(row["dec_deg"])), row):
        doc = lc.load(st["key"])
        out.append({"name": st["name"], "period_days": st.get("period_days"),
                    "coverage": lc.coverage(doc.get("records") or [], st.get("period_days"), st.get("epoch_jd"))})
    return out


def evaluate(rows: list) -> list:
    """candidates() on the live verdicts. Never raises."""
    try:
        from fits_processing.convergence import is_dso_done
        from datetime import datetime
        from control import filter_plan
        return candidates(rows, is_dso_done, advise_offline, filter_plan.night_for(datetime.now()),
                          curve_fn=curve_coverage)
    except Exception:                          # noqa: BLE001
        _logger.exception("auto-stop evaluation failed")
        return []
