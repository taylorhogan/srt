"""An explicit `filters` plan that ends by itself (C2 of docs/AUTOMATION_PLAN.md, 2026-10-30).

    filters ngc1961 L=2 R=1 G=1 B=1 for 2     -> tonight and tomorrow night, then the advisor

A `filters` plan on a queue row beats everything else, every night, until
someone clears it; a ratio set for one night's reason (a soft forecast, a
channel to top up) ran for weeks. `for N` stores the LAST NIGHT the plan
applies as the row's `filter_plan_until` (an ISO date), and every reader of
the plan -- the sequence generator through instructions.get_filter_plan, the
noon plan's own decision (control/night_plan), the moon judge -- goes through
active_plan() here, so an expired plan is no plan everywhere at once. The
noon queue refresh drops the expired ones from the file and says so.

N counts IMAGED nights (owner, 2026-11-11: "cloudy nights with no imaging
should not count"). A plan set with `for N` since then also carries
`filter_plan_count` = {"nights": N, "from": first night, "imaged": [nights]}:
a night counts once the target's folder holds a frame for it
(<image_dir>/<target>/<camera>/<night>/LIGHT/*.fits), so a cloudy night, a
night the target got no slot and a night it lost to another target are all
free. The noon refresh recounts from disk (count_imaged) before it expires
anything, and the plan applies while fewer than N nights are counted; the
Nth imaged night is still the plan's, the next noon drops it. A night with
one frame counts. `filter_plan_until` stays as the CAP: max(CAP_DAYS, 2 N)
calendar nights, so a plan for a target that is out of season or never
scheduled cannot stand forever. A row with an until and no count (set
before) expires on its date as it always did.

Night dates. A night is named after the evening it starts. The first night a
new plan can affect is today's date whatever the hour: typed at 14:00 it
reaches tonight's sequence (the sunset check regenerates on a plan change);
typed at 02:00 the running night cannot change and the next one is today's
evening. A sequence WRITTEN before NIGHT_STARTS_HOUR, on the other hand, is
for the night in progress, which is yesterday's date (night_for). Only
.date() and .hour are read, so the DST change is nothing here.

Pure and stdlib-only: control/instructions needs astropy and the chat server,
and this must be testable on the pytest-only CI runner.
"""
import os
from datetime import date, datetime, timedelta
from typing import Optional

UNTIL_KEY = "filter_plan_until"
COUNT_KEY = "filter_plan_count"
NIGHT_STARTS_HOUR = 6      # before this a generated sequence is for the night in progress
MAX_NIGHTS = 60
CAP_DAYS = 30              # an imaged-night plan ends by this many calendar nights (or 2 N) regardless


def night_for(now: datetime) -> date:
    """The night a sequence written at *now* images: yesterday's date before dawn."""
    d = now.date()
    return d - timedelta(days=1) if now.hour < NIGHT_STARTS_HOUR else d


def first_night(now: datetime) -> date:
    """The first night a plan set at *now* can affect: tonight, whatever the hour."""
    return now.date()


def until_for(now: datetime, nights: int) -> str:
    """The last night of a plan set at *now* for *nights* nights, ISO."""
    return (first_night(now) + timedelta(days=max(1, int(nights)) - 1)).isoformat()


def limit_for(now: datetime, nights: int) -> tuple:
    """`for N` set at *now*: (cap until ISO, count record) for the row."""
    n = max(1, int(nights))
    cap = until_for(now, max(CAP_DAYS, 2 * n))
    return cap, {"nights": n, "from": first_night(now).isoformat(), "imaged": []}


def _until(row) -> Optional[date]:
    raw = (row or {}).get(UNTIL_KEY)
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None          # a malformed date never applies: an unreadable "until" is no limit


def _count(row) -> Optional[dict]:
    """The row's imaged-night record {nights, from, imaged}, or None (no count, or unreadable)."""
    c = (row or {}).get(COUNT_KEY)
    if not isinstance(c, dict):
        return None
    try:
        n = int(c.get("nights"))
        start = date.fromisoformat(str(c.get("from"))[:10])
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    imaged = [str(d)[:10] for d in (c.get("imaged") or []) if d]
    return {"nights": n, "from": start, "imaged": imaged}


def imaged_used(row) -> Optional[tuple]:
    """(imaged nights counted, N) for an imaged-night plan, else None."""
    c = _count(row)
    return (len(c["imaged"]), c["nights"]) if c else None


def is_expired(row, night: date) -> bool:
    """Has the row's plan run out before *night*? False without a limit.

    Out when the cap date is behind *night*, or (an imaged-night plan) when
    N imaged nights are already counted."""
    u = _until(row)
    if u is not None and u < night:
        return True
    used = imaged_used(row)
    return used is not None and used[0] >= used[1]


def nights_with_frames(target_dir) -> set:
    """The nights a target folder holds light frames for: <camera>/<YYYY-MM-DD>/LIGHT/*.fits.

    A frame marked `bad` (.fits.bad) is not a frame. Stdlib only; a missing
    or unreadable folder is no nights."""
    out = set()
    try:
        cams = [e for e in os.scandir(target_dir) if e.is_dir()]
    except OSError:
        return out
    for cam in cams:
        try:
            nights = [e for e in os.scandir(cam.path) if e.is_dir()]
        except OSError:
            continue
        for n in nights:
            try:
                d = date.fromisoformat(n.name[:10])
            except ValueError:
                continue
            try:
                if any(f.name.lower().endswith(".fits")
                       for f in os.scandir(os.path.join(n.path, "LIGHT"))):
                    out.add(d)
            except OSError:
                continue
    return out


def count_imaged(row, frame_nights, night: date) -> Optional[list]:
    """The nights that count against the row's plan, given the nights with frames. Pure.

    Nights from the plan's first night up to, not including, *night* (the
    night being planned; its frames do not exist yet). None for a row
    without an imaged-night plan."""
    c = _count(row)
    if c is None:
        return None
    return sorted(d.isoformat() for d in (frame_nights or ()) if c["from"] <= d < night)


def active_plan(row, night: date) -> Optional[dict]:
    """The row's {filter: count} plan if it applies to *night*, else None.

    None for no row, no plan, a plan with no positive count, or a plan whose
    `filter_plan_until` is before *night*.
    """
    plan = (row or {}).get("filter_plan")
    if not isinstance(plan, dict) or not plan:
        return None
    out = {}
    for f, n in plan.items():
        try:
            out[str(f)] = int(n)
        except (TypeError, ValueError):
            continue
    if not any(v > 0 for v in out.values()):
        return None
    if is_expired(row, night):
        return None
    return out


def nights_left(row, now: datetime) -> Optional[int]:
    """How many nights the plan still covers counting tonight; None = no limit."""
    u = _until(row)
    if u is None:
        return None
    return max(0, (u - first_night(now)).days + 1)


def describe_until(row, now: datetime) -> str:
    """' (until 2026-10-31, 2 more nights)' or '' for a plan with no limit.

    An imaged-night plan: ' (1 of 5 imaged nights used; ends by 2026-12-10 regardless)'."""
    u = _until(row)
    used = imaged_used(row)
    if used is not None:
        k, n = used
        if k >= n:
            return " (expired: %d imaged night%s)" % (n, "" if n == 1 else "s")
        if u is not None and u < first_night(now):
            return " (expired %s, %d of %d imaged nights used)" % (u.isoformat(), k, n)
        cap = "; ends by %s regardless" % u.isoformat() if u is not None else ""
        return " (%d of %d imaged night%s used%s)" % (k, n, "" if n == 1 else "s", cap)
    if u is None:
        return ""
    n = nights_left(row, now)
    if n == 0:
        return " (expired %s)" % u.isoformat()
    return " (until %s, %s)" % (u.isoformat(), "tonight only" if n == 1 else "%d more nights" % n)


def split_for_nights(tokens: list) -> tuple:
    """Take a trailing `for N` off a token list. -> (tokens, nights or None, error or None).

    `for` alone, `for 0`, `for x` and `for 999` are errors, said plainly, so a
    typo does not become a permanent plan.
    """
    toks = list(tokens or [])
    if len(toks) >= 1 and toks[-1].lower() == "for":
        return toks[:-1], None, "`for` needs a number of nights, e.g. `for 2`"
    if len(toks) >= 2 and toks[-2].lower() == "for":
        n = toks[-1]
        if not n.isdigit() or int(n) < 1:
            return toks[:-2], None, "`for %s`: nights must be a whole number from 1 to %d" % (n, MAX_NIGHTS)
        if int(n) > MAX_NIGHTS:
            return toks[:-2], None, "`for %s`: at most %d nights (leave it off for no limit)" % (n, MAX_NIGHTS)
        return toks[:-2], int(n), None
    return toks, None, None


def expired_rows(rows, night: date) -> list:
    """The queue rows whose plan has run out before *night*: [(row, plan, until)]. Pure."""
    out = []
    for r in rows or []:
        plan = (r or {}).get("filter_plan")
        if isinstance(plan, dict) and plan and is_expired(r, night):
            used = imaged_used(r)
            if used is not None and used[0] >= used[1]:
                c = _count(r)
                when = "%d imaged night%s, %s" % (used[1], "" if used[1] == 1 else "s",
                                                  " ".join(c["imaged"][-used[1]:]))
            else:
                when = str(r.get(UNTIL_KEY))[:10]
            out.append((r, dict(plan), when))
    return out


def describe_expired(expired: list) -> str:
    """The noon chat line for plans that ran out, '' when none.

    *expired* is the plan record's list: [{dso, plan, until}]; *until* is
    the last night's date, or for an imaged-night plan the nights it used."""
    if not expired:
        return ""
    lines = ["Filter plans expired (the advisor decides for them again):"]
    for e in expired:
        ratio = " ".join("%s=%s" % kv for kv in (e.get("plan") or {}).items()) or "-"
        until = str(e.get("until"))
        when = ("last night %s" % until) if until[:4].isdigit() and until[4:5] == "-" else "after %s" % until
        lines.append("  %s -- %s, %s" % (e.get("dso"), ratio, when))
    return "\n".join(lines)
