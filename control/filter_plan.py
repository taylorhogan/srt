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
from datetime import date, datetime, timedelta
from typing import Optional

UNTIL_KEY = "filter_plan_until"
NIGHT_STARTS_HOUR = 6      # before this a generated sequence is for the night in progress
MAX_NIGHTS = 60


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


def _until(row) -> Optional[date]:
    raw = (row or {}).get(UNTIL_KEY)
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None          # a malformed date never applies: an unreadable "until" is no limit


def is_expired(row, night: date) -> bool:
    """Has the row's plan run out before *night*? False without a limit."""
    u = _until(row)
    return u is not None and u < night


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
    """' (until 2026-10-31, 2 more nights)' or '' for a plan with no limit."""
    u = _until(row)
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
            out.append((r, dict(plan), str(r.get(UNTIL_KEY))[:10]))
    return out


def describe_expired(expired: list) -> str:
    """The noon chat line for plans that ran out, '' when none.

    *expired* is the plan record's list: [{dso, plan, until}]."""
    if not expired:
        return ""
    lines = ["Filter plans expired (the advisor decides for them again):"]
    for e in expired:
        ratio = " ".join("%s=%s" % kv for kv in (e.get("plan") or {}).items()) or "-"
        lines.append("  %s -- %s, last night %s" % (e.get("dso"), ratio, e.get("until")))
    return "\n".join(lines)
