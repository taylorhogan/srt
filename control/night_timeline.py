"""Tonight's plan as a timeline: one row per event, time in one column.

Asked for 2026-10-08 in place of the noon report's paragraph: when the roof
opens and closes, when each target starts and in which filters, and when the
night is wrapped up (flats and the end-of-night analysis done).

Pure: the scheduler passes in sunset, the planner's slots and the filter
counts written into the sequence; nothing here reads the clock or the disk.

The offsets are measured, not designed (iris.log, nights 10-03 .. 10-08):
  * roof open: 3-4 min after the pre-sunset check (vision + conductor + the
    11 s move), so about 6 min before sunset;
  * roof closed: 4-5 min after the end sequence begins (park, then close);
  * flats: 70-75 min after the close;
  * SNR + night-contribution analysis: 14-20 min after the flats, longer with
    more targets and frames.
Every derived time carries a "~" in the output.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

PRE_SUNSET_CHECK = timedelta(minutes=10)     # scheduler: 10 min before sunset
ROOF_OPEN_AFTER_CHECK = timedelta(minutes=4)
ROOF_CLOSED_AFTER_END = timedelta(minutes=5)
FLATS = timedelta(minutes=75)
ANALYSIS = timedelta(minutes=20)

_FILTER_ORDER = ("L", "R", "G", "B", "Ha", "O-III", "S-II")


@dataclass
class SlotPlan:
    name: str
    start: Optional[datetime]
    end: Optional[datetime]
    counts: dict                 # {"L": 18, "R": 9, ...} as written into the sequence
    exposure_s: dict             # {"L": 300.0, ...}


def _filters_text(counts: dict, exposure_s: dict) -> str:
    keys = sorted(counts, key=lambda f: (_FILTER_ORDER.index(f) if f in _FILTER_ORDER else 99, f))
    parts, total = [], 0.0
    for f in keys:
        n = int(counts[f])
        if n <= 0:
            continue
        exp = float(exposure_s.get(f) or 0.0)
        total += n * exp
        parts.append("%s %d" % (f, n))
    if not parts:
        return "template filters"
    exps = {float(exposure_s.get(f) or 0.0) for f in keys if counts.get(f)}
    each = (" x %.0fs" % exps.pop()) if len(exps) == 1 and 0.0 not in exps else ""
    return "%s%s (%.1fh)" % (", ".join(parts), each, total / 3600.0)


def build(sunset: datetime, slots: list, tz=None) -> list:
    """[(time or None, approximate?, event), ...] in time order."""
    loc = (lambda t: t.astimezone(tz)) if tz else (lambda t: t)
    rows = []
    check = sunset - PRE_SUNSET_CHECK
    rows.append((loc(check), False, "Imaging check (weather and plan re-checked)"))
    rows.append((loc(check + ROOF_OPEN_AFTER_CHECK), True, "Roof opens, camera cools, sequence starts"))
    rows.append((loc(sunset), False, "Sunset"))
    for i, s in enumerate(slots):
        what = "%s starts: %s" % (s.name, _filters_text(s.counts, s.exposure_s))
        if i > 0:
            what += " (earlier if %s finishes early and it is above the horizon)" % slots[i - 1].name
        rows.append((loc(s.start) if s.start else None, False, what))
    end = slots[-1].end if slots and slots[-1].end else None
    if end is not None:
        rows.append((loc(end), False, "Imaging ends"))
        closed = end + ROOF_CLOSED_AFTER_END
        rows.append((loc(closed), True, "Scope parked, roof closed"))
        rows.append((loc(closed + FLATS), True, "Flats done"))
        rows.append((loc(closed + FLATS + ANALYSIS), True, "All wrapped up: SNR and night analysis posted"))
    else:
        rows.append((None, False, "Imaging ends at dawn; roof close, flats (~75 min) and analysis (~20 min) follow"))
    return rows


def render(rows, title: str = "Tonight") -> str:
    """Two columns: time, event. Approximate times are marked with ~."""
    lines = [title, "Time    Event"]
    for t, approx, event in rows:
        stamp = "  ?  " if t is None else t.strftime("%H:%M")
        lines.append("%s%s  %s" % ("~" if approx else " ", stamp, event))
    return "\n".join(lines)
