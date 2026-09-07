"""slot_plan.py -- fit two targets into one night's dark hours.

    slots = plan_slots(rows, dark_hours, weather_by_hour)

Pure: hours in, slots out. No astronomy here; ``rows`` are exactly what
iris_astronomy.astro_dso_visibility.rank_targets_tonight returns, one per
waiting target, best first:

    (dso_name, good_count, max_alt, dso_type, start_time, symbols, priority)

where ``symbols[i]`` is "+" if the target is above the treeline in
``dark_hours[i]`` and ``weather_by_hour[dark_hours[i].hour]`` says whether
that hour's forecast is acceptable. A GOOD hour is both.

WHY. A one-target night leaves the tail of the dark window unused. Measured
2026-09-06/07: the squid sequence ended 02:16, astronomical dawn 04:46 -- 2.5
of 7.9 dark hours idle, and the window grows to 9 h by October. A second slot
in that tail is a free extra night every three on a target that is counted in
nights.

RULES (the small version of the architecture plan's Phase 7 slot-fitting;
one roof open, one prelude, one set of flats -- only the main section gets a
second target):

  * Slot 1 is the first row: rank_targets_tonight already sorts operator pins
    (priority > 5) first, then good hours, then altitude.
  * Slot 2 is the row with the most good hours OUTSIDE slot 1's window, pins
    first, needing at least ``min_slot_hours`` of them. For a PINNED
    candidate both orders are tried -- a pinned target that sets early may
    deserve to go first -- and the order with more total good hours wins.
  * A slot's window is its first good hour to the hour after its last good
    hour, clipped by the other slot. The end is a hard TimeCondition in the
    sequence, so a slot that runs long cannot eat the next one.
  * Two PINS that overlap all night split it: slot 2 starts at the later of
    its own first good hour and the point where slot 1 has had at least half
    its hours, and both halves must clear ``min_slot_hours``. An unpinned
    candidate never takes hours slot 1 could use.
  * If no second target clears the bar the plan is one slot, exactly as
    before. A second slot is a bonus, never a reason to lower the bar.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class Slot:
    name: str
    start: datetime          # first good hour (inclusive)
    end: datetime            # hard end (exclusive)
    good_hours: int          # good hours inside [start, end)
    priority: int = 5

    @property
    def seconds(self) -> float:
        return self.good_hours * 3600.0


def _good_flags(row, dark_hours, weather_by_hour):
    symbols = row[5]
    return [symbols[i] == "+" and bool(weather_by_hour.get(dt.hour, False))
            for i, dt in enumerate(dark_hours)]


def _window(flags, dark_hours, lo=None, hi=None):
    """(start, end, count) of the good hours in [lo, hi), or None."""
    idx = [i for i, ok in enumerate(flags) if ok
           and (lo is None or dark_hours[i] >= lo)
           and (hi is None or dark_hours[i] < hi)]
    if not idx:
        return None
    start = dark_hours[idx[0]]
    end = dark_hours[idx[-1]] + timedelta(hours=1)
    if hi is not None and end > hi:
        end = hi
    return start, end, len(idx)


def plan_slots(rows, dark_hours, weather_by_hour, min_slot_hours: float = 2.0,
               max_slots: int = 2) -> list:
    """The night's slots, in the order they run. Empty if nothing is usable."""
    if not rows or not dark_hours:
        return []
    flags = {r[0]: _good_flags(r, dark_hours, weather_by_hour) for r in rows}
    first = rows[0]
    w1 = _window(flags[first[0]], dark_hours)
    if w1 is None:
        return []
    slot1 = Slot(first[0], w1[0], w1[1], w1[2], int(first[6]))
    if max_slots < 2 or len(rows) < 2:
        return [slot1]

    best = None
    for cand in rows[1:]:
        f = flags[cand[0]]
        pinned = int(cand[6]) > 5
        # order A: slot1 first, candidate in what is left after it
        after = _window(f, dark_hours, lo=slot1.end)
        if after and after[2] >= min_slot_hours:
            total = slot1.good_hours + after[2]
            plan = [slot1, Slot(cand[0], after[0], after[1], after[2], int(cand[6]))]
            key = (pinned, total, after[2])
            if best is None or key > best[0]:
                best = (key, plan)
        # order B: candidate first (its full window), slot1 in what is left.
        # Only for a PINNED candidate: this order takes hours from the pick,
        # and an unpinned target is never allowed to do that.
        wc = _window(f, dark_hours) if pinned else None
        if wc and wc[2] >= min_slot_hours:
            rest = _window(flags[first[0]], dark_hours, lo=wc[1])
            if rest and rest[2] >= min_slot_hours:
                total = wc[2] + rest[2]
                plan = [Slot(cand[0], wc[0], wc[1], wc[2], int(cand[6])),
                        Slot(first[0], rest[0], rest[1], rest[2], slot1.priority)]
                key = (pinned, total, min(wc[2], rest[2]))
                if best is None or key > best[0]:
                    best = (key, plan)
    if best:
        return best[1]
    # Two pins, no free tail: split the night between them. Either may lead;
    # the leader keeps at least half its hours and the order with more total
    # good hours wins (tie: the pick leads).
    if slot1.priority > 5:
        best = None
        for cand in rows[1:]:
            if int(cand[6]) <= 5:
                continue
            for lead, follow in ((first, cand), (cand, first)):
                fl, ff = flags[lead[0]], flags[follow[0]]
                wl = _window(fl, dark_hours)
                wf = _window(ff, dark_hours)
                if wl is None or wf is None:
                    continue
                half = wl[0] + timedelta(hours=(wl[2] + 1) // 2)
                cut = max(wf[0], half)
                head = _window(fl, dark_hours, hi=cut)
                tail = _window(ff, dark_hours, lo=cut)
                if not (head and tail) or head[2] < min_slot_hours or tail[2] < min_slot_hours:
                    continue
                total = head[2] + tail[2]
                key = (total, lead is first)
                if best is None or key > best[0]:
                    best = (key, [Slot(lead[0], head[0], head[1], head[2], int(lead[6])),
                                  Slot(follow[0], tail[0], tail[1], tail[2], int(follow[6]))])
        if best:
            return best[1]
    return [slot1]


def describe(slots, tz=None) -> str:
    """One line per slot for the chat: 'squid 21:00-02:00 (5h)'."""
    out = []
    for s in slots:
        a = s.start.astimezone(tz) if tz else s.start
        b = s.end.astimezone(tz) if tz else s.end
        out.append("%s %s-%s (%dh)" % (s.name, a.strftime("%H:%M"), b.strftime("%H:%M"), s.good_hours))
    return "\n".join(out)
