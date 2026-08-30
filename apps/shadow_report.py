"""Morning report: did the shadow conductor's journal match last night?

Run daily (scheduled task) during the shadow period. Compares the journal
against the ground truth the legacy system already writes:

  * imaging_state.log -- every NINA-originated transition (the .bat appends a
    timestamped line). Each one must have a matching journal event within
    TOLERANCE_S. A miss means the shadow's file-watching lost an edge; a
    spurious journal NINA event means it invented one. Either is a divergence
    and the phase does not cut over with unexplained divergences.
  * guard counterfactuals -- every transition whose `guard_would` is non-null
    is a place the new guards WOULD have refused where the legacy system
    proceeded. Each is either a guard bug or a legacy bug; Phase 2 needs the
    list either way.
  * ignored events -- reality did something the machine has no row for
    (e.g. a manual image!! outside the modelled night). Phase 3's homework.

Posts one message to the webchat and exits 0 always: a report that fails
silently is worse than no report.

    python apps/shadow_report.py [--night YYYY-MM-DD] [--quiet]
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from iris.core.journal import Journal

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_S = 300          # a shadow poll is 5 s; 5 min allows for restarts

# imaging_state.log lines are written by the .bat, in cmd.exe's locale format:
#   Tue 08/25/2026 20:11:14.84 IMAGING_STATE set to DONE_PRELUDE
#   Wed 08/26/2026  4:38:04.38 IMAGING_STATE set to DONE_FLATS
# Note the DOUBLE space before a single-digit hour, and the fractional
# seconds. Verified against the live file before this regex was written --
# the first draft assumed ISO timestamps and matched nothing.
_ISLOG_RE = re.compile(
    r"^\w{3}\s+(\d\d)/(\d\d)/(\d{4})\s+(\d{1,2}):(\d\d):(\d\d)\.\d+\s+"
    r"IMAGING_STATE set to ([A-Z_]+)\s*$")

# Which journal event corresponds to each bat-written imaging state.
_BAT_TO_EVENT = {
    "DONE_PRELUDE": "NINA_PRELUDE_DONE",
    "DONE_FLATS": "NINA_FLATS_DONE",
    "IN_MAIN": "NINA_SLOT_DONE",
    "IN_FLATS": "NINA_SLOT_DONE",
}


def _night_bounds(night: str):
    d = datetime.fromisoformat(night)
    return d.replace(hour=12), d.replace(hour=12) + timedelta(days=1)


def _bat_transitions(night):
    lo, hi = _night_bounds(night)
    out = []
    path = ROOT / "imaging_state.log"
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ISLOG_RE.match(ln.strip())
        if not m:
            continue
        mo, dy, yr, hh, mi, ss, state = m.groups()
        try:
            when = datetime(int(yr), int(mo), int(dy), int(hh), int(mi), int(ss))
        except ValueError:
            continue
        if lo <= when < hi and state in _BAT_TO_EVENT:
            out.append((when, state, _BAT_TO_EVENT[state]))
    return out


def _journal_entries(night):
    lo, hi = _night_bounds(night)
    j = Journal(ROOT / "local" / "journal")
    out = []
    for e in j.replay(days=3):
        try:
            when = datetime.fromisoformat(e.ts).replace(tzinfo=None)
        except ValueError:
            continue
        if lo <= when < hi:
            out.append((when, e))
    return out


def build_report(night: str) -> str:
    entries = _journal_entries(night)
    transitions = [(w, e) for w, e in entries if e.kind == "transition"]
    lines = [f"Shadow report — night {night}"]

    if not transitions:
        lines.append("No journal transitions this night (shadow idle or not running).")
    else:
        lines.append(f"{len(transitions)} transitions:")
        for w, e in transitions:
            # ASCII arrow on purpose: u2192 crashed print() under the
            # scheduled task's cp1252 console, which killed the report BEFORE
            # it posted -- on the first night that had any transitions to
            # print. main() also reconfigures stdout, but the text itself
            # staying encodable is the fix that cannot regress.
            lines.append("  %s  %s: %s -> %s" % (
                w.strftime("%H:%M:%S"), e.event, e.from_state, e.to_state))

    # --- NINA ground truth vs journal
    bat = _bat_transitions(night)
    misses, matched = [], 0
    for when, raw, event in bat:
        hit = any(e.event == event and abs((w - when).total_seconds()) <= TOLERANCE_S
                  for w, e in transitions)
        if hit:
            matched += 1
        else:
            misses.append("  %s %s (expected %s in journal)" %
                          (when.strftime("%H:%M:%S"), raw, event))
    if bat:
        lines.append(f"NINA ground truth: {matched}/{len(bat)} matched within {TOLERANCE_S}s")
        if misses:
            lines.append("DIVERGENCE — missed NINA transitions:")
            lines += misses

    # --- guard counterfactuals
    would = [(w, e) for w, e in transitions if e.data.get("guard_would")]
    if would:
        lines.append(f"Guard counterfactuals ({len(would)}) — guards WOULD have refused:")
        for w, e in would:
            lines.append("  %s  %s: %s" % (w.strftime("%H:%M:%S"), e.event,
                                           e.data["guard_would"]))
    elif transitions:
        lines.append("Guard counterfactuals: none — evidence supported every guarded transition.")

    # --- roof relay fires (decision-diff): every observed fire, with what
    # Invariant A's guards would have said from live evidence at that moment.
    fires = [(w, e) for w, e in entries if e.event == "ROOF_FIRE_OBSERVED"]
    if fires:
        refused = [(w, e) for w, e in fires if e.data.get("guard_would")]
        lines.append(f"Roof relay fires observed: {len(fires)}, "
                     f"guards would have refused {len(refused)}")
        for w, e in refused:
            lines.append("  %s  %s: %s   evidence=%s" % (
                w.strftime("%H:%M:%S"), e.data.get("direction"),
                e.data["guard_would"], e.data.get("evidence")))

    # --- ignored events (reality outside the model)
    ignored = [(w, e) for w, e in entries
               if e.kind == "note" and e.data.get("ignored_in_state")]
    if ignored:
        lines.append(f"Events with no table row ({len(ignored)}):")
        for w, e in ignored[:10]:
            lines.append("  %s  %s ignored in %s" % (
                w.strftime("%H:%M:%S"), e.event, e.data["ignored_in_state"]))

    rejected = [(w, e) for w, e in entries if e.kind == "rejected"]
    if rejected:
        lines.append(f"Rejected events ({len(rejected)}):")
        for w, e in rejected[:10]:
            lines.append("  %s  %s: %s" % (w.strftime("%H:%M:%S"), e.event, e.guard))

    verdict = "CLEAN" if (not misses and bat) else (
        "no NINA activity to verify against" if not bat else "DIVERGED")
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


def main():
    # The report must never die on encoding: it prints before it posts, so an
    # unprintable character under the task's cp1252 console silently costs the
    # morning verdict.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    night = None
    for i, a in enumerate(sys.argv):
        if a == "--night" and i + 1 < len(sys.argv):
            night = sys.argv[i + 1]
    if night is None:
        night = (datetime.now() - timedelta(hours=12)).date().isoformat()
    report = build_report(night)
    print(report)
    if "--quiet" not in sys.argv:
        try:
            from cmd_processing import social_server
            social_server.post_social_message(report)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
