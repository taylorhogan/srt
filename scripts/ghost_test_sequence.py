"""Write the N.I.N.A sequence for the ghost test: one bright star at five places on the sensor.

    python scripts/ghost_test_sequence.py [--date YYYY-MM-DD] [--start HH:MM] [--out FILE]
                                          [--star NAME] [--ra-hours H] [--dec-deg D]
                                          [--offset-ra-arcmin 8] [--offset-dec-arcmin 6]
                                          [--exposure 30] [--frames 4]

The reflection ghost seen on NGC 1055 (2026-11-19) is an inverted image of a bright star
through a centre: a star at offset r from that centre puts its ghost at about -m r. The
centre and m are measured by putting one star at five known places and reading where
the ghost lands each time. This writes five target containers from the normal template,
the star centred (twice: the template carries three filters a container, so L, R, G then
L, B) and offset east, west, north and south in L, short exposures, with the template's Center step before each
so the pointing is solved, not assumed.

Nothing here moves hardware: it writes a file for the owner to load. The same roof and
mount rules apply to running it as to any sequence.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

# alpha Trianguli (Mothallah), V 3.4, J2000; transits around 21:30 local in late November
# and sits 78 deg up: bright enough for a ghost a thousandth as bright to stand clear of the
# sky in 30 s, high enough to be above every tree.
DEFAULT_STAR = ("alphaTri", 1.884694, 29.5786)


def positions(name, ra_hours, dec_deg, off_ra_arcmin, off_dec_arcmin):
    """(slot name, ra_hours, dec_deg, filters) for centre, E, W, N, S of the star."""
    import math
    dra_h = (off_ra_arcmin / 60.0) / (15.0 * math.cos(math.radians(dec_deg)))
    ddec = off_dec_arcmin / 60.0
    return [
        (name + "_c", ra_hours, dec_deg, ["L", "R", "G"]),
        (name + "_c2", ra_hours, dec_deg, ["L", "B"]),
        (name + "_e", ra_hours + dra_h, dec_deg, ["L"]),
        (name + "_w", ra_hours - dra_h, dec_deg, ["L"]),
        (name + "_n", ra_hours, dec_deg + ddec, ["L"]),
        (name + "_s", ra_hours, dec_deg - ddec, ["L"]),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", help="evening (local), default today")
    ap.add_argument("--start", default="19:30", help="first slot's local start (HH:MM)")
    ap.add_argument("--minutes", type=float, default=12, help="minutes per position")
    ap.add_argument("--out", help="output path (default local/ghost_test_<date>.json)")
    ap.add_argument("--star", default=DEFAULT_STAR[0])
    ap.add_argument("--ra-hours", type=float, default=DEFAULT_STAR[1])
    ap.add_argument("--dec-deg", type=float, default=DEFAULT_STAR[2])
    ap.add_argument("--offset-ra-arcmin", type=float, default=8.0)
    ap.add_argument("--offset-dec-arcmin", type=float, default=6.0)
    ap.add_argument("--exposure", type=float, default=30.0, help="seconds per frame")
    ap.add_argument("--frames", type=int, default=4, help="frames per filter per position")
    ap.add_argument("--template", help="sequence template (default cfg nina.sequence_input, else local/sequence_template.json)")
    ap.add_argument("--lint-scripts", action="store_true", help="lint the Windows script paths (PC only)")
    args = ap.parse_args(argv)

    from configs import config
    from nina_gen import nina_sequence_gen as gen
    import pytz
    cfg = config.data()
    tz = pytz.timezone(cfg["location"]["timezone"])
    day = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now(tz).date()
    hh, mm = (int(x) for x in args.start.split(":"))
    t0 = tz.localize(datetime(day.year, day.month, day.day, hh, mm))
    template = Path(args.template) if args.template else Path(cfg["nina"].get("sequence_input", ""))
    if not template.exists():
        template = Path("local/sequence_template.json")
    out = Path(args.out) if args.out else Path("local/ghost_test_%s.json" % day.isoformat())

    slots = []
    for k, (name, ra, dec, filters) in enumerate(positions(args.star, args.ra_hours, args.dec_deg,
                                                           args.offset_ra_arcmin, args.offset_dec_arcmin)):
        start = t0 + timedelta(minutes=args.minutes * k)
        slots.append({"name": name, "ra_hours": ra, "dec_degrees": dec,
                      "seconds": args.minutes * 60.0, "start": start,
                      "end": start + timedelta(minutes=args.minutes),
                      "filters": {f: 1 for f in filters}, "_want": filters})
    gen.generate_slots_sequence(template, slots, out, lint_scripts=args.lint_scripts)

    # The generator sizes 300 s frames to the slot; the test wants short frames and a fixed
    # count per filter. Rewrite every block of every slot, then read it back.
    with open(out, "r", encoding="utf-8") as f:
        seq = json.load(f)
    area = gen._find_target_area(seq)
    containers = [it for it in gen._items_of(area) if gen._short_type(it) == "DeepSkyObjectContainer"]
    if len(containers) != len(slots):
        raise SystemExit("expected %d target containers, found %d" % (len(slots), len(containers)))
    report = []
    for c, slot in zip(containers, slots):
        want = slot["_want"]
        for se in gen._collect_smart_exposures(c):
            node = gen._filter_node(se)
            fname = (node or {}).get("_name") if node else None
            plan = gen.written_plan(c)
            # keep the blocks the plan gave a count; set their count and exposure
            n = gen._iterations(se)
            if n > 0:
                gen._update_smart_exposure(se, args.frames)
                gen._set_exposure_time(se, args.exposure)
        got = gen.written_plan(c)
        report.append((slot["name"], slot["ra_hours"], slot["dec_degrees"], got, slot["start"], slot["end"]))
        missing = [f for f in want if got.get(f, 0) <= 0]
        if missing:
            raise SystemExit("%s: the template could not carry %s (blocks: %s)" % (slot["name"], missing, got))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(seq, f, indent=2)
    print("wrote %s from %s" % (out, template))
    for name, ra, dec, got, s, e in report:
        print("  %-12s RA %9.5fh Dec %+8.4f  %s-%s  %s x %.0f s" % (
            name, ra, dec, s.strftime("%H:%M"), e.strftime("%H:%M"),
            ", ".join("%s %d" % kv for kv in got.items()), args.exposure))
    print("Load it in N.I.N.A by hand. It opens the roof and parks like any night's file; the")
    print("conductor's roof and mount rules apply. Afterwards: python scripts/ghost_test_measure.py <LIGHT folder>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
