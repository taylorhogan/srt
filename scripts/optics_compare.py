"""
optics_compare.py
Pool the nightly optics_trend.json files and say whether the OPTICS have
actually changed -- as distinct from the seeing having changed.

WHY THESE METRICS AND NOT FWHM. Raw FWHM moves with seeing, altitude and which
field was imaged; measured here, it swung +88%/-41% across nights with no
optical change at all. The trend metrics are built to survive that:
seeing_floor is the best patch of field (the seeing), and field_excess and
edge_excess are the blur ADDED on top of it, seeing removed in quadrature. So
those are comparable between nights; FWHM is not.

WHAT THIS REFUSES TO DO. It will not report a sweet-spot trend from rows whose
fit was clamped. Before 28846c5 a degenerate fit was written as +/-1.5 per axis
=> r = 2.121, which is an ATTRACTOR: every bad night lands on the same number,
so a run of them looks like a smooth rise in collimation error. Two such rows
(2026-08-23, 08-24) sit in the existing history. They are detected by value
here, because they predate the sweet_spot_status field, and excluded.

The direction of a saturated fit IS kept -- if several degenerate nights all
run off the same way, that is weak evidence the true minimum is moving that
way, and it is reported separately from any magnitude.

Usage:
    python scripts/optics_compare.py
    python scripts/optics_compare.py --recent 5      # compare last 5 vs earlier
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# The pre-fix clamp. Rows at this value are degenerate fits, not measurements.
_CLAMP = 1.5
_CLAMP_R = math.hypot(_CLAMP, _CLAMP)


def sweet_ok(row):
    """True only if this row's sweet spot is a real measurement.

    Handles both schemas: rows written after 28846c5 carry sweet_spot_status,
    older ones do not and must be judged by value.
    """
    stat = row.get("sweet_spot_status")
    if stat is not None:
        return stat == "ok"
    x, y, r = (row.get("sweet_spot_x"), row.get("sweet_spot_y"),
               row.get("sweet_spot_r"))
    if x is None or y is None or r is None:
        return False
    if abs(abs(x) - _CLAMP) < 1e-6 or abs(abs(y) - _CLAMP) < 1e-6:
        return False
    return True


def load(root=None):
    root = root or r"C:/Users/iriso/Documents/N.I.N.A/Targets"
    rows = []
    for f in glob.glob(os.path.join(root, "**", "optics_trend.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, list):
            rows += d
    # One row per night; a night imaged on two targets would otherwise weight twice.
    rows.sort(key=lambda r: r.get("night", ""))
    return rows


def _fmt(v, spec="%.2f"):
    return spec % v if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--recent", type=int, default=3,
                    help="how many recent nights form the 'now' group (default 3)")
    args = ap.parse_args()

    rows = load(args.root)
    if not rows:
        print("no optics_trend.json data found")
        return 1

    print("%d night-rows, %s to %s\n" % (len(rows), rows[0]["night"], rows[-1]["night"]))
    print("night       dso      seeing  field_ex  edge_ex   radial  uniform@deg  sweet")
    for r in rows:
        if sweet_ok(r):
            sw = "r=%.2f (%+.2f,%+.2f)" % (r["sweet_spot_r"], r["sweet_spot_x"],
                                           r["sweet_spot_y"])
        else:
            sw = r.get("sweet_spot_status") or (
                "SATURATED" if r.get("sweet_spot_r") is not None else "no fit")
            d = r.get("sweet_spot_direction_deg")
            if d is not None:
                sw += " -> %.0f deg" % d
            elif r.get("sweet_spot_x") is not None:
                sw += " -> %.0f deg" % math.degrees(
                    math.atan2(r["sweet_spot_y"], r["sweet_spot_x"]))
        print("  %-10s %-8s %5s   %6s   %6s   %6s  %4s@%-4s  %s" % (
            r.get("night", "?"), r.get("dso", "?")[:8],
            _fmt(r.get("seeing_floor_arcsec")), _fmt(r.get("field_excess_arcsec")),
            _fmt(r.get("edge_excess_arcsec")), _fmt(r.get("radial_fraction"), "%+.2f"),
            _fmt(r.get("uniform_fraction")), _fmt(r.get("uniform_angle_deg"), "%.0f"),
            sw))

    usable = [r for r in rows if sweet_ok(r)]
    print("\nsweet spot usable on %d of %d nights" % (len(usable), len(rows)))

    # --- the comparison. Seeing-robust metrics only.
    print("\n%s" % ("-" * 66))
    n = min(args.recent, max(1, len(rows) // 2))
    recent, earlier = rows[-n:], rows[:-n]
    if not earlier:
        print("not enough history to compare (need more than %d nights)" % n)
        return 0
    print("recent %d nights vs earlier %d" % (len(recent), len(earlier)))
    # Target confounding. These metrics are meant to be properties of the
    # optics, but they are measured from whatever stars the field provides:
    # a sparse field fills fewer cells and biases the per-cell percentiles that
    # field_excess and edge_excess are built from. If the two groups do not
    # share a target, a difference between them can be the field rather than
    # the telescope, and the comparison cannot tell which.
    t_recent = {r.get("dso") for r in recent}
    t_earlier = {r.get("dso") for r in earlier}
    if not (t_recent & t_earlier):
        print("  !! recent group is %s, earlier is %s -- NO SHARED TARGET."
              % (",".join(sorted(t_recent)), ",".join(sorted(t_earlier))))
        print("     Any difference below may be the field, not the optics.")
        print("     Re-image an earlier target to separate them.")
    print("  metric              earlier    recent    change    verdict")
    for key, label, higher_is_worse in (
            ("seeing_floor_arcsec", "seeing floor", None),
            ("field_excess_arcsec", "field excess", True),
            ("edge_excess_arcsec", "edge excess", True),
            ("median_ecc", "median ecc", True),
            ("radial_fraction", "radial (coma)", True),
            ("uniform_fraction", "uniform (NOT optics)", True)):
        a = [r[key] for r in earlier if isinstance(r.get(key), (int, float))]
        b = [r[key] for r in recent if isinstance(r.get(key), (int, float))]
        if not a or not b:
            continue
        ma, mb = st.median(a), st.median(b)
        spread = (max(a) - min(a)) if len(a) > 1 else float("inf")
        delta = mb - ma
        # A change only counts if it clears the night-to-night scatter of the
        # baseline. With a handful of nights that bar is high, and it should be.
        if higher_is_worse is None:
            verdict = "(context only -- this is the weather)"
        elif abs(delta) > spread:
            verdict = "CHANGED (exceeds baseline spread %.2f)" % spread
        else:
            verdict = "within scatter (spread %.2f)" % spread
        print("  %-20s %7.2f   %7.2f   %+7.2f   %s" % (label, ma, mb, delta, verdict))

    if len(usable) >= 4:
        print("\n  sweet spot drift over %d usable nights:" % len(usable))
        first, last = usable[0], usable[-1]
        print("    %s (%+.2f,%+.2f) -> %s (%+.2f,%+.2f)" % (
            first["night"], first["sweet_spot_x"], first["sweet_spot_y"],
            last["night"], last["sweet_spot_x"], last["sweet_spot_y"]))
    else:
        print("\n  sweet spot: only %d usable night(s) -- not enough to trend."
              % len(usable))
        print("    Collect more before reading anything into it; the metric is")
        print("    fragile and degenerates whenever the field is nearly flat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
