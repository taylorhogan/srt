#!/usr/bin/env python3
"""Write AAVSO Extended Format files for a monitored star (photometry/aavso.py).

    python scripts/aavso_export.py ngc7790 --filter L --obscode XXXX
    python scripts/aavso_export.py hubblev1 --filter L --name "M31 V1" --maglimit 20 --fov 10
    python scripts/aavso_export.py ngc7790 --star "CF Cas" --nights 2026-10-25 2026-10-26

Runs where the frames are (the observatory PC). The files land in
local/lightcurves/aavso/<key>_<FILT>.txt; upload each through WebObs
(https://app.aavso.org/webobs/, "Upload a file"). Nothing is sent from here.
"""
import argparse
import os
import sys

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dso", help="the queue entry (hubblev1, ngc7790)")
    ap.add_argument("--filter", default="L", help="the frames' filter: L -> CV, G -> TG, B -> TB, R -> TR")
    ap.add_argument("--obscode", default="", help="your AAVSO observer code (or cfg['aavso']['obscode'])")
    ap.add_argument("--name", default=None, help="the VSX name when the entry's name is not it (one star only)")
    ap.add_argument("--star", default=None, help="one star of an entry that lists several")
    ap.add_argument("--nights", nargs="*", default=None, help="only these nights (YYYY-MM-DD)")
    ap.add_argument("--fov", type=float, default=30.0, help="VSP chart field, arcmin")
    ap.add_argument("--maglimit", type=float, default=16.0, help="VSP sequence depth")
    ap.add_argument("--refresh-sequence", action="store_true", help="fetch the sequence again")
    a = ap.parse_args(argv)
    from photometry import aavso
    res = aavso.export(a.dso, a.filter, a.obscode, name=a.name, nights=a.nights, fov_arcmin=a.fov,
                       maglimit=a.maglimit, star_name=a.star, progress_cb=print, refresh_sequence=a.refresh_sequence)
    for k, why in res["skipped"].items():
        print("skipped %s: %s" % (k, why))
    for f in res["files"]:
        print("wrote", f)
        n = len(res["lines"].get(os.path.basename(f).rsplit("_", 1)[0], []))
        print("  %d observation%s" % (n, "" if n == 1 else "s"))
    if not res["files"]:
        print("nothing written")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
