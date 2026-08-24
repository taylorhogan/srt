"""
backup_state.py
Back up the small, irreplaceable observatory state to OneDrive.

WHAT IS ACTUALLY AT RISK. local/ is 5.5 GB, but ~5.5 GB of that is bulk capture
-- sky_frames, sky_bursts, iriscam_rec, allsky_frames, skyfield ephemerides.
Losing it would be a shame and would cost history, but nothing there gates the
observatory and most of it regenerates within a night. What CANNOT be
regenerated is about 4 MB of JSON:

  local/sky_solution.json      the sky camera's plate solution. Without it the
                               `seen` chart, limiting magnitude and the compass
                               annotation all stop meaning anything, and
                               recovering it needs a clear night and a blind
                               solve. The 2026-08-24 re-solve took a deliberate
                               session to produce.
  local/scope_marker_parked.json   the parked-pose reference the roof-close
                               gate compares against. Losing it means the gate
                               says "unknown" and refuses to close the roof
                               until someone re-parks and re-records it.
  local/my_instructions.json   the imaging queue -- hand-authored, priorities
                               and all. Nothing else knows what to image.
  local/convergence.json       per-target per-filter convergence state; without
                               it the scheduler cannot tell a finished target
                               from an unstarted one.
  local/boundary_session*.jsonl   the 2026-08-24 collision measurements behind
                               the 187 px tolerance. Re-measuring means another
                               roof-open session with the operator present.
  local/*_log.jsonl            sky, rain, forecast, shadow. Time series that
                               only accumulate in real time -- a year of them
                               cannot be rebuilt in an afternoon.

DESTINATION is the user's OneDrive, which is the only cloud folder actually
syncing on this machine (Dropbox is installed but has no running process and
nothing newer than May 2026). That makes the backup off-machine without any
new infrastructure.

SECRETS ARE EXCLUDED BY DEFAULT. configs/config_private.py holds API keys and
tokens, and losing it means re-entering every credential -- so it is genuinely
worth backing up, but writing secrets into a cloud folder is the operator's
decision, not this script's. Pass --include-secrets to opt in.

SNAPSHOTS, NOT A MIRROR. Each run writes a dated zip rather than overwriting a
single copy, because the failure this protects against is not only disk loss
but corruption: a mirror faithfully replicates a truncated sky_solution.json
and destroys the good one. Old snapshots are pruned to --keep.

Usage:
    python scripts/backup_state.py
    python scripts/backup_state.py --keep 60 --include-secrets
    python scripts/backup_state.py --dest "D:/somewhere/else"
"""
import argparse
import glob
import json
import os
import sys
import zipfile
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DEST = os.path.join(os.path.expanduser("~"), "OneDrive", "IrisBackup")

# Globs relative to the repo root. Deliberately narrow: everything here is
# small and irreplaceable. Bulk capture directories are NOT listed, and adding
# one would quietly turn a 4 MB daily snapshot into a gigabyte.
INCLUDE = [
    "local/*.json",
    "local/*.jsonl",
    "local/parked_refs/*",          # reference frames for the parked detector
    "configs/config_public.py",     # not secret, but pairs with the private one
    "safety.txt",
    "my_calendar.json",
]
SECRETS = ["configs/config_private.py"]


def collect(include_secrets=False):
    pats = list(INCLUDE) + (SECRETS if include_secrets else [])
    out = []
    for pat in pats:
        for p in glob.glob(os.path.join(ROOT, pat)):
            if os.path.isfile(p):
                out.append(p)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--keep", type=int, default=30,
                    help="how many dated snapshots to retain (default 30)")
    ap.add_argument("--include-secrets", action="store_true",
                    help="also back up configs/config_private.py (API keys!)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = collect(args.include_secrets)
    if not files:
        print("nothing matched -- refusing to write an empty backup")
        return 1
    total = sum(os.path.getsize(f) for f in files)
    print("%d files, %.1f MB" % (len(files), total / 1048576.0))

    if not os.path.isdir(os.path.dirname(args.dest)) and not args.dry_run:
        print("destination parent does not exist: %s" % os.path.dirname(args.dest))
        return 1
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.dest, "iris_state_%s.zip" % stamp)

    if args.dry_run:
        print("DRY RUN -> would write %s" % out)
        for f in files[:12]:
            print("   %s" % os.path.relpath(f, ROOT))
        if len(files) > 12:
            print("   ... and %d more" % (len(files) - 12))
        return 0

    os.makedirs(args.dest, exist_ok=True)
    manifest = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
                "root": ROOT, "secrets_included": bool(args.include_secrets),
                "files": [os.path.relpath(f, ROOT).replace("\\", "/") for f in files]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.relpath(f, ROOT).replace("\\", "/"))
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    # Verify before pruning anything. A backup that was never read back is a
    # guess, and pruning on the strength of a guess is how you end up with N
    # corrupt snapshots and no good one.
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        names = set(z.namelist())
    if bad is not None:
        print("VERIFY FAILED on %s -- keeping older snapshots, not pruning" % bad)
        return 1
    missing = [m for m in manifest["files"] if m not in names]
    if missing:
        print("VERIFY FAILED: %d files missing from the archive -- not pruning"
              % len(missing))
        return 1
    print("wrote and verified %s (%.1f MB compressed)"
          % (out, os.path.getsize(out) / 1048576.0))

    snaps = sorted(glob.glob(os.path.join(args.dest, "iris_state_*.zip")))
    for old in snaps[:-args.keep] if args.keep > 0 else []:
        try:
            os.remove(old)
            print("pruned %s" % os.path.basename(old))
        except OSError as e:
            print("could not prune %s: %s" % (old, e))
    print("%d snapshot(s) retained in %s" % (len(snaps[-args.keep:]), args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
