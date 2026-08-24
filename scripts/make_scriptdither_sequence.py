"""
make_scriptdither_sequence.py
Build a NINA sequence that dithers by calling scripts/dither_now.cmd instead of
using NINA's own Direct Guider.

WHY A SCRIPT AND NOT A HAND-EDIT. The sequence is 67 KB of NINA's
$id/$ref object graph. Editing that by hand is a good way to produce a file
that loads but behaves subtly differently, and there is no diff a human can
usefully read. Generating it means the change is reproducible, re-runnable when
the source sequence changes, and reviewable as ~40 lines instead of as a JSON
blob.

WHAT IT CHANGES, and nothing else:

  Each `DitherAfterExposures` trigger owns a TriggerRunner container holding a
  single `Dither` item. Only that item is swapped for a WhenPlugin
  `ExternalScript` pointing at dither_now.cmd. The trigger itself -- including
  its "AfterExposures" cadence -- is left exactly as it was, so NINA still
  decides WHEN to dither and only WHAT it does changes.

  That is deliberately the smallest possible intervention. Removing the trigger
  and adding a script item to the exposure container instead would also work,
  but it would move the cadence logic out of NINA and into the sequence
  structure, which is a second change riding along with the first.

The source sequence is never modified. The output is a NEW file; point
cfg["nina"]["sequence_input"] at it to use it, and point it back to revert.

Usage:
    python scripts/make_scriptdither_sequence.py
    python scripts/make_scriptdither_sequence.py --src X.json --out Y.json
"""
import argparse
import json
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

DITHER_CMD = r"C:\Users\iriso\Documents\development\srt\scripts\dither_now.cmd"

# The item type is provided by the WhenPlugin plugin, not by core NINA. It is
# already in use in cdk_exo_sequence.json (which calls smessage.bat), so the
# plugin is installed -- but a NINA install without it will fail to load this
# sequence, which is why the type string is recorded here rather than guessed.
EXTERNAL_SCRIPT_TYPE = "WhenPlugin.When.ExternalScript, WhenPlugin"


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def max_id(doc):
    """Highest numeric $id in the document, so new nodes cannot collide."""
    best = 0
    for node in walk(doc):
        try:
            best = max(best, int(node.get("$id", 0)))
        except (TypeError, ValueError):
            pass
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cmd", default=DITHER_CMD)
    args = ap.parse_args()

    from configs import config
    cfg = config.data()
    src = args.src or cfg["nina"]["sequence_input"]
    out = args.out or os.path.join(os.path.dirname(src),
                                   "cdk_full_sequence_scriptdither.json")

    with open(src, encoding="utf-8-sig") as fh:
        doc = json.load(fh)

    nid = max_id(doc) + 1
    swapped = 0
    for node in walk(doc):
        if "DitherAfterExposures" not in str(node.get("$type", "")):
            continue
        runner = node.get("TriggerRunner") or {}
        items = (runner.get("Items") or {}).get("$values")
        if not items:
            continue
        parent_id = runner.get("$id")
        new_items = []
        for it in items:
            if "SequenceItem.Guider.Dither" in str(it.get("$type", "")):
                new_items.append({
                    "$id": str(nid),
                    "$type": EXTERNAL_SCRIPT_TYPE,
                    "Script": '"%s"' % args.cmd,
                    "Parent": {"$ref": parent_id},
                    "ErrorBehavior": 0,
                    "Attempts": 1,
                })
                nid += 1
                swapped += 1
            else:
                new_items.append(it)
        runner["Items"]["$values"] = new_items

    if not swapped:
        print("no Dither items found inside DitherAfterExposures triggers -- "
              "nothing written. Has the source sequence changed?")
        return 1

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    print("source : %s" % src)
    print("output : %s" % out)
    print("swapped %d Dither item(s) for ExternalScript -> %s" % (swapped, args.cmd))
    print("\nTo use it, set in configs/config_private.py (or wherever nina lives):")
    print('    cfg["nina"]["sequence_input"] = r"%s"' % out)
    print("To revert, point it back at:\n    %s" % src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
