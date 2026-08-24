"""
expand_sequence_blocks.py
Give a NINA sequence enough SmartExposure blocks for the `filters` command to
work on it.

THE BUG THIS FIXES. nina_sequence_gen._apply_filter_plan opens with:

    # Template order is L, R, G, B (4 blocks)
    if len(smart_exposures) < 4:
        return {}

so a template with fewer than four SmartExposure blocks silently ignores EVERY
filter plan -- explicit or automatic, two filters or three. cdk_full_sequence
.json, the narrowband template this observatory actually images with, has two
(Ha, O-III). The consequence is not theoretical: on 2026-08-23 the queue held
an explicit plan of {Ha: 20, S-II: 20}, generate_sequence returned {}, and the
template's own built-in Ha x20 / O-III x30 ran instead. No error, no warning,
and the night looked normal -- the plan simply had no effect.

So `filters <dso> ...` has never done anything on this template. Any past use
of it was a no-op.

WHAT THIS DOES. Clones the last SmartExposure block until the sequence has at
least `--blocks` of them (default 4), appending each clone to the same
container as the originals. Cloned blocks are deep copies with every $id in the
subtree renumbered and every internal $ref remapped to match, because NINA's
serialisation is an object graph and duplicating an $id silently aliases two
nodes onto one.

The clones' filter and iteration count do not matter: _apply_explicit_plan
rewrites both, assigning plan entries to blocks in order and zeroing whatever
is left over. What matters is only that the blocks EXIST and carry a filter
node, since that is what the guard and the assignment both count.

Usage:
    python scripts/expand_sequence_blocks.py                       # 2 -> 4
    python scripts/expand_sequence_blocks.py --blocks 4 --out X.json
"""
import argparse
import copy
import json
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def max_id(doc):
    best = 0
    for n in walk(doc):
        try:
            best = max(best, int(n.get("$id", 0)))
        except (TypeError, ValueError):
            pass
    return best


def renumber(sub, start):
    """Give every $id in *sub* a fresh number and remap internal $refs.

    Refs pointing OUTSIDE the subtree (e.g. the block's Parent) are left alone
    so the clone stays attached to the same container as the original.
    """
    mapping = {}
    nid = start
    for n in walk(sub):
        if "$id" in n:
            mapping[str(n["$id"])] = str(nid)
            n["$id"] = str(nid)
            nid += 1
    for n in walk(sub):
        r = n.get("$ref")
        if r is not None and str(r) in mapping:
            n["$ref"] = mapping[str(r)]
    return nid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--blocks", type=int, default=4,
                    help="minimum SmartExposure blocks required (default 4, "
                         "which is what _apply_filter_plan demands)")
    args = ap.parse_args()

    from configs import config
    from nina_gen import nina_sequence_gen as g
    cfg = config.data()
    src = args.src or cfg["nina"]["sequence_input"]
    out = args.out or os.path.join(os.path.dirname(src),
                                   "cdk_full_sequence_4block.json")

    doc = json.load(open(src, encoding="utf-8-sig"))
    ses = g._collect_smart_exposures(doc)
    print("source: %s" % src)
    print("  SmartExposure blocks: %d  filters: %s"
          % (len(ses), [(g._filter_node(s) or {}).get("_name") for s in ses]))
    if len(ses) >= args.blocks:
        print("  already has >= %d blocks; nothing to do" % args.blocks)
        return 0
    if not ses:
        print("  no SmartExposure blocks to clone from")
        return 1

    # The container holding the blocks: find the list that contains the last one.
    last = ses[-1]
    holder = None
    for node in walk(doc):
        vals = node.get("$values") if isinstance(node, dict) else None
        if isinstance(vals, list) and any(v is last for v in vals):
            holder = node
            break
    if holder is None:
        print("  could not locate the container holding the blocks")
        return 1
    parent_ref = last.get("Parent")

    nid = max_id(doc) + 1
    added = 0
    while len(g._collect_smart_exposures(doc)) < args.blocks:
        clone = copy.deepcopy(last)
        nid = renumber(clone, nid)
        # Keep the clone attached to the same parent container as the original.
        if parent_ref is not None:
            clone["Parent"] = copy.deepcopy(parent_ref)
        holder["$values"].append(clone)
        added += 1
        if added > 16:
            print("  runaway clone loop -- aborting")
            return 1

    json.dump(doc, open(out, "w", encoding="utf-8"), indent=2)
    ses2 = g._collect_smart_exposures(json.load(open(out, encoding="utf-8-sig")))
    print("output: %s" % out)
    print("  SmartExposure blocks: %d (added %d)  filters: %s"
          % (len(ses2), added,
             [(g._filter_node(s) or {}).get("_name") for s in ses2]))
    print("\nSelect it with cfg[\"nina\"][\"sequence_input\"].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
