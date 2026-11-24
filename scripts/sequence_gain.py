#!/usr/bin/env python3
"""Put a camera gain on a N.I.N.A calibration sequence (flats, darks, bias).

The lights' gain is written by the sequence generator from ``cfg["nina"]["gain"]``
(``nina_sequence_gen.apply_filter_gain``). The calibration frames come from two
hand-made sequences on the observatory PC, ``flat_sequence.json`` (run after
every night by the flats phase) and ``darks_and_bias.json`` (run by hand), and
the stacker keeps only calibration frames at the lights' (gain, offset)
(``stacker.match_calibration``). This tool makes those two files agree with the
gain table, so a filter moved to another gain is calibrated at it:

``flats``: every ``TrainedFlatExposure`` block whose filter is in the table is
rewritten as an ``AutoExposureFlat`` at that gain and offset. A trained flat
looks its exposure up in the profile's trained table, keyed by filter, binning,
gain and offset, and the table holds nothing at a new gain, so the block would
fail validation ("no trained exposure"); the auto-exposure flat finds the
exposure itself (histogram target 50 %, within 10 %, between --min and --max
seconds) and, when it succeeds, writes the trained entry for next time. The
child items are the same in both (close cover, light on, switch filter, set
brightness, the exposure loop, light off, open cover), so the block is moved
over in place. Other filters' blocks are left alone.

``darks``: every ``TakeExposure`` of type BIAS or DARK gets the gain and offset;
the output is a new sequence beside the original, named after the gain, for a
hand run with the camera covered.

Nothing here moves the mount or the roof. The files are written where asked;
copying them to the PC and loading them in N.I.N.A is the operator's step.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Iterable, Optional

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

TRAINED = "NINA.Sequencer.SequenceItem.FlatDevice.TrainedFlatExposure, NINA.Sequencer"
AUTO = "NINA.Sequencer.SequenceItem.FlatDevice.AutoExposureFlat, NINA.Sequencer"
TAKE = "NINA.Sequencer.SequenceItem.Imaging.TakeExposure, NINA.Sequencer"
SWITCH = "NINA.Sequencer.SequenceItem.FilterWheel.SwitchFilter, NINA.Sequencer"


def _items(node: dict) -> list:
    return (node.get("Items") or {}).get("$values") or []


def _conditions(node: dict) -> list:
    return (node.get("Conditions") or {}).get("$values") or []


def walk(node: Any) -> Iterable[dict]:
    """Every dict in the tree, parents before children."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def block_filter(block: dict) -> Optional[str]:
    for item in _items(block):
        if item.get("$type") == SWITCH:
            flt = item.get("Filter") or {}
            return flt.get("_name") or flt.get("Name")
    return None


def exposure_items(block: dict) -> list:
    """The TakeExposure(s) inside a flat block's imaging container."""
    out = []
    for item in _items(block):
        for sub in walk(item):
            if sub.get("$type") == TAKE:
                out.append(sub)
    return out


def trained_to_auto(block: dict, gain: int, offset: int, min_s: float, max_s: float,
                    target: float = 0.5, tolerance: float = 0.1) -> dict:
    """Rewrite one TrainedFlatExposure block as an AutoExposureFlat at *gain*."""
    if block.get("$type") != TRAINED:
        raise ValueError("not a TrainedFlatExposure block")
    takes = exposure_items(block)
    if not takes:
        raise ValueError("flat block has no TakeExposure")
    block["$type"] = AUTO
    block["Name"] = "Auto Exposure Flat"
    block["MinExposure"] = float(min_s)
    block["MaxExposure"] = float(max_s)
    block["HistogramTargetPercentage"] = float(target)
    block["HistogramTolerancePercentage"] = float(tolerance)
    for te in takes:
        te["Gain"] = int(gain)
        te["Offset"] = int(offset)
        te["ImageType"] = "FLAT"
        # The auto flat overwrites ExposureTime with what it finds; a sane start.
        te["ExposureTime"] = float(min(max(float(te.get("ExposureTime") or 1.0), min_s), max_s))
        te["ExposureCount"] = 0
    return block


def convert_flats(seq: dict, gain_map: dict, min_s: float = 2.0, max_s: float = 120.0) -> dict:
    """{filter: (gain, offset)} applied to the trained flat blocks it names.
    Returns {filter: {"gain", "offset", "was"}} for what changed."""
    changed: dict = {}
    for node in list(walk(seq)):
        if node.get("$type") != TRAINED:
            continue
        name = block_filter(node)
        if name not in gain_map:
            continue
        g, o = gain_map[name]
        was = [(t.get("Gain"), t.get("Offset"), t.get("ExposureTime")) for t in exposure_items(node)]
        trained_to_auto(node, int(g), int(o), min_s, max_s)
        changed[name] = {"gain": int(g), "offset": int(o), "was": was}
    return changed


LOOP = "NINA.Sequencer.Conditions.LoopCondition, NINA.Sequencer"


def reset_progress(seq: dict) -> int:
    """Every loop condition back to 0 completed iterations. Returns how many were not.

    N.I.N.A saves a sequence with its run state: the hand-made darks_and_bias.json
    on the PC was saved after a run with its bias loop at 50/50 and its dark loop
    at 1/20, and the gain-26 copy inherited both. Loaded, N.I.N.A skipped the bias
    block outright (2026-11-24: "Iterations: 50/50 ... Items[SKIPPED]") and took
    19 darks, and the operator, watching the bias block finish at once, took it
    for done. A generated sequence starts from nothing."""
    n = 0
    for node in walk(seq):
        if node.get("$type") == LOOP and int(node.get("CompletedIterations", 0) or 0):
            n += 1
        if node.get("$type") == LOOP:
            node["CompletedIterations"] = 0
    return n


def convert_darks(seq: dict, gain: int, offset: int, name: Optional[str] = None) -> dict:
    """Gain and offset onto every BIAS and DARK exposure, every loop counter back
    to 0 (``reset_progress``). Returns counts by type."""
    counts: dict = {}
    for node in walk(seq):
        if node.get("$type") == TAKE and str(node.get("ImageType", "")).upper() in ("BIAS", "DARK"):
            node["Gain"] = int(gain)
            node["Offset"] = int(offset)
            node["ExposureCount"] = 0
            kind = str(node["ImageType"]).upper()
            counts[kind] = counts.get(kind, 0) + 1
    reset_progress(seq)
    if name and isinstance(seq, dict) and "Name" in seq:
        seq["Name"] = name
    return counts


def check_refs(seq: dict) -> list:
    """Json.NET reference integrity: every $ref names an earlier $id, ids unique."""
    ids: set = set()
    problems = []
    for node in walk(seq):
        i = node.get("$id")
        if i is not None:
            if i in ids:
                problems.append("duplicate $id %s" % i)
            ids.add(i)
        r = node.get("$ref")
        if r is not None and r not in ids:
            problems.append("$ref %s before its $id" % r)
    return problems


def gain_map_from_config() -> dict:
    from configs import config
    raw = (config.data().get("nina") or {}).get("gain") or {}
    out = {}
    for k, v in raw.items():
        try:
            out[str(k)] = (int(v[0]), int(v[1]))
        except Exception:
            continue
    return out


def _load(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def _save(seq: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(seq, fh, indent=2)
        fh.write("\n")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flats", help="trained flats -> auto-exposure flats at the table's gain")
    f.add_argument("src")
    f.add_argument("dst")
    f.add_argument("--filters", default=None,
                   help="comma list, e.g. Ha,O-III,S-II (default: every filter in cfg nina.gain)")
    f.add_argument("--gain", type=int, default=None, help="with --filters: this gain for all of them")
    f.add_argument("--offset", type=int, default=None)
    f.add_argument("--min", type=float, default=2.0, help="shortest flat the search may pick, s")
    f.add_argument("--max", type=float, default=120.0, help="longest, s")
    d = sub.add_parser("darks", help="bias and darks at one gain")
    d.add_argument("src")
    d.add_argument("dst")
    d.add_argument("--gain", type=int, required=True)
    d.add_argument("--offset", type=int, required=True)
    d.add_argument("--name", default=None, help="sequence name (default: <old>_gain<g>)")
    args = ap.parse_args(argv)

    seq = _load(args.src)
    if args.cmd == "flats":
        if args.filters:
            if args.gain is None or args.offset is None:
                ap.error("--filters needs --gain and --offset")
            gmap = {x.strip(): (args.gain, args.offset) for x in args.filters.split(",") if x.strip()}
        else:
            gmap = gain_map_from_config()
        if not gmap:
            print("gain table empty: nothing to change")
            return 1
        changed = convert_flats(seq, gmap, args.min, args.max)
        for name, rec in changed.items():
            print("%s: auto-exposure flat at gain %d offset %d (was %s)" % (name, rec["gain"], rec["offset"], rec["was"]))
        missing = sorted(set(gmap) - set(changed))
        if missing:
            print("no trained flat block for: %s" % ", ".join(missing))
    else:
        name = args.name or "%s_gain%d" % (seq.get("Name") or "darks_and_bias", args.gain)
        counts = convert_darks(seq, args.gain, args.offset, name)
        print("%s: %s at gain %d offset %d" % (name, counts or "no BIAS/DARK exposures", args.gain, args.offset))
    problems = check_refs(seq)
    if problems:
        print("REFUSED: " + "; ".join(problems))
        return 2
    _save(seq, args.dst)
    print("wrote", args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
