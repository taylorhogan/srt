"""Structural lint for a generated N.I.N.A sequence: what N.I.N.A will refuse.

N.I.N.A never says no. A sequence it cannot use loads, validates, and then
SKIPS the parts it cannot run -- 2026-09-14 every exposure block was skipped
in five seconds because an extra item inside a SmartExposure made its
Validate() throw, and the first anyone knew was a target container finishing
without a sub. The rules here are the ones N.I.N.A actually enforces, read
from its source and from that night, and the generator refuses to write a
sequence that breaks one so the failure happens at noon with a Pushover
instead of at 21:19 in silence.

    problems = lint(sequence, template=template_json, check_scripts=True)
    if problems: raise SequenceLintError(problems)

Rules (each a short code, so a report can be grepped):

  IDS       every $id unique, every $ref resolves. Newtonsoft's reference
            tracker aliases a duplicated $id and refuses a dangling $ref.
  PARENT    an item inside a container's Items carries Parent {$ref: that
            container's $id}. N.I.N.A walks Parent to find the target and
            the triggers that apply.
  SMART     a SmartExposure holds exactly [SwitchFilter, TakeExposure], its
            first Condition is a LoopCondition and its first Trigger a
            DitherAfterExposures: SmartExposure.cs indexes those by position.
  FILTER    every SwitchFilter names a filter (inline FilterInfo or a $ref to
            one) with a _name.
  CENTER    every Center / CenterAndRotate is preceded by a SwitchFilter, so
            the plate solve never runs through whatever narrowband filter the
            previous block left in the wheel (2026-09-15 03:00).
  TARGET    every DeepSkyObjectContainer has a Target with a TargetName and
            InputCoordinates.
  TYPES     with a template: every $type in the sequence is one the template
            uses or one on the generator's allow-list. A type nobody meant
            to write is a type nobody tested.
  SCRIPT    with check_scripts: every ExternalScript's program exists on disk.
  ROOT      the document is a SequenceRootContainer.

Pure Python; no config, no N.I.N.A. Runs on the bare CI runner.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

# Types the generator adds that a template need not contain.
GENERATOR_TYPES = {
    "NINA.Sequencer.SequenceItem.Focuser.MoveFocuserByTemperature, NINA.Sequencer",
    "NINA.Sequencer.SequenceItem.Platesolving.CenterAndRotate, NINA.Sequencer",
    "NINA.Sequencer.SequenceItem.FilterWheel.SwitchFilter, NINA.Sequencer",
    "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
    "NINA.Core.Model.Equipment.BinningMode, NINA.Core",
    "NINA.Core.Model.Equipment.FlatWizardFilterSettings, NINA.Core",
    "NINA.Sequencer.Conditions.TimeCondition, NINA.Sequencer",
    "NINA.Sequencer.Utility.DateTimeProvider.TimeProvider, NINA.Sequencer",
}


class SequenceLintError(ValueError):
    """The generated sequence breaks a rule N.I.N.A enforces; nothing was written."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("sequence lint: %d problem(s)\n  " % len(self.problems)
                         + "\n  ".join(self.problems))


def _short(t: Any) -> str:
    return str(t or "").split(",")[0].split(".")[-1]


def _items(node: dict) -> list:
    it = node.get("Items")
    return it.get("$values", []) if isinstance(it, dict) else (it or [])


def _values(node: dict, key: str) -> list:
    v = node.get(key)
    return v.get("$values", []) if isinstance(v, dict) else (v or [])


def _walk(node: Any) -> Iterable[dict]:
    """Every dict in the document, depth first, Parent links not followed."""
    if isinstance(node, dict):
        yield node
        for k, v in node.items():
            if k != "Parent":
                yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _types(node: Any) -> set[str]:
    return {n["$type"] for n in _walk(node) if isinstance(n.get("$type"), str)}


def _script_program(script: str) -> Optional[str]:
    """The program an ExternalScript line runs: the first quoted token, or
    the first whitespace-delimited one."""
    if not script:
        return None
    m = re.match(r'\s*"([^"]+)"', script)
    if m:
        return m.group(1)
    return script.split()[0] if script.split() else None


def lint(sequence: Any, template: Any = None, check_scripts: bool = False,
         extra_types: Iterable[str] = ()) -> list[str]:
    """Return the problems, or [] when N.I.N.A will take the sequence."""
    problems: list[str] = []

    # ROOT
    if not isinstance(sequence, dict) or _short(sequence.get("$type")) != "SequenceRootContainer":
        problems.append("ROOT: document is not a SequenceRootContainer")
        return problems

    # IDS
    ids: dict[str, dict] = {}
    for n in _walk(sequence):
        i = n.get("$id")
        if i is not None:
            if i in ids:
                problems.append(f"IDS: $id {i} appears more than once "
                                f"({_short(ids[i].get('$type'))} and {_short(n.get('$type'))})")
            else:
                ids[i] = n
    for n in _walk(sequence):
        for k, v in n.items():
            if isinstance(v, dict) and "$ref" in v and v["$ref"] not in ids:
                problems.append(f"IDS: {_short(n.get('$type'))}.{k} refers to $id {v['$ref']} "
                                f"which does not exist")

    def filter_of(sw: dict) -> Optional[dict]:
        f = sw.get("Filter")
        if isinstance(f, dict) and "$ref" in f:
            f = ids.get(f["$ref"])
        return f if isinstance(f, dict) else None

    for n in _walk(sequence):
        t = _short(n.get("$type"))
        vals = _items(n)

        # PARENT
        if vals and "$id" in n:
            for it in vals:
                if isinstance(it, dict) and "Parent" in it:
                    ref = (it.get("Parent") or {}).get("$ref")
                    if ref != n["$id"]:
                        problems.append(f"PARENT: {_short(it.get('$type'))} $id {it.get('$id')} "
                                        f"sits in {t} $id {n['$id']} but its Parent is $id {ref}")

        # SMART
        if t == "SmartExposure":
            shape = [_short(v.get("$type")) for v in vals if isinstance(v, dict)]
            if shape != ["SwitchFilter", "TakeExposure"]:
                problems.append(f"SMART: SmartExposure $id {n.get('$id')} holds {shape}; "
                                f"N.I.N.A needs exactly [SwitchFilter, TakeExposure]")
            conds = [_short(c.get("$type")) for c in _values(n, "Conditions") if isinstance(c, dict)]
            if not conds or conds[0] != "LoopCondition":
                problems.append(f"SMART: SmartExposure $id {n.get('$id')} first Condition is "
                                f"{conds[0] if conds else 'missing'}, not LoopCondition")
            trigs = [_short(c.get("$type")) for c in _values(n, "Triggers") if isinstance(c, dict)]
            if not trigs or trigs[0] != "DitherAfterExposures":
                problems.append(f"SMART: SmartExposure $id {n.get('$id')} first Trigger is "
                                f"{trigs[0] if trigs else 'missing'}, not DitherAfterExposures")

        # FILTER
        if t == "SwitchFilter":
            f = filter_of(n)
            if f is None or not f.get("_name"):
                problems.append(f"FILTER: SwitchFilter $id {n.get('$id')} names no filter")

        # CENTER
        if vals:
            prev = None
            for it in vals:
                if isinstance(it, dict):
                    st = _short(it.get("$type"))
                    if st in ("Center", "CenterAndRotate") and \
                            not (isinstance(prev, dict) and _short(prev.get("$type")) == "SwitchFilter"):
                        problems.append(f"CENTER: {st} $id {it.get('$id')} is not preceded by a "
                                        f"SwitchFilter; the plate solve would run through whatever "
                                        f"filter the previous block left")
                    prev = it

        # TARGET
        if t == "DeepSkyObjectContainer":
            tgt = n.get("Target") or {}
            if not tgt.get("TargetName") or not isinstance(tgt.get("InputCoordinates"), dict):
                problems.append(f"TARGET: DeepSkyObjectContainer $id {n.get('$id')} has no "
                                f"TargetName/InputCoordinates")

        # SCRIPT
        if check_scripts and t == "ExternalScript":
            prog = _script_program(str(n.get("Script") or ""))
            if not prog:
                problems.append(f"SCRIPT: ExternalScript $id {n.get('$id')} has no program")
            elif not os.path.exists(prog):
                problems.append(f"SCRIPT: ExternalScript $id {n.get('$id')} runs {prog}, "
                                f"which does not exist")

    # TYPES
    if template is not None:
        allowed = _types(template) | GENERATOR_TYPES | set(extra_types)
        for ty in sorted(_types(sequence) - allowed):
            problems.append(f"TYPES: {ty} is not in the template and not on the allow-list")

    return problems


def lint_file(path: Path, template_path: Optional[Path] = None,
              check_scripts: bool = False) -> list[str]:
    seq = json.load(open(path, encoding="utf-8-sig"))
    tpl = json.load(open(template_path, encoding="utf-8-sig")) if template_path else None
    return lint(seq, template=tpl, check_scripts=check_scripts)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Lint a N.I.N.A sequence for what N.I.N.A will refuse.")
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--template", type=Path, default=None,
                    help="template the sequence was generated from (enables the TYPES rule)")
    ap.add_argument("--scripts", action="store_true", help="check ExternalScript programs exist")
    a = ap.parse_args(argv)
    problems = lint_file(a.sequence, a.template, a.scripts)
    if problems:
        print("%s: %d problem(s)" % (a.sequence, len(problems)))
        for p in problems:
            print("  " + p)
        return 1
    print("%s: clean" % a.sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
