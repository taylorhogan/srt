"""Registry audit -- the no-telescope half of Phase 1's exit criterion.

The phase gate reads: "the registry matches the queue+convergence answers for
every DSO". That is a comparison between things already on disk, so it is
checkable at noon, every day, without waiting for a night. This module holds
the pure logic; apps/registry_audit.py runs it against the live system and
shadow_report.py appends its one-line verdict to the morning post.

Three kinds of output, deliberately separated:

  * MISMATCHES -- the registry's state for a DSO disagrees with an
    independently recomputed expectation from the same legacy answers. Any
    entry here is a derivation bug (or a stale running process) and the gate
    does not tick while one exists.
  * FINDINGS -- the legacy sources disagree with EACH OTHER in ways the
    registry model makes visible (a waiting target that is measurably done,
    a completed one that never converged, duplicate queue rows...). These are
    data problems for the operator, not derivation bugs; they do not block
    the gate but they are the split brain being caught in the act.
  * COUNTS -- how many targets sit in each lifecycle state, so the morning
    report shows the registry breathing.

Everything here is a pure function of injected data -- no config, no astropy,
no network -- so CI's bare pytest runner covers the whole policy.
"""
from typing import Callable, Iterable, Optional

# The derivation rules of iris.conductor.targets.derive_registry, restated
# independently. If the two implementations drift, the audit reports the
# drift as mismatches -- which is the point: two authors of the same rule.
STATES = ("WISHED", "QUEUED", "ACQUIRING", "CONVERGED", "RETIRED")

NARROWBAND = {"ha", "o-iii", "s-ii"}


def is_resolved(row: dict) -> bool:
    """Coordinates resolvable: an explicit position, or a real (non-zero)
    above_horizon computed from a successful name lookup.

    '0' is what calc_and_store_hours_above_horizon writes when the resolver
    FAILED, so it is evidence of unresolvability, not of resolution -- a bare
    truthiness test reads it backwards.
    """
    if row.get("ra_deg") is not None and row.get("dec_deg") is not None:
        return True
    ah = str(row.get("above_horizon") or "").strip()
    return ah not in ("", "0", "None")


def expected_state(row: dict, converged: bool, has_frames: bool) -> str:
    status = str(row.get("status") or "").strip().lower()
    if status == "completed":
        return "RETIRED"
    if converged:
        return "CONVERGED"
    if has_frames:
        return "ACQUIRING"
    if is_resolved(row):
        return "QUEUED"
    return "WISHED"


def audit_registry(queue_rows: Iterable[dict], registry: dict,
                   converged_of: Callable[[str], bool],
                   has_frames_of: Callable[[str], bool]) -> dict:
    """Compare a registry snapshot against re-derived expectations.

    registry: {dso: {"state": ...}} as served by /v1/targets (or derived
    in-process). Returns {"mismatches", "missing", "extra", "counts"}.
    """
    mismatches, missing = [], []
    # Last row wins per name, mirroring derive_registry's dict overwrite:
    # duplicate rows are a DATA problem (find_inconsistencies flags them) and
    # must not double-report as derivation mismatches here.
    last_rows: dict = {}
    for row in queue_rows:
        dso = str(row.get("dso") or "").strip()
        if dso:
            last_rows[dso] = row
    for dso, row in last_rows.items():
        want = expected_state(row, converged_of(dso), has_frames_of(dso))
        got = (registry.get(dso) or {}).get("state")
        if got is None:
            missing.append(dso)
        elif got != want:
            mismatches.append((dso, got, want))
    extra = sorted(set(registry) - set(last_rows))
    counts: dict = {}
    for info in registry.values():
        s = info.get("state", "?")
        counts[s] = counts.get(s, 0) + 1
    return {"mismatches": mismatches, "missing": missing,
            "extra": extra, "counts": counts}


def find_inconsistencies(queue_rows: list, convergence: dict,
                         converged_of: Callable[[str], bool]) -> list:
    """The split brain, caught in the act: legacy sources disagreeing with
    each other. Strings, ready to print; empty list = quiet conscience."""
    out = []
    by_name: dict = {}
    for row in queue_rows:
        dso = str(row.get("dso") or "").strip()
        if dso:
            by_name.setdefault(dso.lower().replace(" ", ""), []).append(row)

    for key, rows in sorted(by_name.items()):
        if len(rows) > 1:
            out.append("duplicate queue rows for '%s' (%d)" % (key, len(rows)))
        row = rows[0]
        status = str(row.get("status") or "").strip().lower()
        entries = convergence.get(key) or {}
        if status in ("waiting", "in process") and converged_of(key):
            out.append("AUTO-STOP candidate: '%s' is %s in the queue but "
                       "measurably converged -- the legacy path would keep "
                       "imaging it" % (key, status))
        if (status == "completed" and entries
                and any(i.get("calibrated") for i in entries.values())
                and not converged_of(key)):
            out.append("retired early: '%s' is completed in the queue but its "
                       "calibrated convergence record says not done" % key)
        if (status != "completed" and entries
                and {str(f).lower() for f in entries} & NARROWBAND
                and str(row.get("obj_type") or "").strip().lower() != "nebula"):
            out.append("narrowband history but no obj_type on '%s' -- SIMBAD "
                       "misclassification would plan LRGB (set obj_type: "
                       "nebula on the instruction)" % key)
        if status != "completed" and not is_resolved(row):
            out.append("unresolvable: '%s' has no stored position and no "
                       "successful lookup -- it can never be scheduled" % key)

    for key in sorted(convergence):
        if key not in by_name:
            out.append("convergence record for '%s' has no queue row (imaged "
                       "under a name no longer queued?)" % key)
    return out


def render(audit: dict, findings: list, source: str) -> str:
    """ASCII-only report body (cp1252 consoles have killed reports before)."""
    lines = ["Target registry audit (%s)" % source]
    counts = ", ".join("%s %d" % (s, audit["counts"].get(s, 0))
                       for s in STATES if audit["counts"].get(s))
    lines.append("  %d targets: %s" % (sum(audit["counts"].values()), counts))
    if audit["mismatches"]:
        lines.append("  MISMATCHES (%d) -- registry vs re-derived expectation:"
                     % len(audit["mismatches"]))
        for dso, got, want in audit["mismatches"]:
            lines.append("    %s: registry says %s, expected %s" % (dso, got, want))
    if audit["missing"]:
        lines.append("  queued but absent from registry: %s"
                     % ", ".join(audit["missing"]))
    if audit["extra"]:
        lines.append("  in registry but not queued: %s" % ", ".join(audit["extra"]))
    if findings:
        lines.append("  Findings (legacy sources disagreeing, %d):" % len(findings))
        for f in findings:
            lines.append("    - %s" % f)
    clean = not (audit["mismatches"] or audit["missing"] or audit["extra"])
    lines.append("  Registry verdict: %s"
                 % ("CLEAN" if clean else "MISMATCHED"))
    return "\n".join(lines)
