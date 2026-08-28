"""The shadow Target registry — the DSO lifecycle, derived not yet owned.

Phase 1 derives each target's state from the two legacy sources that today
hold the split brain (my_instructions.json's status and convergence.json's
verdict), plus the presence of accumulated frames. In Phase 4 this derivation
inverts: the registry becomes authoritative and those files retire.

Derivation rules (deliberately simple, and shadow_report checks them against
the legacy answers):
    status == completed                -> RETIRED
    convergence says done              -> CONVERGED     (the auto-stop fact)
    frames cached for the dso          -> ACQUIRING
    coordinates resolved / plan set    -> QUEUED
    otherwise                          -> WISHED
"""
import json
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_queue():
    from configs import config
    path = _repo_root() / config.data()["location"]["instructions"]
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        _logger.warning("targets: could not read %s", path, exc_info=True)
        return []


def _converged_lookup():
    try:
        from fits_processing.convergence import is_dso_done
        return is_dso_done
    except Exception:
        _logger.warning("targets: convergence unavailable", exc_info=True)
        return lambda name: False


def _has_frames(dso: str) -> bool:
    try:
        from configs import config
        d = Path(config.data()["nina"]["image_dir"]) / dso
        return (d / "frame_stats.json").exists()
    except Exception:
        return False


def derive_registry() -> dict:
    """{dso: {state, priority, filters, ...}} — the shadow Target registry."""
    is_done = _converged_lookup()
    out = {}
    for row in _load_queue():
        dso = str(row.get("dso") or "").strip()
        if not dso:
            continue
        status = str(row.get("status") or "").strip().lower()
        resolved = row.get("ra_deg") is not None or bool(row.get("above_horizon"))
        if status == "completed":
            state = "RETIRED"
        elif is_done(dso):
            state = "CONVERGED"
        elif _has_frames(dso):
            state = "ACQUIRING"
        elif resolved:
            state = "QUEUED"
        else:
            state = "WISHED"
        out[dso] = {
            "state": state,
            "priority": row.get("priority"),
            "requestor": row.get("requestor"),
            "recipe": row.get("recipe") or None,
            "filter_plan": row.get("filter_plan"),
            "legacy_status": status,
        }
    return out
