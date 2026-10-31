"""The target board: every target's state, in one file the site shows (ARCHITECTURE_PLAN phase 4, 2026-10-30).

    build(registry, rows, convergence, plan, drafts, decisions, site_ids, flags, now) -> dict   pure
    gather(root, cfg, now) -> the same, from the live inputs on the PC

/live/targets.json is written by scripts/live_skymap.py beside status.json
and pushed with it (live_push), so the Live tab can show the queue as the
machine sees it: the registry's state for each target (WISHED, QUEUED,
ACQUIRING, CONVERGED, RETIRED), the frames and the convergence per filter,
tonight's slot, the auto-stop verdict from the noon plan, the publish draft
and its decision, and the gallery entry when one exists. A reader of the
page during the shadow weeks sees what the flags would do without opening
the chat.

Every input is a reading the PC already has; nothing here decides.
"""
import math
import re
from datetime import datetime, timezone
from typing import Callable, Optional

STATES = ("ACQUIRING", "CONVERGED", "QUEUED", "WISHED", "RETIRED")   # display order
MAX_AGE_H = 20.0           # a noon plan older than this is yesterday's


def _squash(name) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _plan_tonight(plan: Optional[dict], now: datetime) -> dict:
    """{slots: {dso: 'HH:MM-HH:MM'}, auto_stop: {dso: [why]}, filters: {dso: ratio}, best, will_image}
    from the conductor's latest NIGHT_PLAN record, if it is tonight's."""
    out = {"slots": {}, "auto_stop": {}, "filters": {}, "best": None, "will_image": None, "check": None}
    rec = (plan or {}).get("noon") or (plan or {}).get("sunset") or {}
    if (plan or {}).get("sunset"):
        rec = plan["sunset"]
    p = rec.get("plan") or {}
    ts = rec.get("ts")
    if ts:
        try:
            when = datetime.fromisoformat(str(ts))
            if when.tzinfo is None:
                when = when.replace(tzinfo=now.tzinfo)
            if (now - when).total_seconds() > MAX_AGE_H * 3600:
                return out
        except ValueError:
            pass
    out["best"], out["will_image"], out["check"] = p.get("best"), p.get("will_image"), rec.get("check")
    for s in p.get("slots") or []:
        key = _squash(s.get("dso"))
        out["slots"][key] = "%s-%s" % (str(s.get("start", ""))[11:16], str(s.get("end", ""))[11:16])
        if s.get("filters"):
            out["filters"][key] = s["filters"]
    for v in p.get("auto_stop") or []:
        out["auto_stop"][_squash(v.get("dso"))] = list(v.get("why") or [])
    return out


def progress(advice: Optional[dict], curve_done: bool, nights: int, cap_factor: float = 0.0,
             goals: Optional[dict] = None) -> Optional[dict]:
    """How close a target is to the two things auto-stop waits for. Pure.

    *advice* is filter_advisor.advise()'s result (kind, have, need); the
    depth is the worst filter's sharp frames over its goal (the advisor's
    own arithmetic, so the bar fills exactly when its verdict flips), with
    that filter named. *curve_done* is the convergence verdict. *nights*
    imaged so far gives the pace, and the frames still short divided by it
    is the nights to go. *cap_factor* > 0 adds the cap in frames shot. None
    for a monitor or when there is no advice (no frames, no type).
    """
    if not advice or advice.get("action") == "monitor":
        return None
    kind = advice.get("kind")
    if goals is None:
        from fits_processing.filter_advisor import GOALS
        goals = GOALS.get(kind) or {}
    if not goals:
        return None
    have = advice.get("have") or {}
    sharp = {f: int((have.get(f) or {}).get("sharp") or 0) for f in goals}
    shares = {f: min(1.0, sharp[f] / float(goals[f])) for f in goals if goals[f]}
    limiting = min(shares, key=lambda f: (shares[f], f))
    need_frames = sum(max(int(goals[f]) - sharp[f], 0) for f in goals)
    shot = sum(int((h or {}).get("frames") or 0) for h in have.values())
    per_night = (shot / nights) if nights and shot else None
    if need_frames <= 0:
        nights_to_go = 0
    elif per_night:
        nights_to_go = int(math.ceil(need_frames / per_night))
    else:
        nights_to_go = None
    return {"pct": int(round(100 * shares[limiting])), "limiting": limiting,
            "sharp": sharp, "goals": {f: int(v) for f, v in goals.items()},
            "need_frames": need_frames, "shot": shot, "nights": int(nights or 0),
            "per_night": (round(per_night, 1) if per_night else None), "nights_to_go": nights_to_go,
            "cap": (int(round(cap_factor * sum(goals.values()))) if cap_factor and cap_factor > 0 else None),
            "curve": bool(curve_done), "kind": kind}


_SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[\\/]LIGHT[\\/]")


def nights_from_paths(paths) -> int:
    """Distinct session dates in frame paths like .../<dso>/cdk17/2026-10-27/LIGHT/x.fits. Pure."""
    return len({m.group(1) for p in paths for m in [_SESSION_RE.search(str(p))] if m})


def build(registry: dict, rows: list, convergence: dict, plan: Optional[dict], drafts: dict,
          decisions: dict, site_ids: set, flags: dict, now: Optional[datetime] = None,
          progress_fn: Optional[Callable[[str, dict], Optional[dict]]] = None) -> dict:
    """The board. Pure given *progress_fn* (dso, row) -> progress() or None."""
    now = now or datetime.now(timezone.utc).astimezone()
    tonight = _plan_tonight(plan, now)
    by_key = {_squash(r.get("dso")): r for r in rows or []}
    targets = []
    for dso, reg in (registry or {}).items():
        key = _squash(dso)
        row = by_key.get(key) or {}
        conv = (convergence or {}).get(key) or {}
        filters = {}
        for f, info in conv.items():
            if not isinstance(info, dict):
                continue
            filters[f] = {"frames": info.get("total_frames") or info.get("frame_count"),
                          "used": info.get("frame_count"),
                          "slope": info.get("tail_slope_pct"),
                          "rmse": info.get("final_rmse_pct"),
                          "updated": info.get("updated"),
                          "emission": (info.get("emission") or {}).get("level_adu_s")}
        draft = drafts.get(key)
        decision = decisions.get(key)
        draft_out = None
        if draft:
            d = str((decision or {}).get("decision") or "")
            same = (decision or {}).get("draft_hash") == draft.get("draft_hash")
            draft_out = {"hash": str(draft.get("draft_hash") or "")[:12], "recipe": draft.get("recipe"),
                         "written_by": "model" if draft.get("article_source") == "api" else "facts only",
                         "auto_publish_on": draft.get("auto_publish_on"),
                         "decision": (d if same and d else ("stale" if decision else "pending"))}
        targets.append({
            "dso": dso, "state": reg.get("state"), "priority": reg.get("priority"),
            "status": reg.get("legacy_status") or row.get("status"),
            "obj_type": row.get("obj_type"), "lightcurve": bool(row.get("lightcurve")),
            "held": row.get("status") == "held",
            "filter_plan": row.get("filter_plan"), "filter_plan_until": row.get("filter_plan_until"),
            "max_hours": row.get("max_hours"), "moon": row.get("moon"),
            "auto_stop_hold_until": row.get("auto_stop_hold_until"),
            "filters": filters,
            "frames": sum(int(v["frames"] or 0) for v in filters.values()),
            "tonight": tonight["slots"].get(key),
            "filters_tonight": tonight["filters"].get(key),
            "would_stop": tonight["auto_stop"].get(key),
            "draft": draft_out,
            "on_site": key in site_ids,
            "progress": (progress_fn(dso, row) if progress_fn and not row.get("lightcurve") else None),
        })
    order = {s: i for i, s in enumerate(STATES)}
    targets.sort(key=lambda t: (order.get(t["state"], 9), -(float(t["priority"] or 0)), t["dso"]))
    counts = {}
    for t in targets:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    return {"generated": now.isoformat(timespec="seconds"),
            "flags": dict(flags or {}),
            "tonight": {"best": tonight["best"], "will_image": tonight["will_image"], "check": tonight["check"]},
            "counts": counts, "targets": targets}


def gather(root, cfg: dict, now: Optional[datetime] = None) -> dict:
    """build() from the PC's live inputs; every source that fails is empty, never fatal."""
    import json
    import os
    from pathlib import Path
    root = Path(root)
    registry, rows, conv, plan, drafts, decisions, site = {}, [], {}, None, {}, {}, set()
    try:
        from iris import client
        registry = client.targets() or {}
        plan = client.get_plan()
    except Exception:  # noqa: BLE001
        pass
    if not registry:
        try:
            from iris.conductor.targets import derive_registry
            registry = derive_registry()
        except Exception:  # noqa: BLE001
            registry = {}
    try:
        rows = json.loads((root / cfg["location"]["instructions"]).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        rows = []
    try:
        conv = json.loads((root / (cfg.get("convergence", {}).get("file", "local/convergence.json"))).read_text())
    except Exception:  # noqa: BLE001
        conv = {}
    try:
        from control import publish_gate
        drafts = publish_gate.read_journal_drafts(str(root))
        decisions = publish_gate.load_decisions(os.path.join(str(root), publish_gate.DECISIONS_FILE))
    except Exception:  # noqa: BLE001
        pass
    try:
        from scripts import astro_output as ao
        html = (ao.SITE / "index.html").read_text(encoding="utf-8")
        site = {_squash(a["id"]) for a in ao.gallery_articles(html)} | \
               {_squash(a["title"].split("-")[0]) for a in ao.gallery_articles(html)}
    except Exception:  # noqa: BLE001
        site = set()
    flags = {"auto_stop": (cfg.get("planner") or {}).get("auto_stop", "shadow"),
             "filter_advisor": (cfg.get("nina") or {}).get("filter_advisor", "shadow"),
             "final_mode": (cfg.get("publish") or {}).get("final_mode", "shadow"),
             "gate": (cfg.get("publish") or {}).get("gate", "shadow")}

    def _progress(dso: str, row: dict) -> Optional[dict]:
        # The advisor's own tally (offline, as auto-stop reads it), the
        # convergence verdict and the nights imaged from the frame record.
        try:
            from control import auto_stop
            from fits_processing import convergence as conv_mod
            from fits_processing import filter_advisor
            advice = auto_stop.advise_offline(dso, row, root=str(root))
            if not advice:
                return None
            nights = 0
            try:
                image_dir = Path(cfg["nina"]["image_dir"])
                key = dso.lower().replace(" ", "")
                d = next((p for p in image_dir.iterdir()
                          if p.is_dir() and p.name.lower().replace(" ", "") == key), None)
                if d is not None and (d / "frame_stats.json").exists():
                    nights = nights_from_paths(r.get("path", "") for r in json.loads(
                        (d / "frame_stats.json").read_text(encoding="utf-8")) if isinstance(r, dict))
            except Exception:  # noqa: BLE001
                nights = 0
            return progress(advice, conv_mod.is_dso_done(dso), nights, auto_stop.cap_factor(cfg))
        except Exception:  # noqa: BLE001
            return None
    return build(registry, rows, conv, plan, drafts, decisions, site, flags, now, progress_fn=_progress)
