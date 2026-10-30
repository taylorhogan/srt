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
from datetime import datetime, timezone
from typing import Optional

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


def build(registry: dict, rows: list, convergence: dict, plan: Optional[dict], drafts: dict,
          decisions: dict, site_ids: set, flags: dict, now: Optional[datetime] = None) -> dict:
    """The board. Pure."""
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
    return build(registry, rows, conv, plan, drafts, decisions, site, flags, now)
