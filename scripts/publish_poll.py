"""publish_poll.py -- act on the owner's publish decisions (D4 of docs/AUTOMATION_PLAN.md).

    python scripts/publish_poll.py              # once
    python scripts/publish_poll.py --dry-run    # say what would happen, touch nothing

Spark cron (the owner adds it; docs/tasks/README.md):
    */15 8-22 * * * /home/taylor/Documents/srt/scripts/publish_poll.py >> /tmp/publish_poll.log 2>&1

For every pending record in local/publish_queue (scripts/publish_prepare) it
reads the PC's local/publish_decisions.json over ssh (the owner's `approve`
/ `reject`, by draft hash) and asks control/publish_gate.decide what to do:

  publish   gate live: astro_output.publish_article(replace=True) -- the
            picture to images/<id>.jpg, the entry between its anchors (a
            re-render replaces, never duplicates), commit, push to GitHub;
            then on the PC `git pull --ff-only` of its site checkout and
            scripts/site_push.py --yes (the host pulls now, the subscribers
            are told; the web-host key lives on the PC); then PUBLISH_DONE
            journaled through the PC with --retire, which sets the queue
            row to completed. gate shadow: one chat line "would publish now",
            said once per draft.
  rejected  the record and the folder's draft.json marked rejected, once.
  stale     the decision names an older draft; the fresh card is already
            out (publish_prepare posts one per new draft). Waits.
  wait      nothing.

Never raises past one record: a failure is logged and the next record runs.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

QUEUE_DIR = Path(_root) / "local" / "publish_queue"
LOG = Path("/tmp/publish_poll.log")


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def pending_records(queue_dir: Optional[Path] = None) -> list:
    """The queue records awaiting action, by name."""
    out = []
    for p in sorted(Path(queue_dir or QUEUE_DIR).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        rec["_path"] = str(p)
        out.append(rec)
    return out


def plan_actions(records: list, decisions: dict, today: date, gate: str) -> list:
    """[(record, verdict)] for the records that need anything done. Pure."""
    from control import publish_gate
    out = []
    for rec in records:
        if rec.get("status") != "pending":
            continue
        v = publish_gate.decide(rec, decisions.get(rec.get("dso")), today, gate == "live")
        if v["action"] in ("publish", "rejected") or (v["action"] == "stale" and not rec.get("stale_noted")):
            out.append((rec, v))
    return out


def read_pc_decisions() -> Optional[dict]:
    """The PC's local/publish_decisions.json; None when the PC did not answer, {} when absent."""
    try:
        import morning_trigger as mt
    except Exception:  # noqa: BLE001
        return None
    out = mt._ps("if (Test-Path '%s\\local\\publish_decisions.json') { Get-Content -Raw '%s\\local\\publish_decisions.json' } else { '{}' }"
                 % (mt.SRT, mt.SRT), timeout=40)
    if out is None:
        return None
    text = out.strip()
    for cand in (text, text.splitlines()[-1] if text else ""):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else {}
        except ValueError:
            continue
    return {}


def announce_on_pc(log: Callable[[str], None] = log) -> bool:
    """`git pull --ff-only` of the PC's site checkout, then site_push.py --yes:
    the host pulls now and the subscribers are told. The PC holds the web-host
    key; the Spark does not (reference_site_publishing)."""
    try:
        import morning_trigger as mt
    except Exception:  # noqa: BLE001
        return False
    site = r"C:\Users\iriso\Documents\development\taylorhogan.github.io"
    script = ("git -C '%s' pull -q --ff-only origin main; Set-Location '%s'; "
              "& '.venv\\Scripts\\python.exe' 'scripts\\site_push.py' --yes" % (site, mt.SRT))
    out = mt._ps(script, timeout=400)
    for line in (out or "").splitlines()[-8:]:
        log(f"  | {line}")
    return out is not None


def write_record(rec: dict) -> None:
    p = Path(rec.pop("_path", "") or QUEUE_DIR / f"{rec['dso']}.json")
    p.write_text(json.dumps(rec, indent=1))
    rec["_path"] = str(p)


def mark_draft(folder: str, status: str, extra: Optional[dict] = None) -> None:
    p = Path(folder) / "draft.json"
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        d = {}
    d.update({"status": status, **(extra or {})})
    p.write_text(json.dumps(d, indent=1))


def publish_one(rec: dict, log: Callable[[str], None] = log, dry_run: bool = False) -> dict:
    """The live publish of one approved record. -> what was done."""
    from scripts import astro_output as ao
    folder = Path(rec["folder"])
    art = json.loads((folder / "article.json").read_text())
    facts = json.loads((folder / "article_request.json").read_text())
    image = Path(rec["image"])
    if dry_run:
        log(f"{rec['dso']}: would publish '{art.get('title')}' from {image.name}")
        return {"dry_run": True}
    res = ao.publish_article(art, facts, image, push=True, log=log, replace=True)
    (folder / "article_published.json").write_text(json.dumps(res, indent=1))
    mark_draft(str(folder), "published", {"published": res})
    announced = announce_on_pc(log)
    try:
        from scripts import spark_morning_render as smr
        noted = smr.post_pc_event("PUBLISH_DONE", {
            "dso": rec["dso"], "id": res.get("id"), "commit": res.get("commit"), "pushed": res.get("pushed"),
            "replaced": res.get("replaced"), "draft_hash": rec.get("draft_hash"), "announced": announced,
            "title": art.get("title")}, kind="note", retire=True)
    except Exception as exc:  # noqa: BLE001
        log(f"  PUBLISH_DONE failed: {exc}")
        noted = False
    try:
        from utils.webchat_client import post_to_webchat
        post_to_webchat(f"Published: {art.get('title')} (#{res.get('id')}, site {res.get('commit')}"
                        + (", replaced the earlier entry" if res.get("replaced") else "") + ")"
                        + (" -- subscribers told." if announced else " -- announce not confirmed; `site_push.py --yes` on the PC.")
                        + (" The target is retired." if noted else ""))
    except Exception as exc:  # noqa: BLE001
        log(f"  chat line failed: {exc}")
    return {"site": res, "announced": announced, "noted": noted}


def run(today: Optional[date] = None, dry_run: bool = False, log: Callable[[str], None] = log) -> int:
    from configs import config
    from control import publish_gate
    gate = publish_gate.gate_mode(config.data())
    if gate == "off":
        return 0
    records = pending_records()
    if not any(r.get("status") == "pending" for r in records):
        return 0
    decisions = read_pc_decisions()
    if decisions is None:
        log("PC unreachable; decisions unknown -- nothing done")
        return 1
    today = today or date.today()
    rc = 0
    for rec, v in plan_actions(records, decisions, today, gate):
        dso = rec.get("dso")
        try:
            if v["action"] == "publish":
                if gate == "live":
                    log(f"{dso}: publishing ({v['why']})")
                    out = publish_one(rec, log, dry_run)
                    if not dry_run:
                        rec.update(status="published", published_at=datetime.now().isoformat(timespec="minutes"),
                                   site=out.get("site"), announced=out.get("announced"))
                        write_record(rec)
                elif not rec.get("shadow_noted"):
                    log(f"{dso}: gate {gate}: would publish now ({v['why']})")
                    if not dry_run:
                        try:
                            from utils.webchat_client import post_to_webchat
                            post_to_webchat(f"{dso}: approved; with publish.gate = live this would publish now. Nothing done.")
                        except Exception:  # noqa: BLE001
                            pass
                        rec["shadow_noted"] = True
                        write_record(rec)
            elif v["action"] == "rejected":
                log(f"{dso}: {v['why']}")
                if not dry_run:
                    rec.update(status="rejected", rejected=v["why"])
                    write_record(rec)
                    mark_draft(rec.get("folder", ""), "rejected", {"why": v["why"]})
            elif v["action"] == "stale":
                log(f"{dso}: {v['why']}; the fresh card is out, waiting")
                if not dry_run:
                    rec["stale_noted"] = True
                    write_record(rec)
        except Exception as exc:  # noqa: BLE001
            log(f"{dso}: failed: {exc!r}")
            rc = 1
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
