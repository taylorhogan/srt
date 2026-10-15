"""Tonight's plan, as the site sees it: one small file the scheduler writes and the live panel reads.

Asked for 2026-10-15: "at noon and the pre-image check, post to the website
the picture you send to Pushover for the plan for the night; if it is bad
weather and no imaging that will be obvious."

The scheduler writes local/night_plan_status.json at each decision (noon
check, pre-sunset check, a refusal at start) beside the picture it drew
(iris_astronomy/scratch/night_timeline.jpg: the timeline, or the "no
imaging tonight" card) and triggers a live render. scripts/live_skymap.py
reads it, adds a "plan" block to status.json and pushes the picture as
plan.jpg; the page shows the section while the plan is for TONIGHT and
hides it otherwise -- a plan older than MAX_AGE_H is yesterday's.

Pure file I/O; nothing here draws, decides or pushes.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Optional

MAX_AGE_H = 20.0          # noon today .. the following morning; never yesterday's


def write(path, verdict: str, check: str, image: Optional[str], title: str = "",
          summary: str = "", reason: str = "", when: Optional[datetime] = None) -> dict:
    """verdict: "imaging" | "skip"; check: "noon" | "sunset" | "start"."""
    doc = {"written": (when or datetime.now().astimezone()).isoformat(timespec="seconds"),
           "verdict": verdict, "check": check, "title": title, "summary": summary,
           "reason": reason, "image": str(image) if image else None}
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, str(path))
    return doc


def restamp(path, check: str, when: Optional[datetime] = None) -> Optional[dict]:
    """The same plan, confirmed again (the sunset check with nothing changed)."""
    doc = read(path, max_age_h=None)
    if not doc:
        return None
    return write(path, doc["verdict"], check, doc.get("image"), doc.get("title", ""),
                 doc.get("summary", ""), doc.get("reason", ""), when)


def read(path, max_age_h: Optional[float] = MAX_AGE_H, now: Optional[datetime] = None) -> Optional[dict]:
    """The plan if the file exists, parses, and is younger than *max_age_h*."""
    try:
        with open(str(path), encoding="utf-8") as fh:
            doc = json.load(fh)
        written = datetime.fromisoformat(doc["written"])
    except Exception:
        return None
    if max_age_h is not None:
        now = now or datetime.now().astimezone()
        if written.tzinfo is None:
            written = written.replace(tzinfo=now.tzinfo)
        if now - written > timedelta(hours=max_age_h) or written > now + timedelta(hours=1):
            return None
    return doc
