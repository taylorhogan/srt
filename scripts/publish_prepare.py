"""publish_prepare.py -- a passed final render becomes the publish inputs (D2).

    python scripts/publish_prepare.py ic410            # from <image_dir>/Iris/ic410/final/final.json
    python scripts/publish_prepare.py ic410 --force    # even when the QC failed

docs/AUTOMATION_PLAN.md D2 (2026-10-30). Runs on the Spark after
spark_morning_render's final render passes render_qc. It:

1. copies the final picture into ~/Desktop/astro_output as
   <dso>_<recipe>_<product>_final.jpg (the name astro_output reads the
   target, the recipe and the processing from);
2. runs astro_output.handle_image(draft_only=True): the folder, the subs list
   and stats picture, the site check (a target already on the site stops
   here), article_request.json, and article.json -- the model's write-up when
   a credential is present, else astro_output.fallback_writeup, facts only;
3. hashes the draft (article + picture) and records it twice: draft.json in
   the folder, which gates astro_output's own publish path until approved,
   and local/publish_queue/<dso>.json for the card and the poll (D3, D4):
   {dso, recipe, product, folder, image, draft_hash, article_source,
    posted_at, auto_publish_on, status: pending | on_site};
4. (D3) posts the card to the web chat (control/publish_gate.card_text) and
   journals DRAFT_READY through the PC, so `approve <dso>` knows the hash.

Nothing is published here, whatever the config says.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

QUEUE_DIR = Path(_root) / "local" / "publish_queue"
DEFAULT_AUTO_DAYS = 3


def auto_publish_on(posted: datetime, days: int) -> str:
    """The date a pending draft publishes itself if nobody answers (ISO date)."""
    return (posted + timedelta(days=max(0, int(days)))).date().isoformat()


def product_name(dso: str, recipe: str, product: str) -> str:
    """The name astro_output parses: target first, the recipe, the processing word."""
    return f"{dso.lower().replace(' ', '')}_{recipe}_{product}_final.jpg"


def queue_record(dso: str, final: dict, result: dict, folder: Path, image: Path,
                 dhash: Optional[str], now: datetime, days: int) -> dict:
    """The publish_queue entry. Pure."""
    on_site = (result or {}).get("site")
    return {"dso": dso, "recipe": final.get("recipe"), "product": final.get("product"),
            "final_sha256": final.get("sha256"), "qc": final.get("qc"),
            "folder": str(folder), "image": str(image),
            "article_source": (result or {}).get("article_source"),
            "draft_hash": dhash,
            "posted_at": now.isoformat(timespec="minutes"),
            "auto_publish_on": auto_publish_on(now, days) if dhash else None,
            "status": "on_site" if on_site else ("pending" if dhash else "no_draft"),
            "site": on_site}


def prepare(dso: str, final_json: Optional[Path] = None, folder: Optional[Path] = None,
            force: bool = False, days: int = DEFAULT_AUTO_DAYS, now: Optional[datetime] = None,
            handle=None, log: Callable[[str], None] = print, announce: bool = True) -> dict:
    """Steps 1-3 for one target. -> the queue record (also written)."""
    from scripts import astro_output as ao
    handle = handle or ao.handle_image
    folder = Path(folder or ao.DEFAULT_FOLDER)
    now = now or datetime.now()
    key = dso.lower().replace(" ", "")
    if final_json:
        fj = Path(final_json)
    else:
        from stacking import stacker          # numpy; only when the record must be found
        fj = stacker.results_dir(key) / "final" / "final.json"
    final = json.loads(Path(fj).read_text())
    qc = final.get("qc") or {}
    if not qc.get("pass") and not force:
        log(f"{key}: the final render did not pass QC ({'; '.join(qc.get('reasons') or ['no verdict'])}); "
            "nothing prepared (--force overrides)")
        rec = {"dso": key, "status": "qc_failed", "qc": qc, "posted_at": now.isoformat(timespec="minutes")}
        _write_queue(rec)
        return rec
    src = Path(fj).parent / final["file"]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / product_name(key, final.get("recipe") or "", final.get("product") or "learned")
    shutil.copy2(src, dest)
    log(f"{key}: {src.name} -> {dest}")
    result = handle(dest, folder, publish=True, push=False, log=log, draft_only=True)
    out_dir = Path(result.get("folder") or folder / key)
    dhash = None
    if result.get("article") == "draft":
        art = json.loads((out_dir / "article.json").read_text())
        image = Path(result["image"])
        dhash = ao.draft_hash(art, image.read_bytes())
        (out_dir / "draft.json").write_text(json.dumps(
            {"dso": key, "draft_hash": dhash, "status": "pending",
             "posted_at": now.isoformat(timespec="minutes"),
             "auto_publish_on": auto_publish_on(now, days)}, indent=1))
    rec = queue_record(key, final, result, out_dir, Path(result.get("image") or dest), dhash, now, days)
    _write_queue(rec)
    log(f"{key}: {rec['status']}" + (f" (draft {dhash[:12]}, {rec['article_source']}; auto-publishes "
                                      f"{rec['auto_publish_on']} once the gate is live)" if dhash else ""))
    if dhash and announce:
        announce_draft(rec, out_dir, log)
    return rec


def card_image(image: Path, out_dir: Path, max_px: int = 1600) -> Path:
    """A copy of the picture small enough for the chat card."""
    from PIL import Image
    im = Image.open(image).convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    p = Path(out_dir) / f"{image.stem}_card.jpg"
    im.save(p, quality=88, optimize=True)
    return p


def announce_draft(rec: dict, out_dir: Path, log: Callable[[str], None] = print) -> dict:
    """D3: the card in the chat (utils/webchat_client, over the Tailnet) and the
    DRAFT_READY note in the conductor's journal through the PC
    (spark_morning_render.post_pc_event), which is how `approve` learns the
    hash. Neither failing stops the other; the record is already on disk."""
    out: dict = {"card": False, "note": False}
    try:
        from control import publish_gate
        from configs import config
        gate = publish_gate.gate_mode(config.data())
    except Exception:  # noqa: BLE001
        from control import publish_gate
        gate = publish_gate.DEFAULT_GATE
    if gate == "off":
        log("  gate off: no card, no note")
        return out
    try:
        art = json.loads((Path(out_dir) / "article.json").read_text())
        facts = json.loads((Path(out_dir) / "article_request.json").read_text())
    except (OSError, ValueError) as exc:
        log(f"  card: cannot read the draft ({exc})")
        return out
    text = publish_gate.card_text(rec, art, facts, gate)
    try:
        from utils.webchat_client import post_to_webchat
        pic = card_image(Path(rec["image"]), out_dir)
        out["card"] = bool(post_to_webchat(text, pic))
        log(f"  card: {'posted' if out['card'] else 'NOT posted'}")
    except Exception as exc:  # noqa: BLE001
        log(f"  card failed: {exc}")
    try:
        from scripts import spark_morning_render as smr
        out["note"] = smr.post_pc_event(publish_gate.DRAFT_EVENT, {
            "dso": rec["dso"], "recipe": rec.get("recipe"), "product": rec.get("product"),
            "draft_hash": rec.get("draft_hash"), "article_source": rec.get("article_source"),
            "auto_publish_on": rec.get("auto_publish_on"), "title": art.get("title"),
            "folder": rec.get("folder"), "image": rec.get("image"), "gate": gate})
        log(f"  DRAFT_READY {'journaled' if out['note'] else 'NOT journaled (PC unreachable)'}")
    except Exception as exc:  # noqa: BLE001
        log(f"  DRAFT_READY failed: {exc}")
    return out


def _write_queue(rec: dict) -> Path:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    p = QUEUE_DIR / f"{rec['dso']}.json"
    p.write_text(json.dumps(rec, indent=1))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dso")
    ap.add_argument("--final-json", default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--force", action="store_true", help="prepare even when the QC failed")
    a = ap.parse_args(argv)
    days = DEFAULT_AUTO_DAYS
    try:
        from configs import config
        days = int((config.data().get("publish") or {}).get("auto_approve_days", DEFAULT_AUTO_DAYS))
    except Exception:  # noqa: BLE001
        pass
    rec = prepare(a.dso, a.final_json, a.folder, a.force, days)
    return 0 if rec.get("status") in ("pending", "on_site") else 1


if __name__ == "__main__":
    sys.exit(main())
