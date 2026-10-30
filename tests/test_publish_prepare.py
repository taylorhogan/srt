"""D2 of docs/AUTOMATION_PLAN.md (2026-10-30): a passed final render becomes the publish inputs.

The fallback article, the draft hash, the gate in astro_output.pending_articles
and publish_prepare's record are pure; prepare() runs here with a stubbed
handle_image. Nothing touches a site.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts import astro_output as ao
from scripts import publish_prepare as pp

FACTS = {"dso": "ic410", "name": "IC 410", "object_type": "nebula (SIMBAD HII)", "recipe": "SHO",
         "processing": ["denoised", "deconvolved (learned, luminance only)"],
         "filters": {"Ha": [{"exptime": 300.0, "used": 35, "hours": 2.9, "nights": ["2026-10-27"]}],
                     "S-II": [{"exptime": 300.0, "used": 25, "hours": 2.1, "nights": ["2026-10-28"]}]},
         "hours": 5.0, "nights": ["2026-10-27", "2026-10-28"], "meta_frames": "35 Ha + 25 S-II &times; 300 s",
         "image": "ic410_SHO_learned_final.jpg", "date": "2026-10-30"}


def test_fallback_writeup_fills_every_field_from_the_facts_alone():
    art = ao.fallback_writeup(FACTS, distances={})
    assert all(k in art for k in ao.ARTICLE_FIELDS) and art["source"] == "fallback"
    assert art["title"] == "IC410 - Emission Nebula" and art["type_text"] == "EMISSION NEBULA"
    assert art["distance_ly"] == 0 and art["distance_text"] == "NOT ON RECORD"
    assert art["time_travel"] == "" and art["time_travel_title"] == ""
    assert "35 Ha, 25 S-II" in art["paragraphs"][0] and "5.0 hours in all over two nights" in art["paragraphs"][0]
    assert "combined as SHO" in art["paragraphs"][1] and "measured facts alone" in art["paragraphs"][1]
    # a recorded distance fills the box
    art = ao.fallback_writeup(FACTS, distances={"ic410": 12000})
    assert art["distance_ly"] == 12000 and art["distance_text"] == "~12,000 LIGHT YEARS"
    assert art["time_travel"].startswith("The light in this picture left IC 410 about 12,000 light years ago")
    assert ao.distance_text(70_000_000) == "~70 MILLION LIGHT YEARS" and ao.distance_text(None) == ""


def test_fragment_omits_an_empty_light_travel_box():
    art = ao.fallback_writeup(FACTS, distances={})
    frag = ao.build_fragment(art, FACTS, "ic410.jpg", "2026-10-30")
    assert "time-travel-box" not in frag and 'data-distance-ly="0"' in frag and "DISTANCE: NOT ON RECORD" in frag
    art = ao.fallback_writeup(FACTS, distances={"ic410": 12000})
    assert "time-travel-box" in ao.build_fragment(art, FACTS, "ic410.jpg", "2026-10-30")


def test_draft_hash_changes_with_the_article_or_the_picture():
    art = ao.fallback_writeup(FACTS, distances={})
    h = ao.draft_hash(art, b"jpg")
    assert h == ao.draft_hash(dict(art, source="api"), b"jpg")             # the source is not the draft
    assert h != ao.draft_hash(dict(art, title="other"), b"jpg") and h != ao.draft_hash(art, b"jpg2")


def test_pending_articles_skips_a_draft_until_approved(tmp_path, monkeypatch):
    d = tmp_path / "ic410"
    d.mkdir()
    art = ao.fallback_writeup(FACTS, distances={})
    (d / "article.json").write_text(json.dumps(art))
    (d / "article_request.json").write_text(json.dumps(FACTS))
    (d / "draft.json").write_text(json.dumps({"status": "pending"}))
    published = []
    monkeypatch.setattr(ao, "publish_article", lambda *a, **k: published.append(a) or {"id": "ic410"})
    monkeypatch.setattr(ao, "SITE", tmp_path / "nosite")
    logs = []
    assert ao.pending_articles(tmp_path, push=False, log=logs.append) == []
    assert published == [] and any("awaiting `approve ic410`" in m for m in logs)


def test_queue_record_and_auto_publish_date():
    final = {"recipe": "SHO", "product": "learned", "sha256": "abc", "qc": {"pass": True, "reasons": []}}
    now = datetime(2026, 11, 6, 9, 0)
    rec = pp.queue_record("ic410", final, {"article_source": "fallback"}, Path("/f/ic410"),
                          Path("/f/ic410/x.jpg"), "deadbeef", now, 3)
    assert rec["status"] == "pending" and rec["auto_publish_on"] == "2026-11-09" and rec["recipe"] == "SHO"
    rec = pp.queue_record("ic410", final, {"site": {"id": "ic410", "match": "id"}}, Path("/f"), Path("/f/x"), None, now, 3)
    assert rec["status"] == "on_site" and rec["auto_publish_on"] is None
    assert pp.product_name("IC 410", "SHO", "learned") == "ic410_SHO_learned_final.jpg"


def test_prepare_copies_the_final_drafts_and_records(tmp_path, monkeypatch):
    fdir = tmp_path / "Iris" / "ic410" / "final"
    fdir.mkdir(parents=True)
    (fdir / "ic410_SHO_final.jpg").write_bytes(b"picture")
    (fdir / "final.json").write_text(json.dumps({"recipe": "SHO", "product": "learned", "file": "ic410_SHO_final.jpg",
                                                 "sha256": "abc", "qc": {"pass": True, "reasons": []}}))
    folder = tmp_path / "astro_output"
    monkeypatch.setattr(pp, "QUEUE_DIR", tmp_path / "queue")
    seen = []

    def handle(image, folder_, publish, push, log, draft_only):
        seen.append((image.name, publish, push, draft_only))
        out = Path(folder_) / "ic410"
        out.mkdir()
        dest = out / image.name
        image.rename(dest)
        (out / "article.json").write_text(json.dumps(ao.fallback_writeup(FACTS, distances={})))
        return {"dso": "ic410", "folder": str(out), "image": str(dest), "article": "draft",
                "article_source": "fallback", "site": None}
    rec = pp.prepare("IC 410", fdir / "final.json", folder, handle=handle, now=datetime(2026, 11, 6, 9, 0), log=lambda m: None)
    assert seen == [("ic410_SHO_learned_final.jpg", True, False, True)]
    assert rec["status"] == "pending" and rec["article_source"] == "fallback" and rec["auto_publish_on"] == "2026-11-09"
    gate = json.loads((folder / "ic410" / "draft.json").read_text())
    assert gate["status"] == "pending" and gate["draft_hash"] == rec["draft_hash"]
    assert json.loads((tmp_path / "queue" / "ic410.json").read_text())["draft_hash"] == rec["draft_hash"]
    # a failed QC prepares nothing unless forced
    (fdir / "final.json").write_text(json.dumps({"recipe": "SHO", "product": "learned", "file": "ic410_SHO_final.jpg",
                                                 "qc": {"pass": False, "reasons": ["sky at 0.3% grey"]}}))
    rec = pp.prepare("ic410", fdir / "final.json", folder, handle=handle, log=lambda m: None)
    assert rec["status"] == "qc_failed" and len(seen) == 1
