"""D4 of docs/AUTOMATION_PLAN.md (2026-10-30): the Spark acts on the owner's decisions.

Anchors in the site markup (a re-render replaces, a hand-made entry is left
alone), the poll's plan over records and decisions, and one live publish with
the site, the ssh and the chat stubbed. Stdlib; nothing touches a site.
"""
import json
import sys
import types
from datetime import date
from pathlib import Path

from scripts import astro_output as ao
from scripts import publish_poll as pp
from tests.test_astro_output import GALLERY

FACTS = {"dso": "ic410", "name": "IC 410", "processing": [], "meta_frames": "35 Ha &times; 300 s", "hours": 2.9,
         "nights": ["2026-10-27"]}
ART = {"title": "IC410 - Emission Nebula", "distance_ly": 0, "distance_text": "NOT ON RECORD",
       "type_text": "EMISSION NEBULA", "alt": "IC 410", "paragraphs": ["One.", "Two."],
       "time_travel_title": "", "time_travel": ""}


def test_anchors_let_a_re_render_replace_its_own_entry():
    frag = ao.build_fragment(ART, FACTS, "ic410.jpg", "2026-11-07")
    assert frag.startswith("        <!-- dso:ic410:start -->\n") and frag.rstrip().endswith("<!-- dso:ic410:end -->")
    html = ao.insert_fragment(GALLERY, frag)
    assert [a["id"] for a in ao.gallery_articles(html)] == ["ngc2146", "stephans-quintet", "ic410"]
    frag2 = ao.build_fragment(dict(ART, paragraphs=["Newer.", "Words."]), FACTS, "ic410.jpg", "2026-11-09")
    html2 = ao.insert_fragment(html, frag2, replace_id="ic410")
    assert [a["id"] for a in ao.gallery_articles(html2)] == ["ngc2146", "stephans-quintet", "ic410"]
    assert "Newer." in html2 and "One." not in html2 and html2.count("dso:ic410:start") == 1
    assert html2.index('id="notes-panel"') > html2.index("Newer.")
    # a hand-made entry (no anchors) is not replaced: the fragment is appended instead
    html3 = ao.insert_fragment(GALLERY, ao.build_fragment(dict(ART, title="NGC2146 - x"), dict(FACTS, dso="ngc2146"),
                                                          "ngc2146.jpg", "2026-11-09"), replace_id="ngc2146")
    assert [a["id"] for a in ao.gallery_articles(html3)].count("ngc2146") == 2
    assert ao.anchored_span(GALLERY, "ngc2146") is None


def _rec(dso="ic410", status="pending", h="a" * 64, **kw):
    return {"dso": dso, "status": status, "draft_hash": h, "auto_publish_on": "2026-11-09", "folder": f"/f/{dso}",
            "image": f"/f/{dso}/x.jpg", **kw}


def test_plan_actions_picks_what_needs_doing():
    recs = [_rec(), _rec("m1", h="c" * 64), _rec("m31", status="published"), _rec("m76", h="e" * 64),
            _rec("m42", h="f" * 64, stale_noted=True)]
    decisions = {"ic410": {"decision": "approve", "draft_hash": "a" * 64, "by": "t", "at": ""},
                 "m1": {"decision": "reject", "draft_hash": "c" * 64, "by": "t", "why": "red"},
                 "m42": {"decision": "approve", "draft_hash": "old"}}
    plan = pp.plan_actions(recs, decisions, date(2026, 11, 7), "shadow")
    assert [(r["dso"], v["action"]) for r, v in plan] == [("ic410", "publish"), ("m1", "rejected")]
    # the timer, live only; a stale decision is noted once
    plan = pp.plan_actions([_rec("m76", h="e" * 64)], {}, date(2026, 11, 9), "live")
    assert [(r["dso"], v["action"]) for r, v in plan] == [("m76", "publish")]
    assert pp.plan_actions([_rec("m76", h="e" * 64)], {}, date(2026, 11, 9), "shadow") == []
    plan = pp.plan_actions([_rec("m42", h="f" * 64)], {"m42": {"decision": "approve", "draft_hash": "old"}},
                           date(2026, 11, 7), "live")
    assert [(r["dso"], v["action"]) for r, v in plan] == [("m42", "stale")]


def test_run_publishes_an_approved_draft_when_live(tmp_path, monkeypatch):
    folder = tmp_path / "ic410"
    folder.mkdir()
    (folder / "article.json").write_text(json.dumps(ART))
    (folder / "article_request.json").write_text(json.dumps(FACTS))
    (folder / "x.jpg").write_bytes(b"jpg")
    (folder / "draft.json").write_text(json.dumps({"status": "pending"}))
    q = tmp_path / "queue"
    q.mkdir()
    rec = _rec(folder=str(folder), image=str(folder / "x.jpg"))
    (q / "ic410.json").write_text(json.dumps(rec))
    monkeypatch.setattr(pp, "QUEUE_DIR", q)
    monkeypatch.setattr(pp, "read_pc_decisions",
                        lambda: {"ic410": {"decision": "approve", "draft_hash": "a" * 64, "by": "taylor", "at": "t"}})
    published, events, chat = [], [], []
    monkeypatch.setattr(ao, "publish_article", lambda art, facts, image, push, log, replace: published.append(
        (art["title"], image.name, push, replace)) or {"id": "ic410", "commit": "abc1234", "pushed": True, "replaced": False})
    monkeypatch.setattr(pp, "announce_on_pc", lambda log: True)
    import scripts.spark_morning_render as smr
    monkeypatch.setattr(smr, "post_pc_event", lambda ev, data, kind="note", retire=False: events.append((ev, data["dso"], retire)) or True)
    # the chat client needs `requests`, absent on the pytest-only CI: a fake module
    monkeypatch.setitem(sys.modules, "utils.webchat_client",
                        types.SimpleNamespace(post_to_webchat=lambda m, image_path=None: chat.append(m) or True))
    import configs.config as cfg
    monkeypatch.setattr(cfg, "data", lambda: {"publish": {"gate": "live"}}, raising=False)
    logs = []
    assert pp.run(today=date(2026, 11, 7), log=logs.append) == 0
    assert published == [("IC410 - Emission Nebula", "x.jpg", True, True)]
    assert events == [("PUBLISH_DONE", "ic410", True)]
    assert chat and chat[0].startswith("Published: IC410 - Emission Nebula (#ic410, site abc1234)") and "retired" in chat[0]
    rec2 = json.loads((q / "ic410.json").read_text())
    assert rec2["status"] == "published" and rec2["site"]["commit"] == "abc1234"
    assert json.loads((folder / "draft.json").read_text())["status"] == "published"
    assert (folder / "article_published.json").exists()
    # the same approve again does nothing: the record is no longer pending
    assert pp.run(today=date(2026, 11, 7), log=logs.append) == 0 and len(published) == 1


def test_run_in_shadow_says_so_once_and_records_a_reject(tmp_path, monkeypatch):
    q = tmp_path / "queue"
    q.mkdir()
    folder = tmp_path / "m1"
    folder.mkdir()
    (folder / "draft.json").write_text(json.dumps({"status": "pending"}))
    (q / "ic410.json").write_text(json.dumps(_rec()))
    (q / "m1.json").write_text(json.dumps(_rec("m1", h="c" * 64, folder=str(folder))))
    monkeypatch.setattr(pp, "QUEUE_DIR", q)
    monkeypatch.setattr(pp, "read_pc_decisions", lambda: {
        "ic410": {"decision": "approve", "draft_hash": "a" * 64, "by": "t", "at": ""},
        "m1": {"decision": "reject", "draft_hash": "c" * 64, "by": "t", "why": "red stars"}})
    chat = []
    # the chat client needs `requests`, absent on the pytest-only CI: a fake module
    monkeypatch.setitem(sys.modules, "utils.webchat_client",
                        types.SimpleNamespace(post_to_webchat=lambda m, image_path=None: chat.append(m) or True))
    import configs.config as cfg
    monkeypatch.setattr(cfg, "data", lambda: {"publish": {"gate": "shadow"}}, raising=False)
    published = []
    monkeypatch.setattr(ao, "publish_article", lambda *a, **k: published.append(1))
    logs = []
    assert pp.run(today=date(2026, 11, 7), log=logs.append) == 0
    assert published == [] and len(chat) == 1 and "would publish now" in chat[0]
    assert json.loads((q / "ic410.json").read_text())["shadow_noted"] is True
    assert json.loads((q / "m1.json").read_text())["status"] == "rejected"
    assert json.loads((folder / "draft.json").read_text()) == {"status": "rejected", "why": "rejected by t: red stars"}
    pp.run(today=date(2026, 11, 7), log=logs.append)
    assert len(chat) == 1                                   # said once


def test_pc_unreachable_does_nothing(tmp_path, monkeypatch):
    q = tmp_path / "queue"
    q.mkdir()
    (q / "ic410.json").write_text(json.dumps(_rec()))
    monkeypatch.setattr(pp, "QUEUE_DIR", q)
    monkeypatch.setattr(pp, "read_pc_decisions", lambda: None)
    import configs.config as cfg
    monkeypatch.setattr(cfg, "data", lambda: {"publish": {"gate": "live"}}, raising=False)
    logs = []
    assert pp.run(today=date(2026, 11, 7), log=logs.append) == 1 and any("unreachable" in m for m in logs)
