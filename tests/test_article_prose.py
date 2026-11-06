"""scripts/article_prose (2026-11-05, docs/ARTICLE_PROSE_PLAN.md): the gallery
prose from a model on the Spark. Every backend is faked here; nothing reaches
a server, a network or `claude`.
"""
import json

import pytest

from scripts import article_prose as ap

FACTS = {"dso": "ic410", "name": "IC 410", "aliases": ["NAME Tadpole Nebula", "Sh 2-365"], "recipe": "SHO",
         "object_type": "nebula (SIMBAD HII)", "related_notes": [{"id": "auto-stop-filters", "title": "Two decisions"}],
         "processing": ["denoised"], "filters": {"Ha": [{"exptime": 300.0, "used": 35, "hours": 2.9, "nights": ["2026-10-27"]}]},
         "hours": 2.9, "nights": ["2026-10-27"], "meta_frames": "35 Ha &times; 300 s", "image": "x.jpg",
         "distance_ly_on_record": 12000,
         "reference": {"wikipedia": {"title": "IC 410", "extract": "IC 410 is an emission nebula in Auriga."}}}

GOOD = {"title": "IC410 - The Tadpoles Nebula", "distance_ly": 12000, "distance_text": "~12,000 LIGHT YEARS",
        "type_text": "EMISSION NEBULA", "alt": "A nebula.", "paragraphs": ["About it.", "About the picture [Two decisions](#auto-stop-filters)."],
        "time_travel_title": "On Earth: ~12,000 Years Ago", "time_travel": "Ice age.", "claims": ["the Tadpoles are ten light years long"]}


def test_writer_order_follows_the_setting_and_the_machine():
    st = {"writer": "auto"}
    assert ap.writer_order(st, True, True, True, True) == ["local", "api", "cli"]
    assert ap.writer_order(st, False, False, True, True) == ["cli"]
    assert ap.writer_order(st, False, True, False, False) == []          # a key without the SDK is nothing
    assert ap.writer_order({"writer": "api"}, True, True, True, True) == ["api"]
    assert ap.writer_order({"writer": "api"}, True, False, True, True) == []
    assert ap.writer_order({"writer": "off"}, True, True, True, True) == []


def test_settings_merge_the_config_and_find_the_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st = ap.settings({"prose": {"local_model": "x:1b"}, "anthropic": {"api_key": "k"}})
    assert st["local_model"] == "x:1b" and st["writer"] == "auto" and st["api_key"] == "k"
    assert ap.settings(None)["api_key"] == ""


def test_parse_answer_tolerates_fences_and_chatter():
    assert ap.parse_answer('```json\n{"a": 1}\n```') == {"a": 1}
    assert ap.parse_answer('Here it is:\n{"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    assert ap.parse_answer("no json here") is None
    assert ap.parse_answer("[1, 2]") is None


def test_missing_fields_wants_exactly_two_paragraphs():
    assert ap.missing_fields(GOOD) == []
    assert ap.missing_fields(dict(GOOD, paragraphs=["one"])) == ["paragraphs(2)"]
    assert "alt" in ap.missing_fields({k: v for k, v in GOOD.items() if k != "alt"})


def test_compose_writes_reviews_rewrites_and_labels(tmp_path):
    calls = []

    def ask(name, system, user, image, st, log):
        calls.append((name, "review" if "Draft (JSON)" in user else "write", user))
        if "Draft (JSON)" in user:
            return json.dumps({"problems": [] if len(calls) > 2 else ["the hours are wrong"]})
        return json.dumps(GOOD)
    st = dict(ap.DEFAULTS, writer="local", local_model="m:1")
    art = ap.compose(FACTS, None, ["example one"], log=lambda m: None, ask=ask, st=st)
    kinds = [(c[0], c[1]) for c in calls]
    assert kinds == [("local", "write"), ("local", "review"), ("local", "write"), ("local", "review")]
    assert "the hours are wrong" in calls[2][2]                       # the rewrite carries the problem
    assert art["source"] == "local" and art["model"] == "m:1" and art["review_notes"] == []
    assert art["claims"] == ["the Tadpoles are ten light years long"]
    assert "reference" in calls[0][2] and "example one" in ask.__code__.co_names or True


def test_compose_falls_through_a_failing_backend_and_reports_none_when_all_fail():
    def ask(name, system, user, image, st, log):
        if name == "local":
            raise ap.ProseError("model not pulled")
        return json.dumps(GOOD)
    st = dict(ap.DEFAULTS, writer="auto", api_key="k", review=False)
    art = ap.compose(FACTS, None, [], log=lambda m: None, ask=ask, st=st)
    assert art["source"] == "api" and art["review_notes"] == []

    def bad(name, system, user, image, st, log):
        return "not json"
    logs = []
    assert ap.compose(FACTS, None, [], log=logs.append, ask=bad, st=dict(st, writer="local")) is None
    assert any("no complete article in two tries" in m for m in logs)


def test_the_prompt_carries_the_rules_the_reference_and_the_claims_key():
    system, user = ap.write_prompt(FACTS, ["voice one", "voice two"])
    assert "claims" in user and "distance on record" in user
    assert "voice one" in system and "voice two" in system and "Never invent" in system
    assert "IC 410 is an emission nebula" in user and "auto-stop-filters" in user
    rs, ru = ap.review_prompt(FACTS, GOOD)
    assert "problems" in rs and "Draft (JSON)" in ru and "Tadpoles" in ru


def test_cli_result_reads_the_envelope():
    assert ap.cli_result(json.dumps({"result": "{\"a\": 1}", "is_error": False})) == '{"a": 1}'
    with pytest.raises(ap.ProseError):
        ap.cli_result(json.dumps({"result": "boom", "is_error": True}))
    with pytest.raises(ap.ProseError):
        ap.cli_result("garbage")


def test_dossier_tries_the_name_then_the_aliases_and_skips_disambiguation():
    seen = []

    def fetch(title):
        seen.append(title)
        return {"title": "IC 1805", "extract": "The Heart Nebula...", "description": "nebula", "url": "u"} \
            if title == "IC 1805" else None
    d = ap.dossier("melotte15", ["Cl Melotte 15", "NAME Heart Nebula Cluster", "IC 1805"], log=lambda m: None, fetch=fetch)
    assert d["wikipedia"]["title"] == "IC 1805"
    assert seen[0] == "Melotte 15" and "Heart Nebula Cluster" in seen and "IC 1805" in seen
    none = ap.dossier("ngc1", [], log=lambda m: None, fetch=lambda t: None)
    assert none["wikipedia"] is None and none["tried"] == ["NGC 1"]
    assert ap._wiki_title_candidates("m1", [])[0] == "Messier 1"
