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
        calls.append((name, "review" if "Draft (JSON)" in user else "write", user, st["local_model"]))
        if "Draft (JSON)" in user:
            return json.dumps({"problems": [] if len(calls) > 2 else ["the hours are wrong"]})
        return json.dumps(GOOD)
    st = dict(ap.DEFAULTS, writer="local", local_model="m:1", review_model="r:1")
    art = ap.compose(FACTS, None, ["example one"], log=lambda m: None, ask=ask, st=st)
    kinds = [(c[0], c[1], c[3]) for c in calls]
    assert kinds == [("local", "write", "m:1"), ("local", "review", "r:1"),
                     ("local", "write", "m:1"), ("local", "review", "r:1")]   # the other model reviews
    assert "the hours are wrong" in calls[2][2]                       # the rewrite carries the problem
    assert "outrank the reviewer" in calls[2][2]                      # and the facts beat a wrong note
    assert art["source"] == "local" and art["model"] == "m:1" and art["review_notes"] == []
    assert art["reviewer"] == "r:1" and art["spelling_fixed"] == []
    assert art["claims"] == ["the Tadpoles are ten light years long"]
    assert "reference" in calls[0][2]


def test_reviewer_is_the_other_model_and_falls_back_to_the_writer():
    st = dict(ap.DEFAULTS, local_model="m:1", review_model="r:1", review_backend="local")
    assert ap.reviewer_for("local", st)[1]["local_model"] == "r:1"
    assert ap.reviewer_for("local", dict(st, review_backend="same"))[1]["local_model"] == "m:1"
    assert ap.reviewer_for("api", st) == ("local", dict(st, local_model="r:1", local_temperature=ap.REVIEW_TEMPERATURE))
    assert ap.reviewer_for("local", dict(st, review_backend="cli"))[0] == "cli"

    seen = []

    def ask(name, system, user, image, st_, log):
        if "Draft (JSON)" in user:
            seen.append(st_["local_model"])
            if st_["local_model"] == "r:1":
                raise ap.ProseError("model r:1 is not pulled")
            return json.dumps({"problems": []})
        return json.dumps(GOOD)
    logs = []
    art = ap.compose(FACTS, None, [], log=logs.append, ask=ask, st=dict(st, writer="local"))
    assert seen == ["r:1", "m:1"] and art["reviewer"] == "m:1"
    assert any("the writer reviews its own draft" in m for m in logs)

    def mute(name, system, user, image, st_, log):
        if "Draft (JSON)" in user:
            raise ap.ProseError("down")
        return json.dumps(GOOD)
    art = ap.compose(FACTS, None, [], log=lambda m: None, ask=mute, st=dict(st, writer="local"))
    assert art["reviewer"] == "" and art["review_notes"] == []


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
    assert "written by another model" in rs
    assert "500 BC" in system and "about 12,000 years ago" not in user and "about 10 thousand years ago" not in user
    _, u2 = ap.write_prompt(dict(FACTS, distance_ly_on_record=2500), [])
    assert "left the object about 500 BC" in u2
    _, r2 = ap.review_prompt(dict(FACTS, distance_ly_on_record=12000), GOOD)
    assert "left the object about 10,000 BC" in r2                    # the critic gets the same arithmetic


def test_light_left_words():
    assert ap.light_left_words(2500) == "about 500 BC"
    assert ap.light_left_words(1500) == "about AD 500"
    assert ap.light_left_words(640) == "about AD 1390"
    assert ap.light_left_words(70e6) == "about 70 million years ago"
    assert ap.light_left_words(12000) == "about 10,000 BC"
    assert ap.light_left_words(25000) == "about 25 thousand years ago"
    assert ap.light_left_words(None) == "" and ap.light_left_words("x") == ""


def test_problem_lines_keep_defects_and_drop_praise():
    probs = ["the hours are wrong",
             "The draft's title 'M76 - Little Dumbbell Nebula' matches the required format.",
             "The distance of 2500 light years matches the reference.",
             {"class": 1, "quote": "~500 AD", "problem": "2,500 light years is about 500 BC, not AD 500"},
             {"class": 2, "quote": "the Soap Bubble entry", "problem": "refers to an entry the facts do not list; should be dropped"},
             {"class": 3, "quote": "", "problem": ""},
             "The draft correctly attributes the discovery to Messier.",
             {"class": 2, "quote": "two dense structures", "problem": "The reference block confirms this description. No defect."},
             {"class": 2, "quote": "radiation erodes", "problem": "A plausible explanation not contradicted by the reference, commonly accepted."},
             {"class": 1, "quote": "82 frames", "problem": "The frame count is right.", "must_change": False},
             {"class": 2, "quote": "ten light years", "problem": "the reference says each tadpole is about 10 light years long, which this confirms, but the draft says ten parsecs instead", "must_change": True}]
    notes, dropped = ap.problem_lines(probs)
    assert dropped == 6
    assert notes[0] == "the hours are wrong"
    assert notes[1].startswith("2,500 light years is about 500 BC") and '(draft: "~500 AD")' in notes[1]
    assert len(notes) == 4 and notes[3].startswith("the reference says each tadpole")
    assert ap.problem_lines("nonsense") == ([], 0)


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
    none = ap.dossier("ngc1", [], log=lambda m: None, fetch=lambda t: None, search=lambda n: [])
    assert none["wikipedia"] is None and none["tried"] == ["NGC 1"]
    assert ap._wiki_title_candidates("m1", [])[0] == "Messier 1"


def test_local_payload_turns_thinking_off_only_where_it_exists():
    st = dict(ap.DEFAULTS, local_model="g:1")
    pl = ap.local_payload(st, "sys", "usr", ["b64"], ["completion", "vision", "thinking"])
    assert pl["think"] is False and pl["messages"][1]["images"] == ["b64"] and pl["format"] == "json"
    assert "think" not in ap.local_payload(st, "sys", "usr", [], ["completion", "vision"])
    assert ap.local_payload(dict(st, local_think=True), "s", "u", [], ["thinking"])["think"] is True
    assert ap.local_payload(dict(st, local_temperature=0.2), "s", "u", [], [])["options"]["temperature"] == 0.2


def test_dossier_falls_back_to_search_when_the_intro_names_the_object():
    pages = {"Heart Nebula": {"title": "Heart Nebula", "extract": "The Heart Nebula ... its core cluster Melotte 15 ...", "description": "", "url": "u"},
             "Melotte catalogue": {"title": "Melotte catalogue", "extract": "A list.", "description": "", "url": "u"}}
    d = ap.dossier("melotte15", [], log=lambda m: None, fetch=lambda t: pages.get(t),
                   search=lambda n: ["Heart Nebula", "Wolf-Lundmark-Melotte"])
    assert d["wikipedia"]["title"] == "Heart Nebula" and d["by_search"] == "Melotte 15"
    assert d["tried"] == ["Melotte 15", "Heart Nebula"]
    miss = ap.dossier("melotte15", [], log=lambda m: None, fetch=lambda t: pages.get(t),
                      search=lambda n: ["Wolf-Lundmark-Melotte"])
    assert miss["wikipedia"] is None                      # a result that does not name the object is no reference


def test_britishise_replaces_the_unambiguous_words_and_keeps_names():
    t, w = ap.britishise("The colorful center of the catalog, in gray, 300 meters from the Center for Astrophysics. Color is key.")
    assert t == "The colourful centre of the catalogue, in grey, 300 metres from the Center for Astrophysics. Colour is key."
    assert w == ["colorful", "center", "catalog", "gray", "meters", "Color"]
    assert ap.britishise("a program of practice with a meter")[1] == []
    art = {"title": "x", "alt": "A colorful nebula", "paragraphs": ["ionized gas", "fine"], "time_travel": "ok", "time_travel_title": "On Earth"}
    assert ap.apply_spelling(art) == ["colorful", "ionized"] and art["alt"] == "A colourful nebula" and art["paragraphs"][0] == "ionised gas"


def test_the_prompts_carry_the_new_rules():
    system, _ = ap.write_prompt(FACTS, [])
    assert "British spelling" in system and "never describe what it found" in system
    rs, _ = ap.review_prompt(FACTS, GOOD)
    assert "American spelling" in rs and "NGC2146 - The Dusty Hand" in rs and "span of dates" in rs and "what the observatory did" in rs


def test_britishise_leaves_a_linked_note_title_alone():
    text = "See [Weighing a star cluster's age with a color\u2013magnitude diagram](#hr-diagram) for the color of the center."
    out, w = ap.britishise(text)
    assert "color\u2013magnitude diagram](#hr-diagram)" in out and out.endswith("the colour of the centre.")
    assert w == ["color", "center"]
    assert ap.britishise("totaling 8.4 hours, funneling gas")[0] == "totalling 8.4 hours, funnelling gas"


def test_the_note_rule_is_capped():
    _, user = ap.write_prompt(FACTS, [])
    assert "one to three" in user and "never a long list" in user
    rs, _ = ap.review_prompt(FACTS, GOOD)
    assert "links none of them" in rs and "a note left unlinked is not a defect" in rs
