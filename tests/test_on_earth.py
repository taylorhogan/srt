"""scripts/on_earth (2026-11-06): the 'On Earth' box from a standard table of
eras, the model out of it. The table is contiguous from 0 to 14 billion years
and every distance maps to exactly one era."""
import json

from scripts import on_earth as oe
from scripts import article_prose as ap
from scripts import astro_output as ao


def test_the_table_is_contiguous_and_complete():
    eras = oe.load()
    assert eras[0]["from"] == 0 and eras[-1]["to"] >= 13.8e9
    for a, b in zip(eras, eras[1:]):
        assert a["to"] == b["from"], (a["id"], b["id"])
    ids = [e["id"] for e in eras]
    assert len(ids) == len(set(ids))
    for e in eras:
        assert e["name"] and e["span"] and e["summary"] and len(e["paragraph"]) > 150, e["id"]
        assert not e["summary"].endswith("."), e["id"]


def test_every_distance_the_telescope_can_reach_has_one_era():
    for ly in (1, 440, 2500, 6500, 12000, 25000, 2.7e6, 25e6, 180e6, 300e6, 2.4e9, 13e9, 14e9):
        assert oe.era_for(ly) is not None, ly
    assert oe.era_for(-1) is None and oe.era_for("x") is None and oe.era_for(15e9) is None
    assert oe.era_for(2500)["id"] == "classical"          # M76, 500 BC
    assert oe.era_for(6500)["id"] == "neolithic"          # M1, 4500 BC
    assert oe.era_for(12000)["id"] == "last-glacial"      # IC 410
    assert oe.era_for(25e6)["id"] == "oligocene"          # NGC 6946
    assert oe.era_for(180e6)["id"] == "jurassic-early"    # NGC 1961
    assert oe.era_for(300e6)["id"] == "carboniferous"     # the Quintet
    assert oe.era_for(5e9)["id"] == "before-earth"


def test_words_and_titles():
    assert oe.years_ago_words(2500) == "about 500 BC" and oe.years_ago_words(1500) == "about AD 500"
    assert oe.years_ago_words(12000) == "about 12,000 years ago"
    assert oe.years_ago_words(25e6) == "about 25 million years ago" and oe.years_ago_words(2.4e9) == "about 2.4 billion years ago"
    assert oe.title_for(2500) == "On Earth: ~500 BC" and oe.title_for(1500) == "On Earth: ~AD 500"
    assert oe.title_for(25000) == "On Earth: ~25,000 Years Ago" and oe.title_for(70e6) == "On Earth: ~70 Million Years Ago"
    assert oe.phase(25e6, oe.era_for(25e6)) == "late" and oe.phase(33e6, oe.era_for(33e6)) == "early"
    assert oe.phase(2500, oe.era_for(2500)) == ""
    assert oe.phase(12000, oe.era_for(12000)) == "" and "in [the last ice age](#era-last-glacial)" in oe.box_for("IC 410", 12000)["time_travel"]
    assert "in [the first galaxies](#era-first-galaxies)" in oe.box_for("x", 13e9)["time_travel"]


def test_box_links_the_era_and_says_when():
    box = oe.box_for("NGC 6946", 25e6)
    assert box["time_travel_title"] == "On Earth: ~25 Million Years Ago" and box["on_earth_era"] == "oligocene"
    assert box["time_travel"].startswith("This light left NGC 6946 about 25 million years ago, in [the late Oligocene](#era-oligocene): ")
    assert oe.box_for("M 76", 2500)["time_travel"].startswith("This light left M 76 about 500 BC, in [classical antiquity](#era-classical): ")
    assert oe.box_for("x", None) is None and oe.box_for("x", 0) is None


def test_compose_takes_the_box_from_the_table(tmp_path):
    good = {"title": "NGC6946 - Fireworks", "distance_ly": 25000000, "distance_text": "~25 MILLION LIGHT YEARS", "type_text": "SPIRAL GALAXY",
            "alt": "A galaxy.", "paragraphs": ["One.", "Two."], "time_travel_title": "On Earth: wrong", "time_travel": "The Triassic.", "claims": []}
    facts = {"dso": "ngc6946", "name": "NGC 6946", "distance_ly_on_record": 25000000, "related_notes": [], "reference": {}}
    st = dict(ap.DEFAULTS, writer="local", local_model="m:1", review=False)
    art = ap.compose(facts, None, [], log=lambda m: None, ask=lambda *a: json.dumps(good), st=st)
    assert art["on_earth_era"] == "oligocene" and "(#era-oligocene)" in art["time_travel"]
    assert art["time_travel_title"] == "On Earth: ~25 Million Years Ago"
    # no record: the writer's own distance picks the era; none at all: the writer's sentence stands
    art2 = ap.compose(dict(facts, distance_ly_on_record=None), None, [], log=lambda m: None, ask=lambda *a: json.dumps(good), st=st)
    assert art2["on_earth_era"] == "oligocene"
    art3 = ap.compose(dict(facts, distance_ly_on_record=None), None, [], log=lambda m: None,
                      ask=lambda *a: json.dumps(dict(good, distance_ly=None)), st=st)
    assert "on_earth_era" not in art3 and art3["time_travel"] == "The Triassic."


def test_the_prompts_no_longer_ask_for_history():
    _, user = ap.write_prompt({"dso": "x"}, [])
    assert "write nothing about history here" in user
    rs, _ = ap.review_prompt({"dso": "x"}, {"title": "x"})
    assert "do not review it" in rs and "500 BC (not AD 500)" not in rs


def test_timeline_section_round_trips_into_the_page():
    page = ('<body>\n    <div class="container" id="gallery-panel">\n        <article id="m1" class="dso-section"></article>\n\n'
            '    </div>\n\n    <div class="container" id="notes-panel">\n    </div>\n</body>')
    out = oe.with_timeline(page, oe.timeline_html())
    assert out.count(oe.START) == 1 and out.count(oe.END) == 1
    gallery_close = out.index("</div>", out.index('id="m1"'))
    assert gallery_close < out.index(oe.START) < out.index('id="notes-panel"')     # a sibling between the panels
    assert 'body[data-mode="gallery"] .on-earth { display: block; }' in out
    for e in oe.load():
        assert f'id="era-{e["id"]}"' in out
    assert "location.hash.indexOf('#era-')" in out and "dataset.mode = 'gallery'" in out   # a deep link opens the gallery tab
    again = oe.with_timeline(out, oe.timeline_html())
    assert again == out                                       # idempotent
    changed = oe.with_timeline(out, oe.timeline_html().replace("When the light left", "When the light set out"))
    assert changed.count("When the light set out") == 1 and "When the light left</h2>" not in changed
    assert changed.count(oe.START) == 1 and changed.count('id="notes-panel"') == 1
    # an earlier version put the block INSIDE the gallery panel; it is moved out
    inside = page.replace('</article>\n\n', '</article>\n\n' + oe.timeline_html() + '\n', 1)
    moved = oe.with_timeline(inside, oe.timeline_html())
    assert moved.count(oe.START) == 1 and moved == out


def test_the_gallery_box_renders_the_era_link():
    art = {"title": "NGC6946 - Fireworks", "distance_ly": 25000000, "distance_text": "~25 MILLION LIGHT YEARS", "type_text": "SPIRAL GALAXY",
           "alt": "A galaxy.", "paragraphs": ["One.", "Two."]}
    art.update(oe.box_for("NGC 6946", 25e6))
    frag = ao.build_fragment(art, {"meta_frames": "49 L", "hours": 8.4, "nights": ["2026-10-15"], "dso": "ngc6946", "name": "NGC 6946"}, "ngc6946.jpg", "2026-11-06")
    assert '<a href="#era-oligocene">the late Oligocene</a>' in frag
