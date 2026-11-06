"""iris_astronomy/name_fallback (2026-11-06): a position and a type for a name
when SIMBAD is down (both Sesame servers unreachable for days; every `image`
answered "Not a known object"). Wikipedia's infobox by the object's own page,
then the suggestion catalogue; never a text search."""
import pytest

from iris_astronomy import name_fallback as nf

NGC7008 = """{{Infobox planetary nebula
| name = NGC 7008
| image = x.jpg
| ra = {{RA|21|00|32.503}}<ref name="simbad">{{cite web |title=x}}</ref>
| dec = {{DEC|+54|32|36.18}}<ref name="simbad" />
| dist_ly = 2800
}}
'''NGC 7008''' ('''PK 93+5.2'''), also known as the '''Fetus Nebula''', is a [[planetary nebula]] with a diameter of approximately 1 [[light-year]]<ref name=APOD>{{cite web|url=x}}</ref> located at a distance of 2800 [[light years]] in northern [[Cygnus (constellation)|Cygnus]]. It was discovered by [[William Herschel]] in 1787.
"""
M13 = """{{Infobox globular cluster
| ra = {{RA|16|41|41.24}}<ref name="aj"/>
| dec = {{DEC|+36|27|35.5}}<ref name=aj/>
}}
'''Messier 13''' or '''M13''' (also designated '''NGC 6205''') is a [[globular cluster]] of several hundred thousand stars in the constellation of [[Hercules (constellation)|Hercules]].
"""
IC410 = """{{Infobox nebula
| ra = {{RA|05|21|60.00}}
| dec = {{DEC|+33|29|0.0}}
}}
'''IC 410''' is a faint and dusty [[emission nebula]] of more than 100 light-years across, located some 12,000 light-years away.
"""
DISAMBIG = "'''M13''' may refer to:\n* [[Messier 13]]\n{{disambiguation}}\n"
NO_COORDS = "{{Infobox person\n| name = Someone\n}}\n'''Someone''' is a person.\n"
SOUTH = "{{Infobox galaxy\n| ra = {{RA|01|33|50.9}}\n| dec = {{DEC|-30|39|36}}\n}}\n'''NGC 613''' is a [[barred spiral galaxy]] in [[Sculptor (constellation)|Sculptor]].\n"
PAGES = {"NGC 7008": NGC7008, "Messier 13": M13, "M13": DISAMBIG, "IC 410": IC410, "Someone": NO_COORDS, "NGC 613": SOUTH}


def fetch(title):
    t = PAGES.get(title)
    return {"title": title, "text": t} if t else None


def test_titles_are_the_catalogue_forms():
    assert nf.wiki_titles("ngc7008") == ["NGC 7008", "ngc7008"]
    assert nf.wiki_titles("m13")[0] == "Messier 13" and nf.wiki_titles("M 13")[0] == "Messier 13"
    assert nf.wiki_titles("sh2-101")[0] == "Sh 2-101" and nf.wiki_titles("ic 410")[0] == "IC 410"
    assert nf.wiki_titles("melotte15")[0] == "Melotte 15" and nf.wiki_titles("Hubble V1") == ["Hubble V1"]
    assert nf.wiki_titles("") == []


def test_the_infobox_position_matches_simbad():
    ra, dec = nf.parse_position(NGC7008)
    assert ra == pytest.approx(315.13543, abs=1e-4) and dec == pytest.approx(54.54338, abs=1e-4)
    ra, dec = nf.parse_position(SOUTH)
    assert ra == pytest.approx(23.4621, abs=1e-3) and dec == pytest.approx(-30.66, abs=1e-3)   # a negative declination keeps its sign
    ra, dec = nf.parse_position(IC410)
    assert ra == pytest.approx(80.5, abs=1e-4)                                                  # 60.00 s carries into the minute
    assert nf.parse_position(NO_COORDS) is None and nf.parse_position("") is None


def test_the_first_sentence_gives_the_family():
    assert nf.first_sentence(NGC7008).startswith("NGC 7008 (PK 93+5.2), also known as the Fetus Nebula, is a planetary nebula")
    assert nf.kind_letter(nf.first_sentence(NGC7008)) == "N"
    assert nf.kind_letter(nf.first_sentence(M13)) == "C"
    assert nf.kind_letter(nf.first_sentence(IC410)) == "N"
    assert nf.kind_letter(nf.first_sentence(SOUTH)) == "G"
    assert nf.kind_letter("Abell 2218 is a cluster of galaxies about 2 billion light-years away.") == "G"   # as SIMBAD's ClG on the grid
    assert nf.kind_letter("Stephan's Quintet is a visual grouping of five galaxies.") == "G"
    assert nf.kind_letter("NGC 6888 is an emission nebula around a Wolf-Rayet star in the galaxy's disk.") == "N"   # the earliest word
    assert nf.kind_letter("Someone is a person.") == "?"


def test_wikipedia_resolves_by_the_page_never_a_search():
    hit = nf.wikipedia("ngc7008", fetch)
    assert hit["source"] == "wikipedia" and hit["letter"] == "N" and hit["title"] == "NGC 7008"
    assert hit["ra_deg"] == pytest.approx(315.13543, abs=1e-4)
    hit = nf.wikipedia("m13", fetch)                      # "M13" is a disambiguation page; "Messier 13" is tried first
    assert hit["letter"] == "C" and hit["title"] == "Messier 13"
    assert nf.wikipedia("Someone", fetch) is None         # a page with no position is no answer
    assert nf.wikipedia("ngc99999", fetch) is None
    assert nf.wikipedia("x", lambda t: None) is None      # offline


def test_the_catalogue_is_the_last_resort_and_approximate():
    hit = nf.catalogue("ic1795")
    assert hit and hit["source"] == "catalogue" and hit["letter"] == "N" and abs(hit["ra_deg"] - 38.2) < 1
    assert nf.catalogue("ngc7008") is None
    assert nf.resolve("ic1795", fetch=lambda t: None)["source"] == "catalogue"


def test_resolve_remembers_hits_and_misses(monkeypatch):
    calls = []

    def counting(title):
        calls.append(title)
        return fetch(title)
    nf._HITS.clear(); nf._MISSES.clear()
    assert nf.resolve("ngc7008", counting)["letter"] == "N"
    assert nf.letter_for("ngc 7008", counting) == "N" and len(calls) == 1        # the hit answers the type, no fetch
    assert nf.letter_for("gravwav", counting) == "?"
    n = len(calls)
    assert nf.resolve("gravwav", counting) is None and len(calls) == n            # a miss is not asked again for an hour
    nf._HITS.clear(); nf._MISSES.clear()


def test_is_a_dso_object_falls_back_when_sesame_fails(monkeypatch):
    adv = pytest.importorskip("iris_astronomy.astro_dso_visibility")
    from astropy.coordinates import SkyCoord

    def down(name, cache=False):
        raise RuntimeError("All Sesame queries failed")
    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(down))
    monkeypatch.setattr(nf, "_fetch_wikitext", fetch)
    nf._HITS.clear(); nf._MISSES.clear()
    t = adv.is_a_dso_object("ngc7008")
    assert t is not None and t.coord.ra.deg == pytest.approx(315.13543, abs=1e-4) and t.name == "ngc7008"
    assert adv.resolved_by("ngc7008") == "wikipedia"
    assert adv.is_a_dso_object("ngc99999") is None and adv.resolved_by("ngc99999") is None
    assert adv.is_a_dso_object("ic1795").coord.dec.deg == pytest.approx(61.5, abs=1) and adv.resolved_by("ic1795") == "catalogue"
    # the type lookup reads the same hit when SIMBAD is not set up
    monkeypatch.setattr(adv, "_get_simbad", lambda: None)
    monkeypatch.setattr(adv, "_load_type_cache", lambda: {})
    assert adv.get_dso_type("ngc7008") == "N" and adv.get_dso_type("ic1795") == "N" and adv.get_dso_type("gravwav") == "?"
    nf._HITS.clear(); nf._MISSES.clear()


def test_a_sesame_outage_is_not_retried_on_every_name(monkeypatch):
    adv = pytest.importorskip("iris_astronomy.astro_dso_visibility")
    from astropy.coordinates import SkyCoord
    calls = []

    def down(name, cache=False):
        calls.append(name)
        raise RuntimeError("All Sesame queries failed. Unable to retrieve coordinates. See errors per URL below: \n x: timed out")
    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(down))
    monkeypatch.setattr(nf, "_fetch_wikitext", fetch)
    monkeypatch.setattr(adv, "_sesame_retry_after", 0.0)
    nf._HITS.clear(); nf._MISSES.clear()
    assert adv.is_a_dso_object("ngc7008") is not None and len(calls) == 1
    assert adv.is_a_dso_object("m13") is not None and len(calls) == 1          # inside the cooldown: straight to the fallback
    monkeypatch.setattr(adv, "_sesame_retry_after", 0.0)
    nf._HITS.clear(); nf._MISSES.clear()
