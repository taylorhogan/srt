"""control/suggest and configs/candidates.json (E1 of docs/AUTOMATION_PLAN.md, 2026-10-30).

Pure: fixture visibilities, the size band, the exclusions and the ranking;
the catalogue is checked for shape so a hand edit cannot break the Monday card.
"""
import json
from pathlib import Path

from control import suggest as S

ROOT = Path(__file__).resolve().parents[1]
FOV = S.fov_arcmin(0.26)      # (41.5, 27.7)


def test_catalogue_is_well_formed():
    cands = S.load_catalogue(ROOT / "configs" / "candidates.json")
    assert len(cands) >= 80
    names = [c["name"] for c in cands]
    assert len(names) == len(set(names))
    for c in cands:
        assert set(c) >= {"name", "common", "kind", "ra_deg", "dec_deg", "size_arcmin", "emission", "palette", "note"}
        assert c["kind"] in S.KINDS and c["palette"] in S.DEFAULT_PLANS
        assert 0 <= c["ra_deg"] < 360 and -30 <= c["dec_deg"] <= 90, c["name"]      # reachable from Boston
        assert c["size_arcmin"] > 0 and c["name"] == S.squash(c["name"]) or "-" in c["name"]
        if c["kind"] in ("nebula", "snr", "pn"):
            assert c["emission"] and c["palette"] in ("SHO", "HOO", "HALRGB")
        else:
            assert c["palette"] in ("LRGB", "HALRGB")


def test_field_and_size_fit():
    assert round(FOV[0], 1) == 41.5 and round(FOV[1], 1) == 27.7
    assert S.size_fit(10, FOV) == 1.0 and S.size_fit(24, FOV) == 1.0           # in the band
    assert 0 < S.size_fit(2.5, FOV) < 1.0 and S.size_fit(1, FOV) == 0.0        # a dot
    assert 0 < S.size_fit(35, FOV) < 1.0 and S.size_fit(60, FOV) == 0.0        # a mosaic
    assert S.size_fit(0, FOV) == 0.5


def test_visibility_and_season():
    assert S.visibility([4, 4, 2, 0]) == (0.5, 2)
    assert S.visibility([]) == (0.0, 0)
    assert S.season([5] * 8) == 1.0
    assert S.season([5, 5, 4, 4, 3, 3, 2, 2]) < 1.0                           # setting
    assert S.season([2, 2, 3, 3, 4, 4, 5, 5]) == 1.0                          # rising
    assert S.season([0, 0, 0, 0]) == 0.5 and S.season([3]) == 1.0


def test_exclusions_by_name_or_alias():
    q, im, st = {"ngc2403", "m82"}, {"ic410", "trunk"}, {"ngc6888", "trunk", "pelican"}
    assert S.excluded_by("ngc2403", [], q, im, st) == "already in the queue"
    assert S.excluded_by("IC 410", [], q, im, st) == "already has frames on disk"
    assert S.excluded_by("ngc6888", ["Crescent Nebula"], q, im, st) == "already on the site"
    assert S.excluded_by("ic5070", ["Pelican"], q, im, st) == "already on the site"
    assert S.excluded_by("ngc281", ["Pacman Nebula"], q, im, st) is None


def test_rank_orders_by_the_product_and_explains():
    cands = [{"name": "neb1", "common": "A", "kind": "nebula", "size_arcmin": 20, "palette": "SHO"},
             {"name": "gal1", "common": "B", "kind": "galaxy", "size_arcmin": 8, "palette": "LRGB"},
             {"name": "gal2", "common": "C", "kind": "galaxy", "size_arcmin": 8, "palette": "LRGB"},
             {"name": "clu1", "common": "D", "kind": "cluster", "size_arcmin": 15, "palette": "LRGB"}]
    hours = {"neb1": [5] * 30, "gal1": [5] * 30, "gal2": [4] * 9 + [0] * 21, "clu1": [5] * 30}
    moon = {"neb1": 0.95, "gal1": 0.5, "gal2": 0.9, "clu1": 0.5}
    r = S.rank(cands, hours, moon, FOV, queue={"clu1"}, imaged=set(), site=set())
    assert [v["name"] for v in r] == ["neb1", "gal1", "clu1", "gal2"]       # ties and zeros by name
    assert r[0]["score"] == round(1.0 * 1.0 * 0.95 * 1.0, 4) and r[1]["score"] == 0.5
    assert r[3]["score"] == 0 and "only 9 of the next 30 nights" in r[3]["reasons"][0]
    assert r[2]["excluded"] == "already in the queue" and r[2]["score"] == 0
    text = S.describe(r, top=2)
    assert text.startswith("Worth adding this month") and "  neb1 -- A (nebula, SHO): 30 of 30 nights with 3 h+" in text
    assert "gal2 --" not in text and "clu1 --" not in text
    assert S.describe([r[3]], 3).startswith("Nothing worth adding")
