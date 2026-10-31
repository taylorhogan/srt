"""photometry/aavso.py: the AAVSO Extended Format from the light-curve pipeline (2026-10-31).

Pure parts only: the VSP sequence parsed from the chart JSON, the check star
held back from the ensemble, the header and the observation line.
"""
from photometry import aavso

VSP = {"chartid": "X43358XVR", "star": "CF Cas", "auid": "000-BCV-173", "ra": "23:58:17.98", "dec": "61:13:15.8",
       "photometry": [
           {"auid": "000-BFX-116", "ra": "23:58:38.21", "dec": "61:19:19.5", "label": "110",
            "bands": [{"band": "V", "mag": 11.017, "error": 0.026}, {"band": "B", "mag": 11.474, "error": 0.03}]},
           {"auid": "000-BFX-117", "ra": "23:57:40.22", "dec": "61:03:11.5", "label": "112",
            "bands": [{"band": "V", "mag": 11.211, "error": 0.024}]},
           {"auid": "000-BFX-118", "ra": "23:58:00.00", "dec": "-61:03:11.5", "label": "125",
            "bands": [{"band": "V", "mag": 12.5, "error": 0.02}]},
           {"auid": "000-BFX-119", "ra": "23:58:00.00", "dec": "61:00:00.0", "label": "131",
            "bands": [{"band": "V", "mag": 13.1, "error": 0.02}]},
           {"auid": "000-BFX-120", "ra": "23:58:00.00", "dec": "61:00:00.0", "label": "140",
            "bands": [{"band": "V", "mag": 14.0, "error": 0.02}]},
           {"auid": "000-BFX-999", "ra": "23:58:00.00", "dec": "61:00:00.0", "label": "B only",
            "bands": [{"band": "B", "mag": 14.0, "error": 0.02}]},
       ]}


def test_sexagesimal_to_degrees():
    assert abs(aavso.sexa_to_deg("23:58:17.98", hours=True) - 359.5749) < 1e-3
    assert abs(aavso.sexa_to_deg("61:13:15.8") - 61.2211) < 1e-3
    assert abs(aavso.sexa_to_deg("-61:03:11.5") + 61.0532) < 1e-3
    assert aavso.sexa_to_deg("359.911") == 359.911
    assert aavso.sexa_to_deg("junk") is None


def test_sequence_keeps_only_stars_with_a_v_magnitude():
    seq = aavso.parse_sequence(VSP)
    assert seq["chartid"] == "X43358XVR" and seq["star"] == "CF Cas"
    assert [c["label"] for c in seq["comps"]] == ["110", "112", "125", "131", "140"]
    assert seq["comps"][0]["v"] == 11.017 and abs(seq["comps"][0]["ra"] - 359.659) < 1e-2


def test_the_check_star_is_held_back_from_the_ensemble():
    comps = aavso.parse_sequence(VSP)["comps"]
    pick = aavso.choose(comps, (10.0, 15.0), target_mag=11.3)
    assert pick["check"]["label"] == "112"                       # closest in brightness to the target
    assert [c["label"] for c in pick["ensemble"]] == ["110", "125", "131", "140"]
    tight = aavso.choose(comps, (12.0, 13.5))                    # too few for a check star: all comps
    assert tight["check"] is None and [c["label"] for c in tight["ensemble"]] == ["125", "131"]


def test_the_header_and_the_line_are_extended_format():
    hdr = aavso.header("TXYZ")
    assert hdr[0] == "#TYPE=EXTENDED" and hdr[1] == "#OBSCODE=TXYZ" and hdr[3] == "#DELIM=," and hdr[4] == "#DATE=JD"
    assert hdr[-1].startswith("#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES")
    line = aavso.format_line("CF Cas", 2461000.61234, 11.2141, 0.0123, "L", "X43358XVR", "112", 11.205,
                             aavso.notes_text(11, 60, 6.0, 4, "check star catalog V 11.211"))
    f = line.split(",")
    assert len(f) == 15
    assert f[:7] == ["CF Cas", "2461000.61234", "11.214", "0.012", "CV", "NO", "STD"]
    assert f[7:11] == ["ENSEMBLE", "na", "112", "11.205"]
    assert f[13] == "X43358XVR" and "11 x 60 s" in f[14] and "," not in f[14]
    assert aavso.format_line("x", 1.0, 1.0, 0.1, "G", None, None, None).split(",")[4] == "TG"


def test_an_unknown_filter_refuses():
    import pytest
    with pytest.raises(ValueError):
        aavso.filter_code("Ha")


def test_the_queue_name_maps_to_the_vsx_name():
    assert aavso.VSX_NAMES["hubblev1"] == "M31_V1"


def test_a_thin_sequence_borrows_the_neighbours_comps():
    own = {"star": "CG Cas", "chartid": "X1", "comps": [{"auid": "a", "v": 12.0}, {"auid": "b", "v": 13.0}]}
    cf = {"star": "CF Cas", "chartid": "X2", "comps": [{"auid": "b", "v": 13.0}] + [{"auid": "c%d" % i, "v": 11 + i} for i in range(6)]}
    pooled = aavso.pool_sequences(own, [cf], 5)
    assert pooled["chartid"] == "X2" and pooled["borrowed_from"] == "CF Cas"
    assert [c["auid"] for c in pooled["comps"]] == ["a", "b"] + ["c%d" % i for i in range(6)]
    assert aavso.pool_sequences(cf, [own], 5) is cf                   # enough of its own: untouched
