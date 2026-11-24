"""scripts/sequence_gain: a gain onto the flats and darks sequences (no N.I.N.A, no files)."""
import copy
import json

import pytest

from scripts import sequence_gain as sg


def _take(idn, kind="FLAT", exp=2.0):
    return {"$id": idn, "$type": sg.TAKE, "ExposureTime": exp, "Gain": -1, "Offset": -1,
            "Binning": {"$ref": "5"}, "ImageType": kind, "ExposureCount": 7}


def _trained(idn, name, position, take_id):
    base = int(idn)
    return {
        "$id": idn, "$type": sg.TRAINED, "ErrorBehavior": 0, "Attempts": 1, "KeepPanelClosed": False,
        "Name": "Trained Flat Exposure",
        "Conditions": {"$values": []},
        "Items": {"$values": [
            {"$id": str(base + 1), "$type": "NINA.Sequencer.SequenceItem.FlatDevice.CloseCover, NINA.Sequencer"},
            {"$id": str(base + 2), "$type": "NINA.Sequencer.SequenceItem.FlatDevice.ToggleLight, NINA.Sequencer", "OnOff": True},
            {"$id": str(base + 3), "$type": sg.SWITCH,
             "Filter": {"$id": str(base + 4), "_name": name, "_position": position,
                        "FlatWizardFilterSettings": {"Gain": -1, "Offset": -1}}},
            {"$id": str(base + 5), "$type": "NINA.Sequencer.SequenceItem.FlatDevice.SetBrightness, NINA.Sequencer", "Brightness": 255},
            {"$id": str(base + 6), "$type": "NINA.Sequencer.Container.SequentialContainer, NINA.Sequencer",
             "Conditions": {"$values": [{"$id": str(base + 7), "$type": "NINA.Sequencer.Conditions.LoopCondition, NINA.Sequencer", "Iterations": 20}]},
             "Items": {"$values": [_take(take_id)]}},
            {"$id": str(base + 8), "$type": "NINA.Sequencer.SequenceItem.FlatDevice.ToggleLight, NINA.Sequencer", "OnOff": False},
            {"$id": str(base + 9), "$type": "NINA.Sequencer.SequenceItem.FlatDevice.OpenCover, NINA.Sequencer"},
        ]},
    }


def _flats_seq():
    first = _trained("10", "L", 0, "20")
    first["Items"]["$values"][4]["Items"]["$values"][0]["Binning"] = {"$id": "5", "X": 1, "Y": 1}
    return {"$id": "1", "$type": "NINA.Sequencer.Container.SequenceRootContainer, NINA.Sequencer",
            "Name": "flat_sequence",
            "Items": {"$values": [first, _trained("30", "R", 1, "40"), _trained("50", "Ha", 6, "60"),
                                  _trained("70", "S-II", 4, "80")]}}


def test_named_filters_become_auto_exposure_flats_at_the_gain_and_the_rest_stay_trained():
    seq = _flats_seq()
    before = copy.deepcopy(seq)
    changed = sg.convert_flats(seq, {"Ha": (26, 30), "S-II": (26, 30)}, min_s=2.0, max_s=120.0)
    assert set(changed) == {"Ha", "S-II"}
    assert changed["Ha"]["was"] == [(-1, -1, 2.0)]
    blocks = {sg.block_filter(b): b for b in seq["Items"]["$values"]}
    for name in ("Ha", "S-II"):
        b = blocks[name]
        assert b["$type"] == sg.AUTO and b["Name"] == "Auto Exposure Flat"
        assert (b["MinExposure"], b["MaxExposure"]) == (2.0, 120.0)
        assert (b["HistogramTargetPercentage"], b["HistogramTolerancePercentage"]) == (0.5, 0.1)
        te = sg.exposure_items(b)[0]
        assert (te["Gain"], te["Offset"], te["ImageType"], te["ExposureCount"]) == (26, 30, "FLAT", 0)
        # The child layout N.I.N.A indexes into (cover, light, filter, brightness, loop, light, cover) is kept.
        kinds = [i["$type"].split(",")[0].split(".")[-1] for i in b["Items"]["$values"]]
        assert kinds == ["CloseCover", "ToggleLight", "SwitchFilter", "SetBrightness",
                         "SequentialContainer", "ToggleLight", "OpenCover"]
        # Json.NET wants the metadata first: nothing was inserted ahead of $id/$type.
        assert list(b.keys())[:2] == ["$id", "$type"]
    for name in ("L", "R"):
        assert blocks[name] == {k: v for k, v in before["Items"]["$values"][["L", "R"].index(name)].items()}
    # A table naming a filter with no block is reported by the caller, not an error here.
    assert sg.convert_flats(copy.deepcopy(_flats_seq()), {"O-III": (26, 30)}) == {}


def test_the_search_window_bounds_the_starting_exposure():
    seq = _flats_seq()
    sg.convert_flats(seq, {"Ha": (26, 30)}, min_s=5.0, max_s=60.0)
    b = {sg.block_filter(x): x for x in seq["Items"]["$values"]}["Ha"]
    assert sg.exposure_items(b)[0]["ExposureTime"] == 5.0


def test_trained_to_auto_refuses_the_wrong_block():
    with pytest.raises(ValueError):
        sg.trained_to_auto({"$type": sg.AUTO, "Items": {"$values": []}}, 26, 30, 2, 120)


def test_darks_and_bias_take_the_gain_and_a_new_name():
    seq = {"$id": "1", "$type": "root", "Name": "darks_and_bias",
           "Items": {"$values": [
               {"$id": "2", "$type": "x", "Items": {"$values": [_take("3", "BIAS", 0.0001)]}},
               {"$id": "4", "$type": "x", "Items": {"$values": [_take("5", "DARK", 300.0)]}},
               {"$id": "6", "$type": "x", "Items": {"$values": [_take("7", "LIGHT", 30.0)]}},
           ]}}
    counts = sg.convert_darks(seq, 26, 30, "darks_and_bias_gain26")
    assert counts == {"BIAS": 1, "DARK": 1} and seq["Name"] == "darks_and_bias_gain26"
    takes = [n for n in sg.walk(seq) if n.get("$type") == sg.TAKE]
    assert [(t["ImageType"], t["Gain"], t["Offset"]) for t in takes] == \
        [("BIAS", 26, 30), ("DARK", 26, 30), ("LIGHT", -1, -1)]
    assert takes[1]["ExposureTime"] == 300.0


def test_darks_start_from_nothing_whatever_the_template_had_run():
    """2026-11-24: the template was saved after a run (bias 50/50, dark 1/20); N.I.N.A
    loaded the copy, skipped the bias block and took 19 darks."""
    loop = lambda idn, done, n: {"$id": idn, "$type": sg.LOOP, "CompletedIterations": done, "Iterations": n}
    seq = {"$id": "1", "$type": "root", "Name": "darks_and_bias",
           "Items": {"$values": [
               {"$id": "2", "$type": "x", "Conditions": {"$values": [loop("8", 50, 50)]},
                "Items": {"$values": [_take("3", "BIAS", 0.0001)]}},
               {"$id": "4", "$type": "x", "Conditions": {"$values": [loop("9", 1, 20)]},
                "Items": {"$values": [_take("5", "DARK", 300.0)]}},
           ]}}
    sg.convert_darks(seq, 26, 30, "darks_and_bias_gain26")
    loops = [n for n in sg.walk(seq) if n.get("$type") == sg.LOOP]
    assert [(l["CompletedIterations"], l["Iterations"]) for l in loops] == [(0, 50), (0, 20)]
    assert sg.reset_progress(seq) == 0


def test_reference_check_catches_a_forward_ref_and_a_duplicate_id():
    assert sg.check_refs(_flats_seq()) == []
    bad = {"$id": "1", "a": {"$ref": "9"}, "b": {"$id": "1"}}
    problems = sg.check_refs(bad)
    assert any("before" in p for p in problems) and any("duplicate" in p for p in problems)


def test_main_writes_the_file_and_reports(tmp_path, capsys):
    src = tmp_path / "flat_sequence.json"
    src.write_text(json.dumps(_flats_seq()))
    dst = tmp_path / "out.json"
    rc = sg.main(["flats", str(src), str(dst), "--filters", "Ha,O-III", "--gain", "26", "--offset", "30"])
    out = capsys.readouterr().out
    assert rc == 0 and dst.exists()
    assert "Ha: auto-exposure flat at gain 26 offset 30" in out and "no trained flat block for: O-III" in out
    written = json.loads(dst.read_text())
    assert sum(1 for n in sg.walk(written) if n.get("$type") == sg.AUTO) == 1
    dsrc = tmp_path / "darks.json"
    dark = _take("2", "DARK", 300.0)
    dark["Binning"] = {"$id": "3", "X": 1, "Y": 1}
    dsrc.write_text(json.dumps({"$id": "1", "Name": "darks_and_bias", "Items": {"$values": [dark]}}))
    ddst = tmp_path / "darks26.json"
    assert sg.main(["darks", str(dsrc), str(ddst), "--gain", "26", "--offset", "30"]) == 0
    assert json.loads(ddst.read_text())["Name"] == "darks_and_bias_gain26"


def test_gain_map_from_config_reads_pairs(monkeypatch):
    from configs import config
    monkeypatch.setattr(config, "data", lambda: {"nina": {"gain": {"Ha": [26, 30], "junk": "x", "O-III": [26, 30]}}})
    assert sg.gain_map_from_config() == {"Ha": (26, 30), "O-III": (26, 30)}
