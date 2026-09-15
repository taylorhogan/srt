"""Focus seeds: per-filter position-vs-temperature from N.I.N.A's autofocus
reports (fits_processing/focus_model.py) and the MoveFocuserByTemperature
the generator writes before every SmartExposure block. Pure stdlib fixtures:
CI has pytest only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fits_processing import focus_model as fm  # noqa: E402
from nina_gen import nina_sequence_gen as g  # noqa: E402

NS = "NINA.Sequencer"


def _report(filt, pos, temp, r2=0.95):
    return {"filter": fm._norm(filt), "position": pos, "temp": temp, "r2": r2, "when": "t"}


def test_fit_recovers_slope_and_intercept_and_rejects_failed_runs():
    reps = [_report("Ha", 71000 - 50 * t, t) for t in (0, 5, 10, 15, 20)]
    reps.append(_report("Ha", 60000, 10, r2=0.3))      # a failed run: ignored
    m = fm.fit_models(reps)
    e = m["filters"]["HA"]
    assert e["fitted"] and e["n"] == 5
    assert abs(e["slope"] + 50) < 1e-6 and abs(e["intercept"] - 71000) < 1e-3
    assert abs(fm.predict(m, "ha", 12.0) - (71000 - 600)) < 1e-3


def test_outlier_is_clipped_once():
    reps = [_report("L", 72000 - 80 * t, t) for t in (0, 4, 8, 12, 16, 20)]
    reps.append(_report("L", 72000 - 80 * 10 + 3000, 10))   # one wild accepted run
    m = fm.fit_models(reps)
    e = m["filters"]["L"]
    assert e.get("n_clipped") == 1
    assert abs(e["slope"] + 80) < 1.0


def test_too_few_points_fall_back_to_the_pooled_slope():
    reps = [_report("L", 72000 - 80 * t, t) for t in (0, 5, 10, 15, 20)]
    reps += [_report("S-II", 70500, 16.0), _report("S-II", 70300, 18.0)]   # n=2, span 2 C
    m = fm.fit_models(reps)
    s2 = m["filters"]["S-II"]
    assert not s2["fitted"]
    assert abs(s2["slope"] + 80) < 1.0                       # L's slope, pooled
    assert abs(fm.predict(m, "S-II", 17.0) - 70400) < 1.0    # through its own median


def test_no_fitted_filter_uses_the_default_slope():
    m = fm.fit_models([_report("B", 71000, 10.0), _report("B", 71100, 11.0)])
    assert m["filters"]["B"]["slope"] == fm.DEFAULT_SLOPE


def test_save_load_roundtrip_and_describe(tmp_path):
    m = fm.fit_models([_report("Ha", 71000 - 50 * t, t) for t in (0, 5, 10, 15)])
    p = fm.save_model(m, str(tmp_path / "focus_model.json"))
    assert fm.load_model(str(p))["filters"]["HA"]["fitted"]
    assert "HA" in fm.describe(m)
    assert fm.describe(None) == "no focus model"


# ---- generator -------------------------------------------------------------

def _se(idn, filt):
    return {"$id": str(idn), "$type": f"{NS}.SequenceItem.Imaging.SmartExposure, {NS}",
            "Items": {"$id": str(idn + 1), "$type": "x", "$values": [
                {"$id": str(idn + 2), "$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
                 "Filter": filt},
                {"$id": str(idn + 3), "$type": f"{NS}.SequenceItem.Imaging.TakeExposure, {NS}",
                 "ExposureTime": 300.0}]},
            "Parent": {"$ref": "30"}}


def _container():
    shared = {"$id": "5", "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
              "_name": "L", "_position": 0}
    return {"$id": "30", "$type": f"{NS}.Container.SequentialContainer, {NS}", "Name": "IMAGING",
            "Shared": shared,
            "Items": {"$id": "31", "$type": "x", "$values": [
                _se(40, {"$ref": "5"}),                                   # block 0 shares L by $ref
                _se(50, {"$id": "55", "$type": "FilterInfo", "_name": "Ha", "_position": 4}),
                _se(60, {"$id": "65", "$type": "FilterInfo", "_name": "S-II", "_position": 6}),
            ]}}


def _model():
    return fm.fit_models([_report("L", 72000 - 80 * t, t) for t in (0, 5, 10, 15)]
                         + [_report("Ha", 70600 - 45 * t, t) for t in (0, 5, 10, 15)])


def test_seed_inserted_before_each_known_block_and_unknown_left_alone():
    c = _container()
    seeded = g.seed_focus(c, _model(), [100])
    vals = c["Items"]["$values"]
    types = [g._short_type(v) for v in vals]
    assert types == ["MoveFocuserByTemperature", "SmartExposure",      # L via $ref
                     "MoveFocuserByTemperature", "SmartExposure",      # Ha
                     "SmartExposure"]                                  # S-II: no model
    assert set(seeded) == {"L", "Ha"}
    seed = vals[0]
    assert seed["$type"] == g.FOCUS_SEED_TYPE and seed["Absolute"] is True
    assert seed["Parent"] == {"$ref": "30"}
    assert seed["Slope"] == -80.0 and seed["Intercept"] == 72000.0
    assert vals[2]["Slope"] == -45.0 and vals[2]["Intercept"] == 70600.0
    assert vals[0]["$id"] == "100" and vals[2]["$id"] == "101"       # fresh, unique ids


def test_no_model_means_untouched_sequence():
    c = _container()
    before = json.dumps(c, sort_keys=True)
    assert g.seed_focus(c, None, [100]) == {}
    assert json.dumps(c, sort_keys=True) == before


def test_config_off_disables_seeding(monkeypatch):
    class _Cfg:
        @staticmethod
        def data():
            return {"nina": {"focus_seed": False}}
    import configs
    monkeypatch.setitem(sys.modules, "configs.config", _Cfg)
    monkeypatch.setattr(configs, "config", _Cfg, raising=False)
    assert g._focus_model() is None


def test_zeroed_spare_block_gets_no_seed():
    """The 4-block template's fourth block is usually zeroed by the plan; a
    seed in front of it is a focuser move for nothing (seen 2026-09-13:
    'seed -> O-III x0' at the end of the narrowband slot)."""
    c = _container()
    spare = c["Items"]["$values"][1]                     # the Ha block
    spare["Conditions"] = {"$values": [
        {"$type": f"{NS}.Conditions.LoopCondition, {NS}", "Iterations": 0}]}
    g.seed_focus(c, _model(), [100])
    types = [g._short_type(v) for v in c["Items"]["$values"]]
    assert types == ["MoveFocuserByTemperature", "SmartExposure",   # L
                     "SmartExposure",                               # Ha x0: no seed
                     "SmartExposure"]                               # S-II: no model


def test_seed_never_touches_the_inside_of_a_smart_exposure():
    """2026-09-14: a relative MoveFocuserByTemperature written INSIDE every
    seeded SmartExposure ("tracking between subs") made N.I.N.A's
    SmartExposure.Validate() throw NullReferenceException 6104 times and
    skip every block; the run took no exposures until it was relaunched on
    a hand-stripped sequence. Seeds go before the block; the block's own
    items stay exactly [SwitchFilter, TakeExposure], and the container's
    autofocus temperature trigger is left as the template wrote it."""
    c = _container()
    c["Triggers"] = {"$values": [
        {"$id": "90", "$type": f"{NS}.Trigger.Autofocus.AutofocusAfterTemperatureChangeTrigger, {NS}",
         "Amount": 2.0}]}
    g.seed_focus(c, _model(), [100])
    blocks = [v for v in c["Items"]["$values"] if g._short_type(v) == "SmartExposure"]
    assert len(blocks) == 3
    for blk in blocks:
        assert [g._short_type(i) for i in blk["Items"]["$values"]] == ["SwitchFilter", "TakeExposure"]
    assert c["Triggers"]["$values"][0]["Amount"] == 2.0
    assert not hasattr(g, "_insert_tracking") and not hasattr(g, "_track_settings")
