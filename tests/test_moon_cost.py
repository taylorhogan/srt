"""The planner's moon rule (iris_astronomy/moon_cost, 2026-10-25).

Pure arithmetic, no astropy: CI runs it. The numbers are the measured fit's,
so these pin the decisions the planner will make, not just the formulas.
"""
from iris_astronomy import moon_cost as mc

FULL_NEAR = mc.moon_brightness(0.95, 30, 30, 50)       # bright moon close by
FULL_FAR = mc.moon_brightness(0.95, 120, 30, 50)
HALF = mc.moon_brightness(0.50, 90, 30, 50)


def test_no_moon_costs_nothing():
    assert mc.moon_brightness(0.95, 30, -5, 50) == 0.0
    assert mc.plan_worth({"L": 1}, 0.0) == 1.0


def test_brightness_grows_with_phase_and_nearness():
    assert FULL_NEAR > FULL_FAR > HALF > mc.moon_brightness(0.25, 90, 30, 50) > 0


def test_the_measured_worths():
    # scripts/moon_sky_cost.py, 2026-10-25: 95% moon 30 deg away.
    assert round(mc.plan_worth({"L": 1}, FULL_NEAR), 2) == 0.30
    assert round(mc.plan_worth({"O-III": 1}, FULL_NEAR), 2) == 0.69
    assert mc.plan_worth({"Ha": 1}, FULL_NEAR) > 0.75


def test_a_galaxy_waits_and_a_nebula_does_not_under_a_bright_moon():
    galaxy = mc.target_plan({"obj_type": None}, "G")
    nebula = mc.target_plan({"obj_type": "nebula"}, "?")
    assert mc.hour_held("avoid", mc.plan_worth(galaxy, FULL_NEAR), 30, 0.6)
    assert not mc.hour_held("avoid", mc.plan_worth(nebula, FULL_NEAR), 30, 0.6)
    assert not mc.hour_held("avoid", mc.plan_worth(galaxy, HALF), 30, 0.6)       # a half moon is fine


def test_the_entry_plan_wins_over_the_type():
    assert mc.target_plan({"filter_plan": {"Ha": 2, "O-III": 1}}, "G") == {"Ha": 2, "O-III": 1}
    # S-II has no moonlit frames yet and is judged like Ha.
    assert mc.plan_worth({"S-II": 1}, FULL_NEAR) == mc.plan_worth({"Ha": 1}, FULL_NEAR)


def test_modes():
    assert mc.moon_mode({}) == "avoid"
    assert mc.moon_mode({"lightcurve": True}) == "any"          # cadence beats depth
    assert mc.moon_mode({"moon": "dark", "lightcurve": True}) == "dark"
    assert not mc.hour_held("any", 0.1, 60, 0.6)
    assert mc.hour_held("dark", 0.99, 2, 0.6) and not mc.hour_held("dark", 0.2, -1, 0.6)


def test_a_missing_fit_file_falls_back_to_the_measured_values(tmp_path):
    assert mc.load_fit(tmp_path / "none.json") == mc.DEFAULT_FIT
