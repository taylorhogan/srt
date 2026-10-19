"""configs/my.hrz is read two ways; the tree limb must hold in both (2026-10-08).

The planner (astro_dso_visibility.get_horizon_from_azimuth) holds each point's
altitude until the next point. N.I.N.A's "wait until above horizon" and
station_watch interpolate linearly. The limb at az 13-17 was first written as
two points, a plateau to the planner and a thin triangle to N.I.N.A, which put
the horizon at ~40.5 deg at az 15.7: ngc2146 (43 deg) looked clear and was shot
through the branches on 2026-10-07.
"""
import os

HRZ = os.path.join(os.path.dirname(__file__), "..", "configs", "my.hrz")


def _points():
    az, al = [], []
    with open(HRZ) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                az.append(float(parts[0]))
                al.append(float(parts[1]))
    return az, al


def _step(a, az, al):
    for i in range(len(az) - 1):
        if az[i] <= a <= az[i + 1]:
            return al[i]
    return al[-1]


def _linear(a, az, al):
    for i in range(len(az) - 1):
        if az[i] <= a <= az[i + 1]:
            f = (a - az[i]) / (az[i + 1] - az[i])
            return al[i] + f * (al[i + 1] - al[i])
    return al[-1]


def test_tree_limb_holds_for_both_readers():
    az, al = _points()
    for a in (13.5, 14.0, 14.5, 15.0, 15.7, 16.5):
        assert _step(a, az, al) >= 48.0, a
        assert _linear(a, az, al) >= 48.0, a


def test_azimuths_increase():
    az, _ = _points()
    assert az == sorted(az) and len(set(az)) == len(az)


def test_the_planner_reads_the_file_the_way_nina_does():
    """2026-10-19: the planner's staircase put the tree line at 44.1 deg at
    az 28 where N.I.N.A's linear reading has 50.6, so ngc1961 was planned from
    00:00, the generator saw no wait and did not park, and N.I.N.A then waited
    an hour behind the trees. The planner now interpolates linearly too."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from iris_astronomy.horizon import altitude_at as get_horizon_from_azimuth
    az, al = _points()
    for a in (0.0, 5.0, 12.95, 15.0, 20.0, 28.0, 100.0, 200.0, 300.0, 354.0):
        assert abs(get_horizon_from_azimuth(a, az, al) - _linear(a, az, al)) < 1e-9, a
    # Past the last point the profile wraps through 360 to the first.
    assert abs(get_horizon_from_azimuth(357.5, az, al) - (al[-1] + (al[0] - al[-1]) * 2.5 / 5.0)) < 1e-9
