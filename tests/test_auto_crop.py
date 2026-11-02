"""stacking/auto_crop: the subject by type, then the box (shadow, 2026-11-02).

Synthetic fields in 0..1 the size the cropper works at. Skips without
numpy/scipy (CI). The real-picture checks (IC 410, Melotte 15 and the
contact sheet of every composite on the Spark) are in the module notes.
"""
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from stacking import auto_crop as ac  # noqa: E402


def _sky(h=400, w=600, seed=0, level=0.05):
    rng = np.random.default_rng(seed)
    return np.full((h, w), level, np.float32) + rng.normal(0, 0.004, (h, w)).astype(np.float32), rng


def _stars(a, rng, n=150, peak=0.5, where=None):
    h, w = a.shape
    for _ in range(n):
        if where:
            y = rng.integers(where[1], where[3]); x = rng.integers(where[0], where[2])
        else:
            y = rng.integers(2, h - 2); x = rng.integers(2, w - 2)
        a[y - 1:y + 2, x - 1:x + 2] += peak * 0.3
        a[y, x] += peak
    return a


def _blob(a, cy, cx, sy, sx, amp):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a += amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    return a


def _rgb(a):
    return np.clip(np.dstack([a, a, a]), 0, 1)


def _contains(box, inner):
    return box[0] <= inner[0] and box[1] <= inner[1] and box[2] >= inner[2] and box[3] >= inner[3]


def test_a_nebula_off_centre_gets_a_box_around_it_from_the_menu():
    a, rng = _sky()
    _blob(a, 150, 420, 45, 70, 0.5)                 # the nebula, upper right
    _stars(a, rng)
    rec = ac.propose(_rgb(a), "nebula")
    assert not rec["full"]
    x0, y0, x1, y1 = rec["box"]
    assert _contains(rec["box"], (420 - 70, 150 - 45, 420 + 70, 150 + 45))
    assert rec["aspect"].split("~")[0] in dict(ac.ASPECTS)
    assert 0 <= x0 < x1 <= 600 and 0 <= y0 < y1 <= 400
    assert (x1 - x0) * (y1 - y0) < 0.8 * 600 * 400  # it IS a crop
    assert rec["coverage"] > 0.9


def test_a_small_galaxy_keeps_its_context():
    """A box 2x the galaxy would be a postage stamp: the floor gives it sky."""
    a, rng = _sky()
    _blob(a, 200, 300, 10, 16, 0.6)
    _stars(a, rng)
    rec = ac.propose(_rgb(a), "galaxy")
    assert not rec["full"]
    x0, y0, x1, y1 = rec["box"]
    assert x1 - x0 >= ac.MIN_BOX_SHARE["galaxy"] * 600 * 0.95
    assert _contains(rec["box"], (300 - 32, 200 - 20, 300 + 32, 200 + 20))


def test_a_star_cluster_is_found_by_density_without_any_nebula():
    a, rng = _sky()
    _stars(a, rng, n=60, peak=0.4)                                  # the field
    _stars(a, rng, n=120, peak=0.4, where=(380, 80, 520, 200))      # the cluster
    rec = ac.propose(_rgb(a), "cluster")
    assert not rec["full"], rec["why"]
    assert rec["subject"]["source"] == "stars"
    sx0, sy0, sx1, sy1 = rec["subject"]["bbox"]
    assert 340 <= sx0 and sx1 <= 560 and 40 <= sy0 and sy1 <= 240
    assert _contains(rec["box"], (380, 80, 520, 200))


def test_a_cluster_in_a_nebula_is_a_picture_of_both():
    a, rng = _sky()
    _stars(a, rng, n=60, peak=0.4)
    _stars(a, rng, n=120, peak=0.4, where=(380, 80, 520, 200))
    _blob(a, 260, 300, 40, 60, 0.5)                                 # the nebula below it
    rec = ac.propose(_rgb(a), "cluster")
    assert rec["subject"]["source"] == "stars+nebula"
    assert _contains(rec["box"], (380, 80, 520, 200)) and _contains(rec["box"], (240, 220, 360, 300))


def test_a_galaxy_cluster_is_placed_by_its_members():
    a, rng = _sky()
    _stars(a, rng, n=80, peak=0.5)
    for i in range(14):                                             # small faint members, lower left
        cy, cx = 220 + (i % 4) * 40, 80 + (i // 4) * 55 + (i % 3) * 10
        _blob(a, cy, cx, 2, 3, 0.12)
    rec = ac.propose(_rgb(a), "galaxy_cluster")
    assert not rec["full"], rec["why"]
    assert rec["subject"]["source"] == "members"
    assert _contains(rec["box"], (80, 220, 270, 340))


def test_a_subject_that_fills_the_field_is_not_cropped():
    a, rng = _sky()
    a[20:380, 30:570] += 0.5                                       # a plateau over the whole field
    rec = ac.propose(_rgb(a), "nebula")
    assert rec["full"] and rec["box"] == [0, 0, 600, 400]


def test_an_empty_field_is_not_cropped():
    a, rng = _sky()
    _stars(a, rng, n=40)
    rec = ac.propose(_rgb(a), "nebula")
    assert rec["full"]


def test_a_tiny_core_does_not_flood_to_the_background():
    """Faint-first flooded M76's 2 % core to 16 % of background."""
    a, rng = _sky()
    _blob(a, 200, 300, 8, 10, 0.7)                                  # the planetary
    _blob(a, 200, 300, 120, 180, 0.03)                              # a faint, wide glow
    rec = ac.propose(_rgb(a), "nebula")
    assert rec["subject"]["area"] < 0.05


def test_a_bright_star_at_an_edge_moves_the_edge():
    a, rng = _sky()
    _blob(a, 200, 300, 40, 60, 0.5)
    rec0 = ac.propose(_rgb(a.copy()), "nebula")
    x0 = rec0["box"][0]
    b = a.copy()
    b[195:206, x0 - 1:x0 + 2] = 1.0                                 # a saturated star on the left edge
    b[200, x0 - 6:x0 + 7] = 1.0
    rec = ac.propose(_rgb(b), "nebula")
    assert abs(rec["box"][0] - x0) > ac.STAR_EDGE * 600 * 0.5


def test_the_box_stays_inside_the_edge_band_keeping_its_aspect():
    a, rng = _sky(h=400, w=600)
    _blob(a, 200, 300, 150, 150, 0.5)                               # tall subject on a wide field
    rec = ac.propose(_rgb(a), "nebula")
    if not rec["full"]:
        x0, y0, x1, y1 = rec["box"]
        edge = int(ac.EDGE * 600)
        assert y0 >= edge and y1 <= 400 - edge and x0 >= edge and x1 <= 600 - edge


def test_the_type_is_read_offline_in_order(tmp_path):
    queue = tmp_path / "q.json"
    queue.write_text(json.dumps([{"dso": "ic410", "obj_type": "nebula"}, {"dso": "ngc7790", "obj_type": "star"}]))
    cat = tmp_path / "c.json"
    cat.write_text(json.dumps({"candidates": [{"name": "m13", "kind": "cluster"}, {"name": "m27", "kind": "pn"}]}))
    cache = tmp_path / "t.json"
    cache.write_text(json.dumps({"abell2218": "ClG", "ngc6946": "H2G", "x": ""}))
    kw = dict(root=str(tmp_path), queue_path=str(queue), catalogue=str(cat), caches=(str(cache),))
    assert ac.type_for("IC 410", **kw) == "nebula"
    assert ac.type_for("ngc7790", **kw) == "cluster"          # a monitor's stars are the subject
    assert ac.type_for("m13", **kw) == "cluster"
    assert ac.type_for("m27", **kw) == "nebula"
    assert ac.type_for("abell2218", **kw) == "galaxy_cluster"
    assert ac.type_for("ngc6946", **kw) == "galaxy"
    assert ac.type_for("nothing", **kw) == "unknown"


def test_scaled_box_follows_save_rgb_s_rule_and_iou_is_sane():
    assert ac.scaled_box([100, 200, 1100, 1200], (4000, 6000), 2400) == [40, 80, 440, 480]
    assert ac.scaled_box([100, 200, 1100, 1200], (1000, 1500), 2400) == [100, 200, 1100, 1200]
    assert ac.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert ac.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_describe_names_the_box_or_the_reason():
    assert "full frame" in ac.describe({"full": True, "why": "no crop: no subject found"})
    line = ac.describe({"full": False, "box": [10, 20, 410, 320], "aspect": "4:3", "why": "nebula subject 20% of the field (diffuse)",
                        "coverage": 0.98, "empty": 0.4})
    assert "4:3 400x300 at (10,20)" in line and "coverage 0.98" in line
