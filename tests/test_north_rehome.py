"""sentry/north_rehome: when Iris North's aim counts as slipped, and when it may be cycled."""
from sentry.north_rehome import describe, drift_verdict, rehome_refusal

SHUT = [[1047.5, 818.0], [1047.0, 891.0], [975.0, 889.0], [975.0, 816.0]]     # side ~73 px
OPEN = [[1175.0, 185.0], [1167.0, 528.5], [797.0, 523.5], [828.0, 182.5]]     # side ~344 px
REF = {"tag_id": 2, "shut": {"markers": {"2": SHUT}}, "open": {"markers": {"2": OPEN}}}


def _frames(side, off_shut, off_open, n=3):
    return [{"side_px": side, "off_shut_px": off_shut, "off_open_px": off_open} for _ in range(n)]


def test_normal_boot_scatter_is_not_drift():
    """42 px was the post-boot aim on 2026-10-05: inside DRIFT_PX."""
    assert drift_verdict(_frames(72.0, 42.0, 625.0), REF) == {"state": "ok", "end": "shut", "offset_px": 42.0}


def test_the_september_slip_is_drift_at_either_end():
    assert drift_verdict(_frames(72.0, 94.0, 557.0), REF)["state"] == "drift"
    assert drift_verdict(_frames(347.0, 640.0, 99.0), REF) == {"state": "drift", "end": "open", "offset_px": 99.0}
    # past the 100 px tolerance the read is UNKNOWN, but the tag size still says "at an end"
    assert drift_verdict(_frames(72.0, 140.0, 520.0), REF)["state"] == "drift"


def test_a_roof_part_way_is_unclear_not_drift():
    """Mid-travel the tag is neither 72 nor 344 px: that is the roof, not the camera."""
    assert drift_verdict(_frames(180.0, 300.0, 330.0), REF)["state"] == "unclear"
    assert drift_verdict([{"camera": True, "tag": False}], REF)["state"] == "no_tag"


def test_cycle_only_at_rest_and_unknown_refuses():
    ok = dict(night=False, nina_running=False, imaging_state="NONE", scheduler_state="WAITING_FOR_NOON")
    assert rehome_refusal(**ok) is None
    for change in ({"night": None}, {"nina_running": True}, {"imaging_state": "IN_MAIN"},
                   {"scheduler_state": "IMAGING"}, {"scheduler_state": None}):
        assert rehome_refusal(**dict(ok, **change))
    assert rehome_refusal(**ok, roof_moving=True) == "the roof is moving"


def test_describe_says_what_happened():
    before = {"state": "drift", "end": "shut", "offset_px": 94.0}
    good = {"plug": "Iris north camera", "ok": True, "boot_s": 27,
            "after": {"state": "ok", "end": "shut", "offset_px": 8.7}}
    assert describe(before, good) == ("Iris North had slipped 94.0 px off its shut reference; "
                                      "power-cycled 'Iris north camera' (back in 27 s) and re-homed: "
                                      "now 8.7 px off shut")
    assert "FAILED: plug did NOT come back ON" in describe(
        before, {"plug": "x", "error": "plug did NOT come back ON -- Iris North is unpowered"})
