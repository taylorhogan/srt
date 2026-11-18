"""control/overhead: the aircraft passing over the roof, from the public ADS-B
feed (2026-11-18). Pure geometry and selection with a fake feed: the closest
approach of a projected track, what counts as a pass (airborne, inside the
circle now or about to be), the dedupe cooldown, the carrier and type
spelling, the one-line description, one poll end to end, and the chat
report. Nothing here touches the network.
"""
import math

from control import overhead as oh

SITE = (41.8096, -72.8305)


def _ac(lat, lon, alt=30000, gs=450, track=225, flight="JBU1234", hex_="a1b2c3", t="A320", r="N123JB"):
    return {"hex": hex_, "flight": flight, "r": r, "t": t, "lat": lat, "lon": lon,
            "alt_baro": alt, "gs": gs, "track": track}


def _offset(north_km, east_km):
    lat = SITE[0] + north_km / 6371.0 * 180 / math.pi
    lon = SITE[1] + east_km / (6371.0 * math.cos(math.radians(SITE[0]))) * 180 / math.pi
    return lat, lon


def test_closest_approach_of_a_plane_heading_straight_at_the_site():
    lat, lon = _offset(10.0, 10.0)                         # 14 km to the NE
    d_now, d_min, t = oh.closest_approach(lat, lon, 450, 225, *SITE)   # heading SW: straight in
    assert d_now == pytest_approx(14.14, 0.1)
    assert d_min < 0.2
    assert t is not None and 50 < t < 70                   # 14 km at 450 kt (0.23 km/s) ~ 61 s


def test_closest_approach_behind_and_offset():
    lat, lon = _offset(-10.0, -10.0)                       # SW of the site
    _, d_min, t = oh.closest_approach(lat, lon, 450, 225, *SITE)      # still heading SW: going away
    assert d_min < 0.2 and t < 0                           # the closest point is behind it
    lat, lon = _offset(8.0, 0.0)                           # 8 km north, flying east
    _, d_min, t = oh.closest_approach(lat, lon, 300, 90, *SITE)
    assert d_min == pytest_approx(8.0, 0.1) and abs(t) < 1
    _, d_min, t = oh.closest_approach(lat, lon, None, None, *SITE)    # no track: the distance now
    assert d_min == pytest_approx(8.0, 0.1) and t is None


def test_passes_pick_the_airborne_near_and_coming_and_leave_the_rest():
    far = _ac(*_offset(20, 0), track=90, hex_="far")                 # 20 km N heading E: never near
    coming = _ac(*_offset(10, 10), track=225, hex_="coming")         # 14 km out, 61 s from overhead
    here = _ac(*_offset(1, -1), track=0, gs=120, hex_="here", t="PC12", flight="N88QD")
    ground = _ac(*_offset(0.5, 0.5), alt="ground", hex_="ground")
    nofix = {"hex": "nofix", "alt_baro": 5000}
    long_way = _ac(*_offset(40, 40), track=225, hex_="later")        # 56 km out: 4 min away
    got = oh.passes([far, coming, here, ground, nofix, long_way], *SITE)
    assert [g["hex"] for g in got] == ["coming", "here"]             # nearest closest approach first
    assert got[1]["elevation_deg"] > 80                              # 30,000 ft over 1.4 km: nearly overhead
    assert abs(got[0]["bearing_from_site"] - 45) < 1


def test_due_dedupes_per_aircraft_with_a_cooldown_and_prune_forgets():
    state = {}
    assert oh.due(state, "abc", now=1000.0)
    assert not oh.due(state, "abc", now=1000.0 + 600)
    assert oh.due(state, "def", now=1000.0 + 600)
    assert oh.due(state, "abc", now=1000.0 + oh.COOLDOWN_S + 1)
    assert oh.prune({"old": 0.0, "new": 1000.0}, now=1000.0 + 6 * 3600) == {"new": 1000.0}


def test_carrier_and_type_spelling():
    assert oh.carrier("JBU1234") == ("JBU1234", "JetBlue")
    assert oh.carrier("AAL1062 ") == ("AAL1062", "American")
    assert oh.carrier("N88QD") == ("N88QD", "private")
    assert oh.carrier("", "N705JB") == ("N705JB", None)
    assert oh.carrier("ZZZ999") == ("ZZZ999", None)                 # unknown prefix: shown as is
    assert oh.type_name("A21N") == "Airbus A321neo"
    assert oh.type_name("XYZ9") == "XYZ9" and oh.type_name(None) == "aircraft"
    assert oh.compass(0) == "N" and oh.compass(225) == "SW" and oh.compass(350) == "N"


def test_describe_is_one_line_with_the_facts():
    rec = oh.passes([_ac(*_offset(10, 10), track=225)], *SITE)[0]
    line = oh.describe(rec, when="21:04")
    assert line.startswith("21:04 Overhead: JBU1234 JetBlue, Airbus A320 at 30,000 ft heading SW at 450 kt")
    assert "deg up to the NE" in line and "closest in" in line and "km" in line
    assert "\n" not in line
    em = dict(rec, emergency="general", squawk="7700")
    assert "squawking 7700" in oh.describe(em)


def test_poll_once_posts_each_pass_once_and_is_silent_on_a_dead_feed():
    cfg = {"location": {"latitude": SITE[0], "longitude": SITE[1]}, "overhead": {"cooldown_min": 30}}
    feed = {"ac": [_ac(*_offset(10, 10), track=225, hex_="x1"), _ac(*_offset(20, 0), track=90, hex_="x2")]}
    posted = []
    state = {}
    lines = oh.poll_once(cfg, state, posted.append, now=1000.0, get=lambda url: feed)
    assert len(lines) == 1 and posted == lines and "JBU1234" in lines[0]
    again = oh.poll_once(cfg, state, posted.append, now=1060.0, get=lambda url: feed)
    assert again == [] and len(posted) == 1                        # the same plane a minute later: no line

    def dead(url):
        raise OSError("no route")
    assert oh.poll_once(cfg, state, posted.append, now=2000.0, get=dead) == []
    assert oh.poll_once({**cfg, "overhead": {"enabled": False}}, state, posted.append, get=lambda u: feed) == []


def test_nearby_report_lists_the_airborne_nearest_first_and_marks_the_passes():
    cfg = {"location": {"latitude": SITE[0], "longitude": SITE[1]}}
    feed = {"ac": [_ac(*_offset(20, 0), track=90, hex_="far", flight="DAL22", t="B738"),
                   _ac(*_offset(10, 10), track=225, hex_="coming"),
                   _ac(*_offset(0.5, 0.5), alt="ground", hex_="ground")]}
    no_table = lambda url: {"response": "unknown callsign"}            # never the network in a test
    text = oh.nearby_report(cfg, get=lambda url: feed, routes={}, route_get=no_table)
    lines = text.splitlines()
    assert lines[0].startswith("Overhead now (2 airborne")
    assert "JBU1234 JetBlue" in lines[1] and "passing over" in lines[1]
    assert "DAL22 Delta, Boeing 737-800" in lines[2] and "passing over" not in lines[2]
    assert "did not answer" in oh.nearby_report(cfg, get=lambda url: {"ac": []}, route_get=no_table)


def pytest_approx(value, tol):
    import pytest
    return pytest.approx(value, abs=tol)


def test_routes_come_from_the_table_for_airline_callsigns_only_and_are_cached():
    calls = []

    def table(url):
        calls.append(url)
        cs = url.rsplit("/", 1)[-1]
        if cs == "JBU1234":
            return {"response": {"flightroute": {"airline": {"name": "JetBlue Airways"},
                                                 "origin": {"iata_code": "BOS", "municipality": "Boston"},
                                                 "destination": {"iata_code": "FLL", "municipality": "Fort Lauderdale"}}}}
        return {"response": "unknown callsign"}
    routes = {}
    r = oh.route_for("JBU1234", routes, now=1000.0, get=table)
    assert r["origin"] == "BOS" and r["destination_city"] == "Fort Lauderdale" and r["airline"] == "JetBlue Airways"
    assert oh.route_text(r) == "Boston to Fort Lauderdale"
    assert oh.route_for("JBU1234", routes, now=2000.0, get=table) == r and len(calls) == 1   # cached a day
    assert oh.route_for("N88QD", routes, now=1000.0, get=table) is None and len(calls) == 1   # a registration: no lookup
    assert oh.route_for("ZZZ999", routes, now=1000.0, get=table) is None and len(calls) == 2
    assert oh.route_for("ZZZ999", routes, now=1000.0 + 600, get=table) is None and len(calls) == 2   # a miss cached an hour
    assert oh.route_for("ZZZ999", routes, now=1000.0 + 7200, get=table) is None and len(calls) == 3

    def broken(url):
        raise OSError("down")
    assert oh.route_for("DAL401", {}, get=broken) is None                                   # never raises
    assert oh.prune_routes({"OLD": {"at": 0.0}, "NEW": {"at": 1000.0}}, now=1000.0 + oh.ROUTE_TTL_S) == {"NEW": {"at": 1000.0}}


def test_the_line_and_the_poll_carry_the_route():
    rec = oh.passes([_ac(*_offset(10, 10), track=225)], *SITE)[0]
    route = {"origin": "BOS", "origin_city": "Boston", "destination": "FLL", "destination_city": "Fort Lauderdale",
             "airline": "JetBlue Airways"}
    line = oh.describe(rec, route=route)
    assert line.startswith("Overhead: JBU1234 JetBlue Airways, Airbus A320, Boston to Fort Lauderdale, at 30,000 ft")
    assert "Boston" not in oh.describe(rec)                                               # no route: the old line
    cfg = {"location": {"latitude": SITE[0], "longitude": SITE[1]}}
    feed = {"ac": [_ac(*_offset(10, 10), track=225, hex_="x1")]}
    table = lambda url: {"response": {"flightroute": {"airline": {"name": "JetBlue Airways"},
                                                      "origin": {"iata_code": "BOS", "municipality": "Boston"},
                                                      "destination": {"iata_code": "FLL", "municipality": "Fort Lauderdale"}}}}
    posted, routes = [], {}
    oh.poll_once(cfg, {}, posted.append, now=1000.0, get=lambda url: feed, routes=routes, route_get=table)
    assert "Boston to Fort Lauderdale" in posted[0] and "JBU1234" in routes
    off = {**cfg, "overhead": {"routes": False}}
    posted2 = []
    oh.poll_once(off, {}, posted2.append, now=1000.0, get=lambda url: feed, routes={}, route_get=table)
    assert "Boston" not in posted2[0]
    text = oh.nearby_report(cfg, get=lambda url: feed, routes={}, route_get=table)
    assert "JBU1234 JetBlue Airways" in text and "Boston to Fort Lauderdale" in text
