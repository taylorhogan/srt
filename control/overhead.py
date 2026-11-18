"""What is flying over the observatory, from the public ADS-B feed (2026-11-18).

Every airliner and most light aircraft broadcast ADS-B (position, altitude,
speed, track, identity) in the clear, and adsb.lol serves the aggregate as
JSON with no key: `GET /v2/lat/<lat>/lon/<lon>/dist/<nm>` returns every
aircraft its receivers hear within that radius, each with its distance
(`dst`, nautical miles) and bearing (`dir`) from the point. The owner asked
for the passes over the roof to be logged in the web chat, around the clock
("Around the clock", 2026-11-18); matching a pass to a streak in a frame
is a later step and needs nothing more than the same geometry against the
frame's DATE-OBS.

This module is the pure part, shared by the Spark's poller
(`scripts/overhead_watch.py`, cron every minute) and the chat's `overhead`
command on the PC:

- `closest_approach` projects the aircraft's straight-line track and gives
  the closest it will come to the site in km and when (seconds, negative
  when that point is behind it).
- `passes` picks the aircraft that are airborne and either inside
  `radius_km` now or will be, or just were, within `horizon_s`: at a
  one-minute poll a jet covers 8 km, so a 5 km circle alone would miss
  half the passes.
- `due` dedupes by the aircraft's hex code with a cooldown, so a circling
  plane or a slow one is one line, not thirty.
- `describe` is the one line the chat gets: carrier from the callsign's
  ICAO prefix (CARRIERS), the type spelled out (TYPES), altitude, direction
  of travel, how high up it stands from the roof, and where.

Nothing here moves anything or gates anything; a failed fetch is an empty
list, and the callers keep silent on it.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

RADIUS_KM = 5.0          # a pass is inside this horizontal circle: ~45 deg up at airliner height
QUERY_NM = 15.0          # ask the feed for this radius: three polls of warning for a jet
HORIZON_S = 90.0         # a closest approach this near in time, either side, counts now
COOLDOWN_S = 1800.0      # one line per aircraft per half hour
SOURCE = "https://api.adsb.lol/v2"
KM_PER_NM = 1.852
FT_PER_M = 3.28084

CARRIERS = {
    "AAL": "American", "DAL": "Delta", "UAL": "United", "JBU": "JetBlue", "SWA": "Southwest",
    "ASA": "Alaska", "FFT": "Frontier", "NKS": "Spirit", "AAY": "Allegiant", "SCX": "Sun Country",
    "MXY": "Breeze", "HAL": "Hawaiian", "RPA": "Republic", "ENY": "Envoy", "PDT": "Piedmont",
    "JIA": "PSA", "SKW": "SkyWest", "EDV": "Endeavor", "ASH": "Mesa", "GJS": "GoJet",
    "ACA": "Air Canada", "JZA": "Air Canada Express", "WJA": "WestJet", "POE": "Porter",
    "BAW": "British Airways", "VIR": "Virgin Atlantic", "AFR": "Air France", "DLH": "Lufthansa",
    "KLM": "KLM", "EIN": "Aer Lingus", "ICE": "Icelandair", "SAS": "SAS", "FIN": "Finnair",
    "SWR": "Swiss", "AUA": "Austrian", "IBE": "Iberia", "TAP": "TAP Portugal", "ITY": "ITA",
    "LOT": "LOT Polish", "THY": "Turkish", "UAE": "Emirates", "QTR": "Qatar", "ETD": "Etihad",
    "ELY": "El Al", "NAX": "Norwegian", "NOZ": "Norse", "EJU": "easyJet", "RYR": "Ryanair",
    "CPA": "Cathay Pacific", "ANA": "ANA", "JAL": "Japan Airlines", "KAL": "Korean Air",
    "AAR": "Asiana", "CAL": "China Airlines", "EVA": "EVA Air", "SIA": "Singapore",
    "QFA": "Qantas", "ANZ": "Air New Zealand", "AMX": "Aeromexico", "VOI": "Volaris",
    "CMP": "Copa", "AVA": "Avianca", "LAN": "LATAM", "TAM": "LATAM Brasil",
    "FDX": "FedEx", "UPS": "UPS", "GTI": "Atlas Air", "ABX": "ABX Air", "ATN": "Air Transport",
    "CKS": "Kalitta", "GEC": "Lufthansa Cargo", "CLX": "Cargolux", "BOX": "AeroLogic",
    "EJA": "NetJets", "LXJ": "Flexjet", "XOJ": "XOJet", "VJA": "VistaJet", "JTL": "Jet Linx",
    "DCM": "Wheels Up", "GAJ": "Gama", "TWY": "Sunwing", "CJT": "Cargojet",
    "RCH": "US Air Force (AMC)", "CNV": "US Navy", "PAT": "US Army", "CFC": "Canadian Forces",
    "N": "private",
}

TYPES = {
    "A319": "Airbus A319", "A320": "Airbus A320", "A321": "Airbus A321", "A20N": "Airbus A320neo",
    "A21N": "Airbus A321neo", "A306": "Airbus A300", "A310": "Airbus A310", "A332": "Airbus A330-200",
    "A333": "Airbus A330-300", "A339": "Airbus A330neo", "A343": "Airbus A340-300",
    "A346": "Airbus A340-600", "A359": "Airbus A350-900", "A35K": "Airbus A350-1000",
    "A388": "Airbus A380", "BCS1": "Airbus A220-100", "BCS3": "Airbus A220-300",
    "B712": "Boeing 717", "B733": "Boeing 737-300", "B737": "Boeing 737-700", "B738": "Boeing 737-800",
    "B739": "Boeing 737-900", "B38M": "Boeing 737 MAX 8", "B39M": "Boeing 737 MAX 9",
    "B3XM": "Boeing 737 MAX 10", "B744": "Boeing 747-400", "B748": "Boeing 747-8",
    "B752": "Boeing 757-200", "B753": "Boeing 757-300", "B762": "Boeing 767-200",
    "B763": "Boeing 767-300", "B764": "Boeing 767-400", "B772": "Boeing 777-200",
    "B77L": "Boeing 777-200LR", "B773": "Boeing 777-300", "B77W": "Boeing 777-300ER",
    "B788": "Boeing 787-8", "B789": "Boeing 787-9", "B78X": "Boeing 787-10", "MD11": "MD-11",
    "MD82": "MD-82", "MD83": "MD-83", "MD88": "MD-88", "MD90": "MD-90",
    "E135": "Embraer ERJ-135", "E145": "Embraer ERJ-145", "E170": "Embraer E170",
    "E175": "Embraer E175", "E75L": "Embraer E175", "E75S": "Embraer E175", "E190": "Embraer E190",
    "E195": "Embraer E195", "E290": "Embraer E190-E2", "E295": "Embraer E195-E2",
    "CRJ2": "CRJ-200", "CRJ7": "CRJ-700", "CRJ9": "CRJ-900", "CRJX": "CRJ-1000",
    "DH8A": "Dash 8-100", "DH8B": "Dash 8-200", "DH8C": "Dash 8-300", "DH8D": "Dash 8-400",
    "AT43": "ATR 42", "AT72": "ATR 72", "AT75": "ATR 72-500", "AT76": "ATR 72-600",
    "SF34": "Saab 340", "SB20": "Saab 2000", "B190": "Beech 1900", "BE20": "King Air 200",
    "BE9L": "King Air 90", "B350": "King Air 350", "PC12": "Pilatus PC-12", "PC24": "Pilatus PC-24",
    "TBM7": "TBM 700", "TBM8": "TBM 850", "TBM9": "TBM 900", "C208": "Cessna Caravan",
    "C172": "Cessna 172", "C182": "Cessna 182", "C206": "Cessna 206", "C210": "Cessna 210",
    "P28A": "Piper Cherokee", "P28R": "Piper Arrow", "PA31": "Piper Navajo", "PA34": "Piper Seneca",
    "PA46": "Piper Malibu", "M20P": "Mooney M20", "SR20": "Cirrus SR20", "SR22": "Cirrus SR22",
    "SF50": "Cirrus Vision Jet", "DA40": "Diamond DA40", "DA42": "Diamond DA42", "DA62": "Diamond DA62",
    "C25A": "Citation CJ2", "C25B": "Citation CJ3", "C25C": "Citation CJ4", "C510": "Citation Mustang",
    "C525": "Citation CJ1", "C550": "Citation II", "C560": "Citation V", "C56X": "Citation Excel",
    "C650": "Citation III", "C680": "Citation Sovereign", "C68A": "Citation Latitude",
    "C700": "Citation Longitude", "C750": "Citation X", "LJ35": "Learjet 35", "LJ45": "Learjet 45",
    "LJ60": "Learjet 60", "LJ75": "Learjet 75", "E50P": "Phenom 100", "E55P": "Phenom 300",
    "E545": "Praetor 500", "E550": "Praetor 600", "CL30": "Challenger 300", "CL35": "Challenger 350",
    "CL60": "Challenger 600", "GL5T": "Global 5000", "GL7T": "Global 7500", "GLEX": "Global Express",
    "GLF4": "Gulfstream IV", "GLF5": "Gulfstream V", "GLF6": "Gulfstream G650", "G280": "Gulfstream G280",
    "GA5C": "Gulfstream G500", "GA6C": "Gulfstream G600", "GA7C": "Gulfstream G700",
    "F2TH": "Falcon 2000", "F900": "Falcon 900", "FA7X": "Falcon 7X", "FA8X": "Falcon 8X",
    "FA50": "Falcon 50", "H25B": "Hawker 800", "HA4T": "Hawker 4000", "PRM1": "Premier 1",
    "EC35": "Eurocopter EC135", "EC45": "Eurocopter EC145", "H60": "Black Hawk", "UH1": "Huey",
    "A139": "AgustaWestland AW139", "B06": "Bell 206", "B407": "Bell 407", "B429": "Bell 429",
    "R44": "Robinson R44", "R66": "Robinson R66", "S76": "Sikorsky S-76", "S92": "Sikorsky S-92",
    "C130": "C-130 Hercules", "C30J": "C-130J Hercules", "C17": "C-17 Globemaster",
    "K35R": "KC-135 Stratotanker", "KC46": "KC-46 Pegasus", "P8": "P-8 Poseidon", "E3TF": "E-3 Sentry",
}

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg: float) -> str:
    """A 16-point compass name for a bearing in degrees."""
    return _COMPASS[int((float(deg) % 360.0) / 22.5 + 0.5) % 16]


def carrier(flight: Optional[str], reg: Optional[str] = None) -> tuple[str, Optional[str]]:
    """(callsign as shown, carrier name or None) from the ADS-B `flight` field.

    A callsign of three letters and digits is an airline flight (CARRIERS by
    the ICAO prefix); a bare registration (N12345) is a private or corporate
    aircraft; nothing at all shows the registration when there is one.
    """
    cs = (flight or "").strip()
    if not cs:
        return ((reg or "").strip() or "?", None)
    pre = cs[:3].upper()
    if pre.isalpha() and len(cs) > 3 and any(ch.isdigit() for ch in cs[3:]):
        return cs, CARRIERS.get(pre)
    if cs[0].upper() == "N" and cs[1:2].isdigit():
        return cs, CARRIERS["N"]
    return cs, None


def type_name(code: Optional[str]) -> str:
    c = (code or "").strip().upper()
    return TYPES.get(c, c or "aircraft")


def _local_xy(lat: float, lon: float, site_lat: float, site_lon: float) -> tuple[float, float]:
    """East, north offsets in km of (lat, lon) from the site, flat-earth
    (fine over the 30 km the feed is asked for)."""
    x = math.radians(lon - site_lon) * math.cos(math.radians(site_lat)) * 6371.0
    y = math.radians(lat - site_lat) * 6371.0
    return x, y


def closest_approach(lat: float, lon: float, gs_kt: Optional[float], track_deg: Optional[float],
                     site_lat: float, site_lon: float) -> tuple[float, float, Optional[float]]:
    """(distance now km, closest distance km, seconds until closest or None).

    The aircraft is projected along its track at its ground speed; the
    closest point is where that line passes the site. Seconds are negative
    when the closest point is already behind it. With no speed or track the
    closest distance is the distance now and the time None.
    """
    x, y = _local_xy(lat, lon, site_lat, site_lon)
    d_now = math.hypot(x, y)
    if gs_kt is None or track_deg is None or float(gs_kt) <= 1.0:
        return d_now, d_now, None
    v = float(gs_kt) * KM_PER_NM / 3600.0                      # km/s
    tr = math.radians(float(track_deg))
    vx, vy = v * math.sin(tr), v * math.cos(tr)                 # track is from north, clockwise
    # position p + v t; closest where (p + v t) . v = 0
    t = -(x * vx + y * vy) / (vx * vx + vy * vy)
    d_min = math.hypot(x + vx * t, y + vy * t)
    return d_now, d_min, t


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def passes(aircraft: list, site_lat: float, site_lon: float, radius_km: float = RADIUS_KM,
           horizon_s: float = HORIZON_S) -> list[dict]:
    """The aircraft from an adsb.lol `ac` list that are passing over now.

    Airborne (a barometric altitude that is a number, not "ground"), with a
    position, and either inside *radius_km* now or with a closest approach
    inside it within *horizon_s* seconds either way. Each returned record
    carries the geometry: d_now, d_min, t_min (s), bearing_from_site (deg),
    elevation_deg (how far up it stands now), and the raw fields it was
    judged on.
    """
    out = []
    for a in aircraft or []:
        lat, lon = _num(a.get("lat")), _num(a.get("lon"))
        alt_ft = _num(a.get("alt_baro"))
        if alt_ft is None:
            alt_ft = _num(a.get("alt_geom"))
        if lat is None or lon is None or alt_ft is None or alt_ft <= 0:
            continue                                            # on the ground, or no fix
        d_now, d_min, t_min = closest_approach(lat, lon, _num(a.get("gs")), _num(a.get("track")),
                                               site_lat, site_lon)
        inside_now = d_now <= radius_km
        coming = t_min is not None and d_min <= radius_km and abs(t_min) <= horizon_s
        if not (inside_now or coming):
            continue
        x, y = _local_xy(lat, lon, site_lat, site_lon)
        bearing = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
        elev = math.degrees(math.atan2(alt_ft / FT_PER_M / 1000.0, max(d_now, 1e-3)))
        out.append({"hex": str(a.get("hex") or "").lower(), "flight": (a.get("flight") or "").strip(),
                    "reg": a.get("r"), "type": a.get("t"), "alt_ft": alt_ft, "gs_kt": _num(a.get("gs")),
                    "track": _num(a.get("track")), "d_now_km": d_now, "d_min_km": d_min,
                    "t_min_s": t_min, "bearing_from_site": bearing, "elevation_deg": elev,
                    "squawk": a.get("squawk"), "emergency": a.get("emergency")})
    out.sort(key=lambda r: r["d_min_km"])
    return out


def due(state: dict, hex_code: str, now: Optional[float] = None, cooldown_s: float = COOLDOWN_S) -> bool:
    """True, and the state marked, when this aircraft has not been logged
    within *cooldown_s*; *state* is {hex: epoch of the last line}."""
    now = time.time() if now is None else float(now)
    last = state.get(hex_code)
    if last is not None and now - float(last) < cooldown_s:
        return False
    state[hex_code] = now
    return True


def prune(state: dict, now: Optional[float] = None, keep_s: float = 6 * 3600.0) -> dict:
    """The state without entries older than *keep_s*."""
    now = time.time() if now is None else float(now)
    return {k: v for k, v in state.items() if now - float(v) <= keep_s}


def describe(rec: dict, when: Optional[str] = None) -> str:
    """One chat line for a pass, e.g.
    'Overhead: JBU1234 JetBlue, Airbus A320 at 31,000 ft heading SW at 450 kt,
    68 deg up, closest 1.2 km to the E'."""
    cs, who = carrier(rec.get("flight"), rec.get("reg"))
    name = f"{cs} {who}" if who else cs
    alt = f"{int(round(rec['alt_ft'], -2)):,} ft"
    heading = f" heading {compass(rec['track'])}" if rec.get("track") is not None else ""
    speed = f" at {int(round(rec['gs_kt']))} kt" if rec.get("gs_kt") else ""
    where = f"{rec['d_min_km']:.1f} km"
    t = rec.get("t_min_s")
    if t is None or abs(t) < 15:
        where_word = "closest"
    elif t > 0:
        where_word = f"closest in {int(round(t))} s,"
    else:
        where_word = f"closest {int(round(-t))} s ago,"
    side = compass(rec["bearing_from_site"])
    line = (f"Overhead: {name}, {type_name(rec.get('type'))} at {alt}{heading}{speed}, "
            f"{int(round(rec['elevation_deg']))} deg up to the {side}, {where_word} {where}")
    if rec.get("emergency") not in (None, "", "none"):
        line += f" -- squawking {rec.get('squawk')} ({rec.get('emergency')})"
    if when:
        line = f"{when} {line}"
    return line


def fetch(site_lat: float, site_lon: float, dist_nm: float = QUERY_NM, timeout: float = 10.0,
          source: str = SOURCE, get: Optional[Callable] = None) -> list:
    """The feed's `ac` list within *dist_nm* of the site, or [] on any failure."""
    url = f"{source.rstrip('/')}/lat/{site_lat:.4f}/lon/{site_lon:.4f}/dist/{int(dist_nm)}"
    try:
        if get is None:
            import requests
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "iris-observatory overhead_watch"})
            resp.raise_for_status()
            data = resp.json()
        else:
            data = get(url)
        ac = data.get("ac") if isinstance(data, dict) else None
        return list(ac or [])
    except Exception:  # noqa: BLE001 -- a feed outage is silence, never an error card
        return []


def settings(cfg: dict) -> dict:
    """The overhead block of the config with its defaults filled in."""
    c = dict(cfg.get("overhead") or {})
    loc = cfg.get("location") or {}
    return {"enabled": bool(c.get("enabled", True)),
            "radius_km": float(c.get("radius_km", RADIUS_KM)),
            "query_nm": float(c.get("query_nm", QUERY_NM)),
            "horizon_s": float(c.get("horizon_s", HORIZON_S)),
            "cooldown_s": 60.0 * float(c.get("cooldown_min", COOLDOWN_S / 60.0)),
            "source": str(c.get("source", SOURCE)),
            "lat": float(loc.get("latitude")), "lon": float(loc.get("longitude"))}


def poll_once(cfg: dict, state: dict, post: Callable[[str], object], now: Optional[float] = None,
              get: Optional[Callable] = None, when: Optional[str] = None) -> list[str]:
    """One poll: fetch, pick the passes, post each one that is due; returns
    the lines posted. *state* is updated in place (the caller persists it)."""
    s = settings(cfg)
    if not s["enabled"]:
        return []
    ac = fetch(s["lat"], s["lon"], s["query_nm"], source=s["source"], get=get)
    lines = []
    for rec in passes(ac, s["lat"], s["lon"], s["radius_km"], s["horizon_s"]):
        key = rec["hex"] or rec["flight"] or rec.get("reg") or "?"
        if not due(state, key, now, s["cooldown_s"]):
            continue
        line = describe(rec, when)
        post(line)
        lines.append(line)
    return lines


def nearby_report(cfg: dict, get: Optional[Callable] = None, limit: int = 10) -> str:
    """The `overhead` chat command's answer: every airborne aircraft the
    feed hears within the query radius, nearest first, the passes marked."""
    s = settings(cfg)
    ac = fetch(s["lat"], s["lon"], s["query_nm"], source=s["source"], get=get)
    if not ac:
        return (f"Overhead: nothing airborne within {s['query_nm']:.0f} nm, or the feed "
                f"({s['source']}) did not answer.")
    rows = []
    for a in ac:
        lat, lon, alt = _num(a.get("lat")), _num(a.get("lon")), _num(a.get("alt_baro"))
        if lat is None or lon is None or alt is None or alt <= 0:
            continue
        d_now, d_min, t_min = closest_approach(lat, lon, _num(a.get("gs")), _num(a.get("track")), s["lat"], s["lon"])
        x, y = _local_xy(lat, lon, s["lat"], s["lon"])
        cs, who = carrier(a.get("flight"), a.get("r"))
        name = f"{cs} {who}" if who else cs
        over = " <- passing over" if d_min <= s["radius_km"] and (t_min is None or abs(t_min) <= s["horizon_s"]) else ""
        rows.append((d_now, f"{name}, {type_name(a.get('t'))}, {int(round(alt, -2)):,} ft"
                     + (f" heading {compass(a['track'])}" if _num(a.get("track")) is not None else "")
                     + f", {d_now:.1f} km {compass((math.degrees(math.atan2(x, y)) + 360) % 360)}{over}"))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return f"Overhead: {len(ac)} aircraft heard within {s['query_nm']:.0f} nm, none airborne."
    head = f"Overhead now ({len(rows)} airborne within {s['query_nm']:.0f} nm, nearest first):"
    return "\n".join([head] + [f"  {r[1]}" for r in rows[:limit]]
                     + ([f"  ... and {len(rows) - limit} more"] if len(rows) > limit else []))
