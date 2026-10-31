"""AAVSO submissions from the light-curve pipeline (2026-10-31).

    python scripts/aavso_export.py <dso> [--filter L] [--obscode XXXX] [--name "CF Cas"]

photometry/lightcurve.py zero-points every frame against Gaia DR3 G, which is
the right scale for a period and the wrong one for the AAVSO: WebObs wants a
magnitude on a standard system with the comparison sequence named. This
module re-zero-points the same aperture photometry against the AAVSO
sequence for the star, fetched from the Variable Star Plotter (VSP) API and
cached, and writes the AAVSO Extended File Format:

  - the ensemble is every sequence star with a V magnitude inside the frame's
    usable range, except one held back as the check star (KNAME/KMAG), the
    comp closest in brightness to the target, so the check tests the zero
    point where the target sits;
  - the luminance filter is reported as CV (clear, V zero point) and the
    colour filters as TG / TB / TR (untransformed tri-colour), TRANS=NO,
    MTYPE=STD; nothing here invents a transformation;
  - one line per (night, filter): the nightly mean, as the site's curves are.

The pure parts (sequence parsing, the choice of check star, the lines) are
tested without the network or frames; the frame work reuses
lightcurve.solve_frame / measure_star and runs where the frames are (the PC).
"""
import json
import math
from pathlib import Path
from typing import Optional

VSP_URL = "https://apps.aavso.org/vsp/api/chart/"
FILTER_CODES = {"L": "CV", "G": "TG", "B": "TB", "R": "TR", "V": "V"}
SOFTWARE = "Iris photometry/lightcurve.py + aavso.py"
DEFAULT_FOV_ARCMIN = 30.0
DEFAULT_MAGLIMIT = 16.0
HEADER_FIELDS = "NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES"
# Queue names that are not VSX names. VSX wants the exact string: "M31 V1",
# "M31 V0001" and "HUBBLE V1" all return nothing; "M31_V1" is the entry
# (checked 2026-10-31). --name overrides.
VSX_NAMES = {"hubblev1": "M31_V1"}

_ROOT = Path(__file__).resolve().parent.parent
LC_DIR = _ROOT / "local" / "lightcurves"


# --------------------------------------------------------------------------- pure

def sexa_to_deg(text, hours: bool = False) -> Optional[float]:
    """'23:58:17.98' -> degrees (x15 when hours); a plain number passes through. Pure."""
    if text is None:
        return None
    s = str(text).strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.replace(" ", ":").split(":")
    try:
        sign = -1.0 if parts[0].strip().startswith("-") else 1.0
        d = abs(float(parts[0]))
        m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
    except (ValueError, IndexError):
        return None
    v = sign * (d + m / 60.0 + sec / 3600.0)
    return v * 15.0 if hours else v


def parse_sequence(doc: dict) -> dict:
    """The VSP chart JSON -> {chartid, auid, star, comps: [{auid, label, ra, dec, v, v_err}]}.
    Stars without a V magnitude are dropped: the ensemble is a V zero point. Pure."""
    comps = []
    for p in doc.get("photometry") or []:
        v = next((b for b in p.get("bands") or [] if str(b.get("band")).upper() == "V"), None)
        if not v or v.get("mag") is None:
            continue
        ra, dec = sexa_to_deg(p.get("ra"), hours=True), sexa_to_deg(p.get("dec"))
        if ra is None or dec is None:
            continue
        comps.append({"auid": p.get("auid"), "label": str(p.get("label")), "ra": ra, "dec": dec,
                      "v": float(v["mag"]), "v_err": float(v.get("error") or 0.0)})
    return {"chartid": doc.get("chartid"), "auid": doc.get("auid"), "star": doc.get("star"),
            "ra": sexa_to_deg(doc.get("ra"), hours=True), "dec": sexa_to_deg(doc.get("dec")),
            "comps": comps}


def choose(comps: list, mag_range: tuple, target_mag: Optional[float] = None, min_ensemble: int = 3) -> dict:
    """{ensemble: [...], check: {...} | None}: the sequence stars inside the
    frame's usable range, the one closest in brightness to the target held
    back as the check star when enough remain for an ensemble. Pure."""
    lo, hi = mag_range
    usable = sorted((c for c in comps if lo <= c["v"] <= hi), key=lambda c: c["v"])
    if len(usable) <= min_ensemble:
        return {"ensemble": usable, "check": None}
    ref = target_mag if target_mag is not None else usable[len(usable) // 2]["v"]
    check = min(usable, key=lambda c: abs(c["v"] - ref))
    return {"ensemble": [c for c in usable if c is not check], "check": check}


def pool_sequences(own: dict, others: list, min_comps: int = 5) -> dict:
    """A star whose own sequence is thin borrows the comps of the other stars
    in the same field (NGC 7790: CG Cas has 3 within 30 arcmin, CF Cas 13),
    deduplicated by AUID; the chart named is the one that gave the most comps
    when the star's own gave too few. Pure."""
    if len(own.get("comps") or []) >= min_comps or not others:
        return own
    seen = {c["auid"] for c in own.get("comps") or []}
    comps = list(own.get("comps") or [])
    best = own
    for o in others:
        if len(o.get("comps") or []) > len(best.get("comps") or []):
            best = o
        for c in o.get("comps") or []:
            if c["auid"] not in seen:
                seen.add(c["auid"])
                comps.append(c)
    return dict(own, comps=comps, chartid=best.get("chartid"), borrowed_from=best.get("star"))


def filter_code(filt: str) -> str:
    code = FILTER_CODES.get(str(filt).upper()) or FILTER_CODES.get(str(filt))
    if not code:
        raise ValueError("no AAVSO filter code for %r (known: %s)" % (filt, ", ".join(FILTER_CODES)))
    return code


def header(obscode: str, software: str = SOFTWARE) -> list:
    return ["#TYPE=EXTENDED", "#OBSCODE=%s" % obscode, "#SOFTWARE=%s" % software,
            "#DELIM=,", "#DATE=JD", "#OBSTYPE=CCD", "#" + HEADER_FIELDS]


def _clean(text) -> str:
    return str(text if text is not None else "na").replace(",", ";").replace("\n", " ").strip() or "na"


def format_line(name: str, jd: float, mag: float, err: float, filt: str, chartid: Optional[str],
                check_label: Optional[str], check_mag: Optional[float], notes: str = "",
                ensemble: bool = True, comp_label: Optional[str] = None, comp_mag: Optional[float] = None) -> str:
    """One Extended Format observation. Pure."""
    return ",".join([
        _clean(name), "%.5f" % jd, "%.3f" % mag, "%.3f" % err, filter_code(filt), "NO", "STD",
        "ENSEMBLE" if ensemble else _clean(comp_label), "na" if ensemble else ("%.3f" % comp_mag if comp_mag is not None else "na"),
        _clean(check_label) if check_label else "na", "%.3f" % check_mag if check_mag is not None else "na",
        "na", "na", _clean(chartid), _clean(notes)])


def notes_text(n_frames: int, exptime_s: Optional[float], aperture_arcsec: float, n_ensemble: int, scale_note: str = "") -> str:
    exp = ("%g s" % exptime_s) if exptime_s else "?"
    return ("nightly mean of %d x %s frames; %.0f arcsec aperture; ensemble of %d sequence stars%s"
            % (n_frames, exp, aperture_arcsec, n_ensemble, ("; " + scale_note) if scale_note else ""))


# --------------------------------------------------------------------------- network

def fetch_sequence(star: str, fov_arcmin: float = DEFAULT_FOV_ARCMIN, maglimit: float = DEFAULT_MAGLIMIT,
                   cache_key: Optional[str] = None, refresh: bool = False) -> dict:
    """The AAVSO sequence for *star* from the VSP API, parsed, cached as
    local/lightcurves/<key>_vsp.json. A missing or empty sequence raises."""
    import requests
    cache = LC_DIR / ("%s_vsp.json" % (cache_key or "".join(c if c.isalnum() else "_" for c in star.lower())))
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    r = requests.get(VSP_URL, params={"star": star, "fov": fov_arcmin, "maglimit": maglimit, "format": "json"},
                     timeout=60, headers={"User-Agent": "Iris/aavso_export"})
    r.raise_for_status()
    seq = parse_sequence(r.json())
    if not seq["comps"]:
        raise ValueError("VSP has no V sequence for %r within %g arcmin to magnitude %g" % (star, fov_arcmin, maglimit))
    LC_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(seq, indent=1))
    return seq


# --------------------------------------------------------------------------- the export (frames: the PC)

def export(dso: str, filt: str = "L", obscode: str = "", name: Optional[str] = None, nights: Optional[list] = None,
           fov_arcmin: float = DEFAULT_FOV_ARCMIN, maglimit: float = DEFAULT_MAGLIMIT, star_name: Optional[str] = None,
           progress_cb=None, refresh_sequence: bool = False) -> dict:
    """Measure every night of *dso* in *filt* against the AAVSO sequence and
    write local/lightcurves/aavso/<key>_<FILT>.txt. One queue entry may list
    several stars (NGC 7790's Cepheids); *star_name* picks one, else every star
    with a VSP sequence is exported to its own file.

    Returns {"files": [...], "lines": {key: [...]}, "skipped": {name: why}}.
    """
    from configs import config
    from photometry import lightcurve as lc
    cfg = config.data()
    obscode = obscode or (cfg.get("aavso") or {}).get("obscode") or ""
    if not obscode:
        raise ValueError("an AAVSO observer code is needed: --obscode XXXX (or cfg['aavso']['obscode'])")
    image_dir = Path(cfg["nina"]["image_dir"])
    aps = float(cfg["nina"]["arc_sec_per_pixel"])
    astap = (cfg.get("hardware") or {}).get("astap_exe") or "C:/Program Files/astap/astap.exe"
    target, rec = lc._target_for(dso)
    stars = [s for s in lc.stars_for(dso, target, rec) if not star_name or s["name"].lower() == star_name.lower()]
    if not stars:
        raise ValueError("%s lists no star named %r" % (dso, star_name))
    say = progress_cb or (lambda m: None)
    out_dir = LC_DIR / "aavso"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"files": [], "lines": {}, "skipped": {}}
    seqs = {}
    for s in stars:
        vsx = name if (name and len(stars) == 1) else VSX_NAMES.get(s["name"].lower(), s["name"])
        try:
            seqs[s["key"]] = fetch_sequence(vsx, fov_arcmin, maglimit, cache_key=s["key"], refresh=refresh_sequence)
        except Exception as exc:  # noqa: BLE001 -- say which star has no sequence, export the rest
            result["skipped"][s["name"]] = "no AAVSO sequence for %r: %s" % (vsx, exc)
            say(result["skipped"][s["name"]])
    stars = [s for s in stars if s["key"] in seqs]
    for s in stars:
        pooled = pool_sequences(seqs[s["key"]], [seqs[o["key"]] for o in stars if o is not s], lc.MIN_COMPS)
        if pooled.get("borrowed_from"):
            say("%s: its own sequence has %d comps; using %s's chart %s too (%d comps)" % (
                s["name"], len(seqs[s["key"]]["comps"]), pooled["borrowed_from"], pooled["chartid"], len(pooled["comps"])))
        seqs[s["key"]] = pooled
    frames = lc._frames(image_dir, dso, filt)
    lines = {s["key"]: [] for s in stars}
    for (night, fl), paths in sorted(frames.items()):
        if nights and night not in nights:
            continue
        say("%s %s %s: %d frames against the AAVSO sequence" % (dso, night, fl, len(paths)))
        per = {s["key"]: {"t": [], "k": [], "exp": None, "n_ens": 0, "check": None} for s in stars}
        for p in paths:
            solved = lc.solve_frame(p, target, astap, None)
            if solved is None:
                continue
            data, hdr, wcs = solved
            exp = float(hdr.get("EXPTIME", 0) or 0)
            for s in stars:
                seq = seqs[s["key"]]
                pick = choose(seq["comps"], lc.comp_range(exp), target_mag=None)
                ens = [{"ra": c["ra"], "dec": c["dec"], "gmag": c["v"]} for c in pick["ensemble"]]
                if len(ens) < lc.MIN_COMPS:
                    continue
                m = lc.measure_star(data, hdr, wcs, p, (s["ra"], s["dec"]), ens, aps)
                if m:
                    per[s["key"]]["t"].append(m)
                    per[s["key"]]["exp"] = exp
                    per[s["key"]]["n_ens"] = m["n_comps"]
                if pick["check"]:
                    per[s["key"]]["check"] = pick["check"]
                    k = lc.measure_star(data, hdr, wcs, p, (pick["check"]["ra"], pick["check"]["dec"]), ens, aps)
                    if k:
                        per[s["key"]]["k"].append(k)
        for s in stars:
            got = per[s["key"]]
            pt = lc.nightly_point([(m["jd"], m["mag"], m["err"]) for m in got["t"]])
            if pt is None:
                say("%s %s %s: no usable frames for %s" % (dso, night, fl, s["name"]))
                continue
            kpt = lc.nightly_point([(m["jd"], m["mag"], m["err"]) for m in got["k"]]) if got["k"] else None
            seq = seqs[s["key"]]
            line = format_line(seq.get("star") or s["name"], pt["jd"], pt["mag"], pt["err"], fl, seq.get("chartid"),
                               got["check"]["label"] if got["check"] else None, kpt["mag"] if kpt else None,
                               notes_text(pt["n"], got["exp"], 2 * lc.AP_R_ARCSEC, got["n_ens"],
                                          "check star catalog V %.3f" % got["check"]["v"] if got["check"] else ""))
            lines[s["key"]].append(line)
    for s in stars:
        if not lines[s["key"]]:
            continue
        path = out_dir / ("%s_%s.txt" % (s["key"], filter_code(filt)))
        path.write_text("\n".join(header(obscode) + lines[s["key"]]) + "\n")
        result["files"].append(str(path))
        result["lines"][s["key"]] = lines[s["key"]]
    return result
