"""A finished picture dropped in ~/Desktop/astro_output becomes a target folder.

    python scripts/astro_output.py                 # handle every new image once
    python scripts/astro_output.py --poll 120      # keep watching, every 2 minutes
    python scripts/astro_output.py --no-publish    # build the data, touch no site
    python scripts/astro_output.py --no-push       # commit the article, do not push

From Python (a Claude session, the web chat, a loop):

    from scripts import astro_output
    astro_output.process_folder()                  # -> one dict per image handled

An image is NEW when it sits directly in the folder (not in a subfolder). Its
name starts with the target as the frames know it (`ngc2146_LRGB_learned_sat3_crop.png`
-> ngc2146, recipe LRGB). For each new image:

1. `<folder>/<dso>/` is created and the image moved into it.
2. `<dso>_subs.txt` (and `.json`) lists the frames by filter that the stack
   USED: every LIGHT frame of the target put through the stacker's own quality
   gates (`stacking.stacker._select_by_quality` with the defaults `stack()`
   uses, then `_background_gate`) grouped the way the render stacks them, one
   filter at one exposure. The count is the gate's answer today, which is what
   `stack`, `process` and the morning render all get: the stack's own record
   (NFRAMES) is the same number, taken before registration, and a frame that
   then fails to register is dropped silently by every path. Hours and nights
   are of the used frames, for AstroBin.
3. `<dso>_stats.jpg` is the `stats <dso> all` picture: FWHM, eccentricity, sky
   and star count per frame over the whole history, from the same
   frame_stats.json cache (frames missing from it are analysed first).
4. If irisscience.org has no image of the target (no gallery entry whose id,
   title, picture or text names it, by its own name or any SIMBAD alias --
   NGC 7320 is found through "HCG 92a" and the Quintet's text), an article is
   made: the facts go to `article_request.json`, the write-up comes back as
   `article.json` (from the Claude API when the `anthropic` package and a
   credential are present, otherwise whoever runs this fills it in -- a Claude
   session can), and `article.json` is rendered as a gallery entry in the
   site's own markup, the picture copied to `images/<id>.jpg`, committed and
   pushed (live within 5 minutes; scripts/site_push.py announces). A folder
   holding an `article.json` that is not yet published is picked up on the
   next run, so a hand-written write-up needs no flag.

Numbers in the article's meta line (frames, hours, nights) are written from
the data here, never by the model.
"""
from __future__ import annotations

import argparse
import base64
import html as _html
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

_logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FOLDER = Path.home() / "Desktop" / "astro_output"
SITE = Path.home() / "Documents" / "development" / "taylorhogan.github.io"   # scripts/site_push.SITE
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RECIPES = ("LRGB", "HALRGB", "HOO", "SHO", "HSO")
# Defaults of stacking.stacker.stack(): the gate every stacking path runs.
GATE_FWHM_MULT = 1.25
GATE_STAR_FRAC = 0.5
# The write-up model. Opus 5.5 with the server-side refusal fallback; thinking
# is adaptive by default on it, so no thinking parameter is sent.
WRITEUP_MODEL = "claude-opus-5-5"
SITE_IMAGE_MAX_PX = 2400
PROMPT_IMAGE_MAX_PX = 1600

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def squash(s: str) -> str:
    """lower-case alphanumerics only: 'NGC 7320' == 'ngc7320' == 'NGC_7320'."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def parse_image_name(name: str) -> tuple[str, Optional[str]]:
    """(dso, recipe) from a product name: the target is the first `_` token,
    the recipe the first token that is one of RECIPES (case-insensitive)."""
    stem = Path(name).stem
    parts = stem.split("_")
    dso = parts[0].lower()
    recipe = next((p.upper() for p in parts[1:] if p.upper() in RECIPES), None)
    return dso, recipe


def processing_from_name(name: str) -> list[str]:
    """What the product's name says was done to it, in the site's words."""
    toks = [t.lower() for t in Path(name).stem.split("_")]
    out = []
    # The morning render's deblurred/learned composites are the denoised set
    # through the deconvolution, so either name implies denoised.
    if "denoised" in toks or "learned" in toks or "deblurred" in toks:
        out.append("denoised")
    if "learned" in toks:
        out.append("deconvolved (learned, luminance only)")
    elif "deblurred" in toks:
        out.append("deconvolved (starless Richardson-Lucy)")
    for t in toks:
        m = re.fullmatch(r"sat(\d+(?:\.\d+)?)", t)
        if m:
            out.append(f"colour saturation x{m.group(1)}")
    if "crop" in toks:
        out.append("cropped")
    return out


def display_name(dso: str) -> str:
    """'ngc2146' -> 'NGC 2146', 'm33' -> 'M 33', 'sh2-92' -> 'Sh2-92', 'ic1396' -> 'IC 1396'."""
    m = re.fullmatch(r"(ngc|ic|m|ugc|pgc|abell|hcg)(\d+[a-z]?)", dso.lower())
    if m:
        return f"{m.group(1).upper()} {m.group(2).upper()}"
    m = re.fullmatch(r"sh2-(\d+)", dso.lower())
    if m:
        return f"Sh2-{m.group(1)}"
    return dso


def nights_phrase(n: int) -> str:
    return f"{NUMBER_WORDS.get(n, str(n))} night{'s' if n != 1 else ''}"


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    from configs import config
    return config.data()


def find_dso_dir(dso: str, image_dir: Path) -> Optional[Path]:
    """The target's folder under image_dir, as the `stats` command finds it:
    name contains the dso (spaces/underscores ignored); several -> the one
    with the newest LIGHT frame."""
    target = squash(dso)
    cands = [d for d in Path(image_dir).iterdir()
             if d.is_dir() and target in squash(d.name)]
    if not cands:
        return None
    exact = [d for d in cands if squash(d.name) == target]
    if len(exact) == 1:
        return exact[0]

    def newest(d: Path) -> float:
        try:
            return max(f.stat().st_mtime for f in d.rglob("*.fits") if f.parent.name.upper() == "LIGHT")
        except ValueError:
            return 0.0
    return max(exact or cands, key=newest)


def light_frames(dso_dir: Path) -> list[dict]:
    """Every LIGHT frame of the target with its filter (canonical), exposure
    and night (the local date the evening started), headers only."""
    from astropy.io import fits as _fits
    from stacking.color_process import canonical_filter

    out = []
    for fp in sorted(Path(dso_dir).rglob("*.fits")):
        if fp.parent.name.upper() != "LIGHT":
            continue
        try:
            h = _fits.getheader(fp)
        except Exception as e:  # noqa: BLE001
            _logger.warning("unreadable header %s: %s", fp, e)
            continue
        raw = str(h.get("FILTER", "")).strip()
        filt = canonical_filter(raw) or raw or "Unknown"
        exp = float(h.get("EXPTIME", h.get("EXPOSURE", 0.0)) or 0.0)
        t = None
        try:
            t = datetime.fromisoformat(str(h.get("DATE-OBS")).rstrip("Z"))
        except Exception:  # noqa: BLE001
            pass
        if t is None:
            t = datetime.fromtimestamp(fp.stat().st_mtime)
        night = (t - timedelta(hours=12)).date().isoformat()
        out.append({"path": fp, "filter": filt, "exptime": exp, "time": t, "night": night})
    return out


def local_frame_path(path: str, dso_dir: Path) -> Optional[Path]:
    """The local mirror of a cache path written on another machine: the part
    below the target folder (whatever precedes it, '\\' or '/'), under
    *dso_dir*; None unless that file exists here."""
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    name = Path(dso_dir).name.lower()
    idx = next((i for i, p in enumerate(parts) if p.lower() == name), None)
    if idx is None:
        return None
    local = Path(dso_dir).joinpath(*parts[idx + 1:])
    return local if local.exists() else None


def ensure_frame_stats(dso_dir: Path, paths: list[Path], arcsec_per_pixel: float,
                       log: Callable[[str], None] = print) -> None:
    """Analyse the frames frame_stats.json does not know, the way the watcher
    and the `stats` command do, and save them into the cache."""
    from fits_processing import frame_watcher as fw

    cache_path = Path(dso_dir) / "frame_stats.json"
    entries = fw._load_cache(cache_path)
    # The cache is written on the observatory PC and mirrored here with its
    # frames, so its paths are Windows paths (C:\...\Targets\ngc2146\...\LIGHT\x.fits).
    # Every reader keys on the absolute path, so untranslated they match
    # nothing and every frame would be measured again (476 frames, ~1 h on
    # 2026-10-22). Point each entry at the local copy when that file exists.
    translated = 0
    for e in entries:
        p = str(e.get("path", "")) if isinstance(e, dict) else ""
        if not p or os.path.exists(p):
            continue
        local = local_frame_path(p, dso_dir)
        if local is not None:
            e["path"] = str(local)
            translated += 1
    if translated:
        fw._save_cache(cache_path, entries)
        log(f"  frame stats: {translated} cache entries pointed at the local frames")
    known = {os.path.normcase(os.path.abspath(str(e.get("path")))) for e in entries
             if isinstance(e, dict) and e.get("path")}
    missing = [p for p in paths if os.path.normcase(os.path.abspath(str(p))) not in known]
    if not missing:
        return
    log(f"  frame stats: {len(entries)} cached, analysing {len(missing)} new frames")
    with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
        new = list(pool.map(lambda p: fw._analyse(p, arcsec_per_pixel), missing))
    entries.extend(e for e in new if e)
    fw._save_cache(cache_path, entries)


def used_subs(dso_dir: Path, frames: list[dict], filters: Optional[list[str]] = None,
              log: Callable[[str], None] = print) -> dict:
    """Replay the stacker's frame selection per (filter, exposure).

    Returns {filter: [{"exptime", "total", "used", "rejected", "hours",
    "nights": [...], "rejected_files": [...]}, ...]} for the filters asked
    (all present when None), filters ordered as given.
    """
    from fits_processing.frame_cache import load_precomputed_fwhm_stars
    from stacking import stacker

    arcsec = float(_cfg()["nina"]["arc_sec_per_pixel"])
    want = [f for f in (filters or sorted({f["filter"] for f in frames}))]
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for fr in frames:
        if fr["filter"] in want:
            groups[(fr["filter"], fr["exptime"])].append(fr)
    ensure_frame_stats(dso_dir, [fr["path"] for g in groups.values() for fr in g], arcsec, log)

    out: dict[str, list[dict]] = {f: [] for f in want}
    for (filt, exp), grp in sorted(groups.items(), key=lambda kv: (want.index(kv[0][0]), -kv[0][1])):
        paths = [fr["path"] for fr in grp]
        pre = load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
        fwhm = {p: pre.get(p, (0.0, 0))[0] for p in paths}
        stars = {p: pre.get(p, (0.0, 0))[1] for p in paths}
        accepted, rejected, q = stacker._select_by_quality(
            paths, fwhm, stars, max_fwhm_multiplier=GATE_FWHM_MULT, min_star_fraction=GATE_STAR_FRAC)
        if accepted:
            accepted, bg = stacker._background_gate(accepted, None)
            rejected = rejected + bg
        by_path = {fr["path"]: fr for fr in grp}
        nights = sorted({by_path[p]["night"] for p in accepted})
        out[filt].append({
            "exptime": exp, "total": len(paths), "used": len(accepted),
            "rejected": len(rejected), "hours": round(len(accepted) * exp / 3600.0, 2),
            "nights": nights, "gate": q.get("crit_str", "no cut"),
            "rejected_files": sorted(p.name for p in rejected),
        })
        log(f"  {filt} {exp:g} s: {len(accepted)}/{len(paths)} used ({q.get('crit_str', 'no cut')})")
    return out


def render_record(dso: str, recipe: Optional[str]) -> Optional[dict]:
    """The morning render's manifest for this target and recipe
    (local/n2n_lrgb_render/meta_<dso>_<recipe>.json), when it lists the frames
    each channel used -- renders since 2026-10-22 do."""
    if not recipe:
        return None
    p = _ROOT / "local" / "n2n_lrgb_render" / f"meta_{dso}_{recipe}.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    chans = {f: c for f, c in meta.get("channels", {}).items() if isinstance(c.get("used"), list)}
    return chans or None


def subs_from_record(record: dict, frames: list[dict], filters: Optional[list[str]] = None) -> dict:
    """used_subs() from the render's own frame lists instead of a replay.
    Same shape; the exposure and nights come from the frames on disk."""
    by_name = {fr["path"].name: fr for fr in frames}
    want = filters or list(record)
    out: dict[str, list[dict]] = {}
    for filt in want:
        c = record.get(filt)
        if not c:
            continue
        used = [by_name[n] for n in c["used"] if n in by_name]
        exps = sorted({fr["exptime"] for fr in used}) or [0.0]
        for exp in exps:
            u = [fr for fr in used if fr["exptime"] == exp]
            out.setdefault(filt, []).append({
                "exptime": exp, "total": int(c.get("frames", len(u) + len(c.get("rejected", [])))),
                "used": len(u), "rejected": len(c.get("rejected", [])),
                "hours": round(len(u) * exp / 3600.0, 2),
                "nights": sorted({fr["night"] for fr in u}),
                "gate": f"render record {c.get('built', '')}".strip(),
                "rejected_files": sorted(c.get("rejected", [])),
            })
    return out


def subs_summary(subs: dict) -> dict:
    """Totals across filters: hours, nights, and the '56 L + 32 R' style line
    for the site's meta (main exposure only: the groups with the most hours)."""
    nights: set[str] = set()
    hours = 0.0
    parts = []
    for filt, groups in subs.items():
        for g in groups:
            nights.update(g["nights"])
            hours += g["hours"]
        if groups:
            main = max(groups, key=lambda g: g["hours"])
            if main["used"]:
                parts.append((main["used"], filt, main["exptime"]))
    exps = {p[2] for p in parts}
    if len(exps) == 1:
        line = " + ".join(f"{n} {f}" for n, f, _ in parts) + f" &times; {exps.pop():g} s"
    else:
        line = " + ".join(f"{n} {f} &times; {e:g} s" for n, f, e in parts)
    return {"hours": round(hours, 1), "nights": sorted(nights), "line": line}


def subs_text(dso: str, recipe: Optional[str], subs: dict) -> str:
    s = subs_summary(subs)
    from_record = any(g["gate"].startswith("render record") for groups in subs.values() for g in groups)
    lines = [f"{display_name(dso)} -- frames used in the stack"
             + (f" ({recipe})" if recipe else ""),
             f"generated {datetime.now().isoformat(timespec='minutes')}",
             ("counts are the render's own record of the frames it stacked"
              if from_record else
              "counts are the stacker's quality gate (FWHM <= 1.25 x median, stars >= 0.5 x median,"
              " background structure) replayed on the frames on disk from the cached measurements;"
              " the stack measured the frames itself, so its count can differ by a few"),
             ""]
    for filt, groups in subs.items():
        for g in groups:
            lines.append(f"{filt:6s} {g['exptime']:6g} s   used {g['used']:4d} of {g['total']:4d}"
                         f"   {g['hours']:5.2f} h   nights {len(g['nights'])}"
                         + (f"   rejected {g['rejected']}" if g["rejected"] else ""))
    lines += ["", f"total {s['hours']} h over {nights_phrase(len(s['nights']))}: "
                  + (", ".join(s["nights"]) if s["nights"] else "-")]
    rej = [(f, n) for f, groups in subs.items() for g in groups for n in g["rejected_files"]]
    if rej:
        lines += ["", "rejected:"] + [f"  {f}  {n}" for f, n in rej]
    return "\n".join(lines) + "\n"


def stats_picture(dso_dir: Path, out_path: Path) -> Optional[Path]:
    """The `stats <dso> all` plot from the cache (whole history)."""
    from fits_processing import fitsfwhm, imaging_artifacts
    frames = imaging_artifacts.gather_dso_frames(dso_dir, latest_session_only=False)
    if not frames:
        return None
    path, n = fitsfwhm.save_stats_plot_from_cache(frames, out_path)
    return Path(path) if n else None


# ---------------------------------------------------------------------------
# the site
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r'<article\s+id="(?P<id>[^"]+)"\s+class="dso-section"[^>]*>(?P<body>.*?)</article>',
    re.S)


def gallery_articles(html: str) -> list[dict]:
    """Every gallery entry: id, title, image file name, text (tags stripped), span."""
    out = []
    for m in _ARTICLE_RE.finditer(html):
        body = m.group("body")
        t = re.search(r'<h2 class="dso-title">(.*?)</h2>', body, re.S)
        img = re.search(r'<img[^>]+src="([^"?]+)', body)
        text = _html.unescape(re.sub(r"<[^>]+>", " ", body))
        out.append({"id": m.group("id"), "title": _html.unescape(t.group(1)).strip() if t else "",
                    "image": Path(img.group(1)).name if img else "", "text": text,
                    "span": (m.start(), m.end())})
    return out


def site_match(dso: str, aliases: list[str], articles: list[dict]) -> Optional[tuple[dict, str]]:
    """The gallery entry that already shows this target, and how it matched:
    'id', 'title', 'image' (the entry is about it) or 'text' (named in the
    write-up, as NGC 7320 is in the Quintet's). None when the site lacks it."""
    names = [squash(dso)] + [squash(a) for a in aliases]
    # A member designation names its group: "HCG 92a" is the Quintet's HCG 92.
    names += [m.group(1) for n in names for m in [re.fullmatch(r"([a-z]+\d+)[a-z]", n)] if m]
    names = [n for n in names if len(n) >= 4]
    for key in ("id", "title", "image"):
        for a in articles:
            s = squash(a[key])
            if any(n in s for n in names):
                return a, key
    for a in articles:
        s = squash(a["text"])
        if any(n in s for n in names):
            return a, "text"
    return None


def simbad_aliases(dso: str) -> list[str]:
    """Other catalogue names for the target (empty when SIMBAD is unreachable)."""
    try:
        from astroquery.simbad import Simbad
        r = Simbad.query_objectids(display_name(dso))
        return [str(row[0]).strip() for row in r] if r is not None else []
    except Exception as e:  # noqa: BLE001
        _logger.warning("SIMBAD aliases for %s unavailable: %s", dso, e)
        return []


def object_type(dso: str) -> str:
    """'galaxy' / 'nebula' / 'cluster' / 'unknown' from the imaging grid's
    classifier, plus the raw SIMBAD code when cached."""
    try:
        from iris_astronomy import astro_dso_visibility as v
        letter = v.get_dso_type(dso)
    except Exception:  # noqa: BLE001
        letter = "?"
    kind = {"G": "galaxy", "N": "nebula", "C": "star cluster"}.get(letter, "unknown")
    try:
        raw = json.loads((_ROOT / "local" / "dso_types.json").read_text()).get(dso.lower())
    except Exception:  # noqa: BLE001
        raw = None
    return f"{kind} (SIMBAD {raw})" if raw else kind


_NOTE_RE = re.compile(r'<article\s+id="(?P<id>[^"]+)"\s+class="note"[^>]*>(?P<body>.*?)</article>', re.S)


def site_notes(html: str, dso: str) -> list[dict]:
    """Science notes on the site that name the target: [{id, title}], for
    the article to link to."""
    want = {squash(dso)}
    out = []
    for m in _NOTE_RE.finditer(html):
        body = m.group("body")
        if any(n in squash(re.sub(r"<[^>]+>", " ", body)) for n in want):
            t = re.search(r'<h2 class="note-title">(.*?)</h2>', body, re.S)
            title = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", t.group(1)))).strip() if t else m.group("id")
            out.append({"id": m.group("id"), "title": title})
    return out


def article_facts(dso: str, recipe: Optional[str], image: Path, subs: dict,
                  aliases: list[str], notes: Optional[list[dict]] = None) -> dict:
    s = subs_summary(subs)
    return {
        "dso": dso, "name": display_name(dso), "aliases": aliases,
        "related_notes": notes or [],
        "object_type": object_type(dso), "recipe": recipe,
        "processing": processing_from_name(image.name),
        "filters": {f: [{"exptime": g["exptime"], "used": g["used"], "hours": g["hours"],
                         "nights": g["nights"]} for g in groups] for f, groups in subs.items()},
        "hours": s["hours"], "nights": s["nights"], "meta_frames": s["line"],
        "image": image.name, "date": datetime.now().date().isoformat(),
    }


ARTICLE_FIELDS = ("title", "distance_ly", "distance_text", "type_text", "alt",
                  "paragraphs", "time_travel_title", "time_travel")
DISTANCES_FILE = _ROOT / "local" / "dso_distances.json"     # {dso: light years}, hand-kept


def known_distance(dso: str, distances: Optional[dict] = None) -> Optional[int]:
    """A light-year distance from local/dso_distances.json (or *distances*), else None."""
    try:
        table = distances if distances is not None else json.loads(DISTANCES_FILE.read_text())
        v = table.get(squash(dso)) or table.get(dso.lower())
        return int(v) if v else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def distance_text(ly: Optional[int]) -> str:
    """'~70 MILLION LIGHT YEARS' / '~2,300 LIGHT YEARS' / '' for none."""
    if not ly:
        return ""
    if ly >= 1_000_000_000:
        return f"~{ly / 1e9:g} BILLION LIGHT YEARS"
    if ly >= 1_000_000:
        return f"~{ly / 1e6:g} MILLION LIGHT YEARS"
    return f"~{ly:,} LIGHT YEARS"


def fallback_writeup(facts: dict, distances: Optional[dict] = None) -> dict:
    """A complete, facts-only article when no model can write one (D2 of
    docs/AUTOMATION_PLAN.md, 2026-10-30): every ARTICLE_FIELDS key filled from
    the measured facts and the catalogue type, nothing invented. It reads
    plainly, which the card says; the key, when present, replaces it with
    prose. The distance comes from local/dso_distances.json when the owner
    has recorded one, else the meta line says so and the light-travel box is
    left out (build_fragment omits an empty box).
    """
    name = facts.get("name") or display_name(facts.get("dso", ""))
    kind = str(facts.get("object_type") or "unknown").split(" (")[0]
    type_text = {"galaxy": "GALAXY", "nebula": "EMISSION NEBULA", "star cluster": "STAR CLUSTER"}.get(
        kind, "DEEP-SKY OBJECT")
    ly = known_distance(facts.get("dso", ""), distances)
    filters = facts.get("filters") or {}
    parts = []
    for f, groups in filters.items():
        used = sum(int(g.get("used") or 0) for g in groups)
        if used:
            parts.append(f"{used} {f}")
    frames_line = ", ".join(parts) if parts else "the frames on record"
    hours, nights = facts.get("hours"), facts.get("nights") or []
    processing = facts.get("processing") or []
    recipe = facts.get("recipe") or ""
    p1 = (f"{name} is a {kind if kind != 'unknown' else 'deep-sky object'} imaged by Iris, a small "
          f"robotic observatory. The picture is built from {frames_line} exposures of "
          f"{int(next(iter(filters.values()))[0]['exptime']) if filters and filters[next(iter(filters))] else 300} "
          f"seconds each" + (f", {hours} hours in all over {nights_phrase(len(nights))}" if hours else "") + ".")
    p2 = (f"The channels were combined as {recipe}" if recipe else "The channels were combined") + (
        ", then " + " and ".join(processing) if processing else "") +         ". This entry was written from the measured facts alone; a fuller description will follow."
    tt_title = f"On Earth: {distance_text(ly).replace('~', '~').replace(' LIGHT YEARS', ' Years Ago').title()}" if ly else ""
    tt = (f"The light in this picture left {name} about {distance_text(ly).replace('~', '').lower()} ago."
          if ly else "")
    return {"title": f"{squash(facts.get('dso', '')).upper()} - {type_text.title()}",
            "distance_ly": ly or 0, "distance_text": distance_text(ly) or "NOT ON RECORD",
            "type_text": type_text,
            "alt": f"{name}, imaged by Iris" + (f" in {recipe}" if recipe else ""),
            "paragraphs": [p1, p2], "time_travel_title": tt_title, "time_travel": tt,
            "source": "fallback"}


def draft_hash(article: dict, image_bytes: bytes) -> str:
    """What an approval refers to: the article as written plus the picture.
    An edit to either makes a new draft that needs a fresh approve."""
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps({k: article[k] for k in ARTICLE_FIELDS if k in article},
                        sort_keys=True).encode("utf-8"))
    h.update(image_bytes)
    return h.hexdigest()


def writeup_prompt(facts: dict, examples: list[str]) -> tuple[str, str]:
    """(system, user) for the write-up. The examples are two live gallery
    entries, so the voice and the shape are the site's own."""
    system = (
        "You write the gallery entries for irisscience.org, the lab notes of a small robotic "
        "observatory. Each entry is two paragraphs for a college-educated reader who is not an "
        "astronomer: the first about the object itself (what it is, where, why it looks the way it "
        "does, with the distance), the second about this picture (the filters and processing given "
        "in the facts, in plain words), then a short 'On Earth' box about what was happening on "
        "Earth when the light left. Plain prose, no markdown, no HTML, no hedging, no exclamation "
        "marks, no em-dashes. Only state facts about the object that are well established; the "
        "frame counts, hours and nights in the facts are measured and must not be changed. "
        "Two entries from the site follow as examples of the voice and length.\n\n"
        + "\n\n".join(examples)
    )
    user = (
        "Facts for the new entry (JSON):\n" + json.dumps(facts, indent=1) +
        "\n\nThe picture itself is attached. Answer with ONE JSON object and nothing else, with "
        "exactly these keys:\n"
        '  "title": like "NGC2146 - The Dusty Hand" (catalogue id without a space, a dash, the common name or type)\n'
        '  "distance_ly": integer light years (approximate is fine)\n'
        '  "distance_text": as in the meta line, e.g. "~70 MILLION LIGHT YEARS"\n'
        '  "type_text": upper case, e.g. "STARBURST BARRED SPIRAL GALAXY"\n'
        '  "alt": one sentence describing what is visible in this picture, for a blind reader\n'
        '  "paragraphs": a list of exactly two paragraphs (strings); when "related_notes" lists '
        'science notes on the site that used this target, the second paragraph links to each as '
        '[its title](#its id), the way the examples link to theirs\n'
        '  "time_travel_title": e.g. "On Earth: ~70 Million Years Ago"\n'
        '  "time_travel": one paragraph\n'
    )
    return system, user


def _prompt_image(image: Path) -> tuple[str, str]:
    """(media_type, base64) of the picture, shrunk for the request."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image).convert("RGB")
    im.thumbnail((PROMPT_IMAGE_MAX_PX, PROMPT_IMAGE_MAX_PX))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _credentials_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())


def generate_writeup(facts: dict, image: Path, examples: list[str],
                     log: Callable[[str], None] = print) -> Optional[dict]:
    """The write-up from the Claude API, or None (with the reason logged) when
    the package or a credential is missing, the request fails, or the answer
    is not the JSON asked for."""
    try:
        import anthropic
    except ImportError:
        log("  write-up: the anthropic package is not installed (uv pip install anthropic)")
        return None
    if not _credentials_present():
        log("  write-up: no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN and no `ant auth login` profile")
        return None
    system, user = writeup_prompt(facts, examples)
    media, data = _prompt_image(image)
    try:
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=WRITEUP_MODEL, max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
                {"type": "text", "text": user}]}],
        )
    except anthropic.RateLimitError as e:
        log(f"  write-up: rate limited ({e.message}); next run tries again")
        return None
    except anthropic.APIStatusError as e:
        log(f"  write-up: API error {e.status_code}: {e.message}")
        return None
    except anthropic.APIConnectionError as e:
        log(f"  write-up: no connection to the API ({e})")
        return None
    if resp.stop_reason == "refusal":
        why = resp.stop_details.explanation if resp.stop_details else "?"
        log(f"  write-up: the model declined ({why})")
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        art = json.loads(text)
    except json.JSONDecodeError:
        log("  write-up: the answer was not JSON; left in article_request.json as 'raw_answer'")
        facts["raw_answer"] = text
        return None
    missing = [k for k in ARTICLE_FIELDS if k not in art]
    if missing or not isinstance(art.get("paragraphs"), list):
        log(f"  write-up: answer lacks {missing or 'a paragraphs list'}")
        return None
    return art


def site_examples(html: str, ids=("ngc2146", "stephans-quintet")) -> list[str]:
    """The text of two live entries, as the voice to match."""
    out = []
    for a in gallery_articles(html):
        if a["id"] in ids:
            out.append(re.sub(r"\s+", " ", a["text"]).strip())
    return out


def article_id(dso: str, title: str) -> str:
    return squash(dso) or squash(title.split("-")[0])


def md_links(escaped: str) -> str:
    """`[text](#slug)` in an already-escaped paragraph -> an anchor to that
    note on the page; nothing else is markup."""
    return re.sub(r"\[([^\]]+)\]\((#[A-Za-z0-9_-]+)\)", r'<a href="\2">\1</a>', escaped)


def build_fragment(art: dict, facts: dict, image_name: str, date: str) -> str:
    """One gallery entry in the site's exact markup (index.html, dso-section)."""
    e = _html.escape
    aid = article_id(facts["dso"], art["title"])
    processing = " + ".join(p.upper() for p in ("denoised", "deconvolved")
                            if any(p in x for x in facts.get("processing", [])))
    meta = (f"DISTANCE: {e(art['distance_text'])} | TYPE: {e(art['type_text'])} | "
            f"{facts['meta_frames']} | {facts['hours']} HOURS OVER "
            f"{nights_phrase(len(facts['nights'])).upper()}"
            + (f" | {processing}" if processing else ""))
    paras = "\n\n".join(
        f'                <p class="science-text">\n                    {md_links(e(p.strip()))}\n                </p>'
        for p in art["paragraphs"])
    box = (f"""
                <div class="time-travel-box">
                    <span class="time-travel-title">{e(art['time_travel_title'])}</span>
                    <p>
                        {e(art['time_travel'].strip())}
                    </p>
                </div>""" if (art.get("time_travel") or "").strip() else "")
    return f"""{anchor(aid, "start")}
        <!-- {e(facts['name'])} -->
        <article id="{aid}" class="dso-section" data-added="{date}" data-distance-ly="{int(art['distance_ly'])}">
            <div class="dso-image-container">
                <img src="images/{image_name}?v={date}" alt="{e(art['alt'])}" class="dso-image">
            </div>
            <div class="dso-content">
                <h2 class="dso-title">{e(art['title'])}</h2>
                <div class="dso-meta">{meta}</div>

{paras}
{box}
            </div>
        </article>
{anchor(aid, "end")}
"""


def anchor(aid: str, which: str) -> str:
    """The HTML comment that marks a pipeline entry's start or end (D4): a
    re-render of the same target replaces the span between them instead of
    adding a second entry. Entries written by hand have no anchors and are
    never replaced."""
    return f"        <!-- dso:{aid}:{which} -->"


def anchored_span(html: str, aid: str) -> Optional[tuple]:
    """(start, end) of the anchored entry for *aid* in *html*, or None."""
    a, b = anchor(aid, "start").strip(), anchor(aid, "end").strip()
    i = html.find(a)
    if i < 0:
        return None
    j = html.find(b, i)
    if j < 0:
        return None
    start = html.rfind("\n", 0, i) + 1
    end = html.find("\n", j + len(b)) + 1 or len(html)
    return start, end


def insert_fragment(html: str, fragment: str, replace_id: Optional[str] = None) -> str:
    """After the last gallery entry (the notes panel follows it); or, with
    *replace_id* and an anchored entry of that id on the page, in its place."""
    if replace_id:
        span = anchored_span(html, replace_id)
        if span:
            return html[:span[0]] + fragment + html[span[1]:]
    arts = gallery_articles(html)
    if not arts:
        raise ValueError("no gallery entries found in index.html")
    end = arts[-1]["span"][1]
    nl = html.find("\n", end) + 1
    return html[:nl] + "\n" + fragment + html[nl:]


def _git(*args, check=True) -> str:
    return subprocess.run(["git", "-C", str(SITE)] + list(args), check=check,
                          capture_output=True, text=True, timeout=120).stdout.strip()


def publish_article(art: dict, facts: dict, image: Path, push: bool = True,
                    log: Callable[[str], None] = print, replace: bool = False) -> dict:
    """Picture to images/<id>.jpg, entry into index.html, commit, push.

    *replace* (D4): an entry of the same id that the pipeline wrote (between
    its anchors) is replaced by this one -- a re-render re-publishes, never
    duplicates. An entry without anchors was written by hand and is left
    alone (ValueError, as before).
    """
    from PIL import Image
    if not (SITE / "index.html").exists():
        raise FileNotFoundError(f"site repo not at {SITE}")
    aid = article_id(facts["dso"], art["title"])
    date = datetime.now().date().isoformat()
    html = (SITE / "index.html").read_text(encoding="utf-8")
    existing = any(a["id"] == aid for a in gallery_articles(html))
    replacing = bool(replace and existing and anchored_span(html, aid))
    if existing and not replacing:
        raise ValueError(f"index.html already has an entry with id {aid}"
                         + (" (written by hand: not replaced)" if replace else ""))
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image).convert("RGB")
    im.thumbnail((SITE_IMAGE_MAX_PX, SITE_IMAGE_MAX_PX))
    image_name = f"{aid}.jpg"
    im.save(SITE / "images" / image_name, format="JPEG", quality=92)
    fragment = build_fragment(art, facts, image_name, date)
    (SITE / "index.html").write_text(insert_fragment(html, fragment, aid if replacing else None),
                                     encoding="utf-8")
    _git("add", "index.html", f"images/{image_name}")
    _git("commit", "-q", "-m", f"Gallery: {art['title']}" + (" (updated)" if replacing else "")
         + f"\n\n{facts['meta_frames'].replace('&times;', 'x')}, "
         f"{facts['hours']} h over {nights_phrase(len(facts['nights']))}; entry written by astro_output.")
    commit = _git("rev-parse", "--short", "HEAD")
    log(f"  site: committed {commit} ({aid}, images/{image_name}{', replaced' if replacing else ''})")
    pushed = False
    if push:
        try:
            _git("push", "-q", "origin", "main")
            pushed = True
            log("  site: pushed; live within 5 minutes (scripts/site_push.py announces)")
        except subprocess.CalledProcessError as e:
            log(f"  site: push failed, commit kept: {e.stderr.strip()}")
    return {"id": aid, "commit": commit, "pushed": pushed, "image": image_name, "replaced": replacing}


# ---------------------------------------------------------------------------
# the folder
# ---------------------------------------------------------------------------

def new_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))


def handle_image(image: Path, folder: Path, publish: bool = True, push: bool = True,
                 log: Callable[[str], None] = print, draft_only: bool = False) -> dict:
    """Steps 1-4 of the module doc for one new image. Returns what was made.

    *draft_only* (D2): write article.json -- the model's write-up, else
    fallback_writeup -- and stop; nothing reaches the site. result["article"]
    is "draft" and result["article_source"] says who wrote it. The folder is
    then gated by draft.json (scripts/publish_prepare) until approved.
    """
    dso, recipe = parse_image_name(image.name)
    out_dir = Path(folder) / dso
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / image.name
    shutil.move(str(image), dest)
    log(f"{image.name}: {display_name(dso)}" + (f" {recipe}" if recipe else "") + f" -> {out_dir}/")
    result = {"dso": dso, "recipe": recipe, "image": str(dest), "folder": str(out_dir),
              "subs": None, "stats": None, "site": None, "article": None}

    image_dir = Path(_cfg()["nina"]["image_dir"])
    dso_dir = find_dso_dir(dso, image_dir)
    subs = {}
    if dso_dir is None:
        log(f"  no frames folder for {dso} under {image_dir}; no sub counts, no stats")
    else:
        frames = light_frames(dso_dir)
        # A frame taken after the picture was made cannot be in it: the
        # gate replay sees the folder as it is now, the picture saw it then.
        made = datetime.fromtimestamp(dest.stat().st_mtime)
        later = [fr for fr in frames if fr["time"] > made]
        if later:
            frames = [fr for fr in frames if fr["time"] <= made]
            log(f"  {len(later)} frames taken after the picture ({made:%Y-%m-%d %H:%M}) left out")
        filters = None
        if recipe:
            from stacking.color_process import RECIPES as _R
            filters = list(dict.fromkeys(_R[recipe].values()))
            if "L" in filters:
                filters = ["L"] + [f for f in filters if f != "L"]
        ensure_frame_stats(dso_dir, [fr["path"] for fr in frames],
                           float(_cfg()["nina"]["arc_sec_per_pixel"]), log)
        record = render_record(dso, recipe)
        if record and all(f in record for f in (filters or record)):
            subs = subs_from_record(record, frames, filters)
            source = "render record"
            for f, groups in subs.items():
                log(f"  {f}: {sum(g['used'] for g in groups)} used, from the render's record")
        else:
            subs = used_subs(dso_dir, frames, filters, log)
            source = "gate replay"
        txt = out_dir / f"{dso}_subs.txt"
        txt.write_text(subs_text(dso, recipe, subs))
        (out_dir / f"{dso}_subs.json").write_text(json.dumps(
            {"dso": dso, "recipe": recipe, "frames_dir": str(dso_dir), "source": source,
             "filters": subs, **subs_summary(subs)}, indent=1, default=str))
        result["subs"] = str(txt)
        pic = stats_picture(dso_dir, out_dir / f"{dso}_stats.jpg")
        result["stats"] = str(pic) if pic else None
        log(f"  wrote {txt.name}" + (f", {pic.name}" if pic else ", no stats picture (no measured frames)"))

    if not publish:
        return result
    html = (SITE / "index.html").read_text(encoding="utf-8") if (SITE / "index.html").exists() else ""
    if not html:
        log(f"  site repo not found at {SITE}; article skipped")
        return result
    aliases = simbad_aliases(dso)
    hit = site_match(dso, aliases, gallery_articles(html))
    if hit:
        a, how = hit
        result["site"] = {"id": a["id"], "match": how}
        log(f"  on the site already: '{a['title']}' (#{a['id']}, matched by {how}); no article")
        return result
    facts = article_facts(dso, recipe, dest, subs, aliases, site_notes(html, dso))
    (out_dir / "article_request.json").write_text(json.dumps(facts, indent=1))
    art = generate_writeup(facts, dest, site_examples(html), log)
    if draft_only:
        source = "api"
        if art is None:
            art = fallback_writeup(facts)
            source = "fallback"
            log("  write-up: facts-only fallback (no model); the card will say so")
        art = dict(art, source=source)
        (out_dir / "article.json").write_text(json.dumps(art, indent=1))
        result["article"], result["article_source"] = "draft", source
        result["facts"] = facts
        log(f"  draft written: {out_dir / 'article.json'} ({source})")
        return result
    if art is None:
        (out_dir / "article_request.json").write_text(json.dumps(facts, indent=1))
        log(f"  article pending: write {out_dir / 'article.json'} with keys {ARTICLE_FIELDS} "
            "(see article_request.json); the next run publishes it")
        result["article"] = "pending"
        return result
    (out_dir / "article.json").write_text(json.dumps(art, indent=1))
    result["article"] = publish_article(art, facts, dest, push, log)
    return result


def pending_articles(folder: Path, push: bool = True, log: Callable[[str], None] = print) -> list[dict]:
    """Folders with an article.json that is not yet on the site."""
    out = []
    for d in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        aj, rq = d / "article.json", d / "article_request.json"
        if not aj.exists() or not rq.exists() or (d / "article_published.json").exists():
            continue
        gate = d / "draft.json"
        if gate.exists():
            # A draft the pipeline made (scripts/publish_prepare): it publishes
            # only once approved (`approve <dso>`, or the timer), by D4's poll.
            try:
                status = json.loads(gate.read_text()).get("status")
            except (OSError, ValueError):
                status = "unreadable"
            if status != "approved":
                log(f"{d.name}: draft {status}; awaiting `approve {d.name}` -- not published")
                continue
        art, facts = json.loads(aj.read_text()), json.loads(rq.read_text())
        missing = [k for k in ARTICLE_FIELDS if k not in art]
        if missing:
            log(f"{d.name}: article.json lacks {missing}; not published")
            continue
        image = d / facts["image"]
        html = (SITE / "index.html").read_text(encoding="utf-8")
        if any(a["id"] == article_id(facts["dso"], art["title"]) for a in gallery_articles(html)):
            log(f"{d.name}: already on the site; marking published")
            (d / "article_published.json").write_text(json.dumps({"id": article_id(facts["dso"], art["title"])}))
            continue
        log(f"{d.name}: publishing the write-up in article.json")
        res = publish_article(art, facts, image, push, log)
        (d / "article_published.json").write_text(json.dumps(res, indent=1))
        out.append({"dso": facts["dso"], "article": res})
    return out


def process_folder(folder: Path = DEFAULT_FOLDER, publish: bool = True, push: bool = True,
                   log: Callable[[str], None] = print) -> list[dict]:
    """Handle every new image in *folder*, then any write-up left for publishing."""
    folder = Path(folder)
    if not folder.is_dir():
        log(f"no folder {folder}")
        return []
    results = []
    for image in new_images(folder):
        try:
            results.append(handle_image(image, folder, publish, push, log))
        except Exception as e:  # noqa: BLE001 -- one bad image must not stop the rest
            _logger.exception("astro_output: %s failed", image.name)
            log(f"{image.name}: failed: {e!r}")
            results.append({"image": str(image), "error": repr(e)})
    if publish:
        try:
            results.extend(pending_articles(folder, push, log))
        except Exception as e:  # noqa: BLE001
            _logger.exception("astro_output: pending articles failed")
            log(f"pending articles: failed: {e!r}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder", default=str(DEFAULT_FOLDER))
    ap.add_argument("--poll", type=float, default=0, metavar="SECONDS",
                    help="keep watching, checking every SECONDS (0: once)")
    ap.add_argument("--no-publish", dest="publish", action="store_false",
                    help="data only: no site check, no article")
    ap.add_argument("--no-push", dest="push", action="store_false",
                    help="commit the article to the site repo but do not push")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    while True:
        res = process_folder(Path(a.folder).expanduser(), a.publish, a.push)
        if not res:
            print("nothing new")
        if not a.poll:
            return 0 if not any("error" in r for r in res) else 1
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
