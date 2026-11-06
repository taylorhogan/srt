"""name_fallback.py -- a position and a type for a name when SIMBAD is down.

`image <dso>` resolves a name through SIMBAD's Sesame service and classifies
it through SIMBAD's otype; with both unreachable (2026-11-06, Strasbourg
timing out for days and the Harvard mirror redirecting into the same timeout)
every name, M 13 included, answered "Not a known object". Nothing here
replaces SIMBAD when it answers; this is the second opinion:

1. Wikipedia's infobox: the J2000 position is the {{RA|h|m|s}} / {{DEC|d|m|s}}
   templates on the object's own page (reached by its catalogue name, M 13
   as "Messier 13", with redirects followed; never by a text search, since a
   search result is another object's page and its position would point the
   scope at the wrong thing), and the type is the first family word in the
   article's first sentence (planetary nebula, galaxy, globular cluster).
2. configs/candidates.json: the suggestion catalogue's position, rounded to
   0.1 degree, so it is the last resort and the reply says it is approximate.

Pure functions over a fetcher so the tests need no network. stdlib only, so
the PC's chat server can import it.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

WIKI_UA = "IrisObservatory/1.0 (irisscience.org; name resolver) python-urllib"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATALOGUE = os.path.join(_ROOT, "configs", "candidates.json")

# The earliest family word in the first sentence decides. Clusters and groups
# OF galaxies are G, as SIMBAD's ClG/GrG are on the grid (what the telescope
# frames is a field of galaxies); C is a cluster of stars.
_KIND_WORDS = (
    (r"\b(planetary nebula|emission nebula|reflection nebula|dark nebula|supernova remnant|"
     r"h ?ii region|nebula|molecular cloud)\b", "N"),
    (r"\b(open cluster|globular cluster|star cluster|open star cluster|stellar cluster)\b", "C"),
    (r"\b(galaxy|galaxies)\b", "G"),
)


def squash(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def wiki_titles(name: str) -> list[str]:
    """Page titles to try, best first: 'ngc7008' -> 'NGC 7008'; 'm13' -> 'Messier 13';
    'sh2-101' -> 'Sh 2-101'; 'melotte 15' -> 'Melotte 15'; a plain name as given."""
    out: list[str] = []
    s = re.sub(r"\s+", " ", str(name or "")).strip()
    if not s:
        return out
    m = re.fullmatch(r"(sh\s*2|[a-z]+)[\s-]*(\d+[a-z]?)", s.lower())
    if m:
        pre, num = m.groups()
        pre = re.sub(r"\s+", "", pre)
        if pre == "m":
            out.append(f"Messier {num.upper()}")
        elif pre == "sh2":
            out.append(f"Sh 2-{num}")
        else:
            out.append((pre.upper() if len(pre) <= 3 else pre.title()) + " " + num.upper())
    if s not in out:
        out.append(s)
    return out


def _fetch_wikitext(title: str, timeout: float = 15.0) -> Optional[dict]:
    """{title, text} of the page's current wikitext, redirects followed; None when
    missing or offline."""
    q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                "redirects": 1, "format": "json", "titles": title})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q, headers={"User-Agent": WIKI_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for p in ((d.get("query") or {}).get("pages") or {}).values():
        if "missing" in p:
            return None
        try:
            return {"title": p.get("title"), "text": p["revisions"][0]["slots"]["main"]["*"]}
        except (KeyError, IndexError, TypeError):
            return None
    return None


def _sexagesimal(parts: list[str], hours: bool) -> Optional[float]:
    try:
        nums = [float(x) for x in parts if x.strip() != ""]
    except ValueError:
        return None
    if not nums:
        return None
    sign = -1.0 if parts[0].strip().startswith("-") else 1.0
    nums = [abs(nums[0])] + nums[1:3]
    val = sum(v / (60.0 ** i) for i, v in enumerate(nums))
    return sign * val * (15.0 if hours else 1.0)


def parse_position(text: str) -> Optional[tuple[float, float]]:
    """(ra_deg, dec_deg) from an infobox's RA/DEC templates, or None."""
    ra = re.search(r"\|\s*ra\s*=\s*\{\{\s*RA\s*\|([^}]*)\}\}", text, re.I)
    dec = re.search(r"\|\s*dec\s*=\s*\{\{\s*DEC\s*\|([^}]*)\}\}", text, re.I)
    if not (ra and dec):
        return None
    r = _sexagesimal(ra.group(1).split("|"), hours=True)
    d = _sexagesimal(dec.group(1).split("|"), hours=False)
    if r is None or d is None or not (0 <= r < 360) or not (-90 <= d <= 90):
        return None
    return r, d


def first_sentence(text: str) -> str:
    """The article's first sentence, templates, refs, links and bold stripped."""
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    body = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", body, flags=re.S)
    depth, out = 0, []            # drop {{...}} with nesting
    i = 0
    while i < len(body):
        if body.startswith("{{", i):
            depth += 1; i += 2; continue
        if body.startswith("}}", i) and depth:
            depth -= 1; i += 2; continue
        if not depth:
            out.append(body[i])
        i += 1
    body = "".join(out)
    body = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", body)
    body = re.sub(r"'''?", "", body)
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("|", "{", "}", "[[File", "[[Image", "#", "=", "<", "*")):
            m = re.match(r"(.+?\.)(\s|$)", line)
            return (m.group(1) if m else line)[:400]
    return ""


def kind_letter(sentence: str) -> str:
    """G / N / C / ? for the grid, from the family word in a first sentence."""
    s = sentence.lower()
    best = None
    for pat, letter in _KIND_WORDS:
        m = re.search(pat, s)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), letter)
    return best[1] if best else "?"


def wikipedia(name: str, fetch: Callable[[str], Optional[dict]] = _fetch_wikitext) -> Optional[dict]:
    """{ra_deg, dec_deg, letter, title, source:'wikipedia'} or None."""
    for title in wiki_titles(name):
        page = fetch(title)
        if not page:
            continue
        text = page.get("text") or ""
        if re.search(r"\{\{\s*(dis|disambiguation|disambig|hndis)\b", text, re.I) or "may refer to" in text[:600].lower():
            continue
        pos = parse_position(text)
        if pos is None:
            continue
        return {"ra_deg": round(pos[0], 5), "dec_deg": round(pos[1], 5), "letter": kind_letter(first_sentence(text)),
                "title": page.get("title") or title, "source": "wikipedia"}
    return None


def catalogue(name: str, path: str = CATALOGUE) -> Optional[dict]:
    """The suggestion catalogue's entry as a resolver result (approximate position)."""
    try:
        with open(path, encoding="utf-8") as fh:
            cands = json.load(fh).get("candidates") or []
    except (OSError, ValueError):
        return None
    key = squash(name)
    for c in cands:
        if squash(c.get("name")) == key and c.get("ra_deg") is not None and c.get("dec_deg") is not None:
            k = c.get("kind")
            letter = "G" if k == "galaxy" else ("N" if k in ("nebula", "snr", "pn", "reflection") else ("C" if k == "cluster" else "?"))
            return {"ra_deg": float(c["ra_deg"]), "dec_deg": float(c["dec_deg"]), "letter": letter,
                    "title": c.get("common") or c.get("name"), "source": "catalogue"}
    return None


_HITS: dict[str, dict] = {}
_MISSES: dict[str, float] = {}
MISS_TTL_S = 3600.0     # a placeholder name (gravwav) is asked for every night; once an hour is enough


def resolve(name: str, fetch: Callable[[str], Optional[dict]] = _fetch_wikitext, path: str = CATALOGUE) -> Optional[dict]:
    """Wikipedia, then the catalogue. A hit is remembered for `letter_for`; a
    miss is remembered for an hour."""
    key = squash(name)
    if time.monotonic() - _MISSES.get(key, -1e9) < MISS_TTL_S:
        return None
    hit = wikipedia(name, fetch) or catalogue(name, path)
    if hit:
        _HITS[key] = dict(hit, at=time.monotonic())
        _MISSES.pop(key, None)
    else:
        _MISSES[key] = time.monotonic()
    return hit


def last_hit(name: str) -> Optional[dict]:
    return _HITS.get(squash(name))


def letter_for(name: str, fetch: Callable[[str], Optional[dict]] = _fetch_wikitext, path: str = CATALOGUE) -> str:
    """The grid's letter for a name SIMBAD could not type: the remembered hit, else a fresh lookup, else '?'."""
    hit = last_hit(name) or resolve(name, fetch, path)
    return (hit or {}).get("letter") or "?"
