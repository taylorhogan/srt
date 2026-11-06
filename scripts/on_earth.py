"""on_earth.py -- the 'On Earth' box of a gallery entry from a standard table
of eras (2026-11-06, owner: "write a standard template for all the
geological times, and have it point to the item in the template").

The box used to be written by the model, from memory, and memory is where
a language model is confidently wrong about dates: the bench of 2026-11-06
had eight errors in twelve boxes (Uruk writing at 4500 BC, the Triassic at
180 million years, no polar ice in the Oligocene) and neither reviewer had
anything to check them against. Now the year the light left, computed from
the distance on record, picks an era from configs/on_earth.json; the box is
one sentence naming the era with a link to its section of the site's
timeline, and the timeline holds the paragraph, written once and checked
once. The model writes nothing in it.

    python scripts/on_earth.py box ngc6946 25000000     # the box for a name and a distance
    python scripts/on_earth.py era 180000000            # which era 180 Myr ago is
    python scripts/on_earth.py site [--push]            # the timeline section into the site's index.html

The table runs from the last hundred years to the first galaxies, 14 billion
years, contiguous; every distance maps to exactly one era (tests check).
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

TABLE = Path(_root) / "configs" / "on_earth.json"
THIS_YEAR = 2026
START = "<!-- on-earth:start -->"
END = "<!-- on-earth:end -->"


def load(path: Path = TABLE) -> list[dict]:
    eras = json.loads(path.read_text(encoding="utf-8"))["eras"]
    return sorted(eras, key=lambda e: e["from"])


def era_for(years_ago: float, eras: Optional[list[dict]] = None) -> Optional[dict]:
    """The era *years_ago* falls in (from <= years_ago < to; the last era is closed at the top)."""
    eras = eras or load()
    try:
        y = float(years_ago)
    except (TypeError, ValueError):
        return None
    if y < 0:
        return None
    for e in eras:
        if e["from"] <= y < e["to"]:
            return e
    return eras[-1] if y <= eras[-1]["to"] else None


def phase(years_ago: float, era: dict) -> str:
    """'early' / 'middle' / 'late' within a geological era, by thirds; '' for a
    historical one (a dated span reads better without it)."""
    if era["to"] < 11700:
        return ""
    f = (float(years_ago) - era["from"]) / max(1.0, era["to"] - era["from"])
    return "late" if f < 1 / 3 else ("middle" if f < 2 / 3 else "early")   # fewer years ago = later


def years_ago_words(years_ago: float) -> str:
    """'about 500 BC' / 'about AD 500' / 'about 25,000 years ago' / 'about 70 million years ago'."""
    y = float(years_ago)
    if y >= 1e9:
        return f"about {y / 1e9:.3g} billion years ago"
    if y >= 1e6:
        return f"about {y / 1e6:.3g} million years ago"
    if y >= 11700:
        return f"about {int(round(y, -3)):,} years ago"
    step = 100 if y >= 1000 else 10
    year = int(round((THIS_YEAR - y) / step) * step)
    return f"about AD {year}" if year > 0 else f"about {-year:,} BC"


def title_for(years_ago: float) -> str:
    """The box's title line: 'On Earth: ~500 BC', 'On Earth: ~25 Million Years Ago'."""
    y = float(years_ago)
    if y >= 1e9:
        return f"On Earth: ~{y / 1e9:.3g} Billion Years Ago"
    if y >= 1e6:
        return f"On Earth: ~{y / 1e6:.3g} Million Years Ago"
    if y >= 11700:
        return f"On Earth: ~{int(round(y, -3)):,} Years Ago"
    w = years_ago_words(y)
    return "On Earth: ~" + w.replace("about ", "")


def box_for(name: str, distance_ly, eras: Optional[list[dict]] = None) -> Optional[dict]:
    """{time_travel_title, time_travel, on_earth_era} for an object *distance_ly*
    away, or None when the distance is unknown. The sentence links the era's
    section of the site's timeline ([the late Oligocene](#era-oligocene))."""
    try:
        y = float(distance_ly)
    except (TypeError, ValueError):
        return None
    if y <= 0:
        return None
    era = era_for(y, eras)
    if era is None:
        return None
    ph = phase(y, era)
    label = (ph + " " + era["name"]) if ph else era["name"]
    if ph and era["name"].startswith("the "):
        label = "the " + ph + " " + era["name"][4:]
    text = (f"This light left {name} {years_ago_words(y)}, in [{label}](#era-{era['id']}): {era['summary']}.")
    return {"time_travel_title": title_for(y), "time_travel": text, "on_earth_era": era["id"]}


# ------------------------------------------------------------------ the site's timeline

def timeline_html(eras: Optional[list[dict]] = None) -> str:
    """The 'When the light left' section: one anchored block per era, nearest first."""
    eras = eras or load()
    e = _html.escape
    blocks = "\n".join(
        f'            <article class="era" id="era-{era["id"]}">\n'
        f'                <h3 class="era-title">{e(era["name"][0].upper() + era["name"][1:])} '
        f'<span class="era-span">{e(era["span"])}</span></h3>\n'
        f'                <p class="era-text">{e(era["paragraph"])}</p>\n'
        f'            </article>' for era in eras)
    return f"""{START}
        <section class="on-earth" id="on-earth" aria-labelledby="on-earth-title">
            <style>
                .on-earth {{ margin: 3rem auto 1rem; max-width: 70ch; padding-top: 2rem; border-top: 1px solid rgba(128,128,128,.35); }}
                .on-earth h2 {{ margin: 0 0 .4rem; }}
                .on-earth .on-earth-lede {{ opacity: .8; margin: 0 0 1.4rem; }}
                .on-earth .era {{ margin: 0 0 1.2rem; }}
                .on-earth .era-title {{ font-size: 1.05rem; margin: 0 0 .25rem; }}
                .on-earth .era-span {{ font-weight: normal; opacity: .7; font-size: .9rem; margin-left: .5rem; }}
                .on-earth .era-text {{ margin: 0; line-height: 1.55; }}
                .on-earth .era:target {{ background: rgba(127,200,214,.12); outline: 1px solid rgba(127,200,214,.5); border-radius: 4px; padding: .5rem .7rem; }}
            </style>
            <h2 id="on-earth-title">When the light left</h2>
            <p class="on-earth-lede">Every picture on this page is a look into the past: the light that made it left its object as many years ago as the object is light years away. The box under each picture names the era it set out in; these are the eras, nearest first, from the last century to the first galaxies.</p>
{blocks}
            <script>
                // A deep link to an era (#era-oligocene) lands on the gallery tab
                // with that era in view; the page's own deep-link code knows only
                // the entry cards (their _deck), so an era anchor would otherwise
                // leave a visitor on the Live tab.
                (function () {{
                    if (location.hash.indexOf('#era-') !== 0) {{ return; }}
                    var target = document.getElementById(location.hash.slice(1));
                    if (!target) {{ return; }}
                    document.body.dataset.mode = 'gallery';
                    document.querySelectorAll('.mode-btn').forEach(function (btn) {{
                        btn.setAttribute('aria-selected', btn.dataset.target === 'gallery');
                    }});
                    window.requestAnimationFrame(function () {{ target.scrollIntoView({{ block: 'start' }}); }});
                }})();
            </script>
        </section>
        {END}"""


def with_timeline(index_html: str, section: str) -> str:
    """*index_html* with the timeline section in place: replacing the one between
    the markers, else inserted at the end of the gallery panel (before the notes
    panel). Idempotent."""
    a, b = index_html.find(START), index_html.find(END)
    if a >= 0 and b > a:
        return index_html[:a] + section + index_html[b + len(END):]
    m = re.search(r"\n(\s*)</div>\s*\n\s*<div class=\"container\" id=\"notes-panel\"", index_html)
    if not m:
        raise ValueError("gallery panel's end not found in index.html")
    indent = m.group(1)
    return index_html[:m.start()] + "\n\n" + indent + "    " + section + "\n" + index_html[m.start():]


def cmd_site(a) -> int:
    from scripts import astro_output as ao
    p = ao.SITE / "index.html"
    before = p.read_text(encoding="utf-8")
    after = with_timeline(before, timeline_html())
    if after == before:
        print("timeline already current:", p)
    else:
        p.write_text(after, encoding="utf-8")
        print(f"timeline written into {p} ({len(load())} eras)")
    if a.push and not ao._git("status", "--porcelain", "--", "index.html"):
        print("nothing to push")
        return 0
    if a.push:   # whatever index.html holds, current or just written (2026-11-06: a current file with an uncommitted diff was left unpushed)
        ao._git("add", "index.html")
        ao._git("commit", "-m", "Gallery: the 'When the light left' timeline, one section per era (scripts/on_earth.py)")
        ao._git("push")
        print("committed and pushed")
    else:
        print("not committed; `git -C", ao.SITE, "diff --stat` to see it, --push to publish")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("box"); b.add_argument("name"); b.add_argument("distance_ly", type=float)
    b.set_defaults(fn=lambda a: print(json.dumps(box_for(a.name, a.distance_ly), indent=1)) or 0)
    e = sub.add_parser("era"); e.add_argument("years_ago", type=float)
    e.set_defaults(fn=lambda a: print(json.dumps(era_for(a.years_ago), indent=1)) or 0)
    s = sub.add_parser("site"); s.add_argument("--push", action="store_true")
    s.set_defaults(fn=cmd_site)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
