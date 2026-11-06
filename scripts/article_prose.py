"""article_prose.py -- the prose of a gallery entry, written by a model on the
Spark and checked by one (2026-11-05, docs/ARTICLE_PROSE_PLAN.md; item 5 of
the GPU list: "a local model to write the publish articles").

    python scripts/article_prose.py write ic410               # rewrite ~/Desktop/astro_output/ic410/article.json
    python scripts/article_prose.py write ic410 --writer api  # one backend
    python scripts/article_prose.py write --folder DIR        # a folder with article_request.json + the picture
    python scripts/article_prose.py bench --models gemma4:31b-it-q8_0,qwen3-vl:32b-instruct-q8_0
    python scripts/article_prose.py dossier ic410             # the reference block alone

astro_output.handle_image calls compose() for every new draft, and
publish_gate.card_text prints what came back: who wrote it, the claims it
made from general knowledge, and what the reviewer still objects to.

Writers, in the order "auto" tries them (cfg["prose"]["writer"]):
  local  an open-weight vision model served by Ollama on this machine
         (cfg["prose"]["local_model"], cfg["prose"]["local_url"]); the
         server is started when the port is silent (`ollama serve`, the
         user-level install under ~/.local/ollama or on PATH) and the model
         must already be pulled. No account, no key: it outlives everything
         in docs/SUCCESSION.md but the electricity.
  api    the Anthropic SDK with cfg["anthropic"]["api_key"] or
         ANTHROPIC_API_KEY (cfg["prose"]["api_model"]).
  cli    `claude -p` on this machine with the owner's Claude Code login,
         the picture read through its Read tool (cfg["prose"]["cli_model"]).
"off", or none available, or every one failing: None, and
astro_output.fallback_writeup is the draft, facts only (the card says so).

The loop: write (the facts, the reference block, the picture, the newest
gallery entries as the voice) -> review (a second call, text only: measured
numbers contradicted, a statement about the object that the reference does
not support, a style break, a missing note link) -> one rewrite with the
problems -> review again. What remains rides the article as `review_notes`,
and the writer's own list of statements from general knowledge as `claims`;
the card shows both, since the owner's `approve` is the last check and the
site is read by subscribers.

The reviewer is a DIFFERENT model when there is one (the owner's call,
2026-11-06, "the 2 model adversarial approach"): cfg["prose"]["review_backend"]
(local | api | cli | same) and cfg["prose"]["review_model"] (the local
reviewer's tag; qwen3-vl against the gemma4 writer by default). A model
asked to check its own draft tends to agree with itself; a second one has
no stake in the sentences. The reviewer sees the facts, the reference and
the draft, never the picture, so it judges against the record. A reviewer
that cannot answer (not pulled, no server, no key) hands the review back to
the writer and the log says so; the article records `reviewer`.

The reference block (dossier()): the Wikipedia intro for the object (by its
name, then its catalogue aliases), the object type, the distance on record
in local/dso_distances.json, the related notes. A local model is told to say
nothing about the object that is not in the reference or well established,
and to list everything that is not in the reference as a claim; the measured
frame counts, hours and nights are never the model's to change (the meta
line is written from the data, never from the article).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

DEFAULTS = {
    "writer": "auto",                 # auto | local | api | cli | off
    "local_url": "http://127.0.0.1:11434",
    "local_model": "gemma4:31b-it-q8_0",
    "local_start": True,              # start `ollama serve` when the port is silent
    "local_num_ctx": 32768,           # the examples alone are ~10k tokens; Ollama's default is 4k
    "local_think": False,             # a thinking model's private reasoning pass (gemma4: 3,000 tokens, 13 min a call)
    "api_model": "claude-opus-5-5",
    "cli_model": "opus",
    "review": True,
    "review_backend": "local",        # who checks the draft: local | api | cli | same (the writer)
    "review_model": "qwen3-vl:32b-instruct-q8_0",   # the local reviewer; the writer when it is not pulled
    "timeout_s": 900,
}
OLLAMA_CANDIDATES = (Path.home() / ".local" / "ollama" / "bin" / "ollama",)
PROMPT_IMAGE_MAX_PX = {"local": 1024, "api": 1600, "cli": 1600}
WRITE_TEMPERATURE = 0.5
REVIEW_TEMPERATURE = 0.2
WIKI_UA = "IrisObservatory/1.0 (irisscience.org; gallery prose) python-urllib"
WIKI_MAX_CHARS = 2500
ARTICLE_FIELDS = ("title", "distance_ly", "distance_text", "type_text", "alt",
                  "paragraphs", "time_travel_title", "time_travel")
EXTRA_FIELDS = ("claims",)
LOG_DIR = Path(_root) / "local" / "article_prose"


class ProseError(Exception):
    """A backend could not answer; the reason is the message."""


# ------------------------------------------------------------------ settings

def settings(cfg: Optional[dict] = None) -> dict:
    """DEFAULTS under cfg["prose"], plus api_key from cfg["anthropic"] or the environment."""
    st = dict(DEFAULTS)
    st.update((cfg or {}).get("prose") or {})
    st["api_key"] = ((cfg or {}).get("anthropic") or {}).get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or ""
    return st


def ollama_bin() -> Optional[str]:
    p = shutil.which("ollama")
    if p:
        return p
    for c in OLLAMA_CANDIDATES:
        if c.exists():
            return str(c)
    return None


def writer_order(st: dict, have_local: bool, have_key: bool, have_sdk: bool, have_cli: bool) -> list[str]:
    """The backends to try, in order, from the setting and what the machine has. Pure."""
    want = str(st.get("writer") or "auto").lower()
    avail = []
    if have_local:
        avail.append("local")
    if have_key and have_sdk:
        avail.append("api")
    if have_cli:
        avail.append("cli")
    if want == "off":
        return []
    if want == "auto":
        return avail
    return [want] if want in avail else []


# ------------------------------------------------------------------ the reference block

def _wiki_title_candidates(dso: str, aliases: list[str]) -> list[str]:
    """Page titles to try, best first: the display name, then the catalogue
    aliases SIMBAD gave (NAME x -> x; M 1 -> Messier 1)."""
    out: list[str] = []

    def add(t: str) -> None:
        t = re.sub(r"\s+", " ", t).strip()
        m = re.fullmatch(r"M\s*(\d+)", t, re.I)
        if m:
            t = f"Messier {m.group(1)}"
        if t and t not in out:
            out.append(t)

    m0 = re.fullmatch(r"(sh2|[a-z]+)[\s-]*(\d+[a-z]?)", dso.strip().lower())
    if m0:
        pre, num = m0.groups()
        pre = {"m": "Messier", "sh2": "Sh 2-"}.get(pre, pre.upper() if len(pre) <= 3 else pre.title())
        add(pre + ("" if pre.endswith("-") else " ") + num.upper())
    else:
        add(dso)
    for a in aliases or []:
        a = str(a).strip()
        if a.upper().startswith("NAME "):
            add(a[5:])
    for a in aliases or []:
        a = str(a).strip()
        if re.match(r"(NGC|IC|M|Sh\s*2-|Melotte|Cl Melotte|Abell|Arp)\b", a, re.I):
            add(re.sub(r"^Cl\s+", "", a))
    return out[:8]


def _wiki_intro(title: str, timeout: float = 15.0) -> Optional[dict]:
    """{title, description, extract, url} of a page's intro, or None (missing,
    a disambiguation, or no network)."""
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts|description", "explaintext": 1,
                                "exintro": 1, "redirects": 1, "format": "json", "titles": title})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q, headers={"User-Agent": WIKI_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    pages = (d.get("query") or {}).get("pages") or {}
    for p in pages.values():
        if "missing" in p or not (p.get("extract") or "").strip():
            continue
        text = re.sub(r"\s+\n", "\n", p["extract"]).strip()
        if "may refer to" in text[:300]:
            continue
        return {"title": p.get("title"), "description": p.get("description"),
                "extract": text[:WIKI_MAX_CHARS],
                "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(str(p.get("title")).replace(" ", "_"))}
    return None


def _wiki_search(name: str, timeout: float = 15.0, limit: int = 3) -> list[str]:
    """Page titles Wikipedia's search returns for *name*, lists and catalogues left out."""
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": name, "srlimit": limit, "format": "json"})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q, headers={"User-Agent": WIKI_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    out = []
    for hit in ((d.get("query") or {}).get("search") or []):
        t = str(hit.get("title") or "")
        if t and not re.match(r"(List of|Lists of)\b", t) and "catalogue" not in t.lower() and "catalog" not in t.lower():
            out.append(t)
    return out


def dossier(dso: str, aliases: Optional[list[str]] = None, log: Callable[[str], None] = print,
            fetch=_wiki_intro, search=_wiki_search) -> dict:
    """The reference block for the writer: {wikipedia: {...} | None, tried: [...]}.
    The page by name, then by alias, then (2026-11-06, Melotte 15 with no
    aliases on file: no page of its own, the Heart Nebula's is the one) the
    top search results whose intro names the object. Best effort; never raises."""
    tried = []
    for t in _wiki_title_candidates(dso, aliases or []):
        tried.append(t)
        try:
            hit = fetch(t)
        except Exception:  # noqa: BLE001
            hit = None
        if hit:
            log(f"  reference: Wikipedia '{hit['title']}' ({len(hit['extract'])} chars)")
            return {"wikipedia": hit, "tried": tried}
    name = tried[0] if tried else dso
    try:
        found = search(name)
    except Exception:  # noqa: BLE001
        found = []
    for t in found:
        if t in tried:
            continue
        tried.append(t)
        try:
            hit = fetch(t)
        except Exception:  # noqa: BLE001
            hit = None
        if hit and name.lower() in (hit.get("extract") or "").lower():
            log(f"  reference: Wikipedia '{hit['title']}' by search for '{name}' ({len(hit['extract'])} chars)")
            return {"wikipedia": hit, "tried": tried, "by_search": name}
    log(f"  reference: no Wikipedia intro for {', '.join(tried) or dso}; the writer has the facts only")
    return {"wikipedia": None, "tried": tried}


# ------------------------------------------------------------------ prompts

def light_left_words(distance_ly) -> str:
    """When the light left, from a distance in light years: "about 500 BC",
    "about AD 1500", "about 70 million years ago". "" when unknown."""
    try:
        d = float(distance_ly)
    except (TypeError, ValueError):
        return ""
    if d <= 0:
        return ""
    if d >= 1e6:
        return f"about {d / 1e6:g} million years ago"
    if d >= 20000:
        return f"about {round(d / 1000):,} thousand years ago"
    step = 100 if d >= 1000 else 10                 # the box says "about"; a round figure reads right
    year = int(round((2026 - d) / step) * step)
    return f"about AD {year}" if year > 0 else f"about {-year:,} BC"


AMERICAN = {"color": "colour", "colors": "colours", "colorful": "colourful", "colored": "coloured", "colorless": "colourless",
            "center": "centre", "centered": "centred", "centers": "centres", "catalog": "catalogue", "cataloged": "catalogued",
            "catalogs": "catalogues", "gray": "grey", "grays": "greys", "sulfur": "sulphur", "artifact": "artefact",
            "artifacts": "artefacts", "neighbor": "neighbour", "neighbors": "neighbours", "neighboring": "neighbouring",
            "favorite": "favourite", "traveled": "travelled", "traveling": "travelling", "modeled": "modelled",
            "labeled": "labelled", "fiber": "fibre", "fibers": "fibres", "honor": "honour", "meters": "metres",
            "kilometers": "kilometres", "liters": "litres", "harbor": "harbour", "behavior": "behaviour", "vapor": "vapour",
            "ionized": "ionised", "ionizes": "ionises", "ionization": "ionisation", "ionizing": "ionising",
            "analyze": "analyse", "analyzed": "analysed", "recognize": "recognise", "recognized": "recognised",
            "vaporized": "vaporised", "skeptical": "sceptical", "theater": "theatre"}
_AMERICAN_RE = re.compile(r"\b(" + "|".join(sorted(AMERICAN, key=len, reverse=True)) + r")\b", re.I)


def britishise(text: str) -> tuple[str, list[str]]:
    """*text* with the unambiguous American spellings replaced (case kept
    for a capital), and the words changed. A capitalised word inside a
    sentence is a name ("Center for Astrophysics") and is left alone;
    "meter" (the instrument), "program", "practice" are not in the list."""
    changed = []

    def sub(m):
        w = m.group(1)
        before = text[:m.start()].rstrip()
        if w[0].isupper() and before and before[-1] not in ".!?":
            return w
        b = AMERICAN[w.lower()]
        changed.append(w)
        return b.capitalize() if w[0].isupper() else b
    return _AMERICAN_RE.sub(sub, text), changed


def apply_spelling(art: dict) -> list[str]:
    """British spelling on every prose field of *art*, in place; the words changed."""
    words: list[str] = []
    for k in ("title", "alt", "time_travel", "time_travel_title"):
        if isinstance(art.get(k), str):
            art[k], w = britishise(art[k]); words += w
    if isinstance(art.get("paragraphs"), list):
        out = []
        for ptxt in art["paragraphs"]:
            t, w = britishise(str(ptxt)); out.append(t); words += w
        art["paragraphs"] = out
    return words


YEAR_RULE = ("The 'On Earth' box is set at the year the light left, which is the distance in light years "
             "counted back from 2026: 2,500 light years is about 500 BC (not AD 500), 1,500 light years "
             "is about AD 500, 70 million light years is 70 million years ago.")


def write_prompt(facts: dict, examples: list[str]) -> tuple[str, str]:
    """(system, user) for the write-up. The examples are live gallery entries,
    so the voice and the shape are the site's own."""
    system = (
        "You write the gallery entries for irisscience.org, the lab notes of a small robotic "
        "observatory. Each entry is two paragraphs for a college-educated reader who is not an "
        "astronomer: the first about the object itself (what it is, where, why it looks the way it "
        "does, with the distance), the second about this picture (the filters and processing given "
        "in the facts, in plain words, and what is visible in it), then a short 'On Earth' box about "
        "what was happening on Earth when the light left. Plain prose, no markdown, no HTML, no "
        "hedging, no exclamation marks, no em-dashes, no bullet points. British spelling throughout "
        "(colour, centre, catalogue, sulphur, artefact, grey, metres, ionised).\n\n"
        "Rules about truth. The facts carry a reference block (a Wikipedia introduction when one "
        "exists, the object type, a distance on record when the owner has recorded one). Everything "
        "you say about the object must come from that reference or be a fact you are certain is well "
        "established in the astronomical literature; when unsure, leave it out. Never invent a common "
        "name, a discovery story, a size, an age or a number. List every statement about the object "
        "that is NOT in the reference block in \"claims\", so a human can check them before the entry "
        "is published. The frame counts, hours, nights, filters and exposure in the facts are measured "
        "and must not be changed or rounded. Use the distance on record when there is one. "
        + YEAR_RULE + " Nothing about the observatory's own work with this target may be invented either: "
        "the processing in the facts and the titles of the related notes are all you know of it; link a "
        "note, never describe what it found or why the target was chosen.\n\n"
        "Entries from the site follow as examples of the voice and length.\n\n"
        + "\n\n".join(examples)
    )
    when = light_left_words(facts.get("distance_ly_on_record"))
    user = (
        "Facts for the new entry (JSON):\n" + json.dumps(facts, indent=1, ensure_ascii=False) +
        (f"\n\nThe light in this picture left the object {when}." if when else "") +
        "\n\nThe picture itself is attached. Answer with ONE JSON object and nothing else, with "
        "exactly these keys:\n"
        '  "title": like "NGC2146 - The Dusty Hand" (catalogue id without a space, a dash, the common name or type)\n'
        '  "distance_ly": integer light years (the distance on record when given; else the accepted value)\n'
        '  "distance_text": as in the meta line, e.g. "~70 MILLION LIGHT YEARS"\n'
        '  "type_text": upper case, e.g. "STARBURST BARRED SPIRAL GALAXY"\n'
        '  "alt": one sentence describing what is visible in this picture, for a blind reader\n'
        '  "paragraphs": a list of exactly two paragraphs (strings); when "related_notes" lists '
        'science notes on the site that used this target, the second paragraph links to each as '
        '[its title](#its id), the way the examples link to theirs\n'
        '  "time_travel_title": e.g. "On Earth: ~70 Million Years Ago"\n'
        '  "time_travel": one paragraph\n'
        '  "claims": a list of short strings, each a statement about the object in your text that is '
        'not in the reference block (empty when every statement is)\n'
    )
    return system, user


def review_prompt(facts: dict, article: dict) -> tuple[str, str]:
    """(system, user) for the check. Text only; the reviewer lists problems and never rewrites."""
    system = (
        "You check a draft gallery entry for irisscience.org before it is published to subscribers. "
        "The draft was written by another model; trust the measured facts and the reference block, "
        "never the draft. You are given the facts, the reference and the draft. Your answer is a list "
        "of DEFECTS the writer must fix, and nothing else: no checklist of what the draft does right, "
        "no sentence that says something is correct, consistent, matches or is as required. If the "
        "draft is fine the list is empty. A defect is one of:\n"
        "1. a measured number or date in the draft that contradicts the facts (frame counts, hours, "
        "nights, filters, exposure, recipe, dates), or an 'On Earth' year that does not fit the "
        "distance (" + YEAR_RULE + ");\n"
        "2. a statement about the object that contradicts the reference block, or that is not in it "
        "and not well established (a wrong distance, an invented common name, feature, size, age or "
        "discovery), a reference to a note or an entry that the facts do not list, or a statement about "
        "what the observatory did with this target (why it was chosen, what a note found) that the facts "
        "do not state;\n"
        "3. a style break: markdown, HTML, an exclamation mark, an em-dash, hedging, a bullet list, "
        "more or fewer than two paragraphs, American spelling (color, center, catalog, sulfur), a title "
        "not of the form 'NGC2146 - The Dusty Hand' (the catalogue id without a space, a dash, the name);\n"
        "4. a related note listed in the facts that the second paragraph does not link as [title](#id).\n"
        "Answer with ONE JSON object and nothing else: {\"problems\": [{\"class\": 1, \"quote\": "
        "\"the draft's exact words at fault\", \"problem\": \"what is wrong and what it should be\", "
        "\"must_change\": true}, ...]}, or {\"problems\": []}. An item whose must_change is false, or "
        "whose problem says the draft is right, confirmed, plausible or has no defect, does not belong "
        "in the list. A span of dates ('five nights between 27 October and 1 November') states the first "
        "and last night, not that every night between was used; it is not a defect. Do not rewrite the draft."
    )
    art = {k: article.get(k) for k in ARTICLE_FIELDS + EXTRA_FIELDS if k in article}
    when = light_left_words(facts.get("distance_ly_on_record"))
    user = ("Facts (JSON):\n" + json.dumps(facts, indent=1, ensure_ascii=False) +
            (f"\n\nThe light in this picture left the object {when}; an 'On Earth' box set there is right."
             if when else "") +
            "\n\nDraft (JSON):\n" + json.dumps(art, indent=1, ensure_ascii=False))
    return system, user


# ------------------------------------------------------------------ answers

def parse_answer(text: str) -> Optional[dict]:
    """The one JSON object in a model's answer (fences and chatter tolerated), else None."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            v = json.loads(t[i:j + 1])
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def missing_fields(art: Optional[dict]) -> list[str]:
    if not isinstance(art, dict):
        return list(ARTICLE_FIELDS)
    out = [k for k in ARTICLE_FIELDS if k not in art]
    if not isinstance(art.get("paragraphs"), list) or len(art.get("paragraphs") or []) != 2:
        out.append("paragraphs(2)")
    return out


def _image_b64(image: Path, max_px: int) -> tuple[str, str]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image).convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ backends

def _http_json(url: str, payload: Optional[dict] = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def local_models(url: str, timeout: float = 5.0) -> Optional[list[str]]:
    """The models the Ollama server at *url* holds, or None when it does not answer."""
    try:
        d = _http_json(url.rstrip("/") + "/api/tags", timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [m.get("name", "") for m in d.get("models") or []]


def ensure_local(st: dict, log: Callable[[str], None] = print, wait_s: float = 45.0) -> list[str]:
    """The local server up (started when silent and allowed) -> its model list.
    ProseError when it cannot be reached or the configured model is not pulled."""
    url = st["local_url"]
    models = local_models(url)
    if models is None:
        binp = ollama_bin()
        if not st.get("local_start", True) or not binp:
            raise ProseError(f"no Ollama server at {url}" + ("" if binp else " and no ollama binary to start one"))
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = open(LOG_DIR / "ollama_serve.log", "ab")
        env = dict(os.environ)
        env.setdefault("OLLAMA_HOST", urllib.parse.urlparse(url).netloc)
        subprocess.Popen([binp, "serve"], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
        log(f"  local model: started `ollama serve` ({binp})")
        t0 = time.time()
        while models is None and time.time() - t0 < wait_s:
            time.sleep(1.0)
            models = local_models(url)
        if models is None:
            raise ProseError(f"`ollama serve` did not answer at {url} within {wait_s:.0f} s")
    want = st["local_model"]
    if want not in models and f"{want}:latest" not in models:
        raise ProseError(f"model {want} is not pulled (have: {', '.join(models) or 'none'}); run `ollama pull {want}`")
    return models


_CAPS: dict[tuple[str, str], list[str]] = {}


def local_capabilities(url: str, model: str) -> list[str]:
    """What the server says the model can do ("vision", "thinking", ...); cached; [] when unknown."""
    key = (url, model)
    if key not in _CAPS:
        try:
            d = _http_json(url.rstrip("/") + "/api/show", {"model": model}, timeout=20)
            _CAPS[key] = [str(c) for c in d.get("capabilities") or []]
        except (urllib.error.URLError, OSError, ValueError):
            return []
    return _CAPS[key]


def local_payload(st: dict, system: str, user: str, images: list[str], caps: list[str]) -> dict:
    """The /api/chat request. A thinking model (gemma4) reasons to itself
    before the answer unless told not to: 3,566 tokens and 797 s for a 600
    token article on 2026-11-06, at the same tokens per second as qwen. The
    `think` key is sent only to a model that has the capability; Ollama
    refuses it on one that does not."""
    msg: dict = {"role": "user", "content": user}
    if images:
        msg["images"] = images
    payload = {"model": st["local_model"], "stream": False, "format": "json", "keep_alive": "15m",
               "options": {"temperature": float(st.get("local_temperature", WRITE_TEMPERATURE)),
                           "num_ctx": int(st.get("local_num_ctx") or 32768)},
               "messages": [{"role": "system", "content": system}, msg]}
    if "thinking" in caps:
        payload["think"] = bool(st.get("local_think", False))
    return payload


def ask_local(system: str, user: str, image: Optional[Path], st: dict, log: Callable[[str], None] = print) -> str:
    """One answer from the Ollama chat API, JSON-formatted, the picture attached."""
    ensure_local(st, log)
    images = [_image_b64(image, PROMPT_IMAGE_MAX_PX["local"])[1]] if image is not None else []
    payload = local_payload(st, system, user, images, local_capabilities(st["local_url"], st["local_model"]))
    try:
        d = _http_json(st["local_url"].rstrip("/") + "/api/chat", payload, timeout=float(st.get("timeout_s") or 900))
    except urllib.error.HTTPError as e:
        raise ProseError(f"local model HTTP {e.code}: {e.read()[:300]!r}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ProseError(f"local model: {e}") from e
    if d.get("error"):
        raise ProseError(f"local model: {d['error']}")
    text = ((d.get("message") or {}).get("content") or "").strip()
    secs = (d.get("total_duration") or 0) / 1e9
    thought = len((d.get("message") or {}).get("thinking") or "")
    log(f"  local model {st['local_model']}: {d.get('eval_count', '?')} tokens in {secs:.0f} s"
        + (f" ({thought} chars of thinking)" if thought else ""))
    if not text:
        raise ProseError("local model returned an empty answer")
    return text


def ask_api(system: str, user: str, image: Optional[Path], st: dict, log: Callable[[str], None] = print) -> str:
    """One answer from the Anthropic API (the server-side refusal fallback on)."""
    try:
        import anthropic
    except ImportError as e:
        raise ProseError("the anthropic package is not installed (uv pip install anthropic)") from e
    if not st.get("api_key"):
        raise ProseError("no cfg['anthropic']['api_key'] and no ANTHROPIC_API_KEY")
    content: list = []
    if image is not None:
        media, data = _image_b64(image, PROMPT_IMAGE_MAX_PX["api"])
        content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
    content.append({"type": "text", "text": user})
    try:
        client = anthropic.Anthropic(api_key=st["api_key"])
        resp = client.beta.messages.create(
            model=st["api_model"], max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system=system, messages=[{"role": "user", "content": content}])
    except anthropic.RateLimitError as e:
        raise ProseError(f"rate limited ({e.message}); next run tries again") from e
    except anthropic.APIStatusError as e:
        raise ProseError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise ProseError(f"no connection to the API ({e})") from e
    if resp.stop_reason == "refusal":
        why = resp.stop_details.explanation if getattr(resp, "stop_details", None) else "?"
        raise ProseError(f"the model declined ({why})")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def cli_result(envelope: str) -> str:
    """The answer text out of `claude -p --output-format json`; ProseError on an error envelope."""
    try:
        d = json.loads(envelope)
    except json.JSONDecodeError as e:
        raise ProseError(f"claude -p answered no JSON envelope: {envelope[:200]!r}") from e
    if d.get("is_error"):
        raise ProseError(f"claude -p error: {str(d.get('result'))[:300]}")
    return str(d.get("result") or "").strip()


def ask_cli(system: str, user: str, image: Optional[Path], st: dict, log: Callable[[str], None] = print) -> str:
    """One answer from `claude -p` with the owner's login; the picture read by its Read tool."""
    binp = shutil.which("claude")
    if not binp:
        raise ProseError("no `claude` on PATH")
    args = [binp, "-p", "--output-format", "json", "--model", str(st["cli_model"]),
            "--append-system-prompt", system, "--max-turns", "6"]
    prompt = user
    if image is not None:
        args += ["--allowedTools", "Read"]
        prompt += f"\n\nThe picture is the file {image.resolve()}: read it with the Read tool first."
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    try:
        r = subprocess.run(args, input=prompt, capture_output=True, text=True, env=env,
                           cwd=str(image.parent if image is not None else Path.home()),
                           timeout=float(st.get("timeout_s") or 900))
    except subprocess.TimeoutExpired as e:
        raise ProseError("claude -p timed out") from e
    if r.returncode != 0 and not r.stdout.strip():
        raise ProseError(f"claude -p exit {r.returncode}: {r.stderr.strip()[:300]}")
    return cli_result(r.stdout)


ASK = {"local": ask_local, "api": ask_api, "cli": ask_cli}


# ------------------------------------------------------------------ the loop

def _model_name(name: str, st: dict) -> str:
    return {"local": st["local_model"], "api": st["api_model"], "cli": f"claude-code:{st['cli_model']}"}[name]


def reviewer_for(name: str, st: dict) -> tuple[str, dict]:
    """(backend, settings) of the reviewer for a draft written by *name*: the
    configured one (review_backend / review_model), the writer itself when
    the setting is "same" or names the writer's own model. Pure."""
    rb = str(st.get("review_backend") or "same").lower()
    rst = dict(st, local_temperature=REVIEW_TEMPERATURE)
    if rb not in ASK:
        return name, rst
    if rb == "local" and st.get("review_model"):
        rst["local_model"] = str(st["review_model"])
    return rb, rst


PRAISE_RE = re.compile(r"\b(is|are|was|were)\s+(correct|consistent|accurate|appropriate|fine|as required|present|plausible)\b"
                       r"|\bmatches\b|\bno (contradiction|problem|issue|change|defect)s?\b|\bnot a (problem|defect|contradiction)\b"
                       r"|\bcorrectly\b|\bconsistent with\b|\bas required\b|\bnothing to fix\b|\bconfirms?\b"
                       r"|\bnot contradicted\b|\bcommonly accepted\b|\bwell[- ]established\b|\bsupported by\b", re.I)
DEFECT_RE = re.compile(r"\b(should|must|wrong|contradicts?|invent(ed|s)?|missing|not in the (facts|reference)|instead|remove|drop|replace)\b", re.I)


def problem_lines(probs) -> tuple[list[str], int]:
    """The reviewer's list as note strings, praise dropped: (notes, dropped).
    An item is a string or {"class", "quote", "problem"}; a model in
    checklist mode (qwen3-vl, 2026-11-06: 23 "problems", every one a
    sentence saying the draft was right) reports what the draft does well,
    and a note like that sent to the writer as a problem makes it change
    what was right."""
    if not isinstance(probs, list):
        return [], 0
    out, dropped = [], 0
    for p in probs:
        if isinstance(p, dict):
            if p.get("must_change") is False:          # the reviewer's own verdict first
                dropped += 1
                continue
            text = str(p.get("problem") or p.get("what") or "").strip()
            quote = str(p.get("quote") or "").strip()
            if quote and quote.lower() not in text.lower():
                text = f"{text} (draft: \"{quote[:120]}\")" if text else ""
        else:
            text = str(p).strip()
        if not text:
            continue
        if PRAISE_RE.search(text) and not DEFECT_RE.search(text):
            dropped += 1
            continue
        out.append(text)
    return out, dropped


def review(name: str, facts: dict, article: dict, st: dict, ask, log: Callable[[str], None]) -> list[str]:
    """The reviewer's problems (empty when fine). ProseError when it cannot answer."""
    system, user = review_prompt(facts, article)
    text = ask(name, system, user, None, st, log)
    d = parse_answer(text) or {}
    out, dropped = problem_lines(d.get("problems") if isinstance(d, dict) else None)
    log(f"  review ({_model_name(name, st)}): {len(out)} problem(s)" + (f", {dropped} non-problem(s) dropped" if dropped else "")
        + (": " + " | ".join(out) if out else ""))
    return out


def reviewed(name: str, facts: dict, article: dict, st: dict, ask, log: Callable[[str], None]) -> tuple[list[str], str]:
    """(problems, who reviewed): the other model when it answers, else the
    writer on its own draft, else nothing (logged, never fatal)."""
    rname, rst = reviewer_for(name, st)
    own = (rname, _model_name(rname, rst)) == (name, _model_name(name, st))
    if not own:
        try:
            return review(rname, facts, article, rst, ask, log), _model_name(rname, rst)
        except ProseError as e:
            log(f"  reviewer {_model_name(rname, rst)} cannot answer ({e}); the writer reviews its own draft")
    try:
        return review(name, facts, article, st, ask, log), _model_name(name, st)
    except ProseError as e:
        log(f"  review skipped: {e}")
        return [], ""


def _write_once(name: str, system: str, user: str, image: Optional[Path], st: dict, ask, log) -> dict:
    """One draft from a backend, the JSON asked for; a second try when the first is not."""
    for attempt in (1, 2):
        text = ask(name, system, user, image, st, log)
        art = parse_answer(text)
        miss = missing_fields(art)
        if not miss:
            return art  # type: ignore[return-value]
        log(f"  write ({name}, try {attempt}): answer lacks {miss}")
        user = user + "\n\nYour previous answer was not the JSON object described above (it lacked " + \
            ", ".join(miss) + "). Answer again with only that JSON object."
    raise ProseError(f"{name}: no complete article in two tries")


def compose(facts: dict, image: Optional[Path], examples: list[str], cfg: Optional[dict] = None,
            log: Callable[[str], None] = print, ask=None, st: Optional[dict] = None) -> Optional[dict]:
    """The article from the first backend that answers (write, review, one
    rewrite, review), or None with every reason logged. *ask(name, system,
    user, image, st, log) -> text* replaces the real backends in tests."""
    st = st or settings(cfg)
    real = ask is None

    def _ask(name, system, user, img, st_, log_):
        return ASK[name](system, user, img, st_, log_) if real else ask(name, system, user, img, st_, log_)

    have_sdk = True
    if real:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            have_sdk = False
    order = writer_order(st, have_local=(not real) or (st.get("writer") in ("auto", "local")),
                         have_key=bool(st.get("api_key")), have_sdk=have_sdk,
                         have_cli=(not real) or shutil.which("claude") is not None)
    if not order:
        log(f"  write-up: no model backend (writer={st.get('writer')}); facts only")
        return None
    system, user = write_prompt(facts, examples)
    for name in order:
        t0 = time.time()
        try:
            art = _write_once(name, system, user, image, st, _ask, log)
            notes: list[str] = []
            who = ""
            if st.get("review", True):
                notes, who = reviewed(name, facts, art, st, _ask, log)
                if notes:
                    fix = user + "\n\nA reviewer found these problems in a previous draft. Write a new draft " \
                        "that fixes every one that is real. The facts above and the year given with them outrank " \
                        "the reviewer: where a note contradicts them, keep the draft as it was.\n- " + "\n- ".join(notes)
                    art = _write_once(name, system, fix, image, st, _ask, log)
                    notes, who = reviewed(name, facts, art, st, _ask, log)
        except ProseError as e:
            log(f"  write-up ({name}): {e}")
            continue
        claims = art.get("claims")
        art["claims"] = [str(c).strip() for c in claims if str(c).strip()] if isinstance(claims, list) else []
        art["spelling_fixed"] = apply_spelling(art)
        if art["spelling_fixed"]:
            log(f"  spelling: {len(art['spelling_fixed'])} American spelling(s) replaced: {', '.join(art['spelling_fixed'])}")
        art["review_notes"] = notes
        art["reviewer"] = who
        art["source"] = name
        art["model"] = _model_name(name, st)
        art["seconds"] = round(time.time() - t0, 1)
        log(f"  write-up: {name} ({art['model']}) in {art['seconds']} s, {len(art['claims'])} claim(s), "
            f"{len(notes)} reviewer note(s)" + (f" from {who}" if who else ""))
        return art
    return None


# ------------------------------------------------------------------ the command line

def _folder_for(dso: Optional[str], folder: Optional[str]) -> Path:
    from scripts import astro_output as ao
    if folder:
        return Path(folder)
    if not dso:
        raise SystemExit("give a dso or --folder")
    return ao.DEFAULT_FOLDER / ao.squash(dso)


def _inputs(out_dir: Path) -> tuple[dict, Path, list[str]]:
    from scripts import astro_output as ao
    facts = json.loads((out_dir / "article_request.json").read_text())
    img = out_dir / str(facts.get("image") or "")
    if not img.exists():
        pics = [p for p in sorted(out_dir.iterdir()) if p.suffix.lower() in ao.IMAGE_SUFFIXES
                and not p.name.endswith(("_stats.jpg", "_card.jpg"))]
        if not pics:
            raise SystemExit(f"no picture in {out_dir}")
        img = pics[0]
    html = _site_html()
    me = ao.squash(str(facts.get("dso") or out_dir.name))
    return facts, img, ao.site_examples(html, exclude=(me,)) if html else []


def _site_html() -> str:
    from scripts import astro_output as ao
    p = ao.SITE / "index.html"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def cmd_write(a) -> int:
    from scripts import astro_output as ao
    out_dir = _folder_for(a.dso, a.folder)
    facts, img, examples = _inputs(out_dir)
    if "reference" not in facts or a.refresh_reference:
        facts["reference"] = dossier(facts.get("dso", ""), facts.get("aliases") or [])
        facts["distance_ly_on_record"] = ao.known_distance(facts.get("dso", ""), html=_site_html())
        (out_dir / "article_request.json").write_text(json.dumps(facts, indent=1))
    cfg = _cfg()
    st = settings(cfg)
    if a.writer:
        st["writer"] = a.writer
    if a.model:
        st["local_model"] = a.model
    if a.reviewer:
        st["review_model"] = a.reviewer
        st["review_backend"] = "same" if a.reviewer == "same" else "local"
    art = compose(facts, img, examples, cfg, st=st)
    if art is None:
        print("no article written (see the reasons above)")
        return 1
    prev = out_dir / "article.json"
    if prev.exists():
        shutil.copy2(prev, out_dir / "article.prev.json")
    prev.write_text(json.dumps(art, indent=1, ensure_ascii=False))
    print(json.dumps(art, indent=1, ensure_ascii=False))
    print(f"\nwritten: {prev}")
    draft = out_dir / "draft.json"
    if draft.exists():
        from scripts import publish_prepare as pp
        rec_p = pp.QUEUE_DIR / f"{ao.squash(facts.get('dso', ''))}.json"
        new_hash = ao.draft_hash(art, img.read_bytes())
        d = json.loads(draft.read_text())
        d.update({"draft_hash": new_hash, "status": "pending"})
        draft.write_text(json.dumps(d, indent=1))
        if rec_p.exists():
            rec = json.loads(rec_p.read_text())
            rec.update({"draft_hash": new_hash, "article_source": art["source"], "status": "pending"})
            rec_p.write_text(json.dumps(rec, indent=1))
            print(f"draft hash now {new_hash[:12]} (a new draft needs a new approve)")
            if a.announce:
                pp.announce_draft(rec, out_dir)
    return 0


def cmd_dossier(a) -> int:
    from scripts import astro_output as ao
    try:   # SIMBAD can hang for minutes (2026-11-05); the block is useful without it
        from utils.bounded import call_bounded
        aliases = call_bounded(ao.simbad_aliases, 20, a.dso) or []
    except Exception:  # noqa: BLE001
        aliases = []
    d = dossier(a.dso, aliases)
    print(json.dumps(d, indent=1, ensure_ascii=False))
    return 0


def cmd_bench(a) -> int:
    """Every model named, on every folder named (default: the folders with a
    hand-written article.json), the result beside the hand-written entry in
    one Markdown file for the owner to judge. Nothing in the folders changes."""
    from scripts import astro_output as ao
    cfg = _cfg()
    base = settings(cfg)
    models = [m.strip() for m in (a.models or base["local_model"]).split(",") if m.strip()]
    root = Path(a.folder) if a.folder else ao.DEFAULT_FOLDER
    dsos = [d.strip() for d in a.dsos.split(",")] if a.dsos else sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "article.json").exists()
        and (p / "article_request.json").exists())
    out_root = LOG_DIR / "bench"
    out_root.mkdir(parents=True, exist_ok=True)
    report = [f"# Article prose bench ({time.strftime('%Y-%m-%d %H:%M')})\n",
              f"Models: {', '.join(models)}; writer: {a.writer or 'local'}\n"]
    table = ["| target | model | reviewer | seconds | claims | reviewer notes |", "|---|---|---|---|---|---|"]
    for dso in dsos:
        out_dir = root / dso
        try:
            facts, img, examples = _inputs(out_dir)
        except SystemExit as e:
            report.append(f"\n## {dso}\n\nskipped: {e}\n")
            continue
        if "reference" not in facts:
            facts["reference"] = dossier(facts.get("dso", dso), facts.get("aliases") or [])
        facts["distance_ly_on_record"] = ao.known_distance(facts.get("dso", dso), html=_site_html())
        hand = json.loads((out_dir / "article.json").read_text()) if (out_dir / "article.json").exists() else None
        report.append(f"\n## {dso}\n")
        if hand:
            report.append("### Hand-written (on the site)\n\n**" + str(hand.get("title")) + "**\n\n"
                          + "\n\n".join(hand.get("paragraphs") or []) + "\n\n*" + str(hand.get("time_travel_title"))
                          + "*: " + str(hand.get("time_travel")) + "\n")
        for m in models:
            st = dict(base, writer=a.writer or "local", local_model=m)
            if a.reviewer:
                st["review_model"] = a.reviewer
                st["review_backend"] = "same" if a.reviewer == "same" else "local"
            elif st.get("review_model") == m:      # the other model on the bench checks this one
                st["review_model"] = next((x for x in models if x != m), m)
            logs: list[str] = []
            art = compose(facts, img, examples, cfg, log=lambda s: (logs.append(s), print(s)), st=st)
            mdir = out_root / re.sub(r"[^A-Za-z0-9_.-]", "_", m)
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / f"{dso}.json").write_text(json.dumps(art or {"error": logs}, indent=1, ensure_ascii=False))
            (mdir / f"{dso}.log").write_text("\n".join(logs))
            if art is None:
                report.append(f"### {m}\n\nfailed: {logs[-1] if logs else '?'}\n")
                table.append(f"| {dso} | {m} | - | - | - | failed |")
                continue
            report.append(f"### {m} ({art['seconds']} s; reviewed by {art.get('reviewer') or 'nobody'})\n\n"
                          f"**{art.get('title')}**  \n"
                          f"{art.get('distance_text')} | {art.get('type_text')}\n\n"
                          + "\n\n".join(art.get("paragraphs") or []) + f"\n\n*{art.get('time_travel_title')}*: "
                          f"{art.get('time_travel')}\n\n*Alt:* {art.get('alt')}\n\n"
                          f"*Claims:* {'; '.join(art['claims']) or 'none'}  \n"
                          f"*Reviewer:* {'; '.join(art['review_notes']) or 'no objections'}\n")
            table.append(f"| {dso} | {m} | {art.get('reviewer') or '-'} | {art['seconds']} | {len(art['claims'])} | "
                         f"{len(art['review_notes'])} |")
    report.insert(2, "\n".join(table) + "\n")
    out = Path(a.out) if a.out else out_root / "bench.md"
    out.write_text("\n".join(report))
    print(f"\nreport: {out}")
    return 0


def _cfg() -> Optional[dict]:
    try:
        from configs import config
        return config.data()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="(re)write a folder's article.json")
    w.add_argument("dso", nargs="?")
    w.add_argument("--folder")
    w.add_argument("--writer", choices=["auto", "local", "api", "cli"])
    w.add_argument("--model", help="the local model to use")
    w.add_argument("--reviewer", help="the local model that reviews (default cfg prose.review_model; `same` = the writer)")
    w.add_argument("--refresh-reference", action="store_true")
    w.add_argument("--announce", action="store_true", help="re-card a queued draft with its new hash")
    w.set_defaults(fn=cmd_write)
    d = sub.add_parser("dossier", help="print the reference block for a target")
    d.add_argument("dso")
    d.set_defaults(fn=cmd_dossier)
    b = sub.add_parser("bench", help="every model on every hand-written entry, one report")
    b.add_argument("--models", help="comma-separated local model tags")
    b.add_argument("--reviewer", help="one local model that reviews every writer (default: the other model on the bench)")
    b.add_argument("--dsos", help="comma-separated folder names (default: all with a hand-written article)")
    b.add_argument("--folder")
    b.add_argument("--writer", choices=["local", "api", "cli"])
    b.add_argument("--out")
    b.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
