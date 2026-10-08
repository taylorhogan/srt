"""What is on the lab site: every note and gallery entry, as data.

    articles(html, base_url) -> [{id, kind, title, date, summary, image_url, url}, ...]

Read from index.html itself (stdlib html.parser), so the list the publisher
sends to the subscriber service is exactly what visitors see. Two markups:

  science note   <article id=SLUG class="note">      h2.note-title, span.note-date,
                 p.note-abstract, first <figure><img>
  gallery image  <article id=SLUG class="dso-section"> h2.dso-title, img.dso-image,
                 first p.science-text; no date

Image paths are made absolute against *base_url*, keeping any ?v= cache
buster. Nothing here decides what is NEW: the subscriber service on the web
host keeps the record of what has been announced (its `sent` table).
"""
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

KINDS = {"note": "note", "dso-section": "image"}


def _classes(attrs) -> set:
    return set((dict(attrs).get("class") or "").split())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _Parser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url.rstrip("/") + "/"
        self.out = []
        self.cur = None            # the article being read
        self.capture = None        # field name while inside a text-bearing element
        self.capture_depth = 0
        self.depth = 0             # element depth inside the article

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = _classes(attrs)
        if tag == "article":
            kind = next((KINDS[c] for c in cls if c in KINDS), None)
            if kind and a.get("id"):
                self.cur = {"id": a["id"], "kind": kind, "title": "", "date": None,
                            "summary": "", "image_url": None,
                            "url": self.base + "#" + a["id"]}
                self.depth = 0
            return
        if self.cur is None:
            return
        self.depth += 1
        if tag == "img" and self.cur["image_url"] is None and a.get("src"):
            self.cur["image_url"] = urljoin(self.base, a["src"])
        if self.capture:
            self.capture_depth += 1
            return
        if tag == "h2" and ("note-title" in cls or "dso-title" in cls):
            self._start("title")
        elif tag == "span" and "note-date" in cls:
            self._start("date")
        elif tag == "p" and "note-abstract" in cls:
            self._start("summary")
        elif tag == "p" and "science-text" in cls and not self.cur["summary"] \
                and self.cur["kind"] == "image":
            self._start("summary")

    def _start(self, field):
        self.capture = field
        self.capture_depth = 0
        self._buf = []

    def handle_endtag(self, tag):
        if self.cur is None:
            return
        if tag == "article":
            self.out.append(self.cur)
            self.cur = None
            self.capture = None
            return
        if self.capture:
            if self.capture_depth == 0:
                self.cur[self.capture] = _clean("".join(self._buf))
                self.capture = None
            else:
                self.capture_depth -= 1
        if tag != "img":
            self.depth -= 1

    def handle_data(self, data):
        if self.cur is not None and self.capture:
            self._buf.append(data)


def articles(html: str, base_url: str = "https://irisscience.org") -> list:
    """Every note and gallery entry in *html*, in page order. Pure."""
    p = _Parser(base_url)
    p.feed(html)
    p.close()
    return p.out


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "index.html"
    with open(path, encoding="utf-8") as fh:
        print(json.dumps({"articles": articles(fh.read())}, indent=1))
