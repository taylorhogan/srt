"""Render docs/MAINTENANCE.md and push it to the web host's unlisted /mm/.

The manual is served at https://irisscience.org/mm/ — deliberately UNLISTED,
not authenticated: no link on the site points at it, Caddy stamps it
X-Robots-Tag noindex, and it contains chores rather than secrets. Anyone who
is handed the URL (an heir, a house-sitter, future-you on a phone in the
observatory) can read it without a checkout.

Run manually after editing the manual:

    python scripts/push_mm.py            # render + push
    python scripts/push_mm.py --dry-run  # render to scratch, push nothing

The push reuses live_push's bounded scp transport with a different remote
directory (/srv/iris-mm), so it inherits the same timeouts and the same
tmp-then-rename atomicity as the live feed.
"""
import os
import sys
from datetime import date
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "MAINTENANCE.md"
OUT = ROOT / "iris_astronomy" / "scratch" / "mm_index.html"
DEST = "/srv/iris-mm"

# Print-first styling, matching the manual's job: a sheet for the wall, not a
# web page. Single light theme on purpose.
_STYLE = """
body { background:#f7f7f4; color:#1c1c1c; margin:0; padding:0 18px 60px;
       font: 15px/1.6 "Segoe UI", system-ui, sans-serif; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size: 24px; border-bottom: 3px solid #1c1c1c; padding: 28px 0 8px; }
h2 { font-size: 17px; margin-top: 34px; border-bottom: 1px solid #c9c9c2;
     padding-bottom: 4px; }
h3 { font-size: 15px; margin-top: 22px; }
table { border-collapse: collapse; font-size: 13.5px; margin: 12px 0; }
th, td { border: 1px solid #b9b9b0; padding: 5px 10px; text-align: left;
         vertical-align: top; }
th { background: #ebebe4; }
code { font-family: Consolas, monospace; font-size: 0.92em;
       background: #ecece6; padding: 1px 4px; border-radius: 3px; }
li { margin: 5px 0; }
.pushstamp { color: #777; font-size: 12px; margin-top: 40px;
             border-top: 1px solid #c9c9c2; padding-top: 8px; }
div.tablewrap { overflow-x: auto; }
@media print { body { background: #fff; font-size: 12px; } }
"""


def render() -> Path:
    import markdown
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables"])
    # Wide tables scroll rather than break the phone layout.
    body = body.replace("<table>", '<div class="tablewrap"><table>')
    body = body.replace("</table>", "</table></div>")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<meta name='robots' content='noindex, nofollow'>"
            "<title>Iris Maintenance Manual</title>"
            "<style>" + _STYLE + "</style></head><body><main>"
            + body
            + "<p class='pushstamp'>Rendered from docs/MAINTENANCE.md and "
              "pushed " + date.today().isoformat() + " by scripts/push_mm.py. "
              "The repo copy is the source of truth.</p>"
            "</main></body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> int:
    out = render()
    print("rendered", out, f"({out.stat().st_size} bytes)")
    if "--dry-run" in sys.argv:
        return 0
    from scripts import live_push
    live_push.push([(out, "index.html")], dest=DEST)
    print("pushed -> https://irisscience.org/mm/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
