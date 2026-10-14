"""Tonight's plan as a timeline: one row per event, time in one column.

Asked for 2026-10-08 in place of the noon report's paragraph: when the roof
opens and closes, when each target starts and in which filters, and when the
night is wrapped up (flats and the end-of-night analysis done).

Pure: the scheduler passes in sunset, the planner's slots and the filter
counts written into the sequence; nothing here reads the clock or the disk
(picture() writes only the path it is given).

The offsets are measured, not designed (iris.log, nights 10-03 .. 10-08):
  * roof open: 3-4 min after the pre-sunset check (vision + conductor + the
    11 s move), so about 6 min before sunset;
  * roof closed: 4-5 min after the end sequence begins (park, then close);
  * flats: 70-75 min after the close;
  * SNR + night-contribution analysis: 14-20 min after the flats, longer with
    more targets and frames.
Every derived time carries a "~" in the output.

Since 2026-10-14 the report is a picture and a table, not monospaced text:
Pushover sets text in a proportional font and wraps it, so the columns never
lined up. table() is the one model; picture() draws it for Pushover (JPEG,
which is what utils.pushover attaches it as), html() for the chat in the
`tonight` grid's dark palette, and render() keeps the one-line text form as
the fallback. The roof close is its own row, highlighted like the open.
"""
import html as _html
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

PRE_SUNSET_CHECK = timedelta(minutes=10)     # scheduler: 10 min before sunset
ROOF_OPEN_AFTER_CHECK = timedelta(minutes=4)
ROOF_CLOSED_AFTER_END = timedelta(minutes=5)
FLATS = timedelta(minutes=75)
ANALYSIS = timedelta(minutes=20)

_FILTER_ORDER = ("L", "R", "G", "B", "Ha", "O-III", "S-II")


@dataclass
class SlotPlan:
    name: str
    start: Optional[datetime]
    end: Optional[datetime]
    counts: dict                 # {"L": 18, "R": 9, ...} as written into the sequence
    exposure_s: dict             # {"L": 300.0, ...}


def _filters_text(counts: dict, exposure_s: dict) -> str:
    keys = sorted(counts, key=lambda f: (_FILTER_ORDER.index(f) if f in _FILTER_ORDER else 99, f))
    parts, total = [], 0.0
    for f in keys:
        n = int(counts[f])
        if n <= 0:
            continue
        exp = float(exposure_s.get(f) or 0.0)
        total += n * exp
        parts.append("%s %d" % (f, n))
    if not parts:
        return "template filters"
    exps = {float(exposure_s.get(f) or 0.0) for f in keys if counts.get(f)}
    each = (" x %.0fs" % exps.pop()) if len(exps) == 1 and 0.0 not in exps else ""
    return "%s%s (%.1fh)" % (", ".join(parts), each, total / 3600.0)


@dataclass
class Row:
    time: Optional[datetime]
    approx: bool                 # derived from measured offsets, shown with "~"
    event: str
    detail: str = ""
    kind: str = "info"           # check | roof | sun | target | end | wrap
    line: str = ""               # the one-line text form (render)


def table(sunset: datetime, slots: list, tz=None) -> list:
    """Tonight's rows in time order: the model behind every rendering."""
    loc = (lambda t: t.astimezone(tz)) if tz else (lambda t: t)
    rows = []
    check = sunset - PRE_SUNSET_CHECK
    rows.append(Row(loc(check), False, "Imaging check", "weather and plan re-checked", "check",
                    "Imaging check (weather and plan re-checked)"))
    rows.append(Row(loc(check + ROOF_OPEN_AFTER_CHECK), True, "Roof opens",
                    "camera cools, sequence starts", "roof",
                    "Roof opens, camera cools, sequence starts"))
    rows.append(Row(loc(sunset), False, "Sunset", "", "sun", "Sunset"))
    for i, s in enumerate(slots):
        filters = _filters_text(s.counts, s.exposure_s)
        line = "%s starts: %s" % (s.name, filters)
        if i > 0:
            line += " (earlier if %s finishes early and it is above the horizon)" % slots[i - 1].name
        rows.append(Row(loc(s.start) if s.start else None, False, "%s starts" % s.name,
                        filters, "target", line))
    end = slots[-1].end if slots and slots[-1].end else None
    if end is not None:
        rows.append(Row(loc(end), False, "Imaging ends", "", "end", "Imaging ends"))
        closed = end + ROOF_CLOSED_AFTER_END
        rows.append(Row(loc(closed), True, "Roof closes", "scope parked first", "roof",
                        "Scope parked, roof closed"))
        rows.append(Row(loc(closed + FLATS), True, "Flats done", "", "wrap", "Flats done"))
        rows.append(Row(loc(closed + FLATS + ANALYSIS), True, "All wrapped up",
                        "SNR and night analysis posted", "wrap",
                        "All wrapped up: SNR and night analysis posted"))
    else:
        rows.append(Row(None, False, "Imaging ends at dawn",
                        "then the roof closes (~5 min), flats (~75 min), analysis (~20 min)", "end",
                        "Imaging ends at dawn; roof close, flats (~75 min) and analysis (~20 min) follow"))
    return rows


def build(sunset: datetime, slots: list, tz=None) -> list:
    """[(time or None, approximate?, event), ...] in time order."""
    return [(r.time, r.approx, r.line) for r in table(sunset, slots, tz)]


def _stamp(r: Row) -> str:
    return "?" if r.time is None else ("~" if r.approx else "") + r.time.strftime("%H:%M")


def footnotes(rows) -> list:
    notes = ["~ estimated from past nights"]
    if sum(1 for r in rows if r.kind == "target") > 1:
        notes.append("a later target starts earlier if the one before it finishes early")
    return notes


def summary(rows, title: str) -> str:
    """Three lines for the Pushover text above the picture."""
    def first(event):
        r = next((r for r in rows if r.event == event and r.time is not None), None)
        return _stamp(r) if r else "?"
    targets = ", ".join("%s %s" % (r.event[:-len(" starts")], _stamp(r))
                         for r in rows if r.kind == "target")
    return "\n".join((title, "Roof opens %s, closes %s" % (first("Roof opens"), first("Roof closes")),
                      targets))


# The `tonight` grid's palette (iris_astronomy.astro_dso_visibility.build_grid_html).
BG_MAIN, BG_HDR, CLR_TEXT, CLR_DIM, CLR_ACCENT, CLR_BORDER = (
    "#0d1117", "#161b22", "#c9d1d9", "#8b949e", "#58a6ff", "#21262d")
BG_KIND = {"roof": "#3d2e00", "target": "#1a2e40"}
CLR_KIND = {"roof": "#f0c040", "target": "#e6edf3", "check": CLR_DIM, "sun": CLR_DIM, "wrap": CLR_DIM}


def html(rows, title: str) -> str:
    """The chat's table: time | event | details, roof rows amber, targets blue."""
    def td(bg, color, extra=""):
        return ('style="padding:5px 10px;border-bottom:1px solid %s;background:%s;color:%s;'
                'font-size:12px;%s"' % (CLR_BORDER, bg, color, extra))
    th = ('style="padding:5px 10px;border-bottom:2px solid %s;background:%s;color:%s;'
          'font-size:10px;text-transform:uppercase;letter-spacing:0.06em;text-align:left;"'
          % (CLR_BORDER, BG_HDR, CLR_DIM))
    out = ['<div style="font-size:13px;font-weight:600;color:%s;margin-bottom:8px;">%s</div>'
           % (CLR_ACCENT, _html.escape(title)),
           '<div style="overflow-x:auto;border-radius:6px;border:1px solid %s;">' % CLR_BORDER,
           '<table style="border-collapse:collapse;font-family:inherit;background:%s;">' % BG_MAIN,
           "<thead><tr><th %s>Time</th><th %s>Event</th><th %s>Details</th></tr></thead><tbody>"
           % (th, th, th)]
    for r in rows:
        bg = BG_KIND.get(r.kind, BG_MAIN)
        color = CLR_KIND.get(r.kind, CLR_TEXT)
        weight = "font-weight:600;" if r.kind in ("roof", "target") else ""
        out.append("<tr><td %s>%s</td><td %s>%s</td><td %s>%s</td></tr>" % (
            td(bg, color, "font-family:monospace;white-space:nowrap;"), _stamp(r),
            td(bg, color, weight + "white-space:nowrap;"), _html.escape(r.event),
            td(bg, CLR_TEXT if r.kind == "target" else CLR_DIM), _html.escape(r.detail)))
    out.append("</tbody></table></div>")
    out.append('<div style="font-size:11px;color:%s;margin-top:6px;">%s</div>'
               % (CLR_DIM, _html.escape("; ".join(footnotes(rows)))))
    return "".join(out)


def picture(rows, title: str, path) -> str:
    """Draw the table as a JPEG for Pushover. Returns *path*.

    Figure + Agg canvas directly, not pyplot: this runs in the scheduler,
    whose other plots go through pyplot's global state."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    W, pad, line_h = 8.0, 0.25, 0.27
    x_time, x_event, x_detail = pad, 1.15, 3.55
    wrapped = [textwrap.wrap(r.detail, 46) or [""] for r in rows]
    heights = [0.16 + line_h * len(w) for w in wrapped]
    top, hdr = 0.75, 0.38
    notes = footnotes(rows)
    H = top + hdr + sum(heights) + 0.2 + 0.24 * len(notes) + pad

    fig = Figure(figsize=(W, H), dpi=150, facecolor=BG_MAIN)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.text(pad, 0.42, title, color=CLR_ACCENT, fontsize=15, fontweight="bold", va="center")
    y = top
    ax.add_patch(Rectangle((pad - 0.1, y), W - 2 * pad + 0.2, hdr, color=BG_HDR, lw=0))
    for x, label in ((x_time, "TIME"), (x_event, "EVENT"), (x_detail, "DETAILS")):
        ax.text(x, y + hdr / 2, label, color=CLR_DIM, fontsize=9, fontweight="bold", va="center")
    y += hdr
    for r, w, h in zip(rows, wrapped, heights):
        if r.kind in BG_KIND:
            ax.add_patch(Rectangle((pad - 0.1, y), W - 2 * pad + 0.2, h, color=BG_KIND[r.kind], lw=0))
        ax.plot([pad - 0.1, W - pad + 0.1], [y + h, y + h], color=CLR_BORDER, lw=0.8)
        color = CLR_KIND.get(r.kind, CLR_TEXT)
        first = y + 0.08 + line_h / 2
        ax.text(x_time, first, _stamp(r), color=color, fontsize=12, family="monospace", va="center")
        ax.text(x_event, first, r.event, color=color, fontsize=12, va="center",
                fontweight="bold" if r.kind in ("roof", "target") else "normal")
        for j, part in enumerate(w):
            ax.text(x_detail, first + j * line_h, part, va="center", fontsize=11,
                    color=CLR_TEXT if r.kind == "target" else CLR_DIM)
        y += h
    y += 0.2
    for n in notes:
        ax.text(pad, y + 0.12, n, color=CLR_DIM, fontsize=9, va="center")
        y += 0.24
    fig.savefig(str(path), dpi=150, facecolor=BG_MAIN, pil_kwargs={"quality": 92})
    return str(path)


def render(rows, title: str = "Tonight") -> str:
    """Two columns: time, event. Approximate times are marked with ~."""
    lines = [title, "Time    Event"]
    for t, approx, event in rows:
        stamp = "  ?  " if t is None else t.strftime("%H:%M")
        lines.append("%s%s  %s" % ("~" if approx else " ", stamp, event))
    return "\n".join(lines)
