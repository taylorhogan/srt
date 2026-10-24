"""Mock the Core2 screen: real summarize() payloads, painted the way core2_dash.ino paints.

Run from the repo root: .venv/Scripts/python.exe devices/core2_dash/mock_states.py
Writes states.png beside this file. Keep the painter in step with the sketch.
"""
import os
import sys
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cmd_processing import dash

FONT = r"C:\Windows\Fonts\consolab.ttf"
GREEN, ORANGE, RED = (0, 252, 0), (255, 180, 0), (255, 0, 0)
DGREY, LGREY, CYAN, WHITE, BLACK = (123, 125, 123), (214, 211, 214), (0, 252, 255), (255, 255, 255), (0, 0, 0)
_fonts = {}


def col(c):
    return {"g": GREEN, "a": ORANGE, "r": RED}.get((c or "x")[0], DGREY)


W, H = 240, 320          # portrait, as the sketch (2026-10-24)


class Screen:
    def __init__(self):
        self.im = Image.new("RGB", (W, H), BLACK)
        self.d = ImageDraw.Draw(self.im)

    def text(self, x, y, s, size, color):
        f = _fonts.setdefault(size, ImageFont.truetype(FONT, int(8 * size * 1.05)))
        for i, ch in enumerate(s):
            self.d.text((x + i * 6 * size + size // 2, y - size), ch, font=f, fill=color)

    def row(self, y, label, value, color):
        self.text(8, y, label, 1, LGREY)
        big = len(value) * 18 <= W - 12
        self.text(8, y + 11 if big else y + 14, value[:20], 3 if big else 2, color)


def paint(doc, stale_s=None, err=""):
    s = Screen()
    have = doc is not None
    stale = not have or (stale_s is not None and stale_s > 180)
    band = DGREY if stale else col(doc["level"])
    s.d.rectangle((0, 0, W - 1, 29), fill=band)
    s.text(8, 8, "IRIS", 2, BLACK)
    if have:
        s.text(W - 12 * 5 - 8, 8, doc["t"], 2, BLACK)
    if not have:
        s.row(60, "STATUS", "NO DATA", DGREY)
        s.text(8, 110, err, 2, LGREY)
        return s.im
    g = (lambda c: DGREY) if stale else (lambda c: c)
    s.text(8, 38, doc["night"]["v"], 2, g(col(doc["night"]["c"])))
    roof = doc["roof"]["v"]
    if doc["roof"]["age"] is not None:
        a = doc["roof"]["age"]
        roof += "  %dm" % a if a < 60 else "  %dh" % (a // 60)
    s.row(64, "ROOF", roof, g(col(doc["roof"]["c"])))
    s.row(108, "SCOPE", doc["scope"]["v"], g(col(doc["scope"]["c"])))
    if doc["now"]:
        n = doc["now"]
        v = "%s %s %d" % (n["dso"], n["filter"], n["n"]) + ("/%d" % n["of"] if n["of"] is not None else "")
        s.row(152, "NOW", v, g(CYAN))
    elif doc["next"]:
        s.row(152, "NEXT", "%s %s" % (doc["next"]["dso"], doc["next"]["at"]), g(WHITE))
    else:
        s.row(152, "NEXT", "-" if doc["tonight"]["image"] else "no imaging", g(WHITE))
    t = doc["tonight"]
    s.text(8, 202, "open  %s" % (t["opens"] or "--:--"), 2, g(LGREY))
    s.text(8, 224, "close %s" % (t["closes"] or "--:--"), 2, g(LGREY))
    if doc["disk"]:
        s.text(8, 246, "disk  %d GB" % doc["disk"]["gb"], 2, g(col(doc["disk"]["c"])))
    if stale:
        s.text(8, 282, "NO UPDATE %ds" % stale_s, 2, ORANGE)
        s.text(8, 304, err, 1, ORANGE)
    elif doc["alerts"]:
        s.text(8, 290, doc["alerts"][0][:19], 2, col(doc["level"]))
    return s.im


SUMMARY = "Tonight: m76\nRoof opens ~17:52, closes ~06:05"
SLOTS = [("ngc7790", "20:00"), ("hubblev1", "21:00"), ("m76", "22:00"), ("ngc2276", "04:00")]
PLAN = {"noon": {"plan": {"will_image": True, "slots": [
    {"dso": d, "start": ("2026-10-23T" if t < "12" else "2026-10-22T") + t + "-04:00"} for d, t in SLOTS]}}}
NOPLAN = {"noon": {"plan": {"will_image": False, "slots": []}}}


def mk(now, roof_v, roof_at, scope, state, live=None, disk=388.0, plan=PLAN, summary=SUMMARY):
    roof = {"v": roof_v, "at": datetime.fromisoformat(roof_at)} if roof_v else None
    return dash.summarize(datetime.fromisoformat(now), roof, scope,
                          {"state": state} if state else None, plan, summary, live, disk)


CASES = [
    ("1  Afternoon, all quiet", mk("2026-10-22 14:30", "SHUT", "2026-10-22 09:00", "?", "ARMED"), None, ""),
    ("2  Roof just opened", mk("2026-10-22 17:56", "OPEN", "2026-10-22 17:54", "STOPPED", "OPENING_ROOF"), None, ""),
    ("3  Imaging M76", mk("2026-10-22 23:40", "OPEN", "2026-10-22 17:54", "TRACKING", "SLOT_IMAGING",
                          {"dso": "m76", "filter": "Ha", "n": 14, "of": 25}), None, ""),
    ("4  Long line shrinks", mk("2026-10-22 21:20", "OPEN", "2026-10-22 17:54", "TRACKING", "SLOT_IMAGING",
                                {"dso": "hubblev1", "filter": "L", "n": 3, "of": 8}), None, ""),
    ("5  Cloudy night, no imaging", mk("2026-10-22 14:30", "SHUT", "2026-10-22 09:00", "?", "IDLE_DAY",
                                       plan=NOPLAN, summary=""), None, ""),
    ("6  Roof open after dawn", mk("2026-10-23 08:10", "OPEN", "2026-10-23 06:06", "?", "NIGHT_DONE"), None, ""),
    ("7  Emergency stop", mk("2026-10-23 00:35", "?", "2026-10-23 00:34", "STOPPED", "ESTOP"), None, ""),
    ("8  Roof fault", mk("2026-10-23 06:10", "?", "2026-10-23 06:08", "OFF", "FAULT_ROOF_UNKNOWN"), None, ""),
    ("9  Conductor down", mk("2026-10-22 15:00", "SHUT", "2026-10-22 09:00", "?", None), None, ""),
    ("10 Disk low", mk("2026-10-22 15:00", "SHUT", "2026-10-22 09:00", "?", "ARMED", disk=62.0), None, ""),
    ("11 iris-pc stopped answering", mk("2026-10-22 23:40", "OPEN", "2026-10-22 17:54", "TRACKING", "SLOT_IMAGING",
                                        {"dso": "m76", "filter": "Ha", "n": 14, "of": 25}), 420, "http -1"),
    ("12 Never connected", None, None, "http 403"),
    ("13 Roof opened by hand", mk("2026-10-22 15:00", "OPEN", "2026-10-22 14:58", "?", "MANUAL_OPEN"), None, ""),
    ("14 Roof slewing at dusk", mk("2026-10-22 19:58", "OPEN", "2026-10-22 17:54", "SLEWING", "SLOT_SETUP"), None, ""),
    ("15 PWI4 down at night", mk("2026-10-22 23:40", "OPEN", "2026-10-22 17:54", "?", "SLOT_IMAGING"), None, ""),
]

SCALE, PAD, COLS = 2, 24, 5
SW, SH = W * SCALE, H * SCALE
rows = (len(CASES) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * (SW + PAD) + PAD, rows * (SH + PAD + 40) + PAD), (32, 32, 36))
sd = ImageDraw.Draw(sheet)
cap = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 24)
for i, (title, doc, stale, err) in enumerate(CASES):
    x = PAD + (i % COLS) * (SW + PAD)
    y = PAD + (i // COLS) * (SH + PAD + 40)
    sd.text((x, y), title, font=cap, fill=(230, 230, 230))
    im = paint(doc, stale, err).resize((SW, SH), Image.NEAREST)
    sd.rectangle((x - 3, y + 37, x + SW + 2, y + 40 + SH + 2), fill=(70, 70, 75))
    sheet.paste(im, (x, y + 40))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "states.png")
sheet.save(out)
print(out, sheet.size)
