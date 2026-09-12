"""Per-filter focuser position as a function of temperature, fitted from
N.I.N.A's own autofocus reports, to SEED each filter block's autofocus.

Why a seed and not a replacement: on this scope the HFR barely moves across
2000 focuser steps, so an autofocus run that starts far from focus has a flat
curve to fit and fails (four of six runs on 2026-09-11; the first O-III
light of that night was taken at the Ha focus, 1000 steps off, because the
run failed and N.I.N.A restored the position it started from). Centring the
sweep on a predicted position fixes both halves of that: the curve is a V
on both sides, and if the fit still fails the position N.I.N.A restores IS
the prediction, not the previous filter's focus.

What the reports say (571 runs, 2026-03..09): every filter's accepted
position tracks the focuser temperature at roughly -85 steps/degC
(narrowband shallower, O-III steeper), with 200-500 steps of scatter around
the fit, and filters sit at fixed offsets from each other (Ha ~ -1370 from
L, O-III ~ -675, S-II ~ -1300). That scatter is inside the sweep, which is
the whole point: the seed only needs to land inside the V.

The consumer is N.I.N.A's MoveFocuserByTemperature instruction, written by
nina_gen before every SmartExposure block with this filter's slope and
intercept (Absolute: position = slope * T + intercept, T read live from the
focuser at run time). No script, no PWI4: the focuser (an Optec Gemini) is
N.I.N.A's device.

Pure stdlib on purpose: this runs in CI's pytest-only environment and at
noon inside the scheduler.

    python fits_processing/focus_model.py --fit      # refit from the reports, write the model
    python fits_processing/focus_model.py --show     # print the model
    python fits_processing/focus_model.py --predict Ha 15.0
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

DEFAULT_REPORTS_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "NINA", "AutoFocus")
DEFAULT_MODEL_PATH = "local/focus_model.json"

# A run N.I.N.A itself accepted: its hyperbolic R^2 clears the profile's
# threshold (0.8). Anything below is a failed fit and its "calculated" focus
# point is noise.
MIN_R2 = 0.8
MIN_POINTS = 4         # fewer than this and a slope is a coin toss
MIN_TEMP_SPAN_C = 3.0  # a slope over a narrower span is dominated by scatter
CLIP_SIGMA = 3.0       # one pass of outlier rejection around the first fit
# Used when no filter has enough points to fit its own slope. The season's
# broadband fits (L/R/G/B, n=47..101) all land at -83..-87 steps/degC.
DEFAULT_SLOPE = -85.0


def _norm(name: Optional[str]) -> str:
    return (name or "").strip().upper()


def read_reports(reports_dir: str = DEFAULT_REPORTS_DIR) -> list[dict]:
    """Every autofocus report as {filter, position, temp, r2, when}.

    Reports that lack a temperature or a calculated position are skipped;
    they cannot inform a temperature model.
    """
    out = []
    d = Path(reports_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pos = (r.get("CalculatedFocusPoint") or {}).get("Position")
        temp = r.get("Temperature")
        r2 = (r.get("RSquares") or {}).get("Hyperbolic")
        if pos is None or temp is None:
            continue
        try:
            out.append({"filter": _norm(r.get("Filter")), "position": float(pos),
                        "temp": float(temp), "r2": float(r2) if r2 is not None else None,
                        "when": r.get("Timestamp") or p.name[:19]})
        except (TypeError, ValueError):
            continue
    return out


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def _rms(xs, ys, slope, intercept) -> float:
    if not xs:
        return 0.0
    return (sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fit_models(reports: list[dict], min_r2: float = MIN_R2, min_points: int = MIN_POINTS,
               min_temp_span: float = MIN_TEMP_SPAN_C, clip_sigma: float = CLIP_SIGMA,
               default_slope: float = DEFAULT_SLOPE) -> dict:
    """Per-filter {slope, intercept, n, rms, temp_min, temp_max, fitted, ...}.

    A filter with enough accepted runs over a wide enough temperature range
    gets its own slope. One with too few keeps the season's pooled slope
    (median of the fitted ones, else *default_slope*) through its own median
    position at its own median temperature -- a constant-plus-trend seed is
    still far better than the previous filter's focus.
    """
    by: dict[str, list[dict]] = {}
    for r in reports:
        if r["r2"] is None or r["r2"] < min_r2 or not r["filter"]:
            continue
        by.setdefault(r["filter"], []).append(r)

    model: dict[str, dict] = {}
    for f, rs in by.items():
        xs = [r["temp"] for r in rs]
        ys = [r["position"] for r in rs]
        span = max(xs) - min(xs) if xs else 0.0
        entry = {"n": len(rs), "temp_min": min(xs), "temp_max": max(xs),
                 "last_position": rs[-1]["position"], "last_temp": rs[-1]["temp"],
                 "last_when": rs[-1]["when"]}
        if len(rs) >= min_points and span >= min_temp_span:
            slope, intercept = _linfit(xs, ys)
            rms = _rms(xs, ys, slope, intercept)
            # Clip against a ROBUST scale (1.4826 * median |residual|): with a
            # handful of points one wild run inflates the plain rms enough to
            # hide inside its own 3-sigma band.
            res = [abs(y - (slope * x + intercept)) for x, y in zip(xs, ys)]
            scale = 1.4826 * _median(res)
            if clip_sigma and scale > 0:
                keep = [(x, y) for x, y, r in zip(xs, ys, res) if r <= clip_sigma * scale]
                if len(keep) >= min_points and len(keep) < len(xs):
                    xs2, ys2 = [k[0] for k in keep], [k[1] for k in keep]
                    slope, intercept = _linfit(xs2, ys2)
                    rms = _rms(xs2, ys2, slope, intercept)
                    entry["n_clipped"] = len(xs) - len(keep)
            entry.update(slope=slope, intercept=intercept, rms=rms, fitted=True)
        else:
            entry.update(fitted=False, median_position=_median(ys), median_temp=_median(xs))
        model[f] = entry

    fitted_slopes = [e["slope"] for e in model.values() if e.get("fitted")]
    pooled = _median(fitted_slopes) if fitted_slopes else default_slope
    for f, e in model.items():
        if not e.get("fitted"):
            e["slope"] = pooled
            e["intercept"] = e["median_position"] - pooled * e["median_temp"]
            e["rms"] = _rms([r["temp"] for r in by[f]], [r["position"] for r in by[f]],
                            e["slope"], e["intercept"])
    return {"generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "min_r2": min_r2, "pooled_slope": pooled, "filters": model}


def predict(model: dict, filter_name: str, temp_c: float) -> Optional[float]:
    e = (model or {}).get("filters", {}).get(_norm(filter_name))
    if not e:
        return None
    return e["slope"] * float(temp_c) + e["intercept"]


def seed_for(model: dict, filter_name: str) -> Optional[tuple[float, float]]:
    """(slope, intercept) for N.I.N.A's MoveFocuserByTemperature, or None."""
    e = (model or {}).get("filters", {}).get(_norm(filter_name))
    if not e:
        return None
    return float(e["slope"]), float(e["intercept"])


def save_model(model: dict, path: str = DEFAULT_MODEL_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def load_model(path: str = DEFAULT_MODEL_PATH) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh(reports_dir: str = DEFAULT_REPORTS_DIR,
            model_path: str = DEFAULT_MODEL_PATH) -> Optional[dict]:
    """Refit from the reports when they are reachable, else the cached model.

    Called at sequence-generation time so every night's accepted autofocus
    runs feed the next night's seeds. On a box without the reports (CI, a
    dev checkout) the cached file stands in; with neither there are no seeds
    and the sequence is generated exactly as before.
    """
    reports = read_reports(reports_dir)
    if reports:
        model = fit_models(reports)
        if model["filters"]:
            save_model(model, model_path)
            return model
    return load_model(model_path)


def describe(model: Optional[dict]) -> str:
    if not model or not model.get("filters"):
        return "no focus model"
    lines = ["focus model %s  pooled slope %+.0f steps/degC"
             % (model.get("generated", ""), model.get("pooled_slope", 0))]
    for f, e in sorted(model["filters"].items()):
        tag = "fit" if e.get("fitted") else "pooled"
        lines.append("  %-6s %-6s n=%3d  slope %+6.0f  intercept %8.0f  rms %4.0f  "
                     "T %5.1f..%5.1f  last %6.0f @ %.1fC"
                     % (f, tag, e["n"], e["slope"], e["intercept"], e.get("rms", 0),
                        e["temp_min"], e["temp_max"], e["last_position"], e["last_temp"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Focus seed model from N.I.N.A autofocus reports")
    ap.add_argument("--fit", action="store_true", help="refit from the reports and write the model")
    ap.add_argument("--show", action="store_true", help="print the model")
    ap.add_argument("--predict", nargs=2, metavar=("FILTER", "TEMP_C"))
    ap.add_argument("--reports", default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    a = ap.parse_args(argv)
    model = None
    if a.fit:
        reports = read_reports(a.reports)
        model = fit_models(reports)
        save_model(model, a.model)
        print("%d reports read; model written to %s" % (len(reports), a.model))
    if a.show or a.fit:
        print(describe(model or load_model(a.model)))
    if a.predict:
        m = model or load_model(a.model)
        v = predict(m, a.predict[0], float(a.predict[1]))
        print("no model for that filter" if v is None
              else "%s at %s C -> %.0f" % (a.predict[0], a.predict[1], v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
