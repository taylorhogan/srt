"""How much did one night help or hurt each filter's stack?

    python fits_processing/night_contribution.py ngc7380 m33          # newest night of each
    python fits_processing/night_contribution.py ngc7380 2026-10-01   # a given night

The end-of-night `snr` says whether a stack has converged; it cannot say
whether TONIGHT's frames were worth having. Owner's question, 2026-10-02:
"how much last night helped or hurt each filter on the 2 objects".

METHOD. Per filter, every LIGHT frame goes through the stacker's own
convergence prep -- calibration, the FWHM/star quality gate, registration,
downscale to ~512 px -- so a frame the gate rejects contributes nothing here
exactly as it contributes nothing to a stack. Against the median R of all
accepted frames, each frame is fitted F = a*R + b over the pixels that carry
signal: a is its signal scale (transparency x exposure), sigma the robust
noise of its residual. Then, with and without the night's frames,

    optimal-weight SNR  ~ sqrt(sum (a/sigma)^2)        each frame weighted by its worth
    equal-weight SNR    ~ sum(a) / sqrt(sum sigma^2)   roughly what a plain mean does

The equal-weight figure is the one that can go NEGATIVE: a frame weaker than
the stack's own average lowers a plain mean, and that is what "hurt" means.
The optimal figure never goes down; it says what the frames were worth.

First run 2026-10-02 on the 10-01 night: NGC 7380 Ha +3%, O-III +9%, S-II +3%
(14 of 22 S-II frames failed the gate under cloud), M33 Ha +40% (a young
stack). Measured on the downscaled cube, so the percentages are relative
estimates, not absolute SNR.
"""
import json
import logging
import math
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_logger = logging.getLogger(__name__)


def night_of(path) -> str:
    """The session folder a frame was written to: .../<rig>/<date>/LIGHT/x.fits."""
    return Path(path).parent.parent.name


def latest_night(paths) -> str | None:
    nights = sorted({night_of(p) for p in paths})
    return nights[-1] if nights else None


# ------------------------------------------------------------------ pure

def _snr(rows):
    """(optimal, equal) SNR proxies for frame rows {"a", "sigma"}. Pure."""
    rows = [r for r in rows if r["sigma"] > 0]
    if not rows:
        return 0.0, 0.0
    opt = math.sqrt(sum((r["a"] / r["sigma"]) ** 2 for r in rows))
    eq = sum(r["a"] for r in rows) / math.sqrt(sum(r["sigma"] ** 2 for r in rows))
    return opt, eq


def _median(xs):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def summarize(dso, filt, night, rows, shot_by_night, fwhm_by_night):
    """One filter's result. Pure (no numpy), so CI tests it.

    rows          : accepted frames, {"night", "a", "sigma"}
    shot_by_night : {night: frames on disk}, before the quality gate
    fwhm_by_night : {night: [fwhm arcsec, ...]} from the frame cache
    """
    last = [r for r in rows if r["night"] == night]
    prior = [r for r in rows if r["night"] != night]
    out = {"dso": dso, "filter": filt, "night": night,
           "shot": shot_by_night.get(night, 0), "accepted": len(last),
           "prior_frames": len(prior),
           "prior_nights": len({r["night"] for r in prior})}
    o_all, e_all = _snr(rows)
    o_pr, e_pr = _snr(prior)
    out["gain_optimal_pct"] = round(100 * (o_all / o_pr - 1), 1) if o_pr else None
    out["gain_equal_pct"] = round(100 * (e_all / e_pr - 1), 1) if e_pr else None
    if last and prior:
        w = [(r["a"] / r["sigma"]) ** 2 for r in prior if r["sigma"] > 0]
        med_w = _median(w)
        out["worth_prior_frames"] = (round(sum((r["a"] / r["sigma"]) ** 2 for r in last
                                               if r["sigma"] > 0) / med_w, 1)
                                     if med_w else None)
        out["signal_vs_prior"] = round(_median([r["a"] for r in last]) /
                                       _median([r["a"] for r in prior]), 2)
        out["noise_vs_prior"] = round(_median([r["sigma"] for r in last]) /
                                      _median([r["sigma"] for r in prior]), 2)
    fw_last = _median(fwhm_by_night.get(night, []))
    fw_prior = _median([f for n, fs in fwhm_by_night.items() if n != night for f in fs])
    out["fwhm_last"] = round(fw_last, 2) if fw_last else None
    out["fwhm_prior"] = round(fw_prior, 2) if fw_prior else None
    return out


def verdict(r):
    """'helped' / 'hurt' / 'no change' / 'nothing usable' / 'first night'. Pure."""
    if r["accepted"] == 0:
        return "nothing usable"
    if not r["prior_frames"]:
        return "first night"
    g = r.get("gain_equal_pct")
    if g is None:
        return "no change"
    if g < -0.5:
        return "hurt"
    return "helped" if g > 0.5 else "no change"


def sky_words(signal, noise):
    """Why the night's frames weigh what they do, from the per-frame ratios. Pure.

    signal is the median frame's nebula scale against the earlier nights'
    (transparency); noise its residual sigma against theirs, which in a 300 s
    narrowband frame is the sky. IC 410 S-II, 2026-11-01: noise 0.82x with the
    raw sky at the bias level against moonlit earlier nights -> "a darker sky".
    """
    if signal is None or noise is None:
        return ""
    words = []
    if noise <= 0.9:
        words.append("a darker sky")
    elif noise >= 1.1:
        words.append("a brighter sky")
    if signal <= 0.9:
        words.append("thinner air")
    elif signal >= 1.1:
        words.append("clearer air")
    return " and ".join(words) if words else "a sky like the earlier nights'"


def describe(r):
    """One chat line per filter, in sentences. Pure.

    The owner read "stack SNR +32% (plain mean +38%); worth 33.5 earlier
    frames; signal 0.93x, noise 0.82x earlier nights" as a puzzle
    (2026-11-02): two SNR figures with no word on why they differ, "earlier
    frames" that reads as a count, and HELPED beside a worse FWHM. So: the
    gate and the seeing first, then how much deeper the stack is (the plain
    mean only where it carries the verdict, HURT or NO CHANGE), then what
    the new frames were worth and why.
    """
    v = verdict(r)
    head = "%s %s: %s." % (r["dso"], r["filter"], v.upper())
    gate = "%d of %d frames passed the gate" % (r["accepted"], r["shot"])
    if r.get("fwhm_last") and r.get("fwhm_prior"):
        # "now ... earlier ...", never "x vs y": the bare pair did not say which
        # number was the night being judged (owner, 2026-10-08).
        gate += ' (seeing %.2f", earlier nights %.2f")' % (r["fwhm_last"], r["fwhm_prior"])
    if v in ("nothing usable", "first night"):
        return "%s %s." % (head, gate)
    opt, eq = r["gain_optimal_pct"], r["gain_equal_pct"]
    if v == "hurt":
        depth = ("The new frames are weaker than the stack's average: a plain mean of every "
                 "frame comes out %.0f%% shallower, a weighted stack %+.0f%%." % (-eq, opt))
    elif v == "no change":
        depth = "The stack is about as deep as before (weighted %+.0f%%, plain mean %+.0f%%)." % (opt, eq)
    else:
        depth = "The stack is %.0f%% deeper." % opt
    parts = [gate + ".", depth]
    if r.get("worth_prior_frames") is not None:
        why = sky_words(r.get("signal_vs_prior"), r.get("noise_vs_prior"))
        parts.append("The %d new frames are worth %.1f typical earlier ones: %s (per frame: "
                     "noise %.2fx, signal %.2fx the earlier nights)."
                     % (r["accepted"], r["worth_prior_frames"], why,
                        r["noise_vs_prior"], r["signal_vs_prior"]))
    return "%s %s" % (head, " ".join(parts))


# ------------------------------------------------------------------ measuring

def _mad_std(x):
    import numpy as np
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def measure_filter(dso_dir, filt, paths, night, arcsec, cancel_cb=None):
    """Calibrate, gate, register and fit one filter's frames -> summarize() dict."""
    import numpy as np
    from fits_processing.frame_cache import load_precomputed_fwhm_stars
    from stacking import stacker

    paths = sorted(paths)
    pre = load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
    shot = {}
    fwhm = {}
    for p in paths:
        n = night_of(p)
        shot[n] = shot.get(n, 0) + 1
        q = pre.get(p)
        if q and q[0]:
            fwhm.setdefault(n, []).append(q[0] * arcsec)
    frames, accepted, _fw = stacker._prepare_for_convergence(
        paths, calibration=stacker.calibration_from_config(filt, like=paths[0]),
        precomputed_fwhm_stars=pre, downscale_to=512, cancel_cb=cancel_cb)
    rows = []
    if frames:
        cube = np.stack([np.asarray(f, float) for f in frames])
        ref = np.nanmedian(cube, axis=0)
        ok = np.isfinite(ref) & np.all(np.isfinite(cube), axis=0)
        r = ref[ok]
        sig = r > np.median(r) + 3 * _mad_std(r)
        use = sig if sig.sum() > 200 else np.ones_like(r, bool)
        A = np.vstack([r[use], np.ones(int(use.sum()))]).T
        for f, p in zip(cube, accepted):
            v = f[ok]
            a, b = np.linalg.lstsq(A, v[use], rcond=None)[0]
            rows.append({"night": night_of(p), "a": float(a),
                         "sigma": _mad_std(v - (a * r + b))})
    return summarize(Path(dso_dir).name, filt, night, rows, shot, fwhm)


def analyse(dso, night=None, progress_cb=None, cancel_cb=None, workers=3):
    """Every filter of *dso* that has frames from *night* (default: its newest night)."""
    from concurrent.futures import ThreadPoolExecutor
    from configs import config
    from stacking import stacker
    cfg = config.data()
    dso_dir = Path(cfg["nina"]["image_dir"]) / dso
    lights = [p for p in dso_dir.rglob("*.fits") if p.parent.name.upper() == "LIGHT"]
    night = night or latest_night(lights)
    if not night:
        return night, []
    arcsec = float(cfg["nina"]["arc_sec_per_pixel"])
    groups = {f: ps for f, ps in stacker.group_by_filter(lights).items()
              if any(night_of(p) == night for p in ps)}
    if progress_cb:
        progress_cb("%s, night of %s: measuring %s"
                    % (dso, night, ", ".join("%s (%d frames)" % (f, len(ps))
                                             for f, ps in sorted(groups.items()))))

    def one(item):
        f, ps = item
        try:
            return measure_filter(dso_dir, f, ps, night, arcsec, cancel_cb)
        except Exception:  # noqa: BLE001 -- one filter must not sink the others
            _logger.exception("night contribution failed for %s %s", dso, f)
            return None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = [r for r in ex.map(one, sorted(groups.items())) if r]
    return night, results


def report(dso, night, results):
    """The chat message for one target. Pure."""
    if not results:
        return "Night contribution, %s: no frames from %s." % (dso, night or "any night")
    lines = ["Night contribution, %s, night of %s:" % (dso, night)]
    lines += ["- " + describe(r) for r in sorted(results, key=lambda r: r["filter"])]
    return "\n".join(lines)


def main(argv):
    from utils import utils
    utils.set_logger()
    night = next((a for a in argv if len(a) == 10 and a[4] == "-" and a[7] == "-"), None)
    dsos = [a for a in argv if a != night]
    if not dsos:
        print(__doc__.strip().splitlines()[2])
        return 2
    for dso in dsos:
        n, results = analyse(dso, night, progress_cb=print)
        print(report(dso, n, results))
        print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
