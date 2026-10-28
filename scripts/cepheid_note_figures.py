"""Figures for the site's Cepheid note (2026-10-28).

    python scripts/cepheid_note_figures.py <site images dir>

Reads the nightly points the end-of-night `lightcurve` step writes to
local/lightcurves/<dso>.json and the ephemerides stored on the queue entries.

  cepheid_v1.png        Hubble's V1 in M31 against date, with the night the
                        ephemeris on its queue entry puts the maximum.
  cepheid_ngc7790.png   CF Cas and CG Cas folded on their catalog periods and
                        epochs (phase 0 = catalog maximum), and the blended
                        CE Cas A+B against date.

Luminance (L) points only: they are the nightly series; the few G frames per
night are a check, not a curve. Prints the numbers the note quotes.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BG, FG, DIM, GRID, BLUE, ORANGE = "#15192b", "#e6e9f2", "#7a88a1", "#262c40", "#3a8ee6", "#f0a050"


def _jd_to_dt(jd):
    """Local (EDT) time less 18 h: a point sits on the date its night began, so an
    axis labelled 'night of' reads right (23:23 on Oct 17 plots on Oct 17)."""
    return (datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545.0)
            - timedelta(hours=4 + 18))


def _points(name, filt="L"):
    r = json.load(open(os.path.join(ROOT, "local", "lightcurves", name + ".json")))["records"]
    r = sorted((x for x in r if x["filter"] == filt), key=lambda x: x["jd"])
    return (np.array([x["jd"] for x in r]), np.array([x["mag"] for x in r]),
            np.array([x["err"] for x in r]), [x["night"] for x in r])


def _style(ax, title):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=DIM)
    ax.grid(color=GRID, lw=0.8)
    ax.set_title(title, color=FG, loc="left", fontweight="bold")
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)


def sine_fit(t, m, period, epoch):
    """Mean, full amplitude and the phase of maximum light of a sine at *period*. Pure."""
    ph = 2 * np.pi * (t - epoch) / period
    A = np.stack([np.ones_like(t), np.sin(ph), np.cos(ph)], 1)
    c, *_ = np.linalg.lstsq(A, m, rcond=None)
    rms = float(np.sqrt(np.mean((m - A @ c) ** 2)))
    # Brightest = smallest magnitude: where c1 sin + c2 cos is most negative.
    phi_max = (np.arctan2(-c[1], -c[2]) / (2 * np.pi)) % 1.0
    return float(c[0]), float(2 * np.hypot(c[1], c[2])), float(phi_max), rms, c


def v1_figure(out):
    q = {x["dso"]: x for x in json.load(open(os.path.join(ROOT, "local", "my_instructions.json")))}
    e = q["hubblev1"]
    P, E0 = float(e["period_days"]), float(e["epoch_jd"])
    t, m, err, nights = _points("hubblev1")
    n_cyc = np.round((t.mean() - E0) / P)
    jd_pred = E0 + n_cyc * P
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=BG)
    _style(ax, "Hubble's V1 in M31, luminance, one point per night")
    x = [_jd_to_dt(v) for v in t]
    poor = err > 0.2
    ax.errorbar([x[i] for i in np.where(~poor)[0]], m[~poor], yerr=err[~poor], fmt="o",
                color=BLUE, ecolor=BLUE, ms=6, capsize=0, label="nightly mean")
    ax.errorbar([x[i] for i in np.where(poor)[0]], m[poor], yerr=err[poor], fmt="o",
                mfc="none", color=DIM, ecolor=DIM, ms=6, capsize=0, label="poor night (frames disagree)")
    ax.axvline(_jd_to_dt(jd_pred), color=ORANGE, ls="--", lw=1.5,
               label="maximum predicted by the ephemeris")
    ax.invert_yaxis()
    ax.set_ylabel("magnitude (Gaia G scale)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("night of (local evening date)")
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower right")
    leg.get_frame().set_alpha(1)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    i_max = int(np.argmin(np.where(poor, 99, m)))
    print("V1: %d nights %s..%s; range %.2f-%.2f (%.2f mag); brightest %s; "
          "ephemeris max JD %.2f = night of %s; cycle %d"
          % (len(t), nights[0], nights[-1], m[~poor].min(), m[~poor].max(),
             m[~poor].max() - m[~poor].min(), nights[i_max], jd_pred,
             _jd_to_dt(jd_pred).strftime("%Y-%m-%d"), n_cyc))
    decl = np.polyfit(t[i_max + 1:] - t[i_max], m[i_max + 1:], 1)[0]
    print("V1: decline after maximum %.3f mag/day" % decl)


def ngc7790_figure(out):
    q = {x["dso"]: x for x in json.load(open(os.path.join(ROOT, "local", "my_instructions.json")))}
    stars = {s["name"]: s for s in q["ngc7790"]["stars"]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=BG)
    for ax, (name, key) in zip(axes[:2], (("CF Cas", "ngc7790_cf_cas"), ("CG Cas", "ngc7790_cg_cas"))):
        s = stars[name]
        P, E0 = float(s["period_days"]), float(s["epoch_jd"])
        t, m, err, nights = _points(key)
        mean, amp, phi_max, rms, c = sine_fit(t, m, P, E0)
        ph = ((t - E0) / P) % 1.0
        _style(ax, "%s, folded on P = %.3f d" % (name, P))
        for k in (0, 1):
            ax.errorbar(ph + k, m, yerr=err, fmt="o", color=BLUE, ms=6, capsize=0)
        g = np.linspace(0, 2, 400)
        gg = 2 * np.pi * g
        ax.plot(g, c[0] + c[1] * np.sin(gg) + c[2] * np.cos(gg), color=ORANGE, lw=1.2)
        span = m.max() - m.min()
        ax.set_ylim(m.max() + 0.12 * span + 0.03, m.min() - 0.45 * span)
        ax.set_xlim(0, 2)
        ax.set_xlabel("phase (0 = catalog maximum)")
        ax.text(0.03, 0.95, "%d nights, amplitude %.2f mag\nscatter about the sine %.3f mag"
                % (len(t), amp, rms), transform=ax.transAxes, color=DIM, va="top")
        print("%s: %d nights; L %.2f-%.2f; sine amp %.2f, rms %.3f; fitted max at catalog phase %.2f "
              "(%.2f d)" % (name, len(t), m.min(), m.max(), amp, rms, phi_max,
                            ((phi_max + 0.5) % 1 - 0.5) * P))
    axes[0].set_ylabel("magnitude (Gaia G scale)")
    ax = axes[2]
    t, m, err, nights = _points("ngc7790_ce_cas_a_b")
    _style(ax, "CE Cas A+B, blended, against date")
    import matplotlib.dates as mdates
    ax.errorbar([_jd_to_dt(v) for v in t], m, yerr=err, fmt="o", color=BLUE, ms=6, capsize=0)
    ax.set_ylim(m.max() + 0.05, m.min() - 0.15)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlabel("night of (local evening date)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.text(0.03, 0.95, "two Cepheids 2.5\" apart,\nP = 5.141 d and 4.479 d", transform=ax.transAxes,
            color=DIM, va="top")
    print("CE Cas A+B: %d nights; L %.2f-%.2f" % (len(t), m.min(), m.max()))
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "iris_astronomy", "scratch")
    v1_figure(os.path.join(d, "cepheid_v1.png"))
    ngc7790_figure(os.path.join(d, "cepheid_ngc7790.png"))
