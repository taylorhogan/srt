import json, sys, datetime, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
S = sys.argv[1]
M = json.load(open(f"{S}/lc_model.json"))
P, E0 = M["P"], M["E0"]
coef = np.array(M["coef"]); K = M["K"]
def model(p):
    p = np.atleast_1d(p); cols = [np.ones_like(p)]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * p), np.sin(2 * np.pi * k * p)]
    return np.vstack(cols).T @ coef
def jd2dt(j): return datetime.datetime(2000, 1, 1, 12) + datetime.timedelta(days=j - 2451545.0)
C_MEAS, C_LINE, C_UP, C_G, C_BAD = "#1f4e79", "#c0392b", "#e67e22", "#2e8b57", "#999999"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---- 1. the light curve in time
good, bad, gp, up = M["good"], M["bad"], M["gpts"], M["upcoming"]
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=110)
j0, j1 = M["jd_first"] - 3, up[-1]["jd"] + 1
jj = np.linspace(j0, j1, 1200); pp = ((jj - E0) / P) % 1.0
past = jj <= M["jd_last"] + 0.5
ax.plot([jd2dt(j) for j in jj[past]], model(pp[past]), color=C_LINE, lw=2.2, label="fitted curve (3-term Fourier series, P = 31.4 d)")
ax.plot([jd2dt(j) for j in jj[~past]], model(pp[~past]), color=C_LINE, lw=1.6, ls=(0, (4, 3)), alpha=0.8, label="the same curve, carried forward")
ax.errorbar([jd2dt(r["jd"]) for r in good], [r["mag"] for r in good], yerr=[r["err"] for r in good], fmt="o", ms=6,
            color=C_MEAS, ecolor=C_MEAS, elinewidth=1.4, capsize=3, zorder=5, label="measured, L filter (18 nights)")
ax.errorbar([jd2dt(r["jd"]) for r in gp], [r["mag"] for r in gp], yerr=[min(r["err"], 0.3) for r in gp], fmt="D", ms=5,
            color=C_G, ecolor=C_G, elinewidth=1, capsize=2, alpha=0.85, zorder=4, label="measured, G filter (3 nights)")
for r in bad:
    ax.plot(jd2dt(r["jd"]), r["mag"], "x", ms=9, mew=2, color=C_BAD, zorder=4)
ax.plot([], [], "x", ms=9, mew=2, color=C_BAD, label="failed night (error > 0.5 mag, left out)")
ax.plot([jd2dt(u["jd"]) for u in up], [u["mag"] for u in up], "o", ms=5.5, mfc="white", mec=C_UP, mew=1.8, zorder=6,
        label="what each coming clear night should measure")
ax.axvline(jd2dt(M["next_max_jd"]), color=C_UP, lw=1, ls=":", alpha=0.9)
ax.annotate("next maximum\n%s" % M["next_max_date"][:10], (jd2dt(M["next_max_jd"]), M["m_max"] - 0.02), color=C_UP, fontsize=9,
            ha="center", va="bottom")
ax.axvline(jd2dt(M["jd_max_obs"]), color=C_MEAS, lw=1, ls=":", alpha=0.6)
ax.annotate("maximum\n%s" % M["date_max_obs"][:10], (jd2dt(M["jd_max_obs"]), M["m_max"] - 0.02), color=C_MEAS, fontsize=9,
            ha="center", va="bottom")
ax.set_ylim(19.75, 17.95); ax.set_ylabel("magnitude (Gaia G scale)")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
ax.set_title("M31 V1, Hubble's Cepheid: 18 nights from Iris, 2026-10-08 to 11-10, and the nights ahead", fontsize=12)
fig.tight_layout(); fig.savefig(f"{S}/lc_time.svg"); fig.savefig(f"{S}/lc_time.png"); plt.close(fig)

# ---- 2. folded, two cycles
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=110)
bins = M["coverage"]["bins"]
for b in M["coverage"]["missing"]:
    for c in (0, 1):
        ax.axvspan(c + b / bins, c + (b + 1) / bins, color=C_UP, alpha=0.10, lw=0)
ax.plot([], [], color=C_UP, alpha=0.3, lw=10, label="phase bins with no measurement yet")
pp = np.linspace(0, 2, 1441)
ax.plot(pp, model(pp % 1.0), color=C_LINE, lw=2.2, label="fitted curve")
for c in (0, 1):
    ph = [((r["jd"] - E0) / P) % 1.0 + c for r in good]
    ax.errorbar(ph, [r["mag"] for r in good], yerr=[r["err"] for r in good], fmt="o", ms=6, color=C_MEAS, ecolor=C_MEAS,
                elinewidth=1.4, capsize=3, zorder=5, label="measured, L" if c == 0 else None)
    phg = [((r["jd"] - E0) / P) % 1.0 + c for r in gp]
    ax.plot(phg, [r["mag"] for r in gp], "D", ms=5, color=C_G, alpha=0.85, zorder=4, label="measured, G" if c == 0 else None)
    soon = [u for u in up if u["date"] <= "2026-11-30"]
    phu = [u["phase"] + c for u in soon]
    ax.plot(phu, [u["mag"] for u in soon], "o", ms=5.5, mfc="white", mec=C_UP, mew=1.8, zorder=6,
            label="the nights to Nov 30 (predicted)" if c == 0 else None)
    phb = [((r["jd"] - E0) / P) % 1.0 + c for r in bad]
    ax.plot(phb, [r["mag"] for r in bad], "x", ms=9, mew=2, color=C_BAD, zorder=4, label="failed night" if c == 0 else None)
ax.set_xlim(0, 2); ax.set_ylim(19.75, 17.95)
ax.set_xlabel("phase (cycles of 31.4 days from the 2010 epoch of maximum, JD 2455430.5; shown twice)")
ax.set_ylabel("magnitude (Gaia G scale)")
ax2 = ax.secondary_xaxis("top", functions=(lambda x: x * P, lambda d: d / P)); ax2.set_xlabel("days", fontsize=9)
ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=3)
ax.set_title("The same nights folded on the period: one cycle of %.2f mag, a %.0f-day rise and a %.0f-day decline" % (
    M["amplitude"], M["rise_fraction"] * P, (1 - M["rise_fraction"]) * P), fontsize=12)
fig.tight_layout(); fig.savefig(f"{S}/lc_fold.svg"); fig.savefig(f"{S}/lc_fold.png"); plt.close(fig)
print("charts written")
