import json, sys, numpy as np, datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
S = sys.argv[1]
C = json.load(open(f"{S}/comps.json"))
names = [t["name"] for t in C["targets"]]
nights = [n for n in C["nights"] if n["night"] != "2026-10-14"]          # the failed night (clouds) is left out as in the main curve
dates = [datetime.date.fromisoformat(n["night"]) for n in nights]
M = np.array([n["mag_loo"] for n in nights])                            # nights x targets
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 4.4), dpi=110)
cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(names) - 1))
stats = []
for k in range(1, len(names)):
    y = M[:, k] - np.median(M[:, k])
    sd = float(np.std(y)); stats.append((names[k], C["targets"][k]["g"], sd))
    ax.plot(dates, y + 0.0, "o-", ms=4, lw=1, color=cols[k - 1], alpha=0.9, label="%s (G %.1f): scatter %.3f mag" % (names[k], C["targets"][k]["g"], sd))
v = M[:, 0] - np.median(M[:, 0])
ax.plot(dates, v, "o-", ms=6, lw=2, color="#1f4e79", label="V1 (relative to its own median)", zorder=5)
ax.set_ylim(1.0, -0.7); ax.set_ylabel("magnitude relative to the star's own median")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
ax.set_title("Eight comparison stars within 5 arcminutes of V1, measured the same way on the same frames", fontsize=12)
fig.tight_layout(); fig.savefig(f"{S}/comps.svg"); fig.savefig(f"{S}/comps.png"); plt.close(fig)
json.dump({"stats": stats, "v1_range": [float(M[:, 0].min()), float(M[:, 0].max())], "nights": len(nights),
           "median_comp_scatter": float(np.median([s[2] for s in stats]))}, open(f"{S}/comps_stats.json", "w"), indent=1)
print("comps chart:", stats, "V1 range", M[:, 0].min(), M[:, 0].max())
