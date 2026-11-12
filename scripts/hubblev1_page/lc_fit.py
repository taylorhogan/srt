"""Fit V1's folded light curve from the PC's nightly records; write lc_model.json. Pure numpy."""
import json, math, sys, datetime
import numpy as np
sys.path.insert(0, "/home/taylor/Documents/srt")
from photometry import lightcurve as lc
S = sys.argv[1]
doc = json.load(open(f"{S}/pc/hubblev1.json"))
recs = doc["records"]
P, E0 = 31.4, 2455430.5
good = [r for r in recs if r["filter"] == "L" and r["err"] <= lc.MAX_POINT_ERR]
bad = [r for r in recs if r["filter"] == "L" and r["err"] > lc.MAX_POINT_ERR]
gpts = [r for r in recs if r["filter"] == "G"]
jd = np.array([r["jd"] for r in good]); mag = np.array([r["mag"] for r in good]); err = np.array([r["err"] for r in good])
ph = ((jd - E0) / P) % 1.0
# Fourier series of order K in phase, weighted least squares
K = 3
def design(p):
    cols = [np.ones_like(p)]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * p), np.sin(2 * np.pi * k * p)]
    return np.vstack(cols).T
w = 1.0 / np.maximum(err, 0.02)
A = design(ph) * w[:, None]; b = mag * w
coef, *_ = np.linalg.lstsq(A, b, rcond=None)
model = lambda p: design(np.atleast_1d(p)) @ coef
resid = mag - model(ph)
chi2 = float(np.sum((resid / np.maximum(err, 0.02)) ** 2)); dof = len(mag) - (2 * K + 1)
pp = np.linspace(0, 1, 721)
mm = model(pp)
i_max = int(np.argmin(mm)); i_min = int(np.argmax(mm))
ph_max = float(pp[i_max]); m_max = float(mm[i_max]); m_min = float(mm[i_min])
# intensity mean over the whole cycle from the model, and the module's mean over the nights
flux_mean = float(np.mean(10 ** (-0.4 * mm[:-1])))
mean_cycle = -2.5 * math.log10(flux_mean)
mean_nights, n_nights = lc.mean_magnitude(recs, "L")
dist_cycle = lc.leavitt_distance(P, mean_cycle, a_v=0.3, g_to_v=0.3)
dist_nights = lc.leavitt_distance(P, mean_nights, a_v=0.3, g_to_v=0.3)
cov = lc.coverage(recs, P, E0)
# rise time fraction: from min (faintest) to max (brightest) going forward in phase
rise = (ph_max - float(pp[i_min])) % 1.0
# the epoch of maximum nearest our data
E = round((jd.mean() - E0) / P)
jd_max_pred = E0 + P * E
jd_max_obs = E0 + P * (E + ph_max) if ph_max < 0.5 else E0 + P * (E - 1 + ph_max)
def jd2date(j):
    return (datetime.datetime(2000, 1, 1, 12) + datetime.timedelta(days=j - 2451545.0)).strftime("%Y-%m-%d %H:%M")
# upcoming nights: every night from tomorrow to 2026-12-15, predicted at 20:00 local = 01:00 UT next day
up = []
d = datetime.date(2026, 11, 12)
while d <= datetime.date(2026, 12, 20):
    j = 2451545.0 + (datetime.datetime(d.year, d.month, d.day, 1, 0) + datetime.timedelta(days=1) - datetime.datetime(2000, 1, 1, 12)).total_seconds() / 86400
    p = ((j - E0) / P) % 1.0
    up.append({"date": d.isoformat(), "jd": j, "phase": p, "mag": float(model(p)[0]), "bin": min(int(p * cov["bins"]), cov["bins"] - 1)})
    d += datetime.timedelta(days=1)
out = {"P": P, "E0": E0, "K": K, "coef": coef.tolist(), "n_good": len(good), "chi2": chi2, "dof": dof,
       "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
       "ph_max": ph_max, "m_max": m_max, "m_min": m_min, "amplitude": m_min - m_max, "rise_fraction": rise,
       "mean_cycle": mean_cycle, "mean_nights": mean_nights, "n_nights": n_nights,
       "dist_cycle": dist_cycle, "dist_nights": dist_nights, "coverage": cov,
       "jd_max_pred_ephemeris": jd_max_pred, "date_max_ephemeris": jd2date(jd_max_pred),
       "jd_max_obs": jd_max_obs, "date_max_obs": jd2date(jd_max_obs),
       "next_max_jd": jd_max_obs + P, "next_max_date": jd2date(jd_max_obs + P),
       "good": good, "bad": bad, "gpts": gpts, "upcoming": up,
       "model_curve": [[float(p), float(m)] for p, m in zip(pp, mm)],
       "jd_first": float(jd.min()), "jd_last": float(jd.max())}
json.dump(out, open(f"{S}/lc_model.json", "w"), indent=1)
print(f"good L nights {len(good)}, failed {len(bad)}, G nights {len(gpts)}")
print(f"fit K={K}: chi2 {chi2:.1f} / dof {dof}, rms resid {out['rms_resid']:.3f} mag")
print(f"max {m_max:.3f} at phase {ph_max:.3f} ({out['date_max_obs']} UT); min {m_min:.3f}; amplitude {m_min-m_max:.2f}; rise fraction {rise:.2f}")
print(f"ephemeris max would be {out['date_max_ephemeris']}; next max {out['next_max_date']}")
print(f"mean over cycle {mean_cycle:.3f} -> {dist_cycle}")
print(f"mean over nights {mean_nights:.3f} ({n_nights}) -> {dist_nights}")
print("coverage", cov)
print("upcoming bins filled:", sorted({u['bin'] for u in up} & set(cov['missing'])), "of missing", cov["missing"])
