#!/usr/bin/env python3
"""Compare color_process.auto_stretch against the fixed stretch on cached stacks.

    python scripts/auto_stretch_compare.py --out ~/Desktop/auto_stretch_compare

For every (dso, recipe) with a manifest in local/n2n_lrgb_render/, the cached
channel stacks are denoised once, then composed three ways on one shared white
point (p99.95): the routine's fixed default (black at sky - 0.5 sigma in every
channel, softening 0.025), the auto rule, and — where one exists — the stretch
chosen by hand for that target. Raw and denoised composites are written for
each, plus an HTML sheet and a JSON of the numbers auto chose, so the rule can
be judged against the eye on targets whose by-hand renders already exist.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from stacking import color_process
from stacking.color_process import _prepare, RECIPES

OUT = Path(_root) / "local" / "n2n_lrgb_render"

TARGETS = [("ngc7380", "HOO"), ("ngc6888", "SHO"), ("ic1396", "HOO"),
           ("trunk", "HOO"), ("squid", "HOO"), ("ngc5907", "LRGB"), ("m33", "LRGB")]

# Stretches chosen by eye, keyed like TARGETS: per-channel black margin in
# sigma below sky, and the softening. NGC 7380 HOO is the 2026-09-12 deep render.
HAND = {("ngc7380", "HOO"): {"margin": {"R": 0.5, "G": 0.0, "B": 0.0}, "soft": 0.006}}

WHITE_PCT = 99.95
DEFAULT_MARGIN = 0.5
DEFAULT_SOFT = 0.025


def log(m=""):
    print(m, flush=True)


def sky_margin_blacks(subbed: dict, margin) -> dict:
    """Black at sky - k*sigma per channel, sigma from sep like the routine does."""
    import sep
    out = {}
    for c, arr in subbed.items():
        k = margin[c] if isinstance(margin, dict) else margin
        a = np.ascontiguousarray(arr.astype(np.float32))
        bkg = sep.Background(a)
        out[c] = float(np.nanmedian(a)) - k * float(bkg.globalrms)
    return out


def render(subbed: dict, blacks: dict, white: float, soft: float, path: Path, max_px: int):
    def st(chan, black):
        y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
        return np.arcsinh(y / soft) / np.arcsinh(1.0 / soft)
    rgb = np.dstack([st(subbed[c], blacks[c]) for c in ("R", "G", "B")])
    if "L" in subbed:
        lum = st(subbed["L"], blacks["L"])
        rl = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(rl > 1e-4, lum / np.maximum(rl, 1e-4), 0.0)
        rgb = rgb * np.clip(sc, 0.0, color_process.MAX_LUM_BOOST)[:, :, None]
    return color_process.save_rgb(np.clip(np.nan_to_num(rgb), 0, 1), path, max_px=max_px)


def load_model(path):
    import torch
    from nn.noise2noise_model import UNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = UNet(residual="linear")
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-px", type=int, default=1600)
    ap.add_argument("--targets", default="", help="comma list of dso:recipe to restrict to")
    args = ap.parse_args()
    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, os.path.join(_root, "scripts"))
    from n2n_lrgb_render import pick_model
    import torch
    from nn import denoiser
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}

    targets = TARGETS
    if args.targets:
        want = {tuple(t.split(":")) for t in args.targets.split(",")}
        targets = [t for t in TARGETS if t in want]

    report = {}
    for dso, recipe in targets:
        mp = OUT / f"meta_{dso}_{recipe}.json"
        if not mp.exists():
            log(f"{dso} {recipe}: no manifest, skipped")
            continue
        meta = json.loads(mp.read_text())
        mapping = RECIPES[recipe]
        filters = sorted({mapping[ch] for ch in ("L", "R", "G", "B") if ch in mapping} & set(meta["channels"]))
        log(f"\n=== {dso} {recipe}  filters {filters}")
        raw = {f: np.load(OUT / meta["channels"][f]["file"]).astype(np.float32) for f in filters}
        h = min(a.shape[0] for a in raw.values()); w = min(a.shape[1] for a in raw.values())
        raw = {k: v[:h, :w] for k, v in raw.items()}

        domain, model_path = pick_model(filters)
        if model_path not in models:
            models[model_path] = load_model(model_path)
        den = {}
        for f in filters:
            t0 = time.time()
            den[f] = denoiser.denoise_frame(raw[f], models[model_path], device=dev)[:h, :w]
            log(f"  denoised {f} in {time.time() - t0:.0f}s ({domain})")

        def to_ch(d):
            return {ch: d[mapping[ch]] for ch in ("L", "R", "G", "B") if ch in mapping and mapping[ch] in d}

        sub_raw, white = _prepare(to_ch(raw), True, color_process.BG_MESH_FRACTION, WHITE_PCT)
        sub_den, _ = _prepare(to_ch(den), True, color_process.BG_MESH_FRACTION, WHITE_PCT)
        del raw, den

        variants = {}
        variants["default"] = {"blacks": sky_margin_blacks(sub_raw, DEFAULT_MARGIN), "soft": DEFAULT_SOFT}
        log("  auto stretch:")
        auto = color_process.auto_stretch(sub_raw, white, lum="L", log=log)
        variants["auto"] = {"blacks": auto["blacks"], "soft": auto["softening"]}
        if (dso, recipe) in HAND:
            hd = HAND[(dso, recipe)]
            variants["hand"] = {"blacks": sky_margin_blacks(sub_raw, hd["margin"]), "soft": hd["soft"]}

        entry = {"white": white, "filters": filters, "variants": {}, "auto": {
            "channels": auto["channels"], "dominant": auto["dominant"], "anchors": auto["anchors"]}}
        for name, v in variants.items():
            for tag, sub in (("raw", sub_raw), ("denoised", sub_den)):
                p = render(sub, v["blacks"], white, v["soft"],
                           out_dir / f"{dso}_{recipe}_{name}_{tag}.jpg", args.max_px)
            entry["variants"][name] = {"blacks": v["blacks"], "soft": v["soft"]}
            log(f"  {name}: soft {v['soft']:.4f} blacks " + ", ".join(f"{c} {b:+.2f}" for c, b in sorted(v["blacks"].items())))
        report[f"{dso}_{recipe}"] = entry
        del sub_raw, sub_den

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    write_html(out_dir, report)
    log(f"\nwrote {out_dir / 'index.html'}")
    return 0


def write_html(out_dir: Path, report: dict) -> None:
    rows = []
    for key, e in report.items():
        names = list(e["variants"])
        cells = "".join(
            f'<figure><img data-key="{key}_{n}" src="{key}_{n}_denoised.jpg" loading="lazy">'
            f'<figcaption><b>{n}</b> · soft {e["variants"][n]["soft"]:.4f} · black '
            + ", ".join(f'{c} {b:+.2f}' for c, b in sorted(e["variants"][n]["blacks"].items()))
            + '</figcaption></figure>' for n in names)
        a = e["auto"]
        diag = "".join(
            f'<tr><td>{c}</td><td>{d["sky"]:+.3f}</td><td>{d["sigma"]:.3f}</td>'
            f'<td>{100*d["faint_fraction"]:.1f}%</td><td>{100*d["bright_fraction"]:.1f}%</td>'
            f'<td>{d["margin_sigma"]:.2f}σ</td><td>{d["black"]:+.3f}</td></tr>'
            for c, d in sorted(a["channels"].items()))
        an = a["anchors"]
        bright = f' · bright p90 {an["bright_adu"]:.1f} ADU → {an["bright_out"]:.2f}' if "bright_out" in an else ""
        rows.append(f'''
<section><h2>{key.replace("_", " · ")} <span class="dim">white {e["white"]:.1f} ADU</span></h2>
<div class="grid c{len(names)}">{cells}</div>
<details><summary>auto diagnostics · softening from {a["dominant"]}: sky+2σ = {an["faint_adu"]:.2f} ADU → {an["faint_out"]:.2f} of white{bright}</summary>
<table><tr><th>ch</th><th>sky</th><th>σ</th><th>faint excess</th><th>bright</th><th>margin</th><th>black</th></tr>{diag}</table></details>
</section>''')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auto Stretch Comparison</title>
<style>
body{{margin:0;padding:24px 16px 48px;background:#0b0d12;color:#d7dbe3;font:15px/1.5 system-ui,sans-serif}}
h1{{font-size:22px;color:#fff;margin:0 0 4px}} h2{{font-size:16px;color:#fff;margin:28px 0 8px;font-weight:600}}
.dim{{color:#8a92a3;font-weight:400;font-size:13px}} .sub{{color:#8a92a3;max-width:900px;margin-bottom:16px}}
.bar{{position:sticky;top:0;background:#0b0d12;padding:8px 0;display:flex;gap:8px;align-items:center;z-index:2}}
button{{background:#161a24;color:#d7dbe3;border:1px solid #2a3040;border-radius:6px;padding:6px 14px;cursor:pointer;font:inherit}}
button.on{{background:#2b57d9;border-color:#2b57d9;color:#fff}}
.grid{{display:grid;gap:10px}} .c2{{grid-template-columns:1fr 1fr}} .c3{{grid-template-columns:1fr 1fr 1fr}}
@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
figure{{margin:0;background:#000;border:1px solid #222838;border-radius:8px;overflow:hidden}}
figure img{{display:block;width:100%;height:auto}} figcaption{{padding:6px 10px;font-size:12px;color:#aab2c2;background:#10131b}}
details{{margin-top:8px;font-size:13px;color:#aab2c2}} table{{border-collapse:collapse;margin-top:6px}}
td,th{{padding:3px 10px;border-bottom:1px solid #222838;text-align:left}} th{{color:#8a92a3;font-weight:500}}
</style></head><body>
<h1>Auto stretch vs fixed default</h1>
<div class="sub">Same cached stacks, same white point (p99.95 of the raw colour channels), same denoiser. <b>default</b> is the routine's
fixed rule: black at sky − 0.5σ in every channel, softening 0.025. <b>auto</b> measures each channel's faint-signal excess over a
pure-noise sky model and sets the margin from it (0 → 0.5σ), then solves the softening so sky + 2σ of the dominant channel lands at
0.30 of white, capped so the bright p90 stays under 0.90. <b>hand</b> is the stretch chosen by eye where one exists.</div>
<div class="bar"><button id="bd" class="on">denoised</button><button id="br">raw</button><span class="dim">space toggles</span></div>
{"".join(rows)}
<script>
let mode="denoised";
function set(m){{mode=m;document.querySelectorAll("img[data-key]").forEach(i=>i.src=i.dataset.key+"_"+m+".jpg");
 document.getElementById("bd").classList.toggle("on",m=="denoised");document.getElementById("br").classList.toggle("on",m=="raw");}}
document.getElementById("bd").onclick=()=>set("denoised");document.getElementById("br").onclick=()=>set("raw");
document.addEventListener("keydown",e=>{{if(e.key==" "){{e.preventDefault();set(mode=="raw"?"denoised":"raw")}}}});
</script></body></html>'''
    (out_dir / "index.html").write_text(html)


if __name__ == "__main__":
    sys.exit(main())
