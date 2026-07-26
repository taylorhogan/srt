"""Isolate the GIL from disk I/O in the SNR filter parallelism.

The full sh2-92 LIGHT set (23.5 GB) exceeds free RAM (~20 GB), so a full parallel
run is inherently disk/page-cache bound and can't tell us whether threads are held
back by the GIL. This uses a SUBSET small enough to live entirely in page cache,
warms it, then times parallel vs sequential. With the disk removed:
  * per-filter time still inflated under parallel  -> GIL-bound (processes help)
  * per-filter time ~flat under parallel           -> was always I/O (processes won't help)
"""
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import config
from stacking import stacker
# Silence astropy's fit-warning flood — do this AFTER import so we don't clobber
# astropy's custom logger class during its own initialisation.
import logging
logging.getLogger("astropy").setLevel(logging.ERROR)

DSO = "sh2-92"
PER_FILTER = int(sys.argv[1]) if len(sys.argv) > 1 else 24  # frames/filter (fit in RAM)

cfg = config.data()
image_dir = Path(cfg["nina"]["image_dir"])
scratch = Path(os.path.join(os.path.dirname(__file__), "..", cfg["scratch"]["directory"]))
scratch.mkdir(parents=True, exist_ok=True)

dso_dir = image_dir / DSO
fits = sorted(
    (f for f in dso_dir.rglob("*.fits") if f.parent.name.upper() == "LIGHT"),
    key=lambda f: f.stat().st_mtime,
)
by_filter_full = stacker.group_by_filter(fits)
# Cap each filter to PER_FILTER frames so the working set fits in RAM.
by_filter = {fn: paths[:PER_FILTER] for fn, paths in by_filter_full.items()}
subset = [p for paths in by_filter.values() for p in paths]
total_gb = sum(p.stat().st_size for p in subset) / (1024**3)
print(f"{DSO}: subset {len(subset)} frames ({total_gb:.1f} GB) — fits in RAM", flush=True)
for fn, paths in by_filter.items():
    print(f"  {fn}: {len(paths)} frames", flush=True)

# Warm the page cache: read every subset file once so both passes hit RAM, not disk.
print("warming page cache…", flush=True)
t = time.perf_counter()
for p in subset:
    with open(p, "rb") as fh:
        while fh.read(1 << 24):
            pass
print(f"warmed in {time.perf_counter()-t:.1f}s", flush=True)


def run_one(fn, paths, tag):
    out = scratch / f"gil_{tag}_conv_{fn}.jpg"
    t0 = time.perf_counter()
    try:
        stacker.convergence_curve(paths, filter_name=fn, output_path=out)
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"  [{tag}] {fn}: FAILED {dt:.1f}s — {exc}", flush=True)
        return fn, dt
    dt = time.perf_counter() - t0
    print(f"  [{tag}] {fn}: {len(paths)} frames -> {dt:.1f}s", flush=True)
    return fn, dt


# SEQUENTIAL (warm) — clean single-thread baseline
print("\n=== SEQUENTIAL (warm) ===", flush=True)
seq = {}
ts = time.perf_counter()
for fn, paths in by_filter.items():
    _, dt = run_one(fn, paths, "seq")
    seq[fn] = dt
seq_wall = time.perf_counter() - ts
print(f"SEQUENTIAL wall: {seq_wall:.1f}s", flush=True)

# PARALLEL (warm) — disk is out of the picture, so any per-filter inflation is GIL
print("\n=== PARALLEL 2 threads (warm) ===", flush=True)
par = {}
tp = time.perf_counter()
with ThreadPoolExecutor(max_workers=len(by_filter)) as pool:
    futs = [pool.submit(run_one, fn, paths, "par") for fn, paths in by_filter.items()]
    for fut in as_completed(futs):
        fn, dt = fut.result()
        par[fn] = dt
par_wall = time.perf_counter() - tp
print(f"PARALLEL wall: {par_wall:.1f}s", flush=True)

print("\n=== SUMMARY (warm, in-RAM) ===", flush=True)
for fn in by_filter:
    infl = (par[fn] / seq[fn] - 1) * 100 if seq.get(fn) else 0
    print(f"  {fn}: seq {seq[fn]:.1f}s | par {par[fn]:.1f}s  (+{infl:.0f}% under parallel)", flush=True)
print(f"  wall: sequential {seq_wall:.1f}s -> parallel {par_wall:.1f}s  = {seq_wall/par_wall:.2f}x", flush=True)
print("  per-filter inflation ~0% => I/O-bound (processes won't help)", flush=True)
print("  per-filter inflation large => GIL-bound (processes would help small nights)", flush=True)
