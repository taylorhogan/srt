"""Benchmark SNR convergence: parallel (threads) vs sequential, on real data.

Runs stacker.convergence_curve for each filter of a DSO both ways and reports
per-filter and wall-clock times so we can see whether the threaded filter
parallelism actually pays off on this machine/workload. Parallel runs FIRST so
the sequential pass benefits from a warm OS file cache — any parallel win is
therefore conservative.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import config
from stacking import stacker

DSO = sys.argv[1] if len(sys.argv) > 1 else "sh2-92"

cfg = config.data()
image_dir = Path(cfg["nina"]["image_dir"])
scratch = Path(os.path.join(os.path.dirname(__file__), "..", cfg["scratch"]["directory"]))
scratch.mkdir(parents=True, exist_ok=True)

dso_dir = image_dir / DSO
fits = sorted(
    (f for f in dso_dir.rglob("*.fits") if f.parent.name.upper() == "LIGHT"),
    key=lambda f: f.stat().st_mtime,
)
by_filter = stacker.group_by_filter(fits)
print(f"{DSO}: {len(fits)} LIGHT frames", flush=True)
for fn, paths in by_filter.items():
    print(f"  {fn}: {len(paths)} frames", flush=True)


def run_one(fn, paths, tag):
    out = scratch / f"bench_{tag}_convergence_{fn}.jpg"
    gold = scratch / f"bench_{tag}_golden_{fn}.jpg"
    t0 = time.perf_counter()
    try:
        _, _, slope, rmse = stacker.convergence_curve(
            paths, filter_name=fn, output_path=out, golden_output_path=gold,
        )
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"  [{tag}] {fn}: FAILED after {dt:.1f}s — {exc}", flush=True)
        return fn, dt, None
    dt = time.perf_counter() - t0
    print(f"  [{tag}] {fn}: {len(paths)} frames -> {dt:.1f}s  slope {slope:+.4f}%/frame  RMSE {rmse:.2f}%", flush=True)
    return fn, dt, slope


# ---- PARALLEL (threads), first so sequential gets the warm cache ----
print("\n=== PARALLEL (2 threads) ===", flush=True)
tp0 = time.perf_counter()
par_times = {}
with ThreadPoolExecutor(max_workers=min(len(by_filter), 4)) as pool:
    futs = [pool.submit(run_one, fn, paths, "par") for fn, paths in by_filter.items()]
    for fut in as_completed(futs):
        fn, dt, _ = fut.result()
        par_times[fn] = dt
par_wall = time.perf_counter() - tp0
print(f"PARALLEL wall-clock: {par_wall:.1f}s", flush=True)

# ---- SEQUENTIAL ----
print("\n=== SEQUENTIAL ===", flush=True)
ts0 = time.perf_counter()
seq_times = {}
for fn, paths in by_filter.items():
    _, dt, _ = run_one(fn, paths, "seq")
    seq_times[fn] = dt
seq_wall = time.perf_counter() - ts0
print(f"SEQUENTIAL wall-clock: {seq_wall:.1f}s", flush=True)

print("\n=== SUMMARY ===", flush=True)
for fn in by_filter:
    print(f"  {fn}: seq {seq_times.get(fn,0):.1f}s | par {par_times.get(fn,0):.1f}s", flush=True)
print(f"  TOTAL: sequential {seq_wall:.1f}s -> parallel {par_wall:.1f}s", flush=True)
if par_wall > 0:
    print(f"  SPEEDUP: {seq_wall / par_wall:.2f}x  (ideal for {len(by_filter)} filters = {len(by_filter)}.00x)", flush=True)
