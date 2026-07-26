#!/usr/bin/env python3
"""Plot the latest open and close roof-motor wattage traces on one graph.

Finds the most recent open and the most recent close signature across all
library buckets (good/unlabeled/bad), then overlays their power-vs-time traces
in two colours.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SIG_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "sentry", "roof_signatures"
)


def latest_signature(direction):
    """Newest signature JSON for a direction, across all status buckets."""
    pattern = os.path.join(_SIG_ROOT, "*", direction, f"*_{direction}.json")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"no {direction} signatures found under {_SIG_ROOT}")
    # Filenames lead with an ISO timestamp, so lexical sort == chronological.
    latest = max(paths, key=lambda p: os.path.basename(p))
    with open(latest) as f:
        return latest, json.load(f)


def main():
    open_path, open_sig = latest_signature("open")
    close_path, close_sig = latest_signature("close")

    fig, ax = plt.subplots(figsize=(11, 6))

    for sig, color, label in (
        (open_sig, "tab:red", "open"),
        (close_sig, "tab:blue", "close"),
    ):
        t = sig["samples"]["t"]
        p = sig["samples"]["power"]
        feats = sig.get("features", {})
        run_w = feats.get("running_w")
        legend = f"{label}  ({sig.get('timestamp', '?')}"
        if run_w is not None:
            legend += f", running {run_w:.0f} W"
        legend += ")"
        ax.plot(t, p, color=color, linewidth=1.2, label=legend)

    ax.set_xlabel("time since capture start (s)")
    ax.set_ylabel("real power (W)")
    ax.set_title("Roof motor wattage — latest open vs close")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "roof_wattage_open_vs_close.png")
    out = os.path.abspath(out)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"open : {os.path.basename(open_path)}")
    print(f"close: {os.path.basename(close_path)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
