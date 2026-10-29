"""spark_prune.py -- what the Spark could clear when its disk fills, and clear it on `--go`.

    python scripts/spark_prune.py            # report: what would go, and what never will
    python scripts/spark_prune.py --go       # delete the stale caches listed

docs/AUTOMATION_PLAN.md B6 (2026-10-29). The Spark holds every synced frame
(rsync of the observatory's Targets), the morning render's stack caches
(local/n2n_lrgb_render/<dso>_<filter>_raw.npy with a .json sidecar), their
denoised / learned intermediates, and the N2N and deconvolution models. Only
the caches are expendable: a stack is rebuilt from the frames on the next
render that needs it (the sidecar's `built` says when it was last made, and a
target still being imaged is rebuilt every morning, so an old `built` means a
target nobody is imaging).

NEVER touched, and listed so the report says so: anything under the image
directory (the raw lights and the cdk17 calibration frames: CLAUDE.md, raw
lights are never pruned), local/models and local/deconv/models.

The morning trigger runs the report in the disk's warn band; `--go` is run by
hand, or by the trigger once cfg["disk"]["spark_prune_go"] is True.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RENDER_DIR = "local/n2n_lrgb_render"
KEEP_DAYS = 45.0
# Files beside a stack that belong to it: the intermediates the render writes
# from the npy and can write again.
STACK_SIDE_SUFFIXES = (".json", "_den.npz", "_learned.npy")
NEVER = ("Targets", "LIGHT", "FLAT", "DARK", "BIAS", "cdk17", "models")


def protected(path, image_dir=None) -> bool:
    """True for anything that must never be deleted here. Pure on the path."""
    p = Path(str(path))
    parts = set(p.parts)
    if parts & set(NEVER):
        return True
    if image_dir and str(p.resolve()).startswith(str(Path(str(image_dir)).resolve())):
        return True
    return False


def prune_stale_stacks(render_dir, keep_days: float = KEEP_DAYS, now: float = None,
                       image_dir=None) -> list:
    """[(path, bytes, age_days)] of cached stacks older than *keep_days*. Pure
    on the tree: a stack is <dso>_<filter>_raw.npy whose sidecar .json has a
    `built` timestamp; one without a sidecar is judged by the npy's mtime."""
    now = time.time() if now is None else float(now)
    cutoff = now - float(keep_days) * 86400.0
    out = []
    d = Path(str(render_dir))
    if not d.is_dir():
        return out
    for npy in sorted(d.glob("*_raw.npy")):
        if protected(npy, image_dir):
            continue
        built = None
        side = npy.with_suffix(".json")
        try:
            built = _parse_built(json.loads(side.read_text(encoding="utf-8")).get("built"))
        except Exception:  # noqa: BLE001
            built = None
        try:
            stamp = built if built is not None else npy.stat().st_mtime
        except OSError:
            continue
        if stamp < cutoff:
            size = _size(npy)
            for extra in (side,) + tuple(npy.with_name(npy.stem.replace("_raw", "") + sfx)
                                          for sfx in STACK_SIDE_SUFFIXES[1:]):
                if extra.exists():
                    size += _size(extra)
            out.append((npy, size, (now - stamp) / 86400.0))
    return out


def _parse_built(v) -> float:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    from datetime import datetime
    return datetime.fromisoformat(str(v)).timestamp()


def _size(p) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def delete_stack(npy, image_dir=None) -> int:
    """Remove the stack and its sidecar / intermediates. -> bytes freed."""
    if protected(npy, image_dir):
        raise PermissionError("refusing to delete %s" % npy)
    freed = 0
    victims = [npy, npy.with_suffix(".json")]
    victims += [npy.with_name(npy.stem.replace("_raw", "") + sfx) for sfx in STACK_SIDE_SUFFIXES[1:]]
    for v in victims:
        if v.exists() and not protected(v, image_dir):
            freed += _size(v)
            v.unlink()
    return freed


def report(stale: list, keep_days: float, image_dir) -> str:
    lines = ["spark_prune: %d stale stack(s) older than %.0f d, %.1f GB"
             % (len(stale), keep_days, sum(b for _, b, _ in stale) / 1e9)]
    for p, b, age in stale:
        lines.append("  %s  %.2f GB  built %.0f d ago" % (p.name, b / 1e9, age))
    lines.append("never pruned here: %s (raw lights and calibration), local/models, "
                 "local/deconv/models" % image_dir)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="delete what the report lists")
    ap.add_argument("--keep-days", type=float, default=None)
    args = ap.parse_args()
    from configs import config
    cfg = config.data()
    keep = float(args.keep_days or cfg.get("disk", {}).get("stack_keep_days", KEEP_DAYS))
    image_dir = cfg.get("nina", {}).get("image_dir")
    stale = prune_stale_stacks(ROOT / RENDER_DIR, keep, image_dir=image_dir)
    print(report(stale, keep, image_dir))
    if not args.go:
        return 0
    freed = 0
    for p, _b, _age in stale:
        freed += delete_stack(p, image_dir)
        print("deleted", p.name)
    print("freed %.1f GB" % (freed / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
