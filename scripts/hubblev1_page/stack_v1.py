import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/taylor/Documents/srt")

def main():
    from nn import stacks
    S = Path(sys.argv[1])
    paths = sorted(Path("/home/taylor/Desktop/Targets/hubblev1/cdk17").glob("*/LIGHT/*.fits"))
    print("stacking", len(paths), "L frames", flush=True)
    meta = {}
    img = stacks.stack_paths(paths, "L", progress_cb=lambda s: print(s, flush=True), meta_out=meta, min_frames=1)
    np.save(S / "hubblev1_L_raw.npy", img.astype(np.float32))
    json.dump({k: (v if k != "used" else len(v)) for k, v in meta.items()}, open(S / "hubblev1_L_raw.json", "w"), indent=1, default=str)
    print("done", img.shape, flush=True)

if __name__ == "__main__":
    main()
