import sys, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/taylor/Documents/srt")

def main():
    S = Path(sys.argv[1])
    npy = S / "hubblev1_L_raw.npy"
    log = S / "stack_v1.log"
    while not (npy.exists() and log.exists() and "done" in log.read_text()[-400:]):
        time.sleep(15)
    time.sleep(5)
    from nn import denoiser
    raw = np.load(npy).astype(np.float32)
    print("raw", raw.shape, "sky", float(np.median(raw)), flush=True)
    model = denoiser.load_model(Path("/home/taylor/Documents/srt/local/models/n2n_ladder_groups_galaxies_300s.pt"))
    t0 = time.time()
    den = denoiser.denoise_frame(raw, model)[:raw.shape[0], :raw.shape[1]]
    print("denoised in %.0f s" % (time.time() - t0), flush=True)
    np.save(S / "hubblev1_L_den.npy", den.astype(np.float32))
    print("done", flush=True)

if __name__ == "__main__":
    main()
