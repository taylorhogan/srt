# Noise2Noise on the Spark — runbook

Everything N2N runs **only on the Spark** (`spark-3129`). `subs_dir` is defined
for that host alone in `configs/config_public.py`, and the Windows observatory
PC has no torch in its `.venv`. Nothing here can be tested on the observatory
machine.

Paths on the Spark: repo `/home/taylor/Documents/srt`, frames
`/home/taylor/Desktop/Targets`.

The 06:00 cron (`scripts/spark_morning_search.bsh`) does the rsync from the
observatory and then the transit search. **N2N is not wired into it** — this is
a manual chain.

---

## 0. Get the code and check the GPU

```bash
cd /home/taylor/Documents/srt && git pull
source .venv/bin/activate
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 1. Smoke test before the real run

Prove the pipeline runs before spending 130 epochs on it. This step has already
earned its keep once — see "What changed on 2026-08-12" below, where it caught a
divergence that made the whole run worthless.

There is no `--epochs` flag. The value comes from `cfg["nn"]["epochs"]` in
`configs/config_public.py`. Lower it to ~5, run step 2, confirm it completes and
writes a checkpoint, then put it back.

**Do not judge the smoke test by "it finished", or by the loss.** It finishes
either way, and a *constant* predictor also drives the loss down — on 2026-08-12
a fully collapsed net reported val 0.2230 against a constant-predictor baseline
of 0.2263, which looks like a converged run and is not one.

The cheap test that actually catches it, and would have saved ~4 h:

```python
# after training, on any batch
out = model(inp)
print("corr:", np.corrcoef(inp.ravel(), out.ravel())[0,1])
print("std ratio:", out.std()/inp.std())
```

**Compare against the ceiling, not against 1.0.** These frames are noise
dominated, so a *perfect* denoiser cannot approach corr 1 — its output is the
clean image while the input is clean+noise. The ceiling is measurable from any
two frames of the same target:

```python
r12 = np.corrcoef(t_i.ravel(), t_j.ravel())[0, 1]   # two asinh-normalised frames
ceiling = np.sqrt(r12)        # = corr(input, perfect output), also the std ratio
```

On abell2151 R 300s that is **0.37**, not 0.9. Judge the run as a fraction of
it: 2026-08-12 measured 0.3188 against a 0.3705 ceiling (86%) for a healthy net,
against **-0.0158 for the collapsed one**. A first pass at this check used a
naive ">0.9 is healthy" threshold and duly failed a model that was working —
compute the ceiling first.

Then read the per-epoch numbers: every epoch's `train` and `val` must be the
same order of magnitude as epoch 1. If `best` is much lower than the last
epoch's `val`, the run diverged and the checkpoint is whatever early epoch
preceded the blowup. `local/runs/` has the per-epoch curve when the console
only prints every tenth epoch:

```bash
python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for tag in ("loss_train", "loss_val"):
    ea = EventAccumulator(f"local/runs/n2n_R_300s/{tag}"); ea.Reload()
    for t in ea.Tags()["scalars"]:
        print(tag, [f"e{e.step}={e.value:.4f}" for e in ea.Scalars(t)])
EOF
```

Note the training script **deletes** `local/runs/n2n_{filter}_{s}s/` on startup,
so read the curve before re-running.

## 2. Train

```bash
python scripts/n2n_train.py <filter> <seconds>     # e.g. R 300
```

Needs at least 20 frames spread over at least 2 DSOs that have ≥2 frames each.

**Write down the line it prints:**

```
Held-out DSOs for validation: <names> (N/M frames); training on K
```

Step 4 is meaningless without it.

Any existing checkpoint for this filter/exposure is deleted first, and older
ones will fail to load anyway — `load_state_dict` is strict, and all three of
the fixes below changed the model or its inputs. That failure is intended.

## 3. Denoise

```bash
python scripts/n2n_denoise.py <filter> <seconds>
```

Output goes to a flat `denoised/{filter}_{seconds}s/` containing every DSO,
keyed on bare frame names. Checked 2026-08-11: 1189 LIGHT frames, zero name
collisions — but it is a real hazard if NINA's naming ever restarts per target.

## 4. Evaluate — on a held-out DSO from step 2

```bash
python scripts/n2n_evaluate.py <dso> <filter> <seconds>
```

Measuring on a target the model trained on answers a question nobody asked.
Plots land in `local/n2n_eval/`.

### Reading the result

Two measurements, because they fail in different ways.

**Convergence.** `decay_ratio` says how far the curve's tail sits above what
*independent* noise predicts. ~1.0 is textbook 1/√N.

| outcome | meaning |
| --- | --- |
| denoised decay_ratio ≈ raw's, RMSE lower | genuinely bought frames |
| denoised decay_ratio ≫ raw's | bias shared across frames — stacking will never remove it |
| curve flattens to a floor raw does not have | same, and worse |

The reason this is the right test: every frame passes through the same network
with the same learned prior, so whatever it gets wrong it gets wrong the same
way on all of them. Independent noise averages down as √N; a shared bias
survives stacking untouched.

**Photometry.** Aperture flux, raw stack vs denoised, binned by brightness.
Sources are detected on the **raw** stack only — detecting separately would
compare different star lists and hide a source the denoiser erased.

Any bin below ~0.97 is a systematic brightness error. The predicted failure is
specific and faint-end only: the training loss is L1, whose optimum is the
conditional *median*, and for a source near the detection limit the posterior
mass sits at background — so an L1-trained net is structurally inclined to erase
faint sources while leaving bright ones alone.

The instrument was checked against synthetic ground truth: unbiased input reads
0.9996–1.0016 in every bin; 20% faint suppression reads bright 0.9999, mid
0.804, faint 0.800.

---

## What changed on 2026-08-11 (commit 91b43b9)

Three bugs, all of which invalidate older checkpoints:

1. **Registration used one global reference** — the first frame of the first
   DSO — for every frame of every target. A translation-only fit capped at
   200 px cannot align targets degrees apart. Frames of every DSO but the first
   were either silently unregistered or shifted by a spurious median, and
   nothing in the output distinguished the two. Now per-DSO, matching training.
2. **Train/inference normalisation had drifted apart again.** `c617047` added
   background subtraction to inference; training still normalised the raw frame.
   Measured scale error 1.0–1.7×, and *field-dependent* (m13 R 1.73×, ngc5907 L
   1.08×, sh2-92 Ha 1.03×, bubble Ha 0.98×). Both paths now share
   `denoiser.subtract_background()` and `denoiser.normalise()`.
3. **BatchNorm removed** (`UNet(norm=...)`, default `"none"`, still switchable).
   The N2N paper's network has none; this is a regression where the absolute
   output level carries the photometry; and BN bakes training activation
   statistics into running means, which amplifies exactly the shift in (2).

## Root cause of the 2026-08-12 collapse: registration

Read this before touching anything else in the N2N chain.

`nn/registration.py` was taking frames that arrive aligned to **0.0-0.1 px**
and shifting them apart by **3-14 px**:

```
                    BEFORE   AFTER (old code)
m92       frame1     0.10  ->  2.84
          frame3     0.10  ->  5.44
          frame4     8.61  ->  0.67   <- the one real dither, fixed
abell2151 frame1     0.00  ->  5.58
          frame4     0.00  -> 14.02
```

At FWHM ~6 px that is 1-2.3 FWHM, so a training pair's stars did not overlap.
N2N requires two noisy views of the *same* scene; given pairs that far apart the
only learnable answer is the local mean. The net collapsed to a constant
uncorrelated with its input (corr -0.02) and erased every star, which is what
made the denoised frames unregisterable and unmeasurable downstream.

The old estimator decimated by 2 (centroids +-1-2 px) then nearest-neighbour
matched two 50-star lists at a 200 px tolerance. In a dense field that pairs
*different* stars, and the median of those mismatched deltas is a spurious
shift — applied unconditionally, with nothing checking it helped.

Now: phase cross-correlation, star matching kept only as a fallback, and shifts
below `min_shift=0.5 px` are not applied at all (nd_shift's order-3 spline
low-pass filters the frame, which correlates the noise N2N assumes is
independent). Verified — the one genuine 9 px dither is corrected to 0.40 px and
the 11 already-aligned frames are left untouched.

**The lesson for this pipeline:** every one of these bugs produced plausible
output. A diverging run still wrote a checkpoint; a collapsed net still wrote
204 FITS files of the right size; a destructive registration still reported
success for every frame. Nothing short of measuring against ground truth caught
any of them.

## What changed on 2026-08-12

First execution of the 91b43b9 repairs on real hardware. One blocker, found by
the step-1 smoke test.

**Training diverged at epoch 2** — train loss 0.20 → 24834, val 0.19 → 1438 —
and did not recover in the remaining epochs. Because the checkpoint is selected
on best val, the saved model was the *epoch-1* weights: an untrained network
that would have gone through denoise and evaluate looking like a real result.

Cause is the interaction with fix #3. Removing BatchNorm is right for the
photometry, but BN had been renormalising activations between layers and hiding
a heavy-tailed gradient. Measured over 250 batches at `lr=1e-3` with no
clipping: median grad norm 0.2, p99 3.1, **max 45.8** (221x the median), 8% of
batches above 1.0. Adam takes a step off that tail which the plain conv stack
cannot absorb.

Fix: `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` in `nn/trainer.py`.
Clipping rather than rescaling the input, because `denoiser.normalise()` has to
stay bit-identical between training and inference — that has already drifted
twice (c617047, c61cd26) and is what fix #2 repaired. After clipping, 130 epochs
ran monotonically with no divergence.

Worth knowing and **not** the cause: `normalise()` says it scales to `~[0,1]`
and actually produces values up to ~1600, because 99% of pixels are sky so
`p99-p1` spans only the sky noise. The obvious story — bright star, huge
gradient — was measured and is false: correlation between patch peak and grad
norm was -0.001, and the top-5% grad-norm batches had slightly *lower* peaks.
The trigger is still unidentified. Clipping bounds it either way.

Also observed, not fixed:

- `best` was set at **epoch 45** and never beaten in the remaining 85. The
  130-epoch setting is buying nothing at this dataset size.
- `registration.py` copies every frame up front (`[f.copy() for f in frames]`)
  even though `nd_shift` immediately replaces most. That is the bulk of the
  ~98 GB peak and most of the ~25 min of prep per run.
- `n2n_evaluate.py` never passes `precomputed_fwhm_stars`, so FWHM is measured
  four times (raw twice, denoised twice) at ~87 s/frame when the stacker already
  supports reusing it. Two of the four passes are redundant.
- The Spark has **no calibration masters**, so the evaluator's RMSE% is on the
  bias-pedestal scale and is ~100x compressed. Raw-vs-denoised comparison still
  holds; absolute numbers do not.

### Why the frame_stats.json cache does not rescue the evaluate step

The obvious fix for the 5-6 h evaluate is to feed `precomputed_fwhm_stars` from
the `<dso>/frame_stats.json` caches the `stats` command writes. Checked on
2026-08-12 — it does not carry, for four independent reasons, and it fails
*silently* rather than loudly:

1. **Coverage.** m92 has 14 R@300s rows against the 56 frames the eval uses.
   Per-DSO R@300s counts: abell2151 7, abell2218 0, m13 20, m92 14, ngc5033 19,
   ngc5907 15.
2. **Paths are the observatory PC's**, `C:\Users\iriso\...`, in every row of
   all 8 files. `_split_cached` keys on `Path`, so nothing matches on the Spark
   without translation.
3. **FWHM is `fwhm_arcsec`; the stacker wants pixels.** There is no px key.
   Passing arcsec through mis-weights and mis-gates frames instead of erroring
   (~0.26"/px if converting).
4. **No PSF-model stamp** — these rows predate the Moffat change, and d7529e8
   exists to refuse mixing PSF models.

Capping it regardless: **denoised frames are never in the cache**, since they
are written fresh each run. Two of the four passes can never be served from it.

The exception worth knowing: **m13 has 20 of 20 R@300s rows**. Evaluating on
m13 with the cache wired up (translation + arcsec→px) would cover the raw side
completely and leave only 20 denoised frames to measure — ~30 min rather than
5-6 h — at the cost of a 20-frame convergence curve instead of 56.

The bigger and unconditional win is still removing the two redundant passes.

## Timings on the Spark (R 300s, 204 frames, 6 DSOs)

| step | wall |
| --- | --- |
| scan + registration + dataset build | ~25 min |
| train, 130 epochs | ~81 min (37 s/epoch) |
| denoise 204 frames, tiled + registration | ~50 min (14.6 s/frame) |
| evaluate, 56-frame target | **~5-6 h** (FWHM-bound, see above) |

Peak RSS during training is ~99 GB of 121 GB. It holds, but leaves the Spark
with no headroom — do not start a stack or a transit search alongside it.

## Standing rule

Denoised frames are a **display** product until the gate above passes.
Calibrated linear frames stay the science product for the transit work, the
colour–magnitude diagram, and the transient search. A denoiser that makes an
image prettier and one that preserves photometry are different things.

## Cost to expect

Dataset construction now runs one `sep.Background` per frame and holds
normalised copies alongside the originals — roughly 2× frame memory. Fine on
128 GB, but visible.
